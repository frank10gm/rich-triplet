# Rich Triplet — LLM from scratch in Rust

A complete GPT-style language model built from first principles in Rust, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, AdamW optimizer — is written from scratch as a learning exercise.

---

## What this project is

This is an **educational implementation**, not a production tool. The goal was to understand *why* deep learning frameworks like PyTorch work the way they do, by building all the same pieces by hand:

| Layer | What you understand after writing it |
|---|---|
| `tensor.rs` | Row-major storage, matmul, softmax |
| `autograd.rs` | The chain rule as a computation graph, Rc/RefCell ownership |
| `autograd2.rs` | Why PyTorch is fast: matrix VJPs instead of scalar VJPs |
| `nn.rs` / `nn2.rs` | Linear layers, GELU, LayerNorm from first principles |
| `transformer.rs` / `transformer2.rs` | Q/K/V attention, causal masking, residual connections |
| `train.rs` / `train2.rs` | AdamW, gradient clipping, autoregressive generation |

The project contains **two full implementations** of the same model:
- `*scalar*` — one node per weight element (~29,000 nodes for a nano model)
- `*tensor*` — one node per weight matrix (~50 nodes for the same model)

Running both side by side demonstrates the **203x speedup** that tensor-level autodiff gives you — the exact same insight that motivates why PyTorch, JAX, and every modern framework operate on tensors, not scalars.

---

## Can my model actually generate text?

**Yes — and no.** Here is the honest answer:

### What it CAN do

- Train on any UTF-8 text file and generate new text that follows its style
- Learn character-level patterns, word boundaries, punctuation
- Show measurable loss reduction (from ~4.0 at random init to ~2.5–3.0 after training)
- Generate text that "looks like" the training data at the character level

### What it CANNOT do (and why)

The nano model we run has **29,494 parameters**. For comparison:

| Model | Parameters | What it can do |
|---|---|---|
| This model (nano) | 29K | Learn character patterns in a small corpus |
| GPT-2 small | 117M | Coherent paragraphs, follows instructions loosely |
| GPT-2 XL | 1.5B | Strong text generation, knowledge |
| GPT-4 | ~1T (estimated) | Reasoning, code, multilingual |

A 29K parameter model trained on a ~2,000 character corpus for 200 steps will produce text that *looks statistically like* the training data but is not coherent prose. This is expected and intentional — the point was to watch the loss go down, not to write a novel.

**The gap between "loss decreasing" and "useful text" is mostly just scale**: more parameters, more data, more training steps. The architecture and math are identical.

---

## Running the benchmark

```bash
cargo run --release
```

This runs both the scalar and tensor models for 200 training steps on the built-in bilingual corpus, then prints a timing comparison. Expected output:

```
╔══════════════════════════════════════════════════════════════╗
║  Scalar autograd:      ~31s    (~156 ms/step)               ║
║  Tensor autodiff:      ~0.15s  (~1   ms/step)               ║
║  Speedup:              ~203x                                 ║
╚══════════════════════════════════════════════════════════════╝
```

Both models reach the same loss values — they are mathematically identical. Only the graph representation differs.

## Running tests

```bash
cargo test
```

119 tests covering every component: matrix ops, gradient correctness (verified numerically with finite differences), attention shapes, loss values, training convergence.

---

## Training on your own text

Open `src/main.rs` and replace the `CORPUS` constant with your own text:

```rust
const CORPUS: &str = "
    Your text here. The more text, the better the model learns.
    A few thousand characters is the minimum to see anything interesting.
    A few hundred thousand characters starts to produce coherent patterns.
";
```

You can also adjust the model size in the config:

```rust
let config = Config {
    vocab_size: tokenizer.vocab_size(),
    context_length: 64,    // how many characters the model sees at once
    d_model: 128,          // embedding dimension (higher = more capacity)
    n_layers: 4,           // number of transformer blocks
    n_heads: 4,            // number of attention heads (must divide d_model)
};
```

And the training duration:

```rust
let train_cfg = TrainConfig2 {
    max_steps: 2000,       // more steps = better training
    eval_interval: 100,
    learning_rate: 3e-4,
    grad_clip: 1.0,
};
```

**Warning**: the tensor model (`train2`) is fast enough for hundreds of thousands of steps on CPU. The scalar model (`train`) is for demonstration only — it is ~200x slower.

---

## What Path C means — using this as understanding

Path C was described as: *use what you built as understanding, then connect to a real framework.*

Concretely, now that you have written every layer by hand, you could:

### Option 1 — Use Candle (Hugging Face's Rust ML framework)

[Candle](https://github.com/huggingface/candle) is Rust-native, GPU-capable, and has the same conceptual structure as what you built. You can load real GPT-2 weights and run inference:

```toml
# Cargo.toml
[dependencies]
candle-core = "0.8"
candle-nn = "0.8"
candle-transformers = "0.8"
```

```rust
// Load GPT-2 from HuggingFace Hub and run inference
// Every layer in Candle maps 1:1 to what you built:
//   candle_nn::Linear    ↔   your Linear2
//   candle_nn::LayerNorm ↔   your LayerNorm2
//   causal_attention     ↔   your causal_attention()
```

Because you built those layers yourself, reading Candle's source code is straightforward — no magic.

### Option 2 — Scale up this model

The current architecture is correct GPT. You can scale it by:

1. Increasing `d_model` to 256 or 512
2. Increasing `n_layers` to 6–12
3. Training on a large text file (Project Gutenberg, Wikipedia dump, etc.)
4. Training for 10,000–100,000 steps

At `d_model=256, n_layers=6`, you have ~10M parameters — still trainable on CPU overnight, and capable of producing coherent sentences in a single language.

### Option 3 — Load pretrained GPT-2 weights into this model

The `transformer2.rs` architecture is structurally compatible with GPT-2 small (same layer order, same weight shapes). You could:

1. Download GPT-2 weights (available from HuggingFace)
2. Write a weight loader that maps HuggingFace tensor names to your `TensorNode` leaves
3. Use your own `generate2()` function for inference

This would give you a working ~117M parameter model running in your own code.

---

## Project structure

```
src/
├── tensor.rs        Phase 1 — Basic tensor math (Tensor struct, matmul, softmax)
├── tokenizer.rs     Phase 1 — Character tokenizer + BPE tokenizer
├── dataset.rs       Phase 1 — Sliding window dataset, train/val split
│
├── autograd.rs      Phase 2 — Scalar automatic differentiation (Value nodes)
├── nn.rs            Phase 2 — Scalar neural network layers
├── transformer.rs   Phase 3 — Scalar GPT model (slow, educational)
├── train.rs         Phase 4+5 — Scalar AdamW + generation
│
├── autograd2.rs     Path A — Tensor-level autodiff (Mat + TensorNode)
├── nn2.rs           Path A — Tensor-level layers (Linear2, LayerNorm2, Mlp2)
├── transformer2.rs  Path A — Tensor-level GPT (Gpt2)
├── train2.rs        Path A — Tensor-level AdamW + generation
│
└── main.rs          Benchmark: scalar vs tensor, 200 steps each
```

The scalar (`Phase 2–5`) and tensor (`Path A`) stacks are independent. Each has its own tests. The scalar stack exists purely to demonstrate what the tensor stack optimizes.

---

## Key concepts implemented

### Automatic differentiation

```
Forward:  loss = f(weights)      — build computation graph
Backward: ∂loss/∂weights         — walk graph in reverse, apply chain rule
```

The scalar engine (`autograd.rs`) creates one node per number. The tensor engine (`autograd2.rs`) creates one node per matrix operation. Both use the same topological sort + reverse traversal algorithm.

### Matrix VJPs (why the tensor engine is fast)

```
Scalar:  ∂(a·b)/∂a = b                   (one scalar op)
Tensor:  ∂(A@B)/∂A = ∂L/∂C @ B.T        (one matmul)
         ∂(A@B)/∂B = A.T @ ∂L/∂C        (one matmul)
```

A matmul node's backward replaces `rows × cols × inner_dim` scalar multiply-adds with two matrix multiplications that the CPU can vectorize.

### Causal self-attention

```
Q = x @ W_Q    [T, d_head]
K = x @ W_K    [T, d_head]
V = x @ W_V    [T, d_head]
scores = Q @ K.T / sqrt(d_head)    [T, T]
scores[t, s] = -inf  for s > t     (can't look at future tokens)
weights = softmax(scores)           [T, T]
output = weights @ V                [T, d_head]
```

Each token "attends" to all past tokens. The causal mask enforces that the model can only predict the *next* token, not copy future tokens.

### AdamW

```
m = 0.9·m + 0.1·grad          (smooth gradient direction)
v = 0.999·v + 0.001·grad²     (track gradient magnitude)
w -= lr · (m/(1-0.9ᵗ)) / (√(v/(1-0.999ᵗ)) + 1e-8)
w -= lr · 0.1 · w             (weight decay: pull toward zero)
```

The adaptive learning rate (dividing by √v) means parameters with large/noisy gradients get smaller effective updates — stable training without manual tuning.

---

## Stats

- ~7,000 lines of Rust
- 119 tests (all passing)
- Zero ML dependencies
- 203x measured speedup from tensor autodiff
