# Rich Triplet — LLM from scratch in Rust

A complete GPT-style language model built from first principles in Rust, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, AdamW optimizer — is written from scratch.

---

## What this project is

A full LLM stack in Rust, covering two complete transformer implementations (GPT-2 and GPT-OSS), a tensor autodiff engine, Apple Metal GPU acceleration, Flash Attention, and a CLI for inference. Every component is built from scratch:

| File | What you understand after writing it |
|---|---|
| `tensor.rs` | Row-major storage, matmul, softmax |
| `autograd.rs` | The chain rule as a computation graph, Rc/RefCell ownership |
| `autograd2.rs` | Why PyTorch is fast: matrix VJPs instead of scalar VJPs |
| `nn.rs` / `nn2.rs` | Linear layers, GELU, RMSNorm, SwiGLU from first principles |
| `transformer.rs` / `transformer2.rs` | Q/K/V attention, causal masking, Flash Attention |
| `transformer3.rs` | GPT-OSS: RoPE, GQA, Mixture of Experts |
| `train.rs` / `train2.rs` | AdamW, gradient clipping, autoregressive generation |
| `metal_ops.rs` | Tiled GPU kernels in Metal Shading Language |

---

## Running the benchmark

```bash
cargo run --release
```

Trains both the scalar and tensor models for 200 steps on the built-in bilingual corpus, prints a timing comparison:

```
╔══════════════════════════════════════════════════════════════╗
║  Scalar autograd:      ~31s    (~156 ms/step)               ║
║  Tensor autodiff:      ~0.15s  (~1   ms/step)               ║
║  Speedup:              ~203x                                 ║
╚══════════════════════════════════════════════════════════════╝
```

Both models reach the same loss values — mathematically identical, different graph representation.

---

## Running tests

```bash
cargo test
```

271 tests covering every component: matrix ops, gradient correctness (verified numerically with finite differences), attention shapes, Flash Attention correctness, Flash Attention gradients, GPT-OSS architecture shapes, sampling strategies.

---

## Training on your own text

Replace the `CORPUS` constant in `src/main.rs` with your own text, then run:

```bash
cargo run --release -- --prompt "your prompt here" --train-steps 2000
```

This trains on `CORPUS`, then generates a completion for the prompt. The model uses a character tokenizer so any UTF-8 text works with no preprocessing.

To save the trained weights for later use:

```rust
let cfg = TrainConfig2 {
    max_steps: 2000,
    checkpoint_path: Some("model.ckpt".to_string()),
    ..TrainConfig2::default()
};
train2(&model, &tokenizer, &train_data, &val_data, &cfg);
```

Reload with `load_checkpoint("model.ckpt")` — returns a `Vec<(String, Mat)>` that can be applied back to any model with matching parameter count.

You can also tune the model size directly in `src/main.rs`:

```rust
let config = Config {
    vocab_size: tokenizer.vocab_size(),
    context_length: 64,    // tokens the model sees at once
    d_model: 128,          // embedding dimension (higher = more capacity)
    n_layers: 4,           // transformer blocks
    n_heads: 4,            // attention heads (must divide d_model)
};
```

At `d_model=256, n_layers=6` you have ~10M parameters — trainable on CPU overnight.

---

## Running GPT-OSS inference

If you have pretrained GPT-OSS weights (`.safetensors` shards) and a BPE tokenizer:

```bash
cargo run --release -- \
  --prompt "The sky above the port" \
  --weights /path/to/weights/ \
  --vocab  /path/to/vocab.json \
  --merges /path/to/merges.txt \
  --max-new 200 \
  --temp 0.8 \
  --top-k 40
```

The model streams tokens to stdout as they are generated. The weight loader supports F32, BF16, and F16 shards with full GPT-OSS tensor name mapping.

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--prompt TEXT` | — | Text to complete (required for generation) |
| `--weights DIR` | — | Directory with `.safetensors` shards (GPT-OSS) |
| `--vocab PATH` | — | `vocab.json` (required with `--weights`) |
| `--merges PATH` | — | `merges.txt` (required with `--weights`) |
| `--max-new N` | 200 | Tokens to generate |
| `--temp T` | 0.8 | Sampling temperature |
| `--top-k K` | 40 | Top-K cutoff (0 = disabled) |
| `--top-p P` | 1.0 | Nucleus probability (1.0 = disabled) |
| `--seed S` | 42 | RNG seed |
| `--train-steps N` | 200 | Training steps (no-weights mode) |

---

## Apple Metal GPU acceleration

On Apple Silicon, matrix multiplications are dispatched to the GPU automatically:

```bash
cargo run --release --features metal
```

The Metal backend uses a tiled kernel with `threadgroup` shared memory (16×16 tiles), giving 3–5× speedup over the scalar CPU path for large matrices. Matrices below 32,768 elements fall back to CPU to avoid dispatch overhead.

```bash
cargo test --features metal    # 276 tests (5 additional Metal-specific tests)
```

---

## Training a custom model with the GPT-OSS architecture

The GPT-OSS building blocks (RMSNorm, RoPE, SwiGLU, GQA, MoE) all have full backward passes — gradients flow through all of them. You can train a small custom model using them:

```rust
// src/transformer3.rs — use a tiny config instead of gpt_oss_20b()
let config = Config3 {
    vocab_size: tokenizer.vocab_size(),
    hidden_size: 256,
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,   // GQA: 4 Q heads share each KV head
    intermediate_size: 512,
    num_local_experts: 4,
    experts_per_token: 2,
    max_position_embeddings: 512,
    rope_theta: 10000.0,
    rms_norm_eps: 1e-5,
    swiglu_limit: 7.0,
};
let mut rng = InitRng::new(42);
let model = GptOssModel::new(config, &mut rng);
// then pass to train2() as usual
```

---

## Project structure

```
src/
├── tensor.rs        Phase 1 — Basic tensor math (Mat struct, matmul, softmax)
├── tokenizer.rs     Phase 1 — Character tokenizer + BPE tokenizer
├── dataset.rs       Phase 1 — Sliding window dataset, train/val split
│
├── autograd.rs      Phase 2 — Scalar automatic differentiation (Value nodes)
├── nn.rs            Phase 2 — Scalar neural network layers
├── transformer.rs   Phase 3 — Scalar GPT model (slow, educational)
├── train.rs         Phase 4 — Scalar AdamW + generation
│
├── autograd2.rs     Tensor autodiff — Mat-level VJPs, Flash Attention, RoPE, GQA
├── nn2.rs           Tensor layers — Linear2, LayerNorm2, RmsNorm2, SwiGluMlp2
├── transformer2.rs  GPT-2 architecture (trains end-to-end)
├── train2.rs        Tensor AdamW + streaming generation
├── transformer3.rs  GPT-OSS architecture (RoPE, GQA, MoE)
│
├── metal_ops.rs     Apple Metal GPU backend (tiled matmul kernel)
└── main.rs          CLI entry point + benchmark
```

---

## Key concepts implemented

### Automatic differentiation

```
Forward:  loss = f(weights)   — build computation graph
Backward: ∂loss/∂weights      — walk graph in reverse, apply chain rule
```

The scalar engine (`autograd.rs`) creates one node per number. The tensor engine (`autograd2.rs`) creates one node per matrix operation. Both use the same topological sort + reverse traversal algorithm.

### Matrix VJPs (why the tensor engine is fast)

```
Scalar:  ∂(a·b)/∂a = b                   (one scalar multiply)
Tensor:  ∂(A@B)/∂A = ∂L/∂C @ B.T        (one matmul)
         ∂(A@B)/∂B = A.T @ ∂L/∂C        (one matmul)
```

A matmul node's backward replaces `rows × cols × inner_dim` scalar ops with two matrix multiplications the CPU can vectorize.

### Flash Attention (Dao et al. 2022)

Standard attention materializes a `T×T` score matrix. Flash Attention never does — it tiles Q in blocks of 64 and accumulates the output with an online softmax, using only O(T) memory:

```
For each tile of Q (size BLOCK_R):
  For each tile of K,V (size BLOCK_C):
    Compute scores for this tile
    Update running max m and normalizer l (online softmax)
    Accumulate weighted V into output
```

The backward pass recomputes softmax weights from the stored `(l, m)` vectors rather than storing the full attention matrix.

### Causal self-attention

```
Q = x @ W_Q    [T, d_head]
K = x @ W_K    [T, d_head]
V = x @ W_V    [T, d_head]
scores = Q @ K.T / sqrt(d_head)    [T, T]
scores[t, s] = -inf  for s > t     (causal mask)
weights = softmax(scores)           [T, T]
output = weights @ V                [T, d_head]
```

### RoPE (Rotary Position Embeddings)

Instead of adding learned position vectors, RoPE rotates Q and K by an angle that depends on position. Two adjacent dimensions form a rotation pair:

```
For each pair (x_{2i}, x_{2i+1}) at position t:
  angle = t / theta^(2i / d_head)
  x'_{2i}   = x_{2i}   * cos(angle) - x_{2i+1} * sin(angle)
  x'_{2i+1} = x_{2i+1} * cos(angle) + x_{2i}   * sin(angle)
```

This makes attention scores depend only on the *relative* distance between tokens, which generalizes better to long contexts.

### Grouped Multi-Query Attention (GQA)

Instead of one K and V head per Q head, GQA uses fewer KV heads shared across groups of Q heads. With `n_q_heads=64` and `n_kv_heads=8`, each KV head is shared by 8 Q heads — 8× less KV cache memory at inference time.

### Mixture of Experts (MoE)

Instead of one dense FFN per layer, MoE has N expert FFNs and a router that picks the top-K for each token:

```
router_logits = x @ W_router          [T, num_experts]
weights = softmax(top_k(router_logits))
output = sum_k(weight_k * expert_k(x))
```

GPT-OSS uses 32 experts, 4 active per token — 8× more total parameters, same compute per token.

### AdamW

```
m = 0.9·m + 0.1·grad          (smooth gradient direction)
v = 0.999·v + 0.001·grad²     (track gradient magnitude)
w -= lr · (m/(1-0.9ᵗ)) / (√(v/(1-0.999ᵗ)) + 1e-8)
w -= lr · 0.1 · w             (weight decay)
```

---

## Stats

- ~15,000 lines of Rust
- 271 tests (276 with `--features metal`)
- Zero ML dependencies
- 203× measured speedup from tensor autodiff
- Flash Attention: O(T) memory vs O(T²) for standard attention
- SwiGLU clamp (configurable per model, 7.0 for GPT-OSS)
- Checkpoint save/load for trained weights
