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

280 tests covering every component: matrix ops, gradient correctness (verified numerically with finite differences), attention shapes, Flash Attention correctness, Flash Attention gradients, GPT-OSS architecture shapes, sampling strategies.

---

## Training a model from scratch

This project trains a GPT-2 architecture model on any plain-text corpus. The model uses a character tokenizer, so any UTF-8 text works with no preprocessing.

### Step 1 — Put your text in the corpus

Edit `src/main.rs` and replace the `CORPUS` constant with your own text:

```rust
const CORPUS: &str = "
Your text goes here. The more the better — a few thousand words is enough
to see the model learn. A few hundred thousand words gives recognizable output.
";
```

### Step 2 — Tune the model size (optional)

In `src/main.rs`, find the `model_config` inside `run_gpt2_generate` and adjust:

```rust
let model_config = Config {
    vocab_size: tokenizer.vocab_size(),   // set automatically from corpus
    context_length: 64,    // tokens the model sees at once
    d_model: 128,          // embedding dimension — more = more capacity
    n_layers: 4,           // transformer blocks
    n_heads: 4,            // attention heads (must divide d_model)
};
```

Size guide (CPU training):

| d_model | n_layers | ~params | time/step | practical for |
|---|---|---|---|---|
| 64 | 2 | ~0.5M | <1 ms | quick experiments |
| 128 | 4 | ~2M | ~2 ms | short stories, code |
| 256 | 6 | ~10M | ~8 ms | overnight runs |
| 512 | 8 | ~50M | ~50 ms | multi-day runs |

### Step 3 — Train and generate

```bash
cargo run --release -- --prompt "Once upon a time" --train-steps 2000
```

The model trains for 2000 steps, prints train/val loss every 400 steps, then streams a completion of the prompt to stdout.

### Step 4 — Save and resume

To save a checkpoint after training, edit the `TrainConfig2` in `run_gpt2_generate`:

```rust
let cfg = TrainConfig2 {
    max_steps: 2000,
    checkpoint_path: Some("model.ckpt".to_string()),
    early_stopping_patience: 5,   // stop if val loss doesn't improve for 5 evals
    ..TrainConfig2::default()
};
```

Resume from checkpoint without retraining:

```bash
cargo run --release -- --prompt "Once upon a time" --checkpoint model.ckpt
```

### Step 5 — Use the GPT-OSS architecture instead (optional)

All GPT-OSS building blocks (RMSNorm, RoPE, SwiGLU, GQA, MoE) have full backward passes and can be trained from scratch at small scale. To use them, replace the model in `run_gpt2_generate`:

```rust
use transformer3::{GptOssModel, Config3};

let config = Config3 {
    vocab_size: tokenizer.vocab_size(),
    hidden_size: 256,
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,     // GQA: 4 Q heads share each KV head
    intermediate_size: 512,
    num_local_experts: 4,
    experts_per_token: 2,
    max_position_embeddings: 512,
    rope_theta: 10000.0,
    rms_norm_eps: 1e-5,
    swiglu_limit: 7.0,
    sliding_window: None,       // or Some(64) to enable local attention
};
let model = GptOssModel::new(config, &mut rng);
train2(&model, &tokenizer, &train_data, &val_data, &cfg);
```

---

## Running GPT-OSS inference

**Note: this project does not support loading pretrained GPT-2 weights.** The `Gpt2` struct is the GPT-2 *architecture* but trained from scratch on your corpus. For pretrained inference, GPT-OSS is the supported path.

To run GPT-OSS inference, you need:
1. Pretrained `.safetensors` shards from HuggingFace
2. A `vocab.json` and `merges.txt` (BPE tokenizer files)

```bash
# Download weights (requires huggingface_hub)
pip install huggingface_hub
huggingface-cli download openai/gpt-oss-20b --local-dir ./gpt-oss-20b-weights

# Run inference
cargo run --release -- \
  --prompt "The sky above the port" \
  --weights ./gpt-oss-20b-weights/ \
  --vocab  ./gpt-oss-20b-weights/vocab.json \
  --merges ./gpt-oss-20b-weights/merges.txt \
  --max-new 200 \
  --temp 0.8 \
  --top-k 40
```

The model streams tokens to stdout as they are generated. The weight loader supports F32, BF16, and F16 shards with full GPT-OSS tensor name mapping.

To reduce memory from ~40GB to ~10GB before generating, add this to `run_gpt_oss` in `src/main.rs`:

```rust
model.load_weights_from_dir(weights_dir).unwrap();
let stats = model.quantize_for_inference();   // INT4: 4× less RAM
eprintln!("Compressed to {:.1}GB ({:.1}x)", stats.q4_bytes as f64 / 1e9, stats.compression_ratio);
```

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
| `--top-p P` | 0.95 | Nucleus probability (1.0 = disabled) |
| `--rep-penalty R` | 1.1 | Repetition penalty (1.0 = disabled) |
| `--seed S` | 42 | RNG seed |
| `--train-steps N` | 200 | Training steps (no-weights mode) |
| `--checkpoint PATH` | — | Load saved `.ckpt` instead of training |

---

## Running Gemma 3 inference (GGUF)

Download a GGUF weight file (e.g. from HuggingFace) and a copy of the tokenizer:

```bash
# Example with the 4B QAT Q4_0 GGUF
cargo run --release -- \
  --prompt "What is the capital of France?" \
  --weights ./models/gemma-3-4b-it-q4_0.gguf \
  --tokenizer-dir ./models/gemma-3-4b-it/ \
  --model gemma3-4b \
  --max-new 200 \
  --temp 0.8 \
  --top-k 40 \
  --top-p 0.95 \
  --rep-penalty 1.1
```

### Performance tips

The project uses **Apple Accelerate (BLAS)** by default, which gives 4–8× faster matrix
multiplications vs the pure-Rust fallback:

```bash
# Default build already includes BLAS on macOS:
cargo build --release

# To disable BLAS (e.g. for a non-Apple platform):
cargo build --release --no-default-features
```

On Apple Silicon, the `.cargo/config.toml` already sets `-C target-cpu=native` so NEON/AMX
instructions are used automatically.

### GGUF RMSNorm convention

GGUF-converted Gemma 3 weights store RMSNorm scale factors as `(1 + w)` (i.e. the final
effective multiplier), while the `forward_gemma3` implementation applies `(1 + γ) × x̂`.
The GGUF loader therefore subtracts 1 from every loaded norm weight so the math works out
to the correct `(1 + w) × x̂`. The safetensors loader is unaffected.

---

## Apple Metal GPU acceleration

On Apple Silicon, matrix multiplications are dispatched to the GPU automatically:

```bash
cargo run --release --features metal
```

The Metal backend uses a tiled kernel with `threadgroup` shared memory (16×16 tiles), giving 3–5× speedup over the scalar CPU path for large matrices. Matrices below 32,768 elements fall back to CPU to avoid dispatch overhead.

```bash
cargo test --features metal    # 285 tests (5 additional Metal-specific tests)
```

---

## INT4 quantization

After loading pretrained GPT-OSS weights, reduce inference memory by ~4× with dynamic INT4 quantization:

```rust
model.load_weights_from_dir("./gpt-oss-20b-weights").unwrap();
let stats = model.quantize_for_inference();
println!("Quantized {} tensors: {:.1}GB → {:.1}GB ({:.1}x compression)",
    stats.n_tensors,
    stats.f32_bytes as f64 / 1e9,
    stats.q4_bytes  as f64 / 1e9,
    stats.compression_ratio);
// → Quantized 856 tensors: 40.0GB → 10.0GB (4.0x compression)
```

Weights are stored as INT4 nibbles with per-block f32 scales. Each forward call uses `matmul_q4_t` — the dequantization happens per-row during the matmul without materializing the full weight matrix, so peak RAM stays at the compressed size.

Quantization error is at most `absmax / 14` per weight block (~7% of the largest value in each block of 32 elements).

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
- 280 tests (285 with `--features metal`)
- Zero ML dependencies
- 203× measured speedup from tensor autodiff
- Flash Attention: O(T) memory vs O(T²) for standard attention
- SwiGLU clamp (configurable per model, 7.0 for GPT-OSS)
- Sliding window attention (alternating full/local, configurable window size)
- INT4 dynamic quantization (`quantize_for_inference`): ~4× memory reduction
- Checkpoint save/load + resume via `--checkpoint`
- Early stopping on validation loss (`early_stopping_patience`)
- O(T) generation via KV cache (both GPT-2 and GPT-OSS)
