# Rich Triplet — LLM from scratch in Rust

A complete LLM stack built from first principles in Rust, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, quantization, Metal GPU kernels, tokenizer — is written from scratch.

Supports **Gemma 3 inference** on Apple Silicon with Q4_K_M / Q4_0 GGUF weights, Metal GPU decode at **17+ tok/s**, and **~6 GB RAM**.

---

## What this project is

A full LLM stack in Rust covering three transformer implementations (GPT-2, GPT-OSS, Gemma 3), a tensor autodiff engine, Apple Metal GPU acceleration, Flash Attention, GGUF weight loading, and a CLI for inference. Every component is built from scratch:

| File | What you understand after writing it |
|---|---|
| `tensor.rs` | Row-major storage, matmul, softmax |
| `autograd.rs` | The chain rule as a computation graph, Rc/RefCell ownership |
| `autograd2.rs` | Why PyTorch is fast: matrix VJPs, Q4K/BF16 GEMV, ARM NEON + SDOT |
| `nn.rs` / `nn2.rs` | Linear layers, GELU, RMSNorm, SwiGLU from first principles |
| `transformer.rs` / `transformer2.rs` | Q/K/V attention, causal masking, Flash Attention |
| `transformer3.rs` | GPT-OSS: RoPE, GQA, Mixture of Experts |
| `transformer4.rs` | Gemma 3: NeoX RoPE, sliding window, 4-norm blocks, GGUF loading |
| `gguf_loader.rs` | GGUF file format: Q4_0, Q4_K, Q6_K, BF16, F16, F32 tensor types |
| `tokenizer.rs` | Character tokenizer + HuggingFace BPE tokenizer (tokenizer.json) |
| `metal_ops.rs` | Metal GPU tiled matmul, Q4K/BF16 GEMV kernels |
| `metal_decode.rs` | Full-graph Metal decode: 717 dispatches per token in one command buffer |
| `train.rs` / `train2.rs` | AdamW, gradient clipping, autoregressive generation |

---

## Gemma 3 inference (recommended)

Download a GGUF weight file and the tokenizer directory:

```bash
# Build with Metal GPU support (Apple Silicon)
cargo build --release --features metal

# Run inference
./target/release/rich-triplet \
  --model gemma3-4b \
  --weights ./models/gemma-3-4b-it-q4_0.gguf \
  --tokenizer-dir ./models/gemma-3-4b-it/ \
  --prompt "What is the capital of France?" \
  --max-new 200 \
  --temp 0.8
```

Both Q4_K_M and Q4_0 GGUF files are supported. Q4_0 weights are automatically converted to Q4K format at load time for optimal Metal GEMV performance.

### Performance (Apple Silicon, Gemma 3 4B)

| Metric | Value |
|---|---|
| Decode speed (Metal GPU) | **17.3 tok/s** (52 ms/step) |
| Decode speed (CPU SDOT) | 12.4 tok/s (80 ms/step) |
| Metal init time | ~2.1 s |
| Runtime RAM (Q4_K_M) | ~6 GB |
| Runtime RAM (Q4_0) | ~6 GB |

### How it works

1. **CPU prefill**: processes the prompt tokens through all 34 layers
2. **Metal decode**: the full forward pass (RMSNorm, RoPE, GEMV, attention, GELU, residuals) is encoded as ~717 dispatches in a single Metal command buffer — no CPU round-trips per layer
3. **KV cache**: stored in GPU `MTLBuffer`s (StorageModeShared), synced once from CPU after prefill

### GGUF weight types supported

| Type | How it's stored in memory |
|---|---|
| Q4_K | Native Q4K blocks (0.56 bytes/elem) — fastest path |
| Q4_0 | Converted to Q4K at load time (same perf as Q4K) |
| Q6_K | Dequantized to BF16 (2 bytes/elem) |
| BF16, F16, F32 | BF16 in memory (2 bytes/elem) |

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--model NAME` | — | Model architecture (`gemma3-4b`) |
| `--prompt TEXT` | — | Text to complete (required) |
| `--weights PATH` | — | GGUF file or safetensors directory |
| `--tokenizer-dir DIR` | — | Directory containing `tokenizer.json` and `tokenizer.model` |
| `--max-new N` | 200 | Tokens to generate |
| `--temp T` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-k K` | 40 | Top-K cutoff (0 = disabled) |
| `--top-p P` | 0.95 | Nucleus probability (1.0 = disabled) |
| `--rep-penalty R` | 1.1 | Repetition penalty (1.0 = disabled) |
| `--seed S` | 42 | RNG seed |

---

## Running tests

```bash
cargo test --features metal    # 412 tests
```

Covers every component: matrix ops, gradient correctness (verified with finite differences), attention shapes, Flash Attention, Q4K/BF16 quantization, Metal GPU kernels, Gemma 3 architecture, tokenizer encoding/decoding.

---

## Apple Metal GPU acceleration

On Apple Silicon, Metal GPU decode is enabled with `--features metal`:

```bash
cargo build --release --features metal
```

The Metal backend implements:
- **Full-graph decode**: entire forward pass in one `MTLComputeCommandEncoder` (no per-layer dispatch overhead)
- **Custom MSL kernels**: `gemv_q4k_t`, `gemv_bf16_t`, `rms_norm_gemma3`, `rope_neox`, `gelu_tanh_kernel`, `attention_decode`, `kv_cache_append`, `embed_bf16_lookup`, and more
- **Unified memory**: all buffers use `StorageModeShared` — zero-copy between CPU and GPU on Apple Silicon

Without `--features metal`, the CPU path uses NEON SDOT intrinsics + Apple Accelerate BLAS, achieving 12.4 tok/s.

---

## Training from scratch

This project can also train GPT-2 and GPT-OSS models from scratch on any plain-text corpus.

### Quick start

```bash
cargo run --release -- --prompt "Once upon a time" --train-steps 2000
```

Trains on the built-in bilingual corpus, prints train/val loss, then streams a completion.

### Benchmark

```bash
cargo run --release
```

```
Scalar autograd:      ~31s    (~156 ms/step)
Tensor autodiff:      ~0.15s  (~1   ms/step)
Speedup:              ~203x
```

### Model configuration

```rust
let model_config = Config {
    vocab_size: tokenizer.vocab_size(),
    context_length: 64,
    d_model: 128,
    n_layers: 4,
    n_heads: 4,
};
```

| d_model | n_layers | ~params | time/step | practical for |
|---|---|---|---|---|
| 64 | 2 | ~0.5M | <1 ms | quick experiments |
| 128 | 4 | ~2M | ~2 ms | short stories, code |
| 256 | 6 | ~10M | ~8 ms | overnight runs |
| 512 | 8 | ~50M | ~50 ms | multi-day runs |

### GPT-OSS architecture (optional)

All GPT-OSS building blocks (RMSNorm, RoPE, SwiGLU, GQA, MoE) have full backward passes:

```rust
let config = Config3 {
    vocab_size: tokenizer.vocab_size(),
    hidden_size: 256,
    num_hidden_layers: 4,
    num_attention_heads: 8,
    num_key_value_heads: 2,
    intermediate_size: 512,
    num_local_experts: 4,
    experts_per_token: 2,
    max_position_embeddings: 512,
    rope_theta: 10000.0,
    rms_norm_eps: 1e-5,
    swiglu_limit: 7.0,
    sliding_window: None,
};
```

---

## Project structure

```
src/
├── tensor.rs          Basic tensor math (Mat struct, matmul, softmax)
├── tokenizer.rs       Character tokenizer + HuggingFace BPE tokenizer
├── dataset.rs         Sliding window dataset, train/val split
│
├── autograd.rs        Scalar automatic differentiation (Value nodes)
├── autograd2.rs       Tensor autodiff — Mat-level VJPs, Q4K/BF16 GEMV, NEON SDOT
├── nn.rs / nn2.rs     Neural network layers (Linear, RMSNorm, SwiGLU)
│
├── transformer.rs     Scalar GPT model (educational)
├── transformer2.rs    GPT-2 architecture (trains end-to-end)
├── transformer3.rs    GPT-OSS (RoPE, GQA, MoE, safetensors loading)
├── transformer4.rs    Gemma 3 (NeoX RoPE, sliding window, GGUF loading)
│
├── gguf_loader.rs     GGUF file parser (Q4_0, Q4_K, Q6_K, BF16, F16, F32)
├── metal_ops.rs       Metal GPU kernels (tiled matmul, per-dispatch GEMV)
├── metal_decode.rs    Full-graph Metal decode engine (single command buffer)
│
├── train.rs           Scalar AdamW + generation
├── train2.rs          Tensor AdamW + streaming generation
└── main.rs            CLI entry point
```

---

## Key concepts implemented

### Automatic differentiation

```
Forward:  loss = f(weights)   — build computation graph
Backward: ∂loss/∂weights      — walk graph in reverse, apply chain rule
```

The scalar engine (`autograd.rs`) creates one node per number. The tensor engine (`autograd2.rs`) creates one node per matrix operation. Both use topological sort + reverse traversal. The tensor engine is ~200x faster.

### Flash Attention (Dao et al. 2022)

Standard attention materializes a `T*T` score matrix. Flash Attention tiles Q in blocks of 64 and accumulates output with an online softmax, using only O(T) memory. The backward pass recomputes softmax weights from stored `(l, m)` vectors.

### Quantization

- **Q4_K**: 4-bit quantization with 2-level scales (super-block + sub-block). 256 elements in 144 bytes. Dequantized on-the-fly during GEMV via ARM SDOT instruction.
- **Q4_0 -> Q4_K conversion**: Q4_0 weights are automatically re-quantized to Q4K format at load time for unified Metal GEMV and SDOT CPU paths.
- **INT4 dynamic quantization**: `quantize_for_inference()` for GPT-OSS weights (~4x memory reduction).

### RoPE (Rotary Position Embeddings)

Gemma 3 uses the NeoX/half-split pattern — pairs `(i, i+head_dim/2)` rather than interleaved `(2i, 2i+1)`:

```
angle = (pos * freq_scale) / theta^(2i / d_head)
x'_i          = x_i * cos(angle) - x_{i+half} * sin(angle)
x'_{i+half}   = x_{i+half} * cos(angle) + x_i * sin(angle)
```

Local attention layers use theta=10000, freq_scale=1.0. Global layers (every 6th) use theta=1M, freq_scale=1/8.

### Grouped Multi-Query Attention (GQA)

Gemma 3 4B: 8 Q heads, 4 KV heads — each KV head shared by 2 Q heads. Reduces KV cache memory by 2x.

### Mixture of Experts (MoE)

GPT-OSS uses 32 experts, 4 active per token — 8x more total parameters, same compute per token.

---

## Stats

- ~28,000 lines of Rust
- 412 tests (including 23 Metal GPU kernel tests)
- Zero ML dependencies
- 203x measured speedup from tensor autodiff
- 17.3 tok/s Metal GPU decode (Gemma 3 4B, Apple Silicon)
- ~6 GB runtime RAM for 4B parameter model
- ~2.1 s Metal init time
- Flash Attention: O(T) memory vs O(T^2) for standard attention
- KV cache generation: O(1) per new token
