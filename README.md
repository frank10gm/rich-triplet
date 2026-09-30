# Rich Triplet — LLM from scratch in Rust

A complete LLM stack built from first principles in Rust, **with no ML dependencies**. Every component — matrix math, automatic differentiation, attention, quantization, Metal GPU kernels, tokenizer — is written from scratch.

Supports **Gemma 3 inference** on Apple Silicon with Q4_K_M / Q4_0 GGUF weights, Metal GPU decode at **17+ tok/s**, and **~6 GB RAM**.

It also carries two complete text-to-speech stacks — text in, WAV out, neural audio codecs included: **Orpheus**, an autoregressive Llama 3.2 emitting SNAC codes, and **OmniVoice**, a masked-diffusion Qwen3 that unmasks eight codebooks in parallel and clones a voice from a few seconds of reference audio — and a text-to-image stack: **FLUX.1-schnell**, a 12B rectified-flow transformer with its T5-XXL and CLIP-L text encoders and a 16-channel autoencoder, text in and PNG out, with a full-graph Metal engine that keeps the transformer quantized on the GPU and widens each weight only as it is used. The design, the build log and the six bugs between passing tests and a working image are in [docs/text-to-image.md](docs/text-to-image.md).

These three stacks were built first in the C++ port of this project (`rich-triplet-cpp`) and ported back here module by module, checked against the C++ build for identical output. Measurements quoted in their sections were taken on the C++ build unless stated otherwise.

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
| `transformer5.rs` | Llama 3.2: the two RoPE pair conventions, and why GGUF needs the other one |
| `transformer6.rs` | Bidirectional attention, and what a model without a KV cache costs |
| `hubert.rs` | A transformer whose input is a waveform, and whose position is a convolution |
| `conv1d.rs` | Dilated, grouped and transposed 1-D convolution; Snake; weight norm |
| `snac.rs` | Multi-scale residual vector quantization, and a codec decoder |
| `orpheus.rs` | Audio tokens: slot offsets, frame de-interleaving, resynchronisation |
| `omnivoice.rs` | Masked diffusion decoding: confidence unmasking, classifier-free guidance |
| `omnivoice_codec.rs` | Dense residual convolution, and why im2col earns its memory |
| `conv2d.rs` | The same operator one dimension up, and why the layout follows from it |
| `vae.rs` | GroupNorm's statistics span space, and what a latent is scaled by |
| `t5.rs` | An encoder with no positional embedding, and no attention scaling either |
| `clip_text.rs` | Why a text encoder is causal, and where its pooled vector comes from |
| `flux.rs` | Two stream types, adaptive layer norm, and three-axis RoPE |
| `duration.rs` | That Unicode has opinions about how long a character takes to say |
| `resample.rs` | Polyphase rate conversion, and why the filter has to be *that* filter |
| `torch_pickle.rs` | ZIP central directories, and just enough pickle to be safe |
| `wav.rs` | RIFF in both directions, and how to tell speech from noise without listening |
| `png.rs` | That DEFLATE has a mode needing no compressor, and CRC-32 either way |
| `metal_omnivoice.rs` | A GPU GEMM worth writing, and how to know a second implementation agrees |
| `metal_flux.rs` | Keeping 12B quantized at rest, and streaming softmax past the threadgroup limit |

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
| `--system TEXT` | — | ChatML system turn for Qwen 3.5 chat models |

The speech and image stacks add their own flags; `--help` lists them all.

| Flag | Default | Description |
|---|---|---|
| `--model NAME` | — | also `orpheus-3b`, `omnivoice`, `flux-schnell`, `flux-dev` |
| `--voice NAME` | `tara` | Orpheus speaker |
| `--snac PATH` | `models/snac_24khz.bin` | Codec checkpoint — SNAC for Orpheus, the tokenizer GGUF for OmniVoice |
| `--out PATH` | `out.wav` / `out.png` | Where to write the synthesised audio or the image |
| `--no-audio-mask` | off | Let Orpheus sample outside the audio token range |
| `--no-leading-bos` | off | Drop the leading BOS from the Orpheus prompt |
| `--language NAME` | `None` | OmniVoice language hint, e.g. `Italian` |
| `--instruct TEXT` | `None` | OmniVoice voice description, e.g. `a calm young woman` |
| `--duration S` | 0 | OmniVoice audio seconds; 0 estimates from the text |
| `--steps N` | 12 | OmniVoice unmasking steps — the quality/speed dial; FLUX denoising steps (schnell 4, dev 28) |
| `--chunk-seconds S` | 15 | Audio per chunk when splitting long text; 0 never splits |
| `--chunk-threshold S` | 30 | Split only when the estimate exceeds this |
| `--chunk-gap S` | 0.3 | Pause between chunks, in seconds |
| `--guidance G` | 2.0 | OmniVoice classifier-free guidance; 0 halves the work |
| `--ref-audio PATH` | — | WAV of a voice for OmniVoice to clone |
| `--ref-text TEXT` | — | What that WAV says; required alongside it |
| `--rope-interleaved` | off | Use interleaved RoPE pairing (debugging only) |
| `--t5 PATH` | — | T5-XXL encoder GGUF (FLUX) |
| `--t5-tokenizer PATH` | — | `spiece.model`; optional, the GGUF carries one |
| `--clip PATH` / `--clip-tokenizer PATH` | — | CLIP-L safetensors and its `tokenizer.json` |
| `--vae PATH` | — | Autoencoder safetensors |
| `--width N` / `--height N` | 1024 | Image size, multiples of 16 |
| `--tile N` | 0 | VAE tile in latent pixels; 0 decodes whole (tiles automatically above 1024x1024) |
| `--batch N` | 1 | Generate N images from consecutive seeds |
| `--cpu` | off | Run the FLUX transformer on the CPU in a Metal build |

---

## Orpheus text to speech

```bash
# The codec weights (80 MB) -- fetched once, shared by every language
curl -L -o models/snac_24khz.bin \
  https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/pytorch_model.bin

cargo run --release -- \
  --model orpheus-3b \
  --weights ./models/Orpheus-3b-Italian_Spanish-FT-Q8_0.gguf \
  --snac ./models/snac_24khz.bin \
  --prompt "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --voice giulia \
  --out giulia.wav
```

Emotion tags like `<laugh>` and `<sigh>` are ordinary text — the tokenizer
merges them like any other word, so they need no special handling.

### Checkpoints and voices

The published fine-tunes share an architecture, a vocabulary and a codec, and
differ only in language. Any of them loads with no code changes.

| Checkpoint | Languages | Voices |
|---|---|---|
| `canopylabs/orpheus-3b-0.1-ft` | en | tara, leah, jess, leo, dan, mia, zac, zoe |
| `lex-au/Orpheus-3b-Italian_Spanish-FT-Q8_0` | it, es | pietro, giulia, carlo · javi, sergio, maria |

A voice is only a prompt prefix, so naming one the checkpoint was not trained
on still synthesises — it just will not sound like a consistent speaker. The
CLI warns.

The Italian and Spanish weights are a **research release**, and it shows in the
download counts: a few hundred against a quarter of a million for the English
fine-tune. Judge the output by ear before building anything on it.

No `--tokenizer-dir` is needed. The GGUF carries its own 156 940-entry
vocabulary and 280 147 merges, which matters because the Orpheus repositories
are gated on HuggingFace — the weights are freely mirrored as GGUF but
`tokenizer.json` is not.

Nothing in the pipeline is language-specific. The tokenizer is byte-level BPE,
so accented text encodes and round-trips without special handling, and the
codec is phonetically neutral. The language lives entirely in the weights.

### How it works

1. **The backbone is a Llama 3.2 3B** whose vocabulary has been extended by
   28 672 audio codes. It does not emit audio; it emits codec tokens.
2. **Seven tokens make a frame.** `<custom_token_0>` is id 128256, and a code
   is `id - 128266 - (slot * 4096)`, so audio ids run 128266–156937 across
   seven codebook slots of 4096.
3. **A frame is 2048 samples** — 85.33 ms at 24 kHz. Realtime therefore needs
   about 82 tokens a second.
4. **SNAC reconstructs the waveform** from three codebooks running at 1/4, 1/2
   and 1/1 of the frame rate, then four transposed-convolution blocks upsample
   by 8, 8, 4 and 2 for 512 samples per frame.

### How the file was quantized decides what happens at load

The two published checkpoints are packaged differently, and each needs
something the other does not. Both cases are handled automatically; the CLI
prints which branch it took.

| | English Q4_K_M | Italian/Spanish Q8_0 |
|---|---|---|
| On disk | 2.36 GB | 3.52 GB |
| Tensors | 256 | 255 — **lm_head is weight-tied** |
| Widened to BF16 | 29 of 197 projections | **197 of 197** |
| After load | 4.08 GB | 7.57 GB |
| Action taken | requantize lm_head only | requantize every projection |
| Resident | **3.5 GB** | **2.8 GB** |

A Q4_K_M file keeps most projections native and lifts only `attn_v`,
`ffn_down` and `output` to Q6_K — those are the quality-sensitive ones, chosen
deliberately by the quantizer. Flattening them to Q4_K would throw that choice
away, so only the lm_head is requantized: at 156 940 entries it costs 964 MB as
BF16 against 271 MB as Q4_K, and it is the largest single read per token.

A uniformly higher-precision file has no Q4_K tensors at all, so every
projection widens and there is no deliberate choice to preserve. Requantizing
all of them is the right call, and lands *below* the Q4_K_M model.

The decision is made on the fraction of projections widened, not on a byte
threshold, because that fraction is what actually distinguishes the two cases.

Weight tying is free: `MatBf16` is reference-counted, so a checkpoint with no
`output.weight` has its lm_head adopt the embedding's bits rather than copy
964 MB of them. The orientations already agree — the table is `[vocab, hidden]`
and `Linear2` computes `input @ weight.T`.

### Measured on an M3 Pro (18 GB, CPU build of the C++ port)

| | |
|---|---|
| Resident weights | 2.8–3.5 GB depending on the checkpoint |
| Decode | ~16 tokens/s |
| Realtime factor | ~5.3 |
| SNAC decode | ~0.1 s per second of audio |

So a 4 s clip takes about 21 s. Decode runs on the CPU and the Metal build
measures the same, for a duller reason than it looks: with the `blas` feature
on -- which is the default, and stays on under `--features metal` --
`Mat::matmul` dispatches to Accelerate and never reaches the GPU at all, so a
Metal build without a dedicated engine is a CPU build. Two things would close most of the gap — a `metal_decode_llama.rs`
trimmed from the Gemma engine, and speculative decoding, which is already
implemented for Gemma. OmniVoice has such an engine; Orpheus does not yet.

### Diagnosing it

Every fault in a speech pipeline sounds the same — a wrong RoPE convention, a
wrong codebook stride and a wrong convolution padding all produce noise. Two
things make that tractable without a reference implementation to diff against.

**Exact length invariants, asserted in code.** Each upsampling block
multiplies its input length by exactly its stride, and each residual unit
preserves length exactly, so `n_frames * 512` has no slack. Every padding or
`output_padding` mistake breaks the multiple and trips an assertion instead of
degrading the audio.

**Waveform statistics, printed every run.** Speech at 24 kHz sits near an RMS
of 0.03–0.2 with negligible DC offset and a low zero-crossing rate; a decoder
fed bad latents saturates its output `tanh` and lands near 0.5 with most
samples at the rails. `wave_stats` reports both and the CLI warns when the
numbers do not look like speech.

`--debug` adds a per-token dump of which codebook slot each id actually falls
in against the slot the stream expected. A healthy stream walks 0, 1, 2, 3, 4,
5, 6 and repeats; anything else is the frame structure breaking down, which is
invisible in the audio itself.

---

## OmniVoice text to speech

```bash
# Both halves of the model -- the LM and its codec (0.94 GB together)
curl -L -o models/omnivoice-base-Q8_0.gguf \
  https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-Q8_0.gguf
curl -L -o models/omnivoice-tokenizer-Q8_0.gguf \
  https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-Q8_0.gguf

cargo run --release -- \
  --model omnivoice \
  --prompt "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --language Italian \
  --steps 12 \
  --out giulia.wav
```

OmniVoice is a **masked diffusion** language model, not an autoregressive one,
and that single fact reaches all the way down into the attention kernel. It is
also a much smaller model than Orpheus — a Qwen3 0.6B backbone against a Llama
3.2 3B — so both halves together are 0.94 GB on disk against Orpheus's 2.4–3.5.
Speed depends on `--steps`: at 12 a 4 s clip takes about 8 s, at the reference
default of 32 about 21, which is where Orpheus lands too.

`--language` and `--instruct` are free text, not enumerations: there is no
voice list, because a voice is described rather than named. `--instruct
"a calm young woman"` is a valid request, and so is leaving it out.

### Two GGUF files, and what is in each

| | `omnivoice-base` | `omnivoice-tokenizer` |
|---|---|---|
| On disk | 0.66 GB | 0.29 GB |
| Holds | the diffusion LM | the audio codec |
| Loaded | 312 tensors, 1.23 GB resident | 21.6 M to synthesise, 163 M to analyse |
| Also carries | its own 151 676-entry vocabulary | both directions of the codec |

The LM's GGUF embeds its tokenizer, so there is no `--tokenizer-dir`. The
codec's file holds both halves: `acoustic_decoder` and the codebooks for
synthesis, and `acoustic_encoder`, `encoder_semantic` and a 94 M-parameter
`semantic_model` for analysis. Plain synthesis loads only the first, about 190
of its 486 tensors; `--ref-audio` loads the rest.

### How it works

1. **Every audio position starts masked.** The sequence is built once — style
   markers, language, instruction, text, then `N` masked frames — and its
   length is fixed before a single sample exists.
2. **Each step runs the whole sequence**, scores every `(codebook, position)`
   pair by confidence, and unmasks the `k` most confident, where `k` comes from
   a warped timestep schedule. After `--steps` steps everything is decided.
3. **Classifier-free guidance means two passes per step**: one conditioned on
   the text, one on the masked frames alone. `log_probs = c + g * (c - u)`.
4. **A layer penalty makes decoding coarse to fine.** Codebook 0 is unpenalised
   and is decided first; the later codebooks condition on what it chose.
5. **The codec reconstructs the waveform** from all eight codebooks at one
   rate, upsampling by 8, 5, 4, 2 and 3 — exactly 960 samples per frame, so
   25 Hz at 24 kHz.

### Why there is no KV cache

Unmasking a position changes the hidden state of every position that attends to
it, and under a bidirectional mask that is all of them. Nothing computed at
step *n* is still valid at step *n+1*, so there is nothing to cache.

The same reasoning forces bidirectional attention: a masked position has to see
the positions *after* it, or the first step would have nothing to condition on.
Every other attention path in this project is causal, so this one needed its own
kernel, `bidirectional_gqa_attention`.

The head is likewise not the usual one. Eight codebooks are predicted at once by
a single `[1024 -> 8200]` matmul reshaped to `[T, 8, 1025]` — the extra entry
per codebook is its mask token. A position's *input* embedding is the sum across
all eight codebooks, so a fully masked position still has a well-defined
embedding: the sum of the eight mask rows.

### `--steps` is the quality/speed dial

Cost is `steps * 2` full-sequence forward passes and nothing else, so it is
almost exactly linear. Measured on an M3 Pro (18 GB, C++ port), synthesising
4 s of Italian:

| `--steps` | RTF, CPU | RTF, Metal | |
|---|---|---|---|
| 8 | 1.30 | 0.38 | |
| 12 | 1.96 | **0.54** | the default |
| 16 | 2.46 | 0.69 | |
| 32 | 5.17 | 1.31 | the reference's |

On the GPU it synthesises faster than the audio plays at every setting up to
16. A reference clip adds its own frames to every forward pass, so cloning
costs more: a 4 s reference roughly doubles the sequence and takes `--steps 12`
from 0.54 to 1.06.

The default is 12 rather than the reference's 32, which is three times faster
and was judged indistinguishable by ear on Italian. That is a listening call
and could not have been anything else: the waveform statistics look like speech
across the whole range and only stop at the extreme, where a single step
produces something the check flags and warns about. `--steps 32` restores the
reference's setting.

### The Metal path

```bash
cargo run --release --features metal -- --model omnivoice --prompt "..." --language Italian
```

This is the first Metal engine here for a text-to-speech model and the first
for a *prefill* shape rather than a decode one. The Gemma and Qwen engines
exist because a decode step is a chain of GEMVs, memory-bound and too small to
dispatch one at a time. This one exists for the opposite reason: OmniVoice has
no KV cache, so every step is a full-sequence pass over a few hundred
positions — large GEMMs, compute-bound, exactly what the matrix units are for.

**It had to clear a high bar.** Accelerate's sgemm reaches about 1.0 TFLOP/s on
an M3 Pro for these shapes. The tiled matmul in `metal_ops.rs` manages 250
GF/s, so routing the GEMMs through it would have been a 4× regression. What
clears the bar is `simdgroup_matrix`: a 64×64 tile per threadgroup built from
sixteen 8×8 accumulators, which reaches 1.2–2.3 TFLOP/s depending on shape.
Every 8×8 operand loaded from memory feeds four multiply-accumulates instead of
one, and that register blocking is the whole difference.

The GEMMs are only 40% of the CPU runtime, though, so moving them alone would
have capped the win near 1.3×. Profiling said where the rest goes, and all of
it is work the GPU does not have to do:

| | Share of CPU runtime | On the GPU |
|---|---|---|
| sgemm | 40% | the matrix units |
| BF16 dequantization | 12% | **gone** — `bfloat` is an operand type |
| `bzero` in `Mat::zeros` | 8% | **gone** — scratch is allocated once |
| attention, `expf`, norms, SiLU, RoPE | ~33% | kernels |

The BF16 line is the one worth naming. The CPU path dequantizes every weight
into an f32 scratch buffer on *every call* — 437 million conversions per
forward pass, 28 billion for a four-second clip. A Metal kernel takes `bfloat`
straight as a matrix operand, so the checkpoint's own storage format is read
with no conversion pass at all.

Measured on an M3 Pro, four seconds of Italian at the default 12 steps:

| | CPU | Metal |
|---|---|---|
| Generate | 7.53 s | **1.88 s** |
| Realtime factor | 1.96 | **0.54** |
| Resident | 2.0 GB | **1.8 GB** |

Four times faster, and *less* memory: the CPU weights are freed as they are
uploaded, so nothing holds both copies. The remaining 0.26 s is the codec
decode, which is still on the CPU and is now a fifth of the total.

### Trusting a second implementation of the same model

A GPU engine is a rewrite of the forward pass, and a wrong one produces audio
that sounds plausible — the same trap as every other part of this pipeline.
What makes it checkable is that the CPU path is still there.

The test loads the checkpoint twice, uploads one copy, and compares. Logits
agree to a relative 1e-5, which is f32 epsilon — BLAS chunks the weight while
the GPU accumulates 8×8 tiles, so the two sum in different orders and could not
agree further. The check that matters is not the logits but the **argmax per
(position, codebook)**, which is what the sampler actually reads: those agree
**exactly**, on every one of them.

Over a full generation the two are not bit-identical, and the reason is worth
knowing. Which positions get unmasked at each step comes from sorting
confidence scores, and scores that differ in their last bits can sort
differently. One such flip early on changes every decision after it. The
waveforms still correlate at 0.9995 — the same utterance, decided in a
marginally different order.

Sequence length is padded to a multiple of 64 so the GEMM tiles divide evenly
and the inner loop needs no bounds checks. Those padding rows are zeroed and
carry through harmlessly, because every operator except attention is
row-independent — and attention is dispatched over the true length, never the
padded one. There is a test for exactly the lengths that are not multiples of
64.

### Length has to be decided in advance

A diffusion model cannot stop early: every frame exists, masked, from the first
step. Get the length wrong and the failure is not graceful — measured on one
sentence, asking for 2 s of a 4 s phrase truncated it mid-sentence, and asking
for 10 s collapsed into 98% silence at a peak of 0.001.

So `duration.rs` ports the reference's `RuleDurationEstimator`, which is not a
neural model but a lookup table. Every character gets a phonetic weight
relative to one Latin letter — a CJK ideograph is a whole syllable at 3.0, a
combining accent is silent at 0.0, a digit is 3.5 because "2024" is four
characters and fourteen letters' worth of speech — and the sum is scaled
against "Nice to meet you." at 25 frames.

Unicode general category is consulted *before* the script block, which is the
part that is easy to get backwards: a Devanagari vowel sign sits inside the
Devanagari block but is a combining mark, and charging it 1.8 would inflate
every Hindi estimate. Below 50 frames the result is pulled up a cube-root
curve, because the fixed costs of an utterance — onset, final lengthening, the
breath at the end — do not shrink with the text.

The port was checked exhaustively rather than by sampling: all 1 112 064
Unicode code points were classified by both implementations and compared, and
they agree everywhere. That is a one-off harness, not a test in the suite —
running it needs Python's `unicodedata`, which is the thing being replaced.

`--duration S` overrides the whole thing.

### Text longer than a breath

A diffusion model decides its length up front, and OmniVoice was not trained to
hold a voice across a monologue. Past roughly half a minute a single pass stops
sounding like speech at all — the frames are there and the model runs out of
things to put in them. Measured on 40 s of Italian asked for in one pass: RMS
0.023 against the 0.03–0.2 speech band, and a zero-crossing rate of 0.012.
Rumble, not voice.

So above `--chunk-threshold` seconds (30 by default) the text is split into
pieces of about `--chunk-seconds` (15), generated one at a time, and joined.
The same 40 s comes back at RMS 0.102 and a zero-crossing rate of 0.051 —
indistinguishable from a four-second clip.

**Splitting happens at punctuation**, never mid-sentence, because a seam in the
middle of a rising phrase is audible where a seam at a full stop is not.
Sentences merge greedily up to the target, so a chunk overruns only when a
single sentence already does. A full stop that ends a known abbreviation —
`Dr.`, `e.g.`, `No.` — is not a break, and a closing quote or bracket stays
with the sentence it closes rather than opening the next.

**The voice is held by the first chunk.** Every piece after it takes chunk one
as a reference clip, through exactly the machinery voice cloning uses — its
codes go into the prompt as decided frames and its text joins the prompt. Take
that away and each piece invents its own speaker, which is the thing that makes
naive chunking sound like a relay race. With `--ref-audio` the user's clip is
the reference for every chunk instead, and chunk one is no longer special.

**The pieces are joined with a gap, not an overlap.** They are separate
utterances rather than one signal cut in two, so cross-fading them onto each
other would sound like two people talking over one another.

Two things about that join are worth more than they look, and both are
departures from the reference, made after measuring what it produces and then
**confirmed by ear against the reference's own behaviour** — the same standard
the RoPE pairing was settled by, and the only one that can settle a question
about how something sounds.

*Each piece is trimmed to what it actually says.* A chunk's length comes from a
duration estimate, so it ends with however much silence the estimate overshot
by and begins with however much the model took to start. Measured across four
joins of one clip, the pause came out at 113, 127, 191 and **649** ms for the
same intended gap — a hole in the middle of a paragraph. Trimming first makes
every join exactly `--chunk-gap`.

The trim measures RMS over 10 ms windows, not a per-sample peak. That is not a
detail: a chunk's quiet head crosses -50 dBFS on the odd sample while averaging
far below it, so a peak test keeps everything from the first crossing and
leaves half a second of near-silence in the join. A genuinely loud transient is
still kept — that is content.

*The fade is 8 ms, not 100.* The reference ramps a tenth of a second to nothing
at every chunk edge. When the duration estimate is tight — and it usually is —
that ramp lands on speech. Measured on the last 100 ms before four joins, the
level fell 0.176 → 0.011 across the fade: the final syllable of every chunk was
being faded out. At 8 ms the level holds to the edge and the fade only does
what it is for, which is stopping a click.

Measured on an M3 Pro, Metal build of the C++ port:

| text | chunks | audio | wall clock | RMS |
|---|---|---|---|---|
| 683 chars | 3 | 40.6 s | 47 s | 0.102 |
| 1024 chars | 5 | 61.8 s | 64 s | 0.108 |

Roughly linear, and roughly realtime. Nothing caps the total length: 40 000
characters is about 190 chunks and 47 minutes of audio, at 47 minutes of work.

What *is* capped is a single chunk. The GPU engine holds one query's attention
scores in threadgroup memory, which limits a sequence to 2048 positions — and
each chunk after the first carries the reference's frames on top of its own, so
`--chunk-seconds` much past 25 will exceed it. The error says so and names the
number.

### Voice cloning

```bash
cargo run --release -- \
  --model omnivoice \
  --ref-audio giulia.wav \
  --ref-text "Ciao, mi chiamo Giulia. Oggi è una bella giornata a Roma." \
  --prompt "Domani andrò al mercato con mia sorella." \
  --language Italian \
  --out clone.wav
```

A reference clip is not a second mode. It is a prefix:

```
<|denoise|><|lang_start|>...<|instruct_end|>
<|text_start|>{ref_text} {text}<|text_end|>
[ reference frames, decided ][ target frames, masked ]
```

The clip is encoded to codes and those codes go into the sequence as
*already-decided* audio positions, with its transcript joined to the prompt.
So the model is continuing a recording it can see rather than imitating one it
cannot, and every masked position attends to the reference through the same
bidirectional attention it uses for everything else. The unconditional branch
is unchanged — still the masked frames alone — which is what makes guidance
push *towards* the reference voice.

The unmasking loop needed no change: it finds the target frames by counting
back from the end of the sequence, so anything before them is transparent to
it.

`--ref-text` is required. Without it the model cannot tell which part of the
text it has already heard, and would try to say the whole thing again. 3–10 s
of reference is the useful range; the CLI warns past 20.

Two details are the reference implementation's and are not obvious:

**`<|denoise|>` leads the prompt** whenever there is a recording, and only
then. It asks the model to clean the reference up rather than reproduce the
room it was recorded in.

**A quiet clip is levelled before encoding** — brought up to 0.1 RMS — and the
original loudness restored on the way out. The codec was fit on speech at a
particular level, and a quiet recording otherwise encodes into a part of the
codebook space that carries a quiet voice rather than that voice quietly.

Duration estimation switches reference too: with a clip, the estimator
calibrates on *this speaker's* rate rather than on the built-in phrase, which
is strictly better when one is available.

### Reading audio in

Cloning is the first thing here that needs analysis rather than synthesis, and
it pulled in three pieces the project did not have.

**A WAV reader.** `wav.rs` could write a file and not open one. It now takes
8, 16, 24 and 32-bit PCM plus 32 and 64-bit float and WAVE_FORMAT_EXTENSIBLE,
and walks the chunk list rather than assuming the 44-byte header it writes —
almost nothing else writes that, and `LIST`/`INFO` metadata between `fmt ` and
`data` would otherwise be read as samples.

**A resampler.** The codec's two analysis paths run at different rates, so
something has to convert 24 kHz to 16 kHz. It matters that it is *that* filter
and not merely a good one: the features that come out feed a quantizer, so a
different transition band puts the latents somewhere the codebooks were never
fit. `resample.rs` is a port of `torchaudio.functional.resample` at its
defaults, in polyphase form — 24 kHz to 16 kHz is 3 to 2, so two filter phases
of 23 taps and one pass over the input.

**HuBERT.** The codec quantizes the concatenation of an *acoustic* path and a
*semantic* one, so that a code carries what was said and not only how it
sounded. The semantic path is a 94 M-parameter HuBERT, which is the eighth
transformer here and the first that is not a language model:

| | Every other model here | HuBERT |
|---|---|---|
| Input | token ids | a raw waveform |
| Position | RoPE | a grouped convolution, added once |
| Norm placement | before each sublayer | **after each residual add** |
| Norm | RMSNorm | **LayerNorm**, mean subtraction included |
| GELU | tanh approximation | **the erf form** |
| What is used | the last hidden state | **the mean of all thirteen** |

Seven strided convolutions with no padding anywhere reduce 16 kHz audio by
exactly 320, to 50 Hz; the codec keeps every other frame to reach its own 25.
That the two paths arrive at the same frame rate by different arithmetic — 24
kHz over 960 against 16 kHz over 320 and then halved — is the invariant the
whole analysis half rests on, and both ends assert it.

### Checking an encoder with no reference implementation

The decoder could be checked by ear. The encoder cannot: its output is a
thousand integers.

What makes it checkable is that the two halves invert each other. Decode a set
of codes, encode the waveform back, and compare. On real speech codebook 0
recovers **90%** of its indices, falling to 47% by codebook 7 — which is
exactly the shape residual quantization should give, because each codebook only
ever sees the error the ones before it could not represent, and by the eighth
that error is nearly noise. Chance is one in 1024. The waveform correlates at
0.92 with the original and its statistics match to three decimals.

Nothing subtly wrong survives that. A normalisation applied on the wrong axis,
the two paths concatenated in the wrong order, a padding off by one — all of
them land at chance, not near it.

### Diagnosing it

The same rule as Orpheus applies, and harder: every fault sounds identical.
`wave_stats` prints on every run and the exact length invariants hold here too —
each upsampling block multiplies its input length by exactly its stride, and
`frames * 960` has no slack.

One trap is worth naming because no automated check catches it. RoPE can pair
dimension `i` with `i + head_dim/2` (half-split) or `2i` with `2i+1`
(interleaved), and the right answer depends on whether the GGUF converter
permuted the Q and K weight rows — llama.cpp permutes for the `llama`
architecture and not for Qwen3. So Orpheus needs interleaved and OmniVoice needs
half-split, off the same file format.

**Half-split here is confirmed by listening, not by measurement.** Both settings
produce finite, in-range, speech-shaped output that passes every check in the
suite, because the two conventions rotate by the same angles and differ only in
which pairs receive them. Half-split is intelligible Italian; interleaved is
not. The only numeric hint was interleaved's DC offset of -0.03 against
half-split's -0.0001, with a zero-crossing rate of 0.009 — rumble rather than
voice — and that is too weak to have trusted alone. `--rope-interleaved` stays
as the first thing to try when a *new* checkpoint sounds wrong.

The model and its codec are `k2-fsa/OmniVoice` (Apache 2.0, Xiaomi Corp.),
re-uploaded as GGUF by a third party; the backbone is Qwen3-0.6B, also
Apache 2.0.

---

## FLUX text to image

A 12B rectified-flow transformer, its two text encoders and a 16-channel
autoencoder. Text in, PNG out.

```bash
cargo run --release --features metal -- \
  --model flux-schnell \
  --prompt "a photograph of a harbour at dawn, long exposure" \
  --weights models/flux1-schnell-Q4_K_S.gguf \
  --t5 models/t5-v1_1-xxl-encoder-Q4_K_M.gguf \
  --clip models/clip_l.safetensors \
  --clip-tokenizer models/clip_tokenizer.json \
  --vae models/flux_vae.safetensors \
  --width 1024 --height 1024 --steps 4 --seed 42 \
  --out harbour.png
```

`--t5-tokenizer` is optional: the encoder GGUF carries its own SentencePiece
vocabulary, so there is one fewer file to fetch and one fewer way to pair a
checkpoint with the wrong tokenizer.

### What it is

FLUX is not a UNet. It is 19 **double-stream** blocks, where image and text are
separate residual streams with separate weights but a single joint attention
over the concatenation of both, followed by 38 **single-stream** blocks over
that concatenation — with attention and MLP computed in parallel from the same
modulated input and joined before one output projection, rather than run in
sequence.

There is no cross-attention anywhere. The prompt enters as tokens in the text
stream; the timestep and the pooled CLIP vector enter through adaptive layer
norm, which is why every LayerNorm in the model is affine-free. Position is
three-axis RoPE over `(t, h, w)` with per-axis head dimensions 16, 56 and 56 —
image tokens carry their patch-grid coordinates and text tokens carry zeros,
which makes their rotation the identity.

schnell is distilled to four steps **and** guidance-distilled, so there is no
classifier-free guidance and one forward pass per step.

### Memory, on an 18 GB M3 Pro

```
FLUX transformer   11.9B   Q4_K_M    ~6.7 GB
T5-XXL encoder      4.7B   Q4_K      ~2.8 GB   released after encoding
CLIP-L text          123M  F16       ~0.25 GB
VAE decoder           84M  F32       ~0.34 GB
```

The two text encoders run first and are released before the transformer loads;
holding T5 and the transformer at once is the difference between fitting and
not. The autoencoder's last level runs 128 channels at full resolution — half a
gigabyte per activation at 1024x1024 — so `--tile` decodes in overlapping tiles
and blends the seams.

### The Metal engine

Weights stay quantized at rest and are widened per use: before each GEMM, one
dispatch dequantizes that weight into a shared bfloat scratch buffer. This
sounds wasteful and is not — the largest weight is a single-stream block's
fused `linear1` at 3072 → 21504, which is 66 M elements to widen against
575 GFLOP of GEMM to follow, and the scratch buffer is 132 MB reused by every
projection in the model. Widening all 12B at rest would need 24 GB.

The attention kernel is a streaming online softmax rather than the
score-row-in-threadgroup-memory approach the OmniVoice engine uses, which caps
out at 2048 keys; FLUX runs 4352 at 1024x1024.

GPU and CPU agree to **0.5–0.7% RMS-relative**, flat in the number of blocks —
the flatness being what identifies the residual as bfloat rounding in the
matrix unit rather than a structural disagreement, which would compound with
depth.

`--cpu` runs the transformer on the CPU instead, which is correct and slow.

### Weights

| Component | Source | Size |
|---|---|---|
| Transformer | `city96/FLUX.1-schnell-gguf` → `flux1-schnell-Q4_K_S.gguf` | 7.2 GB |
| T5-XXL encoder | `city96/t5-v1_1-xxl-encoder-gguf` → `…-Q4_K_M.gguf` | 2.7 GB |
| CLIP-L | `comfyanonymous/flux_text_encoders` → `clip_l.safetensors` | 235 MB |
| CLIP tokenizer | `openai/clip-vit-large-patch14` → `tokenizer.json` | 2 MB |
| Autoencoder | `John6666/flux1-schnell-fp8-flux` → `vae/diffusion_pytorch_model.safetensors` | 160 MB |

`black-forest-labs/FLUX.1-schnell` is the canonical home of the autoencoder and
the tokenizers, but it is gated, so the table points at ungated mirrors. Both
autoencoder naming conventions load — see the design note for why that
distinction is more than cosmetic.

FLUX.1-schnell is Apache 2.0. `--model flux-dev` runs the dev config — same
architecture plus a distilled-guidance embedding, 28 steps, non-commercial
licence — but its weights are gated and have not been tested.

### Speed, measured on an 18 GB M3 Pro (C++ port)

| Resolution | Tokens | Per step | 4 steps | Total incl. VAE decode |
|---|---|---|---|---|
| 512x512 | 1280 | 10.2 s | 41 s | 45 s |
| 1024x1024 | 4352 | 40.6 s | 162 s | 192 s |

The attention kernel went through two rounds. Comparing the two sizes says how
much of a step it is without needing a profiler — about half at 1024x1024 —
and it started out reading the whole of K and V once per *query*, which is
460 GB of traffic per call. Blocking it over 32 queries per threadgroup divides
that by 32; putting `simdgroup_matrix` on both of its matmuls took the rest:

| | 1024x1024 |
|---|---|
| one query per threadgroup, scalar | 156 s/step |
| 32-query block, scalar | 105 s/step |
| 32-query block, matrix units | **43.5 s/step** |

3.6x on sampling, identical output. Two changes along the way that looked like
obvious wins both made it *slower*; the
[design note](docs/text-to-image.md#blocking-the-attention) records which and
why.

The autoencoder went the same way: its mid-block attention was 76 s of a 102 s
decode as a triple loop, and 1.08 s once both of its products were `sgemm`
calls. **1024x1024 end to end: 688 s → 192 s.**

The arithmetic is checked against `diffusers` reading the same GGUF file —
token ids identical, dequantization bit-exact, whole-transformer velocity
agreeing to 4.4e-3. That comparison
[found a real bug](docs/text-to-image.md#parity-against-diffusers): the
batch-of-one GEMV shortcut quantizes activations to int8, and the only
batch-of-one matmuls here are the modulation projections, whose scale and gate
multiply every token in the block.

T5 encoding is 6.7 s. The autoencoder decodes on the CPU.

### Status

Working end to end against the real checkpoints in the C++ port this stack
came from, where `--model flux-schnell` at 1024x1024 in four steps draws what
was asked for. Here the modules were ported from it and checked against it
stage by stage; the Rust CLI takes the same flags, validates them the same way
and drives the same pipeline.

What is *not* verified is numerical parity with diffusers: the output is a
photograph of what was asked for, which rules out every bug that produces noise
or the wrong subject, but it does not prove a given seed reproduces the
reference implementation's exact image.
[docs/text-to-image.md](docs/text-to-image.md) has the full design, the build
log, the six bugs that stood between passing tests and a working image, and the
rest of what remains unverified.

---

## Running tests

```bash
cargo test --features metal    # 412 tests
```

Covers every component: matrix ops, gradient correctness (verified with finite differences), attention shapes, Flash Attention, Q4K/BF16 quantization, Metal GPU kernels, Gemma 3 architecture, tokenizer encoding/decoding.

The speech and image stacks bring their own tests, ported one for one from the C++ suite: 1-D and 2-D convolution against index-by-index references, both codec decoders and the OmniVoice encoder, RIFF in both directions, resampling against the reference filter, the duration estimator's character classes, the unmask schedule, the autoencoder's block algebra and tile blending, PNG containers chunk by chunk, T5's relative-position bucketing, CLIP's causal mask and pooling, and FLUX's patch order and three-axis RoPE. Most need no weights. The ones that do look in `models/` (relative to the crate root) and skip with a message when the file is missing:

```
models/omnivoice-base-Q8_0.gguf  models/omnivoice-tokenizer-Q8_0.gguf  models/snac_24khz.bin
models/t5-v1_1-xxl-encoder-Q4_K_M.gguf  models/clip_l.safetensors  models/clip_tokenizer.json
models/flux_vae.safetensors  plus an Orpheus GGUF and a FLUX transformer GGUF
```

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
├── transformer_qwen35.rs  Qwen 3.5 (Gated DeltaNet + softmax attention hybrid)
├── transformer5.rs    Llama 3.2 (Orpheus): RoPE pairings, GGUF loading, embedded tokenizer
├── transformer6.rs    Bidirectional Qwen3 (OmniVoice), multi-codebook head
├── hubert.rs          HuBERT: waveform in, semantic features out
│
├── conv1d.rs          1-D convolution: dilated, grouped, transposed; Snake, im2col
├── snac.rs            SNAC 24 kHz codec decoder and residual vector quantizer
├── orpheus.rs         Audio-token protocol, frame de-interleaving, synthesis
├── omnivoice.rs       Masked-diffusion sampler: schedules, guidance, unmasking
├── omnivoice_codec.rs OmniVoice codec, both directions — 8 codebooks, 960x
├── duration.rs        Rule-based duration estimation from character weights
├── resample.rs        Polyphase windowed-sinc sample rate conversion
├── torch_pickle.rs    PyTorch .bin reader: ZIP container + pickle manifest
├── wav.rs             RIFF reading and writing, and waveform statistics
│
├── conv2d.rs          2-D convolution, nearest upsampling, GroupNorm
├── vae.rs             16-channel AutoencoderKL decoder, whole and tiled
├── qlinear.rs         Inference-only linear over f32 / BF16 / Q4_K weights
├── t5.rs              T5 v1.1 encoder: relative position bias, gated GELU
├── clip_text.rs       CLIP-L text tower and its pooled output
├── flux.rs            FLUX MMDiT, three-axis RoPE, rectified-flow sampler
├── png.rs             PNG writing with stored DEFLATE blocks, image statistics
│
├── gguf_loader.rs     GGUF file parser (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0, BF16, F16, F32)
├── metal_ops.rs       Metal GPU kernels (tiled matmul, per-dispatch GEMV)
├── metal_decode.rs    Full-graph Metal decode engine (single command buffer)
├── metal_decode_qwen35.rs  Full-graph Metal decode for Qwen 3.5
├── metal_omnivoice.rs Full-graph Metal forward pass for OmniVoice
├── metal_flux.rs      Full-graph Metal forward pass for FLUX
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
