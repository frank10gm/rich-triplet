//! # Metal full-graph forward pass (OmniVoice)
//!
//! The first Metal engine here for a text-to-speech model, and the first for a
//! *prefill* shape rather than a decode one. The two Gemma and Qwen engines
//! exist because a decode step is a chain of GEMVs, memory-bound and too small
//! to dispatch one at a time. This one exists for the opposite reason:
//! OmniVoice has no KV cache, so every step is a full-sequence forward pass over
//! a few hundred positions -- large GEMMs, compute-bound, and exactly the shape
//! a GPU is built for.
//!
//! ## Why it is worth it, measured rather than assumed
//!
//! Accelerate's sgemm reaches about 1.0 TFLOP/s on an M3 Pro for these shapes,
//! which is a high bar -- the tiled matmul in `metal_ops.rs` manages 250 GF/s
//! and would be a large regression. What clears the bar is `simdgroup_matrix`:
//! a 64x64 tile per threadgroup built from sixteen 8x8 accumulators reaches
//! 1.2-2.3 TFLOP/s depending on shape, since the matrix units do the work the
//! tiled kernel does with scalar multiply-adds.
//!
//! The GEMMs are only 40% of the CPU runtime, though, so moving them alone
//! would cap the win near 1.3x. The rest goes to three things the GPU gets for
//! free:
//!
//!   * **BF16 weights are read natively.** The CPU path dequantizes every
//!     weight to an f32 scratch buffer on every call -- 12% of its runtime, and
//!     437 million conversions per forward pass. A Metal kernel reads `bfloat`
//!     as an operand.
//!   * **No zero-fill.** `Mat::zeros` costs 8% of the CPU runtime in `bzero`
//!     for buffers that sgemm immediately overwrites. GPU scratch is allocated
//!     once and reused.
//!   * **The elementwise work moves too.** RMSNorm, RoPE, SiLU, the attention
//!     softmax and the residual adds are another quarter of the runtime, all of
//!     it memory-bound scalar loops.
//!
//! ## Structure
//!
//! One command buffer per forward pass: 28 layers of 11 dispatches plus 3, so
//! about 311. Within a single compute encoder on Apple Silicon dispatches run in
//! order and each one's writes are visible to the next through the unified L2,
//! so there are no barriers.
//!
//! Sequence length is padded up to a multiple of 64 so the GEMM tiles divide
//! evenly and need no bounds checks in their inner loop. The padding rows are
//! zeroed once and carry through harmlessly -- every operator except attention
//! is row-independent, and attention is dispatched over the true length.
//!
//! The RoPE kernel is half-split only: `Config6::rope_pairing` is not consulted
//! here, exactly as in the C++ engine.

#![allow(dead_code)]

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::autograd2::{Mat, MatBf16};
use crate::nn2::Linear2;
use crate::transformer6::{Config6, OmniForward, OmniLm, OmniToken};

// =============================================================================
// MSL kernel source
// =============================================================================

/// Every kernel of the engine, compiled as one library. Verbatim copy of the C++
/// tree's `src/shaders/omnivoice.msl`, including the leading newline the C++
/// embedding adds.
const OMNIVOICE_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// =============================================================================
// GEMM
// =============================================================================
//
// C[M,N] = A[M,K] @ W[N,K]^T, with the weight in bfloat exactly as the
// checkpoint stores it -- there is no dequantization step, because
// `simdgroup_matrix` takes bfloat as an operand and the matrix units convert
// on the way in.
//
// One threadgroup is four simdgroups covering a 64x64 tile of C, each owning a
// 32x32 quadrant as sixteen 8x8 accumulators. That register blocking is the
// whole difference against the scalar tiled kernel in metal_ops.mm: every 8x8
// operand loaded from memory feeds four multiply-accumulates instead of one.
//
// M and N are always multiples of 64 here, which is why the inner loop has no
// bounds checks. The caller pads the sequence length and the weight rows.

kernel void omni_gemm_bt(
    device const float*  A [[buffer(0)]],
    device const bfloat* W [[buffer(1)]],
    device       float*  C [[buffer(2)]],
    constant uint& K [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]])
{
    const uint m0 = tgid.y * 64u + (sgid / 2u) * 32u;
    const uint n0 = tgid.x * 64u + (sgid % 2u) * 32u;

    simdgroup_float8x8 acc[4][4];
    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    simdgroup_float8x8  a[4];
    simdgroup_bfloat8x8 b[4];

    for (uint k0 = 0; k0 < K; k0 += 8u) {
        for (uint i = 0; i < 4; ++i)
            simdgroup_load(a[i], A + (m0 + i * 8u) * K + k0, K);
        // transpose=true reads W[n,k] as the [k,n] operand the product wants,
        // so the weight never has to be stored twice.
        for (uint j = 0; j < 4; ++j)
            simdgroup_load(b[j], W + (n0 + j * 8u) * K + k0, K, ulong2(0, 0), true);
        for (uint i = 0; i < 4; ++i)
            for (uint j = 0; j < 4; ++j)
                simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
    }

    for (uint i = 0; i < 4; ++i)
        for (uint j = 0; j < 4; ++j)
            simdgroup_store(acc[i][j], C + (m0 + i * 8u) * N + (n0 + j * 8u), N);
}

// =============================================================================
// Normalisation
// =============================================================================

// RMSNorm over each row: out = x / sqrt(mean(x^2) + eps) * weight.
//
// One threadgroup per row, 256 threads reducing through threadgroup memory.
kernel void omni_rms_norm(
    device const float*  x   [[buffer(0)]],
    device const float*  w   [[buffer(1)]],
    device       float*  out [[buffer(2)]],
    constant uint&  D   [[buffer(3)]],
    constant float& eps [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint nth [[threads_per_threadgroup]])
{
    threadgroup float partial[256];
    device const float* src = x + row * D;

    float sum = 0.0f;
    for (uint i = tid; i < D; i += nth) {
        const float v = src[i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(D) + eps);
    device float* dst = out + row * D;
    for (uint i = tid; i < D; i += nth)
        dst[i] = src[i] * scale * w[i];
}

// Qwen3 normalises each head of Q and K on its own before rotating, so this is
// an RMSNorm over `head_dim` slices rather than the whole row. One threadgroup
// per (row, head).
kernel void omni_head_rms_norm(
    device       float* x   [[buffer(0)]],
    device const float* w   [[buffer(1)]],
    constant uint&  head_dim [[buffer(2)]],
    constant uint&  n_heads  [[buffer(3)]],
    constant float& eps      [[buffer(4)]],
    // MSL wants every position attribute in a kernel to have the same width,
    // so these are uint2 with an unused y rather than plain uints.
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid_v [[thread_position_in_threadgroup]],
    uint2 nth_v [[threads_per_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint nth = nth_v.x;
    threadgroup float partial[128];
    device float* slice = x + gid.y * (n_heads * head_dim) + gid.x * head_dim;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nth) {
        const float v = slice[i];
        sum += v * v;
    }
    partial[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) partial[tid] += partial[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = rsqrt(partial[0] / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += nth)
        slice[i] = slice[i] * scale * w[i];
}

// =============================================================================
// Rotary embedding
// =============================================================================

// Half-split pairing: dimension i rotates with i + head_dim/2.
//
// Positions run from zero every pass, because there is no cache -- the whole
// sequence is rotated each time.
kernel void omni_rope(
    device       float* x        [[buffer(0)]],
    device const float* inv_freq [[buffer(1)]],
    constant uint& head_dim [[buffer(2)]],
    constant uint& n_heads  [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint half_dim = head_dim / 2u;
    if (gid.x >= half_dim) return;

    device float* slice = x + gid.z * (n_heads * head_dim) + gid.y * head_dim;
    const float angle = float(gid.z) * inv_freq[gid.x];
    const float c = cos(angle);
    const float s = sin(angle);

    const float lo = slice[gid.x];
    const float hi = slice[gid.x + half_dim];
    slice[gid.x] = lo * c - hi * s;
    slice[gid.x + half_dim] = hi * c + lo * s;
}

// =============================================================================
// Attention
// =============================================================================

// Bidirectional grouped-query attention, one threadgroup per (query, head).
//
// No mask of any kind: a masked diffusion model conditions each position on
// both sides, so every query attends to every key. The scores for one query
// live in threadgroup memory, which caps the sequence length this kernel
// accepts -- the host checks it before dispatching. 2048 floats is 8 KB of the
// 32 KB a threadgroup may hold, so the cap is a choice about scratch buffers
// rather than a hardware limit.
kernel void omni_attention(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device       float* out [[buffer(3)]],
    constant uint&  T          [[buffer(4)]],
    constant uint&  n_q_heads  [[buffer(5)]],
    constant uint&  n_kv_heads [[buffer(6)]],
    constant uint&  head_dim   [[buffer(7)]],
    constant float& scale      [[buffer(8)]],
    // MSL wants every position attribute in a kernel to have the same width,
    // so these are uint2 with an unused y rather than plain uints.
    uint2 gid [[threadgroup_position_in_grid]],
    uint2 tid_v [[thread_position_in_threadgroup]],
    uint2 nth_v [[threads_per_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint nth = nth_v.x;
    threadgroup float scores[2048];
    threadgroup float reduce[256];

    const uint qh = gid.x;
    const uint row = gid.y;
    const uint kvh = qh / (n_q_heads / n_kv_heads);

    device const float* qv = q + row * (n_q_heads * head_dim) + qh * head_dim;
    device const float* kb = k + kvh * head_dim;
    device const float* vb = v + kvh * head_dim;
    const uint kv_stride = n_kv_heads * head_dim;

    // 1. scores
    for (uint t = tid; t < T; t += nth) {
        device const float* kv = kb + t * kv_stride;
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) dot += qv[d] * kv[d];
        scores[t] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. softmax, in two reductions
    float local_max = -INFINITY;
    for (uint t = tid; t < T; t += nth) local_max = max(local_max, scores[t]);
    reduce[tid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) reduce[tid] = max(reduce[tid], reduce[tid + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float row_max = reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_sum = 0.0f;
    for (uint t = tid; t < T; t += nth) {
        const float e = exp(scores[t] - row_max);
        scores[t] = e;
        local_sum += e;
    }
    reduce[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv_sum = 1.0f / reduce[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. weighted sum of V, one output dimension per thread
    device float* dst = out + row * (n_q_heads * head_dim) + qh * head_dim;
    for (uint d = tid; d < head_dim; d += nth) {
        float acc = 0.0f;
        for (uint t = 0; t < T; ++t) acc += scores[t] * vb[t * kv_stride + d];
        dst[d] = acc * inv_sum;
    }
}

// =============================================================================
// Elementwise
// =============================================================================

// SwiGLU's gate: gate = silu(gate) * up, in place.
kernel void omni_silu_mul(
    device       float* gate [[buffer(0)]],
    device const float* up   [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    const float g = gate[i];
    gate[i] = (g / (1.0f + exp(-g))) * up[i];
}

kernel void omni_add(
    device       float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] += src[i];
}

kernel void omni_zero(
    device float* dst [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] = 0.0f;
}
"#;

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

// A GEMM tile is 64x64, so both the sequence length and every weight's row
// count are rounded up to that. The pad rows are zero and stay harmless: every
// operator except attention is row-independent, and attention is dispatched
// over the true length.
const TILE: usize = 64;

// The attention kernel keeps one query's scores in threadgroup memory.
const MAX_ATTENTION_LENGTH: usize = 2048;

fn round_up(v: usize, to: usize) -> usize {
    (v + to - 1) / to * to
}

/// Truncating f32 -> bf16, as the C++ upload does (not the rounding
/// `MatBf16::f32_to_bf16`). Only reached for a projection held as f32.
fn f32_bits_to_bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn set_buffer(enc: &Encoder, buf: &Buffer, index: usize) {
    unsafe { enc.setBuffer_offset_atIndex(Some(buf), 0, index) };
}

fn set_bytes<T>(enc: &Encoder, value: &T, index: usize) {
    unsafe {
        enc.setBytes_length_atIndex(
            NonNull::from(value).cast::<c_void>(),
            std::mem::size_of::<T>(),
            index,
        )
    };
}

fn size(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize { width, height, depth }
}

// =============================================================================
// Weights and scratch
// =============================================================================

struct Layer {
    input_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    down: Buffer,
}

/// Allocates shared buffers and counts the bytes, for `buffer_bytes`.
struct Uploader<'a> {
    device: &'a ProtocolObject<dyn MTLDevice>,
    bytes: usize,
}

impl Uploader<'_> {
    fn alloc(&mut self, n_bytes: usize) -> Buffer {
        let b = self
            .device
            .newBufferWithLength_options(n_bytes, MTLResourceOptions::StorageModeShared)
            .expect("metal omnivoice: buffer allocation failed");
        self.bytes += n_bytes;
        b
    }

    /// Upload a projection's BF16 weight, padding its row count to a tile
    /// multiple, and free the CPU copy.
    fn upload_projection(
        &mut self,
        linear: &mut Linear2,
        out_features: usize,
        in_features: usize,
    ) -> Buffer {
        let rows = round_up(out_features, TILE);
        let mut padded = vec![0u16; rows * in_features];

        if let Some(w) = &linear.bf16_weight {
            let n = w.data.len().min(out_features * in_features);
            padded[..n].copy_from_slice(&w.data[..n]);
        } else {
            let w = linear.weight.data();
            let n = (out_features * in_features).min(w.data.len());
            for i in 0..n {
                padded[i] = f32_bits_to_bf16(w.data[i]);
            }
        }

        let buf = self.alloc(padded.len() * std::mem::size_of::<u16>());
        unsafe {
            std::ptr::copy_nonoverlapping(
                padded.as_ptr(),
                buf.contents().as_ptr() as *mut u16,
                padded.len(),
            );
        }
        linear.clear_weight_data();
        buf
    }

    fn upload_floats(&mut self, v: &[f32]) -> Buffer {
        let buf = self.alloc(v.len().max(1) * std::mem::size_of::<f32>());
        if !v.is_empty() {
            unsafe {
                let dst = buf.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(v.as_ptr(), dst, v.len());
            }
        }
        buf
    }
}

fn make_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
) -> Result<Pipeline, String> {
    let Some(func) = library.newFunctionWithName(&NSString::from_str(name)) else {
        return Err(format!("metal omnivoice: no kernel named '{}'", name));
    };
    device.newComputePipelineStateWithFunction_error(&func).map_err(|e| {
        format!("metal omnivoice: pipeline '{}' failed: {}", name, e.localizedDescription())
    })
}

// =============================================================================
// Context
// =============================================================================

/// Owns every GPU resource a forward pass needs: uploaded weights, activation
/// scratch, and the compiled pipelines.
pub struct MetalOmniContext {
    config: Config6,
    max_tokens: usize,
    max_padded: usize,
    /// Audio head rows, padded to a tile multiple.
    head_rows: usize,

    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    gemm: Pipeline,
    rms_norm: Pipeline,
    head_rms_norm: Pipeline,
    rope: Pipeline,
    attention: Pipeline,
    silu_mul: Pipeline,
    add: Pipeline,
    zero: Pipeline,

    layers: Vec<Layer>,

    final_norm: Buffer,
    head: Buffer,
    inv_freq: Buffer,

    /// The embedding tables stay in CPU memory and are shared with the model
    /// rather than copied -- `MatBf16` is reference-counted, so this costs a
    /// pointer. Only `T` rows are read per pass, and the text table is 310 MB.
    text_embed: MatBf16,
    audio_embed: MatBf16,

    // Activation scratch, all sized for `max_padded` rows.
    x: Buffer,
    normed: Buffer,
    buf_q: Buffer,
    buf_k: Buffer,
    buf_v: Buffer,
    attn: Buffer,
    proj: Buffer,
    buf_gate: Buffer,
    buf_up: Buffer,
    logits: Buffer,

    bytes: usize,
}

impl MetalOmniContext {
    /// The longest sequence the attention kernel can hold -- it keeps one
    /// query's scores in threadgroup memory, 8 KB of the 32 KB a threadgroup
    /// may use. Long text is split into chunks well below this, so what the
    /// headroom buys is room for a generous `--chunk-seconds` and for the
    /// reference clip every chunk after the first carries.
    pub const MAX_TOKENS: usize = 2048;

    /// Upload `lm`'s weights and allocate scratch for sequences up to
    /// `max_tokens`. `None` (with the reason on stderr) when the GPU cannot be
    /// set up.
    ///
    /// **Consumes the model's projection weights.** Each is freed as it is
    /// uploaded, so peak memory never holds both copies -- and `lm` cannot run
    /// a forward pass of its own afterwards. It is still needed for its config,
    /// its embedding tables and its prompt layout.
    ///
    /// The embedding tables stay on the CPU: the text table alone is 310 MB and
    /// only `T` of its rows are ever read, so gathering on the CPU and
    /// uploading the result is cheaper than holding it twice.
    pub fn create(lm: &mut OmniLm, max_tokens: usize) -> Option<MetalOmniContext> {
        let config = lm.config.clone();
        let max_padded = round_up(max_tokens.max(1), TILE);

        let Some(device) = MTLCreateSystemDefaultDevice() else {
            eprintln!("[ Metal ] no GPU device");
            return None;
        };
        let Some(queue) = device.newCommandQueue() else {
            eprintln!("[ Metal ] no command queue");
            return None;
        };

        let source = NSString::from_str(OMNIVOICE_MSL);
        let library = match device.newLibraryWithSource_options_error(&source, None) {
            Ok(library) => library,
            Err(e) => {
                eprintln!("[ Metal ] shader compilation failed: {}", e.localizedDescription());
                return None;
            }
        };

        let pipelines = (|| -> Result<[Pipeline; 8], String> {
            Ok([
                make_pipeline(&device, &library, "omni_gemm_bt")?,
                make_pipeline(&device, &library, "omni_rms_norm")?,
                make_pipeline(&device, &library, "omni_head_rms_norm")?,
                make_pipeline(&device, &library, "omni_rope")?,
                make_pipeline(&device, &library, "omni_attention")?,
                make_pipeline(&device, &library, "omni_silu_mul")?,
                make_pipeline(&device, &library, "omni_add")?,
                make_pipeline(&device, &library, "omni_zero")?,
            ])
        })();
        let [gemm, rms_norm, head_rms_norm, rope, attention, silu_mul, add, zero] = match pipelines
        {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[ Metal ] {}", e);
                return None;
            }
        };

        let c = &config;
        let hidden = c.hidden_size;
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;
        let mut up = Uploader { device: &device, bytes: 0 };

        // -- weights ---------------------------------------------------------
        let mut layers = Vec::with_capacity(lm.layers.len());
        for block in lm.layers.iter_mut() {
            let input_norm = up.upload_floats(&block.input_layernorm.gamma.data().data);
            let post_norm = up.upload_floats(&block.post_attention_layernorm.gamma.data().data);
            let q_norm = up.upload_floats(&block.self_attn.q_norm.gamma.data().data);
            let k_norm = up.upload_floats(&block.self_attn.k_norm.gamma.data().data);

            let q = up.upload_projection(&mut block.self_attn.q_proj, q_dim, hidden);
            let k = up.upload_projection(&mut block.self_attn.k_proj, kv_dim, hidden);
            let v = up.upload_projection(&mut block.self_attn.v_proj, kv_dim, hidden);
            let o = up.upload_projection(&mut block.self_attn.o_proj, hidden, q_dim);
            let gate = up.upload_projection(&mut block.mlp.gate_proj, c.intermediate_size, hidden);
            let up_w = up.upload_projection(&mut block.mlp.up_proj, c.intermediate_size, hidden);
            let down = up.upload_projection(&mut block.mlp.down_proj, hidden, c.intermediate_size);
            layers.push(Layer {
                input_norm,
                post_norm,
                q_norm,
                k_norm,
                q,
                k,
                v,
                o,
                gate,
                up: up_w,
                down,
            });
        }

        let (Some(text_embed), Some(audio_embed)) = (lm.text_embed.clone(), lm.audio_embed.clone())
        else {
            eprintln!("[ Metal ] the model has no embedding tables loaded");
            return None;
        };

        let final_norm = up.upload_floats(&lm.norm.gamma.data().data);
        let head_rows = round_up(c.audio_table_size(), TILE);
        let head = up.upload_projection(&mut lm.audio_head, c.audio_table_size(), hidden);
        let inv_freq = up.upload_floats(&lm.inv_freq_cache);

        // -- scratch ---------------------------------------------------------
        let rows = max_padded;
        let f32_size = std::mem::size_of::<f32>();
        let x = up.alloc(rows * hidden * f32_size);
        let normed = up.alloc(rows * hidden * f32_size);
        let proj = up.alloc(rows * hidden * f32_size);
        let buf_q = up.alloc(rows * q_dim * f32_size);
        let buf_k = up.alloc(rows * kv_dim * f32_size);
        let buf_v = up.alloc(rows * kv_dim * f32_size);
        let attn = up.alloc(rows * q_dim * f32_size);
        let buf_gate = up.alloc(rows * c.intermediate_size * f32_size);
        let buf_up = up.alloc(rows * c.intermediate_size * f32_size);
        let logits = up.alloc(rows * head_rows * f32_size);
        let bytes = up.bytes;

        Some(MetalOmniContext {
            config,
            max_tokens,
            max_padded,
            head_rows,
            device,
            queue,
            gemm,
            rms_norm,
            head_rms_norm,
            rope,
            attention,
            silu_mul,
            add,
            zero,
            layers,
            final_norm,
            head,
            inv_freq,
            text_embed,
            audio_embed,
            x,
            normed,
            buf_q,
            buf_k,
            buf_v,
            attn,
            proj,
            buf_gate,
            buf_up,
            logits,
            bytes,
        })
    }

    /// Longest sequence the allocated scratch can take.
    pub fn max_tokens(&self) -> usize {
        self.max_tokens
    }

    /// Bytes held in GPU buffers.
    pub fn buffer_bytes(&self) -> usize {
        self.bytes
    }

    /// Run the whole sequence and return audio logits, `[T, codebooks * vocab]`
    /// -- the same contract as `OmniLm::forward`.
    pub fn forward(&self, tokens: &[OmniToken]) -> Result<Mat, String> {
        let c = &self.config;

        if tokens.is_empty() {
            return Err("metal omnivoice: empty sequence".into());
        }
        if tokens.len() > self.max_tokens {
            return Err(format!(
                "metal omnivoice: {} tokens exceeds the {} this context was built for",
                tokens.len(),
                self.max_tokens
            ));
        }
        if tokens.len() > MAX_ATTENTION_LENGTH {
            return Err(format!(
                "metal omnivoice: the attention kernel holds at most {} positions",
                MAX_ATTENTION_LENGTH
            ));
        }

        let t = tokens.len();
        let tp = round_up(t, TILE);
        let hidden = c.hidden_size;
        let q_dim = c.num_attention_heads * c.head_dim;
        let kv_dim = c.num_key_value_heads * c.head_dim;

        // The embedding tables stay on the CPU: only `t` rows are read, and the
        // text table is 310 MB.
        {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(self.x.contents().as_ptr() as *mut f32, tp * hidden)
            };
            dst.fill(0.0);
            let text_bits: &[u16] = &self.text_embed.data;
            let audio_bits: &[u16] = &self.audio_embed.data;
            for (i, token) in tokens.iter().enumerate() {
                let row = &mut dst[i * hidden..(i + 1) * hidden];
                if token.is_audio() {
                    if token.audio.len() != c.num_audio_codebook {
                        return Err(format!(
                            "metal omnivoice: position {} has {} codebooks",
                            i,
                            token.audio.len()
                        ));
                    }
                    // A position's embedding is the sum across all eight
                    // codebooks, so a fully masked one is still well defined.
                    for cb in 0..c.num_audio_codebook {
                        let code = token.audio[cb] as usize;
                        if code >= c.audio_vocab_size {
                            return Err(format!("metal omnivoice: code {} is out of range", code));
                        }
                        let r = c.audio_row(cb, code);
                        let src = &audio_bits[r * self.audio_embed.cols..][..hidden];
                        for d in 0..hidden {
                            row[d] += f32::from_bits((src[d] as u32) << 16);
                        }
                    }
                } else {
                    if token.text_id >= c.text_vocab_size {
                        return Err(format!(
                            "metal omnivoice: text id {} is out of range",
                            token.text_id
                        ));
                    }
                    let src = &text_bits[token.text_id * self.text_embed.cols..][..hidden];
                    for d in 0..hidden {
                        row[d] = f32::from_bits((src[d] as u32) << 16);
                    }
                }
            }
        }

        let Some(cmd) = self.queue.commandBuffer() else {
            return Err("metal omnivoice: no command buffer".into());
        };
        let Some(enc) = cmd.computeCommandEncoder() else {
            return Err("metal omnivoice: no compute encoder".into());
        };
        let enc: &Encoder = &enc;

        let hidden32 = hidden as u32;
        let head_dim32 = c.head_dim as u32;
        let n_q32 = c.num_attention_heads as u32;
        let n_kv32 = c.num_key_value_heads as u32;
        let t32 = t as u32;
        let eps: f32 = c.rms_norm_eps;
        let attn_scale = 1.0f32 / (c.head_dim as f32).sqrt();

        let encode_norm = |src: &Buffer, w: &Buffer, dst: &Buffer| {
            enc.setComputePipelineState(&self.rms_norm);
            set_buffer(enc, src, 0);
            set_buffer(enc, w, 1);
            set_buffer(enc, dst, 2);
            set_bytes(enc, &hidden32, 3);
            set_bytes(enc, &eps, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(size(tp, 1, 1), size(256, 1, 1));
        };

        let encode_head_norm = |buf: &Buffer, w: &Buffer, heads: usize| {
            let heads32 = heads as u32;
            enc.setComputePipelineState(&self.head_rms_norm);
            set_buffer(enc, buf, 0);
            set_buffer(enc, w, 1);
            set_bytes(enc, &head_dim32, 2);
            set_bytes(enc, &heads32, 3);
            set_bytes(enc, &eps, 4);
            enc.dispatchThreadgroups_threadsPerThreadgroup(size(heads, t, 1), size(128, 1, 1));
        };

        let encode_rope = |buf: &Buffer, heads: usize| {
            let heads32 = heads as u32;
            enc.setComputePipelineState(&self.rope);
            set_buffer(enc, buf, 0);
            set_buffer(enc, &self.inv_freq, 1);
            set_bytes(enc, &head_dim32, 2);
            set_bytes(enc, &heads32, 3);
            enc.dispatchThreads_threadsPerThreadgroup(
                size(c.head_dim / 2, heads, t),
                size((c.head_dim / 2).min(64), 1, 1),
            );
        };

        for layer in &self.layers {
            encode_norm(&self.x, &layer.input_norm, &self.normed);

            encode_gemm(enc, &self.gemm, &self.normed, &layer.q, &self.buf_q, tp, hidden, q_dim);
            encode_gemm(enc, &self.gemm, &self.normed, &layer.k, &self.buf_k, tp, hidden, kv_dim);
            encode_gemm(enc, &self.gemm, &self.normed, &layer.v, &self.buf_v, tp, hidden, kv_dim);

            encode_head_norm(&self.buf_q, &layer.q_norm, c.num_attention_heads);
            encode_head_norm(&self.buf_k, &layer.k_norm, c.num_key_value_heads);
            encode_rope(&self.buf_q, c.num_attention_heads);
            encode_rope(&self.buf_k, c.num_key_value_heads);

            enc.setComputePipelineState(&self.attention);
            set_buffer(enc, &self.buf_q, 0);
            set_buffer(enc, &self.buf_k, 1);
            set_buffer(enc, &self.buf_v, 2);
            set_buffer(enc, &self.attn, 3);
            set_bytes(enc, &t32, 4);
            set_bytes(enc, &n_q32, 5);
            set_bytes(enc, &n_kv32, 6);
            set_bytes(enc, &head_dim32, 7);
            set_bytes(enc, &attn_scale, 8);
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                size(c.num_attention_heads, t, 1),
                size(256, 1, 1),
            );

            encode_gemm(enc, &self.gemm, &self.attn, &layer.o, &self.proj, tp, q_dim, hidden);
            encode_elementwise(enc, &self.add, &self.x, Some(&self.proj), tp * hidden);

            encode_norm(&self.x, &layer.post_norm, &self.normed);
            encode_gemm(
                enc,
                &self.gemm,
                &self.normed,
                &layer.gate,
                &self.buf_gate,
                tp,
                hidden,
                c.intermediate_size,
            );
            encode_gemm(
                enc,
                &self.gemm,
                &self.normed,
                &layer.up,
                &self.buf_up,
                tp,
                hidden,
                c.intermediate_size,
            );
            encode_elementwise(
                enc,
                &self.silu_mul,
                &self.buf_gate,
                Some(&self.buf_up),
                tp * c.intermediate_size,
            );
            encode_gemm(
                enc,
                &self.gemm,
                &self.buf_gate,
                &layer.down,
                &self.proj,
                tp,
                c.intermediate_size,
                hidden,
            );
            encode_elementwise(enc, &self.add, &self.x, Some(&self.proj), tp * hidden);
        }

        encode_norm(&self.x, &self.final_norm, &self.normed);
        encode_gemm(
            enc,
            &self.gemm,
            &self.normed,
            &self.head,
            &self.logits,
            tp,
            hidden,
            self.head_rows,
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
        if let Some(e) = cmd.error() {
            return Err(format!(
                "metal omnivoice: command buffer failed: {}",
                e.localizedDescription()
            ));
        }

        // The logits buffer is padded on both axes; the caller wants neither pad.
        let table = c.audio_table_size();
        let mut out = Mat::zeros(t, table);
        let src = unsafe {
            let ptr = self.logits.contents().as_ptr() as *const f32;
            std::slice::from_raw_parts(ptr, t * self.head_rows)
        };
        for r in 0..t {
            out.data[r * table..(r + 1) * table]
                .copy_from_slice(&src[r * self.head_rows..r * self.head_rows + table]);
        }
        Ok(out)
    }
}

impl OmniForward for MetalOmniContext {
    fn forward(&self, tokens: &[OmniToken]) -> Result<Mat, String> {
        MetalOmniContext::forward(self, tokens)
    }
}

// =============================================================================
// Encoding helpers
// =============================================================================

/// `C[Mp, N] = A[Mp, K] @ W[N, K]^T`, both dimensions already tile-aligned.
#[allow(clippy::too_many_arguments)]
fn encode_gemm(
    enc: &Encoder,
    pso: &Pipeline,
    a: &Buffer,
    w: &Buffer,
    c: &Buffer,
    m_padded: usize,
    k: usize,
    n_padded: usize,
) {
    let k32 = k as u32;
    let n32 = n_padded as u32;
    enc.setComputePipelineState(pso);
    set_buffer(enc, a, 0);
    set_buffer(enc, w, 1);
    set_buffer(enc, c, 2);
    set_bytes(enc, &k32, 3);
    set_bytes(enc, &n32, 4);
    // Four simdgroups of 32 threads cover the 64x64 tile.
    enc.dispatchThreadgroups_threadsPerThreadgroup(
        size(n_padded / TILE, m_padded / TILE, 1),
        size(128, 1, 1),
    );
}

fn encode_elementwise(enc: &Encoder, pso: &Pipeline, dst: &Buffer, src: Option<&Buffer>, n: usize) {
    enc.setComputePipelineState(pso);
    set_buffer(enc, dst, 0);
    if let Some(src) = src {
        set_buffer(enc, src, 1);
    }
    let width = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatchThreads_threadsPerThreadgroup(size(n, 1, 1), size(width, 1, 1));
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const LM_PATH: &str = "models/omnivoice-base-Q8_0.gguf";

    /// Each test holds one or two copies of the model plus a full set of GPU
    /// buffers; Catch2 runs the C++ cases one after another, and so do these.
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn lock() -> std::sync::MutexGuard<'static, ()> {
        GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn load_lm(cfg: &Config6) -> OmniLm {
        OmniLm::load(LM_PATH, cfg.clone()).unwrap_or_else(|e| panic!("{}", e))
    }

    /// A sequence of the shape a diffusion step actually runs: a style and text
    /// prefix, then masked frames.
    fn sample_sequence(cfg: &Config6, frames: usize) -> Vec<OmniToken> {
        let mut seq = Vec::new();
        seq.push(OmniToken::text(cfg.lang_start));
        for i in 0..6 {
            seq.push(OmniToken::text(1000 + i));
        }
        seq.push(OmniToken::text(cfg.lang_end));
        seq.push(OmniToken::text(cfg.text_start));
        for i in 0..10 {
            seq.push(OmniToken::text(2000 + i));
        }
        seq.push(OmniToken::text(cfg.text_end));

        // A few decided frames, as voice cloning produces, then masked ones.
        for t in 0..4usize {
            let mut token = OmniToken::default();
            token.audio.resize(cfg.num_audio_codebook, 0);
            for c in 0..cfg.num_audio_codebook {
                token.audio[c] = ((c * 37 + t * 11) % 1024) as u32;
            }
            seq.push(token);
        }
        for _ in 0..frames {
            seq.push(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
        }
        seq
    }

    #[test]
    fn test_metal_omni_context_matches_the_cpu_forward_pass() {
        if !std::path::Path::new(LM_PATH).exists() {
            eprintln!("skip: {} not present", LM_PATH);
            return;
        }
        let _guard = lock();
        // Two copies of the model: `create` consumes the projections of the one it
        // uploads, so the reference has to be a second load.
        let cfg = Config6::omnivoice();
        let cpu = load_lm(&cfg);
        let mut gpu_weights = load_lm(&cfg);

        let ctx = MetalOmniContext::create(&mut gpu_weights, MetalOmniContext::MAX_TOKENS)
            .expect("metal context");
        assert_eq!(ctx.max_tokens(), MetalOmniContext::MAX_TOKENS);
        assert!(ctx.buffer_bytes() > 900_000_000);

        let seq = sample_sequence(&cfg, 40);
        let want = cpu.forward(&seq).expect("cpu forward");
        let got = ctx.forward(&seq).expect("gpu forward");
        assert_eq!(got.rows, want.rows);
        assert_eq!(got.cols, want.cols);
        assert_eq!(got.cols, cfg.audio_table_size());

        // The two paths sum in different orders -- BLAS chunks the weight while
        // the GPU accumulates 8x8 tiles -- so they agree to f32 epsilon and not
        // further. At logits of order 10 that is a few units in the last place.
        let mut max_rel = 0.0f64;
        for i in 0..want.data.len() {
            assert!(got.data[i].is_finite());
            let scale = 1.0f64.max((want.data[i] as f64).abs());
            max_rel = max_rel.max((want.data[i] as f64 - got.data[i] as f64).abs() / scale);
        }
        assert!(max_rel < 1e-3, "max_rel {}", max_rel);

        // What the sampler actually reads is the argmax per (position, codebook),
        // and those have to agree exactly or the two paths would diverge after the
        // first unmasking step.
        let vocab = cfg.audio_vocab_size;
        for r in 0..want.rows {
            for c in 0..cfg.num_audio_codebook {
                let mut best_cpu = 0usize;
                let mut best_gpu = 0usize;
                let mut v_cpu = f32::NEG_INFINITY;
                let mut v_gpu = f32::NEG_INFINITY;
                for v in 0..vocab {
                    if want.at(r, c * vocab + v) > v_cpu {
                        v_cpu = want.at(r, c * vocab + v);
                        best_cpu = v;
                    }
                    if got.at(r, c * vocab + v) > v_gpu {
                        v_gpu = got.at(r, c * vocab + v);
                        best_gpu = v;
                    }
                }
                assert_eq!(best_cpu, best_gpu, "position {} codebook {}", r, c);
            }
        }
    }

    #[test]
    fn test_metal_omni_context_is_deterministic() {
        if !std::path::Path::new(LM_PATH).exists() {
            eprintln!("skip: {} not present", LM_PATH);
            return;
        }
        let _guard = lock();
        let cfg = Config6::omnivoice();
        let mut lm = load_lm(&cfg);
        let ctx =
            MetalOmniContext::create(&mut lm, MetalOmniContext::MAX_TOKENS).expect("metal context");

        let seq = sample_sequence(&cfg, 20);
        let a = ctx.forward(&seq).expect("first pass");
        let b = ctx.forward(&seq).expect("second pass");
        for i in 0..a.data.len() {
            assert!(a.data[i] == b.data[i]);
        }
    }

    #[test]
    fn test_metal_omni_context_handles_lengths_that_are_not_tile_multiples() {
        if !std::path::Path::new(LM_PATH).exists() {
            eprintln!("skip: {} not present", LM_PATH);
            return;
        }
        let _guard = lock();
        // The GEMM tile is 64 wide, so every sequence but a multiple of it runs
        // with padding rows. Those rows are zeroed and must not leak into the real
        // ones -- attention is the only operator that crosses positions, and it is
        // dispatched over the true length rather than the padded one.
        let cfg = Config6::omnivoice();
        let cpu = load_lm(&cfg);
        let mut gpu_weights = load_lm(&cfg);
        let ctx = MetalOmniContext::create(&mut gpu_weights, MetalOmniContext::MAX_TOKENS)
            .expect("metal context");

        for frames in [1usize, 5, 44] {
            let seq: Vec<OmniToken> = (0..frames)
                .map(|_| OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id))
                .collect();
            let want = cpu.forward(&seq).expect("cpu forward");
            let got = ctx.forward(&seq).expect("gpu forward");
            assert_eq!(got.rows, frames);
            for i in 0..want.data.len() {
                let scale = 1.0f64.max((want.data[i] as f64).abs());
                assert!((want.data[i] as f64 - got.data[i] as f64).abs() / scale < 1e-3);
            }
        }
    }

    #[test]
    fn test_metal_omni_context_rejects_sequences_it_cannot_run() {
        if !std::path::Path::new(LM_PATH).exists() {
            eprintln!("skip: {} not present", LM_PATH);
            return;
        }
        let _guard = lock();
        let cfg = Config6::omnivoice();
        let mut lm = load_lm(&cfg);
        // Deliberately small, so the limit is reachable without a huge allocation.
        let ctx = MetalOmniContext::create(&mut lm, 128).expect("metal context");

        assert!(ctx.forward(&[]).is_err());
        assert!(ctx.forward(&sample_sequence(&cfg, 400)).is_err());

        // A bad code has to be caught on the host, before anything is dispatched.
        let mut bad = sample_sequence(&cfg, 4);
        bad.last_mut().unwrap().audio[0] = 99999;
        assert!(ctx.forward(&bad).is_err());
    }
}
