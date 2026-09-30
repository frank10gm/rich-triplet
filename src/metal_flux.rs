#![allow(dead_code)]
// =============================================================================
// MetalFluxContext -- the FLUX forward pass as one GPU command buffer
// =============================================================================
//
// The third full-graph engine in this project, after Gemma 3 and OmniVoice, and
// the one with the most work per dispatch: 57 blocks of a 12B transformer over
// four thousand tokens, encoded once and submitted once per denoising step.
//
// ## Weights stay quantized; activations do not
//
// A 12B model is 6.7 GB as Q4_K and 24 GB as bfloat, so the weights live on the
// GPU in their packed form and cannot be widened at rest. They are widened per
// use instead: before each GEMM, one dispatch dequantizes that weight into a
// shared bfloat scratch buffer, and the GEMM reads it from there.
//
// This sounds wasteful and is not. The largest weight is a single-stream
// block's fused `linear1` at 3072 -> 21504 -- 66 M elements to widen, against
// 575 GFLOP of GEMM to follow. The scratch buffer is 132 MB and is reused by
// every projection in the model.
//
// ## Shape constraints
//
// The GEMM kernel is a 64x64 register-blocked `simdgroup_matrix` tile with no
// bounds checks, so every projection width must be a multiple of 64 and every
// contracted dimension a multiple of 8. FLUX satisfies all of them naturally --
// 3072, 9216, 12288, 15360, 18432, 21504 -- and `create` checks rather than
// assumes, because a config that violates one produces silent corruption in
// the last partial tile rather than a fault.
//
// The token count is padded up to 64 with zero rows. Every operator except
// attention is row-independent, so the pad rows are harmless, and attention is
// dispatched over the true length.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;

use crate::autograd2::Mat;
use crate::flux::{
    flux_check_axes, flux_patchify, flux_schedule, flux_timestep_embedding, flux_unpatchify, FluxConfig,
    FluxModel, FluxSampleParams,
};
use crate::nn::InitRng;
use crate::qlinear::QLinear;

// =============================================================================
// MSL kernel source
// =============================================================================

/// Every FLUX kernel, compiled once into a single library. Verbatim from CPP
/// `src/shaders/flux.msl`.
const FLUX_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// =============================================================================
// Q4_K dequantization
// =============================================================================
//
// The existing quantized kernels in this project are GEMV: one output row per
// dispatch, which is exactly right for autoregressive decode and exactly wrong
// here. A diffusion transformer pushes four thousand tokens through every
// projection at once, so what it needs is a GEMM.
//
// Writing a Q4_K x dense GEMM is possible. Dequantizing the weight once per
// layer per step and running an ordinary bfloat GEMM is better, and it is not
// close. The largest FLUX weight is a single-stream block's fused `linear1` at
// 3072 -> 21504: 66 M parameters, 132 MB as bfloat, and 575 GFLOP of GEMM to
// follow. Touching 66 M elements to save writing a fused kernel is noise
// against that, and the block-decode arithmetic below is the same arithmetic
// the GEMV kernels already carry.
//
// A Q4_K super-block is 144 bytes covering 256 weights:
//
//   bytes[0..2]    f16 super-block scale
//   bytes[2..4]    f16 super-block minimum
//   bytes[4..16]   twelve packed 6-bit scale/min pairs, eight sub-blocks
//   bytes[16..144] 128 bytes of nibbles, four chunks of 32
//
// Within a chunk the low nibbles are elements 0..31 of its 64 and the high
// nibbles are elements 32..63 -- a split, not an interleave. Reading them as
// interleaved produces weights that are individually plausible and collectively
// scrambled, which trains no alarm anywhere downstream.

#define Q4K_BLOCK_BYTES 144u
#define Q4K_BLOCK_ELEMS 256u

inline float2 flux_scale_min(device const uchar* sc, uint j) {
    float sv, mv;
    if (j < 4u) {
        sv = float(sc[j] & 0x3Fu);
        mv = float(sc[j + 4u] & 0x3Fu);
    } else {
        sv = float((sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u));
        mv = float((sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u));
    }
    return float2(sv, mv);
}

// One thread per super-block. `out` is the bfloat weight in [N, K] order, the
// same order the blocks are stored in, so this is a straight streaming write.
kernel void flux_dequant_q4k(
    device const uchar*  blocks [[buffer(0)]],
    device       bfloat* out    [[buffer(1)]],
    constant     uint&   n_blocks [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n_blocks) return;

    device const uchar* bp = blocks + gid * Q4K_BLOCK_BYTES;
    ushort d_bits    = ushort(bp[0]) | (ushort(bp[1]) << 8u);
    ushort dmin_bits = ushort(bp[2]) | (ushort(bp[3]) << 8u);
    float d    = float(as_type<half>(d_bits));
    float dmin = float(as_type<half>(dmin_bits));

    device const uchar* sc = bp + 4u;
    device const uchar* qs = bp + 16u;
    device bfloat* dst = out + gid * Q4K_BLOCK_ELEMS;

    for (uint chunk = 0u; chunk < 4u; ++chunk) {
        float2 sm1 = flux_scale_min(sc, chunk * 2u);
        float2 sm2 = flux_scale_min(sc, chunk * 2u + 1u);
        float scale1 = d * sm1.x;  float min1 = dmin * sm1.y;
        float scale2 = d * sm2.x;  float min2 = dmin * sm2.y;

        device const uchar* qq = qs + chunk * 32u;
        uint base = chunk * 64u;
        for (uint l = 0u; l < 32u; ++l) {
            dst[base + l]       = bfloat(scale1 * float(qq[l] & 0x0Fu) - min1);
            dst[base + 32u + l] = bfloat(scale2 * float(qq[l] >> 4u)   - min2);
        }
    }
}

// =============================================================================
// GEMM
// =============================================================================
//
// C[M,N] = A[M,K] @ W[N,K]^T, the weight in bfloat.
//
// One threadgroup is four simdgroups covering a 64x64 tile of C, each owning a
// 32x32 quadrant as sixteen 8x8 accumulators. Every 8x8 operand loaded from
// memory feeds four multiply-accumulates rather than one, which is the whole
// difference against a scalar tiled kernel.
//
// M and N are always multiples of 64 and K a multiple of 8, because the caller
// pads. That is what lets the inner loop carry no bounds checks at all.

kernel void flux_gemm_bt(
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
        // so the weight is never stored twice.
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

// Add a per-column bias to every row of a [M, N] matrix.
kernel void flux_add_bias(
    device       float* C    [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    constant uint& N [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    C[gid.y * N + gid.x] += bias[gid.x];
}

// =============================================================================
// Attention
// =============================================================================
//
// The OmniVoice attention kernel keeps a full row of scores in threadgroup
// memory, which caps it at 2048 keys. FLUX runs 4352 at 1024x1024, so this one
// streams instead: it walks the keys in tiles, keeping only a running maximum,
// a running denominator and the output accumulator. That is the online-softmax
// recurrence -- on seeing a tile whose maximum exceeds the running one, the
// accumulator and the denominator are rescaled by `exp(old_max - new_max)`
// before the tile is folded in, which bounds every exponential without ever
// materialising the score row.
//
// ## Why a block of queries, and not one
//
// The obvious shape is one threadgroup per (head, query), and it is a trap.
// Each threadgroup then streams the whole of K and V for its head -- 4352 keys
// of 128 floats, twice, is 4.4 MB -- and there are 24 x 4352 threadgroups.
// That is 460 GB of traffic per attention call, 57 blocks per step, four steps
// per image. The kernel is not ALU-bound at that point and no amount of
// arithmetic tuning helps it.
//
// So a threadgroup owns a *block* of `FLUX_ATTN_BQ` queries and streams K and V
// once for all of them. The tile every query needs at a given moment is the
// same tile, so the traffic divides by the block size directly: 460 GB becomes
// 14 GB, and the accumulators stay in threadgroup memory where the rescale can
// reach them.
//
// 32 queries of 128 dimensions is 16 KB of accumulator and 4 KB of scores,
// inside the 32 KB a threadgroup may address.
//
// Neither Q nor V is staged in threadgroup memory, and that is measured rather
// than assumed. Staging Q as half removes a 32x redundancy in the score loop --
// each query row is otherwise re-read once per key in the tile -- and it made
// the whole image *slower*, 63.5 s to 68.5 s at 512x512. Threadgroup memory is
// the occupancy currency on this GPU: 20 KB per threadgroup keeps enough of
// them resident to hide the device-memory latency, 28 KB does not, and the
// reads being saved were hitting cache anyway.

#define FLUX_ATTN_BQ 32u
#define FLUX_ATTN_BK 32u
#define FLUX_ATTN_THREADS 256u
#define FLUX_ATTN_MAX_HEAD_DIM 128u
#define FLUX_ATTN_SIMDGROUPS 8u

kernel void flux_attention(
    device const float* q   [[buffer(0)]],
    device const float* k   [[buffer(1)]],
    device const float* v   [[buffer(2)]],
    device       float* out [[buffer(3)]],
    constant uint&  T        [[buffer(4)]],
    constant uint&  n_heads  [[buffer(5)]],
    constant uint&  head_dim [[buffer(6)]],
    constant float& scale    [[buffer(7)]],
    uint2 gid   [[threadgroup_position_in_grid]],
    uint2 tid_v [[thread_position_in_threadgroup]],
    uint  sgid  [[simdgroup_index_in_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint head = gid.x;
    const uint q0 = gid.y * FLUX_ATTN_BQ;
    // Every thread here shares `gid`, so the whole threadgroup leaves together
    // and no barrier below is reached by only part of it.
    if (q0 >= T) return;
    const uint nq = min(FLUX_ATTN_BQ, T - q0);

    const uint stride = n_heads * head_dim;
    device const float* qb = q + head * head_dim;
    device const float* kb = k + head * head_dim;
    device const float* vb = v + head * head_dim;

    threadgroup float acc[FLUX_ATTN_BQ * FLUX_ATTN_MAX_HEAD_DIM];
    threadgroup float sc[FLUX_ATTN_BQ * FLUX_ATTN_BK];
    threadgroup float m_state[FLUX_ATTN_BQ];   // running maximum
    threadgroup float l_state[FLUX_ATTN_BQ];   // running denominator
    threadgroup float alpha[FLUX_ATTN_BQ];     // this tile's rescale

    const uint dq = head_dim / 8u;
    const uint s_tiles = (FLUX_ATTN_BQ / 8u) * (FLUX_ATTN_BK / 8u);
    const uint o_tiles = (FLUX_ATTN_BQ / 8u) * dq;

    for (uint i = tid; i < FLUX_ATTN_BQ * head_dim; i += FLUX_ATTN_THREADS) {
        acc[i] = 0.0f;
    }
    if (tid < FLUX_ATTN_BQ) {
        m_state[tid] = -INFINITY;
        l_state[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint base = 0u; base < T; base += FLUX_ATTN_BK) {
        const uint nk = min(FLUX_ATTN_BK, T - base);

        // --- S = Q K^T, as sixteen 8x8 fragments over eight simdgroups -------
        for (uint t = sgid; t < s_tiles; t += FLUX_ATTN_SIMDGROUPS) {
            const uint ti = t / (FLUX_ATTN_BK / 8u);
            const uint tj = t % (FLUX_ATTN_BK / 8u);
            simdgroup_float8x8 sacc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 a, b;
            for (uint d0 = 0u; d0 < head_dim; d0 += 8u) {
                simdgroup_load(a, qb + (q0 + ti * 8u) * stride + d0, stride);
                // transpose=true reads K[key, d] as the [d, key] operand.
                simdgroup_load(b, kb + (base + tj * 8u) * stride + d0, stride, ulong2(0, 0),
                               true);
                simdgroup_multiply_accumulate(sacc, a, b, sacc);
            }
            simdgroup_store(sacc, sc + (ti * 8u) * FLUX_ATTN_BK + tj * 8u, FLUX_ATTN_BK);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- online softmax, one thread per query row ------------------------
        //
        // The scale is folded in here rather than in the matmul: the fragments
        // come back unscaled and this pass already touches every score.
        //
        // Rows past `nq` and columns past `nk` are zeroed rather than left
        // alone. The matmul below contracts over all 32 keys whatever the tail
        // holds, so a stale value in the masked region would leak into every
        // output element of the block.
        if (tid < FLUX_ATTN_BQ) {
            const uint row = tid * FLUX_ATTN_BK;
            if (tid >= nq) {
                for (uint j = 0u; j < FLUX_ATTN_BK; ++j) {
                    sc[row + j] = 0.0f;
                }
                alpha[tid] = 0.0f;
            } else {
                float tile_max = -INFINITY;
                for (uint j = 0u; j < nk; ++j) {
                    tile_max = max(tile_max, sc[row + j] * scale);
                }
                const float new_max = max(m_state[tid], tile_max);
                const float a = (m_state[tid] == -INFINITY) ? 0.0f : exp(m_state[tid] - new_max);

                float tile_sum = 0.0f;
                for (uint j = 0u; j < nk; ++j) {
                    const float e = exp(sc[row + j] * scale - new_max);
                    sc[row + j] = e;
                    tile_sum += e;
                }
                for (uint j = nk; j < FLUX_ATTN_BK; ++j) {
                    sc[row + j] = 0.0f;
                }
                m_state[tid] = new_max;
                l_state[tid] = l_state[tid] * a + tile_sum;
                alpha[tid] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // --- rescale the accumulator, then O += P V --------------------------
        //
        // The rescale has to happen before the matmul and separately from it.
        // `alpha` is per query *row*, and an 8x8 fragment held in registers
        // cannot be scaled by a row vector -- but it can be *initialised* from
        // the rescaled accumulator, which is the same thing and costs no
        // scratch tile. That is what keeps this kernel at 20 KB of threadgroup
        // memory: the obvious formulation, with a separate [32, head_dim]
        // product buffer, needs 36 KB and does not fit, and even at 28 KB the
        // occupancy loss outweighs everything the matrix units win.
        for (uint p = tid; p < FLUX_ATTN_BQ * head_dim; p += FLUX_ATTN_THREADS) {
            acc[p] *= alpha[p / head_dim];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint t = sgid; t < o_tiles; t += FLUX_ATTN_SIMDGROUPS) {
            const uint ti = t / dq;
            const uint td = t % dq;
            simdgroup_float8x8 oacc;
            simdgroup_load(oacc, acc + (ti * 8u) * head_dim + td * 8u, head_dim);
            simdgroup_float8x8 pm, vm;
            for (uint kk = 0u; kk < FLUX_ATTN_BK; kk += 8u) {
                simdgroup_load(pm, sc + (ti * 8u) * FLUX_ATTN_BK + kk, FLUX_ATTN_BK);
                simdgroup_load(vm, vb + (base + kk) * stride + td * 8u, stride);
                simdgroup_multiply_accumulate(oacc, pm, vm, oacc);
            }
            simdgroup_store(oacc, acc + (ti * 8u) * head_dim + td * 8u, head_dim);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint p = tid; p < nq * head_dim; p += FLUX_ATTN_THREADS) {
        const uint qi = p / head_dim;
        const uint d = p % head_dim;
        out[(q0 + qi) * stride + head * head_dim + d] = acc[p] / l_state[qi];
    }
}

// =============================================================================
// Normalisation
// =============================================================================

// LayerNorm with no affine parameters. Every norm in FLUX is one of these: the
// scale and the shift arrive from the modulation path instead.
kernel void flux_layer_norm(
    device const float* x   [[buffer(0)]],
    device       float* out [[buffer(1)]],
    constant uint&  width [[buffer(2)]],
    constant float& eps   [[buffer(3)]],
    uint  row [[threadgroup_position_in_grid]],
    uint  tid [[thread_position_in_threadgroup]],
    uint  nth [[threads_per_threadgroup]])
{
    threadgroup float reduce[256];
    device const float* src = x + row * width;
    device float* dst = out + row * width;

    float local = 0.0f;
    for (uint c = tid; c < width; c += nth) local += src[c];
    reduce[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float mean = reduce[0] / float(width);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float lv = 0.0f;
    for (uint c = tid; c < width; c += nth) {
        const float d = src[c] - mean;
        lv += d * d;
    }
    reduce[tid] = lv;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(reduce[0] / float(width) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < width; c += nth) dst[c] = (src[c] - mean) * inv;
}

// Per-head RMSNorm with a learned scale, applied to Q and K before rotation.
//
// `row_stride` and `col_offset` let this run directly on a slice of a fused
// QKV projection: Q sits at offset 0 and K at offset `hidden` inside a row that
// is three or four times as wide. Normalising before the split is what keeps
// the image and text streams on their own learned scales, which they must be --
// the two halves of a double block share an attention but not a parameter.
//
// One threadgroup per (row, head).
kernel void flux_head_rms_norm(
    device       float* x     [[buffer(0)]],
    device const float* gain  [[buffer(1)]],
    constant uint&  row_stride [[buffer(2)]],
    constant uint&  col_offset [[buffer(3)]],
    constant uint&  head_dim  [[buffer(4)]],
    constant float& eps       [[buffer(5)]],
    // MSL wants every position attribute in a kernel to have the same width,
    // so these are uint2 with an unused y rather than plain uints.
    uint2 gid   [[threadgroup_position_in_grid]],
    uint2 tid_v [[thread_position_in_threadgroup]],
    uint2 nth_v [[threads_per_threadgroup]])
{
    const uint tid = tid_v.x;
    const uint nth = nth_v.x;
    threadgroup float reduce[128];
    device float* head = x + gid.y * row_stride + col_offset + gid.x * head_dim;

    float local = 0.0f;
    for (uint c = tid; c < head_dim; c += nth) local += head[c] * head[c];
    reduce[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = nth / 2u; s > 0u; s >>= 1u) {
        if (tid < s) reduce[tid] += reduce[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float inv = rsqrt(reduce[0] / float(head_dim) + eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < head_dim; c += nth) head[c] = head[c] * inv * gain[c];
}

// =============================================================================
// Modulation
// =============================================================================

// `(1 + scale) * x + shift`, with the scale and shift broadcast from one row.
//
// `params` points at the modulation projection's output; `shift_off` and
// `scale_off` are element offsets into it, which is what lets one kernel serve
// the six-way, three-way and two-way splits without repacking any of them.
kernel void flux_modulate(
    device       float* x      [[buffer(0)]],
    device const float* params [[buffer(1)]],
    constant uint& width      [[buffer(2)]],
    constant uint& shift_off  [[buffer(3)]],
    constant uint& scale_off  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint c = gid.x;
    const uint i = gid.y * width + c;
    x[i] = (1.0f + params[scale_off + c]) * x[i] + params[shift_off + c];
}

// `dst += gate * src`, the gate broadcast from one row.
kernel void flux_gated_add(
    device       float* dst    [[buffer(0)]],
    device const float* src    [[buffer(1)]],
    device const float* params [[buffer(2)]],
    constant uint& width     [[buffer(3)]],
    constant uint& gate_off  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    const uint i = gid.y * width + gid.x;
    dst[i] += params[gate_off + gid.x] * src[i];
}

// =============================================================================
// RoPE
// =============================================================================
//
// Three axes over (t, h, w), each owning a contiguous slice of the head
// dimension and rotating adjacent pairs within it. The cosine and sine tables
// are precomputed on the CPU -- they depend only on the image size, so they are
// built once per generation rather than once per block.
//
// One thread per (row, head, pair).
kernel void flux_rope(
    device       float* x   [[buffer(0)]],
    device const float* cos_tab [[buffer(1)]],
    device const float* sin_tab [[buffer(2)]],
    constant uint& n_heads  [[buffer(3)]],
    constant uint& head_dim [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]])
{
    const uint pairs = head_dim / 2u;
    const uint row = gid.z;
    const uint head = gid.y;
    const uint p = gid.x;

    device float* h = x + row * (n_heads * head_dim) + head * head_dim;
    const float a = h[2u * p];
    const float b = h[2u * p + 1u];
    const float c = cos_tab[row * pairs + p];
    const float s = sin_tab[row * pairs + p];
    h[2u * p]      = a * c - b * s;
    h[2u * p + 1u] = a * s + b * c;
}

// =============================================================================
// Elementwise
// =============================================================================

kernel void flux_gelu_tanh(
    device float* x [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    const float v = x[i];
    // The cubic term grows fast: an activation of 27 -- ordinary in a FLUX MLP
    // -- puts the argument near 724, and Metal compiles `tanh` under fast-math
    // into a form that evaluates exp(2x). At 724 that is inf, and inf/inf is
    // NaN. One NaN in a residual stream poisons every later block, so it shows
    // up as a fully black image and nothing more specific.
    //
    // tanh is saturated to within a float's resolution well before 10, so
    // clamping the argument is exact rather than approximate.
    const float inner =
        clamp(0.7978845608028654f * (v + 0.044715f * v * v * v), -10.0f, 10.0f);
    x[i] = 0.5f * v * (1.0f + tanh(inner));
}

kernel void flux_silu(
    device float* x [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    x[i] = x[i] / (1.0f + exp(-x[i]));
}

kernel void flux_add(
    device       float* dst [[buffer(0)]],
    device const float* src [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    dst[i] += src[i];
}

kernel void flux_zero(
    device float* x [[buffer(0)]],
    uint i [[thread_position_in_grid]])
{
    x[i] = 0.0f;
}

// Copy a column range out of a [rows, src_width] matrix into a
// [rows, count] one. Splitting a fused QKV projection is three of these.
kernel void flux_slice_cols(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant uint& src_width [[buffer(2)]],
    constant uint& dst_width [[buffer(3)]],
    constant uint& from      [[buffer(4)]],
    constant uint& src_row0  [[buffer(5)]],
    constant uint& dst_row0  [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]])
{
    dst[(dst_row0 + gid.y) * dst_width + gid.x] =
        src[(src_row0 + gid.y) * src_width + from + gid.x];
}

// The inverse: write a [rows, count] block into a column range.
kernel void flux_paste_cols(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant uint& src_width [[buffer(2)]],
    constant uint& dst_width [[buffer(3)]],
    constant uint& at        [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    dst[gid.y * dst_width + at + gid.x] = src[gid.y * src_width + gid.x];
}

// Copy a row range, for splitting and rejoining the text and image streams.
kernel void flux_copy_rows(
    device const float* src [[buffer(0)]],
    device       float* dst [[buffer(1)]],
    constant uint& width    [[buffer(2)]],
    constant uint& src_from [[buffer(3)]],
    constant uint& dst_from [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    dst[(dst_from + gid.y) * width + gid.x] = src[(src_from + gid.y) * width + gid.x];
}
"#;

/// A GEMM tile is 64x64, so both the token count and every projection width
/// are rounded up to that.
const K_TILE: usize = 64;

/// The attention kernel's threadgroup accumulator.
const K_MAX_HEAD_DIM: usize = 128;

fn round_up(v: usize, to: usize) -> usize {
    (v + to - 1) / to * to
}

/// Truncating f32 -> bfloat, the same bits the CPP upload writes.
fn f32_to_bf16_bits(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

type Device = ProtocolObject<dyn MTLDevice>;
type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
type BufferRef = ProtocolObject<dyn MTLBuffer>;
type Pipeline = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
type Encoder = ProtocolObject<dyn MTLComputeCommandEncoder>;

fn size3(width: usize, height: usize, depth: usize) -> MTLSize {
    MTLSize { width, height, depth }
}

fn set_u32(enc: &Encoder, v: u32, index: usize) {
    unsafe { enc.setBytes_length_atIndex(NonNull::from(&v).cast::<c_void>(), std::mem::size_of::<u32>(), index) }
}

fn set_f32(enc: &Encoder, v: f32, index: usize) {
    unsafe { enc.setBytes_length_atIndex(NonNull::from(&v).cast::<c_void>(), std::mem::size_of::<f32>(), index) }
}

fn set_buf(enc: &Encoder, buf: Option<&BufferRef>, index: usize) {
    unsafe { enc.setBuffer_offset_atIndex(buf, 0, index) }
}

/// The whole of a shared buffer as host floats.
///
/// # Safety
/// No other slice over the same buffer may be alive, and the GPU must not be
/// running work that touches it.
unsafe fn host_f32<'a>(buf: &'a BufferRef) -> &'a mut [f32] {
    unsafe { std::slice::from_raw_parts_mut(buf.contents().as_ptr() as *mut f32, buf.length() / 4) }
}

// =============================================================================
// Context
// =============================================================================

/// One uploaded projection: either packed Q4_K blocks or an already-wide
/// bfloat weight, plus its bias.
#[derive(Default)]
struct Weight {
    data: Option<Buffer>, // Q4_K bytes, or bfloat when `packed` is false
    bias: Option<Buffer>,
    packed: bool,
    rows: usize,     // N, padded to K_TILE
    cols: usize,     // K
    n_blocks: usize, // Q4_K super-blocks, when packed
}

struct DoubleBlock {
    img_mod: Weight,
    img_qkv: Weight,
    img_proj: Weight,
    img_mlp_in: Weight,
    img_mlp_out: Weight,
    txt_mod: Weight,
    txt_qkv: Weight,
    txt_proj: Weight,
    txt_mlp_in: Weight,
    txt_mlp_out: Weight,
    img_q_norm: Option<Buffer>,
    img_k_norm: Option<Buffer>,
    txt_q_norm: Option<Buffer>,
    txt_k_norm: Option<Buffer>,
}

struct SingleBlock {
    modulation: Weight,
    linear1: Weight,
    linear2: Weight,
    q_norm: Option<Buffer>,
    k_norm: Option<Buffer>,
}

pub struct MetalFluxContext {
    cfg: FluxConfig,
    max_tokens: usize, // text + image, padded
    max_img_tokens: usize,
    max_txt_tokens: usize,

    device: Retained<Device>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

    dequant_q4k: Pipeline,
    gemm: Pipeline,
    add_bias: Pipeline,
    attention: Pipeline,
    layer_norm: Pipeline,
    head_rms_norm: Pipeline,
    modulate: Pipeline,
    gated_add: Pipeline,
    rope: Pipeline,
    gelu_tanh: Pipeline,
    silu: Pipeline,
    add: Pipeline,
    zero: Pipeline,
    slice_cols: Pipeline,
    paste_cols: Pipeline,
    copy_rows: Pipeline,

    img_in: Weight,
    txt_in: Weight,
    time_1: Weight,
    time_2: Weight,
    vector_1: Weight,
    vector_2: Weight,
    guidance_1: Weight,
    guidance_2: Weight,
    final_mod: Weight,
    final_linear: Weight,

    doubles: Vec<DoubleBlock>,
    singles: Vec<SingleBlock>,

    /// Shared bfloat scratch, sized for the widest weight in the model.
    wide: Buffer,
    wide_elems: usize,

    // Activations.
    img: Buffer,
    txt: Buffer,
    x: Buffer,
    tmp: Buffer, // [max_tokens, hidden]
    qb: Buffer,
    kb: Buffer,
    vb: Buffer,
    attn: Buffer,
    mlp: Buffer,    // [max_tokens, mlp_hidden]
    fused: Buffer,  // [max_tokens, 3*hidden + mlp_hidden]
    joined: Buffer, // [max_tokens, hidden + mlp_hidden]
    vec: Buffer,    // [K_TILE, hidden]
    vec_tmp: Buffer,
    /// A double block needs the image and text modulations alive at the same
    /// time: the gates are applied after a joint attention that sits between
    /// the two projections. Hence two buffers rather than one.
    mod_img: Buffer, // [K_TILE, 6*hidden]
    mod_txt: Buffer,
    /// Staging for the three vectors that enter through a projection: the
    /// timestep embedding (256), the guidance embedding (256) and the pooled
    /// CLIP vector (768). Each needs its own buffer rather than one reused
    /// three times -- the host writes them all before the command buffer is
    /// submitted, so a shared buffer would hand every reader the last value
    /// written rather than the one encoded alongside it. Sized for the widest.
    emb_time: Buffer,
    emb_guidance: Buffer,
    emb_pooled: Buffer,
    emb_width: usize,
    cos_tab: Buffer,
    sin_tab: Buffer,

    device_bytes: usize,
}

// =============================================================================
// Dispatch helpers
// =============================================================================

impl MetalFluxContext {
    /// Run `C[M,N] = A[M,K] @ W[N,K]^T`, widening the weight first if it is
    /// still packed.
    ///
    /// Only Q4_K weights take the dequantization step. Everything else went up
    /// as bfloat and is fed to the GEMM where it lies -- widening it again
    /// would reinterpret pairs of bfloat as single floats, which produces
    /// finite, plausibly-scaled garbage rather than anything that faults.
    fn gemm_into(&self, enc: &Encoder, w: &Weight, a: &BufferRef, c: &BufferRef, m: usize) {
        let mut operand: Option<&BufferRef> = w.data.as_deref();
        if w.packed {
            enc.setComputePipelineState(&self.dequant_q4k);
            set_buf(enc, w.data.as_deref(), 0);
            set_buf(enc, Some(&self.wide), 1);
            set_u32(enc, w.n_blocks as u32, 2);
            enc.dispatchThreads_threadsPerThreadgroup(size3(w.n_blocks, 1, 1), size3(64, 1, 1));
            operand = Some(&self.wide);
        }

        enc.setComputePipelineState(&self.gemm);
        set_buf(enc, Some(a), 0);
        set_buf(enc, operand, 1);
        set_buf(enc, Some(c), 2);
        set_u32(enc, w.cols as u32, 3);
        set_u32(enc, w.rows as u32, 4);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size3(w.rows / K_TILE, m / K_TILE, 1), size3(32, 4, 1));

        if let Some(bias) = w.bias.as_deref() {
            enc.setComputePipelineState(&self.add_bias);
            set_buf(enc, Some(c), 0);
            set_buf(enc, Some(bias), 1);
            set_u32(enc, w.rows as u32, 2);
            enc.dispatchThreads_threadsPerThreadgroup(size3(w.rows, m, 1), size3(64, 1, 1));
        }
    }

    fn run_layer_norm(&self, enc: &Encoder, src: &BufferRef, dst: &BufferRef, rows: usize, width: usize) {
        enc.setComputePipelineState(&self.layer_norm);
        set_buf(enc, Some(src), 0);
        set_buf(enc, Some(dst), 1);
        set_u32(enc, width as u32, 2);
        set_f32(enc, self.cfg.layer_norm_eps, 3);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size3(rows, 1, 1), size3(256, 1, 1));
    }

    fn run_modulate(
        &self,
        enc: &Encoder,
        target: &BufferRef,
        params: &BufferRef,
        rows: usize,
        shift_off: usize,
        scale_off: usize,
    ) {
        enc.setComputePipelineState(&self.modulate);
        set_buf(enc, Some(target), 0);
        set_buf(enc, Some(params), 1);
        set_u32(enc, self.cfg.hidden_size as u32, 2);
        set_u32(enc, shift_off as u32, 3);
        set_u32(enc, scale_off as u32, 4);
        enc.dispatchThreads_threadsPerThreadgroup(size3(self.cfg.hidden_size, rows, 1), size3(64, 1, 1));
    }

    fn run_gated_add(
        &self,
        enc: &Encoder,
        dst: &BufferRef,
        src: &BufferRef,
        params: &BufferRef,
        rows: usize,
        gate_off: usize,
    ) {
        enc.setComputePipelineState(&self.gated_add);
        set_buf(enc, Some(dst), 0);
        set_buf(enc, Some(src), 1);
        set_buf(enc, Some(params), 2);
        set_u32(enc, self.cfg.hidden_size as u32, 3);
        set_u32(enc, gate_off as u32, 4);
        enc.dispatchThreads_threadsPerThreadgroup(size3(self.cfg.hidden_size, rows, 1), size3(64, 1, 1));
    }

    /// Per-head RMSNorm on a column slice of a fused projection, so Q and K
    /// keep their own learned scales before the streams are concatenated.
    fn run_qk_norm(
        &self,
        enc: &Encoder,
        target: &BufferRef,
        gain: Option<&BufferRef>,
        row_stride: usize,
        col_offset: usize,
        rows: usize,
    ) {
        let head_dim = self.cfg.head_dim();
        enc.setComputePipelineState(&self.head_rms_norm);
        set_buf(enc, Some(target), 0);
        set_buf(enc, gain, 1);
        set_u32(enc, row_stride as u32, 2);
        set_u32(enc, col_offset as u32, 3);
        set_u32(enc, head_dim as u32, 4);
        set_f32(enc, self.cfg.qk_norm_eps, 5);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            size3(self.cfg.n_heads, rows, 1),
            size3(head_dim.min(128), 1, 1),
        );
    }

    /// Rotate an assembled [n_tok, hidden] buffer. The tables are built for the
    /// concatenated stream, so this runs after the two halves are in place.
    fn run_rope(&self, enc: &Encoder, target: &BufferRef, rows: usize) {
        let head_dim = self.cfg.head_dim();
        enc.setComputePipelineState(&self.rope);
        set_buf(enc, Some(target), 0);
        set_buf(enc, Some(&self.cos_tab), 1);
        set_buf(enc, Some(&self.sin_tab), 2);
        set_u32(enc, self.cfg.n_heads as u32, 3);
        set_u32(enc, head_dim as u32, 4);
        enc.dispatchThreads_threadsPerThreadgroup(
            size3(head_dim / 2, self.cfg.n_heads, rows),
            size3(head_dim / 2, 1, 1),
        );
    }

    fn run_attention(&self, enc: &Encoder, rows: usize) {
        let head_dim = self.cfg.head_dim();
        enc.setComputePipelineState(&self.attention);
        set_buf(enc, Some(&self.qb), 0);
        set_buf(enc, Some(&self.kb), 1);
        set_buf(enc, Some(&self.vb), 2);
        set_buf(enc, Some(&self.attn), 3);
        set_u32(enc, rows as u32, 4);
        set_u32(enc, self.cfg.n_heads as u32, 5);
        set_u32(enc, head_dim as u32, 6);
        set_f32(enc, 1.0f32 / (head_dim as f32).sqrt(), 7);
        // One threadgroup per (head, block of 32 queries). Blocking the
        // queries is what keeps K and V from being re-streamed once per query.
        const K_QUERY_BLOCK: usize = 32;
        let blocks = rows.div_ceil(K_QUERY_BLOCK);
        enc.dispatchThreadgroups_threadsPerThreadgroup(size3(self.cfg.n_heads, blocks, 1), size3(256, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn run_slice(
        &self,
        enc: &Encoder,
        src: &BufferRef,
        dst: &BufferRef,
        src_width: usize,
        dst_width: usize,
        from: usize,
        src_row0: usize,
        dst_row0: usize,
        rows: usize,
    ) {
        if rows == 0 {
            return;
        }
        enc.setComputePipelineState(&self.slice_cols);
        set_buf(enc, Some(src), 0);
        set_buf(enc, Some(dst), 1);
        set_u32(enc, src_width as u32, 2);
        set_u32(enc, dst_width as u32, 3);
        set_u32(enc, from as u32, 4);
        set_u32(enc, src_row0 as u32, 5);
        set_u32(enc, dst_row0 as u32, 6);
        enc.dispatchThreads_threadsPerThreadgroup(size3(dst_width, rows, 1), size3(64, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn run_paste(
        &self,
        enc: &Encoder,
        src: &BufferRef,
        dst: &BufferRef,
        src_width: usize,
        dst_width: usize,
        at: usize,
        rows: usize,
    ) {
        enc.setComputePipelineState(&self.paste_cols);
        set_buf(enc, Some(src), 0);
        set_buf(enc, Some(dst), 1);
        set_u32(enc, src_width as u32, 2);
        set_u32(enc, dst_width as u32, 3);
        set_u32(enc, at as u32, 4);
        enc.dispatchThreads_threadsPerThreadgroup(size3(src_width, rows, 1), size3(64, 1, 1));
    }

    #[allow(clippy::too_many_arguments)]
    fn run_copy_rows(
        &self,
        enc: &Encoder,
        src: &BufferRef,
        dst: &BufferRef,
        width: usize,
        src_from: usize,
        dst_from: usize,
        rows: usize,
    ) {
        if rows == 0 {
            return;
        }
        enc.setComputePipelineState(&self.copy_rows);
        set_buf(enc, Some(src), 0);
        set_buf(enc, Some(dst), 1);
        set_u32(enc, width as u32, 2);
        set_u32(enc, src_from as u32, 3);
        set_u32(enc, dst_from as u32, 4);
        enc.dispatchThreads_threadsPerThreadgroup(size3(width, rows, 1), size3(64, 1, 1));
    }

    fn run_elem(&self, enc: &Encoder, pipe: &ProtocolObject<dyn MTLComputePipelineState>, buf: &BufferRef, n: usize) {
        enc.setComputePipelineState(pipe);
        set_buf(enc, Some(buf), 0);
        enc.dispatchThreads_threadsPerThreadgroup(size3(n, 1, 1), size3(256, 1, 1));
    }

    fn run_zero(&self, enc: &Encoder, buf: &BufferRef, n: usize) {
        self.run_elem(enc, &self.zero, buf, n);
    }
}

// =============================================================================
// Construction
// =============================================================================

fn make_pipeline(device: &Device, lib: &ProtocolObject<dyn MTLLibrary>, name: &str) -> Result<Pipeline, String> {
    let func = lib
        .newFunctionWithName(&NSString::from_str(name))
        .ok_or_else(|| format!("metal flux: shader has no kernel '{}'", name))?;
    device
        .newComputePipelineStateWithFunction_error(&func)
        .map_err(|e| format!("metal flux: cannot build pipeline '{}': {}", name, e.localizedDescription()))
}

fn buffer_with_bytes(device: &Device, ptr: *const c_void, len: usize) -> Result<Buffer, String> {
    let ptr = NonNull::new(ptr as *mut c_void).ok_or("metal flux: null upload source")?;
    unsafe { device.newBufferWithBytes_length_options(ptr, len, MTLResourceOptions::StorageModeShared) }
        .ok_or_else(|| format!("metal flux: cannot allocate a {} byte buffer", len))
}

/// Upload one projection, consuming it.
///
/// Q4_K weights go up packed and are widened per use. Everything else is
/// converted to bfloat once here, which is what the small projections want --
/// widening 256x3072 on every step to save 1.5 MB would be a poor trade.
fn upload(
    device: &Device,
    src: &mut QLinear,
    device_bytes: &mut usize,
    wide_elems: &mut usize,
) -> Result<Weight, String> {
    let mut w = Weight { rows: src.out_features, cols: src.in_features, ..Default::default() };
    if w.rows == 0 || w.cols == 0 {
        return Ok(w); // an absent optional layer, such as schnell's guidance path
    }
    if w.rows % K_TILE != 0 {
        return Err(format!("metal flux: projection width {} is not a multiple of {}", w.rows, K_TILE));
    }
    if w.cols % 8 != 0 {
        return Err(format!("metal flux: contracted dimension {} is not a multiple of 8", w.cols));
    }
    *wide_elems = (*wide_elems).max(w.rows * w.cols);

    if let Some(q) = &src.q4k {
        let blocks = &q.blocks;
        w.packed = true;
        w.n_blocks = blocks.len() / 144;
        w.data = Some(buffer_with_bytes(device, blocks.as_ptr().cast(), blocks.len())?);
        *device_bytes += blocks.len();
    } else {
        let converted: Vec<u16>;
        let bits: &[u16] = if let Some(b) = &src.bf16 {
            b.data.as_slice()
        } else {
            converted = (0..w.rows * w.cols).map(|i| f32_to_bf16_bits(src.f32.data[i])).collect();
            &converted
        };
        w.data = Some(buffer_with_bytes(device, bits.as_ptr().cast(), std::mem::size_of_val(bits))?);
        *device_bytes += std::mem::size_of_val(bits);
    }

    if !src.bias.is_empty() {
        w.bias = Some(buffer_with_bytes(device, src.bias.as_ptr().cast(), std::mem::size_of_val(src.bias.as_slice()))?);
        *device_bytes += std::mem::size_of_val(src.bias.as_slice());
    }

    // Freed as it goes up, so peak memory never holds both copies.
    src.free_weight();
    src.bias = Vec::new();
    Ok(w)
}

fn upload_vec(device: &Device, v: &[f32], device_bytes: &mut usize) -> Result<Option<Buffer>, String> {
    if v.is_empty() {
        return Ok(None);
    }
    *device_bytes += std::mem::size_of_val(v);
    Ok(Some(buffer_with_bytes(device, v.as_ptr().cast(), std::mem::size_of_val(v))?))
}

/// A zeroed shared buffer. At least 16 bytes, as the other Metal modules do,
/// so a degenerate zero-size request still yields a buffer; the byte count
/// reported is the one asked for.
fn alloc(device: &Device, elems: usize, elem_size: usize, device_bytes: &mut usize) -> Result<Buffer, String> {
    *device_bytes += elems * elem_size;
    device
        .newBufferWithLength_options((elems * elem_size).max(16), MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| format!("metal flux: cannot allocate a {} byte buffer", elems * elem_size))
}

impl MetalFluxContext {
    /// Upload `model`'s weights and allocate scratch for a latent of at most
    /// `max_lat_h x max_lat_w` with at most `max_text_tokens` of prompt.
    ///
    /// **Consumes the model's weights.** Each is freed as it is uploaded, so
    /// peak memory never holds both copies -- and `model` cannot run a forward
    /// pass of its own afterwards. Its config is still read.
    pub fn create(
        model: &mut FluxModel,
        max_lat_h: usize,
        max_lat_w: usize,
        max_text_tokens: usize,
    ) -> Result<MetalFluxContext, String> {
        let cfg = model.cfg.clone();
        if cfg.head_dim() > K_MAX_HEAD_DIM {
            return Err(format!(
                "metal flux: head_dim {} exceeds the attention kernel's {}",
                cfg.head_dim(),
                K_MAX_HEAD_DIM
            ));
        }
        // The attention kernel tiles both of its matmuls into 8x8 simdgroup
        // fragments along the head dimension.
        if cfg.head_dim() % 8 != 0 {
            return Err(format!("metal flux: head_dim {} is not a multiple of 8", cfg.head_dim()));
        }
        if cfg.hidden_size % K_TILE != 0 {
            return Err(format!("metal flux: hidden_size must be a multiple of {}", K_TILE));
        }
        if max_lat_h % cfg.patch_size != 0 || max_lat_w % cfg.patch_size != 0 {
            return Err("metal flux: latent bounds must be multiples of the patch size".to_string());
        }
        flux_check_axes(&cfg)?;

        let device = MTLCreateSystemDefaultDevice().ok_or("metal flux: no Metal device")?;
        let queue = device.newCommandQueue().ok_or("metal flux: cannot create a command queue")?;

        let lib = device
            .newLibraryWithSource_options_error(&NSString::from_str(FLUX_MSL), None)
            .map_err(|e| format!("metal flux: shader compilation failed: {}", e.localizedDescription()))?;

        let dequant_q4k = make_pipeline(&device, &lib, "flux_dequant_q4k")?;
        let gemm = make_pipeline(&device, &lib, "flux_gemm_bt")?;
        let add_bias = make_pipeline(&device, &lib, "flux_add_bias")?;
        let attention = make_pipeline(&device, &lib, "flux_attention")?;
        let layer_norm = make_pipeline(&device, &lib, "flux_layer_norm")?;
        let head_rms_norm = make_pipeline(&device, &lib, "flux_head_rms_norm")?;
        let modulate = make_pipeline(&device, &lib, "flux_modulate")?;
        let gated_add = make_pipeline(&device, &lib, "flux_gated_add")?;
        let rope = make_pipeline(&device, &lib, "flux_rope")?;
        let gelu_tanh = make_pipeline(&device, &lib, "flux_gelu_tanh")?;
        let silu = make_pipeline(&device, &lib, "flux_silu")?;
        let add = make_pipeline(&device, &lib, "flux_add")?;
        let zero = make_pipeline(&device, &lib, "flux_zero")?;
        let slice_cols = make_pipeline(&device, &lib, "flux_slice_cols")?;
        let paste_cols = make_pipeline(&device, &lib, "flux_paste_cols")?;
        let copy_rows = make_pipeline(&device, &lib, "flux_copy_rows")?;

        // --- Weights ---------------------------------------------------------
        let mut bytes = 0usize;
        let mut wide_elems = 0usize;

        let img_in = upload(&device, &mut model.img_in, &mut bytes, &mut wide_elems)?;
        let txt_in = upload(&device, &mut model.txt_in, &mut bytes, &mut wide_elems)?;
        let time_1 = upload(&device, &mut model.time_in_1, &mut bytes, &mut wide_elems)?;
        let time_2 = upload(&device, &mut model.time_in_2, &mut bytes, &mut wide_elems)?;
        let vector_1 = upload(&device, &mut model.vector_in_1, &mut bytes, &mut wide_elems)?;
        let vector_2 = upload(&device, &mut model.vector_in_2, &mut bytes, &mut wide_elems)?;
        let guidance_1 = upload(&device, &mut model.guidance_in_1, &mut bytes, &mut wide_elems)?;
        let guidance_2 = upload(&device, &mut model.guidance_in_2, &mut bytes, &mut wide_elems)?;
        let final_mod = upload(&device, &mut model.final_mod, &mut bytes, &mut wide_elems)?;
        let final_linear = upload(&device, &mut model.final_linear, &mut bytes, &mut wide_elems)?;

        let mut doubles = Vec::with_capacity(model.double_blocks.len());
        for b in model.double_blocks.iter_mut() {
            let img_mod = upload(&device, &mut b.img_mod, &mut bytes, &mut wide_elems)?;
            let img_qkv = upload(&device, &mut b.img_qkv, &mut bytes, &mut wide_elems)?;
            let img_proj = upload(&device, &mut b.img_proj, &mut bytes, &mut wide_elems)?;
            let img_mlp_in = upload(&device, &mut b.img_mlp_in, &mut bytes, &mut wide_elems)?;
            let img_mlp_out = upload(&device, &mut b.img_mlp_out, &mut bytes, &mut wide_elems)?;
            let txt_mod = upload(&device, &mut b.txt_mod, &mut bytes, &mut wide_elems)?;
            let txt_qkv = upload(&device, &mut b.txt_qkv, &mut bytes, &mut wide_elems)?;
            let txt_proj = upload(&device, &mut b.txt_proj, &mut bytes, &mut wide_elems)?;
            let txt_mlp_in = upload(&device, &mut b.txt_mlp_in, &mut bytes, &mut wide_elems)?;
            let txt_mlp_out = upload(&device, &mut b.txt_mlp_out, &mut bytes, &mut wide_elems)?;
            let img_q_norm = upload_vec(&device, &b.img_norm.query_scale, &mut bytes)?;
            let img_k_norm = upload_vec(&device, &b.img_norm.key_scale, &mut bytes)?;
            let txt_q_norm = upload_vec(&device, &b.txt_norm.query_scale, &mut bytes)?;
            let txt_k_norm = upload_vec(&device, &b.txt_norm.key_scale, &mut bytes)?;
            doubles.push(DoubleBlock {
                img_mod,
                img_qkv,
                img_proj,
                img_mlp_in,
                img_mlp_out,
                txt_mod,
                txt_qkv,
                txt_proj,
                txt_mlp_in,
                txt_mlp_out,
                img_q_norm,
                img_k_norm,
                txt_q_norm,
                txt_k_norm,
            });
        }

        let mut singles = Vec::with_capacity(model.single_blocks.len());
        for b in model.single_blocks.iter_mut() {
            let modulation = upload(&device, &mut b.modulation, &mut bytes, &mut wide_elems)?;
            let linear1 = upload(&device, &mut b.linear1, &mut bytes, &mut wide_elems)?;
            let linear2 = upload(&device, &mut b.linear2, &mut bytes, &mut wide_elems)?;
            let q_norm = upload_vec(&device, &b.norm.query_scale, &mut bytes)?;
            let k_norm = upload_vec(&device, &b.norm.key_scale, &mut bytes)?;
            singles.push(SingleBlock { modulation, linear1, linear2, q_norm, k_norm });
        }

        // --- Scratch ---------------------------------------------------------
        let hidden = cfg.hidden_size;
        let mlp_hidden = cfg.mlp_hidden();
        let img_tokens = (max_lat_h / cfg.patch_size) * (max_lat_w / cfg.patch_size);
        let max_tokens = round_up(img_tokens + max_text_tokens, K_TILE);

        let wide = alloc(&device, wide_elems, std::mem::size_of::<u16>(), &mut bytes)?;

        let rows = max_tokens;
        let f = std::mem::size_of::<f32>();
        let img = alloc(&device, rows * hidden, f, &mut bytes)?;
        let txt = alloc(&device, rows * hidden, f, &mut bytes)?;
        let x = alloc(&device, rows * hidden, f, &mut bytes)?;
        let tmp = alloc(&device, rows * hidden, f, &mut bytes)?;
        let qb = alloc(&device, rows * hidden, f, &mut bytes)?;
        let kb = alloc(&device, rows * hidden, f, &mut bytes)?;
        let vb = alloc(&device, rows * hidden, f, &mut bytes)?;
        let attn = alloc(&device, rows * hidden, f, &mut bytes)?;
        let mlp = alloc(&device, rows * mlp_hidden, f, &mut bytes)?;
        let fused = alloc(&device, rows * (3 * hidden + mlp_hidden), f, &mut bytes)?;
        let joined = alloc(&device, rows * (hidden + mlp_hidden), f, &mut bytes)?;
        let vec = alloc(&device, K_TILE * hidden, f, &mut bytes)?;
        let vec_tmp = alloc(&device, K_TILE * hidden, f, &mut bytes)?;
        let mod_img = alloc(&device, K_TILE * 6 * hidden, f, &mut bytes)?;
        let mod_txt = alloc(&device, K_TILE * 6 * hidden, f, &mut bytes)?;
        let emb_width = 256usize.max(cfg.pooled_dim);
        let emb_time = alloc(&device, K_TILE * emb_width, f, &mut bytes)?;
        let emb_guidance = alloc(&device, K_TILE * emb_width, f, &mut bytes)?;
        let emb_pooled = alloc(&device, K_TILE * emb_width, f, &mut bytes)?;
        let cos_tab = alloc(&device, rows * (cfg.head_dim() / 2), f, &mut bytes)?;
        let sin_tab = alloc(&device, rows * (cfg.head_dim() / 2), f, &mut bytes)?;

        Ok(MetalFluxContext {
            cfg,
            max_tokens,
            max_img_tokens: img_tokens,
            max_txt_tokens: max_text_tokens,
            device,
            queue,
            dequant_q4k,
            gemm,
            add_bias,
            attention,
            layer_norm,
            head_rms_norm,
            modulate,
            gated_add,
            rope,
            gelu_tanh,
            silu,
            add,
            zero,
            slice_cols,
            paste_cols,
            copy_rows,
            img_in,
            txt_in,
            time_1,
            time_2,
            vector_1,
            vector_2,
            guidance_1,
            guidance_2,
            final_mod,
            final_linear,
            doubles,
            singles,
            wide,
            wide_elems,
            img,
            txt,
            x,
            tmp,
            qb,
            kb,
            vb,
            attn,
            mlp,
            fused,
            joined,
            vec,
            vec_tmp,
            mod_img,
            mod_txt,
            emb_time,
            emb_guidance,
            emb_pooled,
            emb_width,
            cos_tab,
            sin_tab,
            device_bytes: bytes,
        })
    }

    /// Bytes held in GPU buffers.
    pub fn device_bytes(&self) -> usize {
        self.device_bytes
    }

    /// The largest latent the allocated scratch can take.
    pub fn max_latent_tokens(&self) -> usize {
        self.max_img_tokens
    }
}

// =============================================================================
// Forward
// =============================================================================

impl MetalFluxContext {
    /// One velocity prediction -- the same contract as `FluxModel::forward`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &Mat,
        lat_h: usize,
        lat_w: usize,
        context: &Mat,
        pooled: &[f32],
        timestep: f32,
        guidance: f32,
    ) -> Result<Mat, String> {
        let s = self;
        let cfg = &s.cfg;
        let hidden = cfg.hidden_size;
        let mlp_hidden = cfg.mlp_hidden();
        let head_dim = cfg.head_dim();
        let patch = cfg.patch_size;

        if latent.cols != cfg.in_channels || latent.rows != lat_h * lat_w {
            return Err("metal flux: latent shape mismatch".to_string());
        }
        if lat_h % patch != 0 || lat_w % patch != 0 {
            return Err("metal flux: latent dimensions must be multiples of the patch size".to_string());
        }
        if context.cols != cfg.context_dim {
            return Err("metal flux: context width mismatch".to_string());
        }
        if pooled.len() != cfg.pooled_dim {
            return Err("metal flux: pooled width mismatch".to_string());
        }

        let n_img = (lat_h / patch) * (lat_w / patch);
        let n_txt = context.rows;
        let n_tok = n_img + n_txt;
        if n_img > s.max_img_tokens || n_txt > s.max_txt_tokens {
            return Err("metal flux: sequence exceeds the allocated scratch".to_string());
        }
        let rows = round_up(n_tok, K_TILE);
        let img_rows = round_up(n_img, K_TILE);
        let txt_rows = round_up(n_txt, K_TILE);

        // --- Host-side inputs ------------------------------------------------
        // The RoPE tables depend only on the image size, and the id layout is
        // simple enough that building them here costs less than a kernel would.
        {
            let cosp = unsafe { host_f32(&s.cos_tab) };
            let sinp = unsafe { host_f32(&s.sin_tab) };
            let pairs = head_dim / 2;
            let gw = lat_w / patch;
            for t in 0..rows {
                // Text tokens sit at the origin on every axis, so their rotation is
                // the identity. Pad rows follow them and never reach attention.
                let mut pos_axis = [0.0f32; 3];
                if t >= n_txt && t < n_tok {
                    let i = t - n_txt;
                    pos_axis[1] = (i / gw) as f32;
                    pos_axis[2] = (i % gw) as f32;
                }
                let mut p = 0usize;
                for a in 0..cfg.axes_dim.len() {
                    let dim = cfg.axes_dim[a];
                    for i in 0..dim / 2 {
                        let exponent = (2 * i) as f32 / dim as f32;
                        let angle = pos_axis[a] / cfg.rope_theta.powf(exponent);
                        cosp[t * pairs + p] = angle.cos();
                        sinp[t * pairs + p] = angle.sin();
                        p += 1;
                    }
                }
            }
        }

        // The patchified latent and the T5 sequence go into `tmp` and `attn` as
        // staging, since both are wide enough to hold them and are dead here.
        let patches = flux_patchify(latent, lat_h, lat_w, patch);
        {
            let dst = unsafe { host_f32(&s.tmp) };
            dst[..img_rows * cfg.patch_dim()].fill(0.0);
            dst[..patches.data.len()].copy_from_slice(&patches.data);
        }
        {
            // The T5 sequence is 4096 wide, which is wider than the hidden size --
            // `mlp` is the only activation buffer that can hold it.
            let dst = unsafe { host_f32(&s.mlp) };
            dst[..txt_rows * cfg.context_dim].fill(0.0);
            dst[..context.data.len()].copy_from_slice(&context.data);
        }
        {
            let t_emb = flux_timestep_embedding(timestep, 256, 10000.0, 1000.0);
            let dst = unsafe { host_f32(&s.emb_time) };
            dst[..K_TILE * s.emb_width].fill(0.0);
            dst[..t_emb.len()].copy_from_slice(&t_emb);
        }
        {
            let dst = unsafe { host_f32(&s.emb_pooled) };
            dst[..K_TILE * s.emb_width].fill(0.0);
            dst[..pooled.len()].copy_from_slice(pooled);
        }
        if cfg.guidance_embed {
            let g_emb = flux_timestep_embedding(guidance, 256, 10000.0, 1000.0);
            let dst = unsafe { host_f32(&s.emb_guidance) };
            dst[..K_TILE * s.emb_width].fill(0.0);
            dst[..g_emb.len()].copy_from_slice(&g_emb);
        }

        let cmd = s.queue.commandBuffer().ok_or("metal flux: cannot create a command buffer")?;
        let enc = cmd.computeCommandEncoder().ok_or("metal flux: cannot create a compute encoder")?;
        let enc: &Encoder = &enc;

        // Pad rows are never read downstream, but zeroing them keeps a stray NaN
        // in fresh memory from turning into a NaN the debugger has to chase.
        s.run_zero(enc, &s.img, rows * hidden);
        s.run_zero(enc, &s.txt, rows * hidden);
        s.run_zero(enc, &s.x, rows * hidden);

        // --- Conditioning vector ---------------------------------------------
        s.gemm_into(enc, &s.time_1, &s.emb_time, &s.vec_tmp, K_TILE);
        s.run_elem(enc, &s.silu, &s.vec_tmp, K_TILE * hidden);
        s.gemm_into(enc, &s.time_2, &s.vec_tmp, &s.vec, K_TILE);

        if cfg.guidance_embed {
            s.gemm_into(enc, &s.guidance_1, &s.emb_guidance, &s.vec_tmp, K_TILE);
            s.run_elem(enc, &s.silu, &s.vec_tmp, K_TILE * hidden);
            s.gemm_into(enc, &s.guidance_2, &s.vec_tmp, &s.mod_img, K_TILE);
            enc.setComputePipelineState(&s.add);
            set_buf(enc, Some(&s.vec), 0);
            set_buf(enc, Some(&s.mod_img), 1);
            enc.dispatchThreads_threadsPerThreadgroup(size3(hidden, 1, 1), size3(64, 1, 1));
        }

        s.gemm_into(enc, &s.vector_1, &s.emb_pooled, &s.vec_tmp, K_TILE);
        s.run_elem(enc, &s.silu, &s.vec_tmp, K_TILE * hidden);
        s.gemm_into(enc, &s.vector_2, &s.vec_tmp, &s.mod_txt, K_TILE);
        enc.setComputePipelineState(&s.add);
        set_buf(enc, Some(&s.vec), 0);
        set_buf(enc, Some(&s.mod_txt), 1);
        enc.dispatchThreads_threadsPerThreadgroup(size3(hidden, 1, 1), size3(64, 1, 1));

        // Every modulation projection reads `silu(vec)`, not `vec`.
        s.run_copy_rows(enc, &s.vec, &s.vec_tmp, hidden, 0, 0, K_TILE);
        s.run_elem(enc, &s.silu, &s.vec_tmp, K_TILE * hidden);

        // --- Streams ---------------------------------------------------------
        s.gemm_into(enc, &s.img_in, &s.tmp, &s.img, img_rows);
        s.gemm_into(enc, &s.txt_in, &s.mlp, &s.txt, txt_rows);

        let qkv_width = 3 * hidden;

        // --- Double-stream blocks --------------------------------------------
        for b in &s.doubles {
            s.gemm_into(enc, &b.img_mod, &s.vec_tmp, &s.mod_img, K_TILE);
            s.gemm_into(enc, &b.txt_mod, &s.vec_tmp, &s.mod_txt, K_TILE);

            // Image: norm, modulate, project, per-head norm on Q and K in place.
            s.run_layer_norm(enc, &s.img, &s.tmp, img_rows, hidden);
            s.run_modulate(enc, &s.tmp, &s.mod_img, img_rows, 0, hidden);
            s.gemm_into(enc, &b.img_qkv, &s.tmp, &s.fused, img_rows);
            s.run_qk_norm(enc, &s.fused, b.img_q_norm.as_deref(), qkv_width, 0, img_rows);
            s.run_qk_norm(enc, &s.fused, b.img_k_norm.as_deref(), qkv_width, hidden, img_rows);

            // Text: the same, into a separate staging buffer.
            s.run_layer_norm(enc, &s.txt, &s.tmp, txt_rows, hidden);
            s.run_modulate(enc, &s.tmp, &s.mod_txt, txt_rows, 0, hidden);
            s.gemm_into(enc, &b.txt_qkv, &s.tmp, &s.joined, txt_rows);
            s.run_qk_norm(enc, &s.joined, b.txt_q_norm.as_deref(), qkv_width, 0, txt_rows);
            s.run_qk_norm(enc, &s.joined, b.txt_k_norm.as_deref(), qkv_width, hidden, txt_rows);

            // Assemble [txt | img] -- the order the position ids were built in.
            s.run_slice(enc, &s.joined, &s.qb, qkv_width, hidden, 0, 0, 0, n_txt);
            s.run_slice(enc, &s.joined, &s.kb, qkv_width, hidden, hidden, 0, 0, n_txt);
            s.run_slice(enc, &s.joined, &s.vb, qkv_width, hidden, 2 * hidden, 0, 0, n_txt);
            s.run_slice(enc, &s.fused, &s.qb, qkv_width, hidden, 0, 0, n_txt, n_img);
            s.run_slice(enc, &s.fused, &s.kb, qkv_width, hidden, hidden, 0, n_txt, n_img);
            s.run_slice(enc, &s.fused, &s.vb, qkv_width, hidden, 2 * hidden, 0, n_txt, n_img);

            s.run_rope(enc, &s.qb, n_tok);
            s.run_rope(enc, &s.kb, n_tok);
            s.run_attention(enc, n_tok);

            // Image half of the attention, projected and gated back in.
            s.run_copy_rows(enc, &s.attn, &s.tmp, hidden, n_txt, 0, n_img);
            s.gemm_into(enc, &b.img_proj, &s.tmp, &s.fused, img_rows);
            s.run_gated_add(enc, &s.img, &s.fused, &s.mod_img, n_img, 2 * hidden);

            s.run_layer_norm(enc, &s.img, &s.tmp, img_rows, hidden);
            s.run_modulate(enc, &s.tmp, &s.mod_img, img_rows, 3 * hidden, 4 * hidden);
            s.gemm_into(enc, &b.img_mlp_in, &s.tmp, &s.mlp, img_rows);
            s.run_elem(enc, &s.gelu_tanh, &s.mlp, img_rows * mlp_hidden);
            s.gemm_into(enc, &b.img_mlp_out, &s.mlp, &s.fused, img_rows);
            s.run_gated_add(enc, &s.img, &s.fused, &s.mod_img, n_img, 5 * hidden);

            // Text half, the same shape of work against its own parameters.
            s.run_copy_rows(enc, &s.attn, &s.tmp, hidden, 0, 0, n_txt);
            s.gemm_into(enc, &b.txt_proj, &s.tmp, &s.joined, txt_rows);
            s.run_gated_add(enc, &s.txt, &s.joined, &s.mod_txt, n_txt, 2 * hidden);

            s.run_layer_norm(enc, &s.txt, &s.tmp, txt_rows, hidden);
            s.run_modulate(enc, &s.tmp, &s.mod_txt, txt_rows, 3 * hidden, 4 * hidden);
            s.gemm_into(enc, &b.txt_mlp_in, &s.tmp, &s.mlp, txt_rows);
            s.run_elem(enc, &s.gelu_tanh, &s.mlp, txt_rows * mlp_hidden);
            s.gemm_into(enc, &b.txt_mlp_out, &s.mlp, &s.joined, txt_rows);
            s.run_gated_add(enc, &s.txt, &s.joined, &s.mod_txt, n_txt, 5 * hidden);
        }

        // --- Single-stream blocks --------------------------------------------
        s.run_copy_rows(enc, &s.txt, &s.x, hidden, 0, 0, n_txt);
        s.run_copy_rows(enc, &s.img, &s.x, hidden, 0, n_txt, n_img);

        let fused_width = 3 * hidden + mlp_hidden;
        for b in &s.singles {
            s.gemm_into(enc, &b.modulation, &s.vec_tmp, &s.mod_img, K_TILE);

            s.run_layer_norm(enc, &s.x, &s.tmp, rows, hidden);
            s.run_modulate(enc, &s.tmp, &s.mod_img, rows, 0, hidden);

            // One projection produces QKV and the MLP's up-projection together.
            s.gemm_into(enc, &b.linear1, &s.tmp, &s.fused, rows);
            s.run_qk_norm(enc, &s.fused, b.q_norm.as_deref(), fused_width, 0, rows);
            s.run_qk_norm(enc, &s.fused, b.k_norm.as_deref(), fused_width, hidden, rows);
            s.run_slice(enc, &s.fused, &s.qb, fused_width, hidden, 0, 0, 0, n_tok);
            s.run_slice(enc, &s.fused, &s.kb, fused_width, hidden, hidden, 0, 0, n_tok);
            s.run_slice(enc, &s.fused, &s.vb, fused_width, hidden, 2 * hidden, 0, 0, n_tok);
            s.run_slice(enc, &s.fused, &s.mlp, fused_width, mlp_hidden, 3 * hidden, 0, 0, rows);

            s.run_rope(enc, &s.qb, n_tok);
            s.run_rope(enc, &s.kb, n_tok);
            s.run_attention(enc, n_tok);
            s.run_elem(enc, &s.gelu_tanh, &s.mlp, rows * mlp_hidden);

            // Attention and MLP are joined before a single output projection,
            // rather than run in sequence.
            s.run_paste(enc, &s.attn, &s.joined, hidden, hidden + mlp_hidden, 0, rows);
            s.run_paste(enc, &s.mlp, &s.joined, mlp_hidden, hidden + mlp_hidden, hidden, rows);
            s.gemm_into(enc, &b.linear2, &s.joined, &s.tmp, rows);
            s.run_gated_add(enc, &s.x, &s.tmp, &s.mod_img, n_tok, 2 * hidden);
        }

        // --- Final layer -----------------------------------------------------
        s.gemm_into(enc, &s.final_mod, &s.vec_tmp, &s.mod_img, K_TILE);
        s.run_copy_rows(enc, &s.x, &s.tmp, hidden, n_txt, 0, n_img);
        s.run_layer_norm(enc, &s.tmp, &s.attn, img_rows, hidden);
        // Two values here, not three, and in the order shift then scale.
        s.run_modulate(enc, &s.attn, &s.mod_img, img_rows, 0, hidden);
        s.gemm_into(enc, &s.final_linear, &s.attn, &s.fused, img_rows);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();

        if let Some(e) = cmd.error() {
            return Err(format!("metal flux: command buffer failed: {}", e.localizedDescription()));
        }

        let patch_dim = cfg.patch_dim();
        let mut tokens = Mat::zeros(n_img, patch_dim);
        let src = unsafe { host_f32(&s.fused) };
        tokens.data.copy_from_slice(&src[..n_img * patch_dim]);
        Ok(flux_unpatchify(&tokens, lat_h, lat_w, cfg.in_channels, patch))
    }
}

// =============================================================================
// Sampling
// =============================================================================

/// Sample with the GPU engine. The CPU-side `flux_sample` with the loop body
/// swapped, so the schedule and the Euler step stay in one place.
pub fn flux_sample_metal(
    ctx: &MetalFluxContext,
    cfg: &FluxConfig,
    context: &Mat,
    pooled: &[f32],
    params: &FluxSampleParams,
) -> Result<Mat, String> {
    const K_VAE_FACTOR: usize = 8;
    if params.width % (K_VAE_FACTOR * cfg.patch_size) != 0 || params.height % (K_VAE_FACTOR * cfg.patch_size) != 0 {
        return Err(format!(
            "flux sample: width and height must be multiples of {}",
            K_VAE_FACTOR * cfg.patch_size
        ));
    }
    if params.steps == 0 {
        return Err("flux sample: steps must be > 0".to_string());
    }

    let lat_h = params.height / K_VAE_FACTOR;
    let lat_w = params.width / K_VAE_FACTOR;

    let mut rng = InitRng::new(params.seed);
    let mut x = Mat::zeros(lat_h * lat_w, cfg.in_channels);
    for v in x.data.iter_mut() {
        *v = rng.next_normal();
    }

    let sigmas = flux_schedule(params.steps, params.shift);
    for i in 0..params.steps {
        let velocity = ctx.forward(&x, lat_h, lat_w, context, pooled, sigmas[i], params.guidance)?;
        let dt = sigmas[i + 1] - sigmas[i];
        for k in 0..x.data.len() {
            x.data[k] += dt * velocity.data[k];
        }
    }
    Ok(x)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flux::{flux_sample, FluxDoubleBlock, FluxSingleBlock};

    fn spread(rows: usize, cols: usize, scale: f32, salt: usize) -> Mat {
        Mat::from_fn(rows, cols, |r, c| {
            let i = ((r * 131 + c * 37 + salt * 7919) % 251) as f32;
            ((i * 0.41).sin() * 0.7 + (i * 0.13).cos() * 0.3) * scale
        })
    }

    fn spread_vec(n: usize, scale: f32, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = ((i * 53 + salt * 6151) % 241) as f32;
                ((k * 0.29).sin() * 0.6 + (k * 0.17).cos() * 0.4) * scale
            })
            .collect()
    }

    fn linear(out: usize, inp: usize, salt: usize) -> QLinear {
        QLinear::from_f32(spread(out, inp, 0.1, salt), spread_vec(out, 0.02, salt + 1))
    }

    /// The GEMM kernel is a 64x64 register-blocked tile with no bounds checks, so
    /// every width here is a multiple of 64 and every contracted dimension a
    /// multiple of 8 -- the same constraints the real FLUX widths satisfy.
    fn gpu_config() -> FluxConfig {
        FluxConfig {
            in_channels: 16, // patch_dim = 16 * 4 = 64
            hidden_size: 64,
            n_heads: 8, // head_dim 8
            n_double_blocks: 2,
            n_single_blocks: 2,
            mlp_ratio: 2.0, // mlp_hidden 128
            context_dim: 64,
            pooled_dim: 64,
            axes_dim: vec![2, 2, 4],
            guidance_embed: false,
            patch_size: 2,
            ..Default::default()
        }
    }

    fn make_model(cfg: &FluxConfig) -> FluxModel {
        let mut m = FluxModel { cfg: cfg.clone(), ..Default::default() };
        let h = cfg.hidden_size;
        let hd = cfg.head_dim();
        let mlp = cfg.mlp_hidden();

        m.img_in = linear(h, cfg.patch_dim(), 1);
        m.txt_in = linear(h, cfg.context_dim, 3);
        m.time_in_1 = linear(h, 256, 5);
        m.time_in_2 = linear(h, h, 7);
        m.vector_in_1 = linear(h, cfg.pooled_dim, 9);
        m.vector_in_2 = linear(h, h, 11);
        if cfg.guidance_embed {
            m.guidance_in_1 = linear(h, 256, 13);
            m.guidance_in_2 = linear(h, h, 15);
        }
        for i in 0..cfg.n_double_blocks {
            let mut b = FluxDoubleBlock::default();
            b.img_mod = linear(6 * h, h, 20 + i * 20);
            b.img_qkv = linear(3 * h, h, 22 + i * 20);
            b.img_proj = linear(h, h, 24 + i * 20);
            b.img_mlp_in = linear(mlp, h, 26 + i * 20);
            b.img_mlp_out = linear(h, mlp, 28 + i * 20);
            b.img_norm.query_scale = spread_vec(hd, 0.3, 40 + i);
            b.img_norm.key_scale = spread_vec(hd, 0.3, 41 + i);
            for v in b.img_norm.query_scale.iter_mut() {
                *v += 1.0;
            }
            for v in b.img_norm.key_scale.iter_mut() {
                *v += 1.0;
            }
            b.txt_mod = linear(6 * h, h, 30 + i * 20);
            b.txt_qkv = linear(3 * h, h, 32 + i * 20);
            b.txt_proj = linear(h, h, 34 + i * 20);
            b.txt_mlp_in = linear(mlp, h, 36 + i * 20);
            b.txt_mlp_out = linear(h, mlp, 38 + i * 20);
            b.txt_norm.query_scale = spread_vec(hd, 0.3, 42 + i);
            b.txt_norm.key_scale = spread_vec(hd, 0.3, 43 + i);
            for v in b.txt_norm.query_scale.iter_mut() {
                *v += 1.0;
            }
            for v in b.txt_norm.key_scale.iter_mut() {
                *v += 1.0;
            }
            m.double_blocks.push(b);
        }
        for i in 0..cfg.n_single_blocks {
            let mut b = FluxSingleBlock::default();
            b.modulation = linear(3 * h, h, 60 + i * 10);
            b.linear1 = linear(3 * h + mlp, h, 62 + i * 10);
            b.linear2 = linear(h, h + mlp, 64 + i * 10);
            b.norm.query_scale = spread_vec(hd, 0.3, 70 + i);
            b.norm.key_scale = spread_vec(hd, 0.3, 71 + i);
            for v in b.norm.query_scale.iter_mut() {
                *v += 1.0;
            }
            for v in b.norm.key_scale.iter_mut() {
                *v += 1.0;
            }
            m.single_blocks.push(b);
        }
        m.final_mod = linear(2 * h, h, 90);
        m.final_linear = linear(cfg.patch_dim(), h, 92);
        m
    }

    #[derive(Debug, Default)]
    struct Deviation {
        max_abs: f32,
        /// Error energy as a fraction of signal energy.
        ///
        /// The right metric for a velocity field. A pointwise relative error is
        /// not: the field crosses zero all over, and a value that happens to land
        /// near zero reports an enormous relative deviation for an absolute one
        /// that is pure rounding.
        rel_rms: f32,
    }

    fn compare(a: &Mat, b: &Mat) -> Deviation {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        let mut d = Deviation::default();
        let mut err_sq = 0.0f64;
        let mut ref_sq = 0.0f64;
        for i in 0..a.data.len() {
            let diff = a.data[i] as f64 - b.data[i] as f64;
            err_sq += diff * diff;
            ref_sq += a.data[i] as f64 * a.data[i] as f64;
            d.max_abs = d.max_abs.max(diff.abs() as f32);
        }
        d.rel_rms = if ref_sq > 0.0 { (err_sq / ref_sq).sqrt() as f32 } else { 0.0 };
        d
    }

    /// The GEMM contracts against a bfloat weight, which keeps eight mantissa
    /// bits, so the two paths agree to a fraction of a percent and no further.
    /// Measured at 0.5-0.7% RMS-relative across block counts -- and, importantly,
    /// flat in the number of blocks: a structural disagreement would compound with
    /// depth, and this does not.
    const K_BF16_TOLERANCE: f32 = 0.02;

    #[test]
    fn matches_the_cpu_forward_pass() {
        let cfg = gpu_config();
        // Two copies: `create` consumes the weights of the model it uploads, so the
        // reference has to be a second build of the same deterministic weights.
        let mut gpu_model = make_model(&cfg);
        let cpu_model = make_model(&cfg);

        let lat_h = 8;
        let lat_w = 6;
        let n_txt = 5;
        let z = spread(lat_h * lat_w, cfg.in_channels, 1.0, 500);
        let ctx = spread(n_txt, cfg.context_dim, 1.0, 501);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 502);

        let engine = MetalFluxContext::create(&mut gpu_model, lat_h, lat_w, n_txt).expect("create");

        let expected = cpu_model.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.75, 0.0).expect("cpu forward");
        let got = engine.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.75, 0.0).expect("gpu forward");

        let d = compare(&expected, &got);
        assert!(d.rel_rms < K_BF16_TOLERANCE, "max abs {}, rel rms {}", d.max_abs, d.rel_rms);
    }

    #[test]
    fn is_deterministic() {
        let cfg = gpu_config();
        let mut model = make_model(&cfg);
        let z = spread(4 * 4, cfg.in_channels, 1.0, 503);
        let ctx = spread(3, cfg.context_dim, 1.0, 504);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 505);

        let engine = MetalFluxContext::create(&mut model, 4, 4, 3).expect("create");

        let a = engine.forward(&z, 4, 4, &ctx, &pooled, 0.5, 0.0).expect("forward a");
        let b = engine.forward(&z, 4, 4, &ctx, &pooled, 0.5, 0.0).expect("forward b");
        assert!(a.data == b.data);
    }

    #[test]
    fn handles_a_latent_smaller_than_its_allocation() {
        let cfg = gpu_config();
        let mut gpu_model = make_model(&cfg);
        let cpu_model = make_model(&cfg);

        let engine = MetalFluxContext::create(&mut gpu_model, 12, 12, 8).expect("create");

        // A token count that is not a tile multiple, so the pad rows are exercised.
        let lat_h = 6;
        let lat_w = 6;
        let n_txt = 3;
        let z = spread(lat_h * lat_w, cfg.in_channels, 1.0, 506);
        let ctx = spread(n_txt, cfg.context_dim, 1.0, 507);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 508);
        assert_eq!((lat_h / 2) * (lat_w / 2) + n_txt, 12); // not a multiple of 64

        let expected = cpu_model.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.3, 0.0).expect("cpu forward");
        let got = engine.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.3, 0.0).expect("gpu forward");

        let d = compare(&expected, &got);
        assert!(d.rel_rms < K_BF16_TOLERANCE, "max abs {}, rel rms {}", d.max_abs, d.rel_rms);
    }

    #[test]
    fn follows_the_guidance_embedding_when_the_config_has_one() {
        let mut cfg = gpu_config();
        cfg.guidance_embed = true;
        let mut gpu_model = make_model(&cfg);
        let cpu_model = make_model(&cfg);

        let z = spread(4 * 4, cfg.in_channels, 1.0, 509);
        let ctx = spread(3, cfg.context_dim, 1.0, 510);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 511);

        let engine = MetalFluxContext::create(&mut gpu_model, 4, 4, 3).expect("create");

        let expected = cpu_model.forward(&z, 4, 4, &ctx, &pooled, 0.5, 3.5).expect("cpu forward");
        let got = engine.forward(&z, 4, 4, &ctx, &pooled, 0.5, 3.5).expect("gpu forward");
        let d = compare(&expected, &got);
        assert!(d.rel_rms < K_BF16_TOLERANCE, "max abs {}, rel rms {}", d.max_abs, d.rel_rms);

        // And the guidance value actually reaches the graph.
        let other = engine.forward(&z, 4, 4, &ctx, &pooled, 0.5, 1.0).expect("gpu forward");
        assert!(other.data != got.data);
    }

    #[test]
    fn survives_activations_large_enough_to_saturate_tanh() {
        // The GELU's cubic term grows fast: a pre-activation of 27 -- ordinary in a
        // real FLUX MLP -- puts the tanh argument near 724, and Metal's fast-math
        // tanh evaluates exp(2x), which is inf there. inf/inf is NaN, one NaN in a
        // residual stream poisons every later block, and the image comes out black.
        //
        // The synthetic weights the other cases use never reach that magnitude, so
        // this one drives the activations up on purpose.
        let cfg = gpu_config();
        let mut gpu_model = make_model(&cfg);
        let cpu_model = make_model(&cfg);

        let lat_h = 4;
        let lat_w = 4;
        let n_txt = 3;
        let z = spread(lat_h * lat_w, cfg.in_channels, 40.0, 600);
        let ctx = spread(n_txt, cfg.context_dim, 40.0, 601);
        let pooled = spread_vec(cfg.pooled_dim, 40.0, 602);

        let engine = MetalFluxContext::create(&mut gpu_model, lat_h, lat_w, n_txt).expect("create");

        let expected = cpu_model.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.5, 0.0).expect("cpu forward");
        let got = engine.forward(&z, lat_h, lat_w, &ctx, &pooled, 0.5, 0.0).expect("gpu forward");

        for &v in &got.data {
            assert!(v.is_finite());
        }
        let d = compare(&expected, &got);
        assert!(d.rel_rms < K_BF16_TOLERANCE, "max abs {}, rel rms {}", d.max_abs, d.rel_rms);
    }

    #[test]
    fn rejects_work_it_has_no_scratch_for() {
        let cfg = gpu_config();
        let mut model = make_model(&cfg);
        let engine = MetalFluxContext::create(&mut model, 4, 4, 3).expect("create");

        let pooled = spread_vec(cfg.pooled_dim, 1.0, 512);
        // Larger latent than the allocation.
        assert!(engine
            .forward(&spread(8 * 8, cfg.in_channels, 1.0, 1), 8, 8, &spread(3, cfg.context_dim, 1.0, 1), &pooled, 0.5, 0.0)
            .is_err());
        // Longer prompt than the allocation.
        assert!(engine
            .forward(&spread(4 * 4, cfg.in_channels, 1.0, 1), 4, 4, &spread(9, cfg.context_dim, 1.0, 1), &pooled, 0.5, 0.0)
            .is_err());
        // A latent that cannot be patched.
        assert!(engine
            .forward(&spread(3 * 4, cfg.in_channels, 1.0, 1), 3, 4, &spread(3, cfg.context_dim, 1.0, 1), &pooled, 0.5, 0.0)
            .is_err());
    }

    #[test]
    fn refuses_a_config_its_gemm_cannot_tile() {
        let mut cfg = gpu_config();
        cfg.hidden_size = 96; // not a multiple of 64
        cfg.n_heads = 8;
        let mut model = make_model(&cfg);
        assert!(MetalFluxContext::create(&mut model, 4, 4, 3).is_err());
    }

    #[test]
    fn flux_sample_metal_reproduces_the_cpu_samplers_trajectory() {
        let cfg = gpu_config();
        let mut gpu_model = make_model(&cfg);
        let cpu_model = make_model(&cfg);

        let ctx = spread(4, cfg.context_dim, 1.0, 513);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 514);

        let p = FluxSampleParams { width: 64, height: 32, steps: 2, seed: 11, ..Default::default() };

        let engine = MetalFluxContext::create(&mut gpu_model, 32 / 8, 64 / 8, 4).expect("create");

        let cpu = flux_sample(&cpu_model, &ctx, &pooled, &p).expect("cpu sample");
        let gpu = flux_sample_metal(&engine, &cfg, &ctx, &pooled, &p).expect("gpu sample");

        let d = compare(&cpu, &gpu);
        // The latent is mostly the initial noise after two steps, so the agreement
        // is tighter here than on a raw velocity.
        assert!(d.rel_rms < K_BF16_TOLERANCE / 2.0, "max abs {}, rel rms {}", d.max_abs, d.rel_rms);
    }
}
