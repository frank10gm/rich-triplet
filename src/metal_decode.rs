/// # Metal full-graph decode engine
///
/// Encodes the **entire** Gemma3 decode forward pass (embedding → 34 layers →
/// final norm → lm_head) into a **single Metal command buffer**.  This avoids
/// the ~0.5 ms per-dispatch overhead that makes individual GEMV calls slower
/// than CPU SDOT on Apple Silicon.
///
/// ## Design
///
/// * `MetalDecodeContext` is created once after weight loading.  It uploads all
///   weights into persistent `MTLBuffer`s, pre-allocates activation scratch
///   buffers, and compiles all MSL pipelines.
/// * Each `decode_step(token_id, position)` call:
///   1. Creates **one** command buffer + **one** compute encoder
///   2. Encodes ~717 dispatches (21 per layer × 34 layers + 3)
///   3. `endEncoding()`, `commit()`, `waitUntilCompleted()`
///   4. Reads logits back from shared-memory buffer
/// * Within a single compute encoder on Apple Silicon, dispatches execute
///   in order and writes are visible to subsequent dispatches (unified L2).
/// * Per-dispatch constants (dimensions, position, theta) are passed via
///   `setBytes` — Metal copies them inline, no buffer allocation needed.

#[cfg(feature = "metal")]
pub mod inner {
    use std::ptr::NonNull;
    use std::ffi::c_void;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::*;

    use crate::transformer4::{Config4, Gemma3Model, Gemma3KvCache};

    // =========================================================================
    // MSL kernel source
    // =========================================================================

    /// All graph-mode kernels compiled from a single MSL source string.
    const DECODE_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── RMSNorm (Gemma3 variant: multiply by (1 + gamma)) ────────────────────
//
// Input:  x[D], gamma[D]
// Output: out[D] = x[i] / rms * (1 + gamma[i])
// One threadgroup, D threads (D ≤ 1024 for Gemma3 hidden_size, but we use
// 256 threads and loop).
// Uses threadgroup memory for parallel sum-of-squares reduction.

kernel void rms_norm_gemma3(
    device const float* x     [[ buffer(0) ]],
    device const float* gamma [[ buffer(1) ]],
    device       float* out   [[ buffer(2) ]],
    constant     uint&  D     [[ buffer(3) ]],
    constant     float& eps   [[ buffer(4) ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    // Phase 1: compute sum of squares via simd_sum
    float partial_sq = 0.0f;
    for (uint i = lid; i < D; i += tg_size) {
        float v = x[i];
        partial_sq += v * v;
    }
    float sum_sq = simd_sum(partial_sq);

    // If tg_size > 32, need threadgroup reduction across simd groups
    threadgroup float shared_sq[32];
    uint simd_lane = lid % 32u;
    uint simd_group = lid / 32u;
    uint n_simd_groups = (tg_size + 31u) / 32u;

    if (simd_lane == 0u) {
        shared_sq[simd_group] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (lid == 0u) {
        float total = 0.0f;
        for (uint g = 0u; g < n_simd_groups; g++) {
            total += shared_sq[g];
        }
        shared_sq[0] = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms_inv = rsqrt(shared_sq[0] / float(D) + eps);

    // Phase 2: normalize and scale
    for (uint i = lid; i < D; i += tg_size) {
        out[i] = x[i] * rms_inv * (1.0f + gamma[i]);
    }
}

// ── Per-head RMSNorm ──────────────────────────────────────────────────────
//
// Input:  x[n_heads * head_dim], gamma[head_dim] (shared across heads)
// Output: out[n_heads * head_dim]
// One threadgroup per head.  32 threads per threadgroup.

kernel void rms_norm_per_head(
    device const float* x     [[ buffer(0) ]],
    device const float* gamma [[ buffer(1) ]],
    device       float* out   [[ buffer(2) ]],
    constant     uint&  n_heads   [[ buffer(3) ]],
    constant     uint&  head_dim  [[ buffer(4) ]],
    constant     float& eps       [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    uint h = tgid;
    if (h >= n_heads) return;
    uint base = h * head_dim;

    float partial_sq = 0.0f;
    for (uint i = lid; i < head_dim; i += tg_size) {
        float v = x[base + i];
        partial_sq += v * v;
    }
    float sum_sq = simd_sum(partial_sq);

    // With 32 threads (one simd group), simd_sum is sufficient
    float rms_inv = rsqrt(sum_sq / float(head_dim) + eps);

    for (uint i = lid; i < head_dim; i += tg_size) {
        out[base + i] = x[base + i] * rms_inv * (1.0f + gamma[i]);
    }
}

// ── RoPE (NeoX half-split) ───────────────────────────────────────────────
//
// Pairs (i, i + half) where half = head_dim / 2.
// One threadgroup per head, 128 threads (one per pair).

kernel void rope_neox(
    device const float* x         [[ buffer(0) ]],
    device       float* out       [[ buffer(1) ]],
    constant     uint&  n_heads   [[ buffer(2) ]],
    constant     uint&  head_dim  [[ buffer(3) ]],
    constant     float& theta     [[ buffer(4) ]],
    constant     float& freq_scale[[ buffer(5) ]],
    constant     uint&  position  [[ buffer(6) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]])
{
    uint h = tgid;
    if (h >= n_heads) return;
    uint half_dim = head_dim / 2u;
    if (lid >= half_dim) return;

    uint i = lid;
    uint base = h * head_dim;
    float angle = (float(position) * freq_scale) / pow(theta, 2.0f * float(i) / float(head_dim));
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    float x0 = x[base + i];
    float x1 = x[base + i + half_dim];
    out[base + i]        = x0 * cos_a - x1 * sin_a;
    out[base + i + half_dim] = x0 * sin_a + x1 * cos_a;
}

// ── GELU (tanh approximation) ────────────────────────────────────────────

kernel void gelu_tanh_kernel(
    device const float* x   [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    constant     uint&  N   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= N) return;
    float v = x[gid];
    float v3 = v * v * v;
    float inner = 0.7978845608f * (v + 0.044715f * v3);
    // Clamp inner before tanh to avoid fast-math NaN:
    // Metal's fast-math tanh uses exp(2x), which overflows for |x| > ~44.
    float t = tanh(clamp(inner, -10.0f, 10.0f));
    out[gid] = 0.5f * v * (1.0f + t);
}

// ── Element-wise multiply ────────────────────────────────────────────────

kernel void elem_mul_kernel(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  N   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= N) return;
    out[gid] = a[gid] * b[gid];
}

// ── Vector add ───────────────────────────────────────────────────────────

kernel void vec_add_kernel(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  N   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= N) return;
    out[gid] = a[gid] + b[gid];
}

// ── KV cache append ──────────────────────────────────────────────────────
//
// Write new K and V rows into the cache at row `seq_len`.

kernel void kv_cache_append(
    device const float* k_new   [[ buffer(0) ]],
    device const float* v_new   [[ buffer(1) ]],
    device       float* k_cache [[ buffer(2) ]],
    device       float* v_cache [[ buffer(3) ]],
    constant     uint&  seq_len [[ buffer(4) ]],
    constant     uint&  kv_dim  [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= kv_dim) return;
    uint offset = seq_len * kv_dim + gid;
    k_cache[offset] = k_new[gid];
    v_cache[offset] = v_new[gid];
}

// ── BF16 embedding lookup ────────────────────────────────────────────────

kernel void embed_bf16_lookup(
    device const ushort* embed_table [[ buffer(0) ]],
    device       float*  out         [[ buffer(1) ]],
    constant     uint&   token_id    [[ buffer(2) ]],
    constant     uint&   hidden_size [[ buffer(3) ]],
    constant     float&  scale       [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= hidden_size) return;
    ushort bits = embed_table[token_id * hidden_size + gid];
    float w = as_type<float>(uint(bits) << 16u);
    out[gid] = w * scale;
}

// ── Fused decode attention ───────────────────────────────────────────────
//
// For a single query token (t_q = 1), compute GQA attention over the
// cached K and V.
//
// One threadgroup per Q head.  Each threadgroup has TG_ATTN threads.
// GQA mapping: kvh = qh / group_size.
//
// Phase 1: Compute scores[c] = dot(q_h, k_h[c]) * scale for c in [k_start..k_end)
// Phase 2: Softmax (max-subtract, exp, sum, normalize)
// Phase 3: out_h = sum_c( scores[c] * v_h[c] )

#define TG_ATTN 32u

kernel void attention_decode(
    device const float* q         [[ buffer(0)  ]],
    device const float* k_cache   [[ buffer(1)  ]],
    device const float* v_cache   [[ buffer(2)  ]],
    device       float* out       [[ buffer(3)  ]],
    constant     uint&  k_start   [[ buffer(4)  ]],
    constant     uint&  k_end     [[ buffer(5)  ]],
    constant     uint&  n_q_heads [[ buffer(6)  ]],
    constant     uint&  n_kv_heads[[ buffer(7)  ]],
    constant     uint&  d_head    [[ buffer(8)  ]],
    constant     float& scale     [[ buffer(9)  ]],
    constant     uint&  kv_stride [[ buffer(10) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint lid  [[ thread_index_in_threadgroup ]])
{
    uint qh = tgid;
    if (qh >= n_q_heads) return;
    uint group = n_q_heads / n_kv_heads;
    uint kvh = qh / group;
    uint q_off = qh * d_head;
    uint kv_off = kvh * d_head;
    uint t_kv = k_end - k_start;

    // ── Phase 1: compute scores ──
    // Each thread handles a strided subset of context positions.
    // Threadgroup memory for scores (max 2048 context tokens is typical).
    threadgroup float tg_scores[2048];

    for (uint ci = lid; ci < t_kv; ci += TG_ATTN) {
        uint cache_row = k_start + ci;
        uint k_base = cache_row * kv_stride + kv_off;
        float dot = 0.0f;
        for (uint di = 0u; di < d_head; di++) {
            dot += q[q_off + di] * k_cache[k_base + di];
        }
        tg_scores[ci] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: softmax ──
    // Find max via parallel reduction
    float local_max = -INFINITY;
    for (uint ci = lid; ci < t_kv; ci += TG_ATTN) {
        local_max = max(local_max, tg_scores[ci]);
    }
    float max_s = simd_max(local_max);

    // Exp and sum
    float local_sum = 0.0f;
    for (uint ci = lid; ci < t_kv; ci += TG_ATTN) {
        float e = exp(tg_scores[ci] - max_s);
        tg_scores[ci] = e;
        local_sum += e;
    }
    float sum_exp = simd_sum(local_sum);

    // Normalize
    float inv_sum = 1.0f / sum_exp;
    for (uint ci = lid; ci < t_kv; ci += TG_ATTN) {
        tg_scores[ci] *= inv_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 3: weighted V sum ──
    // Each thread accumulates partial weighted V, then simd_sum across lanes.
    for (uint di = 0u; di < d_head; di++) {
        float partial = 0.0f;
        for (uint ci = lid; ci < t_kv; ci += TG_ATTN) {
            uint cache_row = k_start + ci;
            uint v_base = cache_row * kv_stride + kv_off;
            partial += tg_scores[ci] * v_cache[v_base + di];
        }
        float total = simd_sum(partial);
        if (lid == 0u) {
            out[q_off + di] = total;
        }
    }
}

// ── Q4K GEMV (copied from metal_ops.rs) ──────────────────────────────────

#define Q4K_BLOCK_BYTES 144u
#define Q4K_BLOCK_ELEMS 256u
#define TG_K 32u

inline float2 scale_min(device const uchar* sc, uint j) {
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

kernel void gemv_q4k_t(
    device const float* A         [[ buffer(0) ]],
    device const uchar* blocks    [[ buffer(1) ]],
    device       float* C         [[ buffer(2) ]],
    constant     uint&  K         [[ buffer(3) ]],
    constant     uint&  N         [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    uint n_sb = K / Q4K_BLOCK_ELEMS;
    float acc = 0.0f;

    for (uint b = lid; b < n_sb; b += TG_K) {
        uint boff = (j * n_sb + b) * Q4K_BLOCK_BYTES;
        device const uchar* bp = blocks + boff;

        ushort d_bits    = ushort(bp[0]) | (ushort(bp[1]) << 8u);
        ushort dmin_bits = ushort(bp[2]) | (ushort(bp[3]) << 8u);
        float d    = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = bp + 4u;
        device const uchar* qs = bp + 16u;
        uint a_base = b * Q4K_BLOCK_ELEMS;

        for (uint chunk = 0u; chunk < 4u; chunk++) {
            float2 sm1 = scale_min(sc, chunk * 2u);
            float2 sm2 = scale_min(sc, chunk * 2u + 1u);
            float scale1 = d * sm1.x;  float min1 = dmin * sm1.y;
            float scale2 = d * sm2.x;  float min2 = dmin * sm2.y;

            device const uchar* qq = qs + chunk * 32u;
            uint a_off = a_base + chunk * 64u;

            float dot_lo = 0.0f, dot_hi = 0.0f;
            float sum_lo = 0.0f, sum_hi = 0.0f;

            for (uint l = 0u; l < 32u; l++) {
                float a_lo = A[a_off + l];
                float a_hi = A[a_off + 32u + l];
                dot_lo += float(qq[l] & 0x0Fu) * a_lo;
                dot_hi += float(qq[l] >> 4u)   * a_hi;
                sum_lo += a_lo;
                sum_hi += a_hi;
            }

            acc += scale1 * dot_lo - min1 * sum_lo
                 + scale2 * dot_hi - min2 * sum_hi;
        }
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
}

// ── BF16 GEMV (copied from metal_ops.rs) ─────────────────────────────────

#define TG_K_BF16 32u

kernel void gemv_bf16_t(
    device const float*  A      [[ buffer(0) ]],
    device const ushort* W      [[ buffer(1) ]],
    device       float*  C      [[ buffer(2) ]],
    constant     uint&   K      [[ buffer(3) ]],
    constant     uint&   N      [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    float acc = 0.0f;
    uint row_start = j * K;

    for (uint p = lid; p < K; p += TG_K_BF16) {
        ushort bits = W[row_start + p];
        float w = as_type<float>(uint(bits) << 16u);
        acc += A[p] * w;
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
}
"#;

    // =========================================================================
    // Per-layer weight buffer collection
    // =========================================================================

    pub struct LayerWeightBuffers {
        pub q_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub q_proj_is_bf16: bool,
        pub k_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub k_proj_is_bf16: bool,
        pub v_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub v_proj_is_bf16: bool,
        pub o_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub o_proj_is_bf16: bool,
        pub gate_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub gate_proj_is_bf16: bool,
        pub up_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub up_proj_is_bf16: bool,
        pub down_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub down_proj_is_bf16: bool,

        // Norm gamma buffers (f32)
        pub input_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub post_attn_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub pre_ffn_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub post_ffn_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub q_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub k_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,

        // Per-layer attention config
        pub rope_theta: f32,
        pub rope_freq_scale: f32,
        pub sliding_window: Option<usize>,
    }

    // =========================================================================
    // MetalDecodeContext
    // =========================================================================

    pub struct MetalDecodeContext {
        // Core Metal objects
        #[allow(dead_code)]
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

        // Pipelines
        pipe_rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_rms_norm_per_head: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_rope_neox: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gelu_tanh: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_elem_mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_vec_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_attention_decode: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_kv_append: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_embed_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gemv_q4k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gemv_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

        // Per-layer weights
        layer_weights: Vec<LayerWeightBuffers>,

        // Embedding table (BF16 u16 data)
        embed_buf: Retained<ProtocolObject<dyn MTLBuffer>>,

        // lm_head weight — for weight-tied models this IS embed_buf, but we
        // track separately in case they differ.
        lm_head_buf: Retained<ProtocolObject<dyn MTLBuffer>>,

        // Final norm gamma
        final_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,

        // ── Activation scratch buffers ──
        buf_hidden: Retained<ProtocolObject<dyn MTLBuffer>>,      // [1, hidden_size]
        buf_hidden2: Retained<ProtocolObject<dyn MTLBuffer>>,     // [1, hidden_size]
        buf_normed: Retained<ProtocolObject<dyn MTLBuffer>>,      // [1, hidden_size]
        buf_q: Retained<ProtocolObject<dyn MTLBuffer>>,           // [1, nq * head_dim]
        buf_k: Retained<ProtocolObject<dyn MTLBuffer>>,           // [1, nkv * head_dim]
        buf_v: Retained<ProtocolObject<dyn MTLBuffer>>,           // [1, nkv * head_dim]
        buf_q_normed: Retained<ProtocolObject<dyn MTLBuffer>>,    // [1, nq * head_dim]
        buf_k_normed: Retained<ProtocolObject<dyn MTLBuffer>>,    // [1, nkv * head_dim]
        buf_q_roped: Retained<ProtocolObject<dyn MTLBuffer>>,     // [1, nq * head_dim]
        buf_k_roped: Retained<ProtocolObject<dyn MTLBuffer>>,     // [1, nkv * head_dim]
        buf_attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,    // [1, nq * head_dim]
        buf_o_proj_out: Retained<ProtocolObject<dyn MTLBuffer>>,  // [1, hidden_size]
        buf_gate: Retained<ProtocolObject<dyn MTLBuffer>>,        // [1, intermediate_size]
        buf_up: Retained<ProtocolObject<dyn MTLBuffer>>,          // [1, intermediate_size]
        buf_gate_act: Retained<ProtocolObject<dyn MTLBuffer>>,    // [1, intermediate_size]
        buf_mlp_hidden: Retained<ProtocolObject<dyn MTLBuffer>>,  // [1, intermediate_size]
        buf_down_out: Retained<ProtocolObject<dyn MTLBuffer>>,    // [1, hidden_size]
        buf_logits: Retained<ProtocolObject<dyn MTLBuffer>>,      // [1, vocab_size]

        // KV cache buffers
        kv_k_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        kv_v_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,

        // Model config
        config: Config4,
        #[allow(dead_code)]
        max_seq_len: usize,
    }

    // =========================================================================
    // Buffer helpers
    // =========================================================================

    fn alloc_buf(
        device: &ProtocolObject<dyn MTLDevice>,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        // Ensure we always allocate at least 16 bytes (Metal requirement)
        let len = byte_len.max(16);
        device
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .expect("Metal: buffer allocation failed")
    }

    fn upload_f32(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[f32],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_len = data.len() * 4;
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len.max(16),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: buffer allocation failed")
        }
    }

    fn upload_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_len = data.len();
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len.max(16),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: buffer allocation failed")
        }
    }

    fn upload_u16(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[u16],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_len = data.len() * 2;
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len.max(16),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: buffer allocation failed")
        }
    }

    fn compile_pipeline(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
        func_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let src = NSString::from_str(source);
        let library = device
            .newLibraryWithSource_options_error(&src, None)
            .expect(&format!("Metal: failed to compile MSL for {}", func_name));
        let name = NSString::from_str(func_name);
        let func = library
            .newFunctionWithName(&name)
            .expect(&format!("Metal: function '{}' not found", func_name));
        device
            .newComputePipelineStateWithFunction_error(&func)
            .expect(&format!("Metal: pipeline creation failed for {}", func_name))
    }

    // =========================================================================
    // Construction
    // =========================================================================

    impl MetalDecodeContext {
        /// Create a new Metal decode context by uploading all model weights
        /// and allocating activation buffers.
        ///
        /// Call after model weights are loaded.  The `model` is borrowed
        /// read-only; its weight data is copied into persistent Metal buffers.
        pub fn new(model: &Gemma3Model) -> Self {
            let t_start = std::time::Instant::now();
            let cfg = &model.config;
            let device = MTLCreateSystemDefaultDevice()
                .expect("Metal: no GPU device found");
            let queue = device
                .newCommandQueue()
                .expect("Metal: command queue creation failed");

            // Compile all pipelines from the combined MSL source
            let pipe_rms_norm = compile_pipeline(&device, DECODE_MSL, "rms_norm_gemma3");
            let pipe_rms_norm_per_head = compile_pipeline(&device, DECODE_MSL, "rms_norm_per_head");
            let pipe_rope_neox = compile_pipeline(&device, DECODE_MSL, "rope_neox");
            let pipe_gelu_tanh = compile_pipeline(&device, DECODE_MSL, "gelu_tanh_kernel");
            let pipe_elem_mul = compile_pipeline(&device, DECODE_MSL, "elem_mul_kernel");
            let pipe_vec_add = compile_pipeline(&device, DECODE_MSL, "vec_add_kernel");
            let pipe_attention_decode = compile_pipeline(&device, DECODE_MSL, "attention_decode");
            let pipe_kv_append = compile_pipeline(&device, DECODE_MSL, "kv_cache_append");
            let pipe_embed_bf16 = compile_pipeline(&device, DECODE_MSL, "embed_bf16_lookup");
            let pipe_gemv_q4k = compile_pipeline(&device, DECODE_MSL, "gemv_q4k_t");
            let pipe_gemv_bf16 = compile_pipeline(&device, DECODE_MSL, "gemv_bf16_t");

            // Upload per-layer weights
            let mut layer_weights = Vec::with_capacity(cfg.num_hidden_layers);
            for (_i, layer) in model.layers.iter().enumerate() {
                let attn = &layer.self_attn;
                let mlp = &layer.mlp;

                // Upload projection weights — detect Q4K vs BF16 for each
                let (q_proj, q_is_bf16) = upload_linear_detect_type(&device, &attn.q_proj);
                let (k_proj, k_is_bf16) = upload_linear_detect_type(&device, &attn.k_proj);
                let (v_proj, v_is_bf16) = upload_linear_detect_type(&device, &attn.v_proj);
                let (o_proj, o_is_bf16) = upload_linear_detect_type(&device, &attn.o_proj);
                let (gate_proj, gate_is_bf16) = upload_linear_detect_type(&device, &mlp.gate_proj);
                let (up_proj, up_is_bf16) = upload_linear_detect_type(&device, &mlp.up_proj);
                let (down_proj, down_is_bf16) = upload_linear_detect_type(&device, &mlp.down_proj);

                // Upload norm gamma vectors
                let input_ln_g = upload_f32(&device, &layer.input_layernorm.gamma.data().data);
                let post_attn_ln_g = upload_f32(&device, &layer.post_attention_layernorm.gamma.data().data);
                let pre_ffn_ln_g = upload_f32(&device, &layer.pre_feedforward_layernorm.gamma.data().data);
                let post_ffn_ln_g = upload_f32(&device, &layer.post_feedforward_layernorm.gamma.data().data);
                let q_norm_g = upload_f32(&device, &attn.q_norm.gamma.data().data);
                let k_norm_g = upload_f32(&device, &attn.k_norm.gamma.data().data);

                layer_weights.push(LayerWeightBuffers {
                    q_proj,
                    q_proj_is_bf16: q_is_bf16,
                    k_proj,
                    k_proj_is_bf16: k_is_bf16,
                    v_proj,
                    v_proj_is_bf16: v_is_bf16,
                    o_proj,
                    o_proj_is_bf16: o_is_bf16,
                    gate_proj,
                    gate_proj_is_bf16: gate_is_bf16,
                    up_proj,
                    up_proj_is_bf16: up_is_bf16,
                    down_proj,
                    down_proj_is_bf16: down_is_bf16,
                    input_layernorm_gamma: input_ln_g,
                    post_attn_layernorm_gamma: post_attn_ln_g,
                    pre_ffn_layernorm_gamma: pre_ffn_ln_g,
                    post_ffn_layernorm_gamma: post_ffn_ln_g,
                    q_norm_gamma: q_norm_g,
                    k_norm_gamma: k_norm_g,
                    rope_theta: attn.rope_theta,
                    rope_freq_scale: attn.rope_freq_scale,
                    sliding_window: attn.sliding_window,
                });
            }

            // Upload embedding table (BF16)
            let embed_buf = if let Some(ref bf16) = model.embed_bf16 {
                upload_u16(&device, &bf16.data)
            } else {
                // Fallback: upload f32 embed as-is (shouldn't happen in practice)
                upload_f32(&device, &model.embed_tokens.data().data)
            };

            // lm_head: weight-tied to embed_tokens in Gemma3
            // The lm_head Linear2 should have a bf16_weight that points to the
            // same data as embed_bf16.  Upload it (or reuse embed_buf).
            let lm_head_buf = if let Some(ref bf16) = model.lm_head.bf16_weight {
                // Check if it's the same Arc as embed_bf16
                if model.embed_bf16.is_some()
                    && std::sync::Arc::ptr_eq(&bf16.data, &model.embed_bf16.as_ref().unwrap().data)
                {
                    embed_buf.clone()
                } else {
                    upload_u16(&device, &bf16.data)
                }
            } else if let Some(ref q4k) = model.lm_head.q4k_weight {
                upload_bytes(&device, &q4k.blocks)
            } else {
                upload_f32(&device, &model.lm_head.weight.data().data)
            };

            // Final norm gamma
            let final_norm_gamma = upload_f32(&device, &model.norm.gamma.data().data);

            // Activation scratch buffers
            let h = cfg.hidden_size;
            let nq = cfg.num_attention_heads;
            let nkv = cfg.num_key_value_heads;
            let d = cfg.head_dim;
            let inter = cfg.intermediate_size;
            let vocab = cfg.vocab_size;

            let buf_hidden = alloc_buf(&device, h * 4);
            let buf_hidden2 = alloc_buf(&device, h * 4);
            let buf_normed = alloc_buf(&device, h * 4);
            let buf_q = alloc_buf(&device, nq * d * 4);
            let buf_k = alloc_buf(&device, nkv * d * 4);
            let buf_v = alloc_buf(&device, nkv * d * 4);
            let buf_q_normed = alloc_buf(&device, nq * d * 4);
            let buf_k_normed = alloc_buf(&device, nkv * d * 4);
            let buf_q_roped = alloc_buf(&device, nq * d * 4);
            let buf_k_roped = alloc_buf(&device, nkv * d * 4);
            let buf_attn_out = alloc_buf(&device, nq * d * 4);
            let buf_o_proj_out = alloc_buf(&device, h * 4);
            let buf_gate = alloc_buf(&device, inter * 4);
            let buf_up = alloc_buf(&device, inter * 4);
            let buf_gate_act = alloc_buf(&device, inter * 4);
            let buf_mlp_hidden = alloc_buf(&device, inter * 4);
            let buf_down_out = alloc_buf(&device, h * 4);
            let buf_logits = alloc_buf(&device, vocab * 4);

            // KV cache buffers
            let max_seq_len = cfg.max_position_embeddings.min(8192); // cap for memory
            let kv_dim = nkv * d;
            let mut kv_k_bufs = Vec::with_capacity(cfg.num_hidden_layers);
            let mut kv_v_bufs = Vec::with_capacity(cfg.num_hidden_layers);
            for _ in 0..cfg.num_hidden_layers {
                kv_k_bufs.push(alloc_buf(&device, max_seq_len * kv_dim * 4));
                kv_v_bufs.push(alloc_buf(&device, max_seq_len * kv_dim * 4));
            }

            let elapsed = t_start.elapsed();
            eprintln!(
                "[ Metal ] Decode context initialized in {:.0} ms ({} layers, {} pipelines)",
                elapsed.as_millis(),
                cfg.num_hidden_layers,
                11
            );

            MetalDecodeContext {
                device,
                queue,
                pipe_rms_norm,
                pipe_rms_norm_per_head,
                pipe_rope_neox,
                pipe_gelu_tanh,
                pipe_elem_mul,
                pipe_vec_add,
                pipe_attention_decode,
                pipe_kv_append,
                pipe_embed_bf16,
                pipe_gemv_q4k,
                pipe_gemv_bf16,
                layer_weights,
                embed_buf,
                lm_head_buf,
                final_norm_gamma,
                buf_hidden,
                buf_hidden2,
                buf_normed,
                buf_q,
                buf_k,
                buf_v,
                buf_q_normed,
                buf_k_normed,
                buf_q_roped,
                buf_k_roped,
                buf_attn_out,
                buf_o_proj_out,
                buf_gate,
                buf_up,
                buf_gate_act,
                buf_mlp_hidden,
                buf_down_out,
                buf_logits,
                kv_k_bufs,
                kv_v_bufs,
                config: cfg.clone(),
                max_seq_len,
            }
        }
    }

    /// Upload and detect whether a Linear2 is BF16 or Q4K.
    fn upload_linear_detect_type(
        device: &ProtocolObject<dyn MTLDevice>,
        linear: &crate::nn2::Linear2,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, bool) {
        if let Some(ref q4k) = linear.q4k_weight {
            (upload_bytes(device, &q4k.blocks), false)
        } else if let Some(ref bf16) = linear.bf16_weight {
            (upload_u16(device, &bf16.data), true)
        } else {
            (upload_f32(device, &linear.weight.data().data), false)
        }
    }

    // =========================================================================
    // KV cache sync (from CPU prefill)
    // =========================================================================

    impl MetalDecodeContext {
        /// Copy KV cache contents from CPU `Gemma3KvCache` into GPU buffers.
        /// Call once after CPU prefill, before starting Metal decode loop.
        pub fn sync_kv_from_cpu(&mut self, cache: &Gemma3KvCache) {
            let kv_dim = self.config.num_key_value_heads * self.config.head_dim;
            for (i, layer_cache) in cache.layers.iter().enumerate() {
                let lc = layer_cache.borrow();
                let seq_len = lc.seq_len;
                if seq_len == 0 {
                    continue;
                }
                let n_floats = seq_len * kv_dim;
                unsafe {
                    let k_dst = self.kv_k_bufs[i].contents().as_ptr() as *mut f32;
                    std::ptr::copy_nonoverlapping(lc.k.data.as_ptr(), k_dst, n_floats);
                    let v_dst = self.kv_v_bufs[i].contents().as_ptr() as *mut f32;
                    std::ptr::copy_nonoverlapping(lc.v.data.as_ptr(), v_dst, n_floats);
                }
            }
        }
    }

    // =========================================================================
    // Dispatch helpers — encode a single kernel dispatch into the encoder
    // =========================================================================

    impl MetalDecodeContext {
        /// Encode a RMSNorm dispatch.
        fn dispatch_rms_norm(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            gamma_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            dim: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_rms_norm);
            let d = dim as u32;
            let eps: f32 = self.config.rms_norm_eps;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&d as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&eps as *const f32 as *mut c_void).unwrap(),
                    4, 4,
                );
            }
            // Use 256 threads for hidden_size=2560 (each handles 10 elements)
            let tg = 256.min(dim);
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode a per-head RMSNorm dispatch.
        fn dispatch_rms_norm_per_head(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            gamma_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n_heads: usize,
            head_dim: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_rms_norm_per_head);
            let nh = n_heads as u32;
            let hd = head_dim as u32;
            let eps: f32 = self.config.rms_norm_eps;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&nh as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&hd as *const u32 as *mut c_void).unwrap(),
                    4, 4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&eps as *const f32 as *mut c_void).unwrap(),
                    4, 5,
                );
            }
            // One threadgroup per head, 32 threads each (one SIMD group)
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_heads, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }

        /// Encode a RoPE dispatch.
        fn dispatch_rope(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n_heads: usize,
            head_dim: usize,
            theta: f32,
            freq_scale: f32,
            position: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_rope_neox);
            let nh = n_heads as u32;
            let hd = head_dim as u32;
            let pos = position as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&nh as *const u32 as *mut c_void).unwrap(),
                    4, 2,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&hd as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&theta as *const f32 as *mut c_void).unwrap(),
                    4, 4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&freq_scale as *const f32 as *mut c_void).unwrap(),
                    4, 5,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&pos as *const u32 as *mut c_void).unwrap(),
                    4, 6,
                );
            }
            // One threadgroup per head, head_dim/2 threads each
            let half = head_dim / 2;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_heads, height: 1, depth: 1 },
                MTLSize { width: half, height: 1, depth: 1 },
            );
        }

        /// Encode a GELU_tanh dispatch.
        fn dispatch_gelu_tanh(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_gelu_tanh);
            let n_u32 = n as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 2,
                );
            }
            let tg = 256;
            let grids = (n + tg - 1) / tg;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grids, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode an element-wise multiply dispatch.
        fn dispatch_elem_mul(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            a_buf: &ProtocolObject<dyn MTLBuffer>,
            b_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_elem_mul);
            let n_u32 = n as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
            }
            let tg = 256;
            let grids = (n + tg - 1) / tg;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grids, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode a vector add dispatch.
        fn dispatch_vec_add(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            a_buf: &ProtocolObject<dyn MTLBuffer>,
            b_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_vec_add);
            let n_u32 = n as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(a_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(b_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
            }
            let tg = 256;
            let grids = (n + tg - 1) / tg;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grids, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode a KV cache append dispatch.
        fn dispatch_kv_append(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            k_new_buf: &ProtocolObject<dyn MTLBuffer>,
            v_new_buf: &ProtocolObject<dyn MTLBuffer>,
            k_cache_buf: &ProtocolObject<dyn MTLBuffer>,
            v_cache_buf: &ProtocolObject<dyn MTLBuffer>,
            seq_len: usize,
            kv_dim: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_kv_append);
            let sl = seq_len as u32;
            let kvd = kv_dim as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(k_new_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(v_new_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(k_cache_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(v_cache_buf), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&sl as *const u32 as *mut c_void).unwrap(),
                    4, 4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&kvd as *const u32 as *mut c_void).unwrap(),
                    4, 5,
                );
            }
            let tg = 256.min(kv_dim);
            let grids = (kv_dim + tg - 1) / tg;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grids, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode embedding lookup dispatch.
        fn dispatch_embed_lookup(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            token_id: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_embed_bf16);
            let tid = token_id as u32;
            let h = self.config.hidden_size as u32;
            let scale = (self.config.hidden_size as f32).sqrt();
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&self.embed_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&self.buf_hidden), 0, 1);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&tid as *const u32 as *mut c_void).unwrap(),
                    4, 2,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&h as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&scale as *const f32 as *mut c_void).unwrap(),
                    4, 4,
                );
            }
            let tg = 256;
            let grids = (self.config.hidden_size + tg - 1) / tg;
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: grids, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        /// Encode Q4K GEMV dispatch: out[1,N] = act[1,K] @ Q4K[N,K]^T
        fn dispatch_gemv_q4k(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            act_buf: &ProtocolObject<dyn MTLBuffer>,
            weight_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            k: usize,
            n: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_gemv_q4k);
            let k_u32 = k as u32;
            let n_u32 = n as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&k_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 4,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }

        /// Encode BF16 GEMV dispatch: out[1,N] = act[1,K] @ BF16[N,K]^T
        fn dispatch_gemv_bf16(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            act_buf: &ProtocolObject<dyn MTLBuffer>,
            weight_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            k: usize,
            n: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_gemv_bf16);
            let k_u32 = k as u32;
            let n_u32 = n as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(act_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&k_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 4,
                );
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }

        /// Encode attention decode dispatch.
        fn dispatch_attention(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            q_buf: &ProtocolObject<dyn MTLBuffer>,
            k_cache_buf: &ProtocolObject<dyn MTLBuffer>,
            v_cache_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            k_start: usize,
            k_end: usize,
        ) {
            encoder.setComputePipelineState(&self.pipe_attention_decode);
            let ks = k_start as u32;
            let ke = k_end as u32;
            let nq = self.config.num_attention_heads as u32;
            let nkv = self.config.num_key_value_heads as u32;
            let dh = self.config.head_dim as u32;
            let scale = 1.0f32 / (self.config.query_pre_attn_scalar as f32).sqrt();
            let kv_stride = (self.config.num_key_value_heads * self.config.head_dim) as u32;
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(q_buf), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(k_cache_buf), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(v_cache_buf), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(out_buf), 0, 3);
                encoder.setBytes_length_atIndex(
                    NonNull::new(&ks as *const u32 as *mut c_void).unwrap(),
                    4, 4,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&ke as *const u32 as *mut c_void).unwrap(),
                    4, 5,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&nq as *const u32 as *mut c_void).unwrap(),
                    4, 6,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&nkv as *const u32 as *mut c_void).unwrap(),
                    4, 7,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&dh as *const u32 as *mut c_void).unwrap(),
                    4, 8,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&scale as *const f32 as *mut c_void).unwrap(),
                    4, 9,
                );
                encoder.setBytes_length_atIndex(
                    NonNull::new(&kv_stride as *const u32 as *mut c_void).unwrap(),
                    4, 10,
                );
            }
            // One threadgroup per Q head, 32 threads each
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: self.config.num_attention_heads, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }
    }

    // =========================================================================
    // Graph encoding: full layer + full decode step
    // =========================================================================

    impl MetalDecodeContext {
        /// Dispatch Q4K or BF16 GEMV depending on the weight type.
        fn dispatch_gemv(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            act_buf: &ProtocolObject<dyn MTLBuffer>,
            weight_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            k: usize,
            n: usize,
            is_bf16: bool,
        ) {
            if is_bf16 {
                self.dispatch_gemv_bf16(encoder, act_buf, weight_buf, out_buf, k, n);
            } else {
                self.dispatch_gemv_q4k(encoder, act_buf, weight_buf, out_buf, k, n);
            }
        }

        /// Encode one transformer layer's forward pass into the encoder.
        fn encode_layer(
            &self,
            encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            layer_idx: usize,
            position: usize,
        ) {
            let lw = &self.layer_weights[layer_idx];
            let h = self.config.hidden_size;
            let nq = self.config.num_attention_heads;
            let nkv = self.config.num_key_value_heads;
            let d = self.config.head_dim;
            let inter = self.config.intermediate_size;
            let kv_dim = nkv * d;

            // 1. Input layernorm: hidden → normed
            self.dispatch_rms_norm(encoder, &self.buf_hidden, &lw.input_layernorm_gamma, &self.buf_normed, h);

            // 2. Q projection: normed → q
            self.dispatch_gemv(encoder, &self.buf_normed, &lw.q_proj, &self.buf_q, h, nq * d, lw.q_proj_is_bf16);

            // 3. K projection: normed → k
            self.dispatch_gemv(encoder, &self.buf_normed, &lw.k_proj, &self.buf_k, h, nkv * d, lw.k_proj_is_bf16);

            // 4. V projection: normed → v
            self.dispatch_gemv(encoder, &self.buf_normed, &lw.v_proj, &self.buf_v, h, nkv * d, lw.v_proj_is_bf16);

            // 5. Per-head RMSNorm on Q: q → q_normed
            self.dispatch_rms_norm_per_head(encoder, &self.buf_q, &lw.q_norm_gamma, &self.buf_q_normed, nq, d);

            // 6. Per-head RMSNorm on K: k → k_normed
            self.dispatch_rms_norm_per_head(encoder, &self.buf_k, &lw.k_norm_gamma, &self.buf_k_normed, nkv, d);

            // 7. RoPE on Q: q_normed → q_roped
            self.dispatch_rope(encoder, &self.buf_q_normed, &self.buf_q_roped, nq, d,
                             lw.rope_theta, lw.rope_freq_scale, position);

            // 8. RoPE on K: k_normed → k_roped
            self.dispatch_rope(encoder, &self.buf_k_normed, &self.buf_k_roped, nkv, d,
                             lw.rope_theta, lw.rope_freq_scale, position);

            // 9. KV cache append: write k_roped and v into cache at position
            self.dispatch_kv_append(encoder,
                &self.buf_k_roped, &self.buf_v,
                &self.kv_k_bufs[layer_idx], &self.kv_v_bufs[layer_idx],
                position, kv_dim);

            // 10. Attention: q_roped + kv_cache → attn_out
            let k_end = position + 1;
            let k_start = match lw.sliding_window {
                Some(w) => k_end.saturating_sub(w),
                None => 0,
            };
            self.dispatch_attention(encoder,
                &self.buf_q_roped,
                &self.kv_k_bufs[layer_idx], &self.kv_v_bufs[layer_idx],
                &self.buf_attn_out,
                k_start, k_end);

            // 11. O projection: attn_out → o_proj_out
            self.dispatch_gemv(encoder, &self.buf_attn_out, &lw.o_proj, &self.buf_o_proj_out, nq * d, h, lw.o_proj_is_bf16);

            // 12. Post-attention norm: o_proj_out → normed
            self.dispatch_rms_norm(encoder, &self.buf_o_proj_out, &lw.post_attn_layernorm_gamma, &self.buf_normed, h);

            // 13. Residual: hidden + normed → hidden2
            self.dispatch_vec_add(encoder, &self.buf_hidden, &self.buf_normed, &self.buf_hidden2, h);

            // 14. Pre-FFN norm: hidden2 → normed
            self.dispatch_rms_norm(encoder, &self.buf_hidden2, &lw.pre_ffn_layernorm_gamma, &self.buf_normed, h);

            // 15. Gate projection: normed → gate
            self.dispatch_gemv(encoder, &self.buf_normed, &lw.gate_proj, &self.buf_gate, h, inter, lw.gate_proj_is_bf16);

            // 16. Up projection: normed → up
            self.dispatch_gemv(encoder, &self.buf_normed, &lw.up_proj, &self.buf_up, h, inter, lw.up_proj_is_bf16);

            // 17. GELU_tanh: gate → gate_act
            self.dispatch_gelu_tanh(encoder, &self.buf_gate, &self.buf_gate_act, inter);

            // 18. Element-wise multiply: gate_act * up → mlp_hidden
            self.dispatch_elem_mul(encoder, &self.buf_gate_act, &self.buf_up, &self.buf_mlp_hidden, inter);

            // 19. Down projection: mlp_hidden → down_out
            if lw.down_proj_is_bf16 {
                self.dispatch_gemv_bf16(encoder, &self.buf_mlp_hidden, &lw.down_proj, &self.buf_down_out, inter, h);
            } else {
                self.dispatch_gemv_q4k(encoder, &self.buf_mlp_hidden, &lw.down_proj, &self.buf_down_out, inter, h);
            }

            // 20. Post-FFN norm: down_out → normed
            self.dispatch_rms_norm(encoder, &self.buf_down_out, &lw.post_ffn_layernorm_gamma, &self.buf_normed, h);

            // 21. Residual: hidden2 + normed → hidden (ready for next layer)
            self.dispatch_vec_add(encoder, &self.buf_hidden2, &self.buf_normed, &self.buf_hidden, h);
        }

        /// Execute one full decode step: embed → 34 layers → final norm → lm_head.
        /// Returns logits as Vec<f32>.
        pub fn decode_step(&mut self, token_id: usize, position: usize) -> Vec<f32> {
            let cmd_buf = self.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            // 1. Embedding lookup → buf_hidden
            self.dispatch_embed_lookup(&encoder, token_id);

            // 2. All layers
            for layer_idx in 0..self.config.num_hidden_layers {
                self.encode_layer(&encoder, layer_idx, position);
            }

            // 3. Final norm: hidden → normed
            self.dispatch_rms_norm(
                &encoder,
                &self.buf_hidden,
                &self.final_norm_gamma,
                &self.buf_normed,
                self.config.hidden_size,
            );

            // 4. lm_head: normed → logits (BF16, weight-tied to embed in Gemma3)
            self.dispatch_gemv_bf16(
                &encoder,
                &self.buf_normed,
                &self.lm_head_buf,
                &self.buf_logits,
                self.config.hidden_size,
                self.config.vocab_size,
            );

            // End encoding, commit, wait
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            // Read logits from shared memory
            let vocab = self.config.vocab_size;
            let ptr = self.buf_logits.contents().as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, vocab).to_vec() }
        }
    }

    // =========================================================================
    // Tests
    // =========================================================================
    #[cfg(test)]
    mod tests {
        use super::*;
        use std::collections::HashMap;

        /// Create a Metal device + compile all pipelines (tests that MSL compiles).
        fn test_context() -> (
            Retained<ProtocolObject<dyn MTLDevice>>,
            Retained<ProtocolObject<dyn MTLCommandQueue>>,
            HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        ) {
            let device = MTLCreateSystemDefaultDevice().expect("no Metal device");
            let queue = device.newCommandQueue().expect("no queue");

            let kernel_names = [
                "rms_norm_gemma3",
                "rms_norm_per_head",
                "rope_neox",
                "gelu_tanh_kernel",
                "elem_mul_kernel",
                "vec_add_kernel",
                "kv_cache_append",
                "embed_bf16_lookup",
                "attention_decode",
                "gemv_q4k_t",
                "gemv_bf16_t",
            ];

            let mut pipelines = HashMap::new();
            for name in &kernel_names {
                let pipe = compile_pipeline(&device, DECODE_MSL, name);
                pipelines.insert(name.to_string(), pipe);
            }

            (device, queue, pipelines)
        }

        fn run_kernel_1d(
            _device: &ProtocolObject<dyn MTLDevice>,
            queue: &ProtocolObject<dyn MTLCommandQueue>,
            pipe: &ProtocolObject<dyn MTLComputePipelineState>,
            buffers: Vec<&ProtocolObject<dyn MTLBuffer>>,
            byte_args: Vec<(Vec<u8>, usize)>, // (bytes, index)
            grid: MTLSize,
            tg: MTLSize,
        ) {
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pipe);
            for (i, buf) in buffers.iter().enumerate() {
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(*buf), 0, i);
                }
            }
            for (bytes, idx) in &byte_args {
                unsafe {
                    enc.setBytes_length_atIndex(
                        NonNull::new(bytes.as_ptr() as *mut c_void).unwrap(),
                        bytes.len(),
                        *idx,
                    );
                }
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();
        }

        fn read_f32(buf: &ProtocolObject<dyn MTLBuffer>, n: usize) -> Vec<f32> {
            let ptr = buf.contents().as_ptr() as *const f32;
            unsafe { std::slice::from_raw_parts(ptr, n).to_vec() }
        }

        // ── RMSNorm test ─────────────────────────────────────────────────────

        #[test]
        fn test_rms_norm_gemma3() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["rms_norm_gemma3"];

            let d = 8usize;
            let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let gamma: Vec<f32> = vec![0.1, 0.2, 0.0, -0.1, 0.05, 0.0, 0.0, 0.15];
            let eps: f32 = 1e-6;

            // CPU reference
            let sq_sum: f32 = x.iter().map(|v| v * v).sum();
            let rms = (sq_sum / d as f32 + eps).sqrt();
            let expected: Vec<f32> = x.iter().zip(gamma.iter())
                .map(|(xi, gi)| xi / rms * (1.0 + gi))
                .collect();

            let buf_x = upload_f32(&device, &x);
            let buf_g = upload_f32(&device, &gamma);
            let buf_out = alloc_buf(&device, d * 4);

            let d_u32 = d as u32;
            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_x, &buf_g, &buf_out],
                vec![
                    (d_u32.to_ne_bytes().to_vec(), 3),
                    (eps.to_ne_bytes().to_vec(), 4),
                ],
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: d, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, d);
            for i in 0..d {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "rms_norm_gemma3 mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── RoPE test ────────────────────────────────────────────────────────

        #[test]
        fn test_rope_neox() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["rope_neox"];

            let n_heads = 2usize;
            let head_dim = 4usize;
            let half = head_dim / 2;
            let theta: f32 = 10000.0;
            let freq_scale: f32 = 1.0;
            let position = 3usize;

            // Input: [n_heads * head_dim] = 8 elements
            let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

            // CPU reference
            let mut expected = x.clone();
            for h in 0..n_heads {
                for i in 0..half {
                    let angle = (position as f32 * freq_scale) / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let c0 = h * head_dim + i;
                    let c1 = h * head_dim + i + half;
                    expected[c0] = x[c0] * cos_a - x[c1] * sin_a;
                    expected[c1] = x[c0] * sin_a + x[c1] * cos_a;
                }
            }

            let buf_x = upload_f32(&device, &x);
            let buf_out = alloc_buf(&device, x.len() * 4);

            let nh = n_heads as u32;
            let hd = head_dim as u32;
            let pos = position as u32;
            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_x, &buf_out],
                vec![
                    (nh.to_ne_bytes().to_vec(), 2),
                    (hd.to_ne_bytes().to_vec(), 3),
                    (theta.to_ne_bytes().to_vec(), 4),
                    (freq_scale.to_ne_bytes().to_vec(), 5),
                    (pos.to_ne_bytes().to_vec(), 6),
                ],
                MTLSize { width: n_heads, height: 1, depth: 1 },
                MTLSize { width: half, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, x.len());
            for i in 0..x.len() {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "rope_neox mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── GELU test ────────────────────────────────────────────────────────

        #[test]
        fn test_gelu_tanh() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["gelu_tanh_kernel"];

            // Include large values (>10) that trigger fast-math tanh overflow without clamping
            let x: Vec<f32> = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0,
                                   10.0, 12.655, -15.0, 20.0, -25.0, 100.0];
            let n = x.len();

            // CPU reference
            let expected: Vec<f32> = x.iter().map(|&v| {
                let v3 = v * v * v;
                let inner = 0.7978845608_f32 * (v + 0.044715 * v3);
                0.5 * v * (1.0 + inner.tanh())
            }).collect();

            let buf_x = upload_f32(&device, &x);
            let buf_out = alloc_buf(&device, n * 4);
            let n_u32 = n as u32;

            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_x, &buf_out],
                vec![(n_u32.to_ne_bytes().to_vec(), 2)],
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, n);
            for i in 0..n {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "gelu_tanh mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── Vec add test ─────────────────────────────────────────────────────

        #[test]
        fn test_vec_add() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["vec_add_kernel"];

            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
            let n = a.len();
            let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a + b).collect();

            let buf_a = upload_f32(&device, &a);
            let buf_b = upload_f32(&device, &b);
            let buf_out = alloc_buf(&device, n * 4);
            let n_u32 = n as u32;

            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_a, &buf_b, &buf_out],
                vec![(n_u32.to_ne_bytes().to_vec(), 3)],
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, n);
            for i in 0..n {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "vec_add mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── Elem mul test ────────────────────────────────────────────────────

        #[test]
        fn test_elem_mul() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["elem_mul_kernel"];

            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b: Vec<f32> = vec![0.5, 1.5, 2.5, 3.5];
            let n = a.len();
            let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(a, b)| a * b).collect();

            let buf_a = upload_f32(&device, &a);
            let buf_b = upload_f32(&device, &b);
            let buf_out = alloc_buf(&device, n * 4);
            let n_u32 = n as u32;

            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_a, &buf_b, &buf_out],
                vec![(n_u32.to_ne_bytes().to_vec(), 3)],
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, n);
            for i in 0..n {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-5,
                    "elem_mul mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── Per-head RMSNorm test ────────────────────────────────────────────

        #[test]
        fn test_rms_norm_per_head() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["rms_norm_per_head"];

            let n_heads = 2usize;
            let head_dim = 4usize;
            let eps: f32 = 1e-6;

            let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let gamma: Vec<f32> = vec![0.1, 0.2, 0.0, -0.1]; // shared across heads

            // CPU reference
            let mut expected = vec![0.0f32; n_heads * head_dim];
            for h in 0..n_heads {
                let base = h * head_dim;
                let sq_sum: f32 = (0..head_dim).map(|i| x[base + i] * x[base + i]).sum();
                let rms = (sq_sum / head_dim as f32 + eps).sqrt();
                for i in 0..head_dim {
                    expected[base + i] = x[base + i] / rms * (1.0 + gamma[i]);
                }
            }

            let buf_x = upload_f32(&device, &x);
            let buf_g = upload_f32(&device, &gamma);
            let buf_out = alloc_buf(&device, x.len() * 4);
            let nh = n_heads as u32;
            let hd = head_dim as u32;

            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_x, &buf_g, &buf_out],
                vec![
                    (nh.to_ne_bytes().to_vec(), 3),
                    (hd.to_ne_bytes().to_vec(), 4),
                    (eps.to_ne_bytes().to_vec(), 5),
                ],
                MTLSize { width: n_heads, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );

            let result = read_f32(&buf_out, x.len());
            for i in 0..x.len() {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-4,
                    "rms_norm_per_head mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── KV cache append test ─────────────────────────────────────────────

        #[test]
        fn test_kv_cache_append() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["kv_cache_append"];

            let kv_dim = 4usize;
            let max_seq = 8usize;
            let seq_len = 3usize; // write at row 3

            let k_new: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let v_new: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0];

            let buf_k_new = upload_f32(&device, &k_new);
            let buf_v_new = upload_f32(&device, &v_new);
            let buf_k_cache = alloc_buf(&device, max_seq * kv_dim * 4);
            let buf_v_cache = alloc_buf(&device, max_seq * kv_dim * 4);

            // Zero the cache
            unsafe {
                std::ptr::write_bytes(buf_k_cache.contents().as_ptr() as *mut u8, 0, max_seq * kv_dim * 4);
                std::ptr::write_bytes(buf_v_cache.contents().as_ptr() as *mut u8, 0, max_seq * kv_dim * 4);
            }

            let sl = seq_len as u32;
            let kvd = kv_dim as u32;

            run_kernel_1d(
                &device, &queue, pipe,
                vec![&buf_k_new, &buf_v_new, &buf_k_cache, &buf_v_cache],
                vec![
                    (sl.to_ne_bytes().to_vec(), 4),
                    (kvd.to_ne_bytes().to_vec(), 5),
                ],
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: kv_dim, height: 1, depth: 1 },
            );

            let k_cache = read_f32(&buf_k_cache, max_seq * kv_dim);
            let v_cache = read_f32(&buf_v_cache, max_seq * kv_dim);

            // Check that row 3 has the expected values
            for i in 0..kv_dim {
                assert_eq!(k_cache[seq_len * kv_dim + i], k_new[i], "k_cache mismatch at {}", i);
                assert_eq!(v_cache[seq_len * kv_dim + i], v_new[i], "v_cache mismatch at {}", i);
            }
            // Check that other rows are still zero
            for row in 0..max_seq {
                if row == seq_len { continue; }
                for i in 0..kv_dim {
                    assert_eq!(k_cache[row * kv_dim + i], 0.0, "k_cache non-zero at row {} col {}", row, i);
                }
            }
        }

        // ── Attention decode test ────────────────────────────────────────────

        #[test]
        fn test_attention_decode() {
            let (device, queue, pipes) = test_context();
            let pipe = &pipes["attention_decode"];

            // Simple setup: 2 Q heads, 1 KV head, head_dim=4, 3 context tokens
            let n_q_heads = 2usize;
            let n_kv_heads = 1usize;
            let d_head = 4usize;
            let kv_stride = n_kv_heads * d_head;
            let scale = 1.0f32 / (d_head as f32).sqrt();
            let k_start = 0usize;
            let k_end = 3usize;
            let t_kv = k_end - k_start;

            // Q: [n_q_heads * d_head] = 8 values
            let q: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0];
            // K cache: [3, 4]
            let k_cache: Vec<f32> = vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
            ];
            // V cache: [3, 4]
            let v_cache: Vec<f32> = vec![
                10.0, 20.0, 30.0, 40.0,
                50.0, 60.0, 70.0, 80.0,
                90.0, 100.0, 110.0, 120.0,
            ];

            // CPU reference
            let group = n_q_heads / n_kv_heads;
            let mut expected = vec![0.0f32; n_q_heads * d_head];
            for qh in 0..n_q_heads {
                let kvh = qh / group;
                let q_off = qh * d_head;
                let kv_off = kvh * d_head;

                // Compute scores
                let mut scores = vec![0.0f32; t_kv];
                for ci in 0..t_kv {
                    let k_base = (k_start + ci) * kv_stride + kv_off;
                    let mut dot = 0.0f32;
                    for di in 0..d_head {
                        dot += q[q_off + di] * k_cache[k_base + di];
                    }
                    scores[ci] = dot * scale;
                }

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                for e in &mut exps { *e /= sum_exp; }

                // Weighted V sum
                for ci in 0..t_kv {
                    let v_base = (k_start + ci) * kv_stride + kv_off;
                    for di in 0..d_head {
                        expected[q_off + di] += exps[ci] * v_cache[v_base + di];
                    }
                }
            }

            let buf_q = upload_f32(&device, &q);
            let buf_k = upload_f32(&device, &k_cache);
            let buf_v = upload_f32(&device, &v_cache);
            let buf_out = alloc_buf(&device, n_q_heads * d_head * 4);

            let ks = k_start as u32;
            let ke = k_end as u32;
            let nq = n_q_heads as u32;
            let nkv = n_kv_heads as u32;
            let dh = d_head as u32;
            let kvs = kv_stride as u32;

            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();
            enc.setComputePipelineState(pipe);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_q), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_k), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_v), 0, 2);
                enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 3);
                enc.setBytes_length_atIndex(NonNull::new(&ks as *const u32 as *mut c_void).unwrap(), 4, 4);
                enc.setBytes_length_atIndex(NonNull::new(&ke as *const u32 as *mut c_void).unwrap(), 4, 5);
                enc.setBytes_length_atIndex(NonNull::new(&nq as *const u32 as *mut c_void).unwrap(), 4, 6);
                enc.setBytes_length_atIndex(NonNull::new(&nkv as *const u32 as *mut c_void).unwrap(), 4, 7);
                enc.setBytes_length_atIndex(NonNull::new(&dh as *const u32 as *mut c_void).unwrap(), 4, 8);
                enc.setBytes_length_atIndex(NonNull::new(&scale as *const f32 as *mut c_void).unwrap(), 4, 9);
                enc.setBytes_length_atIndex(NonNull::new(&kvs as *const u32 as *mut c_void).unwrap(), 4, 10);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n_q_heads, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let result = read_f32(&buf_out, n_q_heads * d_head);
            for i in 0..result.len() {
                assert!(
                    (result[i] - expected[i]).abs() < 1e-3,
                    "attention_decode mismatch at {}: got {} expected {}",
                    i, result[i], expected[i]
                );
            }
        }

        // ── Multi-dispatch single-encoder test ───────────────────────────────
        // Verify that sequential dispatches within a single encoder see each
        // other's writes (the core assumption of our graph approach).

        #[test]
        fn test_multi_dispatch_visibility() {
            let (device, queue, pipes) = test_context();

            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
            let n = a.len();

            let buf_a = upload_f32(&device, &a);
            let buf_b = upload_f32(&device, &b);
            let buf_mid = alloc_buf(&device, n * 4);  // a + b
            let buf_out = alloc_buf(&device, n * 4);   // (a + b) * b

            let n_u32 = n as u32;

            // Single command buffer, single encoder, TWO dispatches
            let cmd = queue.commandBuffer().unwrap();
            let enc = cmd.computeCommandEncoder().unwrap();

            // Dispatch 1: mid = a + b
            enc.setComputePipelineState(&pipes["vec_add_kernel"]);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_mid), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n, height: 1, depth: 1 },
            );

            // Dispatch 2: out = mid * b (reads from mid written by dispatch 1)
            enc.setComputePipelineState(&pipes["elem_mul_kernel"]);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(&buf_mid), 0, 0);
                enc.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(&buf_out), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(),
                    4, 3,
                );
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: n, height: 1, depth: 1 },
            );

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            let result = read_f32(&buf_out, n);
            for i in 0..n {
                let expected = (a[i] + b[i]) * b[i];
                assert!(
                    (result[i] - expected).abs() < 1e-5,
                    "multi-dispatch visibility failed at {}: got {} expected {}",
                    i, result[i], expected
                );
            }
        }
    }
}
