/// # Metal full-graph decode engine for Qwen 3.5 (hybrid DeltaNet + softmax)
///
/// Encodes the entire Qwen3.5 decode forward pass into a single Metal command
/// buffer.  Handles two layer types: DeltaNet (recurrent) and FullAttention
/// (softmax with GQA).
///
/// DeltaNet layers have persistent recurrent state (S matrix + conv history)
/// stored in GPU buffers.  FullAttention layers have KV cache in half precision.

#[cfg(feature = "metal")]
pub mod inner {
    use std::ptr::NonNull;
    use std::ffi::c_void;

    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::*;

    use crate::transformer_qwen35::{
        ConfigQwen35, Qwen35Model, Qwen35Cache, LayerCache, TokenMixer,
    };

    // =========================================================================
    // MSL kernel source
    // =========================================================================

    const DECODE_QWEN35_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ── RMSNorm (Gemma3 variant: multiply by (1 + gamma)) ────────────────────
kernel void rms_norm_gemma3(
    device const float* x      [[ buffer(0) ]],
    device const float* gamma  [[ buffer(1) ]],
    device       float* out    [[ buffer(2) ]],
    constant     uint&  D      [[ buffer(3) ]],
    constant     float& eps    [[ buffer(4) ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    // Sum of squares with SIMD + threadgroup reduction
    float sum_sq = 0;
    for (uint i = tid; i < D; i += tg_size) {
        float v = x[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);

    threadgroup float tg_buf[8];
    uint lane = tid % 32;
    uint grp  = tid / 32;
    if (lane == 0) tg_buf[grp] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        uint n_grps = (tg_size + 31) / 32;
        for (uint i = 0; i < n_grps; i++) total += tg_buf[i];
        tg_buf[0] = rsqrt(total / float(D) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = tg_buf[0];

    for (uint i = tid; i < D; i += tg_size)
        out[i] = x[i] * rms_inv * (1.0 + gamma[i]);
}

// ── Per-head RMSNorm (1 + gamma) ─────────────────────────────────────────
kernel void rms_norm_per_head(
    device const float* x      [[ buffer(0) ]],
    device const float* gamma  [[ buffer(1) ]],
    device       float* out    [[ buffer(2) ]],
    constant     uint&  n_heads [[ buffer(3) ]],
    constant     uint&  head_dim [[ buffer(4) ]],
    constant     float& eps    [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    uint h = tgid;
    if (h >= n_heads) return;
    uint base = h * head_dim;

    float sum_sq = 0;
    for (uint i = tid; i < head_dim; i += tg_size)
        sum_sq += x[base + i] * x[base + i];
    sum_sq = simd_sum(sum_sq);

    threadgroup float tg_red[4];
    uint lane = tid % 32;
    uint grp  = tid / 32;
    if (lane == 0) tg_red[grp] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (uint i = 0; i < (tg_size + 31)/32; i++) t += tg_red[i];
        tg_red[0] = rsqrt(t / float(head_dim) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = tg_red[0];

    for (uint i = tid; i < head_dim; i += tg_size)
        out[base + i] = x[base + i] * rms_inv * (1.0 + gamma[i]);
}

// ── Embedding lookup (BF16, no scaling) ───────────────────────────────────
kernel void embed_bf16_lookup(
    device const ushort* embed_table [[ buffer(0) ]],
    device       float*  out         [[ buffer(1) ]],
    constant     uint&   token_id    [[ buffer(2) ]],
    constant     uint&   hidden_size [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= hidden_size) return;
    ushort bits = embed_table[token_id * hidden_size + gid];
    out[gid] = as_type<float>(uint(bits) << 16u);
}

// ── Vector operations ─────────────────────────────────────────────────────
kernel void vec_add_kernel(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    out[gid] = a[gid] + b[gid];
}

kernel void elem_mul_kernel(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    out[gid] = a[gid] * b[gid];
}

kernel void silu_kernel(
    device const float* x   [[ buffer(0) ]],
    device       float* out [[ buffer(1) ]],
    constant     uint&  N   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= N) return;
    float v = x[gid];
    out[gid] = v / (1.0 + exp(-v));
}

kernel void scale_inplace(
    device float* x        [[ buffer(0) ]],
    constant float& scale  [[ buffer(1) ]],
    constant uint&  N      [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid < N) x[gid] *= scale;
}

// ── Conv1d depthwise + SiLU ──────────────────────────────────────────────
// One thread per channel.  kernel_size = 4, hist = 3.
kernel void conv1d_silu(
    device       float* data        [[ buffer(0) ]],   // [qkv_dim] input → output (in-place)
    device       float* conv_state  [[ buffer(1) ]],   // [qkv_dim * hist]
    device const float* conv_weight [[ buffer(2) ]],   // [qkv_dim * ks]
    constant     uint&  qkv_dim     [[ buffer(3) ]],
    constant     uint&  kernel_size [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= qkv_dim) return;
    uint c = gid;
    uint hist = kernel_size - 1;
    uint w_base = c * kernel_size;
    uint s_base = c * hist;

    float input_c = data[c];

    float val = 0;
    for (uint t = 0; t < hist; t++)
        val += conv_weight[w_base + t] * conv_state[s_base + t];
    val += conv_weight[w_base + hist] * input_c;

    // SiLU
    data[c] = val / (1.0 + exp(-val));

    // Shift history
    for (uint t = 0; t < hist - 1; t++)
        conv_state[s_base + t] = conv_state[s_base + t + 1];
    conv_state[s_base + hist - 1] = input_c;
}

// ── L2 normalize per head (in-place) ─────────────────────────────────────
kernel void l2_normalize_heads(
    device float* x          [[ buffer(0) ]],
    constant uint& n_heads   [[ buffer(1) ]],
    constant uint& dim       [[ buffer(2) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    uint h = tgid;
    if (h >= n_heads) return;
    uint base = h * dim;

    float sum_sq = 0;
    for (uint i = tid; i < dim; i += tg_size)
        sum_sq += x[base + i] * x[base + i];
    sum_sq = simd_sum(sum_sq);

    threadgroup float tg_buf[4];
    uint lane = tid % 32;
    uint grp  = tid / 32;
    if (lane == 0) tg_buf[grp] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (uint i = 0; i < (tg_size + 31)/32; i++) t += tg_buf[i];
        tg_buf[0] = max(sqrt(t), 1e-12f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float norm = tg_buf[0];

    for (uint i = tid; i < dim; i += tg_size)
        x[base + i] /= norm;
}

// ── DeltaNet gate computation ─────────────────────────────────────────────
// decay[h] = exp(-exp(a_log[h]) * softplus(a[h] + dt_bias[h]))
// beta[h]  = sigmoid(b[h])
kernel void deltanet_compute_gates(
    device const float* a_proj  [[ buffer(0) ]],
    device const float* b_proj  [[ buffer(1) ]],
    device const float* a_log   [[ buffer(2) ]],
    device const float* dt_bias [[ buffer(3) ]],
    device       float* decay   [[ buffer(4) ]],
    device       float* beta    [[ buffer(5) ]],
    constant     uint&  nv      [[ buffer(6) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= nv) return;
    float a = a_proj[gid];
    float b = b_proj[gid];
    float al = a_log[gid];
    float dt = dt_bias[gid];

    float sp_in = a + dt;
    float sp = sp_in > 20.0f ? sp_in : log(1.0f + exp(sp_in));
    float g = -exp(al) * sp;
    decay[gid] = exp(g);
    beta[gid] = 1.0f / (1.0f + exp(-b));
}

// ── DeltaNet recurrent state update ───────────────────────────────────────
// One threadgroup per value head, vd threads per threadgroup.
// Thread j handles column j of state S[kd, vd].
// v_per_k for repeat-interleave of Q/K heads.
kernel void deltanet_recurrent(
    device       float* state    [[ buffer(0) ]],   // [nv * kd * vd]
    device const float* q        [[ buffer(1) ]],   // [nk * kd] (L2-normed, scaled)
    device const float* k        [[ buffer(2) ]],   // [nk * kd] (L2-normed)
    device const float* v        [[ buffer(3) ]],   // [nv * vd]
    device const float* decay_b  [[ buffer(4) ]],   // [nv]
    device const float* beta_b   [[ buffer(5) ]],   // [nv]
    device       float* output   [[ buffer(6) ]],   // [nv * vd]
    constant     uint&  kd       [[ buffer(7) ]],
    constant     uint&  vd       [[ buffer(8) ]],
    constant     uint&  v_per_k  [[ buffer(9) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]])
{
    uint h = tgid;   // value head index
    uint j = tid;       // column index
    if (j >= vd) return;

    uint qk_group = h / v_per_k;
    uint s_base = h * kd * vd;
    float decay_h = decay_b[h];
    float beta_h  = beta_b[h];
    float v_j     = v[h * vd + j];

    // Load Q and K for this group into threadgroup memory
    threadgroup float tg_k[256];   // max kd
    threadgroup float tg_q[256];
    for (uint i = tid; i < kd; i += vd) {
        tg_k[i] = k[qk_group * kd + i];
        tg_q[i] = q[qk_group * kd + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 1: Decay state + kv_mem[j]
    float kv_mem_j = 0;
    for (uint i = 0; i < kd; i++) {
        float s_ij = state[s_base + i * vd + j] * decay_h;
        state[s_base + i * vd + j] = s_ij;
        kv_mem_j += s_ij * tg_k[i];
    }

    // Step 2: State update — S += outer(k, (v - kv_mem) * beta)
    float delta_j = (v_j - kv_mem_j) * beta_h;
    for (uint i = 0; i < kd; i++)
        state[s_base + i * vd + j] += tg_k[i] * delta_j;

    // Step 3: Query — output = S^T @ q
    float out_j = 0;
    for (uint i = 0; i < kd; i++)
        out_j += state[s_base + i * vd + j] * tg_q[i];

    output[h * vd + j] = out_j;
}

// ── Gated RMSNorm per head ────────────────────────────────────────────────
// out[h*vd+j] = (input[h*vd+j] / rms_h) * weight[j] * silu(z[h*vd+j])
kernel void gated_rms_norm(
    device const float* input  [[ buffer(0) ]],
    device const float* z      [[ buffer(1) ]],
    device const float* weight [[ buffer(2) ]],
    device       float* output [[ buffer(3) ]],
    constant     uint&  nv     [[ buffer(4) ]],
    constant     uint&  vd     [[ buffer(5) ]],
    constant     float& eps    [[ buffer(6) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]],
    uint tg_size [[ threads_per_threadgroup ]])
{
    uint h = tgid;
    if (h >= nv) return;
    uint base = h * vd;

    float sum_sq = 0;
    for (uint j = tid; j < vd; j += tg_size)
        sum_sq += input[base + j] * input[base + j];
    sum_sq = simd_sum(sum_sq);

    threadgroup float tg_buf[4];
    uint lane = tid % 32;
    uint grp  = tid / 32;
    if (lane == 0) tg_buf[grp] = sum_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0;
        for (uint i = 0; i < (tg_size + 31)/32; i++) t += tg_buf[i];
        tg_buf[0] = rsqrt(t / float(vd) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float rms_inv = tg_buf[0];

    for (uint j = tid; j < vd; j += tg_size) {
        float normed = input[base + j] * rms_inv;
        float z_val  = z[base + j];
        float silu_z = z_val / (1.0 + exp(-z_val));
        output[base + j] = normed * weight[j] * silu_z;
    }
}

// ── Split Q and gate from interleaved q_proj output ───────────────────────
// q_proj: [nq * d * 2] with layout [Q₀, gate₀, Q₁, gate₁, ...]
kernel void split_q_gate(
    device const float* qg   [[ buffer(0) ]],
    device       float* q    [[ buffer(1) ]],
    device       float* gate [[ buffer(2) ]],
    constant     uint&  nq   [[ buffer(3) ]],
    constant     uint&  d    [[ buffer(4) ]],
    uint gid [[ thread_position_in_grid ]])
{
    uint total = nq * d;
    if (gid >= total) return;
    uint h = gid / d;
    uint j = gid % d;
    q[gid]    = qg[h * d * 2 + j];
    gate[gid] = qg[h * d * 2 + d + j];
}

// ── Sigmoid gate: out = a * sigmoid(b) ────────────────────────────────────
kernel void sigmoid_gate(
    device const float* a   [[ buffer(0) ]],
    device const float* b   [[ buffer(1) ]],
    device       float* out [[ buffer(2) ]],
    constant     uint&  N   [[ buffer(3) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= N) return;
    out[gid] = a[gid] / (1.0 + exp(-b[gid]));
}

// ── Partial RoPE (NeoX half-split, only first rope_dim dims) ──────────────
kernel void rope_neox_partial(
    device float* x           [[ buffer(0) ]],
    constant uint& n_heads    [[ buffer(1) ]],
    constant uint& head_dim   [[ buffer(2) ]],
    constant uint& rope_dim   [[ buffer(3) ]],
    constant float& theta     [[ buffer(4) ]],
    constant uint& position   [[ buffer(5) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]])
{
    uint h = tgid;
    if (h >= n_heads) return;
    uint half_rope = rope_dim / 2;
    if (tid >= half_rope) return;

    uint i = tid;
    float freq = 1.0 / pow(theta, 2.0 * float(i) / float(rope_dim));
    float angle = float(position) * freq;
    float cos_a = cos(angle);
    float sin_a = sin(angle);

    uint base = h * head_dim;
    uint idx0 = base + i;
    uint idx1 = base + i + half_rope;
    float x0 = x[idx0];
    float x1 = x[idx1];
    x[idx0] = x0 * cos_a - x1 * sin_a;
    x[idx1] = x1 * cos_a + x0 * sin_a;
}

// ── GQA attention decode ──────────────────────────────────────────────────
// One threadgroup per Q head, 32 threads.
#define TG_ATTN 32
kernel void attention_decode(
    device const float* q_buf    [[ buffer(0) ]],
    device const half*  k_cache  [[ buffer(1) ]],
    device const half*  v_cache  [[ buffer(2) ]],
    device       float* out_buf  [[ buffer(3) ]],
    constant     uint&  nq       [[ buffer(4) ]],
    constant     uint&  nkv      [[ buffer(5) ]],
    constant     uint&  head_dim [[ buffer(6) ]],
    constant     float& scale    [[ buffer(7) ]],
    constant     uint&  k_end    [[ buffer(8) ]],
    constant     uint&  sliding_window [[ buffer(9) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]])
{
    uint qh = tgid;
    if (qh >= nq) return;
    uint kvh = qh / (nq / nkv);
    uint kv_dim = nkv * head_dim;
    uint k_start = (sliding_window > 0 && k_end > sliding_window) ? k_end - sliding_window : 0;
    uint seq_len = k_end - k_start;

    threadgroup float tg_scores[4096];

    // Phase 1: scores
    for (uint c = tid; c < seq_len; c += TG_ATTN) {
        float dot = 0;
        uint k_pos = k_start + c;
        for (uint j = 0; j < head_dim; j++)
            dot += q_buf[qh * head_dim + j] * float(k_cache[k_pos * kv_dim + kvh * head_dim + j]);
        tg_scores[c] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: softmax
    if (tid == 0) {
        float max_s = -1e30;
        for (uint c = 0; c < seq_len; c++)
            max_s = max(max_s, tg_scores[c]);
        float sum_exp = 0;
        for (uint c = 0; c < seq_len; c++) {
            tg_scores[c] = exp(tg_scores[c] - max_s);
            sum_exp += tg_scores[c];
        }
        float inv = 1.0 / sum_exp;
        for (uint c = 0; c < seq_len; c++)
            tg_scores[c] *= inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: weighted sum
    for (uint j = tid; j < head_dim; j += TG_ATTN) {
        float acc = 0;
        for (uint c = 0; c < seq_len; c++)
            acc += tg_scores[c] * float(v_cache[(k_start + c) * kv_dim + kvh * head_dim + j]);
        out_buf[qh * head_dim + j] = acc;
    }
}

// ── KV cache append (f32 → half) ─────────────────────────────────────────
kernel void kv_cache_append(
    device const float* k_new   [[ buffer(0) ]],
    device const float* v_new   [[ buffer(1) ]],
    device       half*  k_cache [[ buffer(2) ]],
    device       half*  v_cache [[ buffer(3) ]],
    constant     uint&  seq_len [[ buffer(4) ]],
    constant     uint&  kv_dim  [[ buffer(5) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid >= kv_dim) return;
    k_cache[seq_len * kv_dim + gid] = half(k_new[gid]);
    v_cache[seq_len * kv_dim + gid] = half(v_new[gid]);
}

// ── f32 to f16 bulk conversion ────────────────────────────────────────────
kernel void f32_to_f16_convert(
    device const float* src [[ buffer(0) ]],
    device       half*  dst [[ buffer(1) ]],
    constant     uint&  N   [[ buffer(2) ]],
    uint gid [[ thread_position_in_grid ]])
{
    if (gid < N) dst[gid] = half(src[gid]);
}

// ── GEMV BF16 transpose: C[N] = A[K] @ BF16[N,K]^T ──────────────────────
// One threadgroup per output row j, 32 threads.
kernel void gemv_bf16_t(
    device const float*  act     [[ buffer(0) ]],
    device const ushort* weight  [[ buffer(1) ]],
    device       float*  out     [[ buffer(2) ]],
    constant     uint&   K       [[ buffer(3) ]],
    constant     uint&   N       [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]])
{
    uint j = tgid;
    if (j >= N) return;
    float acc = 0;
    for (uint i = tid; i < K; i += 32) {
        ushort bits = weight[j * K + i];
        float w = as_type<float>(uint(bits) << 16u);
        acc += act[i] * w;
    }
    acc = simd_sum(acc);
    if (tid == 0) out[j] = acc;
}

// ── GEMV Q4K transpose ───────────────────────────────────────────────────
// One threadgroup per output row j, 32 threads.
// Q4K block: 256 elements, 144 bytes.
inline float2 scale_min(device const uchar* sc, uint chunk) {
    // Decode Q4K scale+min from the 12 bytes of scales data
    float d = as_type<half>(*(device const ushort*)(sc + 128));
    float dmin = as_type<half>(*(device const ushort*)(sc + 130));
    uchar sc_byte, m_byte;
    if (chunk < 4) {
        sc_byte = sc[chunk];
        m_byte  = sc[chunk + 4];
    } else {
        uint off = chunk - 4;
        sc_byte = ((sc[off + 8] & 0xF0) >> 4) | ((sc[off] >> 6) << 4);
        m_byte  = ((sc[off + 8] & 0x0F) << 2) | ((sc[off + 4] >> 6) << 4);
        // Correction: m_byte higher bits
    }
    // Simplified Q4K decoding
    float scale, minimum;
    if (chunk < 4) {
        scale   = d * float(sc[chunk] & 63);
        minimum = dmin * float(sc[chunk + 4] & 63);
    } else {
        uint off = chunk - 4;
        uchar s_lo = (sc[off + 8] >> 4) & 0x0F;
        uchar s_hi = (sc[off] >> 6) & 0x03;
        scale = d * float(s_lo | (s_hi << 4));
        uchar m_lo = sc[off + 8] & 0x0F;
        uchar m_hi = (sc[off + 4] >> 6) & 0x03;
        minimum = dmin * float(m_lo | (m_hi << 4));
    }
    return float2(scale, minimum);
}

inline float2 scale_min_q4k(device const uchar* sc, uint j) {
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
    device const float* A       [[ buffer(0) ]],
    device const uchar* blocks  [[ buffer(1) ]],
    device       float* C       [[ buffer(2) ]],
    constant     uint&  K       [[ buffer(3) ]],
    constant     uint&  N       [[ buffer(4) ]],
    uint tgid [[ threadgroup_position_in_grid ]],
    uint tid  [[ thread_index_in_threadgroup ]])
{
    uint j = tgid;
    if (j >= N) return;

    uint n_sb = K / 256;
    float acc = 0.0f;

    for (uint b = tid; b < n_sb; b += 32) {
        uint boff = (j * n_sb + b) * 144;
        device const uchar* bp = blocks + boff;

        ushort d_bits    = ushort(bp[0]) | (ushort(bp[1]) << 8u);
        ushort dmin_bits = ushort(bp[2]) | (ushort(bp[3]) << 8u);
        float d    = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = bp + 4u;
        device const uchar* qs = bp + 16u;
        uint a_base = b * 256;

        for (uint chunk = 0u; chunk < 4u; chunk++) {
            float2 sm1 = scale_min_q4k(sc, chunk * 2u);
            float2 sm2 = scale_min_q4k(sc, chunk * 2u + 1u);
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
    if (tid == 0u) C[j] = acc;
}

"#;

    // =========================================================================
    // Helper functions
    // =========================================================================

    fn alloc_buf(
        device: &ProtocolObject<dyn MTLDevice>,
        byte_len: usize,
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
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
        if byte_len == 0 { return alloc_buf(device, 16); }
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr, byte_len, MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: upload_f32 failed")
        }
    }

    fn upload_u16(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[u16],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_len = data.len() * 2;
        if byte_len == 0 { return alloc_buf(device, 16); }
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr, byte_len, MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: upload_u16 failed")
        }
    }

    fn upload_bytes(
        device: &ProtocolObject<dyn MTLDevice>,
        data: &[u8],
    ) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let byte_len = data.len();
        if byte_len == 0 { return alloc_buf(device, 16); }
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr, byte_len, MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: upload_bytes failed")
        }
    }

    fn compile_library(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
    ) -> Retained<ProtocolObject<dyn MTLLibrary>> {
        let src = NSString::from_str(source);
        device
            .newLibraryWithSource_options_error(&src, None)
            .expect("Metal: failed to compile MSL library")
    }

    fn pipeline_from_library(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
        func_name: &str,
    ) -> Retained<ProtocolObject<dyn MTLComputePipelineState>> {
        let name = NSString::from_str(func_name);
        let func = library
            .newFunctionWithName(&name)
            .unwrap_or_else(|| panic!("Metal: function '{}' not found", func_name));
        device
            .newComputePipelineStateWithFunction_error(&func)
            .unwrap_or_else(|e| panic!("Metal: pipeline for '{}' failed: {:?}", func_name, e))
    }

    fn upload_linear_detect_type(
        device: &ProtocolObject<dyn MTLDevice>,
        linear: &crate::nn2::Linear2,
    ) -> (Retained<ProtocolObject<dyn MTLBuffer>>, bool) {
        if let Some(ref q4k) = linear.q4k_weight {
            (upload_bytes(device, &q4k.blocks), false)
        } else if let Some(ref bf16) = linear.bf16_weight {
            (upload_u16(device, &bf16.data), true)
        } else if let Some(ref q4) = linear.q4_weight {
            let f32_mat = q4.dequantize();
            let bf16_data: Vec<u16> = f32_mat.data.iter()
                .map(|&f| crate::autograd2::MatBf16::f32_to_bf16(f))
                .collect();
            (upload_u16(device, &bf16_data), true)
        } else {
            let d = linear.weight.data();
            if d.rows == 0 && d.cols == 0 {
                (alloc_buf(device, 16), true)
            } else {
                let bf16_data: Vec<u16> = d.data.iter()
                    .map(|&f| crate::autograd2::MatBf16::f32_to_bf16(f))
                    .collect();
                (upload_u16(device, &bf16_data), true)
            }
        }
    }

    // =========================================================================
    // Per-layer weight buffer collections
    // =========================================================================

    pub struct DeltaNetLayerWeights {
        pub in_proj_qkv: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub in_proj_qkv_is_bf16: bool,
        pub in_proj_z: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub in_proj_z_is_bf16: bool,
        pub in_proj_a: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub in_proj_a_is_bf16: bool,
        pub in_proj_b: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub in_proj_b_is_bf16: bool,
        pub out_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub out_proj_is_bf16: bool,

        pub conv1d_weight: Retained<ProtocolObject<dyn MTLBuffer>>,  // f32
        pub a_log: Retained<ProtocolObject<dyn MTLBuffer>>,           // f32
        pub dt_bias: Retained<ProtocolObject<dyn MTLBuffer>>,         // f32
        pub norm_weight: Retained<ProtocolObject<dyn MTLBuffer>>,     // f32

        // MLP
        pub gate_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub gate_proj_is_bf16: bool,
        pub up_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub up_proj_is_bf16: bool,
        pub down_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub down_proj_is_bf16: bool,

        // Norms
        pub input_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub post_attn_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    pub struct FullAttnLayerWeights {
        pub q_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub q_proj_is_bf16: bool,
        pub k_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub k_proj_is_bf16: bool,
        pub v_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub v_proj_is_bf16: bool,
        pub o_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub o_proj_is_bf16: bool,

        pub q_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub k_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,

        // MLP
        pub gate_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub gate_proj_is_bf16: bool,
        pub up_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub up_proj_is_bf16: bool,
        pub down_proj: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub down_proj_is_bf16: bool,

        // Norms
        pub input_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
        pub post_attn_layernorm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,
    }

    pub enum LayerWeightsQwen35 {
        DeltaNet(DeltaNetLayerWeights),
        FullAttn(FullAttnLayerWeights),
    }

    // =========================================================================
    // MetalDecodeContextQwen35
    // =========================================================================

    pub struct MetalDecodeContextQwen35 {
        #[allow(dead_code)]
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,

        // Pipelines
        pipe_rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_rms_norm_per_head: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_embed_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_vec_add: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_elem_mul: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_scale: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_conv1d_silu: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_l2_normalize: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_dn_gates: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_dn_recurrent: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gated_rms_norm: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_split_q_gate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_sigmoid_gate: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_rope_partial: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_attention_decode: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_kv_append: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gemv_bf16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_gemv_q4k: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pipe_f32_to_f16: Retained<ProtocolObject<dyn MTLComputePipelineState>>,

        // Per-layer weights
        layer_weights: Vec<LayerWeightsQwen35>,

        // Embedding + lm_head + final norm
        embed_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        lm_head_buf: Retained<ProtocolObject<dyn MTLBuffer>>,
        lm_head_is_bf16: bool,
        final_norm_gamma: Retained<ProtocolObject<dyn MTLBuffer>>,

        // ── Activation scratch buffers ──
        buf_hidden: Retained<ProtocolObject<dyn MTLBuffer>>,
        buf_hidden2: Retained<ProtocolObject<dyn MTLBuffer>>,
        buf_normed: Retained<ProtocolObject<dyn MTLBuffer>>,

        // DeltaNet
        buf_qkv: Retained<ProtocolObject<dyn MTLBuffer>>,       // [qkv_dim]
        buf_z: Retained<ProtocolObject<dyn MTLBuffer>>,          // [value_dim]
        buf_a: Retained<ProtocolObject<dyn MTLBuffer>>,          // [nv]
        buf_b: Retained<ProtocolObject<dyn MTLBuffer>>,          // [nv]
        buf_decay: Retained<ProtocolObject<dyn MTLBuffer>>,      // [nv]
        buf_beta: Retained<ProtocolObject<dyn MTLBuffer>>,       // [nv]
        buf_dn_output: Retained<ProtocolObject<dyn MTLBuffer>>,  // [value_dim]
        buf_dn_gated: Retained<ProtocolObject<dyn MTLBuffer>>,   // [value_dim]

        // FullAttn
        buf_qg: Retained<ProtocolObject<dyn MTLBuffer>>,         // [nq * d * 2]
        buf_q: Retained<ProtocolObject<dyn MTLBuffer>>,          // [nq * d]
        buf_fa_gate: Retained<ProtocolObject<dyn MTLBuffer>>,    // [nq * d]
        buf_k: Retained<ProtocolObject<dyn MTLBuffer>>,          // [nkv * d]
        buf_v: Retained<ProtocolObject<dyn MTLBuffer>>,          // [nkv * d]
        buf_attn_out: Retained<ProtocolObject<dyn MTLBuffer>>,   // [nq * d]
        buf_sigmoid_out: Retained<ProtocolObject<dyn MTLBuffer>>,// [nq * d]

        // MLP
        buf_mlp_gate: Retained<ProtocolObject<dyn MTLBuffer>>,   // [intermediate]
        buf_mlp_up: Retained<ProtocolObject<dyn MTLBuffer>>,     // [intermediate]
        buf_mlp_hidden: Retained<ProtocolObject<dyn MTLBuffer>>, // [intermediate]
        buf_down_out: Retained<ProtocolObject<dyn MTLBuffer>>,   // [hidden]

        buf_logits: Retained<ProtocolObject<dyn MTLBuffer>>,     // [vocab]

        // Per-layer DeltaNet state (persistent across steps)
        dn_state_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,    // [nv * kd * vd] f32
        dn_conv_state_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,// [qkv_dim * hist] f32

        // Per-layer FullAttn KV cache (half)
        fa_kv_k_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
        fa_kv_v_bufs: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,

        // Config
        config: ConfigQwen35,
        lm_head_vocab: usize,
    }

    // =========================================================================
    // Implementation
    // =========================================================================

    impl MetalDecodeContextQwen35 {
        pub fn new(model: &Qwen35Model) -> Self {
            let cfg = &model.config;
            let device = objc2_metal::MTLCreateSystemDefaultDevice()
                .expect("Metal: no GPU device found");
            let queue = device.newCommandQueue().expect("Metal: queue creation failed");

            eprintln!("[ Qwen3.5-Metal ] Compiling MSL kernels...");
            let t0 = std::time::Instant::now();
            let library = compile_library(&device, DECODE_QWEN35_MSL);
            let pipe = |name: &str| pipeline_from_library(&device, &library, name);

            let pipe_rms_norm       = pipe("rms_norm_gemma3");
            let pipe_rms_norm_per_head = pipe("rms_norm_per_head");
            let pipe_embed_bf16     = pipe("embed_bf16_lookup");
            let pipe_vec_add        = pipe("vec_add_kernel");
            let pipe_elem_mul       = pipe("elem_mul_kernel");
            let pipe_silu           = pipe("silu_kernel");
            let pipe_scale          = pipe("scale_inplace");
            let pipe_conv1d_silu    = pipe("conv1d_silu");
            let pipe_l2_normalize   = pipe("l2_normalize_heads");
            let pipe_dn_gates       = pipe("deltanet_compute_gates");
            let pipe_dn_recurrent   = pipe("deltanet_recurrent");
            let pipe_gated_rms_norm = pipe("gated_rms_norm");
            let pipe_split_q_gate   = pipe("split_q_gate");
            let pipe_sigmoid_gate   = pipe("sigmoid_gate");
            let pipe_rope_partial   = pipe("rope_neox_partial");
            let pipe_attention_decode = pipe("attention_decode");
            let pipe_kv_append      = pipe("kv_cache_append");
            let pipe_gemv_bf16      = pipe("gemv_bf16_t");
            let pipe_gemv_q4k       = pipe("gemv_q4k_t");
            let pipe_f32_to_f16     = pipe("f32_to_f16_convert");
            eprintln!("[ Qwen3.5-Metal ] MSL compile: {:.0}ms", t0.elapsed().as_millis());

            // Upload weights
            eprintln!("[ Qwen3.5-Metal ] Uploading weights...");
            let t1 = std::time::Instant::now();

            let mut layer_weights = Vec::with_capacity(cfg.num_hidden_layers);
            let mut dn_state_bufs = Vec::new();
            let mut dn_conv_state_bufs = Vec::new();
            let mut fa_kv_k_bufs = Vec::new();
            let mut fa_kv_v_bufs = Vec::new();

            let max_seq = 4096usize; // max context length for KV cache
            let qkv_dim = cfg.deltanet_qkv_dim();
            let nk = cfg.linear_num_key_heads;
            let nv = cfg.linear_num_value_heads;
            let kd = cfg.linear_key_head_dim;
            let vd = cfg.linear_value_head_dim;
            let value_dim = nv * vd;
            let conv_hist = cfg.linear_conv_kernel_dim - 1;

            let nq = cfg.num_attention_heads;
            let nkv = cfg.num_key_value_heads;
            let d = cfg.head_dim;
            let kv_dim = nkv * d;

            for (i, layer) in model.layers.iter().enumerate() {
                match &layer.token_mixer {
                    TokenMixer::DeltaNet(dn) => {
                        let (qkv_buf, qkv_bf16) = upload_linear_detect_type(&device, &dn.in_proj_qkv);
                        let (z_buf, z_bf16) = upload_linear_detect_type(&device, &dn.in_proj_z);
                        let (a_buf, a_bf16) = upload_linear_detect_type(&device, &dn.in_proj_a);
                        let (b_buf, b_bf16) = upload_linear_detect_type(&device, &dn.in_proj_b);
                        let (out_buf, out_bf16) = upload_linear_detect_type(&device, &dn.out_proj);
                        let (gate_buf, gate_bf16) = upload_linear_detect_type(&device, &layer.mlp.gate_proj);
                        let (up_buf, up_bf16) = upload_linear_detect_type(&device, &layer.mlp.up_proj);
                        let (down_buf, down_bf16) = upload_linear_detect_type(&device, &layer.mlp.down_proj);

                        let conv_w = upload_f32(&device, &dn.conv1d_weight);
                        let a_log = upload_f32(&device, &dn.a_log);
                        let dt_bias = upload_f32(&device, &dn.dt_bias);
                        let norm_w = upload_f32(&device, &dn.norm_weight);

                        let in_gamma = upload_f32(&device, &layer.input_layernorm.gamma.data().data);
                        let post_gamma = upload_f32(&device, &layer.post_attention_layernorm.gamma.data().data);

                        layer_weights.push(LayerWeightsQwen35::DeltaNet(DeltaNetLayerWeights {
                            in_proj_qkv: qkv_buf, in_proj_qkv_is_bf16: qkv_bf16,
                            in_proj_z: z_buf, in_proj_z_is_bf16: z_bf16,
                            in_proj_a: a_buf, in_proj_a_is_bf16: a_bf16,
                            in_proj_b: b_buf, in_proj_b_is_bf16: b_bf16,
                            out_proj: out_buf, out_proj_is_bf16: out_bf16,
                            conv1d_weight: conv_w, a_log, dt_bias, norm_weight: norm_w,
                            gate_proj: gate_buf, gate_proj_is_bf16: gate_bf16,
                            up_proj: up_buf, up_proj_is_bf16: up_bf16,
                            down_proj: down_buf, down_proj_is_bf16: down_bf16,
                            input_layernorm_gamma: in_gamma,
                            post_attn_layernorm_gamma: post_gamma,
                        }));

                        // Allocate DeltaNet state buffers
                        dn_state_bufs.push(alloc_buf(&device, nv * kd * vd * 4));
                        dn_conv_state_bufs.push(alloc_buf(&device, qkv_dim * conv_hist * 4));
                    }
                    TokenMixer::FullAttn(fa) => {
                        let (q_buf, q_bf16) = upload_linear_detect_type(&device, &fa.q_proj);
                        let (k_buf, k_bf16) = upload_linear_detect_type(&device, &fa.k_proj);
                        let (v_buf, v_bf16) = upload_linear_detect_type(&device, &fa.v_proj);
                        let (o_buf, o_bf16) = upload_linear_detect_type(&device, &fa.o_proj);
                        let (gate_buf, gate_bf16) = upload_linear_detect_type(&device, &layer.mlp.gate_proj);
                        let (up_buf, up_bf16) = upload_linear_detect_type(&device, &layer.mlp.up_proj);
                        let (down_buf, down_bf16) = upload_linear_detect_type(&device, &layer.mlp.down_proj);

                        let q_norm_g = upload_f32(&device, &fa.q_norm.gamma.data().data);
                        let k_norm_g = upload_f32(&device, &fa.k_norm.gamma.data().data);
                        let in_gamma = upload_f32(&device, &layer.input_layernorm.gamma.data().data);
                        let post_gamma = upload_f32(&device, &layer.post_attention_layernorm.gamma.data().data);

                        layer_weights.push(LayerWeightsQwen35::FullAttn(FullAttnLayerWeights {
                            q_proj: q_buf, q_proj_is_bf16: q_bf16,
                            k_proj: k_buf, k_proj_is_bf16: k_bf16,
                            v_proj: v_buf, v_proj_is_bf16: v_bf16,
                            o_proj: o_buf, o_proj_is_bf16: o_bf16,
                            q_norm_gamma: q_norm_g, k_norm_gamma: k_norm_g,
                            gate_proj: gate_buf, gate_proj_is_bf16: gate_bf16,
                            up_proj: up_buf, up_proj_is_bf16: up_bf16,
                            down_proj: down_buf, down_proj_is_bf16: down_bf16,
                            input_layernorm_gamma: in_gamma,
                            post_attn_layernorm_gamma: post_gamma,
                        }));

                        // Allocate KV cache buffers (half precision)
                        fa_kv_k_bufs.push(alloc_buf(&device, max_seq * kv_dim * 2));
                        fa_kv_v_bufs.push(alloc_buf(&device, max_seq * kv_dim * 2));
                    }
                }
            }

            // Embedding
            let embed_buf = if let Some(ref bf16) = model.embed_bf16 {
                upload_u16(&device, &bf16.data)
            } else {
                let e = model.embed_tokens.data();
                let bf16_data: Vec<u16> = e.data.iter()
                    .map(|&f| crate::autograd2::MatBf16::f32_to_bf16(f))
                    .collect();
                upload_u16(&device, &bf16_data)
            };

            // lm_head (weight-tied to embed)
            let lm_head_vocab = if let Some(ref bf16) = model.embed_bf16 {
                bf16.data.len() / cfg.hidden_size
            } else {
                let e = model.embed_tokens.data();
                if e.rows > 0 { e.rows } else { cfg.vocab_size }
            };
            let lm_head_buf = embed_buf.clone();  // weight-tied
            let lm_head_is_bf16 = true;

            // Final norm
            let final_norm_gamma = upload_f32(&device, &model.norm.gamma.data().data);

            eprintln!("[ Qwen3.5-Metal ] Weight upload: {:.0}ms", t1.elapsed().as_millis());

            // Activation buffers
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;

            let buf_hidden = alloc_buf(&device, h * 4);
            let buf_hidden2 = alloc_buf(&device, h * 4);
            let buf_normed = alloc_buf(&device, h * 4);
            let buf_qkv_a = alloc_buf(&device, qkv_dim * 4);
            let buf_z_a = alloc_buf(&device, value_dim * 4);
            let buf_a_a = alloc_buf(&device, nv * 4);
            let buf_b_a = alloc_buf(&device, nv * 4);
            let buf_decay_a = alloc_buf(&device, nv * 4);
            let buf_beta_a = alloc_buf(&device, nv * 4);
            let buf_dn_output_a = alloc_buf(&device, value_dim * 4);
            let buf_dn_gated_a = alloc_buf(&device, value_dim * 4);
            let buf_qg_a = alloc_buf(&device, nq * d * 2 * 4);
            let buf_q_a = alloc_buf(&device, nq * d * 4);
            let buf_fa_gate_a = alloc_buf(&device, nq * d * 4);
            let buf_k_a = alloc_buf(&device, kv_dim * 4);
            let buf_v_a = alloc_buf(&device, kv_dim * 4);
            let buf_attn_out_a = alloc_buf(&device, nq * d * 4);
            let buf_sigmoid_out_a = alloc_buf(&device, nq * d * 4);
            let buf_mlp_gate_a = alloc_buf(&device, inter * 4);
            let buf_mlp_up_a = alloc_buf(&device, inter * 4);
            let buf_mlp_hidden_a = alloc_buf(&device, inter * 4);
            let buf_down_out_a = alloc_buf(&device, h * 4);
            let buf_logits_a = alloc_buf(&device, lm_head_vocab * 4);

            let ctx = MetalDecodeContextQwen35 {
                device, queue,
                pipe_rms_norm, pipe_rms_norm_per_head, pipe_embed_bf16,
                pipe_vec_add, pipe_elem_mul, pipe_silu, pipe_scale,
                pipe_conv1d_silu, pipe_l2_normalize, pipe_dn_gates,
                pipe_dn_recurrent, pipe_gated_rms_norm, pipe_split_q_gate,
                pipe_sigmoid_gate, pipe_rope_partial, pipe_attention_decode,
                pipe_kv_append, pipe_gemv_bf16, pipe_gemv_q4k, pipe_f32_to_f16,

                layer_weights,
                embed_buf, lm_head_buf, lm_head_is_bf16, final_norm_gamma,

                buf_hidden, buf_hidden2, buf_normed,
                buf_qkv: buf_qkv_a, buf_z: buf_z_a,
                buf_a: buf_a_a, buf_b: buf_b_a,
                buf_decay: buf_decay_a, buf_beta: buf_beta_a,
                buf_dn_output: buf_dn_output_a, buf_dn_gated: buf_dn_gated_a,

                buf_qg: buf_qg_a, buf_q: buf_q_a,
                buf_fa_gate: buf_fa_gate_a,
                buf_k: buf_k_a, buf_v: buf_v_a,
                buf_attn_out: buf_attn_out_a, buf_sigmoid_out: buf_sigmoid_out_a,

                buf_mlp_gate: buf_mlp_gate_a, buf_mlp_up: buf_mlp_up_a,
                buf_mlp_hidden: buf_mlp_hidden_a, buf_down_out: buf_down_out_a,
                buf_logits: buf_logits_a,

                dn_state_bufs, dn_conv_state_bufs,
                fa_kv_k_bufs, fa_kv_v_bufs,

                config: cfg.clone(),
                lm_head_vocab,
            };

            ctx
        }

        // =====================================================================
        // Dispatch helpers
        // =====================================================================

        fn dispatch_gemv(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            act_buf: &ProtocolObject<dyn MTLBuffer>,
            weight_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            k: usize, n: usize, is_bf16: bool,
        ) {
            if is_bf16 {
                enc.setComputePipelineState(&self.pipe_gemv_bf16);
            } else {
                enc.setComputePipelineState(&self.pipe_gemv_q4k);
            }
            let k_u32 = k as u32;
            let n_u32 = n as u32;
            unsafe {
                enc.setBuffer_offset_atIndex(Some(act_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(weight_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&k_u32 as *const u32 as *mut c_void).unwrap(), 4, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 4);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: n, height: 1, depth: 1 },
                MTLSize { width: 32, height: 1, depth: 1 },
            );
        }

        fn dispatch_rms_norm(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            gamma_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            dim: usize,
        ) {
            enc.setComputePipelineState(&self.pipe_rms_norm);
            let d = dim as u32;
            let eps = self.config.rms_norm_eps;
            unsafe {
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(gamma_buf), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 2);
                enc.setBytes_length_atIndex(
                    NonNull::new(&d as *const u32 as *mut c_void).unwrap(), 4, 3);
                enc.setBytes_length_atIndex(
                    NonNull::new(&eps as *const f32 as *mut c_void).unwrap(), 4, 4);
            }
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: 1, height: 1, depth: 1 },
                MTLSize { width: 256.min(dim), height: 1, depth: 1 },
            );
        }

        fn dispatch_vec_add(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            a: &ProtocolObject<dyn MTLBuffer>,
            b: &ProtocolObject<dyn MTLBuffer>,
            out: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            enc.setComputePipelineState(&self.pipe_vec_add);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out), 0, 2);
            }
            let tg = 256;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: (n + tg - 1) / tg, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        fn dispatch_elem_mul(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            a: &ProtocolObject<dyn MTLBuffer>,
            b: &ProtocolObject<dyn MTLBuffer>,
            out: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            enc.setComputePipelineState(&self.pipe_elem_mul);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a), 0, 0);
                enc.setBuffer_offset_atIndex(Some(b), 0, 1);
                enc.setBuffer_offset_atIndex(Some(out), 0, 2);
            }
            let tg = 256;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: (n + tg - 1) / tg, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        fn dispatch_silu(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            x_buf: &ProtocolObject<dyn MTLBuffer>,
            out_buf: &ProtocolObject<dyn MTLBuffer>,
            n: usize,
        ) {
            enc.setComputePipelineState(&self.pipe_silu);
            let n_u32 = n as u32;
            unsafe {
                enc.setBuffer_offset_atIndex(Some(x_buf), 0, 0);
                enc.setBuffer_offset_atIndex(Some(out_buf), 0, 1);
                enc.setBytes_length_atIndex(
                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 2);
            }
            let tg = 256;
            enc.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize { width: (n + tg - 1) / tg, height: 1, depth: 1 },
                MTLSize { width: tg, height: 1, depth: 1 },
            );
        }

        // =====================================================================
        // DeltaNet layer encoding
        // =====================================================================

        fn encode_deltanet_layer(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            lw: &DeltaNetLayerWeights,
            dn_idx: usize,  // index into dn_state_bufs
        ) {
            let cfg = &self.config;
            let h = cfg.hidden_size;
            let qkv_dim = cfg.deltanet_qkv_dim();
            let nk = cfg.linear_num_key_heads;
            let nv = cfg.linear_num_value_heads;
            let kd = cfg.linear_key_head_dim;
            let vd = cfg.linear_value_head_dim;
            let value_dim = nv * vd;
            let v_per_k = nv / nk;
            let inter = cfg.intermediate_size;
            let ks = cfg.linear_conv_kernel_dim;

            // 1. Input layernorm
            self.dispatch_rms_norm(enc, &self.buf_hidden, &lw.input_layernorm_gamma, &self.buf_normed, h);

            // 2-5. Projections
            self.dispatch_gemv(enc, &self.buf_normed, &lw.in_proj_qkv, &self.buf_qkv, h, qkv_dim, lw.in_proj_qkv_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.in_proj_z, &self.buf_z, h, value_dim, lw.in_proj_z_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.in_proj_a, &self.buf_a, h, nv, lw.in_proj_a_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.in_proj_b, &self.buf_b, h, nv, lw.in_proj_b_is_bf16);

            // 6. Conv1d + SiLU (in-place on buf_qkv)
            {
                enc.setComputePipelineState(&self.pipe_conv1d_silu);
                let dim_u32 = qkv_dim as u32;
                let ks_u32 = ks as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.dn_conv_state_bufs[dn_idx]), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&lw.conv1d_weight), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&dim_u32 as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&ks_u32 as *const u32 as *mut c_void).unwrap(), 4, 4);
                }
                let tg = 256;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (qkv_dim + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // 7. L2 normalize Q (first nk*kd elements of buf_qkv)
            {
                enc.setComputePipelineState(&self.pipe_l2_normalize);
                let nh = nk as u32;
                let dim = kd as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&dim as *const u32 as *mut c_void).unwrap(), 4, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nk, height: 1, depth: 1 },
                    MTLSize { width: 128.min(kd), height: 1, depth: 1 },
                );
            }

            // 8. L2 normalize K (buf_qkv offset by nk*kd*4 bytes)
            {
                enc.setComputePipelineState(&self.pipe_l2_normalize);
                let nh = nk as u32;
                let dim = kd as u32;
                let key_dim = nk * kd;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), key_dim * 4, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&dim as *const u32 as *mut c_void).unwrap(), 4, 2);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nk, height: 1, depth: 1 },
                    MTLSize { width: 128.min(kd), height: 1, depth: 1 },
                );
            }

            // 9. Scale Q by 1/sqrt(kd) (in-place, first nk*kd elements)
            {
                enc.setComputePipelineState(&self.pipe_scale);
                let scale = 1.0 / (kd as f32).sqrt();
                let n_u32 = (nk * kd) as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&scale as *const f32 as *mut c_void).unwrap(), 4, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 2);
                }
                let tg = 256;
                let key_dim = nk * kd;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (key_dim + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // 10. Compute gates
            {
                enc.setComputePipelineState(&self.pipe_dn_gates);
                let nv_u32 = nv as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_a), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_b), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&lw.a_log), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&lw.dt_bias), 0, 3);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_decay), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_beta), 0, 5);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nv_u32 as *const u32 as *mut c_void).unwrap(), 4, 6);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (nv + 31) / 32, height: 1, depth: 1 },
                    MTLSize { width: 32.min(nv), height: 1, depth: 1 },
                );
            }

            // 11. Recurrent state update
            {
                enc.setComputePipelineState(&self.pipe_dn_recurrent);
                let kd_u32 = kd as u32;
                let vd_u32 = vd as u32;
                let vpk_u32 = v_per_k as u32;
                let key_dim = nk * kd;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.dn_state_bufs[dn_idx]), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), 0, 1);                // Q at offset 0
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), key_dim * 4, 2);      // K at offset
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qkv), key_dim * 2 * 4, 3);  // V at offset
                    enc.setBuffer_offset_atIndex(Some(&self.buf_decay), 0, 4);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_beta), 0, 5);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_dn_output), 0, 6);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&kd_u32 as *const u32 as *mut c_void).unwrap(), 4, 7);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&vd_u32 as *const u32 as *mut c_void).unwrap(), 4, 8);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&vpk_u32 as *const u32 as *mut c_void).unwrap(), 4, 9);
                }
                // One threadgroup per value head, vd threads per threadgroup
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nv, height: 1, depth: 1 },
                    MTLSize { width: vd, height: 1, depth: 1 },
                );
            }

            // 12. Gated RMSNorm
            {
                enc.setComputePipelineState(&self.pipe_gated_rms_norm);
                let nv_u32 = nv as u32;
                let vd_u32 = vd as u32;
                let eps: f32 = 1e-6;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_dn_output), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_z), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&lw.norm_weight), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_dn_gated), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nv_u32 as *const u32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&vd_u32 as *const u32 as *mut c_void).unwrap(), 4, 5);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&eps as *const f32 as *mut c_void).unwrap(), 4, 6);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nv, height: 1, depth: 1 },
                    MTLSize { width: 128.min(vd), height: 1, depth: 1 },
                );
            }

            // 13. Output projection
            self.dispatch_gemv(enc, &self.buf_dn_gated, &lw.out_proj, &self.buf_normed, value_dim, h, lw.out_proj_is_bf16);

            // 14. Residual: hidden + normed → hidden2
            self.dispatch_vec_add(enc, &self.buf_hidden, &self.buf_normed, &self.buf_hidden2, h);

            // 15. Post-attention layernorm
            self.dispatch_rms_norm(enc, &self.buf_hidden2, &lw.post_attn_layernorm_gamma, &self.buf_normed, h);

            // 16-21. MLP (SwiGLU)
            self.dispatch_gemv(enc, &self.buf_normed, &lw.gate_proj, &self.buf_mlp_gate, h, inter, lw.gate_proj_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.up_proj, &self.buf_mlp_up, h, inter, lw.up_proj_is_bf16);
            self.dispatch_silu(enc, &self.buf_mlp_gate, &self.buf_mlp_hidden, inter);
            self.dispatch_elem_mul(enc, &self.buf_mlp_hidden, &self.buf_mlp_up, &self.buf_mlp_gate, inter);
            self.dispatch_gemv(enc, &self.buf_mlp_gate, &lw.down_proj, &self.buf_down_out, inter, h, lw.down_proj_is_bf16);

            // 22. Final residual: hidden2 + down_out → hidden
            self.dispatch_vec_add(enc, &self.buf_hidden2, &self.buf_down_out, &self.buf_hidden, h);
        }

        // =====================================================================
        // FullAttention layer encoding
        // =====================================================================

        fn encode_fullattn_layer(
            &self,
            enc: &ProtocolObject<dyn MTLComputeCommandEncoder>,
            lw: &FullAttnLayerWeights,
            fa_idx: usize,  // index into fa_kv_k_bufs
            position: usize,
        ) {
            let cfg = &self.config;
            let h = cfg.hidden_size;
            let nq = cfg.num_attention_heads;
            let nkv = cfg.num_key_value_heads;
            let d = cfg.head_dim;
            let kv_dim = nkv * d;
            let rope_dim = cfg.rope_dim();
            let inter = cfg.intermediate_size;

            // 1. Input layernorm
            self.dispatch_rms_norm(enc, &self.buf_hidden, &lw.input_layernorm_gamma, &self.buf_normed, h);

            // 2-4. Projections
            self.dispatch_gemv(enc, &self.buf_normed, &lw.q_proj, &self.buf_qg, h, nq * d * 2, lw.q_proj_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.k_proj, &self.buf_k, h, kv_dim, lw.k_proj_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.v_proj, &self.buf_v, h, kv_dim, lw.v_proj_is_bf16);

            // 5. Split Q and gate
            {
                enc.setComputePipelineState(&self.pipe_split_q_gate);
                let nq_u32 = nq as u32;
                let d_u32 = d as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_qg), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_q), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_fa_gate), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nq_u32 as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&d_u32 as *const u32 as *mut c_void).unwrap(), 4, 4);
                }
                let total = nq * d;
                let tg = 256;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (total + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // 6. Per-head RMSNorm Q → buf_q (in-place? no, need separate output)
            // Actually rms_norm_per_head reads from buffer 0, writes to buffer 2.
            // To do in-place, use buf_q as both input and output.
            // Let's use buf_attn_out as temp for normed Q.
            {
                enc.setComputePipelineState(&self.pipe_rms_norm_per_head);
                let nh = nq as u32;
                let hd = d as u32;
                let eps = self.config.rms_norm_eps;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_q), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&lw.q_norm_gamma), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_attn_out), 0, 2); // temp output
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&hd as *const u32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&eps as *const f32 as *mut c_void).unwrap(), 4, 5);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nq, height: 1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
            }

            // 7. Per-head RMSNorm K → buf_sigmoid_out as temp
            {
                enc.setComputePipelineState(&self.pipe_rms_norm_per_head);
                let nh = nkv as u32;
                let hd = d as u32;
                let eps = self.config.rms_norm_eps;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_k), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&lw.k_norm_gamma), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_sigmoid_out), 0, 2); // temp
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&hd as *const u32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&eps as *const f32 as *mut c_void).unwrap(), 4, 5);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nkv, height: 1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
            }

            // 8. Partial RoPE Q (buf_attn_out → in-place)
            {
                enc.setComputePipelineState(&self.pipe_rope_partial);
                let nh = nq as u32;
                let hd = d as u32;
                let rd = rope_dim as u32;
                let theta = cfg.rope_theta;
                let pos = position as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_attn_out), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&hd as *const u32 as *mut c_void).unwrap(), 4, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&rd as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&theta as *const f32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&pos as *const u32 as *mut c_void).unwrap(), 4, 5);
                }
                let half_rope = rope_dim / 2;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nq, height: 1, depth: 1 },
                    MTLSize { width: half_rope, height: 1, depth: 1 },
                );
            }

            // 9. Partial RoPE K (buf_sigmoid_out → in-place)
            {
                enc.setComputePipelineState(&self.pipe_rope_partial);
                let nh = nkv as u32;
                let hd = d as u32;
                let rd = rope_dim as u32;
                let theta = cfg.rope_theta;
                let pos = position as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_sigmoid_out), 0, 0);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nh as *const u32 as *mut c_void).unwrap(), 4, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&hd as *const u32 as *mut c_void).unwrap(), 4, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&rd as *const u32 as *mut c_void).unwrap(), 4, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&theta as *const f32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&pos as *const u32 as *mut c_void).unwrap(), 4, 5);
                }
                let half_rope = rope_dim / 2;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nkv, height: 1, depth: 1 },
                    MTLSize { width: half_rope, height: 1, depth: 1 },
                );
            }

            // 10. KV cache append
            {
                enc.setComputePipelineState(&self.pipe_kv_append);
                let sl = position as u32;
                let kvd = kv_dim as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_sigmoid_out), 0, 0);  // K (after RoPE)
                    enc.setBuffer_offset_atIndex(Some(&self.buf_v), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.fa_kv_k_bufs[fa_idx]), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.fa_kv_v_bufs[fa_idx]), 0, 3);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&sl as *const u32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&kvd as *const u32 as *mut c_void).unwrap(), 4, 5);
                }
                let tg = 256;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (kv_dim + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // 11. Attention decode
            {
                enc.setComputePipelineState(&self.pipe_attention_decode);
                let nq_u32 = nq as u32;
                let nkv_u32 = nkv as u32;
                let d_u32 = d as u32;
                let scale = 1.0f32 / (d as f32).sqrt();
                let k_end = (position + 1) as u32;
                let sw: u32 = 0; // no sliding window
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_attn_out), 0, 0); // Q (after RoPE)
                    enc.setBuffer_offset_atIndex(Some(&self.fa_kv_k_bufs[fa_idx]), 0, 1);
                    enc.setBuffer_offset_atIndex(Some(&self.fa_kv_v_bufs[fa_idx]), 0, 2);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_q), 0, 3);  // reuse as output
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nq_u32 as *const u32 as *mut c_void).unwrap(), 4, 4);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&nkv_u32 as *const u32 as *mut c_void).unwrap(), 4, 5);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&d_u32 as *const u32 as *mut c_void).unwrap(), 4, 6);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&scale as *const f32 as *mut c_void).unwrap(), 4, 7);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&k_end as *const u32 as *mut c_void).unwrap(), 4, 8);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&sw as *const u32 as *mut c_void).unwrap(), 4, 9);
                }
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: nq, height: 1, depth: 1 },
                    MTLSize { width: 32, height: 1, depth: 1 },
                );
            }

            // 12. Sigmoid gate: attn_out * sigmoid(gate) → sigmoid_out
            {
                enc.setComputePipelineState(&self.pipe_sigmoid_gate);
                let n_u32 = (nq * d) as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.buf_q), 0, 0);         // attn output
                    enc.setBuffer_offset_atIndex(Some(&self.buf_fa_gate), 0, 1);   // gate values
                    enc.setBuffer_offset_atIndex(Some(&self.buf_sigmoid_out), 0, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 3);
                }
                let tg = 256;
                let total = nq * d;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (total + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // 13. Output projection
            self.dispatch_gemv(enc, &self.buf_sigmoid_out, &lw.o_proj, &self.buf_normed, nq * d, h, lw.o_proj_is_bf16);

            // 14. Residual
            self.dispatch_vec_add(enc, &self.buf_hidden, &self.buf_normed, &self.buf_hidden2, h);

            // 15. Post-attention layernorm
            self.dispatch_rms_norm(enc, &self.buf_hidden2, &lw.post_attn_layernorm_gamma, &self.buf_normed, h);

            // 16-21. MLP
            self.dispatch_gemv(enc, &self.buf_normed, &lw.gate_proj, &self.buf_mlp_gate, h, inter, lw.gate_proj_is_bf16);
            self.dispatch_gemv(enc, &self.buf_normed, &lw.up_proj, &self.buf_mlp_up, h, inter, lw.up_proj_is_bf16);
            self.dispatch_silu(enc, &self.buf_mlp_gate, &self.buf_mlp_hidden, inter);
            self.dispatch_elem_mul(enc, &self.buf_mlp_hidden, &self.buf_mlp_up, &self.buf_mlp_gate, inter);
            self.dispatch_gemv(enc, &self.buf_mlp_gate, &lw.down_proj, &self.buf_down_out, inter, h, lw.down_proj_is_bf16);

            // 22. Final residual
            self.dispatch_vec_add(enc, &self.buf_hidden2, &self.buf_down_out, &self.buf_hidden, h);
        }

        // =====================================================================
        // Full decode step
        // =====================================================================

        pub fn decode_step(&self, token_id: usize, position: usize) -> Vec<f32> {
            let h = self.config.hidden_size;

            let cmd = self.queue.commandBuffer().expect("commandBuffer failed");
            let enc = cmd.computeCommandEncoder().expect("encoder failed");

            // Embed token → buf_hidden
            {
                enc.setComputePipelineState(&self.pipe_embed_bf16);
                let tid = token_id as u32;
                let hs = h as u32;
                unsafe {
                    enc.setBuffer_offset_atIndex(Some(&self.embed_buf), 0, 0);
                    enc.setBuffer_offset_atIndex(Some(&self.buf_hidden), 0, 1);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&tid as *const u32 as *mut c_void).unwrap(), 4, 2);
                    enc.setBytes_length_atIndex(
                        NonNull::new(&hs as *const u32 as *mut c_void).unwrap(), 4, 3);
                }
                let tg = 256;
                enc.dispatchThreadgroups_threadsPerThreadgroup(
                    MTLSize { width: (h + tg - 1) / tg, height: 1, depth: 1 },
                    MTLSize { width: tg, height: 1, depth: 1 },
                );
            }

            // Process layers
            let mut dn_idx = 0usize;
            let mut fa_idx = 0usize;

            for (_i, lw) in self.layer_weights.iter().enumerate() {
                match lw {
                    LayerWeightsQwen35::DeltaNet(dlw) => {
                        self.encode_deltanet_layer(&enc, dlw, dn_idx);
                        dn_idx += 1;
                    }
                    LayerWeightsQwen35::FullAttn(flw) => {
                        self.encode_fullattn_layer(&enc, flw, fa_idx, position);
                        fa_idx += 1;
                    }
                }
            }

            // Final norm
            self.dispatch_rms_norm(&enc, &self.buf_hidden, &self.final_norm_gamma, &self.buf_normed, h);

            // lm_head
            self.dispatch_gemv(&enc, &self.buf_normed, &self.lm_head_buf, &self.buf_logits,
                              h, self.lm_head_vocab, self.lm_head_is_bf16);

            enc.endEncoding();
            cmd.commit();
            cmd.waitUntilCompleted();

            // Read logits
            let ptr = self.buf_logits.contents().as_ptr() as *const f32;
            let logits = unsafe { std::slice::from_raw_parts(ptr, self.lm_head_vocab) };
            logits.to_vec()
        }

        // =====================================================================
        // Sync state from CPU prefill
        // =====================================================================

        /// Copy DeltaNet states and FullAttn KV caches from CPU to GPU after prefill.
        pub fn sync_state_from_cpu(&self, cache: &Qwen35Cache) {
            let cfg = &self.config;
            let nv = cfg.linear_num_value_heads;
            let kd = cfg.linear_key_head_dim;
            let vd = cfg.linear_value_head_dim;
            let nkv = cfg.num_key_value_heads;
            let d = cfg.head_dim;
            let kv_dim = nkv * d;

            let mut dn_idx = 0usize;
            let mut fa_idx = 0usize;

            for (i, layer_cache) in cache.layers.iter().enumerate() {
                let lc = layer_cache.borrow();
                match &*lc {
                    LayerCache::DeltaNet(state) => {
                        // Copy recurrent state
                        let state_bytes = state.state.len() * 4;
                        let dst_ptr = self.dn_state_bufs[dn_idx].contents().as_ptr() as *mut f32;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                state.state.as_ptr(), dst_ptr, state.state.len());
                        }

                        // Copy conv state
                        let conv_ptr = self.dn_conv_state_bufs[dn_idx].contents().as_ptr() as *mut f32;
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                state.conv_state.as_ptr(), conv_ptr, state.conv_state.len());
                        }
                        dn_idx += 1;
                    }
                    LayerCache::FullAttn(kv_cache) => {
                        // Convert f32 KV → half and copy to GPU buffers
                        let seq_len = kv_cache.seq_len;
                        if seq_len > 0 {
                            let n_floats = seq_len * kv_dim;

                            // Use a temporary staging buffer for f32→f16 conversion
                            let staging = upload_f32(&self.device, &kv_cache.k.data[..n_floats]);
                            let cmd = self.queue.commandBuffer().expect("cmd");
                            let enc = cmd.computeCommandEncoder().expect("enc");
                            enc.setComputePipelineState(&self.pipe_f32_to_f16);
                            let n_u32 = n_floats as u32;
                            unsafe {
                                enc.setBuffer_offset_atIndex(Some(&staging), 0, 0);
                                enc.setBuffer_offset_atIndex(Some(&self.fa_kv_k_bufs[fa_idx]), 0, 1);
                                enc.setBytes_length_atIndex(
                                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 2);
                            }
                            let tg = 256;
                            enc.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize { width: (n_floats + tg - 1) / tg, height: 1, depth: 1 },
                                MTLSize { width: tg, height: 1, depth: 1 },
                            );

                            // V
                            let staging_v = upload_f32(&self.device, &kv_cache.v.data[..n_floats]);
                            enc.setComputePipelineState(&self.pipe_f32_to_f16);
                            unsafe {
                                enc.setBuffer_offset_atIndex(Some(&staging_v), 0, 0);
                                enc.setBuffer_offset_atIndex(Some(&self.fa_kv_v_bufs[fa_idx]), 0, 1);
                                enc.setBytes_length_atIndex(
                                    NonNull::new(&n_u32 as *const u32 as *mut c_void).unwrap(), 4, 2);
                            }
                            enc.dispatchThreadgroups_threadsPerThreadgroup(
                                MTLSize { width: (n_floats + tg - 1) / tg, height: 1, depth: 1 },
                                MTLSize { width: tg, height: 1, depth: 1 },
                            );

                            enc.endEncoding();
                            cmd.commit();
                            cmd.waitUntilCompleted();
                        }
                        fa_idx += 1;
                    }
                }
            }
        }

        pub fn print_memory_stats(&self) {
            // Rough estimate
            let h = self.config.hidden_size;
            let nv = self.config.linear_num_value_heads;
            let kd = self.config.linear_key_head_dim;
            let vd = self.config.linear_value_head_dim;
            let n_dn = self.dn_state_bufs.len();
            let n_fa = self.fa_kv_k_bufs.len();

            let state_mb = (n_dn * nv * kd * vd * 4) as f64 / 1e6;
            let kv_mb = (n_fa * 2 * 4096 * self.config.num_key_value_heads * self.config.head_dim * 2) as f64 / 1e6;
            eprintln!("[ Qwen3.5-Metal ] DeltaNet state: {:.1} MB ({} layers)", state_mb, n_dn);
            eprintln!("[ Qwen3.5-Metal ] KV cache alloc: {:.1} MB ({} layers)", kv_mb, n_fa);
        }
    }
}
