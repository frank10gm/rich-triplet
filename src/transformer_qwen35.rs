/// # Qwen 3.5 — inference-only architecture
///
/// Implements Qwen 3.5 dense models (4B, 9B) for text-only inference.
/// Weights loaded from GGUF files.
///
/// ## Architecture overview
///
/// Qwen 3.5 is a **hybrid** model: ~75% of layers use Gated DeltaNet
/// (linear attention / RNN) and ~25% use standard softmax attention.
///
/// | Feature                  | Gemma 3 (transformer4)   | Qwen 3.5               |
/// |--------------------------|--------------------------|-------------------------|
/// | Attention                | Softmax GQA everywhere   | Hybrid: DeltaNet + GQA  |
/// | Activation               | gelu_tanh                | silu                    |
/// | Norms per block          | 4                        | 2 (pre-norm)            |
/// | RMSNorm variant          | (1+gamma)                | (1+gamma)               |
/// | Embed scaling            | sqrt(hidden)             | none                    |
/// | RoPE                     | full head_dim            | partial (25%)           |
/// | Full attn output gate    | no                       | yes (sigmoid)           |
/// | DeltaNet layers          | n/a                      | conv1d + recurrent state|

use crate::autograd2::{Mat, MatBf16, TensorNode};
use crate::nn2::{Linear2, RmsNorm2};
use std::cell::RefCell;
use std::sync::Arc;

// ============================================================================
// ConfigQwen35
// ============================================================================

#[derive(Clone, Debug)]
pub struct ConfigQwen35 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    // Full attention config
    pub num_attention_heads: usize,   // 16 Q heads
    pub num_key_value_heads: usize,   // 4 KV heads
    pub head_dim: usize,              // 256
    // DeltaNet config
    pub linear_num_key_heads: usize,  // 16
    pub linear_num_value_heads: usize, // 32
    pub linear_key_head_dim: usize,   // 128
    pub linear_value_head_dim: usize, // 128
    pub linear_conv_kernel_dim: usize, // 4
    // MLP
    pub intermediate_size: usize,
    // Normalization
    pub rms_norm_eps: f32,
    // RoPE (for full attention layers only)
    pub rope_theta: f32,
    pub partial_rotary_factor: f32, // 0.25 → only first 64 of 256 dims
    // Architecture
    pub full_attention_interval: usize, // 4 → layers 3,7,11,...,31
    pub max_position_embeddings: usize,
    pub eos_token_id: usize,
    pub tie_word_embeddings: bool,
}

impl ConfigQwen35 {
    pub fn qwen35_4b() -> Self {
        ConfigQwen35 {
            vocab_size: 248320,
            hidden_size: 2560,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            intermediate_size: 9216,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            full_attention_interval: 4,
            max_position_embeddings: 262144,
            eos_token_id: 248044,
            tie_word_embeddings: true,
        }
    }

    pub fn qwen35_9b() -> Self {
        ConfigQwen35 {
            vocab_size: 248320,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            num_key_value_heads: 4,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: 32,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            intermediate_size: 12288,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            full_attention_interval: 4,
            max_position_embeddings: 262144,
            eos_token_id: 248044,
            tie_word_embeddings: false,
        }
    }

    pub fn qwen35_0_8b() -> Self {
        ConfigQwen35 {
            vocab_size: 248320,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            head_dim: 256,
            linear_num_key_heads: 16,
            linear_num_value_heads: 16,
            linear_key_head_dim: 128,
            linear_value_head_dim: 128,
            linear_conv_kernel_dim: 4,
            intermediate_size: 3584,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            partial_rotary_factor: 0.25,
            full_attention_interval: 4,
            max_position_embeddings: 262144,
            eos_token_id: 248044,
            tie_word_embeddings: true,
        }
    }

    pub fn is_full_attention_layer(&self, layer_idx: usize) -> bool {
        (layer_idx + 1) % self.full_attention_interval == 0
    }

    /// Total QKV projection size for DeltaNet: key_dim*2 + value_dim
    pub fn deltanet_qkv_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim * 2
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// RoPE dimension for full attention (partial)
    pub fn rope_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }
}

// ============================================================================
// Cache types
// ============================================================================

/// Cache for a DeltaNet layer: recurrent state + conv1d state.
pub struct DeltaNetState {
    /// Recurrent state: [num_v_heads * key_head_dim, value_head_dim]
    /// Each head h occupies rows [h*kd..(h+1)*kd], cols [0..vd]
    pub state: Vec<f32>,
    /// Conv1d state: [qkv_dim, kernel_size - 1]
    pub conv_state: Vec<f32>,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_dim: usize,
    pub conv_kernel: usize,
}

impl DeltaNetState {
    pub fn new(cfg: &ConfigQwen35) -> Self {
        let nv = cfg.linear_num_value_heads;
        let kd = cfg.linear_key_head_dim;
        let vd = cfg.linear_value_head_dim;
        let conv_dim = cfg.deltanet_qkv_dim();
        let ks = cfg.linear_conv_kernel_dim;
        DeltaNetState {
            state: vec![0.0; nv * kd * vd],
            conv_state: vec![0.0; conv_dim * (ks - 1)],
            num_v_heads: nv,
            key_head_dim: kd,
            value_head_dim: vd,
            conv_dim,
            conv_kernel: ks,
        }
    }

    /// Get mutable slice for head h's state matrix [kd, vd] in row-major.
    fn head_state_mut(&mut self, h: usize) -> &mut [f32] {
        let sz = self.key_head_dim * self.value_head_dim;
        &mut self.state[h * sz..(h + 1) * sz]
    }

    /// Get slice for head h's state matrix.
    fn head_state(&self, h: usize) -> &[f32] {
        let sz = self.key_head_dim * self.value_head_dim;
        &self.state[h * sz..(h + 1) * sz]
    }
}

/// KV cache for a full attention layer.
pub struct FullAttnKvCache {
    pub k: Mat, // [max_tokens, nkv * head_dim]
    pub v: Mat,
    pub seq_len: usize,
}

impl FullAttnKvCache {
    pub fn new(nkv: usize, head_dim: usize, max_tokens: usize) -> Self {
        FullAttnKvCache {
            k: Mat::zeros(max_tokens, nkv * head_dim),
            v: Mat::zeros(max_tokens, nkv * head_dim),
            seq_len: 0,
        }
    }

    pub fn append(&mut self, new_k: &Mat, new_v: &Mat) {
        let n_new = new_k.rows;
        let d = new_k.cols;
        for r in 0..n_new {
            for c in 0..d {
                *self.k.at_mut(self.seq_len + r, c) = new_k.at(r, c);
                *self.v.at_mut(self.seq_len + r, c) = new_v.at(r, c);
            }
        }
        self.seq_len += n_new;
    }
}

/// Per-layer cache: either DeltaNet state or full attention KV cache.
pub enum LayerCache {
    DeltaNet(DeltaNetState),
    FullAttn(FullAttnKvCache),
}

/// Full cache for the entire Qwen3.5 model.
pub struct Qwen35Cache {
    pub layers: Vec<RefCell<LayerCache>>,
}

impl Qwen35Cache {
    pub fn new(cfg: &ConfigQwen35, max_tokens: usize) -> Self {
        let max = max_tokens.min(cfg.max_position_embeddings);
        let layers = (0..cfg.num_hidden_layers)
            .map(|i| {
                if cfg.is_full_attention_layer(i) {
                    RefCell::new(LayerCache::FullAttn(FullAttnKvCache::new(
                        cfg.num_key_value_heads,
                        cfg.head_dim,
                        max,
                    )))
                } else {
                    RefCell::new(LayerCache::DeltaNet(DeltaNetState::new(cfg)))
                }
            })
            .collect();
        Qwen35Cache { layers }
    }
}

// ============================================================================
// Qwen35DeltaNet — Gated DeltaNet linear attention
// ============================================================================

pub struct Qwen35DeltaNet {
    pub in_proj_qkv: Linear2, // [hidden, qkv_dim]
    pub in_proj_z: Linear2,   // [hidden, value_dim] — output gate
    pub in_proj_a: Linear2,   // [hidden, num_v_heads] — alpha/decay gate
    pub in_proj_b: Linear2,   // [hidden, num_v_heads] — beta/write gate
    pub out_proj: Linear2,    // [value_dim, hidden]
    /// Conv1d depthwise weight: [qkv_dim, kernel_size] stored flat
    pub conv1d_weight: Vec<f32>,
    /// A_log: [num_v_heads] — log of base decay rate
    pub a_log: Vec<f32>,
    /// dt_bias: [num_v_heads] — bias for decay gate
    pub dt_bias: Vec<f32>,
    /// Output RMSNorm weight (zero-centered): [value_head_dim]
    pub norm_weight: Vec<f32>,
    // Config
    pub num_k_heads: usize,
    pub num_v_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub qkv_dim: usize,
    pub value_dim: usize,
}

impl Qwen35DeltaNet {
    pub fn new_for_inference(cfg: &ConfigQwen35) -> Self {
        let h = cfg.hidden_size;
        let qkv_dim = cfg.deltanet_qkv_dim();
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

        Qwen35DeltaNet {
            in_proj_qkv: Linear2::new_no_bias_zeros(h, qkv_dim),
            in_proj_z: Linear2::new_no_bias_zeros(h, value_dim),
            in_proj_a: Linear2::new_no_bias_zeros(h, cfg.linear_num_value_heads),
            in_proj_b: Linear2::new_no_bias_zeros(h, cfg.linear_num_value_heads),
            out_proj: Linear2::new_no_bias_zeros(value_dim, h),
            conv1d_weight: vec![0.0; qkv_dim * cfg.linear_conv_kernel_dim],
            a_log: vec![0.0; cfg.linear_num_value_heads],
            dt_bias: vec![0.0; cfg.linear_num_value_heads],
            norm_weight: vec![0.0; cfg.linear_value_head_dim],
            num_k_heads: cfg.linear_num_key_heads,
            num_v_heads: cfg.linear_num_value_heads,
            key_head_dim: cfg.linear_key_head_dim,
            value_head_dim: cfg.linear_value_head_dim,
            conv_kernel: cfg.linear_conv_kernel_dim,
            qkv_dim,
            value_dim,
        }
    }

    /// Forward pass for a single token (decode step).
    ///
    /// x: [1, hidden] → output: [1, hidden]
    pub fn forward_cached(&self, x: &TensorNode, state: &mut DeltaNetState) -> TensorNode {
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        let kd = self.key_head_dim;
        let vd = self.value_head_dim;
        let v_per_k = nv / nk; // 2

        // 1. Projections
        let qkv_tn = self.in_proj_qkv.forward(x); // [1, qkv_dim]
        let z_tn = self.in_proj_z.forward(x);      // [1, value_dim]
        let a_tn = self.in_proj_a.forward(x);      // [1, num_v_heads]
        let b_tn = self.in_proj_b.forward(x);      // [1, num_v_heads]

        let qkv_raw: Vec<f32> = qkv_tn.data().data.clone();
        let z_raw: Vec<f32> = z_tn.data().data.clone();
        let a_raw: Vec<f32> = a_tn.data().data.clone();
        let b_raw: Vec<f32> = b_tn.data().data.clone();

        // 2. Causal Conv1d + SiLU
        let qkv_conv = self.apply_conv1d(&qkv_raw, state);

        // 3. Split QKV: layout is [Q_all | K_all | V_all]
        // Q: [nk * kd], K: [nk * kd], V: [nv * vd]
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let mut q_flat = vec![0.0f32; key_dim];
        let mut k_flat = vec![0.0f32; key_dim];
        let mut v_flat = vec![0.0f32; value_dim];

        q_flat.copy_from_slice(&qkv_conv[0..key_dim]);
        k_flat.copy_from_slice(&qkv_conv[key_dim..key_dim * 2]);
        v_flat.copy_from_slice(&qkv_conv[key_dim * 2..key_dim * 2 + value_dim]);

        // 4. Compute gates
        let mut beta = vec![0.0f32; nv]; // write gate
        let mut g_decay = vec![0.0f32; nv]; // decay (negative)
        for h in 0..nv {
            beta[h] = sigmoid(b_raw[h]);
            // g = -exp(A_log) * softplus(a + dt_bias)
            g_decay[h] = -self.a_log[h].exp() * softplus(a_raw[h] + self.dt_bias[h]);
        }

        // 5. L2 normalize Q and K per head
        l2_normalize_heads(&mut q_flat, nk, kd);
        l2_normalize_heads(&mut k_flat, nk, kd);

        // 6. Repeat-interleave Q, K: nk → nv heads
        let mut q_exp = vec![0.0f32; nv * kd];
        let mut k_exp = vec![0.0f32; nv * kd];
        for g in 0..nk {
            for vi in 0..v_per_k {
                let dst = (g * v_per_k + vi) * kd;
                q_exp[dst..dst + kd].copy_from_slice(&q_flat[g * kd..(g + 1) * kd]);
                k_exp[dst..dst + kd].copy_from_slice(&k_flat[g * kd..(g + 1) * kd]);
            }
        }

        // 7. Scale Q by 1/sqrt(key_head_dim)
        let q_scale = 1.0 / (kd as f32).sqrt();
        for v in &mut q_exp {
            *v *= q_scale;
        }

        // 8. Recurrent state update per head
        let mut output = vec![0.0f32; nv * vd];

        for h in 0..nv {
            let s = state.head_state_mut(h); // [kd * vd]
            let k_h = &k_exp[h * kd..(h + 1) * kd];
            let v_h = &v_flat[h * vd..(h + 1) * vd];
            let q_h = &q_exp[h * kd..(h + 1) * kd];
            let decay = g_decay[h].exp(); // exp(g) where g < 0, so decay < 1
            let beta_h = beta[h];

            // S = S * decay
            for val in s.iter_mut() {
                *val *= decay;
            }

            // kv_mem = S^T @ k  →  kv_mem[j] = sum_i(S[i*vd+j] * k[i])
            // S is stored as [kd, vd] in row-major: S[i][j] = s[i*vd+j]
            let mut kv_mem = vec![0.0f32; vd];
            for i in 0..kd {
                let k_i = k_h[i];
                if k_i != 0.0 {
                    for j in 0..vd {
                        kv_mem[j] += s[i * vd + j] * k_i;
                    }
                }
            }

            // delta = (v - kv_mem) * beta
            // S += outer(k, delta) = k[i] * delta[j]
            for i in 0..kd {
                let k_i = k_h[i];
                if k_i != 0.0 {
                    for j in 0..vd {
                        let delta_j = (v_h[j] - kv_mem[j]) * beta_h;
                        s[i * vd + j] += k_i * delta_j;
                    }
                }
            }

            // output = S^T @ q  →  out[j] = sum_i(S[i*vd+j] * q[i])
            let out_h = &mut output[h * vd..(h + 1) * vd];
            for i in 0..kd {
                let q_i = q_h[i];
                if q_i != 0.0 {
                    for j in 0..vd {
                        out_h[j] += s[i * vd + j] * q_i;
                    }
                }
            }
        }

        // 9. Gated RMSNorm: rms_norm(output) * (1 + norm_weight) * silu(z)
        //    Applied per head (vd dims), shared norm_weight
        let eps = 1e-6f32;
        let mut gated_output = vec![0.0f32; nv * vd];
        for h in 0..nv {
            let out_h = &output[h * vd..(h + 1) * vd];
            let z_h = &z_raw[h * vd..(h + 1) * vd];

            // RMS norm
            let mut sum_sq = 0.0f32;
            for j in 0..vd {
                sum_sq += out_h[j] * out_h[j];
            }
            let rms = (sum_sq / vd as f32 + eps).sqrt();

            let dst = &mut gated_output[h * vd..(h + 1) * vd];
            for j in 0..vd {
                let normed = out_h[j] / rms;
                let scaled = normed * self.norm_weight[j];
                dst[j] = scaled * silu(z_h[j]);
            }
        }

        // 10. Output projection
        let out_mat = Mat::new(gated_output, 1, self.value_dim);
        self.out_proj.forward(&TensorNode::leaf(out_mat))
    }

    /// Batched prefill: process multiple tokens at once.
    /// x: [T, hidden] → output: [T, hidden]
    /// Batches projections as GEMM; loops for recurrent state update.
    pub fn forward_prefill(&self, x: &TensorNode, state: &mut DeltaNetState) -> TensorNode {
        let nk = self.num_k_heads;
        let nv = self.num_v_heads;
        let kd = self.key_head_dim;
        let vd = self.value_head_dim;
        let v_per_k = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let t = x.data().rows;

        // 1. Batch projections (GEMM)
        let qkv_all = self.in_proj_qkv.forward(x); // [T, qkv_dim]
        let z_all = self.in_proj_z.forward(x);      // [T, value_dim]
        let a_all = self.in_proj_a.forward(x);      // [T, nv]
        let b_all = self.in_proj_b.forward(x);      // [T, nv]

        let qkv_d = qkv_all.data();
        let z_d = z_all.data();
        let a_d = a_all.data();
        let b_d = b_all.data();

        // 2-8. Process each token sequentially (conv1d + recurrent)
        let mut all_gated = vec![0.0f32; t * value_dim];

        for tok in 0..t {
            let qkv_row = &qkv_d.data[tok * self.qkv_dim..(tok + 1) * self.qkv_dim];
            let z_row = &z_d.data[tok * value_dim..(tok + 1) * value_dim];
            let a_row = &a_d.data[tok * nv..(tok + 1) * nv];
            let b_row = &b_d.data[tok * nv..(tok + 1) * nv];

            // Conv1d + SiLU
            let qkv_conv = self.apply_conv1d(qkv_row, state);

            // Split QKV
            let mut q_flat = vec![0.0f32; key_dim];
            let mut k_flat = vec![0.0f32; key_dim];
            let mut v_flat = vec![0.0f32; value_dim];
            q_flat.copy_from_slice(&qkv_conv[0..key_dim]);
            k_flat.copy_from_slice(&qkv_conv[key_dim..key_dim * 2]);
            v_flat.copy_from_slice(&qkv_conv[key_dim * 2..key_dim * 2 + value_dim]);

            // Gates
            let mut beta = vec![0.0f32; nv];
            let mut g_decay = vec![0.0f32; nv];
            for h in 0..nv {
                beta[h] = sigmoid(b_row[h]);
                g_decay[h] = -self.a_log[h].exp() * softplus(a_row[h] + self.dt_bias[h]);
            }

            // L2 normalize
            l2_normalize_heads(&mut q_flat, nk, kd);
            l2_normalize_heads(&mut k_flat, nk, kd);

            // Repeat-interleave
            let mut q_exp = vec![0.0f32; nv * kd];
            let mut k_exp = vec![0.0f32; nv * kd];
            for g in 0..nk {
                for vi in 0..v_per_k {
                    let dst = (g * v_per_k + vi) * kd;
                    q_exp[dst..dst + kd].copy_from_slice(&q_flat[g * kd..(g + 1) * kd]);
                    k_exp[dst..dst + kd].copy_from_slice(&k_flat[g * kd..(g + 1) * kd]);
                }
            }

            // Scale Q
            let q_scale = 1.0 / (kd as f32).sqrt();
            for v in &mut q_exp { *v *= q_scale; }

            // Recurrent state update
            let mut output = vec![0.0f32; nv * vd];
            for h in 0..nv {
                let s = state.head_state_mut(h);
                let k_h = &k_exp[h * kd..(h + 1) * kd];
                let v_h = &v_flat[h * vd..(h + 1) * vd];
                let q_h = &q_exp[h * kd..(h + 1) * kd];
                let decay = g_decay[h].exp();
                let beta_h = beta[h];

                for val in s.iter_mut() { *val *= decay; }

                let mut kv_mem = vec![0.0f32; vd];
                for i in 0..kd {
                    let k_i = k_h[i];
                    if k_i != 0.0 {
                        for j in 0..vd { kv_mem[j] += s[i * vd + j] * k_i; }
                    }
                }

                for i in 0..kd {
                    let k_i = k_h[i];
                    if k_i != 0.0 {
                        for j in 0..vd {
                            s[i * vd + j] += k_i * (v_h[j] - kv_mem[j]) * beta_h;
                        }
                    }
                }

                let out_h = &mut output[h * vd..(h + 1) * vd];
                for i in 0..kd {
                    let q_i = q_h[i];
                    if q_i != 0.0 {
                        for j in 0..vd { out_h[j] += s[i * vd + j] * q_i; }
                    }
                }
            }

            // Gated RMSNorm per head
            let eps = 1e-6f32;
            let dst = &mut all_gated[tok * value_dim..(tok + 1) * value_dim];
            for h in 0..nv {
                let out_h = &output[h * vd..(h + 1) * vd];
                let z_h = &z_row[h * vd..(h + 1) * vd];
                let mut sum_sq = 0.0f32;
                for j in 0..vd { sum_sq += out_h[j] * out_h[j]; }
                let rms = (sum_sq / vd as f32 + eps).sqrt();
                for j in 0..vd {
                    let normed = out_h[j] / rms;
                    dst[h * vd + j] = normed * self.norm_weight[j] * silu(z_h[j]);
                }
            }
        }

        // 9. Batch output projection (GEMM)
        let gated_tn = TensorNode::leaf(Mat::new(all_gated, t, value_dim));
        self.out_proj.forward(&gated_tn)
    }

    /// Apply causal conv1d with state update. Returns conv_output after SiLU.
    fn apply_conv1d(&self, input: &[f32], state: &mut DeltaNetState) -> Vec<f32> {
        let dim = self.qkv_dim;
        let ks = self.conv_kernel;
        let hist = ks - 1; // 3
        let mut output = vec![0.0f32; dim];

        for c in 0..dim {
            // Compute convolution: weight[c,0]*state[0] + ... + weight[c,ks-2]*state[ks-2] + weight[c,ks-1]*input
            let w_base = c * ks;
            let s_base = c * hist;
            let mut val = 0.0f32;
            for t in 0..hist {
                val += self.conv1d_weight[w_base + t] * state.conv_state[s_base + t];
            }
            val += self.conv1d_weight[w_base + hist] * input[c];

            output[c] = silu(val);

            // Update state: shift left, append new input
            for t in 0..hist - 1 {
                state.conv_state[s_base + t] = state.conv_state[s_base + t + 1];
            }
            state.conv_state[s_base + hist - 1] = input[c];
        }

        output
    }
}

// ============================================================================
// Qwen35FullAttention — softmax GQA with output gate and partial RoPE
// ============================================================================

pub struct Qwen35FullAttention {
    pub q_proj: Linear2, // [hidden, num_heads * head_dim * 2] (Q + gate interleaved)
    pub k_proj: Linear2, // [hidden, num_kv_heads * head_dim]
    pub v_proj: Linear2, // [hidden, num_kv_heads * head_dim]
    pub o_proj: Linear2, // [num_heads * head_dim, hidden]
    pub q_norm: RmsNorm2, // [head_dim]
    pub k_norm: RmsNorm2, // [head_dim]
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rope_dim: usize, // partial RoPE dimension (64)
}

impl Qwen35FullAttention {
    pub fn new_for_inference(cfg: &ConfigQwen35) -> Self {
        let h = cfg.hidden_size;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;
        let d = cfg.head_dim;
        let rope_dim = cfg.rope_dim();

        Qwen35FullAttention {
            // Q proj outputs both query and gate: [hidden, nq * d * 2]
            q_proj: Linear2::new_no_bias_zeros(h, nq * d * 2),
            k_proj: Linear2::new_no_bias_zeros(h, nkv * d),
            v_proj: Linear2::new_no_bias_zeros(h, nkv * d),
            o_proj: Linear2::new_no_bias_zeros(nq * d, h),
            q_norm: RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            k_norm: RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim: d,
            rope_theta: cfg.rope_theta,
            rope_dim,
        }
    }

    /// Forward pass for a single token with KV cache.
    ///
    /// x: [1, hidden] → output: [1, hidden]
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut FullAttnKvCache) -> TensorNode {
        let d = self.head_dim;
        let nq = self.n_q_heads;
        let nkv = self.n_kv_heads;
        let seq_offset = cache.seq_len;

        // 1. Projections
        let qg_tn = self.q_proj.forward(x); // [1, nq * d * 2]
        let k_tn = self.k_proj.forward(x);  // [1, nkv * d]
        let v_tn = self.v_proj.forward(x);  // [1, nkv * d]

        let qg_data = qg_tn.data();
        let k_data = k_tn.data();
        let v_data = v_tn.data();

        // 2. Split Q and gate from q_proj output
        // Layout: [nq, d*2] → per head: first d = query, last d = gate
        let mut q_raw = vec![0.0f32; nq * d];
        let mut gate_raw = vec![0.0f32; nq * d];
        for h in 0..nq {
            let src_base = h * d * 2;
            for j in 0..d {
                q_raw[h * d + j] = qg_data.data[src_base + j];
                gate_raw[h * d + j] = qg_data.data[src_base + d + j];
            }
        }

        let mut k_raw: Vec<f32> = k_data.data.clone();

        // 3. Per-head RMSNorm on Q and K (zero-centered: (1+gamma) variant)
        apply_per_head_norm_raw(&mut q_raw, &self.q_norm, nq, d);
        apply_per_head_norm_raw(&mut k_raw, &self.k_norm, nkv, d);

        // 4. Partial RoPE: only first rope_dim dimensions
        let rope_dim = self.rope_dim;
        apply_partial_rope(&mut q_raw, nq, d, rope_dim, self.rope_theta, seq_offset);
        apply_partial_rope(&mut k_raw, nkv, d, rope_dim, self.rope_theta, seq_offset);

        // 5. Append K, V to cache
        let k_mat = Mat::new(k_raw, 1, nkv * d);
        let v_mat = Mat::new(v_data.data.clone(), 1, nkv * d);
        cache.append(&k_mat, &v_mat);

        // 6. GQA attention over full cache
        let attn_out = gqa_attention_cached(
            &q_raw, &cache.k, &cache.v,
            0, cache.seq_len,
            nq, nkv, d,
            1.0 / (d as f32).sqrt(),
        );

        // 7. Apply output gate: attn_out * sigmoid(gate)
        let mut gated = vec![0.0f32; nq * d];
        for i in 0..nq * d {
            gated[i] = attn_out[i] * sigmoid(gate_raw[i]);
        }

        // 8. Output projection
        let out_mat = Mat::new(gated, 1, nq * d);
        self.o_proj.forward(&TensorNode::leaf(out_mat))
    }

    /// Batched prefill: process multiple tokens at once with causal attention.
    /// x: [T, hidden] → output: [T, hidden]
    pub fn forward_prefill(&self, x: &TensorNode, cache: &mut FullAttnKvCache) -> TensorNode {
        let d = self.head_dim;
        let nq = self.n_q_heads;
        let nkv = self.n_kv_heads;
        let groups = nq / nkv;
        let t = x.data().rows;
        let start_pos = cache.seq_len;
        let scale = 1.0 / (d as f32).sqrt();

        // 1. Batch projections (GEMM)
        let qg_tn = self.q_proj.forward(x);  // [T, nq*d*2]
        let k_tn = self.k_proj.forward(x);   // [T, nkv*d]
        let v_tn = self.v_proj.forward(x);   // [T, nkv*d]
        let qg_d = qg_tn.data();
        let k_d = k_tn.data();
        let v_d = v_tn.data();

        // 2. Split Q/gate, normalize, RoPE per token
        let mut q_all = vec![0.0f32; t * nq * d];
        let mut gate_all = vec![0.0f32; t * nq * d];
        let mut k_all = vec![0.0f32; t * nkv * d];
        let v_all = v_d.data.clone();

        for tok in 0..t {
            // Split Q and gate
            let qg_off = tok * nq * d * 2;
            let q_off = tok * nq * d;
            for h in 0..nq {
                for j in 0..d {
                    q_all[q_off + h * d + j] = qg_d.data[qg_off + h * d * 2 + j];
                    gate_all[q_off + h * d + j] = qg_d.data[qg_off + h * d * 2 + d + j];
                }
            }

            // Per-head RMSNorm Q
            let q_slice = &mut q_all[q_off..q_off + nq * d];
            apply_per_head_norm_raw(q_slice, &self.q_norm, nq, d);

            // Per-head RMSNorm K
            let k_off = tok * nkv * d;
            k_all[k_off..k_off + nkv * d].copy_from_slice(
                &k_d.data[k_off..k_off + nkv * d],
            );
            let k_slice = &mut k_all[k_off..k_off + nkv * d];
            apply_per_head_norm_raw(k_slice, &self.k_norm, nkv, d);

            // Partial RoPE
            let pos = start_pos + tok;
            apply_partial_rope(
                &mut q_all[q_off..q_off + nq * d],
                nq, d, self.rope_dim, self.rope_theta, pos,
            );
            apply_partial_rope(
                &mut k_all[k_off..k_off + nkv * d],
                nkv, d, self.rope_dim, self.rope_theta, pos,
            );
        }

        // 3. Store all K, V in cache
        for tok in 0..t {
            let k_off = tok * nkv * d;
            let k_mat = Mat::new(k_all[k_off..k_off + nkv * d].to_vec(), 1, nkv * d);
            let v_mat = Mat::new(v_all[tok * nkv * d..(tok + 1) * nkv * d].to_vec(), 1, nkv * d);
            cache.append(&k_mat, &v_mat);
        }

        // 4. Causal attention: each query attends to all keys up to its position
        let mut attn_out = vec![0.0f32; t * nq * d];
        for h in 0..nq {
            let kv_h = h / groups;
            for qi in 0..t {
                let cache_end = start_pos + qi + 1; // attend to positions 0..=qi+start_pos
                let mut scores = vec![0.0f32; cache_end];

                // Compute Q @ K^T for all cached keys
                let q_off = qi * nq * d + h * d;
                for ki in 0..cache_end {
                    let k_row = &cache.k.data[ki * nkv * d + kv_h * d..];
                    let mut dot = 0.0f32;
                    for j in 0..d {
                        dot += q_all[q_off + j] * k_row[j];
                    }
                    scores[ki] = dot * scale;
                }

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                for s in &mut scores { *s /= sum_exp; }

                // Weighted sum of values
                let out_off = qi * nq * d + h * d;
                for ki in 0..cache_end {
                    let v_row = &cache.v.data[ki * nkv * d + kv_h * d..];
                    let w = scores[ki];
                    for j in 0..d {
                        attn_out[out_off + j] += w * v_row[j];
                    }
                }
            }
        }

        // 5. Apply output gate: attn_out * sigmoid(gate)
        let mut gated = vec![0.0f32; t * nq * d];
        for i in 0..t * nq * d {
            gated[i] = attn_out[i] * sigmoid(gate_all[i]);
        }

        // 6. Batch output projection (GEMM)
        let out_mat = Mat::new(gated, t, nq * d);
        self.o_proj.forward(&TensorNode::leaf(out_mat))
    }
}

// ============================================================================
// Qwen35Mlp — SwiGLU with SiLU activation
// ============================================================================

pub struct Qwen35Mlp {
    pub gate_proj: Linear2, // [hidden, intermediate]
    pub up_proj: Linear2,   // [hidden, intermediate]
    pub down_proj: Linear2, // [intermediate, hidden]
}

impl Qwen35Mlp {
    pub fn new_for_inference(cfg: &ConfigQwen35) -> Self {
        Qwen35Mlp {
            gate_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            up_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            down_proj: Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size),
        }
    }

    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let gate = self.gate_proj.forward(x).silu();
        let up = self.up_proj.forward(x);
        let hidden = gate.mul_elem_node(&up);
        self.down_proj.forward(&hidden)
    }
}

// ============================================================================
// Qwen35Block — one transformer block (hybrid: DeltaNet or FullAttn)
// ============================================================================

pub enum TokenMixer {
    DeltaNet(Qwen35DeltaNet),
    FullAttn(Qwen35FullAttention),
}

pub struct Qwen35Block {
    pub input_layernorm: RmsNorm2,
    pub token_mixer: TokenMixer,
    pub post_attention_layernorm: RmsNorm2,
    pub mlp: Qwen35Mlp,
}

impl Qwen35Block {
    pub fn new_for_inference(cfg: &ConfigQwen35, layer_idx: usize) -> Self {
        let mixer = if cfg.is_full_attention_layer(layer_idx) {
            TokenMixer::FullAttn(Qwen35FullAttention::new_for_inference(cfg))
        } else {
            TokenMixer::DeltaNet(Qwen35DeltaNet::new_for_inference(cfg))
        };

        Qwen35Block {
            input_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            token_mixer: mixer,
            post_attention_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            mlp: Qwen35Mlp::new_for_inference(cfg),
        }
    }

    /// Cached forward: x [1, hidden] → [1, hidden]
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut LayerCache) -> TensorNode {
        // Pre-norm → token mixer → residual
        let normed = self.input_layernorm.forward_gemma3(x); // (1+gamma) variant
        let attn = match (&self.token_mixer, cache) {
            (TokenMixer::DeltaNet(dn), LayerCache::DeltaNet(state)) => {
                dn.forward_cached(&normed, state)
            }
            (TokenMixer::FullAttn(fa), LayerCache::FullAttn(kv)) => {
                fa.forward_cached(&normed, kv)
            }
            _ => panic!("Layer/cache type mismatch"),
        };
        let x2 = x.add(&attn);

        // Pre-norm → MLP → residual
        let normed2 = self.post_attention_layernorm.forward_gemma3(&x2);
        let mlp_out = self.mlp.forward(&normed2);
        x2.add(&mlp_out)
    }

    /// Batched prefill: x [T, hidden] → [T, hidden]
    pub fn forward_prefill(&self, x: &TensorNode, cache: &mut LayerCache) -> TensorNode {
        let normed = self.input_layernorm.forward_gemma3(x);
        let attn = match (&self.token_mixer, cache) {
            (TokenMixer::DeltaNet(dn), LayerCache::DeltaNet(state)) => {
                dn.forward_prefill(&normed, state)
            }
            (TokenMixer::FullAttn(fa), LayerCache::FullAttn(kv)) => {
                fa.forward_prefill(&normed, kv)
            }
            _ => panic!("Layer/cache type mismatch"),
        };
        let x2 = x.add(&attn);

        let normed2 = self.post_attention_layernorm.forward_gemma3(&x2);
        let mlp_out = self.mlp.forward(&normed2);
        x2.add(&mlp_out)
    }
}

// ============================================================================
// Qwen35Model — full model
// ============================================================================

pub struct Qwen35Model {
    pub embed_tokens: TensorNode,
    pub embed_bf16: Option<MatBf16>,
    pub layers: Vec<Qwen35Block>,
    pub norm: RmsNorm2,
    pub lm_head: Linear2,
    pub config: ConfigQwen35,
}

impl Qwen35Model {
    pub fn new_for_inference(cfg: ConfigQwen35) -> Self {
        let embed = TensorNode::leaf(Mat::zeros(0, 0));
        let layers: Vec<Qwen35Block> = (0..cfg.num_hidden_layers)
            .map(|i| Qwen35Block::new_for_inference(&cfg, i))
            .collect();
        let norm = RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps);
        let lm_head = Linear2 {
            weight: embed.clone(),
            bias: TensorNode::leaf(Mat::zeros(0, 0)),
            in_features: cfg.hidden_size,
            out_features: cfg.vocab_size,
            q4_weight: None,
            q4k_weight: None,
            bf16_weight: None,
        };
        Qwen35Model {
            embed_tokens: embed,
            embed_bf16: None,
            layers,
            norm,
            lm_head,
            config: cfg,
        }
    }

    // -------------------------------------------------------------------------
    // Generation with cache
    // -------------------------------------------------------------------------

    pub fn generate_cached_streaming(
        &mut self,
        token_ids: &[usize],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        debug: bool,
        mut callback: impl FnMut(usize),
    ) {
        use crate::transformer3::SamplingParams;
        use crate::transformer4::LcgRng;

        let params = SamplingParams {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            eos_token_id: Some(self.config.eos_token_id),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };

        let h = self.config.hidden_size;
        let mut rng = LcgRng::new(seed);
        let mut seen: Vec<usize> = Vec::new();

        // Allocate cache
        let max_tokens = token_ids.len() + max_new;
        let mut cache = Qwen35Cache::new(&self.config, max_tokens);

        // Track seq position for full attention layers
        let mut _seq_pos = 0usize;

        // ----- Prefill: batched (all prompt tokens at once) -----
        let prefill_start = std::time::Instant::now();

        // Embed all tokens into [T, h] matrix
        let t_len = token_ids.len();
        let mut embed_data = Vec::with_capacity(t_len * h);
        for &tok in token_ids {
            embed_data.extend_from_slice(&self.embed_token(tok));
        }
        let mut x = TensorNode::leaf(Mat::new(embed_data, t_len, h));

        if debug {
            let xd = x.data();
            let last_row: Vec<f32> = (0..h).map(|c| xd.at(t_len - 1, c)).collect();
            let rms: f32 = (last_row.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
            eprintln!("[ Qwen3.5-dbg ] embed rms={:.4} first5={:.4?}", rms, &last_row[..5]);
        }

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_prefill(&x, &mut cache.layers[i].borrow_mut());
            if debug && (i < 3 || i == self.layers.len() - 1) {
                let xd = x.data();
                let last_row: Vec<f32> = (0..h).map(|c| xd.at(t_len - 1, c)).collect();
                let rms: f32 = (last_row.iter().map(|v| v * v).sum::<f32>() / h as f32).sqrt();
                let kind = if self.config.is_full_attention_layer(i) { "full" } else { "delta" };
                eprintln!("[ Qwen3.5-dbg ] layer {} ({}) h_rms={:.4}", i, kind, rms);
            }
        }

        // Extract last row [1, h] for logits
        {
            let xd = x.data();
            let last_row: Vec<f32> = (0..h).map(|c| xd.at(t_len - 1, c)).collect();
            let x_last = TensorNode::leaf(Mat::new(last_row, 1, h));
            let normed = self.norm.forward_gemma3(&x_last);
            let logits_node = self.lm_head.forward(&normed);
            let logits = logits_node.data().clone();

            let prefill_ms = prefill_start.elapsed().as_millis();
            let prefill_tps = token_ids.len() as f64 / prefill_start.elapsed().as_secs_f64();
            eprintln!(
                "[ Qwen3.5 ] Prefill: {} tokens in {:.0} ms ({:.1} tok/s)",
                token_ids.len(), prefill_ms, prefill_tps
            );

            if debug {
                let lv = &logits.data;
                let mut indexed: Vec<(usize, f32)> =
                    lv.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
                let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
                eprintln!("[ Qwen3.5-dbg ] prefill top-5: {:?}", top5);
            }

            let first = crate::transformer4::sample_token(
                &logits, 0, &params, &seen, &mut rng,
            );
            callback(first);
            seen.push(first);
            if first == self.config.eos_token_id || first == 248046 {
                return;
            }

            // ── Metal GPU decode path ──
            #[cfg(feature = "metal")]
            let use_metal = true;
            #[cfg(not(feature = "metal"))]
            let use_metal = false;

            #[cfg(feature = "metal")]
            let metal_ctx = {
                let ctx = crate::metal_decode_qwen35::inner::MetalDecodeContextQwen35::new(self);
                ctx.print_memory_stats();
                // Warmup to trigger shader JIT
                let t_warmup = std::time::Instant::now();
                let _ = ctx.decode_step(0, 0);
                eprintln!("[ Qwen3.5-Metal ] GPU warmup in {:.0} ms", t_warmup.elapsed().as_millis());
                // Sync state from CPU prefill
                ctx.sync_state_from_cpu(&cache);
                ctx
            };

            let decode_start = std::time::Instant::now();
            let mut prev = first;
            let mut n_decoded = 1usize;
            let mut metal_seq_len = token_ids.len();

            for _ in 1..max_new {
                #[cfg(feature = "metal")]
                let logits = if use_metal {
                    let logits_vec = metal_ctx.decode_step(prev, metal_seq_len);
                    metal_seq_len += 1;
                    Mat::new(logits_vec.clone(), 1, logits_vec.len())
                } else {
                    let x_data = self.embed_token(prev);
                    let mut x = TensorNode::leaf(Mat::new(x_data, 1, h));
                    for (i, layer) in self.layers.iter().enumerate() {
                        x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
                    }
                    let normed = self.norm.forward_gemma3(&x);
                    let logits_node = self.lm_head.forward(&normed);
                    logits_node.data().clone()
                };

                #[cfg(not(feature = "metal"))]
                let logits = {
                    let x_data = self.embed_token(prev);
                    let mut x = TensorNode::leaf(Mat::new(x_data, 1, h));
                    for (i, layer) in self.layers.iter().enumerate() {
                        x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
                    }
                    let normed = self.norm.forward_gemma3(&x);
                    let logits_node = self.lm_head.forward(&normed);
                    logits_node.data().clone()
                };

                prev = crate::transformer4::sample_token(
                    &logits, 0, &params, &seen, &mut rng,
                );
                callback(prev);
                seen.push(prev);
                n_decoded += 1;

                if prev == self.config.eos_token_id || prev == 248046 {
                    break;
                }
            }

            let decode_elapsed = decode_start.elapsed();
            let decode_tps = n_decoded as f64 / decode_elapsed.as_secs_f64();
            let ms_per_tok = decode_elapsed.as_millis() as f64 / n_decoded as f64;
            let backend = if use_metal { "Metal" } else { "CPU" };
            eprintln!(
                "[ Qwen3.5 ] Decode ({}): {} tokens in {:.0} ms ({:.1} tok/s, {:.0} ms/tok)",
                backend,
                n_decoded,
                decode_elapsed.as_millis(),
                decode_tps,
                ms_per_tok
            );
        }
    }

    /// Embed a single token ID → [hidden_size] f32 vector (no scaling for Qwen3.5).
    fn embed_token(&self, tok: usize) -> Vec<f32> {
        let h = self.config.hidden_size;
        if let Some(ref bf16) = self.embed_bf16 {
            (0..h)
                .map(|c| MatBf16::bf16_to_f32(bf16.data[tok * h + c]))
                .collect()
        } else {
            let te = self.embed_tokens.data();
            (0..h).map(|c| te.at(tok, c)).collect()
        }
    }

    // -------------------------------------------------------------------------
    // GGUF weight loading
    // -------------------------------------------------------------------------

    pub fn load_weights_from_gguf(&mut self, path: &str) -> std::io::Result<()> {
        use crate::autograd2::MatBf16;
        use crate::gguf_loader::{GgufFile, GgufType};

        eprintln!("[ GGUF ] Opening {}...", path);
        let gguf = GgufFile::open(path)?;
        eprintln!("[ GGUF ] Found {} tensors.", gguf.tensor_info.len());

        if let Some(arch) = gguf.metadata.get("general.architecture").and_then(|v| v.as_str()) {
            eprintln!("[ GGUF ] Architecture: {}", arch);
        }

        let n_tensors = gguf.tensor_info.len();
        let mut loaded = 0usize;
        let mut lm_head_explicitly_loaded = false;

        // Helper: load tensor as f32 vec (handles F32/F16/Q8_0)
        let load_f32 = |gguf: &GgufFile, idx: usize| -> std::io::Result<Vec<f32>> {
            match gguf.tensor_info[idx].gguf_type {
                GgufType::F32 => gguf.decode_f32(idx),
                GgufType::F16 => gguf.decode_f16_to_f32(idx),
                GgufType::Q8_0 => gguf.decode_q8_0_to_f32(idx),
                GgufType::Q6K => gguf.decode_q6k_to_f32(idx),
                _ => Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    format!("expected f32/f16/q8_0 for {}", gguf.tensor_info[idx].name),
                )),
            }
        };

        for idx in 0..n_tensors {
            let name = gguf.tensor_info[idx].name.clone();
            let gtype = gguf.tensor_info[idx].gguf_type;

            // ---- token embedding ----
            if name == "token_embd.weight" {
                let shape = gguf.tensor_info[idx].shape.clone();
                let (vocab, hidden) = if shape.len() == 2 { (shape[1], shape[0]) } else { (shape[0], 1) };
                eprintln!("[ GGUF ] token_embd: type={:?} vocab={} hidden={}", gtype, vocab, hidden);
                let bits = match gtype {
                    GgufType::Bf16 => gguf.decode_bf16(idx)?,
                    GgufType::F16 => crate::transformer4::f32s_to_bf16_and_drop(gguf.decode_f16_to_f32(idx)?),
                    GgufType::F32 => crate::transformer4::f32s_to_bf16_and_drop(gguf.decode_f32(idx)?),
                    GgufType::Q4K => crate::transformer4::f32s_to_bf16_and_drop(gguf.decode_q4k_to_f32(idx)?),
                    GgufType::Q6K => crate::transformer4::f32s_to_bf16_and_drop(gguf.decode_q6k_to_f32(idx)?),
                    GgufType::Q4_0 => crate::transformer4::f32s_to_bf16_and_drop(gguf.decode_q4_0_to_f32(idx)?),
                    _ => {
                        eprintln!("[ GGUF ] Warning: unsupported embed type {:?}", gtype);
                        continue;
                    }
                };
                crate::transformer4::model_set_embed_bf16_raw(
                    &mut self.embed_tokens, &mut self.embed_bf16, &mut self.lm_head,
                    bits, vocab, hidden, self.config.tie_word_embeddings,
                );
                loaded += 1;
                continue;
            }

            // ---- lm_head (separate, for non-tied models) ----
            if name == "output.weight" {
                eprintln!("[ GGUF ] Loading explicit output.weight for lm_head (type={:?})", gtype);
                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut self.lm_head)?;
                lm_head_explicitly_loaded = true;
                loaded += 1;
                continue;
            }

            // ---- output norm ----
            if name == "output_norm.weight" {
                let f32s = load_f32(&gguf, idx)?;
                let n = f32s.len();
                // GGUF stores (1 + HF_w); subtract 1 for (1+gamma) variant
                let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                self.norm.gamma.set_data(Mat::new(adjusted, 1, n));
                loaded += 1;
                continue;
            }

            // ---- per-layer tensors: blk.{i}.* ----
            if let Some(rest) = name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    let layer_idx: usize = match rest[..dot].parse() {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if layer_idx >= self.layers.len() {
                        continue;
                    }
                    let tensor_name = &rest[dot + 1..];
                    let layer = &mut self.layers[layer_idx];

                    match tensor_name {
                        // ---- layer norms ----
                        "attn_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            let adj: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer.input_layernorm.gamma.set_data(Mat::new(adj, 1, n));
                            loaded += 1;
                        }
                        "ffn_norm.weight" | "post_attention_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            let adj: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer.post_attention_layernorm.gamma.set_data(Mat::new(adj, 1, n));
                            loaded += 1;
                        }

                        // ---- MLP weights (all layers) ----
                        "ffn_gate.weight" => {
                            crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut layer.mlp.gate_proj)?;
                            loaded += 1;
                        }
                        "ffn_up.weight" => {
                            crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut layer.mlp.up_proj)?;
                            loaded += 1;
                        }
                        "ffn_down.weight" => {
                            crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut layer.mlp.down_proj)?;
                            loaded += 1;
                        }

                        // ---- Full attention layer weights ----
                        "attn_q.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut fa.q_proj)?;
                            }
                            loaded += 1;
                        }
                        "attn_k.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut fa.k_proj)?;
                            }
                            loaded += 1;
                        }
                        "attn_v.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut fa.v_proj)?;
                            }
                            loaded += 1;
                        }
                        "attn_output.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut fa.o_proj)?;
                            }
                            loaded += 1;
                        }
                        "attn_q_norm.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                let f32s = load_f32(&gguf, idx)?;
                                let n = f32s.len();
                                let adj: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                                fa.q_norm.gamma.set_data(Mat::new(adj, 1, n));
                            }
                            loaded += 1;
                        }
                        "attn_k_norm.weight" => {
                            if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                                let f32s = load_f32(&gguf, idx)?;
                                let n = f32s.len();
                                let adj: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                                fa.k_norm.gamma.set_data(Mat::new(adj, 1, n));
                            }
                            loaded += 1;
                        }

                        // ---- DeltaNet layer weights ----
                        "ssm_in.weight" | "attn_qkv.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut dn.in_proj_qkv)?;
                            }
                            loaded += 1;
                        }
                        "ssm_gate.weight" | "attn_gate.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut dn.in_proj_z)?;
                            }
                            loaded += 1;
                        }
                        "ssm_out.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut dn.out_proj)?;
                            }
                            loaded += 1;
                        }
                        "ssm_alpha.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut dn.in_proj_a)?;
                            }
                            loaded += 1;
                        }
                        "ssm_beta.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                crate::transformer4::load_linear_from_gguf(&gguf, idx, &mut dn.in_proj_b)?;
                            }
                            loaded += 1;
                        }
                        "ssm_a.weight" | "ssm_a" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                let f32s = load_f32(&gguf, idx)?;
                                dn.a_log = f32s;
                            }
                            loaded += 1;
                        }
                        "ssm_dt.bias" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                let f32s = load_f32(&gguf, idx)?;
                                dn.dt_bias = f32s;
                            }
                            loaded += 1;
                        }
                        "ssm_norm.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                let f32s = load_f32(&gguf, idx)?;
                                // Gated RMSNorm uses weight directly (init=1.0), NOT (1+gamma).
                                // GGUF should store raw HF values (≈1.0). No adjustment.
                                dn.norm_weight = f32s;
                            }
                            loaded += 1;
                        }
                        "ssm_conv1d.weight" => {
                            if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                                // Conv1d weight shape in GGUF: [qkv_dim, 1, kernel_size]
                                // We store flat as [qkv_dim * kernel_size]
                                let f32s = load_f32(&gguf, idx)?;
                                dn.conv1d_weight = f32s;
                            }
                            loaded += 1;
                        }

                        other => {
                            eprintln!("[ GGUF ] Unknown blk tensor: blk.{}.{}", layer_idx, other);
                        }
                    }
                }
            }
        }

        // If tie_word_embeddings and no explicit lm_head, copy embed to lm_head
        if self.config.tie_word_embeddings && !lm_head_explicitly_loaded {
            if let Some(ref bf16) = self.embed_bf16 {
                self.lm_head.bf16_weight = Some(MatBf16 {
                    data: bf16.data.clone(),
                    rows: bf16.rows,
                    cols: bf16.cols,
                });
                eprintln!("[ GGUF ] lm_head weight-tied to embed_tokens (BF16)");
            }
        }

        eprintln!(
            "[ GGUF ] Loaded {} / {} tensors for Qwen3.5-{}",
            loaded,
            n_tensors,
            if self.config.hidden_size == 2560 { "4B" } else { "9B" }
        );
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Safetensors weight loading
    // -------------------------------------------------------------------------

    pub fn load_weights_from_dir(&mut self, dir: &str) -> Result<(), String> {
        let entries =
            std::fs::read_dir(dir).map_err(|e| format!("cannot read dir {}: {}", dir, e))?;

        let mut loaded_shards = 0usize;
        let mut loaded_tensors = 0usize;
        let mut matched = 0usize;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }

            let mut file = std::fs::File::open(&path)
                .map_err(|e| format!("cannot open {:?}: {}", path, e))?;
            let (data_start, tensor_entries) =
                crate::transformer3::parse_safetensors_header(&mut file)
                    .map_err(|e| format!("parse error in {:?}: {}", path, e))?;

            for te in &tensor_entries {
                let t = crate::transformer3::read_safetensor_from_file(&mut file, data_start, te)
                    .map_err(|e| format!("read error in {:?}: {}", path, e))?;
                loaded_tensors += 1;
                if self.apply_safetensor(t) {
                    matched += 1;
                }
            }
            loaded_shards += 1;
        }

        if loaded_shards == 0 {
            return Err(format!("no .safetensors files found in {}", dir));
        }

        // If tie_word_embeddings and lm_head has no weight, copy from embed
        if self.config.tie_word_embeddings && self.lm_head.bf16_weight.is_none() {
            if let Some(ref bf16) = self.embed_bf16 {
                self.lm_head.bf16_weight = Some(MatBf16 {
                    data: bf16.data.clone(),
                    rows: bf16.rows,
                    cols: bf16.cols,
                });
            }
        }

        eprintln!(
            "Qwen3.5: loaded {} tensors ({} matched) from {} shards in {}",
            loaded_tensors, matched, loaded_shards, dir
        );
        Ok(())
    }

    /// Apply a single safetensor to the model. Returns true if matched.
    fn apply_safetensor(&mut self, t: crate::transformer3::SafeTensor) -> bool {
        use crate::autograd2::MatBf16;

        let name = &t.name;

        // Strip "model.language_model." or "model." prefix
        let inner = name
            .strip_prefix("model.language_model.")
            .or_else(|| name.strip_prefix("model."));
        let inner = match inner {
            Some(s) => s,
            None => return false,
        };

        // Helper: get f32 data from safetensor (BF16 → f32 if needed)
        let get_f32 = |t: &crate::transformer3::SafeTensor| -> Vec<f32> {
            if let Some(ref bits) = t.bf16_data {
                bits.iter().map(|&b| MatBf16::bf16_to_f32(b)).collect()
            } else {
                t.data.clone()
            }
        };

        // Helper: set linear weight from safetensor
        let set_linear = |linear: &mut Linear2, t: crate::transformer3::SafeTensor| {
            let (rows, cols) = if t.shape.len() >= 2 {
                (t.shape[0], t.shape[1])
            } else {
                (1, t.shape[0])
            };
            if let Some(bits) = t.bf16_data {
                linear.load_bf16(bits, rows, cols);
            } else {
                linear.weight.set_data(Mat::new(t.data, rows, cols));
            }
        };

        // ---- embed_tokens ----
        if inner == "embed_tokens.weight" {
            let (vocab, hidden) = (t.shape[0], t.shape[1]);
            eprintln!("[ Safetensors ] embed_tokens: vocab={} hidden={}", vocab, hidden);
            let bits = if let Some(bits) = t.bf16_data {
                bits
            } else {
                crate::transformer4::f32s_to_bf16_and_drop(t.data)
            };
            crate::transformer4::model_set_embed_bf16_raw(
                &mut self.embed_tokens, &mut self.embed_bf16, &mut self.lm_head,
                bits, vocab, hidden, self.config.tie_word_embeddings,
            );
            return true;
        }

        // ---- final norm ----
        if inner == "norm.weight" {
            let f32s = get_f32(&t);
            let n = f32s.len();
            // Safetensors stores raw gamma (zeros-init) — no adjustment needed.
            // forward_gemma3 applies (1+gamma) so gamma=0 gives identity scaling.
            self.norm.gamma.set_data(Mat::new(f32s, 1, n));
            return true;
        }

        // ---- lm_head (for non-tied models) ----
        if inner == "lm_head.weight" {
            let (rows, cols) = (t.shape[0], t.shape[1]);
            if let Some(bits) = t.bf16_data {
                self.lm_head.load_bf16(bits, rows, cols);
            } else {
                self.lm_head.weight.set_data(Mat::new(t.data, rows, cols));
            }
            return true;
        }

        // ---- per-layer tensors: layers.{i}.* ----
        if let Some(rest) = inner.strip_prefix("layers.") {
            if let Some(dot) = rest.find('.') {
                let layer_idx: usize = match rest[..dot].parse() {
                    Ok(v) => v,
                    Err(_) => return false,
                };
                if layer_idx >= self.layers.len() {
                    return false;
                }
                let tensor_name = &rest[dot + 1..];
                let layer = &mut self.layers[layer_idx];

                match tensor_name {
                    // ---- layer norms (safetensors: raw gamma, no adjustment) ----
                    "input_layernorm.weight" => {
                        let f32s = get_f32(&t);
                        let n = f32s.len();
                        layer.input_layernorm.gamma.set_data(Mat::new(f32s, 1, n));
                        return true;
                    }
                    "post_attention_layernorm.weight" => {
                        let f32s = get_f32(&t);
                        let n = f32s.len();
                        layer.post_attention_layernorm.gamma.set_data(Mat::new(f32s, 1, n));
                        return true;
                    }

                    // ---- MLP ----
                    "mlp.gate_proj.weight" => {
                        set_linear(&mut layer.mlp.gate_proj, t);
                        return true;
                    }
                    "mlp.up_proj.weight" => {
                        set_linear(&mut layer.mlp.up_proj, t);
                        return true;
                    }
                    "mlp.down_proj.weight" => {
                        set_linear(&mut layer.mlp.down_proj, t);
                        return true;
                    }

                    // ---- Full attention ----
                    "self_attn.q_proj.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            set_linear(&mut fa.q_proj, t);
                        }
                        return true;
                    }
                    "self_attn.k_proj.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            set_linear(&mut fa.k_proj, t);
                        }
                        return true;
                    }
                    "self_attn.v_proj.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            set_linear(&mut fa.v_proj, t);
                        }
                        return true;
                    }
                    "self_attn.o_proj.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            set_linear(&mut fa.o_proj, t);
                        }
                        return true;
                    }
                    "self_attn.q_norm.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            let f32s = get_f32(&t);
                            let n = f32s.len();
                            fa.q_norm.gamma.set_data(Mat::new(f32s, 1, n));
                        }
                        return true;
                    }
                    "self_attn.k_norm.weight" => {
                        if let TokenMixer::FullAttn(ref mut fa) = layer.token_mixer {
                            let f32s = get_f32(&t);
                            let n = f32s.len();
                            fa.k_norm.gamma.set_data(Mat::new(f32s, 1, n));
                        }
                        return true;
                    }

                    // ---- DeltaNet ----
                    "linear_attn.in_proj_qkv.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            set_linear(&mut dn.in_proj_qkv, t);
                        }
                        return true;
                    }
                    "linear_attn.in_proj_z.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            set_linear(&mut dn.in_proj_z, t);
                        }
                        return true;
                    }
                    "linear_attn.in_proj_a.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            set_linear(&mut dn.in_proj_a, t);
                        }
                        return true;
                    }
                    "linear_attn.in_proj_b.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            set_linear(&mut dn.in_proj_b, t);
                        }
                        return true;
                    }
                    "linear_attn.out_proj.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            set_linear(&mut dn.out_proj, t);
                        }
                        return true;
                    }
                    "linear_attn.A_log" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            dn.a_log = get_f32(&t);
                        }
                        return true;
                    }
                    "linear_attn.dt_bias" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            dn.dt_bias = get_f32(&t);
                        }
                        return true;
                    }
                    "linear_attn.norm.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            let f32s = get_f32(&t);
                            // Safetensors: raw gamma, no adjustment needed
                            dn.norm_weight = f32s;
                        }
                        return true;
                    }
                    "linear_attn.conv1d.weight" => {
                        if let TokenMixer::DeltaNet(ref mut dn) = layer.token_mixer {
                            // Conv1d weight: [qkv_dim, 1, kernel_size] → flatten to [qkv_dim * kernel_size]
                            dn.conv1d_weight = get_f32(&t);
                        }
                        return true;
                    }

                    _ => {
                        // Skip vision/mtp/other tensors silently
                    }
                }
            }
        }

        false
    }
}

// ============================================================================
// Helper functions
// ============================================================================

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x // avoid overflow
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// L2 normalize each head in-place: x[h*dim..(h+1)*dim] /= ||x_h||_2
fn l2_normalize_heads(x: &mut [f32], n_heads: usize, dim: usize) {
    let eps = 1e-12f32;
    for h in 0..n_heads {
        let slice = &mut x[h * dim..(h + 1) * dim];
        let norm: f32 = slice.iter().map(|v| v * v).sum::<f32>().sqrt().max(eps);
        for v in slice.iter_mut() {
            *v /= norm;
        }
    }
}

/// Apply per-head RMSNorm (zero-centered: (1+gamma) variant) in-place.
fn apply_per_head_norm_raw(x: &mut [f32], norm: &RmsNorm2, n_heads: usize, dim: usize) {
    let gamma = norm.gamma.data();
    let eps = norm.eps;
    for h in 0..n_heads {
        let slice = &mut x[h * dim..(h + 1) * dim];
        let sum_sq: f32 = slice.iter().map(|v| v * v).sum();
        let rms = (sum_sq / dim as f32 + eps).sqrt();
        for j in 0..dim {
            slice[j] = (slice[j] / rms) * (1.0 + gamma.at(0, j));
        }
    }
}

/// Apply partial RoPE: only the first `rope_dim` dimensions of each head.
/// Uses standard rotary encoding (not NeoX half-split — Qwen3.5 uses interleaved for partial RoPE).
fn apply_partial_rope(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    pos: usize,
) {
    let half = rope_dim / 2;
    for h in 0..n_heads {
        let base = h * head_dim;
        for i in 0..half {
            let freq = 1.0 / theta.powf(2.0 * i as f32 / rope_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();

            // NeoX half-split: pairs (i, i+half)
            let idx0 = base + i;
            let idx1 = base + i + half;
            let x0 = x[idx0];
            let x1 = x[idx1];
            x[idx0] = x0 * cos_a - x1 * sin_a;
            x[idx1] = x1 * cos_a + x0 * sin_a;
        }
    }
}

/// GQA attention over cached K/V (single query token).
///
/// q: [nq * d] flat, cache.k/v: [max_tokens, nkv * d], k_start..k_end: valid range.
fn gqa_attention_cached(
    q: &[f32],
    k_cache: &Mat,
    v_cache: &Mat,
    k_start: usize,
    k_end: usize,
    nq: usize,
    nkv: usize,
    d: usize,
    scale: f32,
) -> Vec<f32> {
    let ctx_len = k_end - k_start;
    if ctx_len == 0 {
        return vec![0.0; nq * d];
    }

    let heads_per_kv = nq / nkv;
    let mut output = vec![0.0f32; nq * d];

    for qh in 0..nq {
        let kvh = qh / heads_per_kv;
        let q_slice = &q[qh * d..(qh + 1) * d];

        // Compute attention scores
        let mut scores = vec![0.0f32; ctx_len];
        for t in 0..ctx_len {
            let mut dot = 0.0f32;
            for j in 0..d {
                dot += q_slice[j] * k_cache.at(k_start + t, kvh * d + j);
            }
            scores[t] = dot * scale;
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f32 = exp_scores.iter().sum();
        for s in &mut exp_scores {
            *s /= sum;
        }

        // Weighted sum of values
        let out_slice = &mut output[qh * d..(qh + 1) * d];
        for t in 0..ctx_len {
            let w = exp_scores[t];
            if w > 0.0 {
                for j in 0..d {
                    out_slice[j] += w * v_cache.at(k_start + t, kvh * d + j);
                }
            }
        }
    }

    output
}

/// Convert f32 vec to BF16 bits, consuming the input.
pub fn f32s_to_bf16_and_drop(f32s: Vec<f32>) -> Vec<u16> {
    crate::transformer4::f32s_to_bf16_and_drop(f32s)
}

/// Release memory back to the OS.
pub fn release_memory_to_os() {
    crate::transformer4::release_memory_to_os();
}
