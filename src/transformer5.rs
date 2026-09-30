//! # Llama 3.2 -- inference only
//!
//! The backbone Orpheus is fine-tuned from. Structurally it is the plainest
//! architecture in this project: one attention block, one FFN, two norms, full
//! causal attention everywhere.
//!
//! ## What it is not
//!
//! Read alongside `transformer4.rs` (Gemma 3), because the differences are all
//! places a shared implementation would go quietly wrong:
//!
//! | Feature                 | Gemma 3                        | Llama 3.2         |
//! |-------------------------|--------------------------------|-------------------|
//! | Norms per block         | 4                              | **2**             |
//! | Per-head Q/K RMSNorm    | yes                            | **no**            |
//! | Attention scale         | 1/sqrt(query_pre_attn_scalar)  | 1/sqrt(head_dim)  |
//! | Embedding scaling       | x sqrt(hidden_size)            | **none**          |
//! | GGUF norm gamma         | stored as `1 + gamma`          | **stored raw**    |
//! | FFN gate                | gelu_pytorch_tanh              | **SiLU**          |
//! | Attention span          | 5 local : 1 global, two thetas | all global, one   |
//! | lm_head                 | weight-tied                    | **separate**      |
//! | RoPE frequency scaling  | one scalar                     | **per dimension** |
//!
//! Two of those bite hardest. Subtracting 1 from a Llama norm gamma the way the
//! Gemma loader must produces immediate garbage -- loud, easy. The RoPE scaling
//! is the quiet one; see `Config5::inv_freq`.
//!
//! ## Llama 3 RoPE scaling
//!
//! Llama 3.2 does not scale RoPE by a single factor. It divides each frequency
//! band by a different amount: high-frequency dimensions are left alone so
//! local structure is preserved, low-frequency ones are divided by 32 so
//! positions stretch, with a smooth ramp between. GGUF ships the resulting
//! per-dimension divisors in `rope_freqs.weight`, so there is no need to
//! reimplement the piecewise formula -- but they are **divisors**, running from
//! 1.0 up to 32.0, not multipliers running down to 1/32.
//!
//! Applying them the wrong way round scales the low-frequency dimensions by 32
//! instead of by 1/32 -- a factor of 1024 on exactly the dimensions that carry
//! long-range position. Because the unscaled high-frequency dimensions still
//! dominate local structure, the output starts out plausible and degrades as
//! the sequence grows, which is close to the worst failure mode to debug.
//!
//! ## Weight loading from GGUF
//!
//! GGUF's llama naming maps onto the model fields as:
//!
//! ```text
//!   token_embd.weight          -> embed_bf16
//!   output.weight              -> lm_head          (separate, not weight-tied)
//!   output_norm.weight         -> norm.gamma
//!   rope_freqs.weight          -> config.rope_freq_divisors
//!   blk.{i}.attn_norm.weight   -> layers[i].input_layernorm.gamma
//!   blk.{i}.attn_q.weight      -> layers[i].self_attn.q_proj
//!   blk.{i}.attn_k.weight      -> layers[i].self_attn.k_proj
//!   blk.{i}.attn_v.weight      -> layers[i].self_attn.v_proj
//!   blk.{i}.attn_output.weight -> layers[i].self_attn.o_proj
//!   blk.{i}.ffn_norm.weight    -> layers[i].post_attention_layernorm.gamma
//!   blk.{i}.ffn_gate.weight    -> layers[i].mlp.gate_proj
//!   blk.{i}.ffn_up.weight      -> layers[i].mlp.up_proj
//!   blk.{i}.ffn_down.weight    -> layers[i].mlp.down_proj
//! ```
//!
//! **Norm gammas are stored verbatim.** Gemma's GGUF files hold `1 + gamma` and
//! its loader subtracts one; doing that here scales every norm by `gamma - 1`
//! and the model emits garbage from the first token. The mistake is loud, but
//! only if you know to look for it.

#![allow(dead_code)]

use std::sync::Arc;

use crate::autograd2::{Mat, MatBf16, TensorNode};
use crate::gguf_loader::{GgufFile, GgufMetaValue, GgufType};
use crate::nn2::{Linear2, RmsNorm2};
use crate::tokenizer::{HfBpeTokenizer, PreTokenizer};
use crate::transformer3::SamplingParams;
use crate::transformer4::{
    LcgRng, f32s_to_bf16_and_drop, gqa_attention_cached, load_linear_from_gguf,
};

// =============================================================================
// RoPE conventions
// =============================================================================

/// Which dimensions RoPE rotates against each other.
///
/// This is not a free choice -- it has to match how the weights were stored,
/// and the two conventions in circulation disagree:
///
///   HalfSplit    dimension `i` rotates with `i + head_dim/2`
///   Interleaved  dimension `2i` rotates with `2i + 1`
///
/// HuggingFace's `LlamaAttention` uses `rotate_half`, which is HalfSplit.
/// llama.cpp's llama path uses Interleaved -- and reconciles the two by
/// **permuting the Q and K weight rows during conversion**, so that
/// interleaved rotation of the permuted weights reproduces half-split rotation
/// of the originals. `convert_hf_to_gguf.py` does this for the llama
/// architecture but not for Gemma, which is why `transformer4` rotates
/// half-split and this rotates interleaved off the very same file format.
///
/// Getting it wrong is close to undetectable at first. Both conventions rotate
/// by the same angles and differ only in which pairs those angles apply to, so
/// at low positions -- where every angle is small and every rotation near
/// identity -- the output looks fine. It falls apart as positions grow, which
/// for a speech model means the first few tokens are right and the rest is
/// noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopePairing {
    HalfSplit,
    Interleaved,
}

// =============================================================================
// Config5
// =============================================================================

#[derive(Clone, Debug)]
pub struct Config5 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    /// Explicit, though for Llama 3.2 it equals hidden_size / n_heads.
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
    pub max_position_embeddings: usize,
    pub eos_token_id: usize,

    /// Per-dimension RoPE divisors, `head_dim / 2` of them, as shipped in
    /// GGUF's `rope_freqs.weight`. Empty means unscaled RoPE.
    pub rope_freq_divisors: Vec<f32>,

    /// Interleaved for GGUF, which stores permuted Q/K weights. See
    /// `RopePairing`.
    pub rope_pairing: RopePairing,
}

impl Default for Config5 {
    fn default() -> Self {
        Config5 {
            vocab_size: 0,
            hidden_size: 0,
            num_hidden_layers: 0,
            num_attention_heads: 0,
            num_key_value_heads: 0,
            intermediate_size: 0,
            head_dim: 0,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 131072,
            eos_token_id: 128009,
            rope_freq_divisors: Vec::new(),
            rope_pairing: RopePairing::Interleaved,
        }
    }
}

impl Config5 {
    /// canopylabs/orpheus-3b-0.1-ft, whose backbone is Llama 3.2 3B Instruct
    /// with the vocabulary extended by 28 672 audio tokens.
    ///
    /// Values are those in the published GGUF metadata: 28 layers, hidden
    /// 3072, 24 query heads over 8 KV heads, head_dim 128, FFN 8192,
    /// rope_theta 500000, RMS epsilon 1e-5, vocabulary 156 940.
    pub fn orpheus_3b() -> Self {
        Config5 {
            vocab_size: 156940, // 128 256 base + 28 672 audio + 12 markers
            hidden_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            intermediate_size: 8192,
            head_dim: 128,
            rope_theta: 500000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 131072,
            eos_token_id: 128009,
            // Left empty here; the loader fills it from `rope_freqs.weight`.
            ..Config5::default()
        }
    }

    /// Inverse RoPE frequencies, one per rotated pair.
    ///
    /// `inv_freq[i] = theta^(-2i/head_dim) / rope_freq_divisors[i]`, so a
    /// divisor of 32 stretches that band by 32x. With no divisors this is
    /// plain RoPE.
    pub fn inv_freq(&self) -> Vec<f32> {
        let half = self.head_dim / 2;
        let mut out = vec![0.0f32; half];
        for (i, slot) in out.iter_mut().enumerate() {
            let exponent = 2.0f32 * i as f32 / self.head_dim as f32;
            let mut f = 1.0f32 / self.rope_theta.powf(exponent);
            if i < self.rope_freq_divisors.len() {
                // Divisors, not multipliers: `rope_freqs.weight` runs from 1.0 up
                // to 32.0, and dividing by 32 is what stretches a frequency band.
                let d = self.rope_freq_divisors[i];
                if d > 0.0 {
                    f /= d;
                }
            }
            *slot = f;
        }
        out
    }

    /// Query heads sharing each KV head.
    pub fn gqa_group(&self) -> usize {
        if self.num_key_value_heads == 0 {
            1
        } else {
            self.num_attention_heads / self.num_key_value_heads
        }
    }
}

// =============================================================================
// KV cache
// =============================================================================

/// One layer's cached keys and values, preallocated to the session length.
pub struct LlamaLayerKvCache {
    /// [max_seq_len, n_kv_heads * head_dim]
    pub k: Mat,
    pub v: Mat,
    pub seq_len: usize,
}

impl LlamaLayerKvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        LlamaLayerKvCache {
            k: Mat::zeros(max_seq_len, n_kv_heads * head_dim),
            v: Mat::zeros(max_seq_len, n_kv_heads * head_dim),
            seq_len: 0,
        }
    }

    /// Append rows, both [n_new, n_kv_heads * head_dim].
    pub fn append(&mut self, new_k: &Mat, new_v: &Mat) {
        assert!(
            new_k.cols == self.k.cols && new_v.cols == self.v.cols,
            "llama kv cache: width mismatch"
        );
        assert!(self.seq_len + new_k.rows <= self.k.rows, "llama kv cache: overflow");
        let kw = self.k.cols;
        let vw = self.v.cols;
        for r in 0..new_k.rows {
            let dst = self.seq_len + r;
            self.k.data[dst * kw..(dst + 1) * kw].copy_from_slice(&new_k.data[r * kw..(r + 1) * kw]);
            self.v.data[dst * vw..(dst + 1) * vw].copy_from_slice(&new_v.data[r * vw..(r + 1) * vw]);
        }
        self.seq_len += new_k.rows;
    }
}

pub struct LlamaKvCache {
    pub layers: Vec<LlamaLayerKvCache>,
}

impl LlamaKvCache {
    pub fn new(config: &Config5, max_tokens: usize) -> Self {
        let layers = (0..config.num_hidden_layers)
            .map(|_| LlamaLayerKvCache::new(config.num_key_value_heads, config.head_dim, max_tokens))
            .collect();
        LlamaKvCache { layers }
    }

    pub fn clear(&mut self) {
        for layer in &mut self.layers {
            layer.seq_len = 0;
        }
    }

    /// Release the memory, for after a GPU upload.
    pub fn free(&mut self) {
        for layer in &mut self.layers {
            layer.k = Mat::zeros(0, 0);
            layer.v = Mat::zeros(0, 0);
            layer.seq_len = 0;
        }
    }
}

// =============================================================================
// RoPE
// =============================================================================

/// Apply RoPE to every head of a [T, n_heads * head_dim] matrix.
///
/// Row `r` sits at absolute position `offset + r`.
pub fn llama_rope(
    x: &Mat,
    n_heads: usize,
    head_dim: usize,
    offset: usize,
    inv_freq: &[f32],
    pairing: RopePairing,
) -> Mat {
    let half = head_dim / 2;
    assert!(inv_freq.len() >= half, "llama_rope: not enough inverse frequencies");
    assert!(x.cols == n_heads * head_dim, "llama_rope: width is not n_heads * head_dim");

    let mut out = x.clone();
    for row in 0..x.rows {
        let pos = (offset + row) as f32;
        for i in 0..half {
            let angle = pos * inv_freq[i];
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            for h in 0..n_heads {
                // Interleaved pairs (2i, 2i+1) for GGUF's permuted llama
                // weights; half-split (i, i + head_dim/2) for HuggingFace's.
                let c0 = if pairing == RopePairing::Interleaved {
                    h * head_dim + 2 * i
                } else {
                    h * head_dim + i
                };
                let c1 = if pairing == RopePairing::Interleaved { c0 + 1 } else { c0 + half };
                let x0 = x.at(row, c0);
                let x1 = x.at(row, c1);
                *out.at_mut(row, c0) = x0 * cos_a - x1 * sin_a;
                *out.at_mut(row, c1) = x0 * sin_a + x1 * cos_a;
            }
        }
    }
    out
}

// =============================================================================
// LlamaAttention
// =============================================================================

/// Grouped-query attention with full causal span and no per-head norms.
pub struct LlamaAttention {
    pub q_proj: Linear2,
    pub k_proj: Linear2,
    pub v_proj: Linear2,
    pub o_proj: Linear2,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// 1/sqrt(head_dim).
    pub attn_scale: f32,
    /// Shared with the model; `head_dim / 2` entries. `None` until the model
    /// binds it (`LlamaModel::rebind_inv_freq`).
    pub inv_freq: Option<Arc<Vec<f32>>>,
    pub rope_pairing: RopePairing,
}

impl LlamaAttention {
    pub fn new(cfg: &Config5) -> Self {
        LlamaAttention {
            q_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
            k_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            v_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            o_proj: Linear2::new_no_bias_zeros(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            n_q_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            // Plain 1/sqrt(head_dim) -- Gemma's query_pre_attn_scalar has no analogue.
            attn_scale: 1.0f32 / (cfg.head_dim as f32).sqrt(),
            inv_freq: None,
            rope_pairing: cfg.rope_pairing,
        }
    }

    pub fn new_for_inference(cfg: &Config5) -> Self {
        Self::new(cfg)
    }

    /// Project to Q/K/V, rotate Q and K at their absolute positions, append to
    /// the cache, then attend over the whole cache.
    pub fn forward_cached(&self, x: &Mat, cache: &mut LlamaLayerKvCache) -> Mat {
        let inv_freq = self
            .inv_freq
            .as_ref()
            .expect("llama attention: inverse frequencies not bound");
        let seq_offset = cache.seq_len;

        let xn = TensorNode::leaf(x.clone());
        // No per-head Q/K RMSNorm: RoPE goes straight onto the projections.
        let q = llama_rope(
            &self.q_proj.forward(&xn).data(),
            self.n_q_heads,
            self.head_dim,
            seq_offset,
            inv_freq,
            self.rope_pairing,
        );
        let k = llama_rope(
            &self.k_proj.forward(&xn).data(),
            self.n_kv_heads,
            self.head_dim,
            seq_offset,
            inv_freq,
            self.rope_pairing,
        );
        let v = self.v_proj.forward(&xn).data().clone();

        cache.append(&k, &v);

        // Every layer is global: attend over the whole cache. The cache holds only
        // past tokens, so prefill still needs the causal mask that
        // `gqa_attention_cached` applies relative to `k_start`.
        let attn = gqa_attention_cached(
            &q,
            &cache.k,
            &cache.v,
            0,
            cache.seq_len,
            self.n_q_heads,
            self.n_kv_heads,
            self.head_dim,
            self.attn_scale,
        );

        self.o_proj.forward(&TensorNode::leaf(attn)).data().clone()
    }
}

// =============================================================================
// LlamaMlp
// =============================================================================

/// `down_proj(silu(gate_proj(x)) * up_proj(x))`.
pub struct LlamaMlp {
    pub gate_proj: Linear2,
    pub up_proj: Linear2,
    pub down_proj: Linear2,
}

impl LlamaMlp {
    pub fn new(cfg: &Config5) -> Self {
        LlamaMlp {
            gate_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            up_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            down_proj: Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size),
        }
    }

    pub fn new_for_inference(cfg: &Config5) -> Self {
        Self::new(cfg)
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        let xn = TensorNode::leaf(x.clone());
        // SiLU, not Gemma's gelu_pytorch_tanh.
        let gate = self.gate_proj.forward(&xn).silu();
        self.down_proj
            .forward(&gate.mul_elem_node(&self.up_proj.forward(&xn)))
            .data()
            .clone()
    }
}

// =============================================================================
// LlamaBlock
// =============================================================================

/// Pre-norm block:
///
///   x  = x + attn(input_layernorm(x))
///   x  = x + mlp(post_attention_layernorm(x))
///
/// Two norms, not Gemma's four. There is no norm on either sub-block's output.
pub struct LlamaBlock {
    pub input_layernorm: RmsNorm2,
    pub self_attn: LlamaAttention,
    pub post_attention_layernorm: RmsNorm2,
    pub mlp: LlamaMlp,
}

impl LlamaBlock {
    pub fn new(cfg: &Config5) -> Self {
        LlamaBlock {
            input_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            self_attn: LlamaAttention::new_for_inference(cfg),
            post_attention_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            mlp: LlamaMlp::new_for_inference(cfg),
        }
    }

    pub fn new_for_inference(cfg: &Config5) -> Self {
        Self::new(cfg)
    }

    pub fn forward_cached(&self, x: &Mat, cache: &mut LlamaLayerKvCache) -> Mat {
        // Plain RMSNorm, not Gemma's (1 + gamma) variant.
        let normed = self.input_layernorm.forward(&TensorNode::leaf(x.clone())).data().clone();
        let mut h = self.self_attn.forward_cached(&normed, cache);
        h.add_assign(x);

        let normed2 = self
            .post_attention_layernorm
            .forward(&TensorNode::leaf(h.clone()))
            .data()
            .clone();
        let mut ff = self.mlp.forward(&normed2);
        ff.add_assign(&h);
        ff
    }
}

// =============================================================================
// LlamaModel
// =============================================================================

pub struct LlamaModel {
    /// BF16 embedding table, [vocab_size, hidden_size]. Looked up row by row,
    /// so it never needs an f32 copy.
    pub embed_bf16: Option<MatBf16>,
    pub config: Config5,
    pub layers: Vec<LlamaBlock>,
    pub norm: RmsNorm2,
    /// Separate from the embedding: Llama 3.2 3B ties them, but Orpheus does
    /// not, and the extended vocabulary makes this the largest single weight.
    pub lm_head: Linear2,
    /// Precomputed once and shared by every layer.
    pub inv_freq_cache: Arc<Vec<f32>>,
}

impl LlamaModel {
    pub fn new(cfg: Config5) -> Self {
        let norm = RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps);
        let lm_head = Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.vocab_size);
        let layers = (0..cfg.num_hidden_layers)
            .map(|_| LlamaBlock::new_for_inference(&cfg))
            .collect();
        let mut model = LlamaModel {
            embed_bf16: None,
            config: cfg,
            layers,
            norm,
            lm_head,
            inv_freq_cache: Arc::new(Vec::new()),
        };
        model.rebind_inv_freq();
        model
    }

    pub fn new_for_inference(cfg: Config5) -> Self {
        Self::new(cfg)
    }

    /// Wire `head_dim / 2` inverse frequencies into every layer.
    ///
    /// Called after loading, and again after any change to the RoPE fields of
    /// `config`, because every layer holds its own handle on `inv_freq_cache`.
    pub fn rebind_inv_freq(&mut self) {
        self.inv_freq_cache = Arc::new(self.config.inv_freq());
        for layer in &mut self.layers {
            layer.self_attn.inv_freq = Some(self.inv_freq_cache.clone());
            layer.self_attn.rope_pairing = self.config.rope_pairing;
        }
    }

    /// Look up embedding rows. Unlike Gemma there is no sqrt(hidden) scaling.
    pub fn embed_rows(&self, token_ids: &[usize]) -> Mat {
        let t = token_ids.len();
        let h = self.config.hidden_size;
        // No sqrt(hidden_size) scaling -- that is Gemma's.
        let embed = self.embed_bf16.as_ref().expect("llama: embedding table not loaded");
        let bits: &[u16] = &embed.data;
        Mat::from_fn(t, h, |row, col| MatBf16::bf16_to_f32(bits[token_ids[row] * h + col]))
    }

    /// Prefill the cache with `token_ids`, returning logits for the last token.
    pub fn forward_cached(&self, token_ids: &[usize], cache: &mut LlamaKvCache) -> Mat {
        let mut x = self.embed_rows(token_ids);
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[i]);
        }
        // Only the final row's logits matter, so the lm_head runs as a GEMV rather
        // than a T x 156940 gemm.
        let last = Mat::from_fn(1, x.cols, |_, c| x.at(x.rows - 1, c));
        self.lm_head
            .forward(&self.norm.forward(&TensorNode::leaf(last)))
            .data()
            .clone()
    }

    /// Total resident weight bytes, for reporting.
    pub fn weight_bytes(&self) -> usize {
        let mut n = 0usize;
        if let Some(e) = &self.embed_bf16 {
            n += e.size_bytes();
        }
        n += linear_bytes(&self.lm_head);
        for layer in &self.layers {
            n += linear_bytes(&layer.self_attn.q_proj)
                + linear_bytes(&layer.self_attn.k_proj)
                + linear_bytes(&layer.self_attn.v_proj)
                + linear_bytes(&layer.self_attn.o_proj)
                + linear_bytes(&layer.mlp.gate_proj)
                + linear_bytes(&layer.mlp.up_proj)
                + linear_bytes(&layer.mlp.down_proj);
        }
        n
    }

    /// Quantize `lm_head` from BF16 to Q4_K, freeing the BF16 copy.
    ///
    /// Worth doing for a large vocabulary. Orpheus ships `output.weight` as
    /// Q6_K, which this loader widens to BF16 at 2 bytes an element -- 964 MB
    /// for 156 940 x 3072. Q4_K brings that to 271 MB, and since the lm_head
    /// GEMV is the single largest memory read per token, it speeds decode up
    /// as well.
    pub fn quantize_lm_head(&mut self) {
        self.lm_head.quantize_bf16_to_q4k();
    }

    /// Total projections, `7 * layers + 1` for the lm_head.
    pub fn projection_count(&self) -> usize {
        7 * self.layers.len() + 1
    }

    /// How many projections are currently held as BF16 rather than Q4_K.
    ///
    /// The signal for whether a checkpoint was stored at uniform higher
    /// precision. A Q4_K_M file keeps most projections native and widens only
    /// the deliberately-higher-precision minority, so this stays small; a Q8_0
    /// or F16 file widens all of them.
    pub fn bf16_projection_count(&self) -> usize {
        let mut n = if self.lm_head.bf16_weight.is_some() { 1 } else { 0 };
        for layer in &self.layers {
            for l in [
                &layer.self_attn.q_proj,
                &layer.self_attn.k_proj,
                &layer.self_attn.v_proj,
                &layer.self_attn.o_proj,
                &layer.mlp.gate_proj,
                &layer.mlp.up_proj,
                &layer.mlp.down_proj,
            ] {
                if l.bf16_weight.is_some() {
                    n += 1;
                }
            }
        }
        n
    }

    /// Requantize every BF16 projection to Q4_K, freeing the BF16 copies.
    ///
    /// For files stored at a uniform higher precision. A Q8_0 checkpoint has no
    /// Q4_K tensors at all, so this loader widens every one of them to BF16 --
    /// 2 bytes an element, about 6.6 GB for a 3B model, and roughly 3.5x the
    /// per-token memory traffic of Q4_K. Requantizing brings both back down.
    ///
    /// Do **not** call this on a Q4_K_M file. That format deliberately keeps
    /// `attn_v` and `ffn_down` at Q6_K because they are the quality-sensitive
    /// projections; flattening them to Q4_K discards a choice the quantizer
    /// made on purpose. Tensors already stored as Q4_K are untouched either
    /// way, so the damage would be exactly to the ones worth keeping.
    ///
    /// Returns the number of projections converted.
    pub fn quantize_projections_to_q4k(&mut self) -> usize {
        // `quantize_bf16_to_q4k` is a no-op without a BF16 weight, so anything
        // already stored as Q4_K passes through untouched.
        let mut converted = 0usize;
        let mut convert = |l: &mut Linear2| {
            if l.bf16_weight.is_some() {
                l.quantize_bf16_to_q4k();
                converted += 1;
            }
        };
        for layer in &mut self.layers {
            convert(&mut layer.self_attn.q_proj);
            convert(&mut layer.self_attn.k_proj);
            convert(&mut layer.self_attn.v_proj);
            convert(&mut layer.self_attn.o_proj);
            convert(&mut layer.mlp.gate_proj);
            convert(&mut layer.mlp.up_proj);
            convert(&mut layer.mlp.down_proj);
        }
        convert(&mut self.lm_head);
        converted
    }

    /// Autoregressive generation over a KV cache.
    ///
    /// `on_token` receives each sampled id and returns false to stop, which is
    /// how a caller detects an end-of-audio token it recognises but the config
    /// does not. Returns the number of tokens generated.
    pub fn generate(
        &self,
        prompt: &[usize],
        max_new: usize,
        params: &SamplingParams,
        debug: bool,
        mut on_token: impl FnMut(usize) -> bool,
    ) -> usize {
        if prompt.is_empty() {
            return 0;
        }

        let mut cache = LlamaKvCache::new(&self.config, prompt.len() + max_new + 1);
        let mut rng = LcgRng::new(params.seed);

        let mut logits = self.forward_cached(prompt, &mut cache);

        let mut generated: Vec<usize> = Vec::with_capacity(max_new);

        for step in 0..max_new {
            let next = sample_token_large_vocab(&logits, 0, params, &generated, &mut rng);

            if debug {
                let mut max_logit = f32::NEG_INFINITY;
                for c in 0..logits.cols {
                    max_logit = std_max(max_logit, logits.at(0, c));
                }
                eprintln!(
                    "[ llama ] step {} -> id {} (logit {:.3}, max {:.3})",
                    step,
                    next,
                    logits.at(0, next),
                    max_logit
                );
            }

            generated.push(next);
            if !on_token(next) {
                return generated.len();
            }
            if params.eos_token_id == Some(next) {
                return generated.len();
            }
            if step + 1 == max_new {
                break;
            }
            logits = self.forward_cached(&[next], &mut cache);
        }

        generated.len()
    }

    /// Load from a GGUF file with `general.architecture == "llama"`.
    ///
    /// Reads `token_embd`, `output`, `output_norm`, `rope_freqs` and the
    /// `blk.{i}.*` tensors. Norm gammas are taken **verbatim** -- the `1 +
    /// gamma` convention is Gemma's alone.
    pub fn load_weights_from_gguf(&mut self, path: &str) -> Result<(), String> {
        let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;

        // ---- sanity-check the metadata against the config ----
        if let Some(arch) = gguf.metadata.get("general.architecture") {
            if let Some(name) = arch.as_str() {
                if name != "llama" {
                    return Err(format!(
                        "llama: GGUF declares architecture '{}', expected 'llama'",
                        name
                    ));
                }
            }
        }

        let file_layers = meta_u64(&gguf, "llama.block_count", self.config.num_hidden_layers);
        let file_hidden = meta_u64(&gguf, "llama.embedding_length", self.config.hidden_size);
        let file_heads = meta_u64(&gguf, "llama.attention.head_count", self.config.num_attention_heads);
        let file_kv_heads =
            meta_u64(&gguf, "llama.attention.head_count_kv", self.config.num_key_value_heads);
        let file_ffn = meta_u64(&gguf, "llama.feed_forward_length", self.config.intermediate_size);
        let file_head_dim = meta_u64(&gguf, "llama.attention.key_length", self.config.head_dim);

        // A geometry mismatch would otherwise show up as a shape error deep in the
        // first matmul, or worse, as a model that runs and produces noise.
        let mismatch = |what: &str, got: usize, want: usize| -> String {
            format!("llama: GGUF says {} is {} but the config says {}", what, got, want)
        };
        if file_layers != self.config.num_hidden_layers {
            return Err(mismatch("block_count", file_layers, self.config.num_hidden_layers));
        }
        if file_hidden != self.config.hidden_size {
            return Err(mismatch("embedding_length", file_hidden, self.config.hidden_size));
        }
        if file_heads != self.config.num_attention_heads {
            return Err(mismatch("head_count", file_heads, self.config.num_attention_heads));
        }
        if file_kv_heads != self.config.num_key_value_heads {
            return Err(mismatch("head_count_kv", file_kv_heads, self.config.num_key_value_heads));
        }
        if file_ffn != self.config.intermediate_size {
            return Err(mismatch("feed_forward_length", file_ffn, self.config.intermediate_size));
        }
        if file_head_dim != self.config.head_dim {
            return Err(mismatch("attention.key_length", file_head_dim, self.config.head_dim));
        }

        self.config.rope_theta = meta_f32(&gguf, "llama.rope.freq_base", self.config.rope_theta);
        self.config.rms_norm_eps = meta_f32(
            &gguf,
            "llama.attention.layer_norm_rms_epsilon",
            self.config.rms_norm_eps,
        );
        let file_vocab = meta_u64(&gguf, "llama.vocab_size", self.config.vocab_size);
        if file_vocab != self.config.vocab_size {
            return Err(mismatch("vocab_size", file_vocab, self.config.vocab_size));
        }

        eprintln!(
            "[ GGUF ] llama: {} layers, hidden {}, {}/{} heads, ffn {}, vocab {}, theta {:.0}",
            self.config.num_hidden_layers,
            self.config.hidden_size,
            self.config.num_attention_heads,
            self.config.num_key_value_heads,
            self.config.intermediate_size,
            self.config.vocab_size,
            self.config.rope_theta as f64
        );

        // Norm epsilon reaches the layers through their constructors, so refresh
        // the ones already built from the file's value.
        self.norm.eps = self.config.rms_norm_eps;
        for layer in &mut self.layers {
            layer.input_layernorm.eps = self.config.rms_norm_eps;
            layer.post_attention_layernorm.eps = self.config.rms_norm_eps;
        }

        // ---- tensors ----
        let mut loaded = 0usize;
        let mut have_embed = false;
        let mut have_output = false;

        for idx in 0..gguf.tensor_info.len() {
            let name = gguf.tensor_info[idx].name.as_str();
            let gtype = gguf.tensor_info[idx].gguf_type;
            let shape = &gguf.tensor_info[idx].shape;

            if name == "token_embd.weight" {
                // GGUF stores [hidden, vocab]; the lookup table wants [vocab, hidden].
                let vocab = if shape.len() == 2 { shape[1] } else { shape[0] };
                let hidden = if shape.len() == 2 { shape[0] } else { 1 };
                if vocab != self.config.vocab_size || hidden != self.config.hidden_size {
                    return Err(format!(
                        "llama: token_embd is {}x{}, expected {}x{}",
                        vocab, hidden, self.config.vocab_size, self.config.hidden_size
                    ));
                }
                eprintln!(
                    "[ GGUF ] token_embd: type={:?} vocab={} hidden={}",
                    gtype, vocab, hidden
                );

                let from_f32 = |decoded: std::io::Result<Vec<f32>>| -> Result<Vec<u16>, String> {
                    let f32s = decoded.map_err(|e| e.to_string())?;
                    Ok(f32s_to_bf16_and_drop(f32s))
                };

                let bits = match gtype {
                    GgufType::Bf16 => gguf.decode_bf16(idx).map_err(|e| e.to_string())?,
                    GgufType::F16 => from_f32(gguf.decode_f16_to_f32(idx))?,
                    GgufType::F32 => from_f32(gguf.decode_f32(idx))?,
                    GgufType::Q4_0 => from_f32(gguf.decode_q4_0_to_f32(idx))?,
                    GgufType::Q4K => from_f32(gguf.decode_q4k_to_f32(idx))?,
                    GgufType::Q6K => from_f32(gguf.decode_q6k_to_f32(idx))?,
                    GgufType::Q8_0 => from_f32(gguf.decode_q8_0_to_f32(idx))?,
                    GgufType::Q5K => from_f32(gguf.decode_q5k_to_f32(idx))?,
                    _ => {
                        return Err(format!("llama: token_embd has unsupported type {:?}", gtype));
                    }
                };
                self.embed_bf16 = Some(MatBf16 {
                    data: Arc::new(bits),
                    rows: vocab,
                    cols: hidden,
                });
                have_embed = true;
                loaded += 1;
                continue;
            }

            if name == "output.weight" {
                eprintln!("[ GGUF ] output.weight (lm_head): type={:?}", gtype);
                load_linear_from_gguf(&gguf, idx, &mut self.lm_head).map_err(|e| e.to_string())?;
                have_output = true;
                loaded += 1;
                continue;
            }

            if name == "output_norm.weight" {
                load_norm_from_gguf(&gguf, idx, &mut self.norm, self.config.hidden_size)?;
                loaded += 1;
                continue;
            }

            if name == "rope_freqs.weight" {
                // Per-dimension RoPE divisors. Loading these is what keeps long
                // sequences coherent; see the module docs on why they are divisors.
                let divisors = gguf.decode_f32(idx).map_err(|e| e.to_string())?;
                let want = self.config.head_dim / 2;
                if divisors.len() != want {
                    return Err(format!(
                        "llama: rope_freqs has {} entries, expected head_dim/2 = {}",
                        divisors.len(),
                        want
                    ));
                }
                eprintln!(
                    "[ GGUF ] rope_freqs: {} divisors, {:.4}..{:.4}",
                    divisors.len(),
                    divisors[0] as f64,
                    divisors[divisors.len() - 1] as f64
                );
                self.config.rope_freq_divisors = divisors;
                loaded += 1;
                continue;
            }

            let Some((layer_idx, field)) = split_block_name(name) else {
                eprintln!("[ GGUF ] Note: ignoring unrecognised tensor '{}'", name);
                continue;
            };
            if layer_idx >= self.layers.len() {
                return Err(format!(
                    "llama: tensor '{}' names layer {} but the model has {}",
                    name,
                    layer_idx,
                    self.layers.len()
                ));
            }
            let hidden = self.config.hidden_size;
            let layer = &mut self.layers[layer_idx];
            let io = |r: std::io::Result<()>| r.map_err(|e| e.to_string());

            match field {
                "attn_norm.weight" => {
                    load_norm_from_gguf(&gguf, idx, &mut layer.input_layernorm, hidden)?
                }
                "ffn_norm.weight" => {
                    load_norm_from_gguf(&gguf, idx, &mut layer.post_attention_layernorm, hidden)?
                }
                "attn_q.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.q_proj))?,
                "attn_k.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.k_proj))?,
                "attn_v.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.v_proj))?,
                "attn_output.weight" => {
                    io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.o_proj))?
                }
                "ffn_gate.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.gate_proj))?,
                "ffn_up.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.up_proj))?,
                "ffn_down.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.down_proj))?,
                _ => {
                    eprintln!("[ GGUF ] Note: ignoring unrecognised field '{}'", name);
                    continue;
                }
            }
            loaded += 1;
        }

        if !have_embed {
            return Err("llama: GGUF has no token_embd.weight".into());
        }
        if !have_output {
            // Weight tying, which Llama 3.2 3B does and the English Orpheus
            // fine-tune does not. `MatBf16` is Arc-backed, so the lm_head
            // adopts the embedding's bits rather than copying 964 MB of them.
            //
            // The orientations already agree: the table is [vocab, hidden] and
            // Linear2 computes `input @ weight.T` for a weight of
            // [out_features, in_features].
            let Some(embed) = &self.embed_bf16 else {
                return Err("llama: GGUF has neither output.weight nor a usable token_embd".into());
            };
            eprintln!("[ GGUF ] No output.weight; tying lm_head to the embedding");
            let (data, rows, cols) = (embed.data.clone(), embed.rows, embed.cols);
            self.lm_head.load_bf16_arc(data, rows, cols);
        }

        // The divisors are read from the same tensor loop that fills the layers,
        // and every layer holds a handle on the cache, so recompute now.
        self.rebind_inv_freq();

        eprintln!(
            "[ GGUF ] Loaded {} tensors, {:.2} GB resident",
            loaded,
            self.weight_bytes() as f64 / 1e9
        );
        Ok(())
    }
}

/// Bytes held by one projection, whichever form its weight is in.
fn linear_bytes(l: &Linear2) -> usize {
    if let Some(q4k) = &l.q4k_weight {
        return q4k.size_bytes();
    }
    if let Some(bf16) = &l.bf16_weight {
        return bf16.size_bytes();
    }
    l.weight.data().numel() * std::mem::size_of::<f32>()
}

/// `std::max(a, b)`: `b` only when `a < b`, so ties and NaNs keep `a`.
#[inline]
fn std_max(a: f32, b: f32) -> f32 {
    if a < b { b } else { a }
}

// =============================================================================
// GGUF helpers
// =============================================================================

/// Set an RMSNorm gamma from an f32 or F16 GGUF tensor.
fn load_norm_from_gguf(
    gguf: &GgufFile,
    idx: usize,
    norm: &mut RmsNorm2,
    expect: usize,
) -> Result<(), String> {
    let gtype = gguf.tensor_info[idx].gguf_type;
    let values = match gtype {
        GgufType::F32 => gguf.decode_f32(idx).map_err(|e| e.to_string())?,
        GgufType::F16 => gguf.decode_f16_to_f32(idx).map_err(|e| e.to_string())?,
        _ => {
            return Err(format!(
                "llama: norm tensor '{}' has unsupported type {:?}",
                gguf.tensor_info[idx].name, gtype
            ));
        }
    };
    if values.len() != expect {
        return Err(format!(
            "llama: norm tensor '{}' has {} entries, expected {}",
            gguf.tensor_info[idx].name,
            values.len(),
            expect
        ));
    }
    // Verbatim -- no `1 + gamma` adjustment. See the module docs.
    norm.gamma.set_data(Mat::new(values, 1, expect));
    Ok(())
}

/// Parse `blk.{i}.` and return the layer index plus the remaining field name.
fn split_block_name(name: &str) -> Option<(usize, &str)> {
    const PREFIX: &str = "blk.";
    let rest = name.strip_prefix(PREFIX)?;
    let dot = rest.find('.')?;
    let mut value = 0usize;
    for ch in rest[..dot].bytes() {
        if !ch.is_ascii_digit() {
            return None;
        }
        value = value.wrapping_mul(10).wrapping_add((ch - b'0') as usize);
    }
    Some((value, &rest[dot + 1..]))
}

/// Read a u64 metadata value, or fall back.
fn meta_u64(gguf: &GgufFile, key: &str, fallback: usize) -> usize {
    match gguf.metadata.get(key) {
        None => fallback,
        Some(v) => v.as_u64().map(|v| v as usize).unwrap_or(fallback),
    }
}

/// Read an f32 metadata value, or fall back.
fn meta_f32(gguf: &GgufFile, key: &str, fallback: f32) -> f32 {
    match gguf.metadata.get(key) {
        None => fallback,
        Some(v) => v.as_f32().unwrap_or(fallback),
    }
}

// =============================================================================
// Embedded tokenizer
// =============================================================================

/// Build a byte-level BPE tokenizer from a GGUF file's embedded vocabulary.
///
/// Reads `tokenizer.ggml.tokens`, `tokenizer.ggml.merges` and
/// `tokenizer.ggml.pre`. A `pre` of `llama-bpe` selects Llama 3's digit
/// grouping, which differs from GPT-2's in a way that changes the ids for
/// every number in the input.
///
/// This exists so a GGUF file is self-sufficient. Orpheus is gated on
/// HuggingFace, so requiring a separate `tokenizer.json` would mean requiring
/// an account for weights that are otherwise freely mirrored.
pub fn load_gguf_tokenizer(gguf: &GgufFile) -> Result<HfBpeTokenizer, String> {
    let read_strings = |key: &str| -> Result<Vec<String>, String> {
        let Some(value) = gguf.metadata.get(key) else {
            return Err(format!("llama: GGUF has no '{}'", key));
        };
        let GgufMetaValue::Array(array) = value else {
            return Err(format!("llama: GGUF '{}' is not an array", key));
        };
        let mut out = Vec::with_capacity(array.len());
        for entry in array {
            let Some(text) = entry.as_str() else {
                return Err(format!("llama: GGUF '{}' holds a non-string entry", key));
            };
            out.push(text.to_string());
        }
        Ok(out)
    };

    let tokens = read_strings("tokenizer.ggml.tokens")?;
    let merges = read_strings("tokenizer.ggml.merges")?;

    // `tokenizer.ggml.model` says which family; `pre` says which
    // pre-tokenizer regex within it.
    let model = gguf
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .unwrap_or("gpt2")
        .to_string();
    let pre = gguf
        .metadata
        .get("tokenizer.ggml.pre")
        .and_then(|v| v.as_str())
        .unwrap_or("default")
        .to_string();

    if model != "gpt2" {
        return Err(format!(
            "llama: GGUF tokenizer model is '{}'; only byte-level 'gpt2' vocabularies are supported here",
            model
        ));
    }

    let kind = if pre == "llama-bpe" { PreTokenizer::Llama3 } else { PreTokenizer::Gpt2 };

    eprintln!(
        "[ GGUF ] tokenizer: model={} pre={} vocab={} merges={}",
        model,
        pre,
        tokens.len(),
        merges.len()
    );

    HfBpeTokenizer::from_vocab_and_merges(tokens, &merges, /*byte_level=*/ true, kind)
}

// =============================================================================
// Sampling
// =============================================================================

/// Sample from a logits row without sorting the whole vocabulary.
///
/// Same pipeline as `sample_token`: repetition penalty, temperature, softmax,
/// top-k, top-p, then draw. The difference is that top-k uses
/// `select_nth_unstable_by` and only sorts the survivors, which for a
/// 156 940-entry vocabulary is the difference between ~10 ms and well under
/// 1 ms per token. At a realtime budget of 11.9 ms per token, a full sort would
/// consume most of it on its own.
///
/// `seen` holds previously generated ids for the repetition penalty.
pub fn sample_token_large_vocab(
    logits: &Mat,
    row: usize,
    params: &SamplingParams,
    seen: &[usize],
    rng: &mut LcgRng,
) -> usize {
    let v = logits.cols;
    let mut scores: Vec<f32> = (0..v).map(|c| logits.at(row, c)).collect();

    // Vocabulary mask. -infinity survives the temperature division and makes
    // exp() underflow to exactly zero, so masked ids cannot be drawn and
    // cannot be the greedy argmax either.
    if params.allowed_min.is_some() || params.allowed_max.is_some() {
        let lo = params.allowed_min.unwrap_or(0);
        let hi = params.allowed_max.unwrap_or(v);
        for (c, s) in scores.iter_mut().enumerate() {
            if c < lo || c >= hi {
                *s = f32::NEG_INFINITY;
            }
        }
        for &id in &params.allowed_extra {
            if id < v {
                scores[id] = logits.at(row, id);
            }
        }
    }

    // Repetition penalty over the trailing window of generated tokens.
    if params.repetition_penalty != 1.0 && !seen.is_empty() {
        const WINDOW: usize = 64;
        let start = if seen.len() > WINDOW { seen.len() - WINDOW } else { 0 };
        for &id in &seen[start..] {
            if id >= v {
                continue;
            }
            scores[id] = if scores[id] > 0.0 {
                scores[id] / params.repetition_penalty
            } else {
                scores[id] * params.repetition_penalty
            };
        }
    }

    // Greedy: no need to build a distribution at all.
    if params.temperature <= 0.0 {
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (c, &s) in scores.iter().enumerate() {
            if s > best_v {
                best_v = s;
                best = c;
            }
        }
        return best;
    }

    for s in &mut scores {
        *s /= params.temperature;
    }

    let mut max_s = f32::NEG_INFINITY;
    for &s in &scores {
        max_s = std_max(max_s, s);
    }
    let mut probs = vec![0.0f32; v];
    let mut sum_exp = 0.0f32;
    for c in 0..v {
        probs[c] = (scores[c] - max_s).exp();
        sum_exp += probs[c];
    }
    if sum_exp <= 0.0 {
        return 0;
    }
    for p in &mut probs {
        *p /= sum_exp;
    }

    // Descending by probability. Probabilities come out of exp(), so they are
    // never NaN or negative zero, and total_cmp agrees with `>`.
    let by_prob_desc = |a: &usize, b: &usize| probs[*b].total_cmp(&probs[*a]);

    // Candidate set. Top-k selects with select_nth_unstable -- linear, rather
    // than sorting 156 940 entries per token.
    let mut idx: Vec<usize> = Vec::new();
    let k = if params.top_k > 0 { params.top_k.min(v) } else { v };
    if k < v {
        idx = (0..v).collect();
        idx.select_nth_unstable_by(k, by_prob_desc);
        idx.truncate(k);
    } else if params.top_p < 1.0 {
        // Nucleus sampling has to look at the mass in descending order, but it
        // only ever needs the head of the distribution. Anything below
        // 1/(4v) cannot enter a nucleus of any useful size, and dropping it
        // first keeps the sort small.
        let floor_p = 0.25f32 / v as f32;
        idx.reserve(1024);
        for (c, &p) in probs.iter().enumerate() {
            if p > floor_p {
                idx.push(c);
            }
        }
        if idx.is_empty() {
            idx = (0..v).collect();
        }
    }

    if !idx.is_empty() {
        // Only the survivors get sorted.
        idx.sort_unstable_by(by_prob_desc);

        if params.top_p < 1.0 {
            let mut cumulative = 0.0f32;
            let mut keep = 0usize;
            while keep < idx.len() {
                cumulative += probs[idx[keep]];
                if cumulative >= params.top_p {
                    keep += 1; // include the token that crossed the threshold
                    break;
                }
                keep += 1;
            }
            idx.truncate(keep.max(1));
        }

        let mut mass = 0.0f32;
        for &c in &idx {
            mass += probs[c];
        }
        if mass <= 0.0 {
            return idx[0];
        }
        // Inverse CDF over the candidates.
        let draw = rng.next_f32() * mass;
        let mut running = 0.0f32;
        for &c in &idx {
            running += probs[c];
            if running >= draw {
                return c;
            }
        }
        return idx[idx.len() - 1];
    }

    // No filtering: draw straight from the full distribution.
    let draw = rng.next_f32();
    let mut running = 0.0f32;
    for (c, &p) in probs.iter().enumerate() {
        running += p;
        if running >= draw {
            return c;
        }
    }
    v - 1
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// `SamplingParams` with the defaults the C++ struct declares.
    fn default_params() -> SamplingParams {
        SamplingParams {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed: 0,
            eos_token_id: None,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            allowed_min: None,
            allowed_max: None,
            allowed_extra: Vec::new(),
        }
    }

    /// Deterministic activations for a [T, n_heads * head_dim] tensor.
    fn heads(rows: usize, n_heads: usize, head_dim: usize) -> Mat {
        Mat::from_fn(rows, n_heads * head_dim, |r, c| {
            0.01f32 * (r * n_heads * head_dim + c) as f32 - 0.5
        })
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    #[test]
    fn test_orpheus_3b_matches_published_gguf_metadata() {
        let c = Config5::orpheus_3b();
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.hidden_size, 3072);
        assert_eq!(c.num_attention_heads, 24);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.intermediate_size, 8192);
        assert_eq!(c.rope_theta, 500000.0);
        assert_eq!(c.vocab_size, 156940);
        // 24 query heads over 8 KV heads: three queries share each KV head.
        assert_eq!(c.gqa_group(), 3);
        // head_dim * n_heads reconstructs hidden_size, so the projections line up.
        assert_eq!(c.head_dim * c.num_attention_heads, c.hidden_size);
        // Interleaved is the GGUF convention; see RopePairing.
        assert_eq!(c.rope_pairing, RopePairing::Interleaved);
    }

    #[test]
    fn test_inv_freq_decreases_with_dimension() {
        let c = Config5::orpheus_3b();
        let f = c.inv_freq();
        assert_eq!(f.len(), c.head_dim / 2);
        assert!(approx(f[0], 1.0)); // theta^0
        for i in 1..f.len() {
            assert!(f[i] < f[i - 1]);
        }
    }

    #[test]
    fn test_rope_freq_divisors_divide_rather_than_multiply() {
        // `rope_freqs.weight` runs from 1.0 up to 32.0. Treating those as
        // multipliers would raise the low-frequency bands by 32x instead of
        // lowering them -- a factor of 1024 the wrong way, on exactly the
        // dimensions that carry long-range position.
        let mut c = Config5::orpheus_3b();
        let plain = c.inv_freq();

        c.rope_freq_divisors = vec![1.0; c.head_dim / 2];
        *c.rope_freq_divisors.last_mut().unwrap() = 32.0;
        let scaled = c.inv_freq();

        assert!(approx(scaled[0], plain[0]));
        assert!(scaled.last().unwrap() < plain.last().unwrap());
        assert!((scaled.last().unwrap() - plain.last().unwrap() / 32.0).abs() < 1e-12);
    }

    #[test]
    fn test_zero_divisor_is_ignored() {
        let mut c = Config5::orpheus_3b();
        c.rope_freq_divisors = vec![0.0; c.head_dim / 2];
        for f in c.inv_freq() {
            assert!(f.is_finite());
            assert!(f > 0.0);
        }
    }

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------

    #[test]
    fn test_llama_rope_at_position_0_is_identity() {
        let c = Config5::orpheus_3b();
        let f = c.inv_freq();
        let x = heads(1, 2, 128);
        for p in [RopePairing::Interleaved, RopePairing::HalfSplit] {
            let out = llama_rope(&x, 2, 128, 0, &f, p);
            for i in 0..x.data.len() {
                assert!(approx(out.data[i], x.data[i]));
            }
        }
    }

    #[test]
    fn test_llama_rope_preserves_pair_norms() {
        // A rotation changes direction, never length. This holds for both
        // conventions and is the cheapest check that the sin/cos signs are
        // consistent.
        let c = Config5::orpheus_3b();
        let f = c.inv_freq();
        const HEAD_DIM: usize = 128;
        const HALF: usize = HEAD_DIM / 2;
        let x = heads(3, 2, HEAD_DIM);

        for p in [RopePairing::Interleaved, RopePairing::HalfSplit] {
            let out = llama_rope(&x, 2, HEAD_DIM, 5, &f, p);
            for row in 0..x.rows {
                for h in 0..2 {
                    for i in 0..HALF {
                        let c0 = if p == RopePairing::Interleaved {
                            h * HEAD_DIM + 2 * i
                        } else {
                            h * HEAD_DIM + i
                        };
                        let c1 = if p == RopePairing::Interleaved { c0 + 1 } else { c0 + HALF };
                        let before = x.at(row, c0) * x.at(row, c0) + x.at(row, c1) * x.at(row, c1);
                        let after =
                            out.at(row, c0) * out.at(row, c0) + out.at(row, c1) * out.at(row, c1);
                        assert!((before - after).abs() < 1e-3);
                    }
                }
            }
        }
    }

    #[test]
    fn test_llama_rope_pairings_are_genuinely_different() {
        // The regression test for the bug that produced this enum. GGUF permutes
        // llama Q/K weight rows so that interleaved rotation reproduces
        // HuggingFace's half-split `rotate_half`; applying half-split to those
        // permuted weights rotates the wrong pairs together.
        //
        // The reason it is worth a test rather than a comment: both conventions
        // use the same angles, so at low positions -- where every rotation is
        // near identity -- they agree to several decimals. They only separate as
        // position grows, which in a speech model means the first few tokens come
        // out right and everything after is noise.
        let c = Config5::orpheus_3b();
        let f = c.inv_freq();
        let x = heads(1, 1, 128);

        let inter = llama_rope(&x, 1, 128, 40, &f, RopePairing::Interleaved);
        let split = llama_rope(&x, 1, 128, 40, &f, RopePairing::HalfSplit);

        let differing = (0..inter.data.len())
            .filter(|&i| (inter.data[i] - split.data[i]).abs() > 1e-3)
            .count();
        assert!(differing > 0);
    }

    #[test]
    fn test_llama_rope_rotates_unit_pair_by_expected_angle() {
        // One head, head_dim 2, so there is a single pair at frequency inv_freq[0]
        // == 1 and the rotation angle is just the position.
        let c = Config5 { head_dim: 2, rope_theta: 10000.0, ..Config5::default() };
        let f = c.inv_freq();
        assert!(approx(f[0], 1.0));

        let x = Mat::new(vec![1.0, 0.0], 1, 2);
        let out = llama_rope(&x, 1, 2, 3, &f, RopePairing::Interleaved);
        // (1, 0) rotated by 3 radians -> (cos 3, sin 3).
        assert!(approx(out.at(0, 0), 3.0f32.cos()));
        assert!(approx(out.at(0, 1), 3.0f32.sin()));
    }

    #[test]
    fn test_llama_rope_advances_angle_with_row_offset() {
        let c = Config5 { head_dim: 2, ..Config5::default() };
        let f = c.inv_freq();
        let x = Mat::from_fn(3, 2, |_, col| if col == 0 { 1.0 } else { 0.0 });
        // Rows are at absolute positions offset..offset+2.
        let out = llama_rope(&x, 1, 2, 10, &f, RopePairing::Interleaved);
        for r in 0..3 {
            let angle = (10 + r) as f32 * f[0];
            assert!(approx(out.at(r, 0), angle.cos()));
            assert!(approx(out.at(r, 1), angle.sin()));
        }
    }

    // -------------------------------------------------------------------------
    // KV cache
    // -------------------------------------------------------------------------

    #[test]
    fn test_llama_kv_cache_preallocates_and_appends() {
        let mut c = Config5::orpheus_3b();
        c.num_hidden_layers = 2;
        let mut cache = LlamaKvCache::new(&c, 16);
        assert_eq!(cache.layers.len(), 2);

        let kv_dim = c.num_key_value_heads * c.head_dim;
        assert_eq!(cache.layers[0].k.rows, 16);
        assert_eq!(cache.layers[0].k.cols, kv_dim);
        assert_eq!(cache.layers[0].seq_len, 0);

        let k = Mat::ones(3, kv_dim);
        let v = Mat::ones(3, kv_dim).scale(2.0);
        cache.layers[0].append(&k, &v);
        assert_eq!(cache.layers[0].seq_len, 3);
        assert!(approx(cache.layers[0].k.at(2, 0), 1.0));
        assert!(approx(cache.layers[0].v.at(2, 0), 2.0));
        // Rows past seq_len stay zero.
        assert!(approx(cache.layers[0].k.at(3, 0), 0.0));

        cache.layers[0].append(&k, &v);
        assert_eq!(cache.layers[0].seq_len, 6);

        cache.clear();
        assert_eq!(cache.layers[0].seq_len, 0);
        cache.free();
        assert_eq!(cache.layers[0].k.rows, 0);
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    #[test]
    fn test_sample_large_vocab_picks_argmax_when_greedy() {
        let mut logits = Mat::zeros(1, 100);
        *logits.at_mut(0, 57) = 5.0;
        let p = SamplingParams { temperature: 0.0, ..default_params() };
        let mut rng = LcgRng::new(1);
        assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut rng), 57);
    }

    #[test]
    fn test_sample_large_vocab_honours_vocabulary_mask() {
        // The highest logit sits outside the allowed range, so it must not win.
        let mut logits = Mat::zeros(1, 100);
        *logits.at_mut(0, 5) = 100.0; // masked out
        *logits.at_mut(0, 60) = 1.0; // best allowed
        let mut p = SamplingParams {
            temperature: 0.0,
            allowed_min: Some(50),
            allowed_max: Some(70),
            ..default_params()
        };
        let mut rng = LcgRng::new(1);
        assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut rng), 60);

        // allowed_extra reaches back outside the range, for stop markers.
        p.allowed_extra = vec![5];
        let mut rng2 = LcgRng::new(1);
        assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut rng2), 5);
    }

    #[test]
    fn test_sample_large_vocab_never_draws_masked_token() {
        let logits = Mat::from_fn(1, 500, |_, c| (c % 7) as f32);
        let p = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            allowed_min: Some(100),
            allowed_max: Some(120),
            ..default_params()
        };
        let mut rng = LcgRng::new(99);
        for _ in 0..200 {
            let id = sample_token_large_vocab(&logits, 0, &p, &[], &mut rng);
            assert!(id >= 100);
            assert!(id < 120);
        }
    }

    #[test]
    fn test_sample_large_vocab_respects_top_k() {
        let mut logits = Mat::zeros(1, 1000);
        *logits.at_mut(0, 10) = 10.0;
        *logits.at_mut(0, 20) = 9.0;
        let p = SamplingParams { temperature: 1.0, top_k: 2, top_p: 1.0, ..default_params() };
        let mut rng = LcgRng::new(3);
        for _ in 0..100 {
            let id = sample_token_large_vocab(&logits, 0, &p, &[], &mut rng);
            assert!(id == 10 || id == 20);
        }
    }

    #[test]
    fn test_sample_large_vocab_collapses_at_top_p_near_zero() {
        let mut logits = Mat::zeros(1, 1000);
        *logits.at_mut(0, 42) = 20.0;
        let p = SamplingParams { temperature: 1.0, top_p: 0.01, ..default_params() };
        let mut rng = LcgRng::new(5);
        for _ in 0..50 {
            assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut rng), 42);
        }
    }

    #[test]
    fn test_sample_large_vocab_applies_repetition_penalty() {
        let mut logits = Mat::zeros(1, 100);
        *logits.at_mut(0, 7) = 2.0;
        *logits.at_mut(0, 8) = 1.9;
        let p = SamplingParams { temperature: 0.0, repetition_penalty: 2.0, ..default_params() };
        let mut rng = LcgRng::new(1);
        // Unpenalised, 7 wins.
        assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut rng), 7);
        // Having already produced 7, its positive logit is divided and 8 takes it.
        assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[7], &mut rng), 8);
    }

    #[test]
    fn test_sample_large_vocab_is_reproducible_for_a_seed() {
        let logits = Mat::from_fn(1, 2000, |_, c| (c as f32 * 0.01).sin() * 3.0);
        let p = SamplingParams { temperature: 0.9, top_p: 0.9, ..default_params() };

        let mut first = Vec::new();
        let mut a = LcgRng::new(1234);
        for _ in 0..20 {
            first.push(sample_token_large_vocab(&logits, 0, &p, &[], &mut a));
        }
        let mut b = LcgRng::new(1234);
        for want in first {
            assert_eq!(sample_token_large_vocab(&logits, 0, &p, &[], &mut b), want);
        }
    }
}
