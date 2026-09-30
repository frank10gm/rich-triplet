//! # OmniVoice LM -- a Qwen3 backbone used as a masked diffusion model
//!
//! Every other transformer in this project is autoregressive: sample a token,
//! append it to a KV cache, run one more position. This one is not, and the
//! difference reaches all the way down into the attention kernel.
//!
//! OmniVoice starts with every audio position masked, runs the **whole sequence**
//! through the model, unmasks the positions it is most confident about, and
//! repeats -- 32 times by default. So:
//!
//!   * **Attention is bidirectional.** A masked position has to see the
//!     positions after it, or there would be nothing to condition on. Every
//!     attention path elsewhere in this project is causal, so this needs its
//!     own.
//!   * **There is no KV cache.** Unmasking a position changes the hidden states
//!     of every position that attends to it, which under a full mask is all of
//!     them. Nothing carries between steps.
//!   * **The head predicts eight codebooks at once.** Logits are
//!     `[T, 8, 1025]`, produced by a single `[1024 -> 8200]` matmul and
//!     reshaped, not by eight separate heads.
//!
//! The backbone itself is an ordinary Qwen3 0.6B: 28 blocks, hidden 1024, 16
//! query heads over 8 KV heads, head_dim 128, SwiGLU, and per-head RMSNorm on Q
//! and K -- the same `apply_per_head_norm` shape Gemma 3 uses.
//!
//! ## Audio tokens
//!
//! Eight codebooks of 1024 entries share one embedding table of 8200 rows, where
//! row `i * 1025 + c` is code `c` of codebook `i`. The extra entry per codebook
//! is the mask token, id 1024. A position's embedding is the **sum** across all
//! eight codebooks, so a fully masked position still has a well-defined
//! embedding -- the sum of the eight mask rows.
//!
//! ## RoPE pairing
//!
//! The same fork that made Orpheus emit noise for its first few tokens applies
//! here: rotary embeddings can pair `i` with `i + head_dim/2` (half-split) or
//! `2i` with `2i+1` (interleaved), and which is right depends on whether the
//! GGUF converter permuted the Q and K weight rows. llama.cpp permutes for the
//! `llama` architecture and not for Qwen3, which predicts half-split here.
//!
//! **Half-split is confirmed by listening.** Both settings produce output that
//! passes every automated check -- finite, in range, speech-like statistics --
//! because the two conventions rotate by the same angles and differ only in
//! which pairs receive them. Only a human could separate them: half-split is
//! intelligible speech, interleaved is unintelligible. The one metric that did
//! hint was the interleaved output's DC offset of -0.03 against half-split's
//! -0.0001, with a zero-crossing rate of 0.009 -- rumble rather than voice.
//!
//! The flag stays because it is the first thing to try when a new checkpoint
//! sounds wrong, not because the answer is open.
//!
//! ## Weight loading from GGUF
//!
//! The `omnivoice-lm` architecture maps onto the model fields as:
//!
//! ```text
//!   llm.embed_tokens.weight                    -> text_embed
//!   audio_embeddings.weight                    -> audio_embed
//!   audio_heads.weight                         -> audio_head
//!   llm.norm.weight                            -> norm.gamma
//!   llm.layers.{i}.input_layernorm.weight      -> input_layernorm.gamma
//!   llm.layers.{i}.post_attention_layernorm.*  -> post_attention_layernorm.gamma
//!   llm.layers.{i}.self_attn.{q,k,v,o}_proj.*  -> self_attn projections
//!   llm.layers.{i}.self_attn.{q,k}_norm.weight -> per-head Q/K norms
//!   llm.layers.{i}.mlp.{gate,up,down}_proj.*   -> mlp projections
//! ```
//!
//! Norm gammas are stored verbatim -- the `1 + gamma` convention belongs to
//! Gemma alone.
//!
//! The two audio tensors are both `[8200, 1024]`, which is
//! `num_audio_codebook * audio_vocab_size` rows: eight blocks of 1025, one per
//! codebook, each holding 1024 codes plus a mask token.

#![allow(dead_code)]

use std::sync::Arc;

use crate::autograd2::{Mat, MatBf16, TensorNode};
use crate::gguf_loader::{GgufFile, GgufType};
use crate::nn2::{Linear2, RmsNorm2};
use crate::transformer4::{f32s_to_bf16_and_drop, load_linear_from_gguf};
use crate::transformer5::{RopePairing, llama_rope};

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct Config6 {
    pub text_vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,

    /// Codebooks the head predicts in parallel.
    pub num_audio_codebook: usize,
    /// 1024 codes plus the mask token.
    pub audio_vocab_size: usize,
    /// The id standing for "not yet decided".
    pub audio_mask_id: usize,

    /// Special marker ids, read from the checkpoint metadata.
    pub text_start: usize,
    pub text_end: usize,
    pub lang_start: usize,
    pub lang_end: usize,
    pub instruct_start: usize,
    pub instruct_end: usize,
    pub denoise: usize,

    /// See the module docs. Half-split, confirmed by listening; interleaved is
    /// unintelligible despite passing every automated check.
    pub rope_pairing: RopePairing,
}

impl Default for Config6 {
    fn default() -> Self {
        Config6 {
            text_vocab_size: 151676,
            hidden_size: 1024,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            intermediate_size: 3072,
            head_dim: 128,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            num_audio_codebook: 8,
            audio_vocab_size: 1025,
            audio_mask_id: 1024,
            text_start: 151674,
            text_end: 151675,
            lang_start: 151670,
            lang_end: 151671,
            instruct_start: 151672,
            instruct_end: 151673,
            denoise: 151669,
            rope_pairing: RopePairing::HalfSplit,
        }
    }
}

impl Config6 {
    pub fn omnivoice() -> Self {
        Config6::default()
    }

    /// Rows in the shared audio embedding table.
    pub fn audio_table_size(&self) -> usize {
        self.num_audio_codebook * self.audio_vocab_size
    }

    /// Where codebook `i`'s block starts in that table.
    pub fn audio_row(&self, codebook: usize, code: usize) -> usize {
        codebook * self.audio_vocab_size + code
    }

    pub fn inv_freq(&self) -> Vec<f32> {
        let half = self.head_dim / 2;
        let mut out = vec![0.0f32; half];
        for (i, slot) in out.iter_mut().enumerate() {
            let exponent = 2.0f32 * i as f32 / self.head_dim as f32;
            *slot = 1.0f32 / self.rope_theta.powf(exponent);
        }
        out
    }
}

// =============================================================================
// Attention helpers
// =============================================================================

/// Apply an RMSNorm independently to each head's slice of a
/// [T, n_heads * head_dim] matrix.
pub fn apply_head_norm(x: &Mat, norm: &RmsNorm2, n_heads: usize, head_dim: usize) -> Mat {
    assert!(x.cols == n_heads * head_dim, "apply_head_norm: width mismatch");
    let gamma = norm.gamma.data();
    assert!(gamma.cols == head_dim, "apply_head_norm: gamma is not head_dim wide");

    let cols = x.cols;
    let mut out = Mat::zeros(x.rows, x.cols);
    for r in 0..x.rows {
        let row = &x.data[r * cols..(r + 1) * cols];
        let orow = &mut out.data[r * cols..(r + 1) * cols];
        for h in 0..n_heads {
            let base = h * head_dim;
            // Each head normalizes over its own slice, independently.
            let mut sum_sq = 0.0f32;
            for i in 0..head_dim {
                sum_sq += row[base + i] * row[base + i];
            }
            let inv = 1.0f32 / (sum_sq / head_dim as f32 + norm.eps).sqrt();
            for i in 0..head_dim {
                orow[base + i] = row[base + i] * inv * gamma.data[i];
            }
        }
    }
    out
}

/// Full bidirectional grouped-query attention.
///
/// `q` is [T, n_q_heads * d], `k` and `v` are [T, n_kv_heads * d]. No mask of
/// any kind: every query attends to every key. That is the whole point -- a
/// masked diffusion model conditions each position on both sides.
pub fn bidirectional_gqa_attention(
    q: &Mat,
    k: &Mat,
    v: &Mat,
    n_q_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
) -> Mat {
    let t = q.rows;
    let group = n_q_heads / n_kv_heads;
    let mut out = Mat::zeros(t, n_q_heads * head_dim);

    // Per-head contiguous copies so each head's scores are one sgemm. With no
    // causal mask the score matrix is dense, which is what makes this the
    // BLAS-friendly shape despite being O(T^2).
    for qh in 0..n_q_heads {
        let kvh = qh / group;
        let q_h = Mat::from_fn(t, head_dim, |r, c| q.at(r, qh * head_dim + c));
        let k_h = Mat::from_fn(t, head_dim, |r, c| k.at(r, kvh * head_dim + c));
        let v_h = Mat::from_fn(t, head_dim, |r, c| v.at(r, kvh * head_dim + c));

        // `matmul_bt` only exists with BLAS; without it the transposed matmul
        // accumulates in the same order as C++'s scalar `matmul_bt` fallback.
        #[cfg(feature = "blas")]
        let mut scores = q_h.matmul_bt(&k_h);
        #[cfg(not(feature = "blas"))]
        let mut scores = q_h.matmul(&k_h.transpose());

        for r in 0..t {
            let row = &mut scores.data[r * t..(r + 1) * t];
            let mut max_s = f32::NEG_INFINITY;
            for s in row.iter_mut() {
                *s *= scale;
                // `std::max(max_s, s)`: keeps max_s on ties.
                if max_s < *s {
                    max_s = *s;
                }
            }
            let mut sum_exp = 0.0f32;
            for s in row.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            if sum_exp > 0.0 {
                for s in row.iter_mut() {
                    *s /= sum_exp;
                }
            }
        }

        let head_out = scores.matmul(&v_h);
        for r in 0..t {
            for c in 0..head_dim {
                *out.at_mut(r, qh * head_dim + c) = head_out.at(r, c);
            }
        }
    }
    out
}

// =============================================================================
// OmniAttention
// =============================================================================

/// Bidirectional grouped-query attention with per-head Q/K normalization.
///
/// No cache and no causal mask: every position attends to every other.
pub struct OmniAttention {
    pub q_proj: Linear2,
    pub k_proj: Linear2,
    pub v_proj: Linear2,
    pub o_proj: Linear2,
    /// Per-head RMSNorm over `head_dim`, applied before RoPE.
    pub q_norm: RmsNorm2,
    pub k_norm: RmsNorm2,
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub attn_scale: f32,
    pub rope_pairing: RopePairing,
}

impl OmniAttention {
    pub fn new(cfg: &Config6) -> Self {
        OmniAttention {
            q_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim),
            k_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            v_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim),
            o_proj: Linear2::new_no_bias_zeros(cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size),
            q_norm: RmsNorm2::new_with_eps(cfg.head_dim, cfg.rms_norm_eps),
            k_norm: RmsNorm2::new_with_eps(cfg.head_dim, cfg.rms_norm_eps),
            n_q_heads: cfg.num_attention_heads,
            n_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            attn_scale: 1.0f32 / (cfg.head_dim as f32).sqrt(),
            rope_pairing: cfg.rope_pairing,
        }
    }

    /// `inv_freq` is passed in rather than held.
    ///
    /// The C++ original first cached a pointer into the owning model, and
    /// returning the model by value left every layer pointing at a moved-from
    /// vector. Passing the slice down removes the lifetime question entirely.
    pub fn forward(&self, x: &Mat, inv_freq: &[f32]) -> Mat {
        assert!(
            inv_freq.len() >= self.head_dim / 2,
            "omnivoice lm: not enough inverse frequencies"
        );
        // Every projection here runs on a BF16 weight, so `fused_linear` returns a
        // plain leaf and there is no graph to break. See `OmniBlock::forward` for
        // the ops that do build one.
        let xn = TensorNode::leaf(x.clone());

        // Qwen3 normalizes each head of Q and K before rotating, like Gemma 3.
        let q_normed =
            apply_head_norm(&self.q_proj.forward(&xn).data(), &self.q_norm, self.n_q_heads, self.head_dim);
        let k_normed =
            apply_head_norm(&self.k_proj.forward(&xn).data(), &self.k_norm, self.n_kv_heads, self.head_dim);

        // Positions are absolute from 0: there is no cache, so the whole sequence
        // is rotated every pass.
        let q = llama_rope(&q_normed, self.n_q_heads, self.head_dim, 0, inv_freq, self.rope_pairing);
        let k = llama_rope(&k_normed, self.n_kv_heads, self.head_dim, 0, inv_freq, self.rope_pairing);
        let v = self.v_proj.forward(&xn).data().clone();

        let attn = bidirectional_gqa_attention(
            &q,
            &k,
            &v,
            self.n_q_heads,
            self.n_kv_heads,
            self.head_dim,
            self.attn_scale,
        );
        self.o_proj.forward(&TensorNode::leaf(attn)).data().clone()
    }
}

// =============================================================================
// OmniMlp / OmniBlock
// =============================================================================

pub struct OmniMlp {
    pub gate_proj: Linear2,
    pub up_proj: Linear2,
    pub down_proj: Linear2,
}

impl OmniMlp {
    pub fn new(cfg: &Config6) -> Self {
        OmniMlp {
            gate_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            up_proj: Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.intermediate_size),
            down_proj: Linear2::new_no_bias_zeros(cfg.intermediate_size, cfg.hidden_size),
        }
    }

    pub fn forward(&self, x: &Mat) -> Mat {
        let xn = TensorNode::leaf(x.clone());
        let gate = self.gate_proj.forward(&xn).silu();
        let fused = gate.mul_elem_node(&self.up_proj.forward(&xn));
        let out = self.down_proj.forward(&fused).data().clone();

        // `silu` and `mul_elem_node` always wire up a backward closure, and that
        // closure captures its own node -- a reference cycle plain refcounting can
        // never collect. Inference never walks it, so without this the SwiGLU
        // intermediates of every layer of every pass stay resident: at 550
        // positions that is ~13 GB over a single clip, and the kernel kills the
        // process somewhere past a thousand.
        fused.free_graph();
        out
    }
}

pub struct OmniBlock {
    pub input_layernorm: RmsNorm2,
    pub self_attn: OmniAttention,
    pub post_attention_layernorm: RmsNorm2,
    pub mlp: OmniMlp,
}

impl OmniBlock {
    pub fn new(cfg: &Config6) -> Self {
        OmniBlock {
            input_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            self_attn: OmniAttention::new(cfg),
            post_attention_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            mlp: OmniMlp::new(cfg),
        }
    }

    pub fn forward(&self, x: &Mat, inv_freq: &[f32]) -> Mat {
        // Same cycle as the MLP's: `rms_norm` captures its own output node in the
        // backward closure it always builds. Copying the data out and freeing the
        // graph immediately is what keeps a forward-only pass flat in memory.
        let norm1 = self.input_layernorm.forward(&TensorNode::leaf(x.clone()));
        let normed = norm1.data().clone();
        norm1.free_graph();

        let mut h = self.self_attn.forward(&normed, inv_freq);
        h.add_assign(x);

        let norm2 = self.post_attention_layernorm.forward(&TensorNode::leaf(h.clone()));
        let normed2 = norm2.data().clone();
        norm2.free_graph();

        let mut ff = self.mlp.forward(&normed2);
        ff.add_assign(&h);
        ff
    }
}

// =============================================================================
// Model
// =============================================================================

/// One position of the model's input.
///
/// A position is either a text token or a stack of `num_audio_codebook` audio
/// codes; the audio codes may be the mask id. Keeping both in one struct means
/// the sequence is a single flat vector and the audio mask is derived rather
/// than tracked separately.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OmniToken {
    /// Text token id, used when `audio` is empty.
    pub text_id: usize,
    /// One code per codebook, or empty for a text position.
    pub audio: Vec<u32>,
}

impl OmniToken {
    pub fn is_audio(&self) -> bool {
        !self.audio.is_empty()
    }

    pub fn text(id: usize) -> Self {
        OmniToken { text_id: id, audio: Vec::new() }
    }

    pub fn masked(codebooks: usize, mask_id: usize) -> Self {
        OmniToken { text_id: 0, audio: vec![mask_id as u32; codebooks] }
    }
}

// =============================================================================
// Backends
// =============================================================================

/// Anything that can run one full-sequence forward pass.
///
/// `OmniLm` is one; the Metal context is the other, and the diffusion loop
/// does not need to know which it has. The interface is deliberately the whole
/// pass rather than a layer at a time: a masked diffusion model caches nothing
/// between steps, so there is no state for a backend to hold across calls and
/// nothing finer worth abstracting.
pub trait OmniForward {
    /// Audio logits for every position, `[T, codebooks * vocab]`.
    fn forward(&self, tokens: &[OmniToken]) -> Result<Mat, String>;
}

pub struct OmniLm {
    pub config: Config6,
    /// [text_vocab_size, hidden_size]
    pub text_embed: Option<MatBf16>,
    /// [audio_table_size, hidden_size]
    pub audio_embed: Option<MatBf16>,
    pub layers: Vec<OmniBlock>,
    pub norm: RmsNorm2,
    /// [audio_table_size, hidden_size]; logits reshape to [T, codebooks, vocab].
    pub audio_head: Linear2,
    pub inv_freq_cache: Vec<f32>,
}

impl OmniLm {
    pub fn new(cfg: Config6) -> Self {
        let norm = RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps);
        let audio_head = Linear2::new_no_bias_zeros(cfg.hidden_size, cfg.audio_table_size());
        let layers = (0..cfg.num_hidden_layers).map(|_| OmniBlock::new(&cfg)).collect();
        let mut model = OmniLm {
            config: cfg,
            text_embed: None,
            audio_embed: None,
            layers,
            norm,
            audio_head,
            inv_freq_cache: Vec::new(),
        };
        model.refresh_inv_freq();
        model
    }

    /// Recompute the cached inverse frequencies from the config.
    pub fn refresh_inv_freq(&mut self) {
        self.inv_freq_cache = self.config.inv_freq();
        for layer in &mut self.layers {
            layer.self_attn.rope_pairing = self.config.rope_pairing;
        }
    }

    /// Embed a sequence: text rows straight from the table, audio rows summed
    /// across all eight codebooks.
    pub fn embed(&self, tokens: &[OmniToken]) -> Result<Mat, String> {
        if tokens.is_empty() {
            return Err("omnivoice lm: empty sequence".into());
        }
        let (Some(text_embed), Some(audio_embed)) = (&self.text_embed, &self.audio_embed) else {
            return Err("omnivoice lm: embeddings not loaded".into());
        };

        let h = self.config.hidden_size;
        let text_bits: &[u16] = &text_embed.data;
        let audio_bits: &[u16] = &audio_embed.data;

        let mut out = Mat::zeros(tokens.len(), h);
        for (t, tok) in tokens.iter().enumerate() {
            let row = &mut out.data[t * h..(t + 1) * h];

            if !tok.is_audio() {
                if tok.text_id >= text_embed.rows {
                    return Err(format!(
                        "omnivoice lm: text id {} is outside the vocabulary",
                        tok.text_id
                    ));
                }
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot = MatBf16::bf16_to_f32(text_bits[tok.text_id * h + c]);
                }
                continue;
            }

            if tok.audio.len() != self.config.num_audio_codebook {
                return Err(format!(
                    "omnivoice lm: audio position {} carries {} codes, expected {}",
                    t,
                    tok.audio.len(),
                    self.config.num_audio_codebook
                ));
            }
            // Sum across codebooks. A fully masked position is the sum of the eight
            // mask rows, which is a perfectly well-defined embedding -- that is what
            // lets decoding start from nothing.
            for i in 0..self.config.num_audio_codebook {
                let code = tok.audio[i];
                if code as usize >= self.config.audio_vocab_size {
                    return Err(format!(
                        "omnivoice lm: audio code {} at codebook {} is outside the codebook",
                        code, i
                    ));
                }
                let base = self.config.audio_row(i, code as usize) * h;
                for (c, slot) in row.iter_mut().enumerate() {
                    *slot += MatBf16::bf16_to_f32(audio_bits[base + c]);
                }
            }
        }
        Ok(out)
    }

    /// Run the whole sequence and return audio logits, `[T, codebooks * vocab]`.
    ///
    /// Column `i * audio_vocab_size + c` is the logit for code `c` of codebook
    /// `i` at that position.
    pub fn forward(&self, tokens: &[OmniToken]) -> Result<Mat, String> {
        let mut h = self.embed(tokens)?;
        for layer in &self.layers {
            h = layer.forward(&h, &self.inv_freq_cache);
        }

        // Unlike an autoregressive model there is no "last position" shortcut:
        // every position's logits matter, because any of them might be unmasked
        // this step.
        let final_norm = self.norm.forward(&TensorNode::leaf(h));
        let normed = final_norm.data().clone();
        final_norm.free_graph();
        Ok(self.audio_head.forward(&TensorNode::leaf(normed)).data().clone())
    }

    pub fn weight_bytes(&self) -> usize {
        let mut n = linear_bytes(&self.audio_head);
        if let Some(e) = &self.text_embed {
            n += e.size_bytes();
        }
        if let Some(e) = &self.audio_embed {
            n += e.size_bytes();
        }
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

    /// Requantize every BF16 projection to Q4_K.
    pub fn quantize_projections_to_q4k(&mut self) -> usize {
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
        convert(&mut self.audio_head);
        converted
    }

    /// Load the `omnivoice-lm` GGUF. Geometry and special ids come from the
    /// file's metadata, falling back to `cfg`.
    pub fn load(path: &str, mut cfg: Config6) -> Result<OmniLm, String> {
        let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;

        if let Some(arch) = gguf.metadata.get("general.architecture") {
            if let Some(name) = arch.as_str() {
                if name != "omnivoice-lm" {
                    return Err(format!(
                        "omnivoice lm: GGUF declares architecture '{}', expected 'omnivoice-lm'",
                        name
                    ));
                }
            }
        }

        // Geometry comes from the file; a mismatch is an error rather than a
        // silent reinterpretation.
        cfg.num_hidden_layers = meta_u64(&gguf, "omnivoice-lm.block_count", cfg.num_hidden_layers);
        cfg.hidden_size = meta_u64(&gguf, "omnivoice-lm.embedding_length", cfg.hidden_size);
        cfg.num_attention_heads =
            meta_u64(&gguf, "omnivoice-lm.attention.head_count", cfg.num_attention_heads);
        cfg.num_key_value_heads =
            meta_u64(&gguf, "omnivoice-lm.attention.head_count_kv", cfg.num_key_value_heads);
        cfg.intermediate_size =
            meta_u64(&gguf, "omnivoice-lm.feed_forward_length", cfg.intermediate_size);
        cfg.head_dim = meta_u64(&gguf, "omnivoice-lm.attention.key_length", cfg.head_dim);
        cfg.text_vocab_size = meta_u64(&gguf, "omnivoice-lm.vocab_size", cfg.text_vocab_size);
        cfg.rope_theta = meta_f32(&gguf, "omnivoice-lm.rope.freq_base", cfg.rope_theta);
        cfg.rms_norm_eps =
            meta_f32(&gguf, "omnivoice-lm.attention.layer_norm_rms_epsilon", cfg.rms_norm_eps);

        cfg.num_audio_codebook = meta_u64(&gguf, "omnivoice.num_audio_codebook", cfg.num_audio_codebook);
        cfg.audio_vocab_size = meta_u64(&gguf, "omnivoice.audio_vocab_size", cfg.audio_vocab_size);
        cfg.audio_mask_id = meta_u64(&gguf, "omnivoice.audio_mask_id", cfg.audio_mask_id);
        cfg.text_start = meta_u64(&gguf, "omnivoice.special.text_start", cfg.text_start);
        cfg.text_end = meta_u64(&gguf, "omnivoice.special.text_end", cfg.text_end);
        cfg.lang_start = meta_u64(&gguf, "omnivoice.special.lang_start", cfg.lang_start);
        cfg.lang_end = meta_u64(&gguf, "omnivoice.special.lang_end", cfg.lang_end);
        cfg.instruct_start = meta_u64(&gguf, "omnivoice.special.instruct_start", cfg.instruct_start);
        cfg.instruct_end = meta_u64(&gguf, "omnivoice.special.instruct_end", cfg.instruct_end);
        cfg.denoise = meta_u64(&gguf, "omnivoice.special.denoise", cfg.denoise);

        if cfg.audio_mask_id >= cfg.audio_vocab_size {
            return Err(format!(
                "omnivoice lm: mask id {} is outside the audio vocabulary of {}",
                cfg.audio_mask_id, cfg.audio_vocab_size
            ));
        }

        eprintln!(
            "[ GGUF ] omnivoice-lm: {} layers, hidden {}, {}/{} heads, ffn {}, text vocab {}, {} codebooks x {}",
            cfg.num_hidden_layers,
            cfg.hidden_size,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.intermediate_size,
            cfg.text_vocab_size,
            cfg.num_audio_codebook,
            cfg.audio_vocab_size
        );

        let mut model = OmniLm::new(cfg);

        let mut loaded = 0usize;
        for idx in 0..gguf.tensor_info.len() {
            let name = gguf.tensor_info[idx].name.as_str();

            if name == "llm.embed_tokens.weight" {
                let table = load_embedding(
                    &gguf,
                    idx,
                    model.config.text_vocab_size,
                    model.config.hidden_size,
                )?;
                model.text_embed = Some(table);
                loaded += 1;
                continue;
            }
            if name == "audio_embeddings.weight" {
                let table = load_embedding(
                    &gguf,
                    idx,
                    model.config.audio_table_size(),
                    model.config.hidden_size,
                )?;
                model.audio_embed = Some(table);
                loaded += 1;
                continue;
            }
            if name == "audio_heads.weight" {
                load_linear_from_gguf(&gguf, idx, &mut model.audio_head).map_err(|e| e.to_string())?;
                loaded += 1;
                continue;
            }
            if name == "llm.norm.weight" {
                load_norm(&gguf, idx, &mut model.norm, model.config.hidden_size)?;
                loaded += 1;
                continue;
            }

            let Some((layer_idx, field)) = split_layer_name(name) else {
                eprintln!("[ GGUF ] Note: ignoring unrecognised tensor '{}'", name);
                continue;
            };
            if layer_idx >= model.layers.len() {
                return Err(format!(
                    "omnivoice lm: tensor '{}' names layer {} but the model has {}",
                    name,
                    layer_idx,
                    model.layers.len()
                ));
            }
            let hidden = model.config.hidden_size;
            let head_dim = model.config.head_dim;
            let layer = &mut model.layers[layer_idx];
            let io = |r: std::io::Result<()>| r.map_err(|e| e.to_string());

            match field {
                "input_layernorm.weight" => load_norm(&gguf, idx, &mut layer.input_layernorm, hidden)?,
                "post_attention_layernorm.weight" => {
                    load_norm(&gguf, idx, &mut layer.post_attention_layernorm, hidden)?
                }
                // Per head, so this is head_dim wide rather than hidden_size.
                "self_attn.q_norm.weight" => load_norm(&gguf, idx, &mut layer.self_attn.q_norm, head_dim)?,
                "self_attn.k_norm.weight" => load_norm(&gguf, idx, &mut layer.self_attn.k_norm, head_dim)?,
                "self_attn.q_proj.weight" => {
                    io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.q_proj))?
                }
                "self_attn.k_proj.weight" => {
                    io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.k_proj))?
                }
                "self_attn.v_proj.weight" => {
                    io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.v_proj))?
                }
                "self_attn.o_proj.weight" => {
                    io(load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.o_proj))?
                }
                "mlp.gate_proj.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.gate_proj))?,
                "mlp.up_proj.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.up_proj))?,
                "mlp.down_proj.weight" => io(load_linear_from_gguf(&gguf, idx, &mut layer.mlp.down_proj))?,
                _ => {
                    eprintln!("[ GGUF ] Note: ignoring unrecognised field '{}'", name);
                    continue;
                }
            }
            loaded += 1;
        }

        if model.text_embed.is_none() {
            return Err("omnivoice lm: GGUF has no llm.embed_tokens.weight".into());
        }
        if model.audio_embed.is_none() {
            return Err("omnivoice lm: GGUF has no audio_embeddings.weight".into());
        }

        model.refresh_inv_freq();
        eprintln!(
            "[ GGUF ] Loaded {} tensors, {:.2} GB resident",
            loaded,
            model.weight_bytes() as f64 / 1e9
        );
        Ok(model)
    }
}

impl OmniForward for OmniLm {
    fn forward(&self, tokens: &[OmniToken]) -> Result<Mat, String> {
        OmniLm::forward(self, tokens)
    }
}

// =============================================================================
// GGUF helpers
// =============================================================================

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

fn load_norm(gguf: &GgufFile, idx: usize, norm: &mut RmsNorm2, expect: usize) -> Result<(), String> {
    let gtype = gguf.tensor_info[idx].gguf_type;
    let values = match gtype {
        GgufType::F32 => gguf.decode_f32(idx).map_err(|e| e.to_string())?,
        GgufType::F16 => gguf.decode_f16_to_f32(idx).map_err(|e| e.to_string())?,
        _ => {
            return Err(format!(
                "omnivoice lm: norm '{}' has unsupported type {:?}",
                gguf.tensor_info[idx].name, gtype
            ));
        }
    };
    if values.len() != expect {
        return Err(format!(
            "omnivoice lm: norm '{}' has {} entries, expected {}",
            gguf.tensor_info[idx].name,
            values.len(),
            expect
        ));
    }
    norm.gamma.set_data(Mat::new(values, 1, expect));
    Ok(())
}

/// Read an embedding table into BF16, whatever it was stored as.
fn load_embedding(gguf: &GgufFile, idx: usize, rows: usize, cols: usize) -> Result<MatBf16, String> {
    let gtype = gguf.tensor_info[idx].gguf_type;
    let widen = |decoded: std::io::Result<Vec<f32>>| -> Result<MatBf16, String> {
        let f32s = decoded.map_err(|e| e.to_string())?;
        if f32s.len() != rows * cols {
            return Err(format!(
                "omnivoice lm: '{}' holds {} values, expected {}",
                gguf.tensor_info[idx].name,
                f32s.len(),
                rows * cols
            ));
        }
        Ok(MatBf16 { data: Arc::new(f32s_to_bf16_and_drop(f32s)), rows, cols })
    };

    match gtype {
        GgufType::Bf16 => {
            let bits = gguf.decode_bf16(idx).map_err(|e| e.to_string())?;
            Ok(MatBf16 { data: Arc::new(bits), rows, cols })
        }
        GgufType::F16 => widen(gguf.decode_f16_to_f32(idx)),
        GgufType::F32 => widen(gguf.decode_f32(idx)),
        GgufType::Q8_0 => widen(gguf.decode_q8_0_to_f32(idx)),
        GgufType::Q4K => widen(gguf.decode_q4k_to_f32(idx)),
        GgufType::Q6K => widen(gguf.decode_q6k_to_f32(idx)),
        GgufType::Q5K => widen(gguf.decode_q5k_to_f32(idx)),
        GgufType::Q4_0 => widen(gguf.decode_q4_0_to_f32(idx)),
        _ => Err(format!(
            "omnivoice lm: '{}' has unsupported type {:?}",
            gguf.tensor_info[idx].name, gtype
        )),
    }
}

/// Parse `llm.layers.{i}.` and return the layer index plus the remaining field.
fn split_layer_name(name: &str) -> Option<(usize, &str)> {
    const PREFIX: &str = "llm.layers.";
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

fn meta_u64(gguf: &GgufFile, key: &str, fallback: usize) -> usize {
    match gguf.metadata.get(key) {
        None => fallback,
        Some(v) => v.as_u64().map(|v| v as usize).unwrap_or(fallback),
    }
}

fn meta_f32(gguf: &GgufFile, key: &str, fallback: f32) -> f32 {
    match gguf.metadata.get(key) {
        None => fallback,
        Some(v) => v.as_f32().unwrap_or(fallback),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const LM_PATH: &str = "models/omnivoice-base-Q8_0.gguf";

    fn load_lm() -> Option<OmniLm> {
        if !std::path::Path::new(LM_PATH).exists() {
            eprintln!("skip: {} not present", LM_PATH);
            return None;
        }
        match OmniLm::load(LM_PATH, Config6::omnivoice()) {
            Ok(lm) => Some(lm),
            Err(e) => panic!("{}", e),
        }
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    #[test]
    fn test_config6_matches_omnivoice_checkpoint() {
        let c = Config6::omnivoice();
        // A Qwen3 0.6B backbone.
        assert_eq!(c.num_hidden_layers, 28);
        assert_eq!(c.hidden_size, 1024);
        assert_eq!(c.num_attention_heads, 16);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.rope_theta, 1000000.0);

        // Eight codebooks of 1024 codes plus a mask token each.
        assert_eq!(c.num_audio_codebook, 8);
        assert_eq!(c.audio_vocab_size, 1025);
        assert_eq!(c.audio_mask_id, 1024);
        assert_eq!(c.audio_table_size(), 8200);
    }

    #[test]
    fn test_audio_rows_are_one_contiguous_block_per_codebook() {
        let c = Config6::omnivoice();
        // Codebook i occupies rows [i*1025, (i+1)*1025).
        assert_eq!(c.audio_row(0, 0), 0);
        assert_eq!(c.audio_row(0, 1024), 1024); // codebook 0's mask token
        assert_eq!(c.audio_row(1, 0), 1025);
        assert_eq!(c.audio_row(7, 1024), 8199); // the last row of the table
        assert_eq!(c.audio_row(7, 1024) + 1, c.audio_table_size());
    }

    // -------------------------------------------------------------------------
    // Tokens
    // -------------------------------------------------------------------------

    #[test]
    fn test_omni_token_distinguishes_text_from_audio() {
        let t = OmniToken::text(1234);
        assert!(!t.is_audio());
        assert_eq!(t.text_id, 1234);

        let a = OmniToken::masked(8, 1024);
        assert!(a.is_audio());
        assert_eq!(a.audio.len(), 8);
        for &v in &a.audio {
            assert_eq!(v, 1024);
        }
    }

    // -------------------------------------------------------------------------
    // The real model
    // -------------------------------------------------------------------------

    #[test]
    fn test_omni_lm_loads_and_embeds() {
        let Some(lm) = load_lm() else { return };
        assert_eq!(lm.layers.len(), 28);
        assert!(lm.text_embed.is_some());
        assert!(lm.audio_embed.is_some());
        assert_eq!(lm.text_embed.as_ref().unwrap().rows, lm.config.text_vocab_size);
        assert_eq!(lm.audio_embed.as_ref().unwrap().rows, lm.config.audio_table_size());
        assert_eq!(lm.inv_freq_cache.len(), lm.config.head_dim / 2);

        // A masked audio position is the sum of the eight mask rows, which is what
        // lets decoding start from nothing.
        let seq = vec![
            OmniToken::text(1000),
            OmniToken::masked(lm.config.num_audio_codebook, lm.config.audio_mask_id),
        ];
        let e = lm.embed(&seq).expect("embed");
        assert_eq!(e.rows, 2);
        assert_eq!(e.cols, lm.config.hidden_size);
        let nonzero = (0..e.cols).any(|c| e.at(1, c) != 0.0);
        assert!(nonzero);
    }

    #[test]
    fn test_omni_lm_rejects_malformed_positions() {
        let Some(lm) = load_lm() else { return };

        // Too few codebooks for an audio position.
        let bad = OmniToken { text_id: 0, audio: vec![1, 2, 3] };
        assert!(lm.embed(&[bad]).is_err());

        // A code past the end of a codebook.
        let mut oob = OmniToken::masked(lm.config.num_audio_codebook, lm.config.audio_mask_id);
        oob.audio[0] = 5000;
        assert!(lm.embed(&[oob]).is_err());

        // A text id past the end of the vocabulary.
        assert!(lm.embed(&[OmniToken::text(lm.config.text_vocab_size + 1)]).is_err());

        assert!(lm.embed(&[]).is_err());
    }

    #[test]
    fn test_omni_lm_produces_logits_for_every_position() {
        // Unlike an autoregressive model there is no last-position shortcut: any
        // position might be unmasked this step, so all of them need logits.
        let Some(lm) = load_lm() else { return };

        let mut seq = Vec::new();
        for i in 0..4 {
            seq.push(OmniToken::text(1000 + i));
        }
        for _ in 0..4 {
            seq.push(OmniToken::masked(lm.config.num_audio_codebook, lm.config.audio_mask_id));
        }

        let logits = lm.forward(&seq).expect("forward");
        assert_eq!(logits.rows, seq.len());
        assert_eq!(logits.cols, lm.config.audio_table_size());
        for &v in &logits.data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_omni_lm_is_deterministic() {
        let Some(lm) = load_lm() else { return };
        let seq = vec![
            OmniToken::text(500),
            OmniToken::masked(lm.config.num_audio_codebook, lm.config.audio_mask_id),
        ];
        let a = lm.forward(&seq).expect("forward a");
        let b = lm.forward(&seq).expect("forward b");
        for i in 0..a.data.len() {
            assert_eq!(a.data[i], b.data[i]);
        }
    }
}
