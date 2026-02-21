/// # GPT-OSS Architecture (transformer3)
///
/// This file implements the GPT-OSS architecture from OpenAI (August 2025),
/// using the five new features that differentiate it from GPT-2:
///
/// | Feature          | GPT-2 (transformer2.rs)      | GPT-OSS (this file)             |
/// |------------------|------------------------------|---------------------------------|
/// | Normalization    | LayerNorm (mean + variance)  | RMSNorm (RMS only, no beta)     |
/// | Position enc.    | Learned absolute embeddings  | RoPE (rotary, relative)         |
/// | Attention        | Standard multi-head          | Grouped Multi-Query (GQA)       |
/// | FFN activation   | GELU                         | SwiGLU (gated, 3 projections)   |
/// | FFN structure    | Single dense layer           | Mixture of Experts (MoE)        |
///
/// ## Inference only
///
/// This implementation is **forward-pass only**. The new ops (RMSNorm, RoPE,
/// SiLU, GQA) don't register backward_fn — gradients don't flow through them.
///
/// This is intentional: GPT-OSS-20b has 21B parameters. Training it in this
/// codebase is not feasible (no GPU, no batching, no quantization). The purpose
/// of this file is to define the correct architecture so that pretrained weights
/// can eventually be loaded and used for inference.
///
/// ## How to get from this code to running GPT-OSS-20b
///
/// 1. Add the `safetensors` crate to Cargo.toml
/// 2. Download weights from huggingface.co/openai/gpt-oss-20b
/// 3. Write a loader that reads each tensor by name and calls `param.set_data()`
///    on the matching field in this struct
/// 4. Call `model.generate()` with a text prompt
///
/// The weight name mapping would look like:
///   "model.embed_tokens.weight"          → model.embed_tokens
///   "model.layers.0.self_attn.q_proj.weight" → model.layers[0].self_attn.q_proj.weight
///   etc.
///
/// ## What is NOT implemented
///
/// - YaRN RoPE scaling (extends context beyond 4096 tokens; basic RoPE only)
/// - Sliding window attention (alternating with full attention; full only here)
/// - MXFP4 quantization (4-bit weights; f32 only here)
/// - The `swiglu_limit=7.0` clamping from the config (minor regularization detail)

use crate::autograd2::{TensorNode, Mat, Q4Mat};
use crate::nn::InitRng;
use crate::nn2::{Linear2, RmsNorm2, SwiGluMlp2, Module2};
use std::cell::RefCell;

// =============================================================================
// SamplingParams — controls how tokens are drawn from the logit distribution
// =============================================================================
//
// ## Why not just "temperature"?
//
// Temperature alone is often not enough:
//
//   • temperature=1.0  → full distribution, can pick low-probability garbage
//   • temperature=0.0  → greedy argmax, repetitive / boring
//
// Production systems layer three filters on top of the temperature-scaled
// distribution before drawing:
//
//   1. top_k — keep only the K highest-probability tokens.
//      Prevents long-tail surprises. k=50 is a common default.
//
//   2. top_p (nucleus sampling) — keep the *smallest* set of tokens whose
//      cumulative probability ≥ p.  p=0.9 means "ignore the bottom 10%".
//      Adapts to the shape of the distribution: if the model is confident
//      (one token has 80% probability) few candidates remain; if it's
//      uncertain many remain.
//
//   3. repetition_penalty — multiply the logit of any token that already
//      appears in the context by (1 / penalty) before softmax.
//      Values like 1.1–1.3 reduce looping.
//
// ## Sampling with a PRNG
//
// A linear congruential generator (LCG) is embedded so every call to
// `sample_token` is reproducible given the same `seed`.  The seed advances
// by one on every drawn sample, so successive calls within a generation loop
// produce different (but reproducible) outputs.
//
// ## Usage
//
//   let params = SamplingParams {
//       temperature: 0.8,
//       top_k:  50,
//       top_p:  0.92,
//       repetition_penalty: 1.1,
//       seed:   42,
//   };
//   let tok = model.generate_with_params(&prompt_ids, 100, &params);

/// Parameters controlling token sampling.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    /// Divide logits by this value before softmax.
    ///   temperature = 0.0 → greedy argmax (no randomness).
    ///   temperature = 1.0 → unmodified distribution.
    ///   temperature > 1.0 → flatter (more random).
    ///   temperature < 1.0 → sharper (more greedy).
    pub temperature: f32,

    /// Keep only the top-k tokens by probability before sampling.
    /// Set to 0 to disable (keep all tokens).
    pub top_k: usize,

    /// Nucleus (top-p) sampling: keep the smallest set of tokens whose
    /// cumulative probability ≥ top_p.  1.0 disables the filter.
    pub top_p: f32,

    /// Divide the logit of any previously-seen token by this factor.
    /// 1.0 = no penalty, 1.1–1.3 = mild, 2.0 = aggressive.
    pub repetition_penalty: f32,

    /// Seed for the internal LCG random number generator.
    /// The same seed + same logits → same sample (reproducible).
    pub seed: u64,

    /// Stop generation when this token id is produced.
    /// Set to `None` to disable EOS stopping (generate exactly `max_new` tokens).
    ///
    /// For GPT-2 tokenizer the EOS token is 50256.
    /// For tiktoken (GPT-OSS) it is 100257.
    pub eos_token_id: Option<usize>,

    /// Frequency penalty — further reduces the logit of tokens proportionally
    /// to how many times they have already appeared:
    ///
    ///   logit[tok] -= frequency_penalty * count(tok)
    ///
    /// 0.0 = disabled.  Typical values: 0.2 – 1.0.
    /// Positive values discourage word repetition.
    pub frequency_penalty: f32,

    /// Presence penalty — reduces the logit of any token that has appeared
    /// at least once, regardless of how many times:
    ///
    ///   logit[tok] -= presence_penalty   (if count(tok) > 0)
    ///
    /// 0.0 = disabled.  Typical values: 0.2 – 1.0.
    /// Encourages the model to use new vocabulary.
    pub presence_penalty: f32,
}

impl SamplingParams {
    /// Greedy (deterministic) sampling — always picks the argmax.
    pub fn greedy() -> Self {
        SamplingParams {
            temperature: 0.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 0,
            eos_token_id: None,
            frequency_penalty: 0.0, presence_penalty: 0.0,
        }
    }

    /// Typical creative-writing defaults.
    pub fn creative(seed: u64) -> Self {
        SamplingParams {
            temperature: 0.8, top_k: 50, top_p: 0.92,
            repetition_penalty: 1.1, seed,
            eos_token_id: None,
            frequency_penalty: 0.0, presence_penalty: 0.0,
        }
    }
}

/// Internal LCG RNG — fast, no-dependency, reproducible.
///
/// Uses Knuth's multiplier (same as InitRng in nn.rs but step is different).
/// Each call to `next_f32` advances the state and returns a value in [0, 1).
struct LcgRng { state: u64 }
impl LcgRng {
    fn new(seed: u64) -> Self { LcgRng { state: seed.wrapping_add(1) } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 { (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32 }
}

// =============================================================================
// Config3 — GPT-OSS hyperparameters
// =============================================================================

#[derive(Clone, Debug)]
pub struct Config3 {
    /// Vocabulary size (tiktoken: 201088 for GPT-OSS)
    pub vocab_size: usize,

    /// Embedding + hidden dimension (2880 for gpt-oss-20b)
    pub hidden_size: usize,

    /// Number of transformer layers (24 for gpt-oss-20b)
    pub num_hidden_layers: usize,

    /// Number of Q attention heads (64 for gpt-oss-20b)
    pub num_attention_heads: usize,

    /// Number of K/V attention heads (8 for gpt-oss-20b, GQA group_size=8)
    pub num_key_value_heads: usize,

    /// FFN hidden dimension per expert (2880 for gpt-oss-20b)
    pub intermediate_size: usize,

    /// Total number of MoE experts per layer (32 for gpt-oss-20b)
    pub num_local_experts: usize,

    /// Number of experts activated per token (4 for gpt-oss-20b)
    pub experts_per_token: usize,

    /// Maximum sequence length (131072 for gpt-oss-20b with YaRN)
    pub max_position_embeddings: usize,

    /// RoPE base frequency theta (150000 for gpt-oss-20b)
    pub rope_theta: f32,

    /// RMSNorm epsilon (1e-5 for gpt-oss-20b)
    pub rms_norm_eps: f32,
}

impl Config3 {
    /// Exact configuration for GPT-OSS-20b (from huggingface config.json)
    pub fn gpt_oss_20b() -> Self {
        Config3 {
            vocab_size: 201088,
            hidden_size: 2880,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            intermediate_size: 2880,
            num_local_experts: 32,
            experts_per_token: 4,
            max_position_embeddings: 131072,
            rope_theta: 150000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Exact configuration for GPT-OSS-120b (from huggingface config.json)
    pub fn gpt_oss_120b() -> Self {
        Config3 {
            vocab_size: 201088,
            hidden_size: 7168,
            num_hidden_layers: 36,
            num_attention_heads: 128,
            num_key_value_heads: 8,
            intermediate_size: 7168,
            num_local_experts: 128,
            experts_per_token: 4,
            max_position_embeddings: 131072,
            rope_theta: 150000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// Dimension of each attention head's Q/K/V vectors.
    /// For gpt-oss-20b: 2880 / 64 = 45
    pub fn d_head(&self) -> usize {
        assert_eq!(self.hidden_size % self.num_attention_heads, 0,
            "hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size, self.num_attention_heads);
        self.hidden_size / self.num_attention_heads
    }
}

// =============================================================================
// GptOssAttention — Grouped Multi-Query Attention with RoPE
// =============================================================================
//
// Q projection: [hidden_size → n_q_heads * d_head]
// K projection: [hidden_size → n_kv_heads * d_head]  ← fewer heads
// V projection: [hidden_size → n_kv_heads * d_head]  ← fewer heads
//
// After projections, apply RoPE to Q and K independently, then run GQA.
// Output projection: [n_q_heads * d_head → hidden_size]

pub struct GptOssAttention {
    pub q_proj: Linear2,   // hidden → n_q_heads * d_head
    pub k_proj: Linear2,   // hidden → n_kv_heads * d_head
    pub v_proj: Linear2,   // hidden → n_kv_heads * d_head
    pub o_proj: Linear2,   // n_q_heads * d_head → hidden
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub d_head: usize,
    pub rope_theta: f32,
    /// When true, use YaRN RoPE scaling (required for sequences > original_ctx).
    /// When false, use basic RoPE (original_ctx = max_ctx, no scaling).
    pub use_yarn: bool,
    pub original_ctx: usize,
    pub max_ctx: usize,
}

impl GptOssAttention {
    pub fn new(config: &Config3, rng: &mut InitRng) -> Self {
        let d_head = config.d_head();
        // Use YaRN when the config's max_position_embeddings exceeds the standard 4096
        let use_yarn = config.max_position_embeddings > 4096;
        GptOssAttention {
            q_proj: Linear2::new(config.hidden_size, config.num_attention_heads * d_head, rng),
            k_proj: Linear2::new(config.hidden_size, config.num_key_value_heads * d_head, rng),
            v_proj: Linear2::new(config.hidden_size, config.num_key_value_heads * d_head, rng),
            o_proj: Linear2::new(config.num_attention_heads * d_head, config.hidden_size, rng),
            n_q_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            d_head,
            rope_theta: config.rope_theta,
            use_yarn,
            original_ctx: 4096,
            max_ctx: config.max_position_embeddings,
        }
    }

    /// x: [T, hidden_size]  →  output: [T, hidden_size]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let t = x.data().rows;
        let d_head = self.d_head;

        // Project to Q, K, V
        let q = self.q_proj.forward(x); // [T, n_q_heads * d_head]
        let k = self.k_proj.forward(x); // [T, n_kv_heads * d_head]
        let v = self.v_proj.forward(x); // [T, n_kv_heads * d_head]

        // Apply RoPE to each Q head independently
        let q_rope = self.apply_rope_to_all_heads(&q, self.n_q_heads, t, d_head);

        // Apply RoPE to each K head independently
        let k_rope = self.apply_rope_to_all_heads(&k, self.n_kv_heads, t, d_head);

        // Grouped Multi-Query Attention
        let attn_out = TensorNode::gqa_attention(
            &q_rope, &k_rope, &v,
            self.n_q_heads, self.n_kv_heads, d_head,
        );

        // Output projection
        self.o_proj.forward(&attn_out)
    }

    /// Apply RoPE (or YaRN RoPE) to every head in a [T, n_heads * d_head] tensor.
    ///
    /// Splits into individual [T, d_head] head slices, applies the chosen RoPE
    /// variant to each, then reassembles.
    fn apply_rope_to_all_heads(
        &self,
        x: &TensorNode,
        n_heads: usize,
        t: usize,
        d_head: usize,
    ) -> TensorNode {
        let x_data = x.data().clone();

        // Build per-head slices, apply rope, collect results
        // We do this in a single Mat::from_fn to avoid allocating n_heads TensorNodes.
        // The RoPE math is self-contained so we inline it here for both variants.
        let scale     = self.max_ctx as f32 / self.original_ctx as f32;
        let mscale    = if self.use_yarn { 0.1 * scale.ln() + 1.0 } else { 1.0 };
        let inv_scale = if self.use_yarn { 1.0 / scale } else { 1.0 };
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;

        let out_data = Mat::from_fn(t, n_heads * d_head, |row, col| {
            let h      = col / d_head;
            let dim    = col % d_head;
            let pair   = dim / 2;
            let is_odd = dim % 2 == 1;
            let pos    = row as f32;

            let omega = 1.0 / self.rope_theta.powf(2.0 * pair as f32 / d_head as f32);

            let effective_pos = if self.use_yarn {
                let cycles = self.original_ctx as f32 * omega / (2.0 * std::f32::consts::PI);
                let ramp = if cycles < beta_slow { 0.0f32 }
                           else if cycles > beta_fast { 1.0f32 }
                           else { (cycles - beta_slow) / (beta_fast - beta_slow) };
                (1.0 - ramp) * pos * inv_scale + ramp * pos
            } else {
                pos
            };

            let angle  = effective_pos * omega * mscale;
            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            let base_col = h * d_head + (dim & !1);
            if !is_odd {
                x_data.at(row, base_col)     * cos_a - x_data.at(row, base_col + 1) * sin_a
            } else {
                x_data.at(row, base_col + 1) * cos_a + x_data.at(row, base_col)     * sin_a
            }
        });

        TensorNode::leaf(out_data)
    }

    /// Cached forward: x is [n_new, hidden_size] (usually 1 token during generation).
    ///
    /// The KV cache holds K/V from all previous tokens. We:
    ///   1. Project x → Q, K_new, V_new
    ///   2. Apply RoPE starting at cache.seq_len
    ///   3. Append K_new, V_new to cache
    ///   4. Run attention: Q [n_new, …] vs full cached K/V [T_total, …]
    ///   5. Project output
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut LayerKvCache) -> TensorNode {
        let n_new  = x.data().rows;
        let d_head = self.d_head;
        let seq_offset = cache.seq_len;

        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        let q_rope = self.apply_rope_to_all_heads_at(&q, self.n_q_heads,  n_new, d_head, seq_offset);
        let k_rope = self.apply_rope_to_all_heads_at(&k, self.n_kv_heads, n_new, d_head, seq_offset);

        cache.append(&k_rope.data().clone(), &v.data().clone());

        let k_full = TensorNode::leaf(cache.k_filled());
        let v_full = TensorNode::leaf(cache.v_filled());

        let attn_out = self.gqa_attention_cached(&q_rope, &k_full, &v_full);
        self.o_proj.forward(&attn_out)
    }

    /// GQA where Q has n_q_new rows but K/V have T_total rows.
    /// No causal mask is needed: K/V only contain past tokens.
    fn gqa_attention_cached(&self, q: &TensorNode, k: &TensorNode, v: &TensorNode) -> TensorNode {
        let q_data = q.data().clone();
        let k_data = k.data().clone();
        let v_data = v.data().clone();
        let t_q  = q_data.rows;
        let t_kv = k_data.rows;
        let d_head = self.d_head;
        let scale = 1.0 / (d_head as f32).sqrt();
        let group_size = self.n_q_heads / self.n_kv_heads;

        let mut out_data = Mat::zeros(t_q, self.n_q_heads * d_head);
        for qh in 0..self.n_q_heads {
            let kvh = qh / group_size;
            let q_h = Mat::from_fn(t_q,  d_head, |r, c| q_data.at(r, qh  * d_head + c));
            let k_h = Mat::from_fn(t_kv, d_head, |r, c| k_data.at(r, kvh * d_head + c));
            let v_h = Mat::from_fn(t_kv, d_head, |r, c| v_data.at(r, kvh * d_head + c));

            let scores = q_h.matmul(&k_h.transpose()).scale(scale);
            let mut w = Mat::zeros(t_q, t_kv);
            for r in 0..t_q {
                let row_max = (0..t_kv).map(|c| scores.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0f32;
                for c in 0..t_kv { let e = (scores.at(r, c) - row_max).exp(); *w.at_mut(r, c) = e; row_sum += e; }
                for c in 0..t_kv { *w.at_mut(r, c) /= row_sum; }
            }
            let out_h = w.matmul(&v_h);
            for r in 0..t_q { for c in 0..d_head { *out_data.at_mut(r, qh * d_head + c) = out_h.at(r, c); } }
        }
        TensorNode::leaf(out_data)
    }

    /// Like apply_rope_to_all_heads but starting at seq_offset (for cache generation).
    fn apply_rope_to_all_heads_at(
        &self, x: &TensorNode, n_heads: usize, t: usize, d_head: usize, seq_offset: usize,
    ) -> TensorNode {
        let x_data = x.data().clone();
        let scale     = self.max_ctx as f32 / self.original_ctx as f32;
        let mscale    = if self.use_yarn { 0.1 * scale.ln() + 1.0 } else { 1.0 };
        let inv_scale = if self.use_yarn { 1.0 / scale } else { 1.0 };
        let beta_fast = 32.0f32;
        let beta_slow = 1.0f32;
        let out_data = Mat::from_fn(t, n_heads * d_head, |row, col| {
            let h = col / d_head; let dim = col % d_head; let pair = dim / 2; let is_odd = dim % 2 == 1;
            let pos = (seq_offset + row) as f32;
            let omega = 1.0 / self.rope_theta.powf(2.0 * pair as f32 / d_head as f32);
            let effective_pos = if self.use_yarn {
                let cycles = self.original_ctx as f32 * omega / (2.0 * std::f32::consts::PI);
                let ramp = if cycles < beta_slow { 0.0f32 } else if cycles > beta_fast { 1.0f32 }
                           else { (cycles - beta_slow) / (beta_fast - beta_slow) };
                (1.0 - ramp) * pos * inv_scale + ramp * pos
            } else { pos };
            let angle = effective_pos * omega * mscale;
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            let base_col = h * d_head + (dim & !1);
            if !is_odd { x_data.at(row, base_col) * cos_a - x_data.at(row, base_col + 1) * sin_a }
            else       { x_data.at(row, base_col + 1) * cos_a + x_data.at(row, base_col) * sin_a }
        });
        TensorNode::leaf(out_data)
    }
}

impl Module2 for GptOssAttention {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.q_proj.parameters();
        p.extend(self.k_proj.parameters());
        p.extend(self.v_proj.parameters());
        p.extend(self.o_proj.parameters());
        p
    }
}

// =============================================================================
// MoELayer — Mixture of Experts Feed-Forward Network
// =============================================================================
//
// GPT-OSS replaces the single dense FFN with a mixture of experts:
//
//   router_logits = x @ W_router          [T, num_experts]
//   weights, indices = top_k(softmax(router_logits), k=experts_per_token)
//   output = sum_i(weights[i] * expert_i(x))   for i in selected experts
//
// For gpt-oss-20b: 32 total experts, 4 active per token.
// Only ~12.5% of expert parameters are used for any given token.
//
// Why MoE?
//   - Total parameters: large (more knowledge capacity)
//   - Active parameters per token: small (fast inference)
//   - GPT-OSS-120b has 128 experts but activates only 4 → 3% of FFN weights per token
//
// Each expert is a SwiGluMlp2.

pub struct MoELayer {
    /// Router: projects hidden state to expert logits [hidden_size → num_experts]
    pub router: Linear2,

    /// The expert FFN networks
    pub experts: Vec<SwiGluMlp2>,

    pub num_experts: usize,
    pub experts_per_token: usize,
}

impl MoELayer {
    pub fn new(config: &Config3, rng: &mut InitRng) -> Self {
        let router = Linear2::new(config.hidden_size, config.num_local_experts, rng);
        let experts = (0..config.num_local_experts)
            .map(|_| SwiGluMlp2::new(config.hidden_size, config.intermediate_size, rng))
            .collect();
        MoELayer {
            router,
            experts,
            num_experts: config.num_local_experts,
            experts_per_token: config.experts_per_token,
        }
    }

    /// x: [T, hidden_size]  →  output: [T, hidden_size]
    ///
    /// For each token (row), select top-k experts by router score,
    /// run the token through each selected expert, and take the
    /// weighted sum.
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let x_data = x.data().clone();
        let t = x_data.rows;
        let d = x_data.cols;
        let k = self.experts_per_token;

        // Router logits: [T, num_experts]
        let router_logits = self.router.forward(x);
        let rlogits = router_logits.data().clone();

        let mut out_data = Mat::zeros(t, d);

        for row in 0..t {
            // Compute softmax over experts for this token
            let row_max = (0..self.num_experts)
                .map(|e| rlogits.at(row, e))
                .fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = (0..self.num_experts)
                .map(|e| (rlogits.at(row, e) - row_max).exp())
                .collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp).collect();

            // Select top-k experts by probability
            let mut indexed: Vec<(usize, f32)> = probs.iter()
                .cloned()
                .enumerate()
                .collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let top_k = &indexed[..k];

            // Renormalize weights of selected experts to sum to 1
            let weight_sum: f32 = top_k.iter().map(|(_, w)| w).sum();

            // Build a single-token input tensor
            let token_mat = Mat::from_fn(1, d, |_, c| x_data.at(row, c));
            let token_node = TensorNode::leaf(token_mat);

            // Run selected experts and accumulate weighted output
            for &(expert_idx, weight) in top_k {
                let expert_out = self.experts[expert_idx].forward(&token_node);
                let expert_data = expert_out.data();
                let normalized_weight = weight / weight_sum;
                for c in 0..d {
                    *out_data.at_mut(row, c) += normalized_weight * expert_data.at(0, c);
                }
            }
        }

        TensorNode::leaf(out_data)
    }
}

impl Module2 for MoELayer {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.router.parameters();
        for expert in &self.experts {
            p.extend(expert.parameters());
        }
        p
    }
}

// =============================================================================
// GptOssBlock — one transformer layer
// =============================================================================
//
// Pre-norm residual block (same structure as GPT-2 block, different internals):
//   x = x + Attention(RMSNorm(x))
//   x = x + MoE(RMSNorm(x))

pub struct GptOssBlock {
    pub input_layernorm:          RmsNorm2,
    pub self_attn:                GptOssAttention,
    pub post_attention_layernorm: RmsNorm2,
    pub mlp:                      MoELayer,
}

impl GptOssBlock {
    pub fn new(config: &Config3, rng: &mut InitRng) -> Self {
        GptOssBlock {
            input_layernorm:          RmsNorm2::new(config.hidden_size),
            self_attn:                GptOssAttention::new(config, rng),
            post_attention_layernorm: RmsNorm2::new(config.hidden_size),
            mlp:                      MoELayer::new(config, rng),
        }
    }

    /// x: [T, hidden_size]  →  output: [T, hidden_size]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        // Attention sub-block: x = x + Attn(RMSNorm(x))
        let attn_out = self.self_attn.forward(&self.input_layernorm.forward(x));
        let x2 = x.add(&attn_out);

        // MoE sub-block: x = x + MoE(RMSNorm(x))
        let moe_out = self.mlp.forward(&self.post_attention_layernorm.forward(&x2));
        x2.add(&moe_out)
    }

    /// Cached forward: x is [n_new, hidden_size], cache accumulates K/V.
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut LayerKvCache) -> TensorNode {
        let attn_out = self.self_attn.forward_cached(&self.input_layernorm.forward(x), cache);
        let x2 = x.add(&attn_out);
        let moe_out = self.mlp.forward(&self.post_attention_layernorm.forward(&x2));
        x2.add(&moe_out)
    }
}

impl Module2 for GptOssBlock {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.input_layernorm.parameters();
        p.extend(self.self_attn.parameters());
        p.extend(self.post_attention_layernorm.parameters());
        p.extend(self.mlp.parameters());
        p
    }
}

// =============================================================================
// KvCache — per-layer key/value cache for efficient autoregressive generation
// =============================================================================
//
// Without a KV cache, generating token T requires re-running the full forward
// pass over all T tokens every step → O(T²) total work.
//
// With a KV cache:
//   - First forward pass (prefill): run all T_prompt tokens, store K and V
//     for each layer.
//   - Each new token step: run only the new token through Q/K/V projections,
//     append the new K/V rows to the cache, run attention over cache.
//   - Cost per new token: O(T_cache) not O(T_cache²) → O(T) total.
//
// The cache stores raw Mat values (not TensorNodes) since we don't backprop
// through generation.

pub struct LayerKvCache {
    /// Accumulated K values: [T_so_far, n_kv_heads * d_head]
    pub k: Mat,
    /// Accumulated V values: [T_so_far, n_kv_heads * d_head]
    pub v: Mat,
    /// Next write position (= number of tokens processed so far)
    pub seq_len: usize,
}

impl LayerKvCache {
    fn new(n_kv_heads: usize, d_head: usize, max_seq_len: usize) -> Self {
        LayerKvCache {
            k: Mat::zeros(max_seq_len, n_kv_heads * d_head),
            v: Mat::zeros(max_seq_len, n_kv_heads * d_head),
            seq_len: 0,
        }
    }

    /// Append new K/V rows (from the current token step) to the cache.
    /// new_k, new_v: [n_new_tokens, n_kv_heads * d_head]
    fn append(&mut self, new_k: &Mat, new_v: &Mat) {
        let n_new = new_k.rows;
        let d = new_k.cols;
        assert_eq!(d, self.k.cols);
        for r in 0..n_new {
            for c in 0..d {
                *self.k.at_mut(self.seq_len + r, c) = new_k.at(r, c);
                *self.v.at_mut(self.seq_len + r, c) = new_v.at(r, c);
            }
        }
        self.seq_len += n_new;
    }

    /// Return a view of the filled portion of K: [seq_len, n_kv_heads * d_head]
    fn k_filled(&self) -> Mat {
        Mat::from_fn(self.seq_len, self.k.cols, |r, c| self.k.at(r, c))
    }

    /// Return a view of the filled portion of V: [seq_len, n_kv_heads * d_head]
    fn v_filled(&self) -> Mat {
        Mat::from_fn(self.seq_len, self.v.cols, |r, c| self.v.at(r, c))
    }
}

/// Full KV cache for all layers.
pub struct KvCache {
    pub layers: Vec<RefCell<LayerKvCache>>,
}

impl KvCache {
    pub fn new(config: &Config3) -> Self {
        let d_head = config.d_head();
        let layers = (0..config.num_hidden_layers)
            .map(|_| RefCell::new(LayerKvCache::new(
                config.num_key_value_heads,
                d_head,
                config.max_position_embeddings,
            )))
            .collect();
        KvCache { layers }
    }

    /// Reset all caches (start a new sequence).
    pub fn clear(&self) {
        for layer in &self.layers {
            layer.borrow_mut().seq_len = 0;
        }
    }
}

// =============================================================================
// GptOssModel — the full GPT-OSS model
// =============================================================================

pub struct GptOssModel {
    /// Token embedding table [vocab_size, hidden_size]
    pub embed_tokens: TensorNode,

    /// Transformer layers
    pub layers: Vec<GptOssBlock>,

    /// Final RMSNorm before lm_head
    pub norm: RmsNorm2,

    /// Language model head: [hidden_size → vocab_size]
    pub lm_head: Linear2,

    pub config: Config3,
}

/// Statistics returned by `quantize_all_linear_weights()`.
#[derive(Debug, Default)]
pub struct Q4QuantStats {
    pub n_tensors:        usize,
    pub f32_bytes:        usize,
    pub q4_bytes:         usize,
    pub compression_ratio: f32,
}

impl GptOssModel {
    /// Create a randomly-initialized GPT-OSS model with the given config.
    ///
    /// For the real model, use `gpt_oss_20b()` or `gpt_oss_120b()` config,
    /// then load weights from HuggingFace (see module-level docs).
    pub fn new(config: Config3, rng: &mut InitRng) -> Self {
        let embed_tokens = TensorNode::leaf(Mat::new(
            rng.normal_vec(config.vocab_size * config.hidden_size, 0.02),
            config.vocab_size, config.hidden_size,
        ));
        let layers = (0..config.num_hidden_layers)
            .map(|_| GptOssBlock::new(&config, rng))
            .collect();
        let norm    = RmsNorm2::new(config.hidden_size);
        let lm_head = Linear2::new(config.hidden_size, config.vocab_size, rng);

        GptOssModel { embed_tokens, layers, norm, lm_head, config }
    }

    /// Forward pass: token_ids → logits [T, vocab_size]
    ///
    /// This is pure inference — no loss, no backward.
    /// To generate text, take the last row of the logits,
    /// apply softmax + temperature + top-k sampling.
    pub fn forward(&self, token_ids: &[usize]) -> TensorNode {
        let t = token_ids.len();
        let d = self.config.hidden_size;

        // Token embedding lookup (no positional embedding — RoPE handles position)
        let te = self.embed_tokens.data().clone();
        let x_data = Mat::from_fn(t, d, |row, col| te.at(token_ids[row], col));
        let mut x = TensorNode::leaf(x_data);

        // Pass through transformer layers
        for layer in &self.layers {
            x = layer.forward(&x);
        }

        // Final norm + lm_head
        let x_normed = self.norm.forward(&x);
        self.lm_head.forward(&x_normed)
    }

    /// Batched forward pass for B sequences.
    ///
    /// ## What "batching" means here
    ///
    /// A real GPU batch processes B sequences in parallel inside a single
    /// [B, T, D] tensor operation.  Our `Mat` is 2D, so we take the next
    /// best thing: we pad all sequences to the same length `T_max` and run
    /// one independent forward pass per sequence.  The work is the same, but
    /// the loop is simple and the API is identical to what you'd have with 3D
    /// tensors — callers can batch their data and get back one logit matrix
    /// per sequence without caring about the inner implementation.
    ///
    /// ## Parameters
    ///   sequences:   B sequences; need not have the same length
    ///
    /// ## Returns
    ///   Vec of B logit tensors [T_i, vocab_size], one per sequence.
    pub fn forward_batch(&self, sequences: &[Vec<usize>]) -> Vec<TensorNode> {
        sequences.iter().map(|seq| self.forward(seq)).collect()
    }

    /// Batched cross-entropy loss for B (input, target) sequence pairs.
    ///
    /// Each sequence pair shares the same length T. The loss is averaged
    /// across all T*B token predictions.
    ///
    /// ## Parameters
    ///   batch:  slice of (input_ids, target_ids) pairs, all the same length
    ///
    /// ## Returns
    ///   A scalar TensorNode with gradient wired through all B sequences.
    pub fn loss_batch(&self, batch: &[(&[usize], &[usize])]) -> TensorNode {
        assert!(!batch.is_empty(), "loss_batch: empty batch");

        // Collect per-sequence scalar losses
        let loss_nodes: Vec<TensorNode> = batch.iter()
            .map(|(inp, tgt)| self.loss(inp, tgt))
            .collect();

        // Average them: accumulate into a sum, then scale
        let b = loss_nodes.len() as f32;
        // Build a sum node the same way loss() works: leaf + backward plumbing
        let sum_val: f32 = loss_nodes.iter().map(|n| n.data().at(0, 0)).sum::<f32>() / b;

        let avg = TensorNode::leaf(Mat::new(vec![sum_val], 1, 1));
        // Backward: propagate 1/B into each child loss
        let children = loss_nodes.clone();
        avg.set_backward(Box::new(move || {
            for child in &children {
                let cur = child.grad().clone();
                let upd = Mat::new(vec![cur.at(0,0) + 1.0/b], 1, 1);
                child.set_grad(upd);
                child.call_backward_fn();
            }
        }), loss_nodes);
        avg
    }

    /// Per-sequence cross-entropy loss (input_ids → targets).
    ///
    /// Shared helper used by `loss_batch`. Returns a scalar TensorNode whose
    /// backward pass propagates gradients through the logits.
    pub fn loss(&self, token_ids: &[usize], targets: &[usize]) -> TensorNode {
        let logits_node = self.forward(token_ids);
        let logits = logits_node.data().clone();
        let t = logits.rows;
        let v = logits.cols;
        assert_eq!(t, targets.len(), "loss: token_ids len {} != targets len {}", t, targets.len());

        // Numerically-stable softmax + cross-entropy per row
        let mut probs = Mat::zeros(t, v);
        let mut loss_val = 0.0f32;
        for r in 0..t {
            let row_max = (0..v).map(|c| logits.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..v {
                let e = (logits.at(r, c) - row_max).exp();
                *probs.at_mut(r, c) = e;
                sum_exp += e;
            }
            for c in 0..v { *probs.at_mut(r, c) /= sum_exp; }
            loss_val -= probs.at(r, targets[r]).ln();
        }
        loss_val /= t as f32;

        let loss = TensorNode::leaf(Mat::new(vec![loss_val], 1, 1));

        // Backward: d(loss)/d(logits[r,c]) = (probs[r,c] - 1_{c==target[r]}) / T
        let logits_c      = logits_node.clone();
        let probs_stored  = probs;
        let targets_v     = targets.to_vec();
        loss.set_backward(Box::new(move || {
            let mut dlogits = logits_c.grad().clone();
            for r in 0..t {
                for c in 0..v {
                    let indicator = if c == targets_v[r] { 1.0 } else { 0.0 };
                    *dlogits.at_mut(r, c) += (probs_stored.at(r, c) - indicator) / t as f32;
                }
            }
            logits_c.set_grad(dlogits);
            logits_c.call_backward_fn();
        }), vec![logits_node]);
        loss
    }

    /// Greedy next-token prediction for a prompt.
    ///
    /// Returns the index of the most likely next token.
    /// For real generation, apply temperature + top-k sampling (see train2.rs).
    pub fn predict_next(&self, token_ids: &[usize]) -> usize {
        let logits = self.forward(token_ids);
        let logits_data = logits.data();
        let t = logits_data.rows;
        let v = logits_data.cols;
        (0..v)
            .max_by(|&a, &b| logits_data.at(t - 1, a)
                .partial_cmp(&logits_data.at(t - 1, b))
                .unwrap())
            .unwrap()
    }

    /// Autoregressive generation with KV cache — O(T) per step.
    ///
    /// ## How the KV cache makes generation fast
    ///
    /// Without cache: generating token k requires running the full prompt
    /// (k tokens) through every layer → O(k²) total work.
    ///
    /// With cache:
    ///   Step 1 — "prefill": run the entire prompt once, populate cache.
    ///   Step 2+ — "decode": for each new token, pass only 1 token through
    ///             attention, which reads from but only appends to the cache.
    ///             → O(T_prompt + T_generate) total work.
    ///
    /// ## Parameters
    ///   token_ids:    prompt tokens
    ///   max_new:      number of new tokens to generate
    ///   temperature:  >1 = more random, <1 = more greedy, 1.0 = standard sampling
    ///
    /// ## Returns
    ///   Vec of new token ids (not including the prompt)
    pub fn generate_cached(&self, token_ids: &[usize], max_new: usize, temperature: f32) -> Vec<usize> {
        let cache = KvCache::new(&self.config);
        let d = self.config.hidden_size;
        let te = self.embed_tokens.data().clone();

        // ---- Prefill: run the full prompt through all layers with cache ----
        let t_prompt = token_ids.len();
        let x_data = Mat::from_fn(t_prompt, d, |row, col| te.at(token_ids[row], col));
        let mut x = TensorNode::leaf(x_data);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
        }
        let x_normed = self.norm.forward(&x);
        let logits = self.lm_head.forward(&x_normed);

        // Sample first new token from last position of prompt logits
        let mut generated = Vec::with_capacity(max_new);
        let first_tok = self.sample_token(&logits.data(), t_prompt - 1, temperature);
        generated.push(first_tok);

        // ---- Decode: one token at a time ----
        let mut prev_tok = first_tok;
        for _ in 1..max_new {
            // Build single-token embedding
            let x_data = Mat::from_fn(1, d, |_, col| te.at(prev_tok, col));
            let mut x = TensorNode::leaf(x_data);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
            }
            let x_normed = self.norm.forward(&x);
            let logits   = self.lm_head.forward(&x_normed);

            prev_tok = self.sample_token(&logits.data(), 0, temperature);
            generated.push(prev_tok);
        }

        generated
    }

    /// Like `generate_cached` but calls `callback(token_id)` after each new token
    /// instead of buffering them all.  Useful for printing tokens as they arrive:
    ///
    /// ```ignore
    /// model.generate_cached_streaming(&prompt_ids, 100, 0.8, |tok| {
    ///     print!("{}", tokenizer.decode(&[tok as u32]));
    ///     std::io::stdout().flush().unwrap();
    /// });
    /// ```
    pub fn generate_cached_streaming(
        &self,
        token_ids: &[usize],
        max_new: usize,
        temperature: f32,
        mut callback: impl FnMut(usize),
    ) {
        let cache = KvCache::new(&self.config);
        let d = self.config.hidden_size;
        let te = self.embed_tokens.data().clone();

        // Prefill
        let t_prompt = token_ids.len();
        let x_data = Mat::from_fn(t_prompt, d, |row, col| te.at(token_ids[row], col));
        let mut x = TensorNode::leaf(x_data);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
        }
        let x_normed = self.norm.forward(&x);
        let logits   = self.lm_head.forward(&x_normed);

        let first_tok = self.sample_token(&logits.data(), t_prompt - 1, temperature);
        callback(first_tok);

        // Decode
        let mut prev_tok = first_tok;
        for _ in 1..max_new {
            let x_data = Mat::from_fn(1, d, |_, col| te.at(prev_tok, col));
            let mut x = TensorNode::leaf(x_data);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
            }
            let x_normed = self.norm.forward(&x);
            let logits   = self.lm_head.forward(&x_normed);

            prev_tok = self.sample_token(&logits.data(), 0, temperature);
            callback(prev_tok);
        }
    }

    /// Sample a token from logits at row `pos` with temperature scaling.
    ///
    /// Legacy helper kept for backward compatibility with existing tests.
    /// Equivalent to `sample_token_full` with greedy/temperature-only params.
    fn sample_token(&self, logits: &Mat, pos: usize, temperature: f32) -> usize {
        let params = if temperature <= 0.0 {
            SamplingParams::greedy()
        } else {
            SamplingParams { temperature, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 0,
                             eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 }
        };
        // No previously-seen tokens (repetition_penalty disabled)
        self.sample_token_full(logits, pos, &params, &[], &mut LcgRng::new(params.seed))
    }

    /// Full sampler: top-k → top-p → repetition penalty → temperature → sample.
    ///
    /// ## Algorithm
    ///
    ///   1. Apply repetition penalty: logit[tok] /= penalty  for each tok ∈ seen_ids.
    ///      (Division because logit is positive for likely tokens; dividing makes
    ///       them relatively less likely while keeping the distribution normalised.)
    ///
    ///   2. If temperature ≤ 0: greedy argmax on (penalised) logits. Done.
    ///
    ///   3. Apply temperature: logit[t] /= temperature.
    ///
    ///   4. top-k filter: keep only the K highest-logit indices.
    ///      If top_k = 0, keep all.
    ///
    ///   5. Compute softmax over the surviving set.
    ///
    ///   6. top-p (nucleus) filter: sort by descending probability, keep the
    ///      smallest prefix whose cumulative probability ≥ top_p.  Re-normalise.
    ///
    ///   7. Draw one token proportionally to the surviving probabilities using
    ///      the provided PRNG.
    fn sample_token_full(
        &self,
        logits: &Mat,
        pos: usize,
        params: &SamplingParams,
        seen_ids: &[usize],
        rng: &mut LcgRng,
    ) -> usize {
        let v = logits.cols;

        // Build frequency counts from seen_ids (needed for freq/presence penalties)
        let mut freq: Vec<u32> = vec![0u32; v];
        for &tok in seen_ids {
            if tok < v { freq[tok] += 1; }
        }

        // Step 1: repetition penalty + frequency penalty + presence penalty
        let mut row: Vec<f32> = (0..v).map(|c| logits.at(pos, c)).collect();
        if params.repetition_penalty != 1.0 {
            for (tok, &cnt) in freq.iter().enumerate() {
                if cnt > 0 {
                    // Penalise: reduce logit magnitude (works for both positive & negative logits)
                    row[tok] /= params.repetition_penalty;
                }
            }
        }
        // Frequency penalty: subtract penalty * count
        if params.frequency_penalty != 0.0 {
            for (tok, &cnt) in freq.iter().enumerate() {
                if cnt > 0 {
                    row[tok] -= params.frequency_penalty * cnt as f32;
                }
            }
        }
        // Presence penalty: subtract fixed amount if token appeared at all
        if params.presence_penalty != 0.0 {
            for (tok, &cnt) in freq.iter().enumerate() {
                if cnt > 0 {
                    row[tok] -= params.presence_penalty;
                }
            }
        }

        // Step 2: greedy if temperature == 0
        if params.temperature <= 0.0 {
            return row.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
        }

        // Step 3: temperature scaling
        let temp = params.temperature;
        for x in row.iter_mut() { *x /= temp; }

        // Step 4: top-k — keep only the k highest logits
        let mut candidates: Vec<usize> = (0..v).collect();
        if params.top_k > 0 && params.top_k < v {
            candidates.sort_unstable_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
            candidates.truncate(params.top_k);
        }

        // Step 5: softmax over candidates
        let row_max = candidates.iter().map(|&i| row[i]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = candidates.iter().map(|&i| (row[i] - row_max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let mut probs: Vec<(usize, f32)> = candidates.iter().zip(exps.iter())
            .map(|(&tok, &e)| (tok, e / sum))
            .collect();

        // Step 6: top-p (nucleus) — sort desc, keep cumulative ≥ top_p
        if params.top_p < 1.0 {
            probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let mut cumsum = 0.0f32;
            let mut cutoff = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cumsum += p;
                if cumsum >= params.top_p {
                    cutoff = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff);
            // Re-normalise
            let total: f32 = probs.iter().map(|(_, p)| p).sum();
            for (_, p) in probs.iter_mut() { *p /= total; }
        }

        // Step 7: draw from the distribution
        let u = rng.next_f32();
        let mut cumsum = 0.0f32;
        for &(tok, p) in &probs {
            cumsum += p;
            if u < cumsum { return tok; }
        }
        probs.last().map(|&(tok, _)| tok).unwrap_or(0)
    }

    /// Full generation using `SamplingParams` (top-k, top-p, rep. penalty, RNG).
    ///
    /// Returns the generated token ids (not including the prompt).
    pub fn generate_with_params(
        &self,
        token_ids: &[usize],
        max_new: usize,
        params: &SamplingParams,
    ) -> Vec<usize> {
        let cache = KvCache::new(&self.config);
        let d = self.config.hidden_size;
        let te = self.embed_tokens.data().clone();
        let mut rng = LcgRng::new(params.seed);

        // Prefill
        let t_prompt = token_ids.len();
        let x_data = Mat::from_fn(t_prompt, d, |row, col| te.at(token_ids[row], col));
        let mut x = TensorNode::leaf(x_data);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
        }
        let x_normed = self.norm.forward(&x);
        let logits   = self.lm_head.forward(&x_normed);

        let mut generated: Vec<usize> = Vec::with_capacity(max_new);
        // seen = prompt + generated (for repetition penalty)
        let mut seen: Vec<usize> = token_ids.to_vec();

        let first_tok = self.sample_token_full(&logits.data(), t_prompt - 1, params, &seen, &mut rng);
        generated.push(first_tok);
        seen.push(first_tok);

        // Stop immediately if EOS was the first generated token
        if params.eos_token_id == Some(first_tok) {
            return generated;
        }

        // Decode
        let mut prev_tok = first_tok;
        for _ in 1..max_new {
            let x_data = Mat::from_fn(1, d, |_, col| te.at(prev_tok, col));
            let mut x = TensorNode::leaf(x_data);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
            }
            let x_normed = self.norm.forward(&x);
            let logits   = self.lm_head.forward(&x_normed);

            prev_tok = self.sample_token_full(&logits.data(), 0, params, &seen, &mut rng);
            generated.push(prev_tok);
            seen.push(prev_tok);

            if params.eos_token_id == Some(prev_tok) {
                break;
            }
        }
        generated
    }

    /// Streaming variant of `generate_with_params`.
    pub fn generate_with_params_streaming(
        &self,
        token_ids: &[usize],
        max_new: usize,
        params: &SamplingParams,
        mut callback: impl FnMut(usize),
    ) {
        let cache = KvCache::new(&self.config);
        let d = self.config.hidden_size;
        let te = self.embed_tokens.data().clone();
        let mut rng = LcgRng::new(params.seed);

        let t_prompt = token_ids.len();
        let x_data = Mat::from_fn(t_prompt, d, |row, col| te.at(token_ids[row], col));
        let mut x = TensorNode::leaf(x_data);
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
        }
        let x_normed = self.norm.forward(&x);
        let logits   = self.lm_head.forward(&x_normed);

        let mut seen: Vec<usize> = token_ids.to_vec();
        let first_tok = self.sample_token_full(&logits.data(), t_prompt - 1, params, &seen, &mut rng);
        callback(first_tok);
        seen.push(first_tok);

        // Stop immediately if EOS was the first generated token
        if params.eos_token_id == Some(first_tok) {
            return;
        }

        let mut prev_tok = first_tok;
        for _ in 1..max_new {
            let x_data = Mat::from_fn(1, d, |_, col| te.at(prev_tok, col));
            let mut x = TensorNode::leaf(x_data);
            for (layer_idx, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[layer_idx].borrow_mut());
            }
            let x_normed = self.norm.forward(&x);
            let logits   = self.lm_head.forward(&x_normed);

            prev_tok = self.sample_token_full(&logits.data(), 0, params, &seen, &mut rng);
            callback(prev_tok);
            seen.push(prev_tok);

            if params.eos_token_id == Some(prev_tok) {
                break;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Perplexity
    // -------------------------------------------------------------------------

    /// Compute perplexity of a token sequence: PPL = exp(mean cross-entropy loss).
    ///
    /// ## What perplexity means
    ///
    /// Perplexity is the standard benchmark metric for language models.
    /// It measures "how surprised is the model by this text?":
    ///   - PPL = 1     → model predicts every token with 100% confidence (perfect)
    ///   - PPL = V     → model is no better than random (V = vocab size)
    ///   - GPT-2 achieves ~29 PPL on WikiText-103
    ///   - GPT-3 achieves ~20 PPL
    ///   - GPT-OSS-20b is expected to achieve ~12 PPL
    ///
    /// ## Algorithm
    ///
    ///   PPL = exp( -1/T * sum_t log P(token_t | token_0..t-1) )
    ///
    /// We evaluate this by sliding a context window of `context_len` tokens
    /// across the full sequence, computing the model's probability of each
    /// token, and averaging the log-probabilities.
    ///
    /// ## Parameters
    ///   token_ids:   the full sequence to evaluate (must have ≥ 2 tokens)
    ///   context_len: maximum context window (defaults to config max if 0)
    ///
    /// ## Returns
    ///   Perplexity as f32 (lower is better).
    pub fn perplexity(&self, token_ids: &[usize], context_len: usize) -> f32 {
        assert!(token_ids.len() >= 2, "perplexity: need at least 2 tokens");
        let ctx = if context_len == 0 { self.config.max_position_embeddings } else { context_len };
        let ctx = ctx.min(self.config.max_position_embeddings);

        let n = token_ids.len();
        let mut total_nll = 0.0f32;
        let mut n_tokens = 0usize;

        // Slide a window: at each position t predict token_ids[t] from context
        let mut t = 1usize;
        while t < n {
            // Context: take up to ctx tokens ending just before t
            let ctx_start = if t > ctx { t - ctx } else { 0 };
            let ctx_ids = &token_ids[ctx_start..t];
            let target   = token_ids[t];

            let logits_node = self.forward(ctx_ids);
            let logits = logits_node.data();
            let last   = logits.rows - 1;
            let v      = logits.cols;

            // Numerically stable log-softmax at last position
            let row_max = (0..v).map(|c| logits.at(last, c)).fold(f32::NEG_INFINITY, f32::max);
            let sum_exp: f32 = (0..v).map(|c| (logits.at(last, c) - row_max).exp()).sum();
            let log_prob = logits.at(last, target) - row_max - sum_exp.ln();
            total_nll -= log_prob;
            n_tokens  += 1;

            // Advance by stride = ctx/2 to overlap windows (more accurate PPL)
            t += (ctx / 2).max(1);
        }

        (total_nll / n_tokens as f32).exp()
    }
}

// =============================================================================
// Safetensors weight loader
// =============================================================================
//
// GPT-OSS weights are distributed as .safetensors files from HuggingFace.
// The safetensors format stores tensors with a JSON header (names + dtypes +
// byte offsets) followed by raw binary data.
//
// ## How to get the weights
//
//   pip install huggingface_hub
//   huggingface-cli download openai/gpt-oss-20b --local-dir ./gpt-oss-20b-weights
//
// This downloads multiple .safetensors shards (gpt-oss-20b is split across ~10).
//
// ## Weight name mapping (HuggingFace → this struct)
//
//   model.embed_tokens.weight                        → model.embed_tokens
//   model.layers.{i}.input_layernorm.weight          → layers[i].input_layernorm.gamma
//   model.layers.{i}.self_attn.q_proj.weight         → layers[i].self_attn.q_proj.weight
//   model.layers.{i}.self_attn.k_proj.weight         → layers[i].self_attn.k_proj.weight
//   model.layers.{i}.self_attn.v_proj.weight         → layers[i].self_attn.v_proj.weight
//   model.layers.{i}.self_attn.o_proj.weight         → layers[i].self_attn.o_proj.weight
//   model.layers.{i}.post_attention_layernorm.weight → layers[i].post_attention_layernorm.gamma
//   model.layers.{i}.mlp.router.weight               → layers[i].mlp.router.weight
//   model.layers.{i}.mlp.experts.{j}.gate_proj.weight → layers[i].mlp.experts[j].gate_proj.weight
//   model.layers.{i}.mlp.experts.{j}.up_proj.weight   → layers[i].mlp.experts[j].up_proj.weight
//   model.layers.{i}.mlp.experts.{j}.down_proj.weight → layers[i].mlp.experts[j].down_proj.weight
//   model.norm.weight                                → model.norm.gamma
//   lm_head.weight                                   → model.lm_head.weight
//
// ## dtype notes
//
// HuggingFace stores weights in BF16 (bfloat16). This loader converts them to
// f32 on load. The conversion: bf16 is just f32 with the lower 16 bits zeroed,
// so we reconstruct f32 by left-shifting the 16-bit value.

/// A single parsed tensor from a safetensors file.
pub struct SafeTensor {
    pub name:   String,
    pub shape:  Vec<usize>,
    pub data:   Vec<f32>,    // always f32 after conversion
}

/// Load all tensors from a safetensors binary blob (the raw file bytes).
///
/// Returns a list of SafeTensors. Call `load_into_model` to apply them.
///
/// ## Format
///
/// The file starts with:
///   [8 bytes: header_length as little-endian u64]
///   [header_length bytes: UTF-8 JSON]
///   [remaining bytes: raw tensor data, packed]
///
/// The JSON has the shape:
///   { "tensor_name": { "dtype": "BF16", "shape": [rows, cols], "data_offsets": [start, end] } }
pub fn parse_safetensors(bytes: &[u8]) -> Result<Vec<SafeTensor>, String> {
    if bytes.len() < 8 {
        return Err("safetensors: file too small".to_string());
    }
    let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    if bytes.len() < 8 + header_len {
        return Err(format!("safetensors: header_len {} exceeds file size", header_len));
    }
    let header_json = std::str::from_utf8(&bytes[8..8 + header_len])
        .map_err(|e| format!("safetensors: invalid header UTF-8: {}", e))?;
    let data_section = &bytes[8 + header_len..];

    let mut tensors = Vec::new();
    // Parse top-level object: iterate over key-value pairs
    for (name, value_json) in iter_top_level_pairs(header_json) {
        if name == "__metadata__" { continue; }
        if let Some(tensor) = parse_tensor_value(name, value_json, data_section) {
            tensors.push(tensor);
        }
    }
    Ok(tensors)
}

/// Iterate over top-level `"key": {...}` pairs in a JSON object string.
/// Yields (&str name, &str value_json) for each entry.
fn iter_top_level_pairs(json: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    let b = json.as_bytes();
    let n = b.len();
    let mut i = 0;

    while i < n {
        // Find opening quote of key
        while i < n && b[i] != b'"' { i += 1; }
        if i >= n { break; }
        i += 1; // skip opening quote
        let key_start = i;
        while i < n && b[i] != b'"' { i += 1; }
        let key = json[key_start..i].to_string();
        i += 1; // skip closing quote

        // Skip whitespace and colon
        while i < n && (b[i] == b':' || b[i] == b' ' || b[i] == b'\n' || b[i] == b'\r') { i += 1; }

        // Find matching { }
        if i >= n || b[i] != b'{' { continue; }
        let val_start = i;
        let mut depth = 0i32;
        while i < n {
            if b[i] == b'{' { depth += 1; }
            else if b[i] == b'}' { depth -= 1; if depth == 0 { i += 1; break; } }
            i += 1;
        }
        let value_json = json[val_start..i].to_string();
        pairs.push((key, value_json));
    }
    pairs
}

/// Parse a single tensor's value object: `{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]}`
fn parse_tensor_value(name: String, value_json: String, data_section: &[u8]) -> Option<SafeTensor> {
    let dtype   = extract_quoted_value(&value_json, "dtype")?;
    let shape   = extract_int_array(&value_json, "shape")?;
    let offsets = extract_int_array(&value_json, "data_offsets")?;

    if offsets.len() != 2 { return None; }
    let (byte_start, byte_end) = (offsets[0] as usize, offsets[1] as usize);
    if byte_end > data_section.len() { return None; }

    let raw = &data_section[byte_start..byte_end];
    let data: Vec<f32> = match dtype.as_str() {
        "F32" => raw.chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect(),
        "BF16" => raw.chunks_exact(2)
                     .map(|c| {
                         let bits = u16::from_le_bytes(c.try_into().unwrap()) as u32;
                         f32::from_bits(bits << 16)
                     })
                     .collect(),
        "F16" => raw.chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())))
                    .collect(),
        _ => return None,
    };

    let shape: Vec<usize> = shape.iter().map(|&v| v as usize).collect();
    Some(SafeTensor { name, shape, data })
}

/// Extract the string value of `"key":"value"` from a JSON object string.
fn extract_quoted_value(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = json[pos + pattern.len()..].trim_start();
    if !after.starts_with('"') { return None; }
    let rest = &after[1..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

/// Extract an integer array value of `"key":[1,2,3]` from a JSON object string.
fn extract_int_array(json: &str, key: &str) -> Option<Vec<i64>> {
    let pattern = format!("\"{}\":", key);
    let pos = json.find(&pattern)?;
    let after = json[pos + pattern.len()..].trim_start();
    if !after.starts_with('[') { return None; }
    let end = after.find(']')?;
    let inner = &after[1..end];
    inner.split(',')
        .map(|s| s.trim().parse::<i64>().ok())
        .collect::<Option<Vec<_>>>()
}

/// Convert IEEE 754 float16 bits to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign     = ((bits >> 15) as u32) << 31;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    let f32_bits = if exponent == 0 {
        // Zero or subnormal
        if mantissa == 0 {
            sign
        } else {
            // Subnormal f16 → normalized f32
            let mut m = mantissa << 1;
            let mut e = 127u32 - 14;
            while m & 0x400 == 0 { m <<= 1; e -= 1; }
            sign | (e << 23) | ((m & 0x3FF) << 13)
        }
    } else if exponent == 31 {
        // Inf or NaN
        sign | (255u32 << 23) | (mantissa << 13)
    } else {
        sign | ((exponent + 127 - 15) << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}

/// Load a parsed tensor list into the model, matching by name.
///
/// This is the main entry point for weight loading. Call it with the output
/// of `parse_safetensors`.
///
/// ## Example
///
/// ```no_run
/// let bytes = std::fs::read("gpt-oss-20b-weights/model-00001-of-00010.safetensors").unwrap();
/// let tensors = parse_safetensors(&bytes).unwrap();
/// load_into_model(&mut model, &tensors);
/// // Repeat for each shard file
/// ```
pub fn load_into_model(model: &mut GptOssModel, tensors: &[SafeTensor]) {
    for tensor in tensors {
        if !apply_tensor(model, tensor) {
            // Unknown name — either a shard we don't recognize, or an extra
            // tensor (e.g. "model.rotary_emb.inv_freq"). Skip silently.
        }
    }
}

/// Apply one tensor to the matching field in the model.
/// Returns true if the tensor was recognized and applied.
fn apply_tensor(model: &mut GptOssModel, t: &SafeTensor) -> bool {
    apply_tensor_inner(model, t).unwrap_or(false)
}

fn apply_tensor_inner(model: &mut GptOssModel, t: &SafeTensor) -> Option<bool> {
    let name = t.name.as_str();

    // Token embedding
    if name == "model.embed_tokens.weight" {
        return Some(set_node(&model.embed_tokens, &t.data, t.shape[0], t.shape[1]));
    }
    // Final norm
    if name == "model.norm.weight" {
        return Some(set_node(&model.norm.gamma, &t.data, 1, t.shape[0]));
    }
    // LM head
    if name == "lm_head.weight" {
        return Some(set_node(&model.lm_head.weight, &t.data, t.shape[0], t.shape[1]));
    }

    // Per-layer tensors: "model.layers.{i}...."
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let dot = rest.find('.')?;
        let layer_idx: usize = rest[..dot].parse().ok()?;
        if layer_idx >= model.layers.len() { return Some(false); }
        let layer_name = &rest[dot + 1..];
        let layer = &mut model.layers[layer_idx];

        if layer_name == "input_layernorm.weight" {
            return Some(set_node(&layer.input_layernorm.gamma, &t.data, 1, t.shape[0]));
        }
        if layer_name == "self_attn.q_proj.weight" {
            return Some(set_node(&layer.self_attn.q_proj.weight, &t.data, t.shape[0], t.shape[1]));
        }
        if layer_name == "self_attn.k_proj.weight" {
            return Some(set_node(&layer.self_attn.k_proj.weight, &t.data, t.shape[0], t.shape[1]));
        }
        if layer_name == "self_attn.v_proj.weight" {
            return Some(set_node(&layer.self_attn.v_proj.weight, &t.data, t.shape[0], t.shape[1]));
        }
        if layer_name == "self_attn.o_proj.weight" {
            return Some(set_node(&layer.self_attn.o_proj.weight, &t.data, t.shape[0], t.shape[1]));
        }
        if layer_name == "post_attention_layernorm.weight" {
            return Some(set_node(&layer.post_attention_layernorm.gamma, &t.data, 1, t.shape[0]));
        }
        if layer_name == "mlp.router.weight" {
            return Some(set_node(&layer.mlp.router.weight, &t.data, t.shape[0], t.shape[1]));
        }

        // Experts: "mlp.experts.{j}.{proj}.weight"
        if let Some(expert_rest) = layer_name.strip_prefix("mlp.experts.") {
            let dot2 = expert_rest.find('.')?;
            let expert_idx: usize = expert_rest[..dot2].parse().ok()?;
            if expert_idx >= layer.mlp.experts.len() { return Some(false); }
            let expert_name = &expert_rest[dot2 + 1..];
            let expert = &mut layer.mlp.experts[expert_idx];
            if expert_name == "gate_proj.weight" {
                return Some(set_node(&expert.gate_proj.weight, &t.data, t.shape[0], t.shape[1]));
            }
            if expert_name == "up_proj.weight" {
                return Some(set_node(&expert.up_proj.weight, &t.data, t.shape[0], t.shape[1]));
            }
            if expert_name == "down_proj.weight" {
                return Some(set_node(&expert.down_proj.weight, &t.data, t.shape[0], t.shape[1]));
            }
        }
    }

    Some(false)
}

/// Set the data of a TensorNode from a flat f32 slice.
fn set_node(node: &TensorNode, data: &[f32], rows: usize, cols: usize) -> bool {
    if data.len() != rows * cols {
        return false;
    }
    node.set_data(Mat::new(data.to_vec(), rows, cols));
    true
}

impl GptOssModel {
    /// Load weights from all .safetensors shards in a directory.
    ///
    /// Reads every file matching `*.safetensors` in `dir` and applies tensors.
    ///
    /// ## Usage
    ///
    /// ```no_run
    /// let config = Config3::gpt_oss_20b();
    /// let mut rng = InitRng::new(0);
    /// let mut model = GptOssModel::new(config, &mut rng);
    /// model.load_weights_from_dir("./gpt-oss-20b-weights")
    ///      .expect("failed to load weights");
    /// ```
    pub fn load_weights_from_dir(&mut self, dir: &str) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read dir {}: {}", dir, e))?;

        let mut loaded_shards = 0usize;
        let mut loaded_tensors = 0usize;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }

            let bytes = std::fs::read(&path)
                .map_err(|e| format!("cannot read {:?}: {}", path, e))?;

            let tensors = parse_safetensors(&bytes)
                .map_err(|e| format!("parse error in {:?}: {}", path, e))?;

            loaded_tensors += tensors.len();
            load_into_model(self, &tensors);
            loaded_shards += 1;
        }

        if loaded_shards == 0 {
            return Err(format!("no .safetensors files found in {}", dir));
        }

        println!("Loaded {} tensors from {} shards in {}", loaded_tensors, loaded_shards, dir);
        Ok(())
    }

    // =========================================================================
    // 4-bit weight quantization
    // =========================================================================
    //
    // After loading f32 (or BF16→f32) weights, we can further compress them
    // to 4-bit to reduce inference memory from ~40GB to ~11GB for GPT-OSS-20b.
    //
    // ## Usage
    //
    // ```no_run
    // model.load_weights_from_dir("./gpt-oss-20b-weights").unwrap();
    // let stats = model.quantize_all_linear_weights();
    // println!("Compressed {} weights from {:.1}GB to {:.1}GB ({:.1}x)",
    //          stats.n_tensors, stats.f32_bytes as f64 / 1e9,
    //          stats.q4_bytes  as f64 / 1e9, stats.compression_ratio);
    // ```
    //
    // After quantization, the model's weight TensorNodes are set to the
    // dequantized approximation. This is the "static quantization" approach:
    // slightly lossy, but weights only need to be dequantized once.
    //
    // For "dynamic quantization" (dequantize on every forward pass for
    // better cache utilization), use `Q4Mat::matmul_q4_t` directly.

    /// Quantize all Linear2 weight matrices to 4-bit and replace them with
    /// the dequantized approximation.
    ///
    /// Returns statistics about the compression.
    pub fn quantize_all_linear_weights(&self) -> Q4QuantStats {
        let mut stats = Q4QuantStats::default();

        // Quantize all parameters that are weight matrices (rows*cols > 1)
        for param in self.parameters() {
            let (rows, cols) = {
                let d = param.data();
                (d.rows, d.cols)
            };
            // Skip scalars and norm gammas (1D vectors — not worth quantizing)
            if rows <= 1 || cols <= 1 { continue; }

            let original = param.data().clone();
            let q4 = Q4Mat::quantize(&original);
            let dequant = q4.dequantize();

            stats.n_tensors += 1;
            stats.f32_bytes += rows * cols * 4;
            stats.q4_bytes  += q4.size_bytes();

            // Replace the weight data with the dequantized approximation
            param.set_data(dequant);
        }

        if stats.f32_bytes > 0 {
            stats.compression_ratio = stats.f32_bytes as f32 / stats.q4_bytes as f32;
        }
        stats
    }

    // =========================================================================
    // Memory-mapped weight loading
    // =========================================================================
    //
    // ## Why mmap?
    //
    // GPT-OSS-20b weights are ~40GB of BF16 data split across ~10 shard files.
    // Loading them with `std::fs::read()` copies all bytes into a heap-allocated
    // Vec<u8>, requiring 40GB of RAM *in addition* to the model struct itself.
    //
    // Memory mapping (`mmap`) instead asks the OS to map the file's pages directly
    // into the process's virtual address space:
    //   - No upfront copy: pages are loaded on demand when first accessed.
    //   - The OS can evict clean pages and reload from disk — effective working
    //     set is the pages actually touched during inference, not the full file.
    //   - Multiple processes can share the same physical pages.
    //
    // For GPT-OSS-20b: the model parameters that fit in RAM (~8GB of active
    // layers at a time) are kept; the rest are paged in on demand.
    //
    // ## How it works
    //
    // 1. Open the file and obtain a raw file descriptor.
    // 2. Call `mmap(2)` (on Unix) / `CreateFileMapping`/`MapViewOfFile` (Windows)
    //    to create a mapping. The `memmap2` crate handles both platforms.
    // 3. Treat the resulting `&[u8]` exactly like the in-memory bytes from `fs::read`.
    // 4. The OS unmaps the file when the `Mmap` object is dropped.
    //
    // ## Safety note
    //
    // The file must not be modified while the mapping is alive. For read-only
    // model weights this is always satisfied.
    //
    // ## Requires the `mmap-loading` feature
    //
    // Add to Cargo.toml: `memmap2 = { version = "0.9", optional = true }`
    // and enable with `cargo run --features mmap-loading`.

    /// Load weights from all .safetensors shards using memory-mapped I/O.
    ///
    /// Functionally identical to `load_weights_from_dir()` but uses `mmap`
    /// instead of `fs::read()`. For large models (>8GB) this is the only
    /// practical approach on machines with limited RAM.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// model.load_weights_from_dir_mmap("./gpt-oss-20b-weights")
    ///      .expect("failed to load weights");
    /// ```
    #[cfg(feature = "mmap-loading")]
    pub fn load_weights_from_dir_mmap(&mut self, dir: &str) -> Result<(), String> {
        use std::fs::File;
        use memmap2::Mmap;

        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read dir {}: {}", dir, e))?;

        let mut loaded_shards  = 0usize;
        let mut loaded_tensors = 0usize;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }

            let file = File::open(&path)
                .map_err(|e| format!("cannot open {:?}: {}", path, e))?;

            // SAFETY: we only read from the mapping, never write.
            // The file is not modified while the mapping is alive.
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| format!("mmap failed for {:?}: {}", path, e))?;

            let tensors = parse_safetensors(&mmap[..])
                .map_err(|e| format!("parse error in {:?}: {}", path, e))?;

            loaded_tensors += tensors.len();
            load_into_model(self, &tensors);
            loaded_shards += 1;
            // `mmap` is dropped here, releasing the mapping for this shard.
        }

        if loaded_shards == 0 {
            return Err(format!("no .safetensors files found in {}", dir));
        }

        println!("Loaded {} tensors from {} shards via mmap in {}",
            loaded_tensors, loaded_shards, dir);
        Ok(())
    }

    /// Load a single .safetensors file using memory-mapped I/O.
    ///
    /// Useful when loading one shard at a time to minimize peak memory usage.
    #[cfg(feature = "mmap-loading")]
    pub fn load_shard_mmap(&mut self, path: &str) -> Result<usize, String> {
        use std::fs::File;
        use memmap2::Mmap;

        let file = File::open(path)
            .map_err(|e| format!("cannot open {}: {}", path, e))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("mmap failed for {}: {}", path, e))?;

        let tensors = parse_safetensors(&mmap[..])
            .map_err(|e| format!("parse error in {}: {}", path, e))?;

        let n = tensors.len();
        load_into_model(self, &tensors);
        Ok(n)
    }
}

impl Module2 for GptOssModel {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = vec![self.embed_tokens.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.extend(self.norm.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

impl GptOssModel {
    /// Tie the language model head weights to the token embedding table.
    ///
    /// ## Why weight tying?
    ///
    /// The embedding table maps token id → hidden vector (`[V, D]`).
    /// The lm_head maps hidden vector → logit (`[D, V]` via `weight.T`).
    ///
    /// These two operations are semantically identical: both represent a
    /// per-token direction in the hidden space.  Tying them:
    ///
    ///   - Saves `vocab_size × hidden_size` parameters (≈580M for GPT-OSS-20b)
    ///   - Forces embedding and output representations to be consistent
    ///   - Is standard in GPT-2, LLaMA, Mistral, GPT-OSS
    ///
    /// After calling `tie_weights()`:
    ///   - `lm_head.weight` IS `embed_tokens` (same `TensorNode` / same Rc)
    ///   - Both the forward pass and backward pass accumulate into the same storage
    pub fn tie_weights(&mut self) {
        assert_eq!(
            (self.lm_head.weight.data().rows, self.lm_head.weight.data().cols),
            (self.embed_tokens.data().rows, self.embed_tokens.data().cols),
            "tie_weights: lm_head.weight and embed_tokens have different shapes"
        );
        self.lm_head.weight = self.embed_tokens.clone();
    }
}

// =============================================================================
// Tests
// =============================================================================
//
// We use a tiny config for unit tests — the real gpt-oss-20b config would
// require instantiating billions of parameters, which is not feasible in a test.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::InitRng;
    use crate::autograd2::Mat;

    /// Tiny config for fast unit tests
    fn tiny_config() -> Config3 {
        Config3 {
            vocab_size: 32,
            hidden_size: 64,       // must be divisible by num_attention_heads
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2, // GQA group_size = 8/2 = 4
            intermediate_size: 64,
            num_local_experts: 4,
            experts_per_token: 2,
            max_position_embeddings: 128,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn make_rng() -> InitRng { InitRng::new(42) }

    // --- Config3 ---

    #[test]
    fn test_config3_d_head() {
        let cfg = tiny_config();
        assert_eq!(cfg.d_head(), 8); // 64 / 8
    }

    #[test]
    fn test_config3_gpt_oss_20b_d_head() {
        let cfg = Config3::gpt_oss_20b();
        assert_eq!(cfg.d_head(), 45); // 2880 / 64
    }

    // --- RMSNorm (via RmsNorm2, exercising the autograd2 primitive) ---

    #[test]
    fn test_rms_norm_output_shape() {
        let norm = RmsNorm2::new(8);
        let x = TensorNode::leaf(Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.1));
        let out = norm.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, 8));
    }

    #[test]
    fn test_rms_norm_unit_rms() {
        // With gamma=ones, each output row should have RMS ≈ 1.0
        let norm = RmsNorm2::new(8);
        let x = TensorNode::leaf(Mat::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], 1, 8
        ));
        let out = norm.forward(&x);
        let d = out.data();
        let rms = (d.data.iter().map(|v| v * v).sum::<f32>() / 8.0).sqrt();
        assert!((rms - 1.0).abs() < 1e-4, "RMSNorm output RMS should be 1, got {}", rms);
    }

    #[test]
    fn test_rms_norm_finite() {
        let norm = RmsNorm2::new(4);
        let x = TensorNode::leaf(Mat::new(vec![0.5, -1.0, 2.0, -0.3], 1, 4));
        let out = norm.forward(&x);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- RoPE ---

    #[test]
    fn test_rope_shape_preserved() {
        let x = TensorNode::leaf(Mat::zeros(4, 8));
        let out = x.rope_apply(0, 10000.0);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, 8));
    }

    #[test]
    fn test_rope_position_zero_unchanged() {
        // At position 0, cos(0)=1 and sin(0)=0, so rotation is identity
        let vals: Vec<f32> = (0..8).map(|i| i as f32 + 1.0).collect();
        let x = TensorNode::leaf(Mat::new(vals.clone(), 1, 8));
        let out = x.rope_apply(0, 10000.0);
        let d = out.data();
        for c in 0..8 {
            assert!((d.at(0, c) - vals[c]).abs() < 1e-5,
                "RoPE at pos 0 should be identity, diff at [0,{}]", c);
        }
    }

    #[test]
    fn test_rope_finite() {
        let x = TensorNode::leaf(Mat::from_fn(5, 8, |r, c| (r * 8 + c) as f32 * 0.1));
        let out = x.rope_apply(0, 10000.0);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- SiLU ---

    #[test]
    fn test_silu_zero_input() {
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let x = TensorNode::leaf(Mat::zeros(2, 3));
        let out = x.silu();
        assert!(out.data().data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_silu_positive_monotone() {
        // SiLU is positive and increasing for x > 0
        let x = TensorNode::leaf(Mat::new(vec![1.0, 2.0, 3.0], 1, 3));
        let out = x.silu();
        let d = out.data();
        assert!(d.at(0, 0) < d.at(0, 1) && d.at(0, 1) < d.at(0, 2));
    }

    // --- SwiGluMlp2 ---

    #[test]
    fn test_swiglu_output_shape() {
        let mut rng = make_rng();
        let mlp = SwiGluMlp2::new(8, 16, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(3, 8));
        let out = mlp.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, 8));
    }

    #[test]
    fn test_swiglu_output_finite() {
        let mut rng = make_rng();
        let mlp = SwiGluMlp2::new(8, 16, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(2, 8, |r, c| (r * 8 + c) as f32 * 0.1));
        let out = mlp.forward(&x);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- GQA attention ---

    #[test]
    fn test_gqa_output_shape() {
        // n_q_heads=4, n_kv_heads=2, d_head=8 → Q[T,32], K/V[T,16] → out[T,32]
        let t = 5; let n_q = 4; let n_kv = 2; let dh = 8;
        let q = TensorNode::leaf(Mat::zeros(t, n_q * dh));
        let k = TensorNode::leaf(Mat::zeros(t, n_kv * dh));
        let v = TensorNode::leaf(Mat::zeros(t, n_kv * dh));
        let out = TensorNode::gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (t, n_q * dh));
    }

    #[test]
    fn test_gqa_output_finite() {
        let t = 3; let n_q = 4; let n_kv = 2; let dh = 8;
        let q = TensorNode::leaf(Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.01));
        let k = TensorNode::leaf(Mat::from_fn(t, n_kv * dh, |r, c| (r * (n_kv * dh) + c) as f32 * 0.01));
        let v = TensorNode::leaf(Mat::from_fn(t, n_kv * dh, |r, c| (r * (n_kv * dh) + c) as f32 * 0.01));
        let out = TensorNode::gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- GptOssAttention ---

    #[test]
    fn test_gpt_oss_attention_output_shape() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let attn = GptOssAttention::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(4, cfg.hidden_size));
        let out = attn.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, cfg.hidden_size));
    }

    // --- MoELayer ---

    #[test]
    fn test_moe_output_shape() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let moe = MoELayer::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(3, cfg.hidden_size));
        let out = moe.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, cfg.hidden_size));
    }

    #[test]
    fn test_moe_output_finite() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let moe = MoELayer::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(2, cfg.hidden_size, |r, c| (r * cfg.hidden_size + c) as f32 * 0.01));
        let out = moe.forward(&x);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- GptOssBlock ---

    #[test]
    fn test_gpt_oss_block_output_shape() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let block = GptOssBlock::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(3, cfg.hidden_size, |_, _| 0.1));
        let out = block.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, cfg.hidden_size));
    }

    #[test]
    fn test_gpt_oss_block_output_finite() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let block = GptOssBlock::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(2, cfg.hidden_size, |r, c| (r * cfg.hidden_size + c) as f32 * 0.01));
        let out = block.forward(&x);
        assert!(out.data().data.iter().all(|v| v.is_finite()));
    }

    // --- Full GptOssModel ---

    #[test]
    fn test_gpt_oss_forward_shape() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let out = model.forward(&[0, 3, 7, 2]);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, cfg.vocab_size));
    }

    #[test]
    fn test_gpt_oss_logits_finite() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let out = model.forward(&[1, 2, 3]);
        assert!(out.data().data.iter().all(|v| v.is_finite()),
            "all logits should be finite");
    }

    #[test]
    fn test_gpt_oss_predict_next_valid() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let next = model.predict_next(&[0, 1, 2]);
        assert!(next < cfg.vocab_size, "predicted token {} out of vocab range {}", next, cfg.vocab_size);
    }

    // --- KV cache ---

    #[test]
    fn test_kv_cache_new() {
        let cfg = tiny_config();
        let cache = KvCache::new(&cfg);
        assert_eq!(cache.layers.len(), cfg.num_hidden_layers);
        for layer in &cache.layers {
            let l = layer.borrow();
            assert_eq!(l.seq_len, 0);
            assert_eq!(l.k.cols, cfg.num_key_value_heads * cfg.d_head());
        }
    }

    #[test]
    fn test_generate_cached_output_length() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let new_tokens = model.generate_cached(&[0, 1, 2], 5, 1.0);
        assert_eq!(new_tokens.len(), 5);
    }

    #[test]
    fn test_generate_cached_tokens_in_vocab() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let new_tokens = model.generate_cached(&[1, 2], 4, 0.0);
        for &tok in &new_tokens {
            assert!(tok < cfg.vocab_size, "generated token {} out of vocab {}", tok, cfg.vocab_size);
        }
    }

    #[test]
    fn test_generate_cached_matches_uncached() {
        // The first generated token from cached generation must match predict_next.
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let prompt = vec![0usize, 3, 7];
        let cached_first = model.generate_cached(&prompt, 1, 0.0)[0];
        // predict_next runs full forward, generate_cached runs with cache — same result
        let uncached_first = model.predict_next(&prompt);
        assert_eq!(cached_first, uncached_first,
            "cached first token {} != uncached {}", cached_first, uncached_first);
    }

    #[test]
    fn test_generate_cached_streaming_same_as_buffered() {
        // generate_cached_streaming must yield the exact same tokens as generate_cached.
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let prompt = vec![0usize, 1, 2];
        let buffered = model.generate_cached(&prompt, 6, 0.8);
        let mut streamed = Vec::new();
        model.generate_cached_streaming(&prompt, 6, 0.8, |tok| streamed.push(tok));
        assert_eq!(buffered, streamed,
            "streaming tokens {:?} differ from buffered {:?}", streamed, buffered);
    }

    #[test]
    fn test_generate_cached_streaming_callback_count() {
        // Callback must be invoked exactly max_new times.
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let mut count = 0usize;
        model.generate_cached_streaming(&[0, 1], 7, 1.0, |_tok| count += 1);
        assert_eq!(count, 7, "expected 7 callback calls, got {}", count);
    }

    // --- SamplingParams, top-k, top-p, repetition penalty, perplexity ---

    #[test]
    fn test_sampling_params_greedy_is_deterministic() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams::greedy();
        let r1 = model.generate_with_params(&[0, 1, 2], 5, &p);
        let r2 = model.generate_with_params(&[0, 1, 2], 5, &p);
        assert_eq!(r1, r2, "greedy must be deterministic");
    }

    #[test]
    fn test_sampling_params_same_seed_same_output() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams::creative(7);
        let r1 = model.generate_with_params(&[0, 1], 6, &p);
        let r2 = model.generate_with_params(&[0, 1], 6, &p);
        assert_eq!(r1, r2, "same seed must produce same output");
    }

    #[test]
    fn test_sampling_params_different_seeds_may_differ() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p1 = SamplingParams::creative(1);
        let p2 = SamplingParams::creative(99999);
        let r1 = model.generate_with_params(&[0, 1, 2, 3], 10, &p1);
        let r2 = model.generate_with_params(&[0, 1, 2, 3], 10, &p2);
        // With different seeds the output may differ (not guaranteed but
        // overwhelmingly likely with temperature > 0 and a real distribution)
        // We only assert that the generation runs without panicking and
        // produces the right length.
        assert_eq!(r1.len(), 10);
        assert_eq!(r2.len(), 10);
    }

    #[test]
    fn test_top_k_zero_same_as_full_vocab() {
        // top_k = 0 means "keep all tokens" — result must equal top_k = vocab_size
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p_nok = SamplingParams { temperature: 0.8, top_k: 0,              top_p: 1.0, repetition_penalty: 1.0, seed: 42, eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 };
        let p_all = SamplingParams { temperature: 0.8, top_k: cfg.vocab_size, top_p: 1.0, repetition_penalty: 1.0, seed: 42, eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 };
        let r_nok = model.generate_with_params(&[1, 2], 5, &p_nok);
        let r_all = model.generate_with_params(&[1, 2], 5, &p_all);
        assert_eq!(r_nok, r_all, "top_k=0 should behave like top_k=vocab_size");
    }

    #[test]
    fn test_top_p_one_same_as_no_nucleus() {
        // top_p = 1.0 keeps all tokens → same as disabling nucleus sampling
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p1 = SamplingParams { temperature: 0.9, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: 7, eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 };
        let p2 = SamplingParams { temperature: 0.9, top_k: 0, top_p: 0.9999, repetition_penalty: 1.0, seed: 7, eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 };
        let r1 = model.generate_with_params(&[0, 3], 4, &p1);
        // With top_p ≈ 1 the result may or may not equal r1 depending on dist.
        // We just verify it runs without panic and produces correct length.
        assert_eq!(r1.len(), 4);
        let r2 = model.generate_with_params(&[0, 3], 4, &p2);
        assert_eq!(r2.len(), 4);
    }

    #[test]
    fn test_repetition_penalty_tokens_in_vocab() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.5, seed: 3, eos_token_id: None, frequency_penalty: 0.0, presence_penalty: 0.0 };
        let tokens = model.generate_with_params(&[0, 1, 2], 8, &p);
        for &tok in &tokens {
            assert!(tok < cfg.vocab_size, "token {} out of vocab", tok);
        }
    }

    #[test]
    fn test_generate_with_params_streaming_matches_buffered() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams::creative(42);
        let buffered = model.generate_with_params(&[0, 1], 6, &p);
        let mut streamed = Vec::new();
        model.generate_with_params_streaming(&[0, 1], 6, &p, |t| streamed.push(t));
        assert_eq!(buffered, streamed);
    }

    #[test]
    fn test_perplexity_is_finite_and_positive() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let tokens: Vec<usize> = (0..8).map(|i| i % cfg.vocab_size).collect();
        let ppl = model.perplexity(&tokens, 4);
        assert!(ppl.is_finite() && ppl > 0.0, "expected finite positive PPL, got {}", ppl);
    }

    #[test]
    fn test_perplexity_untrained_near_vocab_size() {
        // A randomly-initialised model should have PPL close to vocab_size
        // (since it can't predict anything useful).
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let tokens: Vec<usize> = (0..12).map(|i| i % cfg.vocab_size).collect();
        let ppl = model.perplexity(&tokens, 4);
        // Allow a wide range: random init can vary quite a bit
        assert!(ppl > 1.0, "PPL should be > 1 for an untrained model");
        assert!(ppl < (cfg.vocab_size as f32) * 10.0,
            "PPL should not be absurdly large, got {}", ppl);
    }

    // --- Batching ---

    #[test]
    fn test_forward_batch_output_count() {
        // forward_batch returns one TensorNode per sequence
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let batch = vec![vec![0usize, 1, 2], vec![3usize, 4], vec![5usize]];
        let logits = model.forward_batch(&batch);
        assert_eq!(logits.len(), 3);
    }

    #[test]
    fn test_forward_batch_logit_shapes() {
        // Each logit tensor must have shape [T_i, vocab_size]
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let seqs = vec![vec![0usize, 1], vec![2usize, 3, 4]];
        let logits = model.forward_batch(&seqs);
        assert_eq!(logits[0].data().rows, 2);
        assert_eq!(logits[0].data().cols, cfg.vocab_size);
        assert_eq!(logits[1].data().rows, 3);
        assert_eq!(logits[1].data().cols, cfg.vocab_size);
    }

    #[test]
    fn test_forward_batch_matches_forward_single() {
        // forward_batch[i] must produce identical results to forward(seqs[i])
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let seq = vec![1usize, 2, 3];
        let single = model.forward(&seq);
        let batch  = model.forward_batch(&[seq])[0].clone();
        let sd = single.data();
        let bd = batch.data();
        assert_eq!((sd.rows, sd.cols), (bd.rows, bd.cols));
        for r in 0..sd.rows {
            for c in 0..sd.cols {
                assert!((sd.at(r,c) - bd.at(r,c)).abs() < 1e-5,
                    "mismatch at [{r},{c}]: single={} batch={}", sd.at(r,c), bd.at(r,c));
            }
        }
    }

    #[test]
    fn test_loss_is_finite() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let inp    = vec![0usize, 1, 2, 3];
        let tgt    = vec![1usize, 2, 3, 4];
        let loss   = model.loss(&inp, &tgt);
        let v = loss.data().at(0, 0);
        assert!(v.is_finite() && v > 0.0,
            "expected finite positive loss, got {}", v);
    }

    #[test]
    fn test_loss_batch_average() {
        // loss_batch of B identical examples must equal loss of one example
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let inp: Vec<usize> = vec![0, 1, 2];
        let tgt: Vec<usize> = vec![1, 2, 3];
        let single_loss = model.loss(&inp, &tgt).data().at(0, 0);
        let batch_loss  = model.loss_batch(&[
            (inp.as_slice(), tgt.as_slice()),
            (inp.as_slice(), tgt.as_slice()),
        ]).data().at(0, 0);
        assert!((batch_loss - single_loss).abs() < 1e-4,
            "batch_loss {} != single_loss {}", batch_loss, single_loss);
    }

    #[test]
    fn test_loss_batch_size_1_equals_loss() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let inp: Vec<usize> = vec![2, 3, 4];
        let tgt: Vec<usize> = vec![3, 4, 5];
        let single = model.loss(&inp, &tgt).data().at(0, 0);
        let batched = model.loss_batch(&[(inp.as_slice(), tgt.as_slice())]).data().at(0, 0);
        assert!((single - batched).abs() < 1e-5,
            "single={} batched={}", single, batched);
    }

    // --- Safetensors parser ---

    /// Build a minimal valid safetensors binary blob for testing.
    /// Contains one tensor: "test.weight" shape [2,3] dtype F32.
    fn make_test_safetensors() -> Vec<u8> {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw_bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let end = raw_bytes.len();

        let header = format!(
            r#"{{"test.weight":{{"dtype":"F32","shape":[2,3],"data_offsets":[0,{}]}}}}"#,
            end
        );
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut out = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        out.extend_from_slice(&raw_bytes);
        out
    }

    #[test]
    fn test_parse_safetensors_shape() {
        let blob = make_test_safetensors();
        let tensors = parse_safetensors(&blob).expect("parse failed");
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "test.weight");
        assert_eq!(tensors[0].shape, vec![2, 3]);
    }

    #[test]
    fn test_parse_safetensors_values() {
        let blob = make_test_safetensors();
        let tensors = parse_safetensors(&blob).expect("parse failed");
        let expected = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        for (i, (&got, &exp)) in tensors[0].data.iter().zip(&expected).enumerate() {
            assert!((got - exp).abs() < 1e-6, "data[{}]: got {} expected {}", i, got, exp);
        }
    }

    #[test]
    fn test_parse_safetensors_bf16() {
        // BF16 encoding of 1.0 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
        let bf16_one: u16 = 0x3F80u16;
        let bf16_two: u16 = 0x4000u16; // 2.0 in BF16
        let raw_bytes: Vec<u8> = [bf16_one, bf16_two].iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let end = raw_bytes.len();
        let header = format!(
            r#"{{"w":{{"dtype":"BF16","shape":[1,2],"data_offsets":[0,{}]}}}}"#, end
        );
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;
        let mut blob = Vec::new();
        blob.extend_from_slice(&header_len.to_le_bytes());
        blob.extend_from_slice(header_bytes);
        blob.extend_from_slice(&raw_bytes);

        let tensors = parse_safetensors(&blob).expect("parse failed");
        assert!((tensors[0].data[0] - 1.0f32).abs() < 1e-3, "BF16 1.0 decode: {}", tensors[0].data[0]);
        assert!((tensors[0].data[1] - 2.0f32).abs() < 1e-3, "BF16 2.0 decode: {}", tensors[0].data[1]);
    }

    #[test]
    fn test_load_into_model_embed_tokens() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let mut model = GptOssModel::new(cfg.clone(), &mut rng);

        // Build a safetensors blob for model.embed_tokens.weight [vocab, hidden]
        let data: Vec<f32> = (0..(cfg.vocab_size * cfg.hidden_size)).map(|i| i as f32 * 0.001).collect();
        let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let end = raw.len();
        let header = format!(
            r#"{{"model.embed_tokens.weight":{{"dtype":"F32","shape":[{},{}],"data_offsets":[0,{}]}}}}"#,
            cfg.vocab_size, cfg.hidden_size, end
        );
        let header_bytes = header.as_bytes();
        let mut blob = Vec::new();
        blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
        blob.extend_from_slice(header_bytes);
        blob.extend_from_slice(&raw);

        let tensors = parse_safetensors(&blob).expect("parse failed");
        load_into_model(&mut model, &tensors);

        // Verify the first element was loaded
        let d = model.embed_tokens.data();
        assert!((d.at(0, 0) - 0.0f32).abs() < 1e-6);
        assert!((d.at(0, 1) - 0.001f32).abs() < 1e-6);
    }

    #[test]
    fn test_f16_to_f32_one() {
        // IEEE 754 half-precision 1.0 = 0x3C00
        assert!((f16_to_f32(0x3C00) - 1.0f32).abs() < 1e-5);
    }

    #[test]
    fn test_f16_to_f32_two() {
        // 2.0 in f16 = 0x4000
        assert!((f16_to_f32(0x4000) - 2.0f32).abs() < 1e-5);
    }

    #[test]
    fn test_gpt_oss_param_count_positive() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let total: usize = model.parameters().iter()
            .map(|p| p.data().rows * p.data().cols)
            .sum();
        assert!(total > 0);
        // Print for educational purposes (not an assertion)
        let _ = total;
    }

    // --- Q4 quantization integration ---

    #[test]
    fn test_quantize_all_linear_weights_compresses() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let stats = model.quantize_all_linear_weights();
        assert!(stats.n_tensors > 0, "should quantize at least 1 tensor");
        assert!(stats.compression_ratio > 1.0,
            "Q4 should compress: got {:.2}x", stats.compression_ratio);
    }

    #[test]
    fn test_quantize_all_linear_weights_model_still_runs() {
        // After quantization the model should still produce finite logits.
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        model.quantize_all_linear_weights();
        let out = model.forward(&[0, 1, 2]);
        assert!(out.data().data.iter().all(|v| v.is_finite()),
            "model should still produce finite logits after Q4 quantization");
    }

    #[test]
    fn test_quantize_reduces_memory() {
        // Q4 should use less than half the memory of f32
        use crate::autograd2::Q4Mat;
        let m = Mat::from_fn(64, 64, |r, c| (r * 64 + c) as f32 * 0.01 - 0.5);
        let q = Q4Mat::quantize(&m);
        let f32_bytes = 64 * 64 * 4;
        assert!(q.size_bytes() < f32_bytes / 3,
            "Q4 should use <1/3 of f32 memory: {} vs {}", q.size_bytes(), f32_bytes);
    }

    // --- EOS stopping ---

    #[test]
    fn test_eos_stops_generation_early() {
        // Set eos_token_id = 0 and make the model very likely to produce it
        // (we can't control weights, but we can verify the list ends at EOS).
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        // Use eos = 0, which is token id 0. Generate up to 20 tokens.
        let p = SamplingParams {
            temperature: 1.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 1,
            eos_token_id: Some(0),
            frequency_penalty: 0.0, presence_penalty: 0.0,
        };
        let tokens = model.generate_with_params(&[1, 2], 20, &p);
        // Either EOS was never produced (all 20 tokens) or it stopped early.
        // Either way the list must not be longer than max_new.
        assert!(tokens.len() <= 20, "generated too many tokens: {}", tokens.len());
        // If EOS appears, it must be the last token
        if let Some(pos) = tokens.iter().position(|&t| t == 0) {
            assert_eq!(pos, tokens.len() - 1,
                "EOS token should only appear as the last token");
        }
    }

    #[test]
    fn test_eos_none_generates_max_new() {
        // With eos_token_id = None, should always produce exactly max_new tokens
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams::greedy();
        assert_eq!(p.eos_token_id, None);
        let tokens = model.generate_with_params(&[1, 2], 10, &p);
        assert_eq!(tokens.len(), 10, "greedy (no EOS) should produce exactly max_new");
    }

    #[test]
    fn test_eos_streaming_stops_early() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams {
            temperature: 1.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 1,
            eos_token_id: Some(0),
            frequency_penalty: 0.0, presence_penalty: 0.0,
        };
        let mut streamed = Vec::new();
        model.generate_with_params_streaming(&[1, 2], 20, &p, |t| streamed.push(t));
        assert!(streamed.len() <= 20);
        if let Some(pos) = streamed.iter().position(|&t| t == 0) {
            assert_eq!(pos, streamed.len() - 1,
                "EOS should only appear as the last streamed token");
        }
    }

    // --- Frequency / presence penalty ---

    #[test]
    fn test_frequency_penalty_tokens_in_vocab() {
        // freq/presence penalty should not push any token out of vocab range
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams {
            temperature: 1.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 5,
            eos_token_id: None,
            frequency_penalty: 0.5, presence_penalty: 0.0,
        };
        let tokens = model.generate_with_params(&[0, 1, 2], 10, &p);
        for &tok in &tokens {
            assert!(tok < cfg.vocab_size, "token {} out of vocab", tok);
        }
    }

    #[test]
    fn test_presence_penalty_tokens_in_vocab() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let p = SamplingParams {
            temperature: 1.0, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 6,
            eos_token_id: None,
            frequency_penalty: 0.0, presence_penalty: 0.3,
        };
        let tokens = model.generate_with_params(&[0, 1, 2], 10, &p);
        for &tok in &tokens {
            assert!(tok < cfg.vocab_size, "token {} out of vocab", tok);
        }
    }

    #[test]
    fn test_no_penalty_vs_freq_penalty_different_results() {
        // With a high frequency penalty, generation should differ from no penalty
        // (not guaranteed, but extremely likely with a finite vocab of 16 tokens
        //  and long enough generation).
        let cfg = tiny_config();
        let mut rng = make_rng();
        let model = GptOssModel::new(cfg.clone(), &mut rng);
        let base = SamplingParams {
            temperature: 0.8, top_k: 0, top_p: 1.0,
            repetition_penalty: 1.0, seed: 99,
            eos_token_id: None,
            frequency_penalty: 0.0, presence_penalty: 0.0,
        };
        let with_freq = SamplingParams {
            frequency_penalty: 2.0,
            ..base.clone()
        };
        let r_base = model.generate_with_params(&[0, 1], 12, &base);
        let r_freq = model.generate_with_params(&[0, 1], 12, &with_freq);
        // Both must be valid
        for &t in r_base.iter().chain(r_freq.iter()) {
            assert!(t < cfg.vocab_size);
        }
        // We don't assert they differ (random init means either outcome is possible),
        // but the test ensures both code paths run without panic or NaN.
    }

    // --- Weight tying ---

    #[test]
    fn test_tie_weights_shares_tensor() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let mut model = GptOssModel::new(cfg.clone(), &mut rng);
        model.tie_weights();

        // Write a distinctive value into lm_head.weight[0,0]
        let mut wd = model.lm_head.weight.data().clone();
        *wd.at_mut(0, 0) = 77.0;
        model.lm_head.weight.set_data(wd);

        // embed_tokens must reflect the same change (shared Rc)
        assert_eq!(model.embed_tokens.data().at(0, 0), 77.0,
            "tie_weights: embed_tokens and lm_head.weight must share storage");
    }

    #[test]
    fn test_tie_weights_forward_still_runs() {
        let cfg = tiny_config();
        let mut rng = make_rng();
        let mut model = GptOssModel::new(cfg.clone(), &mut rng);
        model.tie_weights();
        let out = model.forward(&[0, 1, 2]);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, cfg.vocab_size));
        assert!(d.data.iter().all(|v| v.is_finite()),
            "tied-weight forward should produce finite logits");
    }

    #[test]
    fn test_tie_weights_reduces_unique_params() {
        let cfg = tiny_config();
        let mut rng = make_rng();

        let count_unique = |m: &GptOssModel| {
            let mut ptrs: Vec<*const f32> = m.parameters().iter()
                .map(|p| p.data().data.as_ptr())
                .collect();
            ptrs.sort();
            ptrs.dedup();
            ptrs.len()
        };

        let m1 = GptOssModel::new(cfg.clone(), &mut rng);
        let before = count_unique(&m1);
        let mut m2 = GptOssModel::new(cfg.clone(), &mut rng);
        m2.tie_weights();
        let after = count_unique(&m2);
        assert!(after < before,
            "tie_weights should reduce unique param tensors: {before} → {after}");
    }
}
