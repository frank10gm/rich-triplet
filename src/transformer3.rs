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

use crate::autograd2::{TensorNode, Mat};
use crate::nn::InitRng;
use crate::nn2::{Linear2, RmsNorm2, SwiGluMlp2, Module2};

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
}

impl GptOssAttention {
    pub fn new(config: &Config3, rng: &mut InitRng) -> Self {
        let d_head = config.d_head();
        GptOssAttention {
            q_proj: Linear2::new(config.hidden_size, config.num_attention_heads * d_head, rng),
            k_proj: Linear2::new(config.hidden_size, config.num_key_value_heads * d_head, rng),
            v_proj: Linear2::new(config.hidden_size, config.num_key_value_heads * d_head, rng),
            o_proj: Linear2::new(config.num_attention_heads * d_head, config.hidden_size, rng),
            n_q_heads: config.num_attention_heads,
            n_kv_heads: config.num_key_value_heads,
            d_head,
            rope_theta: config.rope_theta,
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

    /// Apply RoPE to every head in a [T, n_heads * d_head] tensor.
    ///
    /// We process head by head, apply rope_apply to each [T, d_head] slice,
    /// then reassemble into [T, n_heads * d_head].
    fn apply_rope_to_all_heads(
        &self,
        x: &TensorNode,
        n_heads: usize,
        t: usize,
        d_head: usize,
    ) -> TensorNode {
        let x_data = x.data().clone();

        let out_data = Mat::from_fn(t, n_heads * d_head, |row, col| {
            let h   = col / d_head;
            let dim = col % d_head;
            let pair = dim / 2;
            let is_odd = dim % 2 == 1;
            let pos = row as f32;
            let angle = pos / self.rope_theta.powf(2.0 * pair as f32 / d_head as f32);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());

            let base_col = h * d_head + (dim & !1); // even partner col
            if !is_odd {
                x_data.at(row, base_col)     * cos_a - x_data.at(row, base_col + 1) * sin_a
            } else {
                x_data.at(row, base_col + 1) * cos_a + x_data.at(row, base_col)     * sin_a
            }
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

    /// Greedy next-token prediction for a prompt.
    ///
    /// Returns the index of the most likely next token.
    /// For real generation, apply temperature + top-k sampling (see train2.rs).
    pub fn predict_next(&self, token_ids: &[usize]) -> usize {
        let logits = self.forward(token_ids);
        let logits_data = logits.data();
        let t = logits_data.rows;
        let v = logits_data.cols;

        // Take last token's logits and return argmax
        (0..v)
            .max_by(|&a, &b| logits_data.at(t - 1, a)
                .partial_cmp(&logits_data.at(t - 1, b))
                .unwrap())
            .unwrap()
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
}
