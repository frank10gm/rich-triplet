/// # Gemma 3 — inference-only architecture
///
/// Implements the Gemma 3 family of models (1b, 4b, 12b, 27b) for
/// text-only inference. Weights are loaded from HuggingFace `.safetensors`
/// shards in bfloat16 or float32.
///
/// ## Architecture overview (vs GPT-OSS / transformer3.rs)
///
/// | Feature                  | GPT-OSS (transformer3) | Gemma 3 (transformer4) |
/// |--------------------------|------------------------|------------------------|
/// | head_dim                 | hidden/n_heads         | explicit (256)         |
/// | Q/K per-head RMSNorm     | no                     | yes                    |
/// | Attention scale          | 1/sqrt(d_head)         | 1/sqrt(query_pre_attn_scalar) |
/// | Local/global attention   | odd layers only        | 5 local : 1 global     |
/// | FFN type                 | MoE                    | dense SwiGLU           |
/// | Logit soft-capping       | no                     | no (null in Gemma 3)   |
/// | Vocab size               | 201088                 | 262144 (1b) / 262208 (4b) |
///
/// ## Inference only
///
/// No backward passes are implemented. This module is for loading Google's
/// pretrained weights and generating text.  To train from scratch with the
/// same architecture use the `Trainable` impl below, which does have backward.
///
/// ## Weight loading
///
/// ```
/// let mut model = Gemma3Model::new(Config4::gemma3_1b(), &mut rng);
/// model.load_weights_from_dir("/path/to/gemma-3-1b-it").unwrap();
/// ```
///
/// ## Tokenizer
///
/// Gemma 3 uses a SentencePiece tokenizer (`tokenizer.model`). Pass token
/// ids produced by an external tokenizer (Python `transformers`, etc.), or
/// use the `sentencepiece` feature once added to this project.

use crate::autograd2::{TensorNode, Mat};
use crate::nn2::{Linear2, RmsNorm2, SwiGluMlp2, Module2, Trainable};
use crate::nn::InitRng;
use std::cell::RefCell;

// ============================================================================
// Config4 — Gemma 3 hyperparameters
// ============================================================================

/// Hyperparameters for a Gemma 3 model.
///
/// Unlike GPT-OSS, `head_dim` is stored explicitly because in Gemma 3
/// `head_dim = 256` regardless of `hidden_size / num_attention_heads`.
#[derive(Clone, Debug)]
pub struct Config4 {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    /// Number of Q heads per layer.
    pub num_attention_heads: usize,
    /// Number of K/V heads per layer (GQA — fewer than Q heads).
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    /// Explicit head dimension. In Gemma 3 this is always 256.
    pub head_dim: usize,
    /// Sliding window size for local attention layers (None = full attention).
    pub sliding_window: Option<usize>,
    /// RoPE base theta. 10_000 for local layers, 1_000_000 for global in Gemma 3.
    pub rope_theta_local: f32,
    pub rope_theta_global: f32,
    pub rms_norm_eps: f32,
    /// Attention score scale = 1 / sqrt(query_pre_attn_scalar).
    /// In Gemma 3 this is 256.0 so scale = 1/16.
    pub query_pre_attn_scalar: f32,
    /// EOS token id (used to stop generation).
    pub eos_token_id: usize,
    /// Maximum sequence length for KV cache allocation.
    pub max_position_embeddings: usize,
}

impl Config4 {
    /// Gemma 3 1B configuration.
    /// Source: google/gemma-3-1b-it config.json
    pub fn gemma3_1b() -> Self {
        Config4 {
            vocab_size: 262144,
            hidden_size: 1152,
            num_hidden_layers: 26,
            num_attention_heads: 4,
            num_key_value_heads: 1,
            intermediate_size: 6912,
            head_dim: 256,
            sliding_window: Some(512),
            rope_theta_local: 10_000.0,
            rope_theta_global: 1_000_000.0,
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            eos_token_id: 1,      // <eos> in Gemma tokenizer
            max_position_embeddings: 32768,
        }
    }

    /// Gemma 3 4B configuration.
    /// Source: google/gemma-3-4b-it config.json / unsloth/gemma-3-4b-pt config.json
    ///
    /// rope_scaling: {factor: 8.0, rope_type: "linear"} applies to global layers only.
    /// Effective global theta = rope_theta (1M) * factor (8) = 8M.
    /// Local layers use rope_local_base_freq = 10_000 (no scaling).
    pub fn gemma3_4b() -> Self {
        Config4 {
            vocab_size: 262208,
            hidden_size: 2560,
            num_hidden_layers: 34,
            num_attention_heads: 8,
            num_key_value_heads: 4,
            intermediate_size: 10240,
            head_dim: 256,
            sliding_window: Some(1024),
            rope_theta_local: 10_000.0,
            rope_theta_global: 8_000_000.0,  // 1_000_000 * rope_scaling.factor(8.0)
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            eos_token_id: 1,  // also 106, checked separately in generate loop
            max_position_embeddings: 32768,
        }
    }

    /// Returns true if layer `layer_idx` uses global (full) attention.
    ///
    /// Gemma 3 pattern: 5 local layers then 1 global, repeating.
    ///   layer 0–4  → local (sliding window)
    ///   layer 5    → global (full causal)
    ///   layer 6–10 → local
    ///   layer 11   → global
    ///   etc.
    pub fn is_global_layer(&self, layer_idx: usize) -> bool {
        layer_idx % 6 == 5
    }
}

// ============================================================================
// Gemma3Attention
// ============================================================================

/// Single attention layer for Gemma 3.
///
/// Differences from GPT-OSS attention:
/// - Per-head RMSNorm on Q and K (after projection, before RoPE).
/// - Explicit `head_dim` (not hidden/n_heads).
/// - Scale = 1/sqrt(query_pre_attn_scalar) instead of 1/sqrt(head_dim).
/// - Local layers use a sliding window causal mask.
/// - RoPE theta differs between local (10k) and global (1M) layers.
pub struct Gemma3Attention {
    pub q_proj:  Linear2,   // [hidden, n_q_heads * head_dim]
    pub k_proj:  Linear2,   // [hidden, n_kv_heads * head_dim]
    pub v_proj:  Linear2,   // [hidden, n_kv_heads * head_dim]
    pub o_proj:  Linear2,   // [n_q_heads * head_dim, hidden]
    /// Per-head RMSNorm on Q (Gemma 3 specific).
    pub q_norm:  RmsNorm2,  // [1, head_dim]
    /// Per-head RMSNorm on K (Gemma 3 specific).
    pub k_norm:  RmsNorm2,  // [1, head_dim]
    pub n_q_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    /// Scale applied to QK dot products = 1/sqrt(query_pre_attn_scalar).
    pub attn_scale: f32,
    /// None = global attention (full causal mask).
    /// Some(w) = local attention (causal mask limited to last w tokens).
    pub sliding_window: Option<usize>,
    /// RoPE theta for this layer (10k for local, 1M for global).
    pub rope_theta: f32,
}

impl Gemma3Attention {
    pub fn new(cfg: &Config4, layer_idx: usize, rng: &mut InitRng) -> Self {
        let h = cfg.hidden_size;
        let d = cfg.head_dim;
        let nq = cfg.num_attention_heads;
        let nkv = cfg.num_key_value_heads;

        let is_global = cfg.is_global_layer(layer_idx);

        Gemma3Attention {
            q_proj:  Linear2::new_no_bias(h, nq * d, rng),
            k_proj:  Linear2::new_no_bias(h, nkv * d, rng),
            v_proj:  Linear2::new_no_bias(h, nkv * d, rng),
            o_proj:  Linear2::new_no_bias(nq * d, h, rng),
            q_norm:  RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            k_norm:  RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim: d,
            attn_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
            sliding_window: if is_global { None } else { cfg.sliding_window },
            rope_theta: if is_global { cfg.rope_theta_global } else { cfg.rope_theta_local },
        }
    }

    /// Forward pass: x [T, hidden] → output [T, hidden].
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let t = x.data().rows;
        let d = self.head_dim;
        let nq = self.n_q_heads;
        let nkv = self.n_kv_heads;

        // Project to Q, K, V
        let q = self.q_proj.forward(x);  // [T, nq*d]
        let k = self.k_proj.forward(x);  // [T, nkv*d]
        let v = self.v_proj.forward(x);  // [T, nkv*d]

        // Per-head RMSNorm on Q and K (Gemma 3 specific)
        let q = apply_per_head_norm(&q, &self.q_norm, t, nq, d);
        let k = apply_per_head_norm(&k, &self.k_norm, t, nkv, d);

        // Apply RoPE to each head
        let q = apply_rope_to_all_heads(&q, nq, t, d, self.rope_theta);
        let k = apply_rope_to_all_heads(&k, nkv, t, d, self.rope_theta);

        // GQA attention
        let attn_out = if let Some(window) = self.sliding_window {
            gqa_attention_windowed(&q, &k, &v, nq, nkv, d, window, self.attn_scale)
        } else {
            gqa_attention_full(&q, &k, &v, nq, nkv, d, self.attn_scale)
        };

        self.o_proj.forward(&attn_out)
    }
}

impl Module2 for Gemma3Attention {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = vec![
            self.q_proj.weight.clone(),
            self.k_proj.weight.clone(),
            self.v_proj.weight.clone(),
            self.o_proj.weight.clone(),
            self.q_norm.gamma.clone(),
            self.k_norm.gamma.clone(),
        ];
        // Linear2 biases are always zero for no-bias layers but we still include them
        // so parameter count is consistent.  (They are frozen at zero.)
        p
    }
}

// ============================================================================
// Gemma3MLP — dense SwiGLU (no MoE)
// ============================================================================

/// Dense SwiGLU FFN as used in Gemma 3.
///
/// output = down_proj(SiLU(gate_proj(x)) * up_proj(x))
pub struct Gemma3Mlp {
    pub gate_proj: Linear2,   // [hidden, intermediate]
    pub up_proj:   Linear2,   // [hidden, intermediate]
    pub down_proj: Linear2,   // [intermediate, hidden]
}

impl Gemma3Mlp {
    pub fn new(cfg: &Config4, rng: &mut InitRng) -> Self {
        Gemma3Mlp {
            gate_proj: Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng),
            up_proj:   Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng),
            down_proj: Linear2::new_no_bias(cfg.intermediate_size, cfg.hidden_size, rng),
        }
    }

    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let gate = self.gate_proj.forward(x).silu();
        let up   = self.up_proj.forward(x);
        let hidden = gate.mul_elem_node(&up);
        self.down_proj.forward(&hidden)
    }
}

impl Module2 for Gemma3Mlp {
    fn parameters(&self) -> Vec<TensorNode> {
        vec![
            self.gate_proj.weight.clone(),
            self.up_proj.weight.clone(),
            self.down_proj.weight.clone(),
        ]
    }
}

// ============================================================================
// Gemma3Block
// ============================================================================

/// One transformer block: pre-norm → attention → residual → pre-norm → MLP → residual.
pub struct Gemma3Block {
    pub input_layernorm:          RmsNorm2,
    pub self_attn:                Gemma3Attention,
    pub post_attention_layernorm: RmsNorm2,
    /// Pre-FFN norm (Gemma 3 uses an extra "pre_feedforward_layernorm").
    pub pre_feedforward_layernorm: RmsNorm2,
    /// Post-FFN norm (Gemma 3 uses "post_feedforward_layernorm").
    pub post_feedforward_layernorm: RmsNorm2,
    pub mlp:                      Gemma3Mlp,
}

impl Gemma3Block {
    pub fn new(cfg: &Config4, layer_idx: usize, rng: &mut InitRng) -> Self {
        Gemma3Block {
            input_layernorm:            RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            self_attn:                  Gemma3Attention::new(cfg, layer_idx, rng),
            post_attention_layernorm:   RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            pre_feedforward_layernorm:  RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            post_feedforward_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            mlp:                        Gemma3Mlp::new(cfg, rng),
        }
    }

    /// Forward: x [T, hidden] → [T, hidden].
    ///
    /// Gemma 3 block structure (differs slightly from GPT-OSS):
    ///   1. normed = input_layernorm(x)
    ///   2. attn   = self_attn(normed)
    ///   3. attn   = post_attention_layernorm(attn)
    ///   4. x2     = x + attn
    ///   5. normed2 = pre_feedforward_layernorm(x2)
    ///   6. mlp    = mlp(normed2)
    ///   7. mlp    = post_feedforward_layernorm(mlp)
    ///   8. output = x2 + mlp
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        // Attention sub-block
        let normed = self.input_layernorm.forward_gemma3(x);
        let attn   = self.self_attn.forward(&normed);
        let attn   = self.post_attention_layernorm.forward_gemma3(&attn);
        let x2     = x.add(&attn);

        // MLP sub-block
        let normed2 = self.pre_feedforward_layernorm.forward_gemma3(&x2);
        let mlp_out = self.mlp.forward(&normed2);
        let mlp_out = self.post_feedforward_layernorm.forward_gemma3(&mlp_out);
        x2.add(&mlp_out)
    }
}

impl Module2 for Gemma3Block {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = Vec::new();
        p.push(self.input_layernorm.gamma.clone());
        p.extend(self.self_attn.parameters());
        p.push(self.post_attention_layernorm.gamma.clone());
        p.push(self.pre_feedforward_layernorm.gamma.clone());
        p.push(self.post_feedforward_layernorm.gamma.clone());
        p.extend(self.mlp.parameters());
        p
    }
}

// ============================================================================
// Gemma3Model
// ============================================================================

/// The full Gemma 3 model.
pub struct Gemma3Model {
    pub embed_tokens: TensorNode,     // [vocab_size, hidden_size]
    pub layers:       Vec<Gemma3Block>,
    pub norm:         RmsNorm2,       // final layer norm
    pub lm_head:      Linear2,        // [hidden_size, vocab_size] — tied with embed_tokens
    pub config:       Config4,
}

impl Gemma3Model {
    pub fn new(cfg: Config4, rng: &mut InitRng) -> Self {
        let embed = TensorNode::leaf(Mat::new(
            rng.normal_vec(cfg.vocab_size * cfg.hidden_size, 0.02),
            cfg.vocab_size, cfg.hidden_size,
        ));
        let layers: Vec<Gemma3Block> = (0..cfg.num_hidden_layers)
            .map(|i| Gemma3Block::new(&cfg, i, rng))
            .collect();
        let norm = RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps);
        // lm_head is weight-tied to embed_tokens in Gemma 3: they share the same
        // underlying data.  We represent this by cloning the TensorNode (which
        // shares the Arc-backed storage).
        let b_data = Mat::zeros(1, cfg.vocab_size);
        let lm_head = Linear2 {
            weight: embed.clone(),
            bias: TensorNode::leaf(b_data),
            in_features: cfg.hidden_size,
            out_features: cfg.vocab_size,
            q4_weight: None,
            bf16_weight: None,
        };

        Gemma3Model { embed_tokens: embed, layers, norm, lm_head, config: cfg }
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    /// Text forward pass: token_ids → logits [T, vocab_size].
    pub fn forward(&self, token_ids: &[usize]) -> TensorNode {
        let t = token_ids.len();
        let h = self.config.hidden_size;

        // Token embedding (Gemma 3 multiplies embeddings by sqrt(hidden_size))
        let scale = (h as f32).sqrt();
        let te = self.embed_tokens.data().clone();
        let x_data = Mat::from_fn(t, h, |row, col| te.at(token_ids[row], col) * scale);
        let x = TensorNode::leaf(x_data);

        // Wire backward: scatter gradient back to embed_tokens rows.
        let embed_node = self.embed_tokens.clone();
        let x_c        = x.clone();
        let ids        = token_ids.to_vec();
        x.set_backward(Box::new(move || {
            let dout = x_c.grad().clone(); // [T, h]
            let mut dte = embed_node.grad().clone();
            for (row, &tid) in ids.iter().enumerate() {
                for col in 0..h {
                    *dte.at_mut(tid, col) += dout.at(row, col) * scale;
                }
            }
            embed_node.set_grad(dte);
        }), vec![self.embed_tokens.clone()]);

        let mut x = x;

        for layer in &self.layers {
            x = layer.forward(&x);
        }

        let normed  = self.norm.forward_gemma3(&x);
        self.lm_head.forward(&normed)
    }

    // -------------------------------------------------------------------------
    // Generation
    // -------------------------------------------------------------------------

    /// Generate `max_new` tokens, calling `callback` for each.
    pub fn generate_streaming(
        &self,
        token_ids: &[usize],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        seed: u64,
        mut callback: impl FnMut(usize),
    ) {
        use crate::transformer3::SamplingParams;

        let params = SamplingParams {
            temperature,
            top_k,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed,
            eos_token_id: Some(self.config.eos_token_id),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };

        let mut rng = LcgRng::new(seed);
        let mut seen: Vec<usize> = token_ids.to_vec();

        // Process prompt
        let logits_node = self.forward(token_ids);
        let logits = logits_node.data().clone();
        let t = logits.rows;

        let next = sample_token(&logits, t - 1, &params, &seen, &mut rng);
        callback(next);
        seen.push(next);
        if params.eos_token_id == Some(next) { return; }

        let mut prev = next;
        for _ in 1..max_new {
            let logits_node = self.forward(&[prev]);
            let logits = logits_node.data().clone();
            prev = sample_token(&logits, 0, &params, &seen, &mut rng);
            callback(prev);
            seen.push(prev);
            if params.eos_token_id == Some(prev) { break; }
        }
    }

    // -------------------------------------------------------------------------
    // Weight loading
    // -------------------------------------------------------------------------

    /// Load weights from a directory containing HuggingFace `.safetensors` shards.
    ///
    /// Tensor names follow the standard Gemma 3 / transformers convention:
    /// - `model.embed_tokens.weight`
    /// - `model.layers.{i}.self_attn.q_proj.weight`
    /// - `model.layers.{i}.self_attn.q_norm.weight`
    /// - etc.
    pub fn load_weights_from_dir(&mut self, dir: &str) -> Result<(), String> {
        let entries = std::fs::read_dir(dir)
            .map_err(|e| format!("cannot read dir {}: {}", dir, e))?;

        let mut loaded_shards = 0usize;
        let mut loaded_tensors = 0usize;
        let mut matched = 0usize;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("safetensors") {
                continue;
            }

            let bytes = std::fs::read(&path)
                .map_err(|e| format!("cannot read {:?}: {}", path, e))?;

            let tensors = crate::transformer3::parse_safetensors(&bytes)
                .map_err(|e| format!("parse error in {:?}: {}", path, e))?;

            for t in &tensors {
                loaded_tensors += 1;
                if apply_tensor(self, t) { matched += 1; }
            }
            loaded_shards += 1;
        }

        if loaded_shards == 0 {
            return Err(format!("no .safetensors files found in {}", dir));
        }

        println!("Gemma3: loaded {} tensors ({} matched) from {} shards in {}",
            loaded_tensors, matched, loaded_shards, dir);
        Ok(())
    }

    /// Quantize all linear projection weights to INT4 (halves memory ~4×).
    pub fn quantize_for_inference(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.q_proj.quantize();
            layer.self_attn.k_proj.quantize();
            layer.self_attn.v_proj.quantize();
            layer.self_attn.o_proj.quantize();
            layer.mlp.gate_proj.quantize();
            layer.mlp.up_proj.quantize();
            layer.mlp.down_proj.quantize();
        }
        self.lm_head.quantize();
    }

    /// Quantize all projection weights to INT4 and free the f32 copies.
    ///
    /// Use this instead of `quantize_for_inference()` when you only need
    /// inference (no training).  Frees ~75% of weight RAM: f32 → INT4 is 8×
    /// smaller, and the f32 copy is dropped so peak usage stays low.
    pub fn quantize_inference_free_f32(&mut self) {
        eprintln!("[ Gemma3 ] Quantizing weights to INT4 and freeing f32 copies...");
        let n_layers = self.layers.len();
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.self_attn.q_proj.quantize_and_free_f32();
            layer.self_attn.k_proj.quantize_and_free_f32();
            layer.self_attn.v_proj.quantize_and_free_f32();
            layer.self_attn.o_proj.quantize_and_free_f32();
            layer.mlp.gate_proj.quantize_and_free_f32();
            layer.mlp.up_proj.quantize_and_free_f32();
            layer.mlp.down_proj.quantize_and_free_f32();
            if (i + 1) % 8 == 0 || i + 1 == n_layers {
                eprintln!("[ Gemma3 ] Quantized {}/{} layers", i + 1, n_layers);
            }
        }
        self.lm_head.quantize_and_free_f32();
        eprintln!("[ Gemma3 ] Quantization done.");
    }
}

// ============================================================================
// Module2 + Trainable for Gemma3Model
// ============================================================================

impl Module2 for Gemma3Model {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = vec![self.embed_tokens.clone()];
        for layer in &self.layers {
            p.extend(layer.parameters());
        }
        p.push(self.norm.gamma.clone());
        p.push(self.lm_head.weight.clone());
        p
    }
}

impl Trainable for Gemma3Model {
    fn forward_tokens(&self, token_ids: &[usize]) -> TensorNode {
        self.forward(token_ids)
    }

    fn loss_tokens(&self, token_ids: &[usize], targets: &[usize]) -> TensorNode {
        let logits_node = self.forward(token_ids);
        let logits = logits_node.data().clone();
        let t = logits.rows;
        let v = logits.cols;
        assert_eq!(t, targets.len());

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
            loss_val -= probs.at(r, targets[r]).ln().max(-100.0);
        }
        loss_val /= t as f32;

        let loss = TensorNode::leaf(Mat::new(vec![loss_val], 1, 1));
        let logits_c     = logits_node.clone();
        let probs_stored = probs;
        let targets_v    = targets.to_vec();

        loss.set_backward(Box::new(move || {
            let mut dlogits = logits_c.grad().clone();
            for r in 0..t {
                for c in 0..v {
                    let ind = if c == targets_v[r] { 1.0f32 } else { 0.0 };
                    *dlogits.at_mut(r, c) += (probs_stored.at(r, c) - ind) / t as f32;
                }
            }
            logits_c.set_grad(dlogits);
            logits_c.call_backward_fn();
        }), vec![logits_node]);

        loss
    }
}

// ============================================================================
// Weight name → model field mapping
// ============================================================================

fn apply_tensor(model: &mut Gemma3Model, t: &crate::transformer3::SafeTensor) -> bool {
    apply_tensor_inner(model, t).unwrap_or(false)
}

fn apply_tensor_inner(
    model: &mut Gemma3Model,
    t: &crate::transformer3::SafeTensor,
) -> Option<bool> {
    // Strip optional "language_model." prefix (present in vision-language model checkpoints)
    let name = t.name.strip_prefix("language_model.").unwrap_or(t.name.as_str());

    // Global tensors — embeddings and norms stay as f32 (they are small
    // and used differently from projection weights).
    if name == "model.embed_tokens.weight" {
        return Some(set_node(&model.embed_tokens, &t.data, t.shape[0], t.shape[1]));
    }
    if name == "model.norm.weight" {
        return Some(set_node(&model.norm.gamma, &t.data, 1, t.shape[0]));
    }
    if name == "lm_head.weight" {
        return Some(set_node(&model.lm_head.weight, &t.data, t.shape[0], t.shape[1]));
    }

    // Per-layer tensors: "model.layers.{i}.{...}"
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let layer_idx: usize = rest[..dot].parse().ok()?;
    if layer_idx >= model.layers.len() { return Some(false); }
    let layer_name = &rest[dot + 1..];
    let layer = &mut model.layers[layer_idx];

    match layer_name {
        // Layer-norm weights are small scalars — keep as f32.
        "input_layernorm.weight" =>
            Some(set_node(&layer.input_layernorm.gamma, &t.data, 1, t.shape[0])),
        "post_attention_layernorm.weight" =>
            Some(set_node(&layer.post_attention_layernorm.gamma, &t.data, 1, t.shape[0])),
        "pre_feedforward_layernorm.weight" =>
            Some(set_node(&layer.pre_feedforward_layernorm.gamma, &t.data, 1, t.shape[0])),
        "post_feedforward_layernorm.weight" =>
            Some(set_node(&layer.post_feedforward_layernorm.gamma, &t.data, 1, t.shape[0])),
        "self_attn.q_norm.weight" =>
            Some(set_node(&layer.self_attn.q_norm.gamma, &t.data, 1, t.shape[0])),
        "self_attn.k_norm.weight" =>
            Some(set_node(&layer.self_attn.k_norm.gamma, &t.data, 1, t.shape[0])),

        // Large projection weights: store as BF16 when available (lossless, 2× RAM).
        "self_attn.q_proj.weight" =>
            Some(set_linear(&mut layer.self_attn.q_proj, t, t.shape[0], t.shape[1])),
        "self_attn.k_proj.weight" =>
            Some(set_linear(&mut layer.self_attn.k_proj, t, t.shape[0], t.shape[1])),
        "self_attn.v_proj.weight" =>
            Some(set_linear(&mut layer.self_attn.v_proj, t, t.shape[0], t.shape[1])),
        "self_attn.o_proj.weight" =>
            Some(set_linear(&mut layer.self_attn.o_proj, t, t.shape[0], t.shape[1])),
        "mlp.gate_proj.weight" =>
            Some(set_linear(&mut layer.mlp.gate_proj, t, t.shape[0], t.shape[1])),
        "mlp.up_proj.weight" =>
            Some(set_linear(&mut layer.mlp.up_proj, t, t.shape[0], t.shape[1])),
        "mlp.down_proj.weight" =>
            Some(set_linear(&mut layer.mlp.down_proj, t, t.shape[0], t.shape[1])),
        _ => Some(false),
    }
}

/// Set a TensorNode's f32 data directly (used for small tensors: norms, embeddings).
fn set_node(node: &TensorNode, data: &[f32], rows: usize, cols: usize) -> bool {
    if data.len() != rows * cols { return false; }
    node.set_data(Mat::new(data.to_vec(), rows, cols));
    true
}

/// Set a Linear2 weight, preferring BF16 storage when the tensor was BF16 on disk.
/// Falls back to f32 if no BF16 data is available (e.g. F32 or F16 safetensors).
fn set_linear(
    linear: &mut crate::nn2::Linear2,
    t: &crate::transformer3::SafeTensor,
    rows: usize,
    cols: usize,
) -> bool {
    if let Some(ref bits) = t.bf16_data {
        if bits.len() == rows * cols {
            linear.load_bf16(bits.clone(), rows, cols);
            return true;
        }
    }
    // Fallback: store as f32
    if t.data.len() != rows * cols { return false; }
    linear.weight.set_data(Mat::new(t.data.clone(), rows, cols));
    true
}

// ============================================================================
// KV cache — per-layer storage for autoregressive decoding
// ============================================================================

/// Per-layer K/V cache for Gemma 3 autoregressive decoding.
///
/// Pre-allocated to `max_position_embeddings` rows.  During generation
/// rows are filled one step at a time; `seq_len` tracks how many are valid.
pub struct Gemma3LayerKvCache {
    /// [max_seq_len, n_kv_heads * head_dim]
    pub k: Mat,
    /// [max_seq_len, n_kv_heads * head_dim]
    pub v: Mat,
    /// Number of valid (filled) rows.
    pub seq_len: usize,
}

impl Gemma3LayerKvCache {
    pub fn new(n_kv_heads: usize, head_dim: usize, max_seq_len: usize) -> Self {
        Gemma3LayerKvCache {
            k: Mat::zeros(max_seq_len, n_kv_heads * head_dim),
            v: Mat::zeros(max_seq_len, n_kv_heads * head_dim),
            seq_len: 0,
        }
    }

    /// Append `new_k` / `new_v` rows (both [n_new, n_kv_heads * head_dim]).
    pub fn append(&mut self, new_k: &Mat, new_v: &Mat) {
        let n_new = new_k.rows;
        let d     = new_k.cols;
        for r in 0..n_new {
            for c in 0..d {
                *self.k.at_mut(self.seq_len + r, c) = new_k.at(r, c);
                *self.v.at_mut(self.seq_len + r, c) = new_v.at(r, c);
            }
        }
        self.seq_len += n_new;
    }

    /// Return filled portion of K: [seq_len, cols].
    pub fn k_filled(&self) -> Mat {
        Mat::from_fn(self.seq_len, self.k.cols, |r, c| self.k.at(r, c))
    }

    /// Return filled portion of V: [seq_len, cols].
    pub fn v_filled(&self) -> Mat {
        Mat::from_fn(self.seq_len, self.v.cols, |r, c| self.v.at(r, c))
    }

    /// Return last `window` rows of K (or all rows when seq_len < window).
    pub fn k_last(&self, window: usize) -> Mat {
        let start = self.seq_len.saturating_sub(window);
        let rows  = self.seq_len - start;
        Mat::from_fn(rows, self.k.cols, |r, c| self.k.at(start + r, c))
    }

    /// Return last `window` rows of V (or all rows when seq_len < window).
    pub fn v_last(&self, window: usize) -> Mat {
        let start = self.seq_len.saturating_sub(window);
        let rows  = self.seq_len - start;
        Mat::from_fn(rows, self.v.cols, |r, c| self.v.at(start + r, c))
    }
}

/// Full KV cache for all Gemma 3 layers.
pub struct Gemma3KvCache {
    pub layers: Vec<RefCell<Gemma3LayerKvCache>>,
}

impl Gemma3KvCache {
    pub fn new(config: &Config4) -> Self {
        let nkv = config.num_key_value_heads;
        let d   = config.head_dim;
        // Cap at 2048 tokens for typical generation — the full 32768 would
        // pre-allocate ~7 GB for Gemma3-4b before any token is processed.
        let max = config.max_position_embeddings.min(2048);
        let layers = (0..config.num_hidden_layers)
            .map(|_| RefCell::new(Gemma3LayerKvCache::new(nkv, d, max)))
            .collect();
        Gemma3KvCache { layers }
    }

    /// Reset all layers (start a new sequence).
    pub fn clear(&self) {
        for layer in &self.layers {
            layer.borrow_mut().seq_len = 0;
        }
    }
}

// ============================================================================
// Cached forward passes
// ============================================================================

impl Gemma3Attention {
    /// Cached forward: `x` is [n_new, hidden] — typically 1 token during decode.
    ///
    /// Steps:
    ///   1. Project → Q [n_new, nq*d], K [n_new, nkv*d], V [n_new, nkv*d]
    ///   2. Per-head RMSNorm on Q and K (before RoPE, Gemma 3 specific)
    ///   3. RoPE at absolute positions [cache.seq_len .. cache.seq_len + n_new)
    ///   4. Append K, V to cache
    ///   5. Select context: last `window` rows (local) or full cache (global)
    ///   6. GQA attention (no causal mask — cache only holds past tokens)
    ///   7. Output projection
    pub fn forward_cached(
        &self,
        x: &TensorNode,
        cache: &mut Gemma3LayerKvCache,
    ) -> TensorNode {
        let n_new      = x.data().rows;
        let d          = self.head_dim;
        let nq         = self.n_q_heads;
        let nkv        = self.n_kv_heads;
        let seq_offset = cache.seq_len;

        // 1. Projections
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 2. Per-head RMSNorm
        let q = apply_per_head_norm(&q, &self.q_norm, n_new, nq,  d);
        let k = apply_per_head_norm(&k, &self.k_norm, n_new, nkv, d);

        // 3. RoPE at absolute positions
        let q = apply_rope_at_offset(&q, nq,  n_new, d, self.rope_theta, seq_offset);
        let k = apply_rope_at_offset(&k, nkv, n_new, d, self.rope_theta, seq_offset);

        // 4. Append to cache
        cache.append(&k.data().clone(), &v.data().clone());

        // 5. Select context window
        let (k_ctx, v_ctx) = match self.sliding_window {
            Some(w) => (cache.k_last(w), cache.v_last(w)),
            None    => (cache.k_filled(), cache.v_filled()),
        };

        // 6. GQA attention
        let attn_out = gqa_attention_cached(
            &q.data(), &k_ctx, &v_ctx, nq, nkv, d, self.attn_scale,
        );

        // 7. Output projection
        self.o_proj.forward(&TensorNode::leaf(attn_out))
    }
}

impl Gemma3Block {
    /// Cached forward: `x` is [n_new, hidden].
    ///
    /// Mirrors the 4-norm Gemma 3 block structure exactly, using the KV cache
    /// for attention.  The MLP always processes n_new tokens (no FFN cache).
    pub fn forward_cached(
        &self,
        x: &TensorNode,
        cache: &mut Gemma3LayerKvCache,
    ) -> TensorNode {
        let normed  = self.input_layernorm.forward_gemma3(x);
        let attn    = self.self_attn.forward_cached(&normed, cache);
        let attn    = self.post_attention_layernorm.forward_gemma3(&attn);
        let x2      = x.add(&attn);

        let normed2  = self.pre_feedforward_layernorm.forward_gemma3(&x2);
        let mlp_out  = self.mlp.forward(&normed2);
        let mlp_out  = self.post_feedforward_layernorm.forward_gemma3(&mlp_out);
        x2.add(&mlp_out)
    }
}

impl Gemma3Model {
    /// Generate `max_new` tokens using a KV cache.
    ///
    /// Each decode step runs only 1 token through the model, reducing
    /// generation from O(N²·T) to O(N·T) — a factor of `max_new` speedup.
    ///
    /// Prefill: run the full prompt through `forward_cached` once per layer,
    /// filling the cache.  Sample the first new token from the last logit row.
    ///
    /// Decode: run 1 token per step.  K/V is appended to the cache; local
    /// layers attend to the last `window` entries, global layers to all.
    pub fn generate_cached_streaming(
        &self,
        token_ids: &[usize],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        seed: u64,
        mut callback: impl FnMut(usize),
    ) {
        use crate::transformer3::SamplingParams;

        let params = SamplingParams {
            temperature,
            top_k,
            top_p: 1.0,
            repetition_penalty: 1.0,
            seed,
            eos_token_id: Some(self.config.eos_token_id),
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        };

        // Gemma 3 uses two EOS token ids: 1 (<eos>) and 106 (<end_of_turn>).
        let is_eos = |tok: usize| tok == 1 || tok == 106;

        let cache  = Gemma3KvCache::new(&self.config);
        let h      = self.config.hidden_size;
        let scale  = (h as f32).sqrt();
        let te     = self.embed_tokens.data().clone();
        let mut rng  = LcgRng::new(seed);
        let mut seen: Vec<usize> = token_ids.to_vec();

        // ----- Prefill -----
        let t_prompt = token_ids.len();
        let x_data = Mat::from_fn(t_prompt, h, |row, col| {
            te.at(token_ids[row], col) * scale
        });
        let mut x = TensorNode::leaf(x_data);
        {
            let xd = x.data();
            let vals: Vec<f32> = (0..5.min(xd.cols)).map(|c| xd.at(0, c)).collect();
            eprintln!("[DBG] embed[tok0, 0..5]: {:?}", vals);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
            if i == 0 || i == 5 || i == 11 || i == 17 || i == 23 || i == 33 {
                let xd = x.data();
                let row = t_prompt - 1;
                let rms: f32 = ((0..xd.cols).map(|c| xd.at(row, c).powi(2)).sum::<f32>() / xd.cols as f32).sqrt();
                eprintln!("[DBG] after_layer{}[last_tok] RMS: {:.4}", i, rms);
            }
        }
        let normed_final = self.norm.forward_gemma3(&x);
        {
            let nd = normed_final.data();
            let row = t_prompt - 1;
            let norm_vals: Vec<f32> = (0..5.min(nd.cols)).map(|c| nd.at(row, c)).collect();
            eprintln!("[DBG] final normed[last_tok, 0..5]: {:?}", norm_vals);
            let rms: f32 = (0..nd.cols).map(|c| nd.at(row, c).powi(2)).sum::<f32>() / nd.cols as f32;
            eprintln!("[DBG] final normed RMS: {:.4}", rms.sqrt());
        }
        let logits_node = self.lm_head.forward(&normed_final);
        {
            let ld = logits_node.data();
            let row = t_prompt - 1;
            let v = ld.cols;
            let mut top: Vec<(usize, f32)> = (0..v).map(|c| (c, ld.at(row, c))).collect();
            top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            eprintln!("[DBG] prefill top-5 logits: {:?}", &top[..5.min(top.len())]);
            // Find ranks of specific tokens
            for (tok_name, tok_id) in [("▁The(669)", 669usize), ("▁Rome(13706)", 13706), ("▁Italy(11702)", 11702), ("\n(107)", 107)] {
                let rank = top.iter().position(|(id, _)| *id == tok_id).unwrap_or(999999);
                let score = if tok_id < v { ld.at(row, tok_id) } else { f32::NAN };
                eprintln!("[DBG] rank of {} = {} (score={:.3})", tok_name, rank, score);
            }
        }
        let first = sample_token(&logits_node.data(), t_prompt - 1, &params, &seen, &mut rng);
        callback(first);
        seen.push(first);
        if is_eos(first) { return; }

        // ----- Decode loop -----
        let mut prev = first;
        for _ in 1..max_new {
            let x_data = Mat::from_fn(1, h, |_, col| te.at(prev, col) * scale);
            let mut x = TensorNode::leaf(x_data);
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
            }
            let logits_node = self.lm_head.forward(&self.norm.forward_gemma3(&x));
            prev = sample_token(&logits_node.data(), 0, &params, &seen, &mut rng);
            callback(prev);
            seen.push(prev);
            if is_eos(prev) { break; }
        }
    }
}

// ============================================================================
// Attention helpers
// ============================================================================

/// Apply per-head RMSNorm to a concatenated head tensor.
///
/// Input:  x [T, n_heads * head_dim]
/// Output: [T, n_heads * head_dim]
///
/// Each head's d_head-dimensional slice is normalized independently.
fn apply_per_head_norm(
    x: &TensorNode,
    norm: &RmsNorm2,
    t: usize,
    n_heads: usize,
    head_dim: usize,
) -> TensorNode {
    let x_data = x.data().clone();
    let mut out = Mat::zeros(t, n_heads * head_dim);

    for h in 0..n_heads {
        // Extract head slice [T, head_dim]
        let head_data = Mat::from_fn(t, head_dim, |row, col| {
            x_data.at(row, h * head_dim + col)
        });
        let head_node = TensorNode::leaf(head_data);
        // Apply RMSNorm (Gemma3 uses (1 + gamma) scaling)
        let normed = norm.forward_gemma3(&head_node);
        let normed_data = normed.data().clone();
        // Write back into out
        for row in 0..t {
            for col in 0..head_dim {
                *out.at_mut(row, h * head_dim + col) = normed_data.at(row, col);
            }
        }
    }

    // Build TensorNode with backward that scatters gradients back.
    // For inference this is a leaf; for training we wire backward below.
    let result = TensorNode::leaf(out);

    // Backward: grad flows through the RMSNorm for each head.
    // We store the per-head forward outputs so we can call their backward.
    // This is a simplified implementation — gradients for the norm gamma
    // are accumulated across heads.
    let x_c      = x.clone();
    let result_c = result.clone();
    let norm_gamma_c = norm.gamma.clone();
    let eps = norm.eps;

    result.set_backward(Box::new(move || {
        let dout = result_c.grad().clone();  // [T, n_heads * head_dim]
        let x_data = x_c.data().clone();
        let mut dx = x_c.grad().clone();

        for h in 0..n_heads {
            // Recompute RMSNorm stats for this head (needed for backward).
            for row in 0..t {
                // Compute RMS
                let mut sq_sum = 0.0f32;
                for col in 0..head_dim {
                    let v = x_data.at(row, h * head_dim + col);
                    sq_sum += v * v;
                }
                let rms = (sq_sum / head_dim as f32 + eps).sqrt();
                let inv_rms = 1.0 / rms;

                // dL/dx uses (1 + gamma) as the effective scale (Gemma3 RMSNorm)
                let mut dot_dy_gamma_x = 0.0f32;
                for col in 0..head_dim {
                    dot_dy_gamma_x += dout.at(row, h * head_dim + col)
                        * (1.0 + norm_gamma_c.data().at(0, col))
                        * x_data.at(row, h * head_dim + col);
                }

                for col in 0..head_dim {
                    let g    = 1.0 + norm_gamma_c.data().at(0, col);
                    let xi   = x_data.at(row, h * head_dim + col);
                    let dy   = dout.at(row, h * head_dim + col);
                    let term1 = dy * g * inv_rms;
                    let term2 = xi * inv_rms * inv_rms * inv_rms * dot_dy_gamma_x / head_dim as f32;
                    *dx.at_mut(row, h * head_dim + col) += term1 - term2;
                }
            }
        }
        x_c.set_grad(dx);
        x_c.call_backward_fn();
    }), vec![x.clone()]);

    result
}

/// Apply RoPE to all heads in a concatenated [T, n_heads * head_dim] tensor.
fn apply_rope_to_all_heads(
    x: &TensorNode,
    n_heads: usize,
    t: usize,
    head_dim: usize,
    theta: f32,
) -> TensorNode {
    let x_data = x.data().clone();
    let mut out = x_data.clone();

    for h in 0..n_heads {
        for pos in 0..t {
            // Rotate pairs (2i, 2i+1) within this head
            let pairs = head_dim / 2;
            for i in 0..pairs {
                let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let c0 = h * head_dim + 2 * i;
                let c1 = h * head_dim + 2 * i + 1;
                let x0 = x_data.at(pos, c0);
                let x1 = x_data.at(pos, c1);
                *out.at_mut(pos, c0) = x0 * cos_a - x1 * sin_a;
                *out.at_mut(pos, c1) = x1 * cos_a + x0 * sin_a;
            }
        }
    }

    // RoPE backward: the rotation matrix is orthogonal, so the backward is
    // just the transpose rotation (negate the sin terms).
    let result = TensorNode::leaf(out);
    let x_c      = x.clone();
    let result_c = result.clone();

    result.set_backward(Box::new(move || {
        let dout   = result_c.grad().clone();
        let mut dx = x_c.grad().clone();
        for h in 0..n_heads {
            for pos in 0..t {
                let pairs = head_dim / 2;
                for i in 0..pairs {
                    let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let c0 = h * head_dim + 2 * i;
                    let c1 = h * head_dim + 2 * i + 1;
                    // Inverse rotation: [cos, sin; -sin, cos] (transpose of forward)
                    let dy0 = dout.at(pos, c0);
                    let dy1 = dout.at(pos, c1);
                    *dx.at_mut(pos, c0) +=  dy0 * cos_a + dy1 * sin_a;
                    *dx.at_mut(pos, c1) += -dy0 * sin_a + dy1 * cos_a;
                }
            }
        }
        x_c.set_grad(dx);
        x_c.call_backward_fn();
    }), vec![x.clone()]);

    result
}

/// Full causal GQA attention (global layers).
///
/// q [T, nq*d], k [T, nkv*d], v [T, nkv*d] → [T, nq*d]
/// scale replaces 1/sqrt(d_head) with 1/sqrt(query_pre_attn_scalar).
fn gqa_attention_full(
    q: &TensorNode,
    k: &TensorNode,
    v: &TensorNode,
    n_q_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    scale: f32,
) -> TensorNode {
    // Delegate to the existing batched_gqa_attention but apply our custom scale.
    // We achieve this by pre-scaling Q: Q' = Q * scale, then calling with scale=1/sqrt(d).
    // Actually simpler: just call gqa_attention and correct for the different scale.
    // The existing gqa_attention uses scale = 1/sqrt(d_head) internally.
    // We need scale = 1/sqrt(query_pre_attn_scalar).
    // Ratio = sqrt(d_head) / sqrt(query_pre_attn_scalar).
    let default_scale = 1.0 / (d_head as f32).sqrt();
    let ratio = scale / default_scale;  // multiply Q by this to get the right scale

    let q_scaled = scale_tensor(q, ratio);
    TensorNode::batched_gqa_attention(&q_scaled, k, v, n_q_heads, n_kv_heads, d_head)
}

/// Sliding-window causal GQA attention (local layers).
fn gqa_attention_windowed(
    q: &TensorNode,
    k: &TensorNode,
    v: &TensorNode,
    n_q_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    window: usize,
    scale: f32,
) -> TensorNode {
    let t = q.data().rows;
    let group = n_q_heads / n_kv_heads;
    let q_data = q.data().clone();
    let k_data = k.data().clone();
    let v_data = v.data().clone();

    let mut out_data = Mat::zeros(t, n_q_heads * d_head);

    for h in 0..n_q_heads {
        let kv_head = h / group;
        let q_mat = Mat::from_fn(t, d_head, |r, c| q_data.at(r, h * d_head + c));
        let k_mat = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kv_head * d_head + c));
        let v_mat = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kv_head * d_head + c));

        // Scores [T, T] with windowed causal mask, scaled by attn_scale
        let kt = k_mat.transpose();
        let scores = q_mat.matmul(&kt).scale(scale); // [T, T]

        // Apply windowed causal mask
        let mut masked = Mat::from_fn(t, t, |r, c| {
            let causal_ok = c <= r;
            let window_ok = r < window || c >= r + 1 - window;
            if causal_ok && window_ok { scores.at(r, c) } else { f32::NEG_INFINITY }
        });

        // Softmax per row
        for r in 0..t {
            let row_max = (0..t).map(|c| masked.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..t {
                let v = if masked.at(r, c) == f32::NEG_INFINITY {
                    0.0
                } else {
                    (masked.at(r, c) - row_max).exp()
                };
                *masked.at_mut(r, c) = v;
                sum_exp += v;
            }
            if sum_exp > 0.0 {
                for c in 0..t { *masked.at_mut(r, c) /= sum_exp; }
            }
        }

        // Output = weights @ V
        let head_out = masked.matmul(&v_mat); // [T, d_head]
        for r in 0..t {
            for c in 0..d_head {
                *out_data.at_mut(r, h * d_head + c) = head_out.at(r, c);
            }
        }
    }

    // Inference-only leaf (no backward for windowed attention)
    TensorNode::leaf(out_data)
}

/// Apply RoPE to [n_new, n_heads * head_dim] with positions starting at `offset`.
///
/// Used in the KV-cached decode path where token row `r` is at absolute
/// sequence position `offset + r`.  This is the inference-only counterpart
/// to `apply_rope_to_all_heads` (which always starts at position 0).
fn apply_rope_at_offset(
    x: &TensorNode,
    n_heads: usize,
    n_new: usize,
    head_dim: usize,
    theta: f32,
    offset: usize,
) -> TensorNode {
    let x_data = x.data().clone();
    let mut out = x_data.clone();
    for h in 0..n_heads {
        for row in 0..n_new {
            let pos   = offset + row;
            let pairs = head_dim / 2;
            for i in 0..pairs {
                let angle = pos as f32 / theta.powf(2.0 * i as f32 / head_dim as f32);
                let cos_a = angle.cos();
                let sin_a = angle.sin();
                let c0 = h * head_dim + 2 * i;
                let c1 = h * head_dim + 2 * i + 1;
                let x0 = x_data.at(row, c0);
                let x1 = x_data.at(row, c1);
                *out.at_mut(row, c0) = x0 * cos_a - x1 * sin_a;
                *out.at_mut(row, c1) = x1 * cos_a + x0 * sin_a;
            }
        }
    }
    // Inference-only: plain leaf, no backward.
    TensorNode::leaf(out)
}

/// GQA attention for the cached decode path (no causal mask needed).
///
/// q [n_new, nq*d]  vs  k_ctx [T', nkv*d], v_ctx [T', nkv*d]
/// → output Mat [n_new, nq*d]
///
/// The cache already contains only past tokens, so no causal masking is
/// required.  Uses the caller-supplied `scale` (Gemma 3's
/// `1/sqrt(query_pre_attn_scalar)`, not the default `1/sqrt(d_head)`).
fn gqa_attention_cached(
    q_data: &Mat,
    k_ctx:  &Mat,
    v_ctx:  &Mat,
    n_q_heads:  usize,
    n_kv_heads: usize,
    d_head:     usize,
    scale:      f32,
) -> Mat {
    let t_q  = q_data.rows;
    let t_kv = k_ctx.rows;
    let group = n_q_heads / n_kv_heads;
    let mut out = Mat::zeros(t_q, n_q_heads * d_head);

    for qh in 0..n_q_heads {
        let kvh = qh / group;

        let q_h = Mat::from_fn(t_q,  d_head, |r, c| q_data.at(r, qh  * d_head + c));
        let k_h = Mat::from_fn(t_kv, d_head, |r, c| k_ctx.at(r, kvh * d_head + c));
        let v_h = Mat::from_fn(t_kv, d_head, |r, c| v_ctx.at(r, kvh * d_head + c));

        // scores [t_q, t_kv] scaled
        let raw_scores = q_h.matmul(&k_h.transpose()).scale(scale);

        // Causal mask: query at absolute position (t_kv - t_q + r) may only
        // attend to keys at positions 0..=(t_kv - t_q + r).
        // During decode t_q==1 so this is a no-op; during prefill it masks
        // the upper triangle so tokens cannot attend to future positions.
        let causal_offset = t_kv - t_q; // absolute position of query row 0
        let mut w = Mat::zeros(t_q, t_kv);
        for r in 0..t_q {
            let max_kv = causal_offset + r; // last valid key index for this query
            let row_max = (0..=max_kv).map(|c| raw_scores.at(r, c))
                .fold(f32::NEG_INFINITY, f32::max);
            let mut row_sum = 0.0f32;
            for c in 0..t_kv {
                let e = if c <= max_kv {
                    (raw_scores.at(r, c) - row_max).exp()
                } else {
                    0.0
                };
                *w.at_mut(r, c) = e;
                row_sum += e;
            }
            if row_sum > 0.0 {
                for c in 0..t_kv { *w.at_mut(r, c) /= row_sum; }
            }
        }

        // Output [t_q, d_head]
        let out_h = w.matmul(&v_h);
        for r in 0..t_q {
            for c in 0..d_head {
                *out.at_mut(r, qh * d_head + c) = out_h.at(r, c);
            }
        }
    }
    out
}

/// Multiply a TensorNode's data by a scalar (with backward).
fn scale_tensor(x: &TensorNode, factor: f32) -> TensorNode {
    if (factor - 1.0).abs() < 1e-9 { return x.clone(); }
    let scaled = x.data().scale(factor);
    let result = TensorNode::leaf(scaled);
    let x_c = x.clone();
    let result_c = result.clone();
    result.set_backward(Box::new(move || {
        let dout = result_c.grad().clone();
        let new_grad = x_c.grad().clone().add(&dout.scale(factor));
        x_c.set_grad(new_grad);
        x_c.call_backward_fn();
    }), vec![x.clone()]);
    result
}

// ============================================================================
// Minimal LCG RNG (copied from transformer3 pattern)
// ============================================================================

struct LcgRng { state: u64 }
impl LcgRng {
    fn new(seed: u64) -> Self { LcgRng { state: seed.wrapping_add(1) } }
    fn next_f32(&mut self) -> f32 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.state >> 32) as f32) / (u32::MAX as f32)
    }
}

fn sample_token(
    logits: &Mat,
    row: usize,
    params: &crate::transformer3::SamplingParams,
    seen: &[usize],
    rng: &mut LcgRng,
) -> usize {
    let v = logits.cols;
    let mut scores: Vec<f32> = (0..v).map(|c| logits.at(row, c)).collect();

    // Repetition penalty
    if params.repetition_penalty != 1.0 {
        for &tok in seen {
            if tok < v {
                if scores[tok] >= 0.0 {
                    scores[tok] /= params.repetition_penalty;
                } else {
                    scores[tok] *= params.repetition_penalty;
                }
            }
        }
    }

    // Temperature
    if params.temperature > 0.0 && params.temperature != 1.0 {
        for s in &mut scores { *s /= params.temperature; }
    }

    // Softmax
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs { *p /= sum; }

    // Top-k
    if params.top_k > 0 && params.top_k < v {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for i in params.top_k..v { probs[indexed[i].0] = 0.0; }
        let sum: f32 = probs.iter().sum();
        for p in &mut probs { *p /= sum; }
    }

    // Greedy
    if params.temperature == 0.0 {
        return probs.iter().cloned().enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
    }

    // Sample
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative { return i; }
    }
    v - 1
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> Config4 {
        Config4 {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 64,
            head_dim: 16,
            sliding_window: Some(8),
            rope_theta_local:  10_000.0,
            rope_theta_global: 1_000_000.0,
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 16.0,  // matches head_dim for this tiny cfg
            eos_token_id: 1,
            max_position_embeddings: 128,
        }
    }

    #[test]
    fn test_config4_global_layer_pattern() {
        let cfg = tiny_cfg();
        // Pattern: 5 local, 1 global (every 6th starting at index 5)
        for i in 0..18 {
            let expected_global = i % 6 == 5;
            assert_eq!(cfg.is_global_layer(i), expected_global,
                "layer {}: is_global should be {}", i, expected_global);
        }
    }

    #[test]
    fn test_gemma3_1b_config_values() {
        let cfg = Config4::gemma3_1b();
        assert_eq!(cfg.vocab_size, 262144);
        assert_eq!(cfg.hidden_size, 1152);
        assert_eq!(cfg.num_hidden_layers, 26);
        assert_eq!(cfg.num_attention_heads, 4);
        assert_eq!(cfg.num_key_value_heads, 1);
        assert_eq!(cfg.intermediate_size, 6912);
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn test_gemma3_4b_config_values() {
        let cfg = Config4::gemma3_4b();
        assert_eq!(cfg.vocab_size, 262208);
        assert_eq!(cfg.hidden_size, 2560);
        assert_eq!(cfg.num_hidden_layers, 34);
        assert_eq!(cfg.num_attention_heads, 8);
        assert_eq!(cfg.num_key_value_heads, 4);
        assert_eq!(cfg.head_dim, 256);
    }

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(42);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let token_ids = vec![0usize, 5, 10, 2];
        let logits = model.forward(&token_ids);
        let d = logits.data();
        assert_eq!(d.rows, 4, "logits rows should equal T=4");
        assert_eq!(d.cols, cfg.vocab_size, "logits cols should equal vocab_size");
    }

    #[test]
    fn test_model_forward_finite() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(7);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let token_ids = vec![1usize, 3, 5];
        let logits = model.forward(&token_ids);
        let d = logits.data();
        for r in 0..d.rows {
            for c in 0..d.cols {
                assert!(d.at(r, c).is_finite(),
                    "logits[{},{}] = {} is not finite", r, c, d.at(r, c));
            }
        }
    }

    #[test]
    fn test_loss_tokens_finite_and_positive() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(3);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let token_ids = vec![1usize, 3, 5, 2];
        let targets   = vec![3usize, 5, 2, 0];
        for p in model.parameters() { p.zero_grad(); }
        let loss = model.loss_tokens(&token_ids, &targets);
        let v = loss.data().at(0, 0);
        assert!(v.is_finite() && v > 0.0, "loss = {}", v);
    }

    #[test]
    fn test_loss_backward_populates_grads() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(11);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let token_ids = vec![0usize, 1, 2];
        let targets   = vec![1usize, 2, 3];
        for p in model.parameters() { p.zero_grad(); }
        let loss = model.loss_tokens(&token_ids, &targets);
        loss.backward();
        // embed_tokens should have non-zero gradient on the rows we looked up
        let eg = model.embed_tokens.grad();
        let any_nonzero = (0..eg.rows).any(|r| (0..eg.cols).any(|c| eg.at(r, c) != 0.0));
        assert!(any_nonzero, "embed_tokens.grad should be non-zero after backward");
    }

    #[test]
    fn test_attention_local_vs_global_different() {
        // Local (layer 0) and global (layer 5) attention should behave differently
        // on the same input because local uses a window mask.
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(5);
        let attn_local  = Gemma3Attention::new(&cfg, 0, &mut rng); // local
        let attn_global = Gemma3Attention::new(&cfg, 5, &mut rng); // global
        assert!(attn_local.sliding_window.is_some(),  "layer 0 should be local");
        assert!(attn_global.sliding_window.is_none(), "layer 5 should be global");
    }

    #[test]
    fn test_rope_per_head_norm_shape_preserved() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(99);
        let attn = Gemma3Attention::new(&cfg, 0, &mut rng);
        let t = 6;
        let x = TensorNode::leaf(Mat::from_fn(t, cfg.hidden_size, |r, c| {
            ((r * cfg.hidden_size + c) as f32) * 0.01
        }));
        let out = attn.forward(&x);
        let d = out.data();
        assert_eq!(d.rows, t);
        assert_eq!(d.cols, cfg.hidden_size);
    }

    #[test]
    fn test_mlp_forward_shape() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(17);
        let mlp = Gemma3Mlp::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(4, cfg.hidden_size, |r, c| {
            ((r * cfg.hidden_size + c) as f32) * 0.01 - 0.5
        }));
        let out = mlp.forward(&x);
        let d = out.data();
        assert_eq!(d.rows, 4);
        assert_eq!(d.cols, cfg.hidden_size);
    }

    #[test]
    fn test_block_output_shape() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(22);
        let block = Gemma3Block::new(&cfg, 0, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(5, cfg.hidden_size, |r, c| {
            ((r + c) as f32) * 0.01
        }));
        let out = block.forward(&x);
        let d = out.data();
        assert_eq!(d.rows, 5);
        assert_eq!(d.cols, cfg.hidden_size);
    }

    #[test]
    fn test_model_parameter_count() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(0);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let params = model.parameters();
        assert!(!params.is_empty(), "model should have parameters");
        // Rough sanity check: at minimum embed + lm_head + norms + attn + mlp per layer
        assert!(params.len() > cfg.num_hidden_layers * 5,
            "expected many parameter tensors, got {}", params.len());
    }

    #[test]
    fn test_generate_streaming_produces_tokens() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(42);
        let model = Gemma3Model::new(cfg, &mut rng);
        let mut generated = Vec::new();
        model.generate_streaming(&[0usize, 1, 2], 5, 1.0, 0, 42, |tok| {
            generated.push(tok);
        });
        assert!(!generated.is_empty(), "should generate at least one token");
        assert!(generated.len() <= 5);
    }

    // -------------------------------------------------------------------------
    // KV cache tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_layer_kv_cache_append_and_len() {
        let mut cache = Gemma3LayerKvCache::new(1, 16, 64);
        assert_eq!(cache.seq_len, 0);
        cache.append(&Mat::zeros(3, 16), &Mat::zeros(3, 16));
        assert_eq!(cache.seq_len, 3);
        assert_eq!(cache.k_filled().rows, 3);
        cache.append(&Mat::zeros(1, 16), &Mat::zeros(1, 16));
        assert_eq!(cache.seq_len, 4);
    }

    #[test]
    fn test_layer_kv_cache_k_last_window() {
        let d = 2;
        let mut cache = Gemma3LayerKvCache::new(1, d, 32);
        for i in 0..10usize {
            let row = Mat::from_fn(1, d, |_, c| (i * d + c) as f32);
            cache.append(&row, &row);
        }
        assert_eq!(cache.seq_len, 10);
        let k_win = cache.k_last(4);
        assert_eq!(k_win.rows, 4);
        // rows 6..10: first row of window is row 6, values 12, 13
        assert!((k_win.at(0, 0) - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_gemma3_kv_cache_new_and_clear() {
        let cfg = tiny_cfg();
        let cache = Gemma3KvCache::new(&cfg);
        assert_eq!(cache.layers.len(), cfg.num_hidden_layers);
        cache.layers[0].borrow_mut().seq_len = 42;
        cache.clear();
        for layer in &cache.layers {
            assert_eq!(layer.borrow().seq_len, 0);
        }
    }

    #[test]
    fn test_attention_forward_cached_shape_and_cache_grows() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(55);
        let attn = Gemma3Attention::new(&cfg, 0, &mut rng);
        let mut cache = Gemma3LayerKvCache::new(
            cfg.num_key_value_heads, cfg.head_dim, cfg.max_position_embeddings);
        let x = TensorNode::leaf(Mat::zeros(1, cfg.hidden_size));
        let out = attn.forward_cached(&x, &mut cache);
        assert_eq!(out.data().rows, 1);
        assert_eq!(out.data().cols, cfg.hidden_size);
        assert_eq!(cache.seq_len, 1);
        // Second call: cache grows to 2
        let out2 = attn.forward_cached(&x, &mut cache);
        assert_eq!(out2.data().rows, 1);
        assert_eq!(cache.seq_len, 2);
    }

    #[test]
    fn test_generate_cached_streaming_produces_tokens() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(7);
        let model = Gemma3Model::new(cfg, &mut rng);
        let mut generated = Vec::new();
        model.generate_cached_streaming(&[0usize, 1, 2], 5, 1.0, 0, 42, |tok| {
            generated.push(tok);
        });
        assert!(!generated.is_empty());
        assert!(generated.len() <= 5);
    }

    #[test]
    fn test_cached_first_token_matches_uncached() {
        // With temperature=0 (greedy), the first generated token must be
        // identical whether we use the cached or the non-cached path.
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(42);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let prompt = vec![0usize, 1, 2, 3];

        // Non-cached: full forward, greedy argmax on last row
        let logits = model.forward(&prompt);
        let ldata  = logits.data().clone();
        let t      = ldata.rows;
        let v      = ldata.cols;
        let uncached = (0..v)
            .max_by(|&a, &b| ldata.at(t-1, a).partial_cmp(&ldata.at(t-1, b)).unwrap())
            .unwrap();

        // Cached: greedy (temperature=0)
        let mut cached_tok = usize::MAX;
        model.generate_cached_streaming(&prompt, 1, 0.0, 0, 0, |tok| { cached_tok = tok; });

        assert_eq!(uncached, cached_tok,
            "cached first token {} must match non-cached {}", cached_tok, uncached);
    }

    #[test]
    fn test_cached_output_finite_after_long_prefill() {
        // Prefill with more tokens than the sliding window to exercise the
        // windowed attention path in local layers.
        let cfg = tiny_cfg(); // window=8
        let mut rng = InitRng::new(11);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let prompt: Vec<usize> = (0..20).map(|i| i % cfg.vocab_size).collect();
        let mut toks = Vec::new();
        model.generate_cached_streaming(&prompt, 1, 0.0, 0, 0, |t| toks.push(t));
        assert_eq!(toks.len(), 1);
        assert!(toks[0] < cfg.vocab_size);
    }
}
