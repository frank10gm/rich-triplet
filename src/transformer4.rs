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
use crate::autograd2::{Mat, TensorNode};
use crate::nn::InitRng;
use crate::nn2::{Linear2, Module2, RmsNorm2, SwiGluMlp2, Trainable};
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
            eos_token_id: 1, // <eos> in Gemma tokenizer
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
            rope_theta_global: 8_000_000.0, // 1_000_000 * rope_scaling.factor(8.0)
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 256.0,
            eos_token_id: 1, // also 106, checked separately in generate loop
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
    pub q_proj: Linear2, // [hidden, n_q_heads * head_dim]
    pub k_proj: Linear2, // [hidden, n_kv_heads * head_dim]
    pub v_proj: Linear2, // [hidden, n_kv_heads * head_dim]
    pub o_proj: Linear2, // [n_q_heads * head_dim, hidden]
    /// Per-head RMSNorm on Q (Gemma 3 specific).
    pub q_norm: RmsNorm2, // [1, head_dim]
    /// Per-head RMSNorm on K (Gemma 3 specific).
    pub k_norm: RmsNorm2, // [1, head_dim]
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
            q_proj: Linear2::new_no_bias(h, nq * d, rng),
            k_proj: Linear2::new_no_bias(h, nkv * d, rng),
            v_proj: Linear2::new_no_bias(h, nkv * d, rng),
            o_proj: Linear2::new_no_bias(nq * d, h, rng),
            q_norm: RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            k_norm: RmsNorm2::new_with_eps(d, cfg.rms_norm_eps),
            n_q_heads: nq,
            n_kv_heads: nkv,
            head_dim: d,
            attn_scale: 1.0 / (cfg.query_pre_attn_scalar as f32).sqrt(),
            sliding_window: if is_global { None } else { cfg.sliding_window },
            rope_theta: if is_global {
                cfg.rope_theta_global
            } else {
                cfg.rope_theta_local
            },
        }
    }

    /// Forward pass: x [T, hidden] → output [T, hidden].
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let t = x.data().rows;
        let d = self.head_dim;
        let nq = self.n_q_heads;
        let nkv = self.n_kv_heads;

        // Project to Q, K, V
        let q = self.q_proj.forward(x); // [T, nq*d]
        let k = self.k_proj.forward(x); // [T, nkv*d]
        let v = self.v_proj.forward(x); // [T, nkv*d]

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
    pub gate_proj: Linear2, // [hidden, intermediate]
    pub up_proj: Linear2,   // [hidden, intermediate]
    pub down_proj: Linear2, // [intermediate, hidden]
}

impl Gemma3Mlp {
    pub fn new(cfg: &Config4, rng: &mut InitRng) -> Self {
        Gemma3Mlp {
            gate_proj: Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng),
            up_proj: Linear2::new_no_bias(cfg.hidden_size, cfg.intermediate_size, rng),
            down_proj: Linear2::new_no_bias(cfg.intermediate_size, cfg.hidden_size, rng),
        }
    }

    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        // Gemma3 uses gelu_pytorch_tanh (approximate GeLU), not SiLU.
        let gate = self.gate_proj.forward(x).gelu_tanh();
        let up = self.up_proj.forward(x);
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
    pub input_layernorm: RmsNorm2,
    pub self_attn: Gemma3Attention,
    pub post_attention_layernorm: RmsNorm2,
    /// Pre-FFN norm (Gemma 3 uses an extra "pre_feedforward_layernorm").
    pub pre_feedforward_layernorm: RmsNorm2,
    /// Post-FFN norm (Gemma 3 uses "post_feedforward_layernorm").
    pub post_feedforward_layernorm: RmsNorm2,
    pub mlp: Gemma3Mlp,
}

impl Gemma3Block {
    pub fn new(cfg: &Config4, layer_idx: usize, rng: &mut InitRng) -> Self {
        Gemma3Block {
            input_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            self_attn: Gemma3Attention::new(cfg, layer_idx, rng),
            post_attention_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            pre_feedforward_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            post_feedforward_layernorm: RmsNorm2::new_with_eps(cfg.hidden_size, cfg.rms_norm_eps),
            mlp: Gemma3Mlp::new(cfg, rng),
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
        let attn = self.self_attn.forward(&normed);
        let attn = self.post_attention_layernorm.forward_gemma3(&attn);
        let x2 = x.add(&attn);

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
    pub embed_tokens: TensorNode, // [vocab_size, hidden_size] — small f32 placeholder; real data in embed_bf16
    /// BF16 embedding table (vocab_size × hidden_size). Populated by load_weights_from_dir.
    /// Used for fast row lookups without a 2.7 GB f32 allocation.
    pub embed_bf16: Option<crate::autograd2::MatBf16>,
    pub layers: Vec<Gemma3Block>,
    pub norm: RmsNorm2,   // final layer norm
    pub lm_head: Linear2, // [hidden_size, vocab_size] — weight-tied with embed_tokens
    pub config: Config4,
}

impl Gemma3Model {
    pub fn new(cfg: Config4, rng: &mut InitRng) -> Self {
        let embed = TensorNode::leaf(Mat::new(
            rng.normal_vec(cfg.vocab_size * cfg.hidden_size, 0.02),
            cfg.vocab_size,
            cfg.hidden_size,
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
            q4k_weight: None,
            bf16_weight: None,
        };

        Gemma3Model {
            embed_tokens: embed,
            embed_bf16: None,
            layers,
            norm,
            lm_head,
            config: cfg,
        }
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
        let x_c = x.clone();
        let ids = token_ids.to_vec();
        x.set_backward(
            Box::new(move || {
                let dout = x_c.grad().clone(); // [T, h]
                let mut dte = embed_node.grad().clone();
                for (row, &tid) in ids.iter().enumerate() {
                    for col in 0..h {
                        *dte.at_mut(tid, col) += dout.at(row, col) * scale;
                    }
                }
                embed_node.set_grad(dte);
            }),
            vec![self.embed_tokens.clone()],
        );

        let mut x = x;

        for layer in &self.layers {
            x = layer.forward(&x);
        }

        let normed = self.norm.forward_gemma3(&x);
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
        if params.eos_token_id == Some(next) {
            return;
        }

        let mut prev = next;
        for _ in 1..max_new {
            let logits_node = self.forward(&[prev]);
            let logits = logits_node.data().clone();
            prev = sample_token(&logits, 0, &params, &seen, &mut rng);
            callback(prev);
            seen.push(prev);
            if params.eos_token_id == Some(prev) {
                break;
            }
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

            let bytes =
                std::fs::read(&path).map_err(|e| format!("cannot read {:?}: {}", path, e))?;

            let tensors = crate::transformer3::parse_safetensors(&bytes)
                .map_err(|e| format!("parse error in {:?}: {}", path, e))?;

            for t in &tensors {
                loaded_tensors += 1;
                if apply_tensor(self, t) {
                    matched += 1;
                }
            }
            loaded_shards += 1;
        }

        if loaded_shards == 0 {
            return Err(format!("no .safetensors files found in {}", dir));
        }

        println!(
            "Gemma3: loaded {} tensors ({} matched) from {} shards in {}",
            loaded_tensors, matched, loaded_shards, dir
        );
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

    // -------------------------------------------------------------------------
    // GGUF weight loading
    // -------------------------------------------------------------------------

    /// Load weights from a GGUF file (e.g. google/gemma-3-4b-it-qat-q4_0-gguf).
    ///
    /// Supports Q4_0, F32, F16, and BF16 tensors.
    /// Q4_0 tensors are loaded directly into `Linear2::q4_weight` (no BF16 copy).
    /// F16 tensors are converted to f32 on load.
    /// BF16 tensors are stored as `MatBf16` (lossless, 2× less RAM than f32).
    ///
    /// Tensor name mapping (GGUF blk.* → HuggingFace model.layers.* convention):
    /// - `token_embd.weight`           → embed_tokens
    /// - `output_norm.weight`          → model.norm
    /// - `blk.{i}.attn_norm.weight`    → input_layernorm
    /// - `blk.{i}.post_attn_norm.weight` or `post_attention_norm.weight` → post_attention_layernorm
    /// - `blk.{i}.ffn_pre_norm.weight` or `ffn_norm.weight` → pre_feedforward_layernorm
    /// - `blk.{i}.ffn_post_norm.weight` or `post_ffw_norm.weight` → post_feedforward_layernorm
    /// - `blk.{i}.attn_q_norm.weight`  → self_attn.q_norm
    /// - `blk.{i}.attn_k_norm.weight`  → self_attn.k_norm
    /// - `blk.{i}.attn_q.weight`       → self_attn.q_proj
    /// - `blk.{i}.attn_k.weight`       → self_attn.k_proj
    /// - `blk.{i}.attn_v.weight`       → self_attn.v_proj
    /// - `blk.{i}.attn_output.weight`  → self_attn.o_proj
    /// - `blk.{i}.ffn_gate.weight`     → mlp.gate_proj
    /// - `blk.{i}.ffn_up.weight`       → mlp.up_proj
    /// - `blk.{i}.ffn_down.weight`     → mlp.down_proj
    pub fn load_weights_from_gguf(&mut self, path: &str) -> std::io::Result<()> {
        use crate::autograd2::MatBf16;
        use crate::gguf_loader::{GgufFile, GgufType};

        eprintln!("[ GGUF ] Opening {}...", path);
        let gguf = GgufFile::open(path)?;
        eprintln!("[ GGUF ] Found {} tensors.", gguf.tensor_info.len());

        // Print architecture metadata
        if let Some(arch) = gguf
            .metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
        {
            eprintln!("[ GGUF ] Architecture: {}", arch);
        }

        let n_tensors = gguf.tensor_info.len();
        let mut loaded = 0usize;
        // Track whether an explicit output.weight was found in the file.
        // If so, we skip the default weight-tying from token_embd at the end.
        let mut lm_head_explicitly_loaded = false;

        for idx in 0..n_tensors {
            let name = gguf.tensor_info[idx].name.clone();
            let gtype = gguf.tensor_info[idx].gguf_type;

            // Helper: load tensor as f32 vec (handles F32/F16 → f32)
            let load_f32 = |gguf: &GgufFile, idx: usize| -> std::io::Result<Vec<f32>> {
                match gguf.tensor_info[idx].gguf_type {
                    GgufType::F32 => gguf.decode_f32(idx),
                    GgufType::F16 => gguf.decode_f16_to_f32(idx),
                    _ => Err(std::io::Error::new(
                        std::io::ErrorKind::Unsupported,
                        format!(
                            "expected f32/f16 for norm tensor {}",
                            gguf.tensor_info[idx].name
                        ),
                    )),
                }
            };

            // ---- token embedding ----
            if name == "token_embd.weight" {
                let shape = gguf.tensor_info[idx].shape.clone();
                // GGUF stores as [cols, rows] — for embed: [hidden, vocab]
                // We want [vocab, hidden]
                let (vocab, hidden) = if shape.len() == 2 {
                    (shape[1], shape[0])
                } else {
                    (shape[0], 1)
                };
                eprintln!(
                    "[ GGUF ] token_embd: type={:?} vocab={} hidden={}",
                    gtype, vocab, hidden
                );
                match gtype {
                    GgufType::Bf16 => {
                        let bits = gguf.decode_bf16(idx)?;
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    GgufType::F16 => {
                        let f32s = gguf.decode_f16_to_f32(idx)?;
                        let bits: Vec<u16> =
                            f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    GgufType::F32 => {
                        let f32s = gguf.decode_f32(idx)?;
                        let bits: Vec<u16> =
                            f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    GgufType::Q4_0 => {
                        // Dequantize Q4_0 embedding to f32, then convert to BF16 for storage.
                        // Memory order: flat[tok * hidden + col] — no transpose needed.
                        let f32s = gguf.decode_q4_0_to_f32(idx)?;
                        let bits: Vec<u16> =
                            f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    GgufType::Q4K => {
                        let f32s = gguf.decode_q4k_to_f32(idx)?;
                        let bits: Vec<u16> =
                            f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    GgufType::Q6K => {
                        let f32s = gguf.decode_q6k_to_f32(idx)?;
                        let bits: Vec<u16> =
                            f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
                        model_set_embed_bf16(self, bits, vocab, hidden);
                    }
                    _ => {
                        eprintln!(
                            "[ GGUF ] Warning: token_embd type {:?} not supported, skipping",
                            gtype
                        );
                    }
                }
                loaded += 1;
                continue;
            }

            // ---- lm_head output projection (some GGUF files include this
            //      separately even for weight-tied models) ----
            if name == "output.weight" {
                eprintln!(
                    "[ GGUF ] Loading explicit output.weight for lm_head (type={:?})",
                    gtype
                );
                load_linear_from_gguf(&gguf, idx, &mut self.lm_head)?;
                lm_head_explicitly_loaded = true;
                loaded += 1;
                continue;
            }

            // ---- output norm ----
            if name == "output_norm.weight" {
                let f32s = load_f32(&gguf, idx)?;
                let n = f32s.len();
                // GGUF stores (1 + w) for Gemma3 RMSNorm weights, but forward_gemma3
                // applies (1 + gamma).  Subtract 1 so the net result is (1 + w) * x_norm.
                let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                self.norm
                    .gamma
                    .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
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
                        // ---- layer norms (f32) ----
                        "attn_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w) for Gemma3 RMSNorm weights, but
                            // forward_gemma3 already applies (1 + gamma).  Subtract 1
                            // so the net effect is the correct (1 + w) * x_norm.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .input_layernorm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        // post_attention_layernorm
                        "post_attn_norm.weight" | "post_attention_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w); subtract 1 to match forward_gemma3.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .post_attention_layernorm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        // pre_feedforward_layernorm
                        "ffn_pre_norm.weight" | "ffn_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w); subtract 1 to match forward_gemma3.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .pre_feedforward_layernorm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        // post_feedforward_layernorm
                        "ffn_post_norm.weight" | "post_ffw_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w); subtract 1 to match forward_gemma3.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .post_feedforward_layernorm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        "attn_q_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w); subtract 1 to match forward_gemma3.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .self_attn
                                .q_norm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        "attn_k_norm.weight" => {
                            let f32s = load_f32(&gguf, idx)?;
                            let n = f32s.len();
                            // GGUF stores (1 + w); subtract 1 to match forward_gemma3.
                            let adjusted: Vec<f32> = f32s.iter().map(|&v| v - 1.0).collect();
                            layer
                                .self_attn
                                .k_norm
                                .gamma
                                .set_data(crate::autograd2::Mat::new(adjusted, 1, n));
                            loaded += 1;
                        }
                        // ---- projection weights ----
                        "attn_q.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.q_proj)?;
                            loaded += 1;
                        }
                        "attn_k.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.k_proj)?;
                            loaded += 1;
                        }
                        "attn_v.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.v_proj)?;
                            loaded += 1;
                        }
                        "attn_output.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.self_attn.o_proj)?;
                            loaded += 1;
                        }
                        "ffn_gate.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.mlp.gate_proj)?;
                            loaded += 1;
                        }
                        "ffn_up.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.mlp.up_proj)?;
                            loaded += 1;
                        }
                        "ffn_down.weight" => {
                            load_linear_from_gguf(&gguf, idx, &mut layer.mlp.down_proj)?;
                            loaded += 1;
                        }
                        other => {
                            eprintln!("[ GGUF ] Unknown blk tensor: blk.{}.{}", layer_idx, other);
                        }
                    }
                }
            } else if name != "token_embd.weight" && name != "output_norm.weight" {
                // Suppress verbose output for vision tower (v.blk.*) and multimodal tensors
                if !name.starts_with("v.") && !name.starts_with("mm.") {
                    eprintln!("[ GGUF ] Skipping unknown top-level tensor: {}", name);
                }
            }

            if loaded % 50 == 0 && loaded > 0 {
                eprintln!("[ GGUF ] Loaded {}/{} tensors...", loaded, n_tensors);
            }
        }

        // Weight tying: lm_head shares embed_tokens weights *unless* the file
        // contained an explicit output.weight tensor (non-tied variant).
        if lm_head_explicitly_loaded {
            eprintln!(
                "[ GGUF ] lm_head loaded from explicit output.weight — skipping weight tying."
            );
        } else if let Some(ref e) = self.embed_bf16 {
            eprintln!(
                "[ GGUF ] embed_bf16 set: {}×{} — applying weight tying to lm_head.",
                e.rows, e.cols
            );
            self.lm_head.load_bf16_arc(e.data.clone(), e.rows, e.cols);
        } else {
            eprintln!("[ GGUF ] WARNING: embed_bf16 is None — token embeddings not loaded!");
        }

        // ── Diagnostics: print first few gamma values of layer-0 norms ──────
        eprintln!("[ GGUF ] Norm gamma diagnostics (layer 0, first 5 values):");
        {
            let g = self.layers[0].input_layernorm.gamma.data();
            let vals: Vec<f32> = (0..5.min(g.cols)).map(|c| g.at(0, c)).collect();
            eprintln!("  input_layernorm gamma (stored, after -1 fix): {:?}", vals);
            let effective: Vec<f32> = vals.iter().map(|&v| 1.0 + v).collect();
            eprintln!(
                "  input_layernorm effective scale (1+gamma):    {:?}",
                effective
            );
        }
        {
            let g = self.layers[0].post_attention_layernorm.gamma.data();
            let vals: Vec<f32> = (0..5.min(g.cols)).map(|c| g.at(0, c)).collect();
            eprintln!("  post_attn_layernorm gamma (stored):           {:?}", vals);
        }
        {
            let g = self.layers[0].self_attn.q_norm.gamma.data();
            let vals: Vec<f32> = (0..5.min(g.cols)).map(|c| g.at(0, c)).collect();
            eprintln!("  q_norm gamma (stored):                        {:?}", vals);
        }
        eprintln!("[ GGUF ] Done. Loaded {} tensors.", loaded);
        Ok(())
    }
}

// ============================================================================
// GGUF loading helpers (free functions)
// ============================================================================

fn model_set_embed_bf16(model: &mut Gemma3Model, bits: Vec<u16>, vocab: usize, hidden: usize) {
    use crate::autograd2::MatBf16;
    // Wrap in Arc once so embed_bf16 and lm_head share the same allocation (no clone).
    let arc = std::sync::Arc::new(bits);
    model.embed_bf16 = Some(MatBf16 {
        data: arc.clone(),   // O(1) refcount bump, not a data copy
        rows: vocab,
        cols: hidden,
    });
    model
        .embed_tokens
        .set_data(crate::autograd2::Mat::zeros(vocab, hidden));
    model.lm_head.load_bf16_arc(arc, vocab, hidden);
}

/// Load a Linear2 weight from a GGUF tensor, supporting Q4_0, BF16, F16, F32.
fn load_linear_from_gguf(
    gguf: &crate::gguf_loader::GgufFile,
    idx: usize,
    linear: &mut crate::nn2::Linear2,
) -> std::io::Result<()> {
    use crate::autograd2::MatBf16;
    use crate::gguf_loader::GgufType;

    let gtype = gguf.tensor_info[idx].gguf_type;
    let shape = &gguf.tensor_info[idx].shape;
    // GGUF stores weight as [in_features, out_features] (Fortran/column-major)
    // Our Linear2 stores weight as [out_features, in_features] (row-major)
    // GgufFile::decode_q4_0_to_q4mat already handles the transpose.
    // For BF16/F16/F32 we need to handle it here.
    let (rows, cols) = if shape.len() >= 2 {
        (shape[1], shape[0]) // transpose: GGUF [cols, rows] → our [rows, cols]
    } else {
        (1, shape[0])
    };

    match gtype {
        GgufType::Q4_0 => {
            let q4 = gguf.decode_q4_0_to_q4mat(idx)?;
            linear.q4_weight = Some(q4);
            linear.bf16_weight = None;
            linear.weight.set_data(crate::autograd2::Mat::zeros(0, 0));
        }
        GgufType::Bf16 => {
            let bits = gguf.decode_bf16(idx)?;
            linear.load_bf16(bits, rows, cols);
        }
        GgufType::F16 => {
            let f32s = gguf.decode_f16_to_f32(idx)?;
            let bits: Vec<u16> = f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
            linear.load_bf16(bits, rows, cols);
        }
        GgufType::F32 => {
            let f32s = gguf.decode_f32(idx)?;
            linear
                .weight
                .set_data(crate::autograd2::Mat::new(f32s, rows, cols));
        }
        GgufType::Q4K => {
            // Load Q4_K natively: keep raw block bytes, dequantize on-the-fly
            // during matmul. Saves ~3.5× RAM vs BF16 and avoids the full
            // decode+convert pass at load time.
            let q4k = gguf.decode_q4k_to_q4kmat(idx)?;
            // Diagnostic: check first 8 values of first Q4K tensor loaded
            linear.q4k_weight = Some(q4k);
            linear.bf16_weight = None;
            linear.weight.set_data(crate::autograd2::Mat::zeros(0, 0));
        }
        GgufType::Q6K => {
            let f32s = gguf.decode_q6k_to_f32(idx)?;
            let bits: Vec<u16> = f32s.iter().map(|&f| MatBf16::f32_to_bf16(f)).collect();
            linear.load_bf16(bits, rows, cols);
        }
        _ => {
            eprintln!(
                "[ GGUF ] Warning: unsupported type {:?} for tensor {}, skipping",
                gtype, gguf.tensor_info[idx].name
            );
        }
    }
    Ok(())
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
            let row_max = (0..v)
                .map(|c| logits.at(r, c))
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..v {
                let e = (logits.at(r, c) - row_max).exp();
                *probs.at_mut(r, c) = e;
                sum_exp += e;
            }
            for c in 0..v {
                *probs.at_mut(r, c) /= sum_exp;
            }
            loss_val -= probs.at(r, targets[r]).ln().max(-100.0);
        }
        loss_val /= t as f32;

        let loss = TensorNode::leaf(Mat::new(vec![loss_val], 1, 1));
        let logits_c = logits_node.clone();
        let probs_stored = probs;
        let targets_v = targets.to_vec();

        loss.set_backward(
            Box::new(move || {
                let mut dlogits = logits_c.grad().clone();
                for r in 0..t {
                    for c in 0..v {
                        let ind = if c == targets_v[r] { 1.0f32 } else { 0.0 };
                        *dlogits.at_mut(r, c) += (probs_stored.at(r, c) - ind) / t as f32;
                    }
                }
                logits_c.set_grad(dlogits);
                logits_c.call_backward_fn();
            }),
            vec![logits_node],
        );

        loss
    }
}

// ============================================================================
// Binary weight cache  (fast save/load, skips safetensors parsing)
// ============================================================================
//
// Format:  MAGIC(8) | N_RECORDS(u32le) | record* | EOF
// Record:  name_len(u32le) | name(utf8) | dtype(u8: 0=f32, 1=bf16) |
//          rows(u32le) | cols(u32le) | data(rows*cols * dtype_bytes)
//
// On load the file is mmap'd; no heap copies until each record is consumed.

const CACHE_MAGIC: &[u8; 8] = b"G3CACHE1";

impl Gemma3Model {
    /// Save all weights to a binary cache file for fast subsequent loads.
    pub fn save_cache(&self, path: &str) -> std::io::Result<()> {
        use std::io::{BufWriter, Write};
        let f = std::fs::File::create(path)?;
        let mut w = BufWriter::new(f);

        // Collect all (name, dtype, rows, cols, raw_bytes) tuples
        let mut records: Vec<(&str, u8, usize, usize, Vec<u8>)> = Vec::new();

        // Helper closures
        let f32_rec = |name: &'static str,
                       data: &[f32],
                       rows: usize,
                       cols: usize|
         -> (&'static str, u8, usize, usize, Vec<u8>) {
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) }
                    .to_vec();
            (name, 0u8, rows, cols, bytes)
        };
        let bf16_rec = |name: &'static str,
                        data: &[u16],
                        rows: usize,
                        cols: usize|
         -> (&'static str, u8, usize, usize, Vec<u8>) {
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) }
                    .to_vec();
            (name, 1u8, rows, cols, bytes)
        };

        // embed_tokens (bf16)
        if let Some(ref e) = self.embed_bf16 {
            records.push(bf16_rec(
                "model.embed_tokens.weight",
                &e.data,
                e.rows,
                e.cols,
            ));
        }

        // final norm
        {
            let g = self.norm.gamma.data();
            records.push(f32_rec("model.norm.weight", &g.data, g.rows, g.cols));
        }

        // per-layer weights
        for (i, layer) in self.layers.iter().enumerate() {
            let push_norm =
                |name: String, node: &crate::autograd2::TensorNode, records: &mut Vec<_>| {
                    let g = node.data();
                    let bytes = unsafe {
                        std::slice::from_raw_parts(g.data.as_ptr() as *const u8, g.data.len() * 4)
                    }
                    .to_vec();
                    records.push((
                        Box::leak(name.into_boxed_str()) as &str,
                        0u8,
                        g.rows,
                        g.cols,
                        bytes,
                    ));
                };
            let prefix = format!("model.layers.{}", i);
            push_norm(
                format!("{}.input_layernorm.weight", prefix),
                &layer.input_layernorm.gamma,
                &mut records,
            );
            push_norm(
                format!("{}.post_attention_layernorm.weight", prefix),
                &layer.post_attention_layernorm.gamma,
                &mut records,
            );
            push_norm(
                format!("{}.pre_feedforward_layernorm.weight", prefix),
                &layer.pre_feedforward_layernorm.gamma,
                &mut records,
            );
            push_norm(
                format!("{}.post_feedforward_layernorm.weight", prefix),
                &layer.post_feedforward_layernorm.gamma,
                &mut records,
            );
            push_norm(
                format!("{}.self_attn.q_norm.weight", prefix),
                &layer.self_attn.q_norm.gamma,
                &mut records,
            );
            push_norm(
                format!("{}.self_attn.k_norm.weight", prefix),
                &layer.self_attn.k_norm.gamma,
                &mut records,
            );

            let push_bf16 = |name: String, lin: &crate::nn2::Linear2, records: &mut Vec<_>| {
                if let Some(ref b) = lin.bf16_weight {
                    let bytes = unsafe {
                        std::slice::from_raw_parts(b.data.as_ptr() as *const u8, b.data.len() * 2)
                    }
                    .to_vec();
                    records.push((
                        Box::leak(name.into_boxed_str()) as &str,
                        1u8,
                        b.rows,
                        b.cols,
                        bytes,
                    ));
                }
            };
            push_bf16(
                format!("{}.self_attn.q_proj.weight", prefix),
                &layer.self_attn.q_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.self_attn.k_proj.weight", prefix),
                &layer.self_attn.k_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.self_attn.v_proj.weight", prefix),
                &layer.self_attn.v_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.self_attn.o_proj.weight", prefix),
                &layer.self_attn.o_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.mlp.gate_proj.weight", prefix),
                &layer.mlp.gate_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.mlp.up_proj.weight", prefix),
                &layer.mlp.up_proj,
                &mut records,
            );
            push_bf16(
                format!("{}.mlp.down_proj.weight", prefix),
                &layer.mlp.down_proj,
                &mut records,
            );
        }

        // Write header
        w.write_all(CACHE_MAGIC)?;
        w.write_all(&(records.len() as u32).to_le_bytes())?;

        // Write records
        for (name, dtype, rows, cols, data) in &records {
            let name_bytes = name.as_bytes();
            w.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
            w.write_all(name_bytes)?;
            w.write_all(&[*dtype])?;
            w.write_all(&(*rows as u32).to_le_bytes())?;
            w.write_all(&(*cols as u32).to_le_bytes())?;
            w.write_all(data)?;
        }
        Ok(())
    }

    /// Load weights from a binary cache file (fast path).
    /// Returns false if the file doesn't exist or has wrong magic.
    pub fn load_cache(&mut self, path: &str) -> std::io::Result<bool> {
        use std::io::Read;
        let mut f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("[ Cache ] No cache file at {}: {}", path, e);
                return Ok(false);
            }
        };
        eprintln!("[ Cache ] Reading cache file {}...", path);
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        eprintln!("[ Cache ] Read {} MB", buf.len() / 1_048_576);

        if buf.len() < 12 || &buf[..8] != CACHE_MAGIC {
            eprintln!("[ Cache ] Bad magic or too short, ignoring cache.");
            return Ok(false);
        }

        let n_records = u32::from_le_bytes(buf[8..12].try_into().unwrap()) as usize;
        eprintln!("[ Cache ] Loading {} records...", n_records);
        let mut pos = 12usize;

        let read_u32 = |buf: &[u8], p: &mut usize| -> u32 {
            let v = u32::from_le_bytes(buf[*p..*p + 4].try_into().unwrap());
            *p += 4;
            v
        };

        for _ in 0..n_records {
            let name_len = read_u32(&buf, &mut pos) as usize;
            let name = std::str::from_utf8(&buf[pos..pos + name_len])
                .unwrap()
                .to_string();
            pos += name_len;
            let dtype = buf[pos];
            pos += 1;
            let rows = read_u32(&buf, &mut pos) as usize;
            let cols = read_u32(&buf, &mut pos) as usize;
            let n_elems = rows * cols;
            let elem_bytes = if dtype == 0 { 4 } else { 2 };
            let data_bytes = &buf[pos..pos + n_elems * elem_bytes];
            pos += n_elems * elem_bytes;

            // Synthesize a SafeTensor and reuse apply_tensor
            let mut st = crate::transformer3::SafeTensor {
                name: name,
                shape: vec![rows, cols],
                data: Vec::new(),
                bf16_data: None,
            };
            if dtype == 0 {
                // f32
                let mut f32s = vec![0.0f32; n_elems];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data_bytes.as_ptr(),
                        f32s.as_mut_ptr() as *mut u8,
                        data_bytes.len(),
                    );
                }
                st.data = f32s;
            } else {
                // bf16
                let mut u16s = vec![0u16; n_elems];
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data_bytes.as_ptr(),
                        u16s.as_mut_ptr() as *mut u8,
                        data_bytes.len(),
                    );
                }
                // Also fill f32 data for the f32 fallback path
                st.data = u16s
                    .iter()
                    .map(|&b| crate::autograd2::MatBf16::bf16_to_f32(b))
                    .collect();
                st.bf16_data = Some(u16s);
            }
            apply_tensor(self, &st);
        }
        Ok(true)
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
    let name = t
        .name
        .strip_prefix("language_model.")
        .unwrap_or(t.name.as_str());

    // Global tensors
    if name == "model.embed_tokens.weight" {
        // Store as BF16 to avoid a 2.7 GB f32 allocation for the 262K×2560 table.
        if let Some(ref bits) = t.bf16_data {
            if bits.len() == t.shape[0] * t.shape[1] {
                use crate::autograd2::MatBf16;
                // Wrap in Arc so embed_bf16 and lm_head share one allocation.
                let arc = std::sync::Arc::new(bits.clone());
                model.embed_bf16 = Some(MatBf16 {
                    data: arc.clone(),
                    rows: t.shape[0],
                    cols: t.shape[1],
                });
                // Keep a tiny f32 placeholder so the TensorNode shape is consistent.
                model
                    .embed_tokens
                    .set_data(crate::autograd2::Mat::zeros(t.shape[0], t.shape[1]));
                // Share the same Arc with lm_head — no data copy.
                model.lm_head.load_bf16_arc(arc, t.shape[0], t.shape[1]);
                return Some(true);
            }
        }
        return Some(set_node(
            &model.embed_tokens,
            &t.data,
            t.shape[0],
            t.shape[1],
        ));
    }
    if name == "model.norm.weight" {
        let n = t.data.len();
        return Some(set_node(&model.norm.gamma, &t.data, 1, n));
    }
    if name == "lm_head.weight" {
        return Some(set_node(
            &model.lm_head.weight,
            &t.data,
            t.shape[0],
            t.shape[1],
        ));
    }

    // Per-layer tensors: "model.layers.{i}.{...}"
    let rest = name.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let layer_idx: usize = rest[..dot].parse().ok()?;
    if layer_idx >= model.layers.len() {
        return Some(false);
    }
    let layer_name = &rest[dot + 1..];
    let layer = &mut model.layers[layer_idx];

    match layer_name {
        // Layer-norm weights: use data.len() as size to handle both
        // 1D [N] (safetensors) and 2D [1,N] (cache file) shapes.
        "input_layernorm.weight" => Some(set_node(
            &layer.input_layernorm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),
        "post_attention_layernorm.weight" => Some(set_node(
            &layer.post_attention_layernorm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),
        "pre_feedforward_layernorm.weight" => Some(set_node(
            &layer.pre_feedforward_layernorm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),
        "post_feedforward_layernorm.weight" => Some(set_node(
            &layer.post_feedforward_layernorm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),
        "self_attn.q_norm.weight" => Some(set_node(
            &layer.self_attn.q_norm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),
        "self_attn.k_norm.weight" => Some(set_node(
            &layer.self_attn.k_norm.gamma,
            &t.data,
            1,
            t.data.len(),
        )),

        // Large projection weights: store as BF16 when available (lossless, 2× RAM).
        "self_attn.q_proj.weight" => {
            let r = set_linear(&mut layer.self_attn.q_proj, t, t.shape[0], t.shape[1]);
            // Debug: print first 8 values of layer 0's q_proj for comparison with GGUF
            if layer_idx == 0 {
                let vals: Vec<f32> = if let Some(ref bf16) = layer.self_attn.q_proj.bf16_weight {
                    bf16.to_f32().data[..8.min(bf16.data.len())].to_vec()
                } else {
                    let d = layer.self_attn.q_proj.weight.data();
                    d.data[..8.min(d.data.len())].to_vec()
                };
                eprint!("[ ST  debug ] model.layers.0.self_attn.q_proj first 8 values: ");
                for v in &vals {
                    eprint!("{:.4} ", v);
                }
                eprintln!("  shape={}x{}", t.shape[0], t.shape[1]);
            }
            Some(r)
        }
        "self_attn.k_proj.weight" => Some(set_linear(
            &mut layer.self_attn.k_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        "self_attn.v_proj.weight" => Some(set_linear(
            &mut layer.self_attn.v_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        "self_attn.o_proj.weight" => Some(set_linear(
            &mut layer.self_attn.o_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        "mlp.gate_proj.weight" => Some(set_linear(
            &mut layer.mlp.gate_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        "mlp.up_proj.weight" => Some(set_linear(
            &mut layer.mlp.up_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        "mlp.down_proj.weight" => Some(set_linear(
            &mut layer.mlp.down_proj,
            t,
            t.shape[0],
            t.shape[1],
        )),
        _ => Some(false),
    }
}

/// Set a TensorNode's f32 data directly (used for small tensors: norms, embeddings).
fn set_node(node: &TensorNode, data: &[f32], rows: usize, cols: usize) -> bool {
    if data.len() != rows * cols {
        return false;
    }
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
    if t.data.len() != rows * cols {
        return false;
    }
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
        let d = new_k.cols;
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
        let rows = self.seq_len - start;
        Mat::from_fn(rows, self.k.cols, |r, c| self.k.at(start + r, c))
    }

    /// Return last `window` rows of V (or all rows when seq_len < window).
    pub fn v_last(&self, window: usize) -> Mat {
        let start = self.seq_len.saturating_sub(window);
        let rows = self.seq_len - start;
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
        let d = config.head_dim;
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
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut Gemma3LayerKvCache) -> TensorNode {
        let n_new = x.data().rows;
        let d = self.head_dim;
        let nq = self.n_q_heads;
        let nkv = self.n_kv_heads;
        let seq_offset = cache.seq_len;

        // 1. Projections
        let q = self.q_proj.forward(x);
        let k = self.k_proj.forward(x);
        let v = self.v_proj.forward(x);

        // 2. Per-head RMSNorm
        let q = apply_per_head_norm(&q, &self.q_norm, n_new, nq, d);
        let k = apply_per_head_norm(&k, &self.k_norm, n_new, nkv, d);

        // 3. RoPE at absolute positions
        let q = apply_rope_at_offset(&q, nq, n_new, d, self.rope_theta, seq_offset);
        let k = apply_rope_at_offset(&k, nkv, n_new, d, self.rope_theta, seq_offset);

        // 4. Append to cache
        cache.append(&k.data().clone(), &v.data().clone());

        // 5. Select context window: compute bounds without copying the cache.
        //    k_start..k_end indexes rows of cache.k / cache.v to attend over.
        let k_end = cache.seq_len;
        let k_start = match self.sliding_window {
            Some(w) => k_end.saturating_sub(w),
            None => 0,
        };

        // 6. GQA attention (no per-layer copy of the KV cache)
        let attn_out = gqa_attention_cached(
            &q.data(),
            &cache.k,
            &cache.v,
            k_start,
            k_end,
            nq,
            nkv,
            d,
            self.attn_scale,
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
    pub fn forward_cached(&self, x: &TensorNode, cache: &mut Gemma3LayerKvCache) -> TensorNode {
        let normed = self.input_layernorm.forward_gemma3(x);
        let attn = self.self_attn.forward_cached(&normed, cache);
        let attn = self.post_attention_layernorm.forward_gemma3(&attn);
        let x2 = x.add(&attn);

        let normed2 = self.pre_feedforward_layernorm.forward_gemma3(&x2);
        let mlp_out = self.mlp.forward(&normed2);
        let mlp_out = self.post_feedforward_layernorm.forward_gemma3(&mlp_out);
        x2.add(&mlp_out)
    }
}

impl Gemma3Model {
    /// Quantize all large projection weights to INT4 and free float storage.
    ///
    /// Call once after all weights are loaded. Converts BF16 (or f32) weights
    /// to Q4 block-wise quantization (block size 32), freeing ~4× the RAM.
    /// Safe for inference-only use — do not call if you need backward passes.
    pub fn quantize_all_weights(&mut self) {
        for layer in &mut self.layers {
            layer.self_attn.q_proj.quantize_bf16_and_free();
            layer.self_attn.k_proj.quantize_bf16_and_free();
            layer.self_attn.v_proj.quantize_bf16_and_free();
            layer.self_attn.o_proj.quantize_bf16_and_free();
            layer.mlp.gate_proj.quantize_bf16_and_free();
            layer.mlp.up_proj.quantize_bf16_and_free();
            layer.mlp.down_proj.quantize_bf16_and_free();
        }
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
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        mut callback: impl FnMut(usize),
    ) {
        use crate::transformer3::SamplingParams;

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

        // Gemma 3 uses two EOS token ids: 1 (<eos>) and 106 (<end_of_turn>).
        let is_eos = |tok: usize| tok == 1 || tok == 106;

        let cache = Gemma3KvCache::new(&self.config);
        let h = self.config.hidden_size;
        let scale = (h as f32).sqrt();
        let mut rng = LcgRng::new(seed);

        // IMPORTANT: `seen` tracks only *generated* tokens for the repetition
        // penalty — NOT prompt tokens.  If we included prompt tokens, the
        // EOS / end-of-turn token (106) that appears in the chat template
        // would be penalised from the very first decode step, making the
        // model much less likely to stop naturally and causing it to keep
        // generating garbage until max_new is reached.
        let mut seen: Vec<usize> = Vec::new();

        // Helper: look up one embedding row from BF16 table (preferred) or f32.
        let embed_row = |tok: usize| -> Vec<f32> {
            if let Some(ref bf16) = self.embed_bf16 {
                use crate::autograd2::MatBf16;
                (0..h)
                    .map(|c| MatBf16::bf16_to_f32(bf16.data[tok * h + c]) * scale)
                    .collect()
            } else {
                let te = self.embed_tokens.data();
                (0..h).map(|c| te.at(tok, c) * scale).collect()
            }
        };

        // ----- Prefill -----
        let t_prompt = token_ids.len();
        let prefill_start = std::time::Instant::now();

        let embed_data: Vec<f32> = token_ids.iter().flat_map(|&tok| embed_row(tok)).collect();

        let x_data = Mat {
            data: embed_data,
            rows: t_prompt,
            cols: h,
        };
        let mut x = TensorNode::leaf(x_data);
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
        }
        // Only run lm_head on the last token row — avoids a T×vocab matmul.
        let last_row = {
            let xd = x.data();
            Mat::from_fn(1, h, |_, c| xd.at(t_prompt - 1, c))
        };
        let normed_final = self.norm.forward_gemma3(&TensorNode::leaf(last_row));
        let logits_node = self.lm_head.forward(&normed_final);

        let prefill_ms = prefill_start.elapsed().as_millis();
        let prefill_tps = t_prompt as f64 / prefill_start.elapsed().as_secs_f64();
        eprintln!(
            "[ Gemma3 ] Prefill: {} tokens in {:.0} ms ({:.1} tok/s)",
            t_prompt, prefill_ms, prefill_tps
        );

        // ── Prefill logit diagnostics ──────────────────────────────────────
        {
            // Hidden-state (post final-norm, last token row)
            let nx = normed_final.data();
            let nx_vals = &nx.data;
            let nx_min = nx_vals.iter().cloned().fold(f32::INFINITY, f32::min);
            let nx_max = nx_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let nx_mean = nx_vals.iter().sum::<f32>() / nx_vals.len() as f32;
            let nx_std = {
                let v = nx_vals.iter().map(|&v| (v - nx_mean).powi(2)).sum::<f32>()
                    / nx_vals.len() as f32;
                v.sqrt()
            };
            // eprintln!(
            //     "[ Gemma3-dbg ] prefill hidden_after_norm: \
            //      min={:.4} max={:.4} mean={:.4} std={:.4}",
            //     nx_min, nx_max, nx_mean, nx_std
            // );

            let ld = logits_node.data();
            let lv = &ld.data;
            let l_min = lv.iter().cloned().fold(f32::INFINITY, f32::min);
            let l_max = lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let l_mean = lv.iter().sum::<f32>() / lv.len() as f32;
            // eprintln!(
            //     "[ Gemma3-dbg ] prefill logits: min={:.2} max={:.2} mean={:.2}  vocab={}",
            //     l_min,
            //     l_max,
            //     l_mean,
            //     lv.len()
            // );
            let mut indexed: Vec<(usize, f32)> = lv.iter().cloned().enumerate().collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
            // eprintln!("[ Gemma3-dbg ] prefill top-5 raw logits: {:?}", top5);

            let n_nan = lv.iter().filter(|&&v| v.is_nan()).count();
            let n_inf = lv.iter().filter(|&&v| v.is_infinite()).count();
            if n_nan > 0 || n_inf > 0 {
                // eprintln!(
                //     "[ Gemma3-dbg ] *** WARNING: {} NaN, {} Inf in prefill logits ***",
                //     n_nan, n_inf
                // );
            }
        }

        let first = sample_token(&logits_node.data(), 0, &params, &seen, &mut rng);
        callback(first);
        seen.push(first);
        if is_eos(first) {
            eprintln!("[ Gemma3 ] EOS after first token — generation complete.");
            return;
        }

        // ----- Decode loop -----
        let decode_start = std::time::Instant::now();
        let mut prev = first;
        let mut n_decoded = 1usize;

        // Timing accumulators for first-step profiling (printed after step 1).
        let mut t_embed_us = 0u128;
        let mut t_layers_us = 0u128;
        let mut t_lmhead_us = 0u128;
        let mut profile_printed = false;

        for step in 1..max_new {
            let t0 = std::time::Instant::now();
            let x_data = Mat {
                data: embed_row(prev),
                rows: 1,
                cols: h,
            };
            let mut x = TensorNode::leaf(x_data);
            t_embed_us += t0.elapsed().as_micros();

            let t1 = std::time::Instant::now();
            for (i, layer) in self.layers.iter().enumerate() {
                x = layer.forward_cached(&x, &mut cache.layers[i].borrow_mut());
            }
            t_layers_us += t1.elapsed().as_micros();

            let t2 = std::time::Instant::now();
            let normed_x = self.norm.forward_gemma3(&x);
            let logits_node = self.lm_head.forward(&normed_x);
            t_lmhead_us += t2.elapsed().as_micros();

            // ── Diagnostics for first 10 decode steps ─────────────────────
            if step <= 10 {
                let ld = logits_node.data();
                let lv = &ld.data;
                let l_max = lv.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut indexed: Vec<(usize, f32)> = lv.iter().cloned().enumerate().collect();
                indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                let top5: Vec<(usize, f32)> = indexed[..5.min(indexed.len())].to_vec();
                eprintln!(
                    "[ Gemma3-dbg ] step={} prev_tok={}  logit_max={:.2}  top-5: {:?}",
                    step, prev, l_max, top5
                );
                let n_nan = lv.iter().filter(|&&v| v.is_nan()).count();
                let n_inf = lv.iter().filter(|&&v| v.is_infinite()).count();
                if n_nan > 0 || n_inf > 0 {
                    eprintln!(
                        "[ Gemma3-dbg ]   *** WARNING: {} NaN, {} Inf in logits ***",
                        n_nan, n_inf
                    );
                }
            }

            prev = sample_token(&logits_node.data(), 0, &params, &seen, &mut rng);
            callback(prev);
            seen.push(prev);
            n_decoded += 1;

            // After the first decode step, print a breakdown so the user can
            // see where time is actually being spent.
            if step == 1 && !profile_printed {
                profile_printed = true;
                eprintln!(
                    "[ Gemma3 ] Step-1 breakdown: embed={:.1}ms  layers={:.1}ms  lm_head={:.1}ms",
                    t_embed_us as f64 / 1000.0,
                    t_layers_us as f64 / 1000.0,
                    t_lmhead_us as f64 / 1000.0,
                );
            }

            if is_eos(prev) {
                break;
            }
        }

        let decode_secs = decode_start.elapsed().as_secs_f64();
        let decode_tps = n_decoded as f64 / decode_secs.max(1e-9);
        eprintln!(
            "[ Gemma3 ] Decode:  {} tokens in {:.0} ms ({:.1} tok/s)",
            n_decoded,
            decode_secs * 1000.0,
            decode_tps
        );
        // Print averaged breakdown if we ran more than one decode step.
        if n_decoded > 1 {
            eprintln!(
                "[ Gemma3 ] Avg/step: embed={:.1}ms  layers={:.1}ms  lm_head={:.1}ms",
                t_embed_us as f64 / (n_decoded as f64 * 1000.0),
                t_layers_us as f64 / (n_decoded as f64 * 1000.0),
                t_lmhead_us as f64 / (n_decoded as f64 * 1000.0),
            );
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
        let head_data = Mat::from_fn(t, head_dim, |row, col| x_data.at(row, h * head_dim + col));
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
    let x_c = x.clone();
    let result_c = result.clone();
    let norm_gamma_c = norm.gamma.clone();
    let eps = norm.eps;

    result.set_backward(
        Box::new(move || {
            let dout = result_c.grad().clone(); // [T, n_heads * head_dim]
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
                        let g = 1.0 + norm_gamma_c.data().at(0, col);
                        let xi = x_data.at(row, h * head_dim + col);
                        let dy = dout.at(row, h * head_dim + col);
                        let term1 = dy * g * inv_rms;
                        let term2 =
                            xi * inv_rms * inv_rms * inv_rms * dot_dy_gamma_x / head_dim as f32;
                        *dx.at_mut(row, h * head_dim + col) += term1 - term2;
                    }
                }
            }
            x_c.set_grad(dx);
            x_c.call_backward_fn();
        }),
        vec![x.clone()],
    );

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
    let x_c = x.clone();
    let result_c = result.clone();

    result.set_backward(
        Box::new(move || {
            let dout = result_c.grad().clone();
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
                        *dx.at_mut(pos, c0) += dy0 * cos_a + dy1 * sin_a;
                        *dx.at_mut(pos, c1) += -dy0 * sin_a + dy1 * cos_a;
                    }
                }
            }
            x_c.set_grad(dx);
            x_c.call_backward_fn();
        }),
        vec![x.clone()],
    );

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
    let ratio = scale / default_scale; // multiply Q by this to get the right scale

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
            if causal_ok && window_ok {
                scores.at(r, c)
            } else {
                f32::NEG_INFINITY
            }
        });

        // Softmax per row
        for r in 0..t {
            let row_max = (0..t)
                .map(|c| masked.at(r, c))
                .fold(f32::NEG_INFINITY, f32::max);
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
                for c in 0..t {
                    *masked.at_mut(r, c) /= sum_exp;
                }
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
            let pos = offset + row;
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

/// GQA attention for the cached decode path.
///
/// `q_data` is [t_q, n_q_heads * d_head].
/// `k_cache` / `v_cache` are the full pre-allocated cache mats
/// [max_seq_len, n_kv_heads * d_head]; only rows `k_start..k_end` are used,
/// so the caller never needs to copy a window slice into a temporary Mat.
///
/// **Decode path (t_q == 1)** — fully zero-alloc: scores and weighted sum are
/// computed with direct index arithmetic (+ BLAS sdot / saxpy when available).
///
/// **Prefill path (t_q > 1)** — per-head Mat copies are still made (same as
/// before) because sgemm needs contiguous data; this path only runs once per
/// generation (during prompt processing), so the cost is acceptable.
fn gqa_attention_cached(
    q_data: &Mat,
    k_cache: &Mat,
    v_cache: &Mat,
    k_start: usize,
    k_end: usize,
    n_q_heads: usize,
    n_kv_heads: usize,
    d_head: usize,
    scale: f32,
) -> Mat {
    let t_q = q_data.rows;
    let t_kv = k_end - k_start; // number of context tokens
    let kv_stride = k_cache.cols; // n_kv_heads * d_head
    let group = n_q_heads / n_kv_heads;
    let mut out = Mat::zeros(t_q, n_q_heads * d_head);

    // ── Decode path (single query token) ────────────────────────────────────
    // Avoids all per-head temporary matrix allocations.
    if t_q == 1 {
        let mut scores = vec![0.0f32; t_kv];

        for qh in 0..n_q_heads {
            let kvh = qh / group;
            let q_off = qh * d_head; // offset into q_data row 0
            let kv_off = kvh * d_head; // offset into each k/v cache row
            let out_off = qh * d_head; // offset into out row 0

            // Step 1: scores[c] = dot(q_h, k_h[c]) * scale
            for (ci, cache_row) in (k_start..k_end).enumerate() {
                let k_base = cache_row * kv_stride + kv_off;

                #[cfg(feature = "blas")]
                {
                    scores[ci] = unsafe {
                        cblas::sdot(
                            d_head as i32,
                            &q_data.data[q_off..],
                            1,
                            &k_cache.data[k_base..],
                            1,
                        )
                    } * scale;
                }
                #[cfg(not(feature = "blas"))]
                {
                    let mut dot = 0.0f32;
                    for di in 0..d_head {
                        dot += q_data.data[q_off + di] * k_cache.data[k_base + di];
                    }
                    scores[ci] = dot * scale;
                }
            }

            // Step 2: softmax
            let max_s = scores[..t_kv]
                .iter()
                .cloned()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for s in &mut scores[..t_kv] {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            if sum_exp > 0.0 {
                for s in &mut scores[..t_kv] {
                    *s /= sum_exp;
                }
            }

            // Step 3: out_h = sum_c(scores[c] * v_h[c])
            for (ci, cache_row) in (k_start..k_end).enumerate() {
                let v_base = cache_row * kv_stride + kv_off;
                let w = scores[ci];

                #[cfg(feature = "blas")]
                unsafe {
                    cblas::saxpy(
                        d_head as i32,
                        w,
                        &v_cache.data[v_base..],
                        1,
                        &mut out.data[out_off..],
                        1,
                    );
                }
                #[cfg(not(feature = "blas"))]
                for di in 0..d_head {
                    out.data[out_off + di] += w * v_cache.data[v_base + di];
                }
            }
        }
        return out;
    }

    // ── Prefill path (multiple query tokens) ────────────────────────────────
    // Per-head copies are necessary to form contiguous mats for sgemm.
    // (causal_offset removed — we use k_start-relative indexing instead)

    for qh in 0..n_q_heads {
        let kvh = qh / group;
        let kv_off = kvh * d_head;

        let q_h = Mat::from_fn(t_q, d_head, |r, c| q_data.at(r, qh * d_head + c));
        let k_h = Mat::from_fn(t_kv, d_head, |r, c| {
            k_cache.data[(k_start + r) * kv_stride + kv_off + c]
        });
        let v_h = Mat::from_fn(t_kv, d_head, |r, c| {
            v_cache.data[(k_start + r) * kv_stride + kv_off + c]
        });

        // Scores [t_q, t_kv] with causal mask
        let raw_scores = q_h.matmul(&k_h.transpose()).scale(scale);

        // Causal masking accounting for k_start offset.
        //
        // Key at window index c has absolute sequence position (k_start + c).
        // Query at row r has absolute position r (fresh prefill, seq_offset=0)
        // or (seq_offset + r) in general — but seq_offset=0 during prefill.
        //
        // A key is causally valid for query r iff: k_start + c <= r
        //                                     iff: c <= r - k_start
        //
        // When r < k_start the query precedes all windowed keys; the row
        // stays all-zero (the output for that position is the zero vector).
        // This happens when the prompt is longer than the window.
        let mut w = Mat::zeros(t_q, t_kv);
        for r in 0..t_q {
            if r < k_start {
                // All windowed keys are causally after this query — attend to nothing.
                continue;
            }
            // max_kv: largest window index the query may attend to (clamped).
            let max_kv = (r - k_start).min(t_kv - 1);
            let row_max = (0..=max_kv)
                .map(|c| raw_scores.at(r, c))
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
                for c in 0..t_kv {
                    *w.at_mut(r, c) /= row_sum;
                }
            }
        }

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
    if (factor - 1.0).abs() < 1e-9 {
        return x.clone();
    }
    let scaled = x.data().scale(factor);
    let result = TensorNode::leaf(scaled);
    let x_c = x.clone();
    let result_c = result.clone();
    result.set_backward(
        Box::new(move || {
            let dout = result_c.grad().clone();
            let new_grad = x_c.grad().clone().add(&dout.scale(factor));
            x_c.set_grad(new_grad);
            x_c.call_backward_fn();
        }),
        vec![x.clone()],
    );
    result
}

// ============================================================================
// Minimal LCG RNG (copied from transformer3 pattern)
// ============================================================================

struct LcgRng {
    state: u64,
}
impl LcgRng {
    fn new(seed: u64) -> Self {
        LcgRng {
            state: seed.wrapping_add(1),
        }
    }
    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
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
        for s in &mut scores {
            *s /= params.temperature;
        }
    }

    // Softmax
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs {
        *p /= sum;
    }

    // Top-k
    if params.top_k > 0 && params.top_k < v {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for i in params.top_k..v {
            probs[indexed[i].0] = 0.0;
        }
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
    }

    // Top-p (nucleus sampling): keep the smallest set of tokens whose
    // cumulative probability exceeds top_p.  Applied after top-k so we
    // work on an already-truncated distribution.
    if params.top_p > 0.0 && params.top_p < 1.0 {
        let mut indexed: Vec<(usize, f32)> = probs
            .iter()
            .cloned()
            .enumerate()
            .filter(|&(_, p)| p > 0.0)
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cumulative = 0.0f32;
        let mut cutoff = indexed.len(); // index after which we zero out
        for (i, &(_, p)) in indexed.iter().enumerate() {
            cumulative += p;
            if cumulative >= params.top_p {
                cutoff = i + 1;
                break;
            }
        }
        // Zero out tokens outside nucleus
        for i in cutoff..indexed.len() {
            probs[indexed[i].0] = 0.0;
        }
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }
    }

    // Greedy
    if params.temperature == 0.0 {
        return probs
            .iter()
            .cloned()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
    }

    // Sample
    let r = rng.next_f32();
    let mut cumulative = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r <= cumulative {
            return i;
        }
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
            rope_theta_local: 10_000.0,
            rope_theta_global: 1_000_000.0,
            rms_norm_eps: 1e-6,
            query_pre_attn_scalar: 16.0, // matches head_dim for this tiny cfg
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
            assert_eq!(
                cfg.is_global_layer(i),
                expected_global,
                "layer {}: is_global should be {}",
                i,
                expected_global
            );
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
        assert_eq!(
            d.cols, cfg.vocab_size,
            "logits cols should equal vocab_size"
        );
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
                assert!(
                    d.at(r, c).is_finite(),
                    "logits[{},{}] = {} is not finite",
                    r,
                    c,
                    d.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_loss_tokens_finite_and_positive() {
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(3);
        let model = Gemma3Model::new(cfg.clone(), &mut rng);
        let token_ids = vec![1usize, 3, 5, 2];
        let targets = vec![3usize, 5, 2, 0];
        for p in model.parameters() {
            p.zero_grad();
        }
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
        let targets = vec![1usize, 2, 3];
        for p in model.parameters() {
            p.zero_grad();
        }
        let loss = model.loss_tokens(&token_ids, &targets);
        loss.backward();
        // embed_tokens should have non-zero gradient on the rows we looked up
        let eg = model.embed_tokens.grad();
        let any_nonzero = (0..eg.rows).any(|r| (0..eg.cols).any(|c| eg.at(r, c) != 0.0));
        assert!(
            any_nonzero,
            "embed_tokens.grad should be non-zero after backward"
        );
    }

    #[test]
    fn test_attention_local_vs_global_different() {
        // Local (layer 0) and global (layer 5) attention should behave differently
        // on the same input because local uses a window mask.
        let cfg = tiny_cfg();
        let mut rng = InitRng::new(5);
        let attn_local = Gemma3Attention::new(&cfg, 0, &mut rng); // local
        let attn_global = Gemma3Attention::new(&cfg, 5, &mut rng); // global
        assert!(
            attn_local.sliding_window.is_some(),
            "layer 0 should be local"
        );
        assert!(
            attn_global.sliding_window.is_none(),
            "layer 5 should be global"
        );
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
        assert!(
            params.len() > cfg.num_hidden_layers * 5,
            "expected many parameter tensors, got {}",
            params.len()
        );
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
            cfg.num_key_value_heads,
            cfg.head_dim,
            cfg.max_position_embeddings,
        );
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
        model.generate_cached_streaming(&[0usize, 1, 2], 5, 1.0, 0, 1.0, 1.0, 42, |tok| {
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
        let ldata = logits.data().clone();
        let t = ldata.rows;
        let v = ldata.cols;
        let uncached = (0..v)
            .max_by(|&a, &b| ldata.at(t - 1, a).partial_cmp(&ldata.at(t - 1, b)).unwrap())
            .unwrap();

        // Cached: greedy (temperature=0)
        let mut cached_tok = usize::MAX;
        model.generate_cached_streaming(&prompt, 1, 0.0, 0, 1.0, 1.0, 0, |tok| {
            cached_tok = tok;
        });

        assert_eq!(
            uncached, cached_tok,
            "cached first token {} must match non-cached {}",
            cached_tok, uncached
        );
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
        model.generate_cached_streaming(&prompt, 1, 0.0, 0, 1.0, 1.0, 0, |t| toks.push(t));
        assert_eq!(toks.len(), 1);
        assert!(toks[0] < cfg.vocab_size);
    }
}
