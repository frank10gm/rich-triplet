#![allow(dead_code)]
// =============================================================================
// T5 v1.1 -- encoder only
// =============================================================================
//
// FLUX conditions on two text encoders. This is the big one: T5-XXL, 4.7B
// parameters of encoder, producing a [256, 4096] sequence embedding that
// carries essentially all of the prompt's meaning. (The CLIP encoder next door
// contributes a single pooled vector.)
//
// It is an ordinary pre-norm transformer with three things that are not
// ordinary, all of them silent when wrong:
//
// ## 1. Attention scores are not scaled
//
// There is no `1 / sqrt(d_kv)`. T5 folds that factor into the initialisation
// of the query projection instead, so applying it here divides every score by
// 8 and the softmax comes out far too flat. The embeddings stay finite,
// smooth, and plausible; the image they condition just comes out generic --
// prompt-shaped but not prompt-specific. It is the single most expensive
// mistake in this file.
//
// ## 2. Relative position bias lives in layer 0 only
//
// T5 has no positional embedding and no RoPE. Instead the first layer owns a
// learned `[n_heads, 32]` table, indexed by a logarithmic bucketing of
// `key_pos - query_pos`, and the resulting `[n_heads, T, T]` bias is added to
// the scores of **every** layer. Recomputing it per layer from each layer's
// own weights fails loudly (the tensors do not exist); computing it once and
// forgetting to add it to the later layers does not.
//
// ## 3. The FFN is gated, and the gate uses the tanh GELU
//
//   h = gelu_tanh(x @ wi_0^T) * (x @ wi_1^T)
//   y = h @ wo^T
//
// T5 v1.0 has a single `wi` and a ReLU. v1.1 -- which is what the XXL
// checkpoint FLUX uses is -- has the gated pair. The two are not
// interchangeable and the tensor names differ, so this fails loudly.
//
// ## Norms
//
// RMSNorm, no bias, no mean subtraction, and the reciprocal square root is
// computed in f32 even when the weights are BF16 -- at `d_model = 4096` the
// sum of squares overflows half precision on ordinary activations.
//
// ## Tokenizer
//
// SentencePiece unigram, 32128 pieces, `</s>` appended and no BOS.
// `tokenizer.rs` already has the Viterbi decoder this needs.

use std::sync::Arc;

use crate::autograd2::{Mat, MatBf16};
use crate::conv2d::gelu_tanh_inplace;
use crate::gguf_loader::{gguf_tensor_to_f32, GgufFile, GgufMetaValue, GgufType};
use crate::qlinear::QLinear;
use crate::tokenizer::SentencePieceTokenizer;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_ff: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub d_kv: usize,
    /// Buckets in the relative position table. Half are exact, half logarithmic.
    pub rel_attn_buckets: usize,
    /// Beyond this distance every offset falls in the last bucket.
    pub rel_attn_max_distance: usize,
    pub rms_norm_eps: f32,
}

impl Default for T5Config {
    fn default() -> Self {
        T5Config {
            vocab_size: 32128,
            d_model: 4096,
            d_ff: 10240,
            n_layers: 24,
            n_heads: 64,
            d_kv: 64,
            rel_attn_buckets: 32,
            rel_attn_max_distance: 128,
            rms_norm_eps: 1e-6,
        }
    }
}

impl T5Config {
    /// google/t5-v1_1-xxl, which is what FLUX and SD3 both use.
    pub fn xxl() -> Self {
        T5Config { vocab_size: 32128, d_model: 4096, d_ff: 10240, n_layers: 24, n_heads: 64, d_kv: 64, ..Default::default() }
    }

    /// google/t5-v1_1-xl, for a smaller test path.
    pub fn xl() -> Self {
        T5Config { vocab_size: 32128, d_model: 2048, d_ff: 5120, n_layers: 24, n_heads: 32, d_kv: 64, ..Default::default() }
    }
}

// =============================================================================
// Norm
// =============================================================================

/// RMSNorm with no bias and no mean subtraction, accumulating in f64.
pub fn t5_rms_norm(x: &Mat, weight: &[f32], eps: f32) -> Mat {
    debug_assert!(weight.len() == x.cols, "t5_rms_norm: weight length != d_model");
    let cols = x.cols;
    let mut out = Mat::zeros(x.rows, cols);
    for r in 0..x.rows {
        let src = &x.data[r * cols..(r + 1) * cols];
        // f64 accumulation: at d_model 4096 the sum of squares of ordinary
        // activations is large enough that f32 loses digits off the end.
        let mut sq = 0.0f64;
        for c in 0..cols {
            sq += src[c] as f64 * src[c] as f64;
        }
        let inv = (1.0 / (sq / cols as f64 + eps as f64).sqrt()) as f32;
        let dst = &mut out.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dst[c] = src[c] * inv * weight[c];
        }
    }
    out
}

// =============================================================================
// Relative position bias
// =============================================================================

/// Map `key_pos - query_pos` onto a bucket index.
///
/// Bidirectional: the sign picks the half of the table, so `num_buckets` is
/// halved first. Within a half, the first quarter of the range is exact and
/// the rest is logarithmic out to `max_distance`.
///
/// The encoder is bidirectional, which is the case that uses both halves. The
/// unidirectional variant clamps negatives to zero instead, and using it here
/// makes every backwards offset collide onto bucket 0 -- the prompt still
/// encodes, it just stops distinguishing word order at range.
pub fn t5_relative_bucket(relative_position: i64, num_buckets: usize, max_distance: usize) -> usize {
    // Bidirectional: the sign selects the half of the table.
    let mut bucket = 0usize;
    let num_buckets = num_buckets / 2;
    if relative_position > 0 {
        bucket = num_buckets;
    }
    let n = relative_position.unsigned_abs() as usize;

    let max_exact = num_buckets / 2;
    if n < max_exact {
        return bucket + n;
    }
    // Logarithmic beyond `max_exact`, saturating at the last bucket.
    let ratio = (n as f64 / max_exact as f64).ln() / (max_distance as f64 / max_exact as f64).ln();
    let scaled = (max_exact as f64 + ratio * (num_buckets - max_exact) as f64) as usize;
    bucket + scaled.min(num_buckets - 1)
}

/// Build the `[n_heads, T, T]` bias, flattened to `[n_heads * T, T]`.
///
/// `table` is the layer-0 `relative_attention_bias.weight`, [num_buckets,
/// n_heads] as stored -- an nn.Embedding, so buckets are the rows.
pub fn t5_position_bias(table: &Mat, seq_len: usize, n_heads: usize, num_buckets: usize, max_distance: usize) -> Mat {
    debug_assert!(table.cols == n_heads, "t5_position_bias: table.cols != n_heads");
    let mut bias = Mat::zeros(n_heads * seq_len, seq_len);
    for q in 0..seq_len {
        for k in 0..seq_len {
            // `memory_position - query_position`, in that order. The other
            // order mirrors the bias and makes the encoder read backwards.
            let rel = k as i64 - q as i64;
            let bucket = t5_relative_bucket(rel, num_buckets, max_distance);
            for h in 0..n_heads {
                *bias.at_mut(h * seq_len + q, k) = table.at(bucket, h);
            }
        }
    }
    bias
}

// =============================================================================
// Layers
// =============================================================================

#[derive(Clone, Default)]
pub struct T5Attention {
    pub q: QLinear,
    pub k: QLinear,
    pub v: QLinear,
    pub o: QLinear,
    pub n_heads: usize,
    pub d_kv: usize,
}

impl T5Attention {
    /// `bias` is [n_heads * T, T] from `t5_position_bias`, shared by every
    /// layer. Scores are **not** divided by sqrt(d_kv).
    pub fn forward(&self, x: &Mat, bias: &Mat) -> Mat {
        let t = x.rows;
        let inner = self.n_heads * self.d_kv;
        let d_kv = self.d_kv;

        let q = self.q.forward(x);
        let k = self.k.forward(x);
        let v = self.v.forward(x);
        debug_assert!(q.cols == inner, "t5 attention: projection width != n_heads * d_kv");

        let mut context = Mat::zeros(t, inner);
        let mut scores = vec![0.0f32; t];

        for h in 0..self.n_heads {
            let off = h * d_kv;
            for i in 0..t {
                let qi = &q.data[i * q.cols + off..i * q.cols + off + d_kv];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..t {
                    let kj = &k.data[j * k.cols + off..j * k.cols + off + d_kv];
                    let mut acc = 0.0f32;
                    for c in 0..d_kv {
                        acc += qi[c] * kj[c];
                    }
                    // No 1/sqrt(d_kv). T5 folds it into the query
                    // initialisation, and dividing here flattens every softmax
                    // in the model.
                    acc += bias.at(h * t + i, j);
                    scores[j] = acc;
                    if max_score < acc {
                        max_score = acc;
                    }
                }
                let mut denom = 0.0f32;
                for j in 0..t {
                    scores[j] = (scores[j] - max_score).exp();
                    denom += scores[j];
                }
                let inv = 1.0f32 / denom;
                let out = &mut context.data[i * inner + off..i * inner + off + d_kv];
                for j in 0..t {
                    let weight = scores[j] * inv;
                    let vj = &v.data[j * v.cols + off..j * v.cols + off + d_kv];
                    for c in 0..d_kv {
                        out[c] += weight * vj[c];
                    }
                }
            }
        }
        self.o.forward(&context)
    }
}

#[derive(Clone, Default)]
pub struct T5FeedForward {
    pub wi_0: QLinear, // gate
    pub wi_1: QLinear, // up
    pub wo: QLinear,
}

impl T5FeedForward {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut gate = self.wi_0.forward(x);
        gelu_tanh_inplace(&mut gate);
        let up = self.wi_1.forward(x);
        for (g, u) in gate.data.iter_mut().zip(&up.data) {
            *g *= *u;
        }
        self.wo.forward(&gate)
    }
}

#[derive(Clone, Default)]
pub struct T5Block {
    pub norm1: Vec<f32>, // RMSNorm weight, [d_model]
    pub attn: T5Attention,
    pub norm2: Vec<f32>,
    pub ff: T5FeedForward,
}

impl T5Block {
    pub fn forward(&self, x: &Mat, bias: &Mat, eps: f32) -> Mat {
        let mut h = self.attn.forward(&t5_rms_norm(x, &self.norm1, eps), bias);
        for (d, s) in h.data.iter_mut().zip(&x.data) {
            *d += *s;
        }
        let mut f = self.ff.forward(&t5_rms_norm(&h, &self.norm2, eps));
        for (d, s) in f.data.iter_mut().zip(&h.data) {
            *d += *s;
        }
        f
    }
}

// =============================================================================
// Encoder
// =============================================================================

#[derive(Clone)]
pub struct T5Encoder {
    pub cfg: T5Config,

    /// [vocab_size, d_model]. Held as f32; at XXL this is 132 M parameters
    /// and only `T` of its rows are ever read, so it is gathered on the CPU and
    /// never quantized.
    pub token_embedding: Mat,
    /// Layer 0's relative attention bias table, [num_buckets, n_heads].
    pub rel_bias_table: Mat,
    pub blocks: Vec<T5Block>,
    pub final_norm: Vec<f32>,
}

impl Default for T5Encoder {
    fn default() -> Self {
        T5Encoder {
            cfg: T5Config::default(),
            token_embedding: Mat::zeros(0, 0),
            rel_bias_table: Mat::zeros(0, 0),
            blocks: Vec::new(),
            final_norm: Vec::new(),
        }
    }
}

// -----------------------------------------------------------------------------
// Loading
// -----------------------------------------------------------------------------

fn join_names(names: &[String]) -> String {
    names.join(", ")
}

fn io_err(e: std::io::Error) -> String {
    e.to_string()
}

/// GGUF stores a 2-D tensor's dimensions fastest-varying first, so a
/// `[out, in]` PyTorch weight has `ne = {in, out}`. Everything here wants the
/// PyTorch reading.
fn load_linear(gguf: &GgufFile, name: &str, out_features: usize, in_features: usize) -> Result<QLinear, String> {
    let idx = gguf.find_tensor(name).ok_or_else(|| format!("t5: missing tensor {}", name))?;
    let info = &gguf.tensor_info[idx];
    if info.n_elements() != out_features * in_features {
        return Err(format!(
            "t5: {} has {} elements, expected {}",
            name,
            info.n_elements(),
            out_features * in_features
        ));
    }

    match info.gguf_type {
        GgufType::Q4K => {
            // The only format kept packed. `decode_q4k_to_q4kmat` also flips
            // GGUF's [in, out] listing to the [out, in] this wants.
            let q = gguf.decode_q4k_to_q4kmat(idx).map_err(io_err)?;
            Ok(QLinear::from_q4k(q, out_features, in_features, Vec::new()))
        }
        GgufType::Bf16 => {
            let bits = gguf.decode_bf16(idx).map_err(io_err)?;
            Ok(QLinear::from_bf16(
                MatBf16 { data: Arc::new(bits), rows: out_features, cols: in_features },
                Vec::new(),
            ))
        }
        GgufType::F32 | GgufType::F16 => {
            // Kept at full precision. Widening f16 to f32 is exact, and
            // folding it down to bfloat instead would throw away two mantissa
            // bits for no saving worth having.
            let f = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
            Ok(QLinear::from_f32(Mat::new(f, out_features, in_features), Vec::new()))
        }
        _ => {
            // Q6_K, Q8_0 and Q5_K land here: decode to f32 and fold to BF16,
            // which halves the resident cost and throws away less than the
            // source format already did.
            let f = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
            let m = Mat::new(f, out_features, in_features);
            Ok(QLinear::from_bf16(m.to_bf16(), Vec::new()))
        }
    }
}

fn load_vec(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let idx = gguf.find_tensor(name).ok_or_else(|| format!("t5: missing tensor {}", name))?;
    let v = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
    if v.len() != len {
        return Err(format!("t5: {} has {} elements, expected {}", name, v.len(), len));
    }
    Ok(v)
}

fn load_mat(gguf: &GgufFile, name: &str, rows: usize, cols: usize) -> Result<Mat, String> {
    let idx = gguf.find_tensor(name).ok_or_else(|| format!("t5: missing tensor {}", name))?;
    let v = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
    if v.len() != rows * cols {
        return Err(format!("t5: {} has {} elements, expected {}", name, v.len(), rows * cols));
    }
    Ok(Mat::new(v, rows, cols))
}

/// Resolve a tensor by trying each name in turn.
///
/// Two naming conventions are in the wild for the same weights. The
/// distributed encoder GGUFs are converted by llama.cpp, which normalises
/// everything to its own scheme -- `enc.blk.3.attn_q.weight`. A file converted
/// straight from the HuggingFace checkpoint keeps T5's own names --
/// `encoder.block.3.layer.0.SelfAttention.q.weight`. Both are accepted, and
/// the error names every candidate so a third convention is easy to add.
fn load_linear_any(gguf: &GgufFile, names: &[String], out_features: usize, in_features: usize) -> Result<QLinear, String> {
    for n in names {
        if gguf.find_tensor(n).is_some() {
            return load_linear(gguf, n, out_features, in_features);
        }
    }
    Err(format!("t5: none of these tensors exist: {}", join_names(names)))
}

fn load_vec_any(gguf: &GgufFile, names: &[String], len: usize) -> Result<Vec<f32>, String> {
    for n in names {
        if gguf.find_tensor(n).is_some() {
            return load_vec(gguf, n, len);
        }
    }
    Err(format!("t5: none of these tensors exist: {}", join_names(names)))
}

fn load_mat_any(gguf: &GgufFile, names: &[String], rows: usize, cols: usize) -> Result<Mat, String> {
    for n in names {
        if gguf.find_tensor(n).is_some() {
            return load_mat(gguf, n, rows, cols);
        }
    }
    Err(format!("t5: none of these tensors exist: {}", join_names(names)))
}

/// llama.cpp's prefix for block `i`, and T5's own.
fn llama_prefix(i: usize) -> String {
    format!("enc.blk.{}.", i)
}
fn hf_prefix(i: usize) -> String {
    format!("encoder.block.{}.layer.", i)
}

fn names2(a: String, b: String) -> Vec<String> {
    vec![a, b]
}

impl T5Encoder {
    /// Load from a GGUF encoder checkpoint (`city96/t5-v1_1-xxl-encoder-gguf`
    /// and friends), which stores the weights under llama.cpp's or the
    /// original T5 names.
    pub fn load_gguf(path: &str, cfg: T5Config) -> Result<T5Encoder, String> {
        let gguf = GgufFile::open(path).map_err(|e| format!("gguf: cannot open {}: {}", path, e))?;

        let mut e = T5Encoder { cfg: cfg.clone(), ..Default::default() };

        e.token_embedding = load_mat_any(
            &gguf,
            &names2("token_embd.weight".into(), "shared.weight".into()),
            cfg.vocab_size,
            cfg.d_model,
        )?;

        // The bias table is stored [buckets, n_heads] -- an nn.Embedding whose
        // rows are the buckets. GGUF lists that as ne = {n_heads, buckets}; the
        // flat buffer is the same either way.
        e.rel_bias_table = load_mat_any(
            &gguf,
            &names2(
                "enc.blk.0.attn_rel_b.weight".into(),
                "encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight".into(),
            ),
            cfg.rel_attn_buckets,
            cfg.n_heads,
        )?;

        let inner = cfg.n_heads * cfg.d_kv;
        e.blocks.reserve(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let lp = llama_prefix(i);
            let hp = hf_prefix(i);
            let mut b = T5Block::default();

            b.norm1 = load_vec_any(&gguf, &names2(format!("{lp}attn_norm.weight"), format!("{hp}0.layer_norm.weight")), cfg.d_model)?;

            b.attn.q = load_linear_any(&gguf, &names2(format!("{lp}attn_q.weight"), format!("{hp}0.SelfAttention.q.weight")), inner, cfg.d_model)?;
            b.attn.k = load_linear_any(&gguf, &names2(format!("{lp}attn_k.weight"), format!("{hp}0.SelfAttention.k.weight")), inner, cfg.d_model)?;
            b.attn.v = load_linear_any(&gguf, &names2(format!("{lp}attn_v.weight"), format!("{hp}0.SelfAttention.v.weight")), inner, cfg.d_model)?;
            b.attn.o = load_linear_any(&gguf, &names2(format!("{lp}attn_o.weight"), format!("{hp}0.SelfAttention.o.weight")), cfg.d_model, inner)?;
            b.attn.n_heads = cfg.n_heads;
            b.attn.d_kv = cfg.d_kv;

            b.norm2 = load_vec_any(&gguf, &names2(format!("{lp}ffn_norm.weight"), format!("{hp}1.layer_norm.weight")), cfg.d_model)?;

            // v1.1's gated pair. llama.cpp calls the gated half `ffn_gate` and
            // the ungated one `ffn_up`, which is `wi_0` and `wi_1` in that
            // order -- and the order matters, because only the gate takes the
            // GELU. A v1.0 checkpoint has a single `wi` and fails here, which
            // is the outcome worth having.
            b.ff.wi_0 = load_linear_any(&gguf, &names2(format!("{lp}ffn_gate.weight"), format!("{hp}1.DenseReluDense.wi_0.weight")), cfg.d_ff, cfg.d_model)?;
            b.ff.wi_1 = load_linear_any(&gguf, &names2(format!("{lp}ffn_up.weight"), format!("{hp}1.DenseReluDense.wi_1.weight")), cfg.d_ff, cfg.d_model)?;
            b.ff.wo = load_linear_any(&gguf, &names2(format!("{lp}ffn_down.weight"), format!("{hp}1.DenseReluDense.wo.weight")), cfg.d_model, cfg.d_ff)?;

            e.blocks.push(b);
        }

        e.final_norm = load_vec_any(
            &gguf,
            &names2("enc.output_norm.weight".into(), "encoder.final_layer_norm.weight".into()),
            cfg.d_model,
        )?;

        Ok(e)
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    /// Encode token ids to `[T, d_model]`.
    ///
    /// The caller pads or truncates to the length the diffusion model expects:
    /// 256 for FLUX.1-schnell, 512 for dev. T5 itself has no length limit --
    /// it has no positional embedding to run out of.
    pub fn forward(&self, tokens: &[u32]) -> Result<Mat, String> {
        if tokens.is_empty() {
            return Err("t5 forward: empty token sequence".to_string());
        }
        if self.token_embedding.rows != self.cfg.vocab_size {
            return Err("t5 forward: embedding table not loaded".to_string());
        }

        let t = tokens.len();
        let d = self.cfg.d_model;
        let mut x = Mat::zeros(t, d);
        for (i, &tok) in tokens.iter().enumerate() {
            if tok as usize >= self.cfg.vocab_size {
                return Err(format!("t5 forward: token id {} out of range", tok));
            }
            let row = &self.token_embedding.data[tok as usize * d..(tok as usize + 1) * d];
            x.data[i * d..(i + 1) * d].copy_from_slice(row);
        }

        // Computed once from layer 0's table and reused by every layer. This is
        // the model's only source of positional information.
        let bias = t5_position_bias(
            &self.rel_bias_table,
            t,
            self.cfg.n_heads,
            self.cfg.rel_attn_buckets,
            self.cfg.rel_attn_max_distance,
        );

        for b in &self.blocks {
            x = b.forward(&x, &bias, self.cfg.rms_norm_eps);
        }
        Ok(t5_rms_norm(&x, &self.final_norm, self.cfg.rms_norm_eps))
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.token_embedding.numel() + self.rel_bias_table.numel() + self.final_norm.len();
        for b in &self.blocks {
            n += b.norm1.len() + b.norm2.len();
            n += b.attn.q.out_features * b.attn.q.in_features;
            n += b.attn.k.out_features * b.attn.k.in_features;
            n += b.attn.v.out_features * b.attn.v.in_features;
            n += b.attn.o.out_features * b.attn.o.in_features;
            n += b.ff.wi_0.out_features * b.ff.wi_0.in_features;
            n += b.ff.wi_1.out_features * b.ff.wi_1.in_features;
            n += b.ff.wo.out_features * b.ff.wo.in_features;
        }
        n
    }

    /// Release every weight. The prompt embedding is computed once, and 2.8 GB
    /// is worth reclaiming before a 12B transformer starts.
    pub fn free_weights(&mut self) {
        self.token_embedding = Mat::zeros(0, 0);
        self.rel_bias_table = Mat::zeros(0, 0);
        for b in self.blocks.iter_mut() {
            b.attn.q.free_weight();
            b.attn.k.free_weight();
            b.attn.v.free_weight();
            b.attn.o.free_weight();
            b.ff.wi_0.free_weight();
            b.ff.wi_1.free_weight();
            b.ff.wo.free_weight();
        }
        self.blocks.clear();
    }
}

/// Build the unigram tokenizer out of the checkpoint's own metadata.
///
/// The distributed encoder GGUFs carry `tokenizer.ggml.tokens` and
/// `tokenizer.ggml.scores`, which is a complete SentencePiece vocabulary. Using
/// it saves asking for a `spiece.model` that is already on disk in another
/// form -- and saves the chance of pairing a checkpoint with the wrong one.
pub fn load_t5_gguf_tokenizer(gguf: &GgufFile) -> Result<SentencePieceTokenizer, String> {
    let (tokens_v, scores_v) = match (gguf.metadata.get("tokenizer.ggml.tokens"), gguf.metadata.get("tokenizer.ggml.scores")) {
        (Some(t), Some(s)) => (t, s),
        _ => return Err("t5: the checkpoint carries no embedded tokenizer".to_string()),
    };
    let (tok_arr, score_arr) = match (tokens_v, scores_v) {
        (GgufMetaValue::Array(t), GgufMetaValue::Array(s)) => (t, s),
        _ => return Err("t5: the embedded tokenizer is not stored as arrays".to_string()),
    };

    let mut tokens = Vec::with_capacity(tok_arr.len());
    for v in tok_arr {
        match v.as_str() {
            Some(s) => tokens.push(s.to_string()),
            None => return Err("t5: a tokenizer entry is not a string".to_string()),
        }
    }
    let scores: Vec<f32> = score_arr.iter().map(|v| v.as_f32().unwrap_or(0.0)).collect();
    SentencePieceTokenizer::from_tokens_and_scores(tokens, &scores)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn spread(rows: usize, cols: usize, scale: f32, salt: usize) -> Mat {
        Mat::from_fn(rows, cols, |r, c| {
            let i = ((r * 131 + c * 37 + salt * 7919) % 251) as f32;
            ((i * 0.41).sin() * 0.7 + (i * 0.13).cos() * 0.3) * scale
        })
    }

    fn linear(out: usize, inp: usize, salt: usize) -> QLinear {
        QLinear::from_f32(spread(out, inp, 0.2, salt), Vec::new())
    }

    fn t5_tiny() -> T5Config {
        T5Config {
            vocab_size: 40,
            d_model: 16,
            d_ff: 32,
            n_layers: 2,
            n_heads: 4,
            d_kv: 4,
            rel_attn_buckets: 32,
            rel_attn_max_distance: 128,
            ..Default::default()
        }
    }

    fn make_t5(cfg: &T5Config) -> T5Encoder {
        let mut e = T5Encoder {
            cfg: cfg.clone(),
            token_embedding: spread(cfg.vocab_size, cfg.d_model, 1.0, 1),
            rel_bias_table: spread(cfg.rel_attn_buckets, cfg.n_heads, 0.5, 2),
            ..Default::default()
        };
        let inner = cfg.n_heads * cfg.d_kv;
        for i in 0..cfg.n_layers {
            let mut b = T5Block::default();
            b.norm1 = vec![1.0; cfg.d_model];
            b.norm2 = vec![1.0; cfg.d_model];
            b.attn.q = linear(inner, cfg.d_model, 10 + i * 8);
            b.attn.k = linear(inner, cfg.d_model, 11 + i * 8);
            b.attn.v = linear(inner, cfg.d_model, 12 + i * 8);
            b.attn.o = linear(cfg.d_model, inner, 13 + i * 8);
            b.attn.n_heads = cfg.n_heads;
            b.attn.d_kv = cfg.d_kv;
            b.ff.wi_0 = linear(cfg.d_ff, cfg.d_model, 14 + i * 8);
            b.ff.wi_1 = linear(cfg.d_ff, cfg.d_model, 15 + i * 8);
            b.ff.wo = linear(cfg.d_model, cfg.d_ff, 16 + i * 8);
            e.blocks.push(b);
        }
        e.final_norm = vec![1.0; cfg.d_model];
        e
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    // -------------------------------------------------------------------------
    // Relative position bias
    // -------------------------------------------------------------------------

    #[test]
    fn relative_bucket_splits_the_table_by_sign() {
        // Bidirectional: negative offsets take the low half, positive the high.
        assert_eq!(t5_relative_bucket(0, 32, 128), 0);
        assert_eq!(t5_relative_bucket(-1, 32, 128), 1);
        assert_eq!(t5_relative_bucket(-7, 32, 128), 7);
        assert_eq!(t5_relative_bucket(1, 32, 128), 17);
        assert_eq!(t5_relative_bucket(7, 32, 128), 23);
    }

    #[test]
    fn relative_bucket_is_exact_below_max_exact_and_logarithmic_above() {
        // num_buckets 32 halves to 16, so the first 8 offsets are exact.
        for n in 0..8i64 {
            assert_eq!(t5_relative_bucket(-n, 32, 128), n as usize);
        }
        // Beyond that, buckets grow slowly and never leave the half.
        let mut previous = 7;
        for n in 8..400i64 {
            let b = t5_relative_bucket(-n, 32, 128);
            assert!(b >= previous);
            assert!(b < 16);
            previous = b;
        }
        // Everything past max_distance saturates on the last bucket of the half.
        assert_eq!(t5_relative_bucket(-100000, 32, 128), 15);
        assert_eq!(t5_relative_bucket(100000, 32, 128), 31);
    }

    #[test]
    fn position_bias_is_not_symmetric() {
        // The bias distinguishes "before" from "after". A symmetric bias would
        // make the encoder blind to word order at range, which still produces
        // fluent embeddings.
        let table = spread(32, 4, 1.0, 30);
        let bias = t5_position_bias(&table, 6, 4, 32, 128);
        assert_eq!(bias.rows, 4 * 6);
        assert_eq!(bias.cols, 6);

        let mut asymmetric = false;
        for q in 0..6 {
            for k in 0..6 {
                if (bias.at(q, k) - bias.at(k, q)).abs() > 1e-6 {
                    asymmetric = true;
                }
            }
        }
        assert!(asymmetric);
    }

    #[test]
    fn position_bias_uses_key_minus_query_in_that_order() {
        // A table that is zero except in one bucket pins the sign convention:
        // with bucket 17 set (relative position +1), the bias must land one step
        // to the *right* of the diagonal.
        let mut table = Mat::zeros(32, 1);
        *table.at_mut(17, 0) = 1.0;
        let bias = t5_position_bias(&table, 4, 1, 32, 128);
        for q in 0..4 {
            for k in 0..4 {
                let expected = if k as i64 - q as i64 == 1 { 1.0 } else { 0.0 };
                assert!(approx(bias.at(q, k), expected, 1e-6));
            }
        }
    }

    #[test]
    fn position_bias_depends_on_the_head() {
        let table = spread(32, 4, 1.0, 31);
        let bias = t5_position_bias(&table, 5, 4, 32, 128);
        let mut differs = false;
        for q in 0..5 {
            for k in 0..5 {
                if (bias.at(q, k) - bias.at(5 + q, k)).abs() > 1e-6 {
                    differs = true;
                }
            }
        }
        assert!(differs);
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    #[test]
    fn rms_norm_normalizes_without_subtracting_the_mean() {
        // A constant row has zero variance but a nonzero RMS, so RMSNorm leaves
        // it at magnitude one rather than at zero. LayerNorm would zero it.
        let mut x = Mat::zeros(2, 8);
        for c in 0..8 {
            *x.at_mut(0, c) = 3.0;
            *x.at_mut(1, c) = -0.5;
        }
        let w = vec![1.0f32; 8];
        let out = t5_rms_norm(&x, &w, 1e-6);
        for c in 0..8 {
            assert!(approx(out.at(0, c), 1.0, 1e-4));
            assert!(approx(out.at(1, c), -1.0, 1e-4));
        }
    }

    #[test]
    fn the_encoder_produces_one_row_per_token() {
        let cfg = t5_tiny();
        let e = make_t5(&cfg);
        let tokens = [3u32, 11, 7, 29, 1];
        let out = e.forward(&tokens).unwrap();
        assert_eq!(out.rows, tokens.len());
        assert_eq!(out.cols, cfg.d_model);
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_encoder_is_bidirectional() {
        // Changing the last token must move the first token's embedding. A
        // causal encoder could not do that, and CLIP next door genuinely cannot.
        let e = make_t5(&t5_tiny());
        let mut tokens = vec![3u32, 11, 7, 29, 1];
        let base = e.forward(&tokens).unwrap();
        *tokens.last_mut().unwrap() = 22;
        let poked = e.forward(&tokens).unwrap();

        let mut delta = 0.0f32;
        for c in 0..base.cols {
            delta = delta.max((base.at(0, c) - poked.at(0, c)).abs());
        }
        assert!(delta > 1e-5);
    }

    #[test]
    fn the_encoder_depends_on_token_order() {
        let e = make_t5(&t5_tiny());
        let a = e.forward(&[3, 11, 7]).unwrap();
        let b = e.forward(&[7, 11, 3]).unwrap();
        assert!(max_delta(&a.data, &b.data) > 1e-4);
    }

    #[test]
    fn attention_scores_carry_no_inverse_sqrt_d_kv() {
        // Scale the q projection by sqrt(d_kv) and the output must be unchanged
        // *only if* the implementation divides by it. It does not, so the
        // outputs must differ -- which is the check that catches the scaling
        // being added.
        let cfg = t5_tiny();
        let plain = make_t5(&cfg);
        let mut scaled = make_t5(&cfg);
        let s = (cfg.d_kv as f32).sqrt();
        for b in scaled.blocks.iter_mut() {
            let mut w = b.attn.q.f32.clone();
            for v in w.data.iter_mut() {
                *v *= s;
            }
            b.attn.q = QLinear::from_f32(w, Vec::new());
        }

        let a = plain.forward(&[3, 11, 7, 29]).unwrap();
        let c = scaled.forward(&[3, 11, 7, 29]).unwrap();
        assert!(max_delta(&a.data, &c.data) > 1e-4);
    }

    #[test]
    fn the_encoder_rejects_out_of_range_tokens_and_empty_input() {
        let e = make_t5(&t5_tiny());
        assert!(e.forward(&[]).is_err());
        assert!(e.forward(&[0, 1, 999]).is_err());
    }

    #[test]
    fn the_xxl_config_is_the_one_flux_uses() {
        let c = T5Config::xxl();
        assert_eq!(c.d_model, 4096);
        assert_eq!(c.d_ff, 10240);
        assert_eq!(c.n_layers, 24);
        assert_eq!(c.n_heads, 64);
        assert_eq!(c.d_kv, 64);
        assert_eq!(c.vocab_size, 32128);
        // 64 heads of 64 is 4096 -- the inner width equals d_model here, which
        // is not true of every T5 size and is worth pinning.
        assert_eq!(c.n_heads * c.d_kv, c.d_model);
    }
}
