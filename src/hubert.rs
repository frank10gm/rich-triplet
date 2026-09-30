// =============================================================================
// HuBERT -- the codec's semantic encoder
// =============================================================================
//
// The eighth transformer here, and the first that is not a language model.
// HuBERT reads a raw waveform and produces one frame of features every 20 ms;
// OmniVoice's codec uses it as half of its analysis path, so that the codes it
// quantizes carry *what was said* and not only how it sounded.
//
// It is a wav2vec2-shaped encoder, which differs from every other transformer
// in this project in ways that all matter:
//
//   * **The input is audio, not tokens.** Seven strided convolutions turn
//     16 kHz samples into 768-wide frames at 50 Hz -- a 320x reduction, done
//     with no padding at all, so the length arithmetic is exact and unforgiving.
//   * **Position is a convolution, not a rotation.** There is no RoPE and no
//     learned table. A single depth-16 grouped convolution with a 128-wide
//     kernel is added to the hidden states once, before the first layer.
//   * **Attention is bidirectional and unmasked**, like the OmniVoice LM and
//     unlike everything else, so it reuses that kernel.
//   * **Post-layer-norm.** The norm comes *after* each residual add, which is
//     the original Transformer arrangement and the opposite of every other
//     model here. Swapping it does not crash; it quietly changes the features.
//   * **LayerNorm, not RMSNorm** -- mean subtraction included.
//   * **GELU is the erf form**, not the tanh approximation Gemma uses.
//
// ## What the codec actually asks for
//
// Not the last hidden state: the **mean of all thirteen**, the embedding output
// plus one per layer. That is what `mean_hidden_states` returns, and it is why
// this cannot be a normal `forward` that keeps only its final output.
//
// ## The length chain
//
// Every stage is a bare convolution with no padding, so lengths only shrink,
// and by exactly `floor((T - K) / S) + 1` each time:
//
//   16 kHz samples -> /5 -> /2 -> /2 -> /2 -> /2 -> /2 -> /2  = 320x
//
// which is 50 Hz. The codec then keeps every other frame to reach its own
// 25 Hz. Feeding it `n * 640 + 320` samples -- one second of 24 kHz audio
// resampled and padded -- yields exactly `2n` frames, and that identity is
// what the caller checks rather than trusting the chain.

#![allow(dead_code)]

use crate::autograd2::Mat;
use crate::conv1d::{conv1d_dense, conv1d_grouped};
use crate::gguf_loader::{GgufFile, load_gguf_conv_weight, load_gguf_vector};
use crate::transformer6::bidirectional_gqa_attention;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct HubertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,

    /// The convolutional feature extractor, one entry per layer.
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,

    /// The positional convolution: kernel width and group count.
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,

    pub layer_norm_eps: f32,
    /// nn.GroupNorm's default, and separate from the above -- they are
    /// different layers and PyTorch does not give them the same epsilon.
    pub group_norm_eps: f32,
}

impl Default for HubertConfig {
    fn default() -> Self {
        HubertConfig {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            conv_dim: vec![512, 512, 512, 512, 512, 512, 512],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            layer_norm_eps: 1e-5,
            group_norm_eps: 1e-5,
        }
    }
}

impl HubertConfig {
    pub fn omnivoice_semantic() -> Self {
        HubertConfig::default()
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Product of the convolution strides -- 320, so 16 kHz in, 50 Hz out.
    pub fn downsample_factor(&self) -> usize {
        let mut n = 1usize;
        for &s in &self.conv_stride {
            n *= s;
        }
        n
    }

    /// How many frames `samples` 16 kHz samples produce, or 0 if the stack
    /// cannot consume them.
    pub fn feature_frames(&self, samples: usize) -> usize {
        let mut t = samples;
        for i in 0..self.conv_kernel.len() {
            // No padding anywhere in the feature extractor, so a clip shorter than
            // the receptive field produces nothing rather than a short frame.
            if t < self.conv_kernel[i] {
                return 0;
            }
            t = (t - self.conv_kernel[i]) / self.conv_stride[i] + 1;
        }
        t
    }
}

// =============================================================================
// Normalisation and activation
// =============================================================================

/// LayerNorm each row of `x` in place: subtract the mean, divide by the
/// standard deviation, scale and shift.
pub fn layer_norm_rows(x: &mut Mat, weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert!(weight.len() == x.cols, "hubert: layer norm width mismatch");
    debug_assert!(bias.len() == x.cols, "hubert: layer norm bias width mismatch");
    let n = x.cols as f64;
    let cols = x.cols;

    for r in 0..x.rows {
        let row = &mut x.data[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for &v in row.iter() {
            sum += v as f64;
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for &v in row.iter() {
            let d = v as f64 - mean;
            var += d * d;
        }
        // Biased variance, which is what PyTorch's LayerNorm uses.
        let inv = (1.0 / (var / n + eps as f64).sqrt()) as f32;
        for c in 0..cols {
            row[c] = ((row[c] as f64 - mean) as f32) * inv * weight[c] + bias[c];
        }
    }
}

/// GroupNorm with one group per channel, over the time axis of a [T, C]
/// matrix. Equivalent to normalising each column on its own.
pub fn group_norm_channels(x: &mut Mat, weight: &[f32], bias: &[f32], eps: f32) {
    debug_assert!(weight.len() == x.cols, "hubert: group norm width mismatch");
    let n = x.rows as f64;
    if x.rows == 0 {
        return;
    }

    // One group per channel, so each column is normalised against its own
    // statistics over time -- the transpose of what LayerNorm does.
    for c in 0..x.cols {
        let mut sum = 0.0f64;
        for r in 0..x.rows {
            sum += x.at(r, c) as f64;
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for r in 0..x.rows {
            let d = x.at(r, c) as f64 - mean;
            var += d * d;
        }
        let inv = (1.0 / (var / n + eps as f64).sqrt()) as f32;
        for r in 0..x.rows {
            let v = ((x.at(r, c) as f64 - mean) as f32) * inv * weight[c] + bias[c];
            *x.at_mut(r, c) = v;
        }
    }
}

unsafe extern "C" {
    /// The C library's single-precision `erf` -- what C++'s `std::erf(float)`
    /// calls. `f32::erf` is still unstable, and a different implementation
    /// would not round the same way.
    fn erff(x: f32) -> f32;
}

/// The exact GELU, `x * 0.5 * (1 + erf(x / sqrt(2)))`.
///
/// Not `gelu_tanh`. The two agree to about 1e-3, which is invisible in a
/// language model's logits and is not what this checkpoint was trained with.
pub fn gelu_erf_inplace(x: &mut Mat) {
    const INV_SQRT2: f32 = 0.70710678118654752;
    for v in x.data.iter_mut() {
        // SAFETY: `erff` is a pure libm function with no preconditions.
        let e = unsafe { erff(*v * INV_SQRT2) };
        *v = 0.5f32 * *v * (1.0f32 + e);
    }
}

// =============================================================================
// Layers
// =============================================================================

/// One convolution of the feature extractor: convolve, normalise, GELU.
///
/// Only the first layer normalises. `feat_extract_norm="group"` puts a
/// GroupNorm with one group per channel -- so, per-channel over time -- on
/// layer 0 and nothing on the other six. A checkpoint carrying norms on every
/// layer would be the `"layer"` variant, which this is not.
#[derive(Clone, Debug)]
pub struct HubertFeatureLayer {
    /// [Cout, Cin * K]
    pub weight: Mat,
    pub kernel: usize,
    pub stride: usize,
    pub group_norm: bool,
    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
}

impl Default for HubertFeatureLayer {
    fn default() -> Self {
        HubertFeatureLayer {
            weight: Mat::zeros(0, 0),
            kernel: 1,
            stride: 1,
            group_norm: false,
            norm_weight: Vec::new(),
            norm_bias: Vec::new(),
        }
    }
}

impl HubertFeatureLayer {
    pub fn forward(&self, x: &Mat, eps: f32) -> Mat {
        let mut h = conv1d_dense(x, &self.weight, self.weight.rows, self.kernel, &[], 1, 0, self.stride);
        if self.group_norm {
            group_norm_channels(&mut h, &self.norm_weight, &self.norm_bias, eps);
        }
        gelu_erf_inplace(&mut h);
        h
    }
}

/// Post-layer-norm encoder block: attention, add, norm, feed-forward, add,
/// norm.
#[derive(Clone, Debug)]
pub struct HubertEncoderLayer {
    /// All four are [hidden, hidden] and all four carry a bias, unlike the
    /// language models here.
    pub q_weight: Mat,
    pub k_weight: Mat,
    pub v_weight: Mat,
    pub o_weight: Mat,
    pub q_bias: Vec<f32>,
    pub k_bias: Vec<f32>,
    pub v_bias: Vec<f32>,
    pub o_bias: Vec<f32>,

    pub attn_norm_weight: Vec<f32>,
    pub attn_norm_bias: Vec<f32>,

    /// [intermediate, hidden] then [hidden, intermediate].
    pub fc1_weight: Mat,
    pub fc2_weight: Mat,
    pub fc1_bias: Vec<f32>,
    pub fc2_bias: Vec<f32>,

    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
}

impl Default for HubertEncoderLayer {
    fn default() -> Self {
        HubertEncoderLayer {
            q_weight: Mat::zeros(0, 0),
            k_weight: Mat::zeros(0, 0),
            v_weight: Mat::zeros(0, 0),
            o_weight: Mat::zeros(0, 0),
            q_bias: Vec::new(),
            k_bias: Vec::new(),
            v_bias: Vec::new(),
            o_bias: Vec::new(),
            attn_norm_weight: Vec::new(),
            attn_norm_bias: Vec::new(),
            fc1_weight: Mat::zeros(0, 0),
            fc2_weight: Mat::zeros(0, 0),
            fc1_bias: Vec::new(),
            fc2_bias: Vec::new(),
            final_norm_weight: Vec::new(),
            final_norm_bias: Vec::new(),
        }
    }
}

/// `x @ weight^T + bias`, the ordinary affine layer.
fn affine(x: &Mat, weight: &Mat, bias: &[f32]) -> Mat {
    let mut out = x.matmul_bt(weight);
    if !bias.is_empty() {
        let cols = out.cols;
        for r in 0..out.rows {
            let row = &mut out.data[r * cols..(r + 1) * cols];
            for c in 0..cols {
                row[c] += bias[c];
            }
        }
    }
    out
}

impl HubertEncoderLayer {
    pub fn forward(&self, x: &Mat, cfg: &HubertConfig) -> Mat {
        let heads = cfg.num_attention_heads;
        let dim = cfg.head_dim();

        let q = affine(x, &self.q_weight, &self.q_bias);
        let k = affine(x, &self.k_weight, &self.k_bias);
        let v = affine(x, &self.v_weight, &self.v_bias);

        // Self-attention over the whole clip: no causal mask, and with one item in
        // the batch there is no padding to mask either.
        let context =
            bidirectional_gqa_attention(&q, &k, &v, heads, heads, dim, 1.0f32 / (dim as f32).sqrt());

        let mut h = affine(&context, &self.o_weight, &self.o_bias);
        h.add_assign(x);
        // Post-norm: after the residual add, not before the sublayer.
        layer_norm_rows(&mut h, &self.attn_norm_weight, &self.attn_norm_bias, cfg.layer_norm_eps);

        let mut ff = affine(&h, &self.fc1_weight, &self.fc1_bias);
        gelu_erf_inplace(&mut ff);
        let mut ff = affine(&ff, &self.fc2_weight, &self.fc2_bias);
        ff.add_assign(&h);
        layer_norm_rows(&mut ff, &self.final_norm_weight, &self.final_norm_bias, cfg.layer_norm_eps);
        ff
    }
}

// =============================================================================
// Model
// =============================================================================

#[derive(Clone, Debug)]
pub struct HubertModel {
    pub config: HubertConfig,

    pub feature_layers: Vec<HubertFeatureLayer>,

    /// `feature_projection`: normalise the 512-wide convolution output, then
    /// widen it to the model's 768.
    pub proj_norm_weight: Vec<f32>,
    pub proj_norm_bias: Vec<f32>,
    pub proj_weight: Mat,
    pub proj_bias: Vec<f32>,

    /// `encoder.pos_conv_embed.conv`: [hidden, (hidden / groups) * K].
    pub pos_conv_weight: Mat,
    pub pos_conv_bias: Vec<f32>,

    pub encoder_norm_weight: Vec<f32>,
    pub encoder_norm_bias: Vec<f32>,
    pub layers: Vec<HubertEncoderLayer>,
}

impl Default for HubertModel {
    fn default() -> Self {
        HubertModel {
            config: HubertConfig::default(),
            feature_layers: Vec::new(),
            proj_norm_weight: Vec::new(),
            proj_norm_bias: Vec::new(),
            proj_weight: Mat::zeros(0, 0),
            proj_bias: Vec::new(),
            pos_conv_weight: Mat::zeros(0, 0),
            pos_conv_bias: Vec::new(),
            encoder_norm_weight: Vec::new(),
            encoder_norm_bias: Vec::new(),
            layers: Vec::new(),
        }
    }
}

impl HubertModel {
    /// Run the convolutional front end: 16 kHz samples to [T, conv_dim.back()].
    pub fn extract_features(&self, samples: &[f32]) -> Result<Mat, String> {
        let frames = self.config.feature_frames(samples.len());
        if frames == 0 {
            return Err(format!(
                "hubert: {} samples are too few for the feature extractor",
                samples.len()
            ));
        }

        // The waveform enters as a one-channel [T, 1] sequence.
        let mut h = Mat::new(samples.to_vec(), samples.len(), 1);
        for layer in &self.feature_layers {
            h = layer.forward(&h, self.config.group_norm_eps);
        }
        if h.rows != frames {
            return Err(format!(
                "hubert: feature extractor produced {} frames where the length arithmetic says {}",
                h.rows, frames
            ));
        }
        Ok(h)
    }

    /// The mean of all `num_hidden_layers + 1` hidden states, [T, hidden_size].
    ///
    /// Averaging is the codec's choice, not HuBERT's: the early layers carry
    /// acoustic detail and the late ones carry phonetic identity, and the
    /// quantizer was fit on the average of both.
    pub fn mean_hidden_states(&self, samples: &[f32]) -> Result<Mat, String> {
        let mut features = self.extract_features(samples)?;

        // feature_projection: normalise the convolution output, then widen it.
        layer_norm_rows(
            &mut features,
            &self.proj_norm_weight,
            &self.proj_norm_bias,
            self.config.layer_norm_eps,
        );
        let mut h = affine(&features, &self.proj_weight, &self.proj_bias);

        // The positional convolution is "same" padded at kernel/2, which for an
        // even kernel leaves one sample too many. HuBERT drops the last one rather
        // than pad asymmetrically.
        let k = self.config.num_conv_pos_embeddings;
        let mut pos = conv1d_grouped(
            &h,
            &self.pos_conv_weight,
            self.config.hidden_size,
            k,
            &self.pos_conv_bias,
            self.config.num_conv_pos_embedding_groups,
            1,
            k / 2,
            1,
        );
        if k % 2 == 0 {
            if pos.rows != h.rows + 1 {
                return Err(format!(
                    "hubert: positional convolution produced {} frames for {} inputs",
                    pos.rows, h.rows
                ));
            }
            pos.data.truncate(h.rows * pos.cols);
            pos.rows = h.rows;
        }
        gelu_erf_inplace(&mut pos);
        h.add_assign(&pos);

        layer_norm_rows(
            &mut h,
            &self.encoder_norm_weight,
            &self.encoder_norm_bias,
            self.config.layer_norm_eps,
        );

        // The codec wants the average of every hidden state, so accumulate as we
        // go rather than keeping thirteen copies of a [T, 768] matrix alive.
        let mut sum = h.clone();
        for layer in &self.layers {
            h = layer.forward(&h, &self.config);
            sum.add_assign(&h);
        }
        let inv = 1.0f32 / (self.layers.len() + 1) as f32;
        for v in sum.data.iter_mut() {
            *v *= inv;
        }
        Ok(sum)
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = 0usize;
        let add_mat = |m: &Mat, n: &mut usize| *n += m.data.len();
        let add_vec = |v: &Vec<f32>, n: &mut usize| *n += v.len();

        for l in &self.feature_layers {
            add_mat(&l.weight, &mut n);
            add_vec(&l.norm_weight, &mut n);
            add_vec(&l.norm_bias, &mut n);
        }
        add_vec(&self.proj_norm_weight, &mut n);
        add_vec(&self.proj_norm_bias, &mut n);
        add_mat(&self.proj_weight, &mut n);
        add_vec(&self.proj_bias, &mut n);
        add_mat(&self.pos_conv_weight, &mut n);
        add_vec(&self.pos_conv_bias, &mut n);
        add_vec(&self.encoder_norm_weight, &mut n);
        add_vec(&self.encoder_norm_bias, &mut n);
        for l in &self.layers {
            add_mat(&l.q_weight, &mut n);
            add_mat(&l.k_weight, &mut n);
            add_mat(&l.v_weight, &mut n);
            add_mat(&l.o_weight, &mut n);
            add_vec(&l.q_bias, &mut n);
            add_vec(&l.k_bias, &mut n);
            add_vec(&l.v_bias, &mut n);
            add_vec(&l.o_bias, &mut n);
            add_vec(&l.attn_norm_weight, &mut n);
            add_vec(&l.attn_norm_bias, &mut n);
            add_mat(&l.fc1_weight, &mut n);
            add_mat(&l.fc2_weight, &mut n);
            add_vec(&l.fc1_bias, &mut n);
            add_vec(&l.fc2_bias, &mut n);
            add_vec(&l.final_norm_weight, &mut n);
            add_vec(&l.final_norm_bias, &mut n);
        }
        n
    }

    // =========================================================================
    // Loading
    // =========================================================================

    /// Load from an open GGUF under `prefix` (`"semantic_model"` in the
    /// OmniVoice tokenizer file).
    pub fn load(gguf: &GgufFile, cfg: HubertConfig, prefix: &str) -> Result<HubertModel, String> {
        if cfg.conv_dim.len() != cfg.conv_kernel.len() || cfg.conv_dim.len() != cfg.conv_stride.len()
        {
            return Err("hubert: conv_dim, conv_kernel and conv_stride disagree in length".into());
        }
        if cfg.num_attention_heads == 0 || cfg.hidden_size % cfg.num_attention_heads != 0 {
            return Err("hubert: hidden size does not divide into attention heads".into());
        }

        let mut m = HubertModel { config: cfg, ..Default::default() };
        let c = m.config.clone();

        // -- feature extractor ---------------------------------------------------
        let mut in_channels = 1usize;
        for i in 0..c.conv_dim.len() {
            let base = format!("{}.feature_extractor.conv_layers.{}", prefix, i);
            let mut layer = HubertFeatureLayer {
                kernel: c.conv_kernel[i],
                stride: c.conv_stride[i],
                ..Default::default()
            };
            layer.weight = load_linear(
                gguf,
                &format!("{}.conv.weight", base),
                c.conv_dim[i],
                in_channels * layer.kernel,
            )?;

            // Only the first layer carries a norm under `feat_extract_norm="group"`.
            // Deciding from the file rather than from a config flag means a
            // checkpoint of the other variant is a load error, not silent drift.
            if gguf.find_tensor(&format!("{}.layer_norm.weight", base)).is_some() {
                let nw = load_sized(gguf, &format!("{}.layer_norm.weight", base), c.conv_dim[i])?;
                let nb = load_sized(gguf, &format!("{}.layer_norm.bias", base), c.conv_dim[i])?;
                layer.group_norm = true;
                layer.norm_weight = nw;
                layer.norm_bias = nb;
            } else if i == 0 {
                return Err(format!(
                    "hubert: '{}.layer_norm.weight' is missing, so the first convolution has no \
                     group norm to apply",
                    base
                ));
            }
            in_channels = c.conv_dim[i];
            m.feature_layers.push(layer);
        }

        // -- feature projection --------------------------------------------------
        let feat_dim = *c.conv_dim.last().unwrap_or(&0);
        let pn_w = load_sized(gguf, &format!("{}.feature_projection.layer_norm.weight", prefix), feat_dim)?;
        let pn_b = load_sized(gguf, &format!("{}.feature_projection.layer_norm.bias", prefix), feat_dim)?;
        let p_w = load_linear(
            gguf,
            &format!("{}.feature_projection.projection.weight", prefix),
            c.hidden_size,
            feat_dim,
        )?;
        let p_b = load_sized(gguf, &format!("{}.feature_projection.projection.bias", prefix), c.hidden_size)?;
        m.proj_norm_weight = pn_w;
        m.proj_norm_bias = pn_b;
        m.proj_weight = p_w;
        m.proj_bias = p_b;

        // -- positional convolution ---------------------------------------------
        let groups = c.num_conv_pos_embedding_groups;
        if groups == 0 || c.hidden_size % groups != 0 {
            return Err("hubert: positional convolution groups do not divide the hidden size".into());
        }
        let pc_w = load_linear(
            gguf,
            &format!("{}.encoder.pos_conv_embed.conv.weight", prefix),
            c.hidden_size,
            (c.hidden_size / groups) * c.num_conv_pos_embeddings,
        )?;
        let pc_b = load_sized(gguf, &format!("{}.encoder.pos_conv_embed.conv.bias", prefix), c.hidden_size)?;
        m.pos_conv_weight = pc_w;
        m.pos_conv_bias = pc_b;

        let en_w = load_sized(gguf, &format!("{}.encoder.layer_norm.weight", prefix), c.hidden_size)?;
        let en_b = load_sized(gguf, &format!("{}.encoder.layer_norm.bias", prefix), c.hidden_size)?;
        m.encoder_norm_weight = en_w;
        m.encoder_norm_bias = en_b;

        // -- encoder layers ------------------------------------------------------
        m.layers.reserve(c.num_hidden_layers);
        for i in 0..c.num_hidden_layers {
            let base = format!("{}.encoder.layers.{}", prefix, i);
            let mut layer = HubertEncoderLayer::default();

            let proj = |name: &str| -> Result<(Mat, Vec<f32>), String> {
                let w = load_linear(
                    gguf,
                    &format!("{}.attention.{}.weight", base, name),
                    c.hidden_size,
                    c.hidden_size,
                )?;
                let b = load_sized(gguf, &format!("{}.attention.{}.bias", base, name), c.hidden_size)?;
                Ok((w, b))
            };
            (layer.q_weight, layer.q_bias) = proj("q_proj")?;
            (layer.k_weight, layer.k_bias) = proj("k_proj")?;
            (layer.v_weight, layer.v_bias) = proj("v_proj")?;
            (layer.o_weight, layer.o_bias) = proj("out_proj")?;

            let an_w = load_sized(gguf, &format!("{}.layer_norm.weight", base), c.hidden_size)?;
            let an_b = load_sized(gguf, &format!("{}.layer_norm.bias", base), c.hidden_size)?;
            layer.attn_norm_weight = an_w;
            layer.attn_norm_bias = an_b;

            let f1_w = load_linear(
                gguf,
                &format!("{}.feed_forward.intermediate_dense.weight", base),
                c.intermediate_size,
                c.hidden_size,
            )?;
            let f1_b = load_sized(
                gguf,
                &format!("{}.feed_forward.intermediate_dense.bias", base),
                c.intermediate_size,
            )?;
            let f2_w = load_linear(
                gguf,
                &format!("{}.feed_forward.output_dense.weight", base),
                c.hidden_size,
                c.intermediate_size,
            )?;
            let f2_b = load_sized(gguf, &format!("{}.feed_forward.output_dense.bias", base), c.hidden_size)?;
            layer.fc1_weight = f1_w;
            layer.fc1_bias = f1_b;
            layer.fc2_weight = f2_w;
            layer.fc2_bias = f2_b;

            let fn_w = load_sized(gguf, &format!("{}.final_layer_norm.weight", base), c.hidden_size)?;
            let fn_b = load_sized(gguf, &format!("{}.final_layer_norm.bias", base), c.hidden_size)?;
            layer.final_norm_weight = fn_w;
            layer.final_norm_bias = fn_b;

            m.layers.push(layer);
        }

        Ok(m)
    }
}

/// Read a `[out, in]` linear weight, checking both dimensions.
fn load_linear(gguf: &GgufFile, name: &str, out_dim: usize, in_dim: usize) -> Result<Mat, String> {
    let m = load_gguf_conv_weight(gguf, name, out_dim).map_err(|e| e.to_string())?;
    if m.cols != in_dim {
        return Err(format!(
            "hubert: '{}' is [{}, {}], expected [{}, {}]",
            name, m.rows, m.cols, out_dim, in_dim
        ));
    }
    Ok(m)
}

fn load_sized(gguf: &GgufFile, name: &str, n: usize) -> Result<Vec<f32>, String> {
    let v = load_gguf_vector(gguf, name).map_err(|e| e.to_string())?;
    if v.len() != n {
        return Err(format!("hubert: '{}' holds {} values, expected {}", name, v.len(), n));
    }
    Ok(v)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resample::resample;

    const CODEC_PATH: &str = "models/omnivoice-tokenizer-Q8_0.gguf";

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// A vowel-ish waveform: a fundamental plus two harmonics, slowly modulated.
    fn voice_like(n: usize, rate: f32) -> Vec<f32> {
        let pi = std::f32::consts::PI;
        let mut out = vec![0.0f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let t = i as f32 / rate;
            let env = 0.6f32 + 0.4f32 * (2.0f32 * pi * 3.0f32 * t).sin();
            *o = 0.2f32
                * env
                * ((2.0f32 * pi * 140.0f32 * t).sin()
                    + 0.5f32 * (2.0f32 * pi * 700.0f32 * t).sin()
                    + 0.25f32 * (2.0f32 * pi * 2400.0f32 * t).sin());
        }
        out
    }

    fn load_model() -> Option<HubertModel> {
        if !std::path::Path::new(CODEC_PATH).exists() {
            eprintln!("skip: no OmniVoice tokenizer checkpoint in models/");
            return None;
        }
        let gguf = GgufFile::open(CODEC_PATH).expect("open tokenizer GGUF");
        Some(
            HubertModel::load(&gguf, HubertConfig::omnivoice_semantic(), "semantic_model")
                .expect("load hubert"),
        )
    }

    // =========================================================================
    // Length arithmetic
    // =========================================================================

    #[test]
    fn the_feature_extractor_reduces_16_khz_by_exactly_320() {
        let c = HubertConfig::omnivoice_semantic();
        assert_eq!(c.downsample_factor(), 320);
        // 5 * 2^6.
        assert_eq!(c.conv_stride.len(), 7);
        assert_eq!(c.conv_kernel.len(), 7);
        assert_eq!(c.conv_dim.len(), 7);
    }

    #[test]
    fn a_padded_second_of_audio_makes_exactly_fifty_frames() {
        // The identity the codec depends on: n frames of 24 kHz audio become
        // n * 640 samples at 16 kHz, are padded by 160 on each side, and come out
        // as exactly 2n frames -- twice the codec's rate, so keeping every other
        // one lands on n.
        let c = HubertConfig::omnivoice_semantic();
        for frames in [1usize, 2, 25, 100, 251] {
            let samples = frames * 640 + 320;
            assert_eq!(c.feature_frames(samples), 2 * frames);
        }
    }

    #[test]
    fn feature_frames_follows_the_convolution_chain() {
        let c = HubertConfig::omnivoice_semantic();
        // Worked by hand: (960-10)/5+1 = 191, then /2 five more times and /2 again.
        assert_eq!(c.feature_frames(960), 2);
        assert_eq!(c.feature_frames(16000), 49);
        // Too short for the stack to consume at all.
        assert_eq!(c.feature_frames(0), 0);
        assert_eq!(c.feature_frames(9), 0);
        assert_eq!(c.feature_frames(100), 0);
    }

    // =========================================================================
    // Normalisation
    // =========================================================================

    #[test]
    fn layer_norm_rows_normalises_each_row() {
        let mut x = Mat::from_fn(3, 4, |r, c| (r * 10 + c) as f32);
        let gamma = vec![1.0f32; 4];
        let beta = vec![0.0f32; 4];
        layer_norm_rows(&mut x, &gamma, &beta, 1e-5);

        for r in 0..x.rows {
            let mut mean = 0.0f32;
            for c in 0..x.cols {
                mean += x.at(r, c);
            }
            assert!((mean / 4.0).abs() < 1e-5);
            let mut var = 0.0f32;
            for c in 0..x.cols {
                var += x.at(r, c) * x.at(r, c);
            }
            // Biased variance, as PyTorch computes it.
            assert!(approx(var / 4.0, 1.0));
        }
        // Every row held the same spread, so every row normalises identically.
        for c in 0..x.cols {
            assert!(approx(x.at(0, c), x.at(2, c)));
        }
    }

    #[test]
    fn layer_norm_rows_applies_gamma_and_beta() {
        let mut x = Mat::from_fn(1, 4, |_, c| c as f32);
        let gamma = vec![2.0f32; 4];
        let beta = vec![1.0f32; 4];
        layer_norm_rows(&mut x, &gamma, &beta, 1e-5);
        let mut mean = 0.0f32;
        for c in 0..x.cols {
            mean += x.at(0, c);
        }
        // Scaled by 2 and shifted by 1, so the mean is the shift.
        assert!(approx(mean / 4.0, 1.0));
    }

    #[test]
    fn group_norm_channels_normalises_down_the_columns() {
        // The transpose of LayerNorm: each channel against its own history. A
        // channel that never moves comes out at exactly the bias.
        let mut x = Mat::from_fn(8, 3, |r, c| {
            if c == 2 { 5.0 } else { r as f32 * if c == 0 { 1.0 } else { -2.0 } }
        });
        let gamma = vec![1.0f32, 1.0, 1.0];
        let beta = vec![0.0f32, 0.0, 0.5];
        group_norm_channels(&mut x, &gamma, &beta, 1e-5);

        for c in 0..2 {
            let mut mean = 0.0f32;
            let mut var = 0.0f32;
            for r in 0..x.rows {
                mean += x.at(r, c);
            }
            assert!((mean / 8.0).abs() < 1e-4);
            for r in 0..x.rows {
                var += x.at(r, c) * x.at(r, c);
            }
            assert!(approx(var / 8.0, 1.0));
        }
        for r in 0..x.rows {
            assert!(approx(x.at(r, 2), 0.5));
        }
    }

    // =========================================================================
    // Activation
    // =========================================================================

    #[test]
    fn gelu_erf_is_the_exact_gelu_not_the_tanh_approximation() {
        let mut x = Mat::new(vec![0.0, 1.0, -1.0, 2.0, 0.5, -0.5, 3.0], 1, 7);
        gelu_erf_inplace(&mut x);
        assert!(approx(x.at(0, 0), 0.0));
        assert!(approx(x.at(0, 1), 0.841344746));
        assert!(approx(x.at(0, 2), -0.158655254));
        assert!(approx(x.at(0, 3), 1.95449974));
        assert!(approx(x.at(0, 4), 0.345731231));
        assert!(approx(x.at(0, 5), -0.154268769));
        assert!(approx(x.at(0, 6), 2.99595031));

        // The tanh form differs by about 1.5e-4 at x = 1 -- invisible in a
        // language model's logits, and not what this checkpoint was trained with.
        let tanh_at_one = 0.5f32
            * (1.0f32 + ((2.0f32 / std::f32::consts::PI).sqrt() * (1.0f32 + 0.044715f32)).tanh());
        assert!((x.at(0, 1) - tanh_at_one).abs() > 1e-5);
        assert!((x.at(0, 1) - tanh_at_one).abs() < 1e-3);
    }

    // =========================================================================
    // The real checkpoint
    // =========================================================================

    #[test]
    fn hubert_model_loads_from_the_tokenizer_gguf() {
        let Some(m) = load_model() else { return };

        assert_eq!(m.feature_layers.len(), 7);
        assert_eq!(m.layers.len(), 12);
        // Only the first convolution carries a norm, which is what
        // feat_extract_norm="group" means.
        assert!(m.feature_layers[0].group_norm);
        for i in 1..m.feature_layers.len() {
            assert!(!m.feature_layers[i].group_norm);
        }
        // HuBERT base, to the nearest hundred thousand.
        assert!(m.parameter_count() > 94_000_000);
        assert!(m.parameter_count() < 95_000_000);
    }

    #[test]
    fn hubert_model_produces_one_frame_per_320_samples() {
        let Some(m) = load_model() else { return };

        // Half a second of 24 kHz audio, prepared the way the codec prepares it.
        const FRAMES: usize = 12;
        let wav = voice_like(FRAMES * 960, 24000.0);
        let at16 = resample(&wav, 24000, 16000);
        assert_eq!(at16.len(), FRAMES * 640);

        let mut padded = vec![0.0f32; 160];
        padded.extend_from_slice(&at16);
        padded.extend(std::iter::repeat_n(0.0f32, 160));

        let h = m.mean_hidden_states(&padded).expect("mean_hidden_states");
        assert_eq!(h.rows, 2 * FRAMES);
        assert_eq!(h.cols, 768);

        // Post-LayerNorm features sit near unit scale; anything wildly outside
        // that is a norm applied on the wrong axis.
        let mut sum_sq = 0.0f64;
        for &v in &h.data {
            assert!(v.is_finite());
            sum_sq += v as f64 * v as f64;
        }
        let rms = (sum_sq / h.data.len() as f64).sqrt();
        assert!(rms > 0.05);
        assert!(rms < 5.0);
    }

    #[test]
    fn hubert_model_is_deterministic() {
        let Some(m) = load_model() else { return };

        let wav = voice_like(4 * 640 + 320, 16000.0);
        let a = m.mean_hidden_states(&wav).expect("first pass");
        let b = m.mean_hidden_states(&wav).expect("second pass");
        assert_eq!(a.data.len(), b.data.len());
        for i in 0..a.data.len() {
            assert!(a.data[i] == b.data[i]);
        }
    }

    #[test]
    fn hubert_model_rejects_a_clip_it_cannot_consume() {
        let Some(m) = load_model() else { return };
        assert!(m.mean_hidden_states(&vec![0.0f32; 100]).is_err());
    }
}
