// =============================================================================
// SNAC -- Multi-Scale Neural Audio Codec, decoder only
// =============================================================================
//
// An autoregressive speech model does not emit audio. It emits codec tokens,
// and something else has to turn those into samples. Orpheus emits SNAC codes,
// so this is the second half of that pipeline: codes in, 24 kHz waveform out.
//
// Only the decoder is implemented. Encoding needs the analysis stack and the
// nearest-neighbour codebook search, neither of which text-to-speech uses.
//
// ## Residual vector quantization at three time scales
//
// A plain codec quantizes every frame at one rate. SNAC uses three codebooks
// running at 1/4, 1/2 and 1/1 of the frame rate, each coding what the previous
// one left behind. Coarse structure gets cheap slow codes and detail gets fast
// ones, so 7 codes cover 4 frames:
//
//   codebook 0, stride 4 -> 1 code per 4 frames   (~11.7 Hz)
//   codebook 1, stride 2 -> 2 codes per 4 frames  (~23.4 Hz)
//   codebook 2, stride 1 -> 4 codes per 4 frames  (~46.9 Hz)
//
// Reconstruction is `sum_i out_proj_i(codebook_i[code]) upsampled by stride_i`.
// The upsampling is a **repeat**, not a tile: stride 4 turns `[a, b]` into
// `[a, a, a, a, b, b, b, b]`. Interleaving instead produces a warble that is
// entirely plausible-sounding and completely wrong.
//
// ## The decoder stack
//
// Each frame carries 512 samples (`decoder_rates` [8, 8, 4, 2] multiply out to
// 512), so 4 frames is 2048 samples -- 85.33 ms at 24 kHz, which is exactly one
// Orpheus 7-token group. Every upsampling block multiplies the length by its
// stride exactly, and every residual unit preserves length exactly, so the
// output length is `n_frames * 512` with no slack. `conv1d.rs` asserts both.
//
// The 24 kHz configuration sets `attn_window_size` to null, so unlike the
// 44 kHz model there is no local attention anywhere in the decoder -- it is
// convolutions and activations end to end.
//
// ## Noise injection is stochastic by design
//
// Each upsampling block ends with a `NoiseBlock`: `x + randn * conv(x)`. The
// noise is drawn per timestep and **shared across all channels** (PyTorch draws
// shape `[B, 1, T]`, not `[B, C, T]`), which correlates it the way the trained
// model expects. Drawing per channel instead is a plausible mistake that only
// makes the output slightly hissier -- unhearable without a reference.
//
// Because of this, decoding the same codes twice does not give the same
// samples. `SnacNoise::Zero` suppresses the draw entirely, which makes the
// decoder deterministic for testing at the cost of a little naturalness.

#![allow(dead_code)]

use crate::autograd2::Mat;
use crate::conv1d::{
    conv_transpose1d, conv1d_dense, conv1d_depthwise, conv1d_pointwise, snake1d, snake1d_inplace,
    weight_norm_combine,
};
use crate::nn::InitRng;
use crate::torch_pickle::{TorchStateDict, load_torch_state_dict};

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct SnacConfig {
    pub sampling_rate: usize,
    /// Quantizer and decoder input width. `encoder_dim * 2^len(encoder_rates)`
    /// in the reference, which is 48 * 16 for the 24 kHz model.
    pub latent_dim: usize,
    /// Channel count entering the first upsampling block; halves each block.
    pub decoder_dim: usize,
    /// Upsampling factors, coarsest first.
    pub decoder_rates: Vec<usize>,
    /// One entry per codebook, in the order they are applied.
    pub vq_strides: Vec<usize>,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    /// Whether each upsampling block carries a `NoiseBlock`.
    pub noise: bool,
    /// Whether the residual units and the input convolution are grouped.
    pub depthwise: bool,
}

impl Default for SnacConfig {
    fn default() -> Self {
        SnacConfig {
            sampling_rate: 24000,
            latent_dim: 768,
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            vq_strides: vec![4, 2, 1],
            codebook_size: 4096,
            codebook_dim: 8,
            noise: true,
            depthwise: true,
        }
    }
}

impl SnacConfig {
    /// `hubertsiuzdak/snac_24khz`, which is the codec Orpheus emits into.
    pub fn snac_24khz() -> Self {
        SnacConfig {
            sampling_rate: 24000,
            // encoder_dim 48 * 2^len(encoder_rates=[2,4,8,8]) = 48 * 16.
            latent_dim: 768,
            decoder_dim: 1024,
            decoder_rates: vec![8, 8, 4, 2],
            vq_strides: vec![4, 2, 1],
            codebook_size: 4096,
            codebook_dim: 8,
            noise: true,
            depthwise: true,
        }
    }

    /// Samples per frame -- the product of `decoder_rates`, 512 at 24 kHz.
    pub fn upsample_factor(&self) -> usize {
        let mut n = 1;
        for &r in &self.decoder_rates {
            n *= r;
        }
        n
    }

    /// Frames covered by one group of codes, which is the coarsest stride.
    pub fn frames_per_group(&self) -> usize {
        self.vq_strides.first().copied().unwrap_or(1)
    }
}

/// How the `NoiseBlock`s draw their noise.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SnacNoise {
    /// Suppress the draw. Makes decoding deterministic, for tests.
    Zero,
    /// Draw N(0,1) per timestep from a seeded LCG. What real output uses.
    Seeded,
}

// =============================================================================
// Helpers (exposed for testing)
// =============================================================================

/// Repeat every row of `x` `factor` times in place of tiling the whole matrix.
///
/// `[a; b]` with factor 3 becomes `[a; a; a; b; b; b]`, which is what
/// `repeat_interleave` does and what the quantizer's stride upsampling needs.
pub fn repeat_rows(x: &Mat, factor: usize) -> Mat {
    if factor <= 1 {
        return x.clone();
    }
    let cols = x.cols;
    let mut out = Mat::zeros(x.rows * factor, cols);
    for r in 0..x.rows {
        let src = &x.data[r * cols..(r + 1) * cols];
        for k in 0..factor {
            let d = (r * factor + k) * cols;
            out.data[d..d + cols].copy_from_slice(src);
        }
    }
    out
}

/// Reconstruct a weight-normalized convolution weight as a `Mat` whose rows are
/// axis 0 of the stored tensor.
///
/// `prefix` names the module, e.g. `decoder.model.2.block.1`; the magnitude and
/// direction are read from `<prefix>.parametrizations.weight.original0` and
/// `...original1`.
pub fn load_weight_norm_mat(sd: &TorchStateDict, prefix: &str) -> Result<Mat, String> {
    let g = sd.require(&format!("{}.parametrizations.weight.original0", prefix))?;
    let v = sd.require(&format!("{}.parametrizations.weight.original1", prefix))?;

    // g is [axis0, 1, 1] and v is [axis0, ...]. Whether axis 0 means output or
    // input channels depends on the layer, and deriving the group count from g
    // rather than assuming either one is what makes transposed convolutions
    // work with no special case.
    if g.shape.is_empty() || v.shape.is_empty() || g.shape[0] != v.shape[0] {
        return Err(format!("snac: weight-norm shapes disagree on axis 0 for '{}'", prefix));
    }
    if g.numel() != g.shape[0] {
        return Err(format!(
            "snac: weight-norm magnitude for '{}' is not one per group",
            prefix
        ));
    }

    let combined = weight_norm_combine(&g.data, &v.data);
    let rows = v.shape[0];
    let cols = v.inner_size();
    Ok(Mat::new(combined, rows, cols))
}

/// Read a `[1, C, 1]` Snake alpha as a flat vector of length C.
pub fn load_alpha(sd: &TorchStateDict, name: &str) -> Result<Vec<f32>, String> {
    let t = sd.require(name)?;
    // Stored as [1, C, 1]; only the channel count matters here.
    Ok(t.data.clone())
}

/// Read an optional bias, returning an empty vector when absent.
fn optional_bias(sd: &TorchStateDict, name: &str) -> Vec<f32> {
    sd.find(name).map(|t| t.data.clone()).unwrap_or_default()
}

// =============================================================================
// Layers
// =============================================================================

/// Snake, dilated depthwise convolution, Snake, pointwise convolution, plus a
/// skip connection.
///
/// The dilated convolution pads by `3 * dilation` against a kernel of 7, which
/// preserves length exactly -- so the reference's centre-crop of the skip
/// branch never triggers, and this can add the input directly.
#[derive(Clone, Debug)]
pub struct SnacResidualUnit {
    pub alpha1: Vec<f32>,
    /// [C, K] when depthwise, [C, C*K] otherwise.
    pub conv1_weight: Mat,
    pub conv1_bias: Vec<f32>,
    pub alpha2: Vec<f32>,
    /// [C, C] pointwise.
    pub conv2_weight: Mat,
    pub conv2_bias: Vec<f32>,
    pub dilation: usize,
    pub kernel: usize,
    pub depthwise: bool,
}

impl Default for SnacResidualUnit {
    fn default() -> Self {
        SnacResidualUnit {
            alpha1: Vec::new(),
            conv1_weight: Mat::zeros(0, 0),
            conv1_bias: Vec::new(),
            alpha2: Vec::new(),
            conv2_weight: Mat::zeros(0, 0),
            conv2_bias: Vec::new(),
            dilation: 1,
            kernel: 7,
            depthwise: true,
        }
    }
}

impl SnacResidualUnit {
    pub fn forward(&self, x: &Mat) -> Mat {
        let y = snake1d(x, &self.alpha1);

        // Padding of 3*dilation against a kernel of 7 preserves length, so the
        // skip connection lines up without a crop.
        let padding = ((self.kernel - 1) * self.dilation) / 2;
        let mut y = if self.depthwise {
            conv1d_depthwise(&y, &self.conv1_weight, &self.conv1_bias, self.dilation, padding)
        } else {
            conv1d_dense(
                &y,
                &self.conv1_weight,
                x.cols,
                self.kernel,
                &self.conv1_bias,
                self.dilation,
                padding,
                1,
            )
        };
        assert!(y.rows == x.rows, "snac: residual unit changed the sequence length");

        snake1d_inplace(&mut y, &self.alpha2);
        let mut y = conv1d_pointwise(&y, &self.conv2_weight, &self.conv2_bias);

        y.add_assign(x);
        y
    }
}

/// `x + noise * conv(x)`, with one noise sample per timestep shared across
/// channels. The convolution has no bias.
#[derive(Clone, Debug)]
pub struct SnacNoiseBlock {
    /// [C, C] pointwise.
    pub weight: Mat,
}

impl SnacNoiseBlock {
    pub fn forward(&self, x: &Mat, mode: SnacNoise, rng: &mut InitRng) -> Mat {
        let h = conv1d_pointwise(x, &self.weight, &[]);
        let mut out = x.clone();
        let cols = out.cols;
        for t in 0..out.rows {
            // One draw per timestep, shared across every channel -- the reference
            // samples shape [B, 1, T]. Drawing per channel would decorrelate what
            // the model was trained to expect.
            let n = if mode == SnacNoise::Zero { 0.0f32 } else { rng.next_normal() };
            if n == 0.0 {
                continue;
            }
            let hrow = &h.data[t * cols..(t + 1) * cols];
            let orow = &mut out.data[t * cols..(t + 1) * cols];
            for c in 0..cols {
                orow[c] += n * hrow[c];
            }
        }
        out
    }
}

/// Snake, transposed convolution, optional noise, then three residual units at
/// dilations 1, 3 and 9.
#[derive(Clone, Debug)]
pub struct SnacDecoderBlock {
    pub alpha: Vec<f32>,
    /// [Cin, Cout * K] -- PyTorch's input-channels-first transposed layout.
    pub up_weight: Mat,
    /// Per **output** channel, so length `out_channels`.
    pub up_bias: Vec<f32>,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub noise: Option<SnacNoiseBlock>,
    pub units: Vec<SnacResidualUnit>,
}

impl Default for SnacDecoderBlock {
    fn default() -> Self {
        SnacDecoderBlock {
            alpha: Vec::new(),
            up_weight: Mat::zeros(0, 0),
            up_bias: Vec::new(),
            out_channels: 0,
            kernel: 0,
            stride: 0,
            padding: 0,
            output_padding: 0,
            noise: None,
            units: Vec::new(),
        }
    }
}

impl SnacDecoderBlock {
    pub fn forward(&self, x: &Mat, mode: SnacNoise, rng: &mut InitRng) -> Mat {
        let h = snake1d(x, &self.alpha);
        let mut h = conv_transpose1d(
            &h,
            &self.up_weight,
            self.out_channels,
            self.kernel,
            &self.up_bias,
            self.stride,
            self.padding,
            self.output_padding,
        );
        assert!(
            h.rows == x.rows * self.stride,
            "snac: upsampling block broke the length multiple"
        );

        if let Some(noise) = &self.noise {
            h = noise.forward(&h, mode, rng);
        }
        for unit in &self.units {
            h = unit.forward(&h);
        }
        h
    }
}

/// One codebook of the residual vector quantizer.
#[derive(Clone, Debug)]
pub struct SnacQuantizerLevel {
    /// [codebook_size, codebook_dim]
    pub codebook: Mat,
    /// [latent_dim, codebook_dim] pointwise.
    pub out_proj_weight: Mat,
    pub out_proj_bias: Vec<f32>,
    pub stride: usize,
}

impl Default for SnacQuantizerLevel {
    fn default() -> Self {
        SnacQuantizerLevel {
            codebook: Mat::zeros(0, 0),
            out_proj_weight: Mat::zeros(0, 0),
            out_proj_bias: Vec::new(),
            stride: 1,
        }
    }
}

/// The residual vector quantizer's synthesis half.
#[derive(Clone, Debug, Default)]
pub struct SnacQuantizer {
    pub levels: Vec<SnacQuantizerLevel>,
    pub latent_dim: usize,
}

impl SnacQuantizer {
    /// Codes to latents: `[T, latent_dim]`.
    ///
    /// `codes[i]` must hold `T / vq_strides[i]` entries, so the finest level
    /// fixes `T`. Every id must be below `codebook_size`.
    pub fn from_codes(&self, codes: &[Vec<u32>]) -> Result<Mat, String> {
        if codes.len() != self.levels.len() {
            return Err(format!(
                "snac: expected {} codebooks, got {}",
                self.levels.len(),
                codes.len()
            ));
        }
        // `codes.back()` in the reference -- an empty list is caught above
        // unless there are no levels at all.
        let Some(finest) = codes.last() else {
            return Err("snac: the finest codebook has no codes".to_string());
        };
        if finest.is_empty() {
            return Err("snac: the finest codebook has no codes".to_string());
        }

        let frames = finest.len();
        let mut z_q = Mat::zeros(frames, self.latent_dim);

        for (i, level) in self.levels.iter().enumerate() {
            let ids = &codes[i];

            if ids.len() * level.stride != frames {
                return Err(format!(
                    "snac: codebook {} has {} codes at stride {}, which does not cover {} frames",
                    i,
                    ids.len(),
                    level.stride,
                    frames
                ));
            }

            // Codebook lookup: one row of [codebook_dim] per code.
            let cb_cols = level.codebook.cols;
            let mut latents = Mat::zeros(ids.len(), cb_cols);
            for (t, &id) in ids.iter().enumerate() {
                if id as usize >= level.codebook.rows {
                    return Err(format!(
                        "snac: code {} at codebook {} index {} is out of range (codebook holds {})",
                        id, i, t, level.codebook.rows
                    ));
                }
                let src = &level.codebook.data[id as usize * cb_cols..(id as usize + 1) * cb_cols];
                latents.data[t * cb_cols..(t + 1) * cb_cols].copy_from_slice(src);
            }

            let projected = conv1d_pointwise(&latents, &level.out_proj_weight, &level.out_proj_bias);
            // Repeat, not tile: stride 4 turns [a, b] into [a,a,a,a,b,b,b,b].
            let upsampled = repeat_rows(&projected, level.stride);
            z_q.add_assign(&upsampled);
        }

        Ok(z_q)
    }
}

// =============================================================================
// Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct SnacDecoder {
    pub config: SnacConfig,
    pub quantizer: SnacQuantizer,

    /// `decoder.model.0` -- [768, 7] depthwise, or [768, 768*7] dense.
    pub in_conv_weight: Mat,
    pub in_conv_bias: Vec<f32>,
    /// `decoder.model.1` -- [1024, 768] pointwise. Absent when not depthwise,
    /// because then `decoder.model.0` already widens the channels.
    pub in_mix_weight: Option<Mat>,
    pub in_mix_bias: Vec<f32>,

    pub blocks: Vec<SnacDecoderBlock>,

    pub out_alpha: Vec<f32>,
    /// [1, C*7] dense, projecting to a single audio channel.
    pub out_weight: Mat,
    pub out_bias: Vec<f32>,
}

impl SnacDecoder {
    /// Load the decoder and quantizer from a `torch.save`d checkpoint.
    ///
    /// Ignores every encoder tensor. Reconstructs each weight-normalized
    /// parameter from its stored magnitude and direction, taking the group
    /// count from the magnitude's own length so that the transposed
    /// convolutions -- whose magnitude is per *input* channel, unlike every
    /// other convolution here -- need no special case.
    pub fn load(path: &str, cfg: SnacConfig) -> Result<SnacDecoder, String> {
        let sd = load_torch_state_dict(path)?;

        let config = cfg;

        // ---- quantizer ----
        let mut quantizer = SnacQuantizer { levels: Vec::new(), latent_dim: config.latent_dim };
        for i in 0..config.vq_strides.len() {
            let base = format!("quantizer.quantizers.{}", i);
            let codebook = sd.require_shape(
                &format!("{}.codebook.weight", base),
                &[config.codebook_size, config.codebook_dim],
            )?;
            let proj = load_weight_norm_mat(&sd, &format!("{}.out_proj", base))?;
            if proj.rows != config.latent_dim || proj.cols != config.codebook_dim {
                return Err(format!(
                    "snac: codebook {} out_proj is {}x{}, expected {}x{}",
                    i, proj.rows, proj.cols, config.latent_dim, config.codebook_dim
                ));
            }

            quantizer.levels.push(SnacQuantizerLevel {
                codebook: Mat::new(codebook.data.clone(), codebook.shape[0], codebook.shape[1]),
                out_proj_weight: proj,
                out_proj_bias: optional_bias(&sd, &format!("{}.out_proj.bias", base)),
                stride: config.vq_strides[i],
            });
        }

        // ---- decoder input ----
        // depthwise: model.0 filters at 768 channels, model.1 widens to 1024.
        // dense:     model.0 widens directly and there is no model.1.
        let in_conv_weight = load_weight_norm_mat(&sd, "decoder.model.0")?;
        let in_conv_bias = optional_bias(&sd, "decoder.model.0.bias");

        let mut next_module = 1;
        let mut in_mix_weight = None;
        let mut in_mix_bias = Vec::new();
        if config.depthwise {
            let mix = load_weight_norm_mat(&sd, "decoder.model.1")?;
            if mix.rows != config.decoder_dim {
                return Err(format!(
                    "snac: decoder.model.1 outputs {} channels, expected {}",
                    mix.rows, config.decoder_dim
                ));
            }
            in_mix_weight = Some(mix);
            in_mix_bias = optional_bias(&sd, "decoder.model.1.bias");
            next_module = 2;
        }

        // ---- upsampling blocks ----
        // With a NoiseBlock the residual units sit at block.{3,4,5}; without one
        // they shift down to block.{2,3,4}.
        let unit_base = if config.noise { 3 } else { 2 };
        let mut channels = config.decoder_dim;
        let mut blocks = Vec::new();

        for i in 0..config.decoder_rates.len() {
            let base = format!("decoder.model.{}", next_module + i);
            let stride = config.decoder_rates[i];
            let out_channels = channels / 2;

            let mut block = SnacDecoderBlock {
                stride,
                kernel: 2 * stride,
                padding: (stride + 1) / 2, // ceil(stride / 2)
                output_padding: stride % 2,
                out_channels,
                ..Default::default()
            };

            let alpha = load_alpha(&sd, &format!("{}.block.0.alpha", base))?;
            if alpha.len() != channels {
                return Err(format!(
                    "snac: {}.block.0.alpha has {} channels, expected {}",
                    base,
                    alpha.len(),
                    channels
                ));
            }
            block.alpha = alpha;

            let up = load_weight_norm_mat(&sd, &format!("{}.block.1", base))?;
            if up.rows != channels || up.cols != out_channels * block.kernel {
                return Err(format!(
                    "snac: {}.block.1 is {}x{}, expected {}x{} ([Cin, Cout*K] -- transposed convolutions store input channels first)",
                    base,
                    up.rows,
                    up.cols,
                    channels,
                    out_channels * block.kernel
                ));
            }
            block.up_weight = up;
            block.up_bias = optional_bias(&sd, &format!("{}.block.1.bias", base));
            if block.up_bias.len() != out_channels {
                return Err(format!(
                    "snac: {}.block.1.bias has {} entries, expected {} (bias is per output channel)",
                    base,
                    block.up_bias.len(),
                    out_channels
                ));
            }

            if config.noise {
                let noise_w = load_weight_norm_mat(&sd, &format!("{}.block.2.linear", base))?;
                if noise_w.rows != out_channels || noise_w.cols != out_channels {
                    return Err(format!(
                        "snac: {}.block.2.linear is not square at {} channels",
                        base, out_channels
                    ));
                }
                block.noise = Some(SnacNoiseBlock { weight: noise_w });
            }

            for u in 0..3 {
                let ub = format!("{}.block.{}", base, unit_base + u);
                let mut unit = SnacResidualUnit {
                    dilation: if u == 0 { 1 } else if u == 1 { 3 } else { 9 },
                    kernel: 7,
                    depthwise: config.depthwise,
                    ..Default::default()
                };

                unit.alpha1 = load_alpha(&sd, &format!("{}.block.0.alpha", ub))?;
                unit.conv1_weight = load_weight_norm_mat(&sd, &format!("{}.block.1", ub))?;
                unit.conv1_bias = optional_bias(&sd, &format!("{}.block.1.bias", ub));
                unit.alpha2 = load_alpha(&sd, &format!("{}.block.2.alpha", ub))?;
                unit.conv2_weight = load_weight_norm_mat(&sd, &format!("{}.block.3", ub))?;
                unit.conv2_bias = optional_bias(&sd, &format!("{}.block.3.bias", ub));

                if unit.alpha1.len() != out_channels || unit.alpha2.len() != out_channels {
                    return Err(format!(
                        "snac: {} alphas do not match {} channels",
                        ub, out_channels
                    ));
                }
                // The first convolution is grouped, the second is dense. Mixing
                // those up is a shape error rather than a silent one.
                let want_c1_cols =
                    if config.depthwise { unit.kernel } else { out_channels * unit.kernel };
                if unit.conv1_weight.rows != out_channels || unit.conv1_weight.cols != want_c1_cols {
                    return Err(format!(
                        "snac: {}.block.1 is {}x{}, expected {}x{}",
                        ub, unit.conv1_weight.rows, unit.conv1_weight.cols, out_channels, want_c1_cols
                    ));
                }
                if unit.conv2_weight.rows != out_channels || unit.conv2_weight.cols != out_channels {
                    return Err(format!(
                        "snac: {}.block.3 is not a dense {}x{} pointwise convolution",
                        ub, out_channels, out_channels
                    ));
                }

                block.units.push(unit);
            }

            blocks.push(block);
            channels = out_channels;
        }

        // ---- output ----
        let tail = next_module + config.decoder_rates.len();
        let out_alpha = load_alpha(&sd, &format!("decoder.model.{}.alpha", tail))?;
        if out_alpha.len() != channels {
            return Err(format!(
                "snac: output alpha has {} channels, expected {}",
                out_alpha.len(),
                channels
            ));
        }

        let out_weight = load_weight_norm_mat(&sd, &format!("decoder.model.{}", tail + 1))?;
        if out_weight.rows != 1 || out_weight.cols != channels * 7 {
            return Err(format!(
                "snac: output convolution is {}x{}, expected 1x{}",
                out_weight.rows,
                out_weight.cols,
                channels * 7
            ));
        }
        let out_bias = optional_bias(&sd, &format!("decoder.model.{}.bias", tail + 1));

        Ok(SnacDecoder {
            config,
            quantizer,
            in_conv_weight,
            in_conv_bias,
            in_mix_weight,
            in_mix_bias,
            blocks,
            out_alpha,
            out_weight,
            out_bias,
        })
    }

    /// Total number of f32 weights held, for reporting.
    pub fn parameter_count(&self) -> usize {
        let mut n = self.in_conv_weight.numel()
            + self.in_conv_bias.len()
            + self.in_mix_bias.len()
            + self.out_alpha.len()
            + self.out_weight.numel()
            + self.out_bias.len();
        if let Some(w) = &self.in_mix_weight {
            n += w.numel();
        }
        for level in &self.quantizer.levels {
            n += level.codebook.numel() + level.out_proj_weight.numel() + level.out_proj_bias.len();
        }
        for block in &self.blocks {
            n += block.alpha.len() + block.up_weight.numel() + block.up_bias.len();
            if let Some(noise) = &block.noise {
                n += noise.weight.numel();
            }
            for unit in &block.units {
                n += unit.alpha1.len()
                    + unit.conv1_weight.numel()
                    + unit.conv1_bias.len()
                    + unit.alpha2.len()
                    + unit.conv2_weight.numel()
                    + unit.conv2_bias.len();
            }
        }
        n
    }

    /// Codes to mono f32 samples in [-1, 1].
    ///
    /// Output length is exactly `codes.last().len() * upsample_factor()`.
    /// `seed` is only read when `mode` is `SnacNoise::Seeded`.
    pub fn decode(&self, codes: &[Vec<u32>], mode: SnacNoise, seed: u64) -> Result<Vec<f32>, String> {
        let z_q = self.quantizer.from_codes(codes)?;
        let frames = z_q.rows;

        let mut rng = InitRng::new(seed);

        // decoder.model.0 -- filters at the latent width, kernel 7, padding 3.
        let mut h = if self.config.depthwise {
            conv1d_depthwise(&z_q, &self.in_conv_weight, &self.in_conv_bias, 1, 3)
        } else {
            conv1d_dense(
                &z_q,
                &self.in_conv_weight,
                self.config.decoder_dim,
                7,
                &self.in_conv_bias,
                1,
                3,
                1,
            )
        };
        if h.rows != frames {
            return Err("snac: input convolution changed the frame count".to_string());
        }

        // decoder.model.1 -- widens to decoder_dim.
        if let Some(mix) = &self.in_mix_weight {
            h = conv1d_pointwise(&h, mix, &self.in_mix_bias);
        }

        for block in &self.blocks {
            h = block.forward(&h, mode, &mut rng);
        }

        if h.rows != frames * self.config.upsample_factor() {
            return Err(format!(
                "snac: decoder produced {} samples for {} frames, expected {}",
                h.rows,
                frames,
                frames * self.config.upsample_factor()
            ));
        }

        snake1d_inplace(&mut h, &self.out_alpha);
        let mono = conv1d_dense(&h, &self.out_weight, 1, 7, &self.out_bias, 1, 3, 1);
        if mono.cols != 1 {
            return Err("snac: output convolution did not collapse to one channel".to_string());
        }

        // Tanh bounds the waveform to [-1, 1], which is what makes the int16
        // conversion downstream safe without clipping.
        Ok(mono.data.iter().map(|v| v.tanh()).collect())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wav::wave_stats;

    const SNAC_PATH: &str = "models/snac_24khz.bin";

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    fn have_weights() -> bool {
        if std::path::Path::new(SNAC_PATH).exists() {
            true
        } else {
            eprintln!("skip: models/snac_24khz.bin not present");
            false
        }
    }

    fn identity(n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for i in 0..n {
            *m.at_mut(i, i) = 1.0;
        }
        m
    }

    /// A quantizer with one-hot codebooks, so a code's contribution is readable
    /// straight off the output.
    fn toy_quantizer(latent_dim: usize, strides: &[usize]) -> SnacQuantizer {
        let mut q = SnacQuantizer { levels: Vec::new(), latent_dim };
        for &stride in strides {
            // 4 codes of width 2; code c maps to [c, -c].
            let codebook = Mat::from_fn(4, 2, |r, c| if c == 0 { r as f32 } else { -(r as f32) });
            // Project [2] -> [latent_dim] by copying into the first two columns.
            let mut out_proj_weight = Mat::zeros(latent_dim, 2);
            *out_proj_weight.at_mut(0, 0) = 1.0;
            *out_proj_weight.at_mut(1, 1) = 1.0;
            q.levels.push(SnacQuantizerLevel {
                codebook,
                out_proj_weight,
                out_proj_bias: Vec::new(),
                stride,
            });
        }
        q
    }

    fn zero_unit(channels: usize, dilation: usize) -> SnacResidualUnit {
        SnacResidualUnit {
            dilation,
            kernel: 7,
            depthwise: true,
            alpha1: vec![1.0; channels],
            alpha2: vec![1.0; channels],
            conv1_weight: Mat::zeros(channels, 7),
            conv2_weight: Mat::zeros(channels, channels),
            ..Default::default()
        }
    }

    // =========================================================================
    // Config
    // =========================================================================

    #[test]
    fn snac_config_24khz_upsamples_by_512() {
        let c = SnacConfig::snac_24khz();
        assert_eq!(c.upsample_factor(), 512);
        assert_eq!(c.sampling_rate, 24000);
        assert_eq!(c.latent_dim, 768);
        assert_eq!(c.decoder_dim, 1024);
        assert_eq!(c.vq_strides, vec![4, 2, 1]);
        assert_eq!(c.frames_per_group(), 4);
    }

    #[test]
    fn one_code_group_is_2048_samples_or_85_33_ms() {
        // 7 Orpheus tokens carry 4 frames; 4 * 512 = 2048 samples at 24 kHz. This
        // is what fixes the realtime token rate at ~82/s, so pin it.
        let c = SnacConfig::snac_24khz();
        let samples = c.frames_per_group() * c.upsample_factor();
        assert_eq!(samples, 2048);
        let ms = 1000.0 * samples as f64 / c.sampling_rate as f64;
        assert!((ms - 85.333).abs() < 0.01);
    }

    // =========================================================================
    // repeat_rows
    // =========================================================================

    #[test]
    fn repeat_rows_repeats_rather_than_tiles() {
        // The distinction that matters: [a; b] with factor 3 must be
        // [a; a; a; b; b; b], not [a; b; a; b; a; b]. Tiling produces a warble
        // that sounds like a plausible codec artefact and is completely wrong.
        let x = Mat::new(vec![1.0, 2.0, 10.0, 20.0], 2, 2);
        let out = repeat_rows(&x, 3);
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, 2);
        for k in 0..3 {
            assert!(approx(out.at(k, 0), 1.0));
            assert!(approx(out.at(k, 1), 2.0));
        }
        for k in 3..6 {
            assert!(approx(out.at(k, 0), 10.0));
            assert!(approx(out.at(k, 1), 20.0));
        }
    }

    #[test]
    fn repeat_rows_at_factor_1_is_the_identity() {
        let x = Mat::new(vec![1.0, 2.0, 3.0], 3, 1);
        let out = repeat_rows(&x, 1);
        assert_eq!(out.rows, 3);
        assert_eq!(out.data, x.data);
    }

    // =========================================================================
    // Quantizer
    // =========================================================================

    #[test]
    fn from_codes_sums_three_time_scales() {
        let q = toy_quantizer(4, &[4, 2, 1]);
        // 4 frames: 1 coarse code, 2 mid codes, 4 fine codes.
        let codes = vec![vec![1u32], vec![2, 3], vec![0, 1, 2, 3]];
        let z = q.from_codes(&codes).expect("from_codes");
        assert_eq!(z.rows, 4);
        assert_eq!(z.cols, 4);

        // Column 0 is the sum of the three levels' code values at each frame:
        //   frame 0: 1 (coarse) + 2 (mid) + 0 (fine) = 3
        //   frame 1: 1 + 2 + 1 = 4
        //   frame 2: 1 + 3 + 2 = 6
        //   frame 3: 1 + 3 + 3 = 7
        assert!(approx(z.at(0, 0), 3.0));
        assert!(approx(z.at(1, 0), 4.0));
        assert!(approx(z.at(2, 0), 6.0));
        assert!(approx(z.at(3, 0), 7.0));
        // Column 1 mirrors it, since the toy codebook is [c, -c].
        assert!(approx(z.at(2, 1), -6.0));
        // Nothing beyond the projected width.
        assert!(approx(z.at(0, 2), 0.0));
    }

    #[test]
    fn from_codes_rejects_code_counts_that_do_not_cover_the_frames() {
        let q = toy_quantizer(4, &[4, 2, 1]);
        // The coarse level needs 1 code for 4 frames, not 2.
        let z = q.from_codes(&[vec![1, 1], vec![2, 3], vec![0, 1, 2, 3]]);
        assert!(z.unwrap_err().contains("does not cover"));
    }

    #[test]
    fn from_codes_rejects_an_out_of_range_code() {
        let q = toy_quantizer(4, &[4, 2, 1]);
        let z = q.from_codes(&[vec![1], vec![2, 3], vec![0, 1, 2, 99]]);
        assert!(z.unwrap_err().contains("out of range"));
    }

    #[test]
    fn from_codes_rejects_the_wrong_number_of_codebooks() {
        let q = toy_quantizer(4, &[4, 2, 1]);
        let z = q.from_codes(&[vec![1], vec![0, 1, 2, 3]]);
        assert!(z.unwrap_err().contains("expected 3 codebooks"));
    }

    #[test]
    fn from_codes_rejects_an_empty_finest_level() {
        let q = toy_quantizer(4, &[4, 2, 1]);
        let z = q.from_codes(&[vec![], vec![], vec![]]);
        assert!(z.is_err());
    }

    // =========================================================================
    // Noise block
    // =========================================================================

    #[test]
    fn snac_noise_block_in_zero_mode_is_the_identity() {
        let nb = SnacNoiseBlock { weight: identity(3) };
        let x = Mat::from_fn(5, 3, |r, c| r as f32 + 0.1 * c as f32);
        let mut rng = InitRng::new(1);
        let out = nb.forward(&x, SnacNoise::Zero, &mut rng);
        for i in 0..x.data.len() {
            assert!(approx(out.data[i], x.data[i]));
        }
    }

    #[test]
    fn snac_noise_block_shares_one_noise_draw_across_all_channels() {
        // The culprit this test exists for: PyTorch draws noise of shape
        // [B, 1, T] -- one sample per timestep, broadcast over channels. Drawing
        // per channel instead only makes the output slightly hissier, which is
        // undetectable by ear, so it has to be pinned here.
        //
        // With an identity weight and an all-ones input, out[t, c] = 1 + noise[t],
        // so every channel in a row must agree exactly, and rows must differ.
        let channels = 8;
        let nb = SnacNoiseBlock { weight: identity(channels) };
        let x = Mat::ones(6, channels);

        let mut rng = InitRng::new(42);
        let out = nb.forward(&x, SnacNoise::Seeded, &mut rng);

        for t in 0..out.rows {
            for c in 1..channels {
                assert!(approx(out.at(t, c), out.at(t, 0)));
            }
        }
        // And the draws actually vary between timesteps.
        let mut any_different = false;
        for t in 1..out.rows {
            if !approx(out.at(t, 0), out.at(0, 0)) {
                any_different = true;
            }
        }
        assert!(any_different);
        // Something was actually added.
        assert!(!approx(out.at(0, 0), 1.0));
    }

    // =========================================================================
    // Residual unit
    // =========================================================================

    #[test]
    fn snac_residual_unit_preserves_length_at_every_dilation() {
        let channels = 4;
        for dilation in [1usize, 3, 9] {
            let unit = zero_unit(channels, dilation);
            let x = Mat::ones(32, channels);
            let out = unit.forward(&x);
            assert_eq!(out.rows, 32);
            assert_eq!(out.cols, channels);
        }
    }

    #[test]
    fn snac_residual_unit_adds_its_input_through_the_skip() {
        // With both convolutions zeroed, the branch contributes nothing and the
        // output must equal the input exactly -- a missing skip connection would
        // show up as zeros.
        let channels = 3;
        let unit = zero_unit(channels, 1);

        let x = Mat::from_fn(10, channels, |r, c| 0.1 * r as f32 - 0.05 * c as f32);
        let out = unit.forward(&x);
        for i in 0..x.data.len() {
            assert!(approx(out.data[i], x.data[i]));
        }
    }

    // =========================================================================
    // Decoder block
    // =========================================================================

    #[test]
    fn snac_decoder_block_multiplies_length_by_its_stride() {
        for stride in [2usize, 4, 8] {
            let c_in = 8;
            let c_out = c_in / 2;

            let mut block = SnacDecoderBlock {
                stride,
                kernel: 2 * stride,
                padding: (stride + 1) / 2,
                output_padding: stride % 2,
                out_channels: c_out,
                alpha: vec![1.0; c_in],
                up_weight: Mat::zeros(c_in, c_out * 2 * stride),
                up_bias: vec![0.0; c_out],
                ..Default::default()
            };
            for u in 0..3 {
                block.units.push(zero_unit(c_out, if u == 0 { 1 } else if u == 1 { 3 } else { 9 }));
            }

            let mut rng = InitRng::new(3);
            let x = Mat::ones(6, c_in);
            let out = block.forward(&x, SnacNoise::Zero, &mut rng);
            assert_eq!(out.rows, 6 * stride);
            assert_eq!(out.cols, c_out);
        }
    }

    // =========================================================================
    // The real decoder
    // =========================================================================

    #[test]
    fn snac_decoder_loads_the_24khz_checkpoint() {
        if !have_weights() {
            return;
        }
        let d = SnacDecoder::load(SNAC_PATH, SnacConfig::snac_24khz()).expect("load");

        assert_eq!(d.blocks.len(), 4);
        assert_eq!(d.quantizer.levels.len(), 3);
        assert!(d.in_mix_weight.is_some());

        // Channel widths halve down the stack: 1024 -> 512 -> 256 -> 128 -> 64.
        let want_out = [512usize, 256, 128, 64];
        let want_stride = [8usize, 8, 4, 2];
        for i in 0..4 {
            assert_eq!(d.blocks[i].out_channels, want_out[i]);
            assert_eq!(d.blocks[i].stride, want_stride[i]);
            assert_eq!(d.blocks[i].kernel, 2 * want_stride[i]);
            assert!(d.blocks[i].noise.is_some());
            assert_eq!(d.blocks[i].units.len(), 3);
            // A transposed convolution's weight is [Cin, Cout*K].
            assert_eq!(d.blocks[i].up_weight.cols, want_out[i] * d.blocks[i].kernel);
            // Its bias is per output channel, not per input channel.
            assert_eq!(d.blocks[i].up_bias.len(), want_out[i]);
        }

        assert_eq!(d.out_alpha.len(), 64);
        assert_eq!(d.out_weight.rows, 1);
        assert_eq!(d.out_weight.cols, 64 * 7);

        // The whole checkpoint is 19.9 M parameters including the encoder; the
        // decoder plus quantizer is the bulk of it.
        let n = d.parameter_count();
        assert!(n > 10_000_000, "decoder parameters: {}", n);
        assert!(n < 20_000_000, "decoder parameters: {}", n);
    }

    #[test]
    fn snac_decoder_produces_exactly_frames_times_512_samples() {
        if !have_weights() {
            return;
        }
        let d = SnacDecoder::load(SNAC_PATH, SnacConfig::snac_24khz()).expect("load");

        // Two Orpheus groups: 8 frames.
        let codes = vec![
            vec![100u32, 200],
            vec![300, 400, 500, 600],
            vec![700, 800, 900, 1000, 1100, 1200, 1300, 1400],
        ];

        let audio = d.decode(&codes, SnacNoise::Zero, 0).expect("decode");
        assert_eq!(audio.len(), 8 * 512);
        assert_eq!(audio.len(), 4096);

        // Tanh must bound the output, which is what makes the int16 conversion
        // safe. Arbitrary codes are not speech, so only the range is asserted.
        let s = wave_stats(&audio);
        assert!(s.in_range(), "{}", s.describe());
        assert!(s.peak <= 1.0);
    }

    #[test]
    fn snac_decoder_is_deterministic_with_noise_suppressed() {
        if !have_weights() {
            return;
        }
        let d = SnacDecoder::load(SNAC_PATH, SnacConfig::snac_24khz()).expect("load");

        let codes = vec![vec![5u32], vec![6, 7], vec![8, 9, 10, 11]];

        let a = d.decode(&codes, SnacNoise::Zero, 0).expect("a");
        let b = d.decode(&codes, SnacNoise::Zero, 99).expect("b");
        // The seed is ignored in Zero mode, so these must be bit-identical.
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i], b[i]);
        }

        // Seeded noise changes the output, and does so reproducibly per seed.
        let c = d.decode(&codes, SnacNoise::Seeded, 7).expect("c");
        let e = d.decode(&codes, SnacNoise::Seeded, 7).expect("e");
        for i in 0..c.len() {
            assert_eq!(c[i], e[i]);
        }
        let differs = a.iter().zip(&c).any(|(x, y)| x != y);
        assert!(differs);
    }

    #[test]
    fn snac_decoder_rejects_a_mismatched_config() {
        if !have_weights() {
            return;
        }
        // The 44.1 kHz shape against 24 kHz weights: every dimension is wrong, and
        // the loader must say which rather than producing a decoder that runs and
        // outputs noise.
        let mut wrong = SnacConfig::snac_24khz();
        wrong.decoder_dim = 1536;
        let err = SnacDecoder::load(SNAC_PATH, wrong).unwrap_err();
        assert!(err.contains("1536"), "{}", err);
    }
}
