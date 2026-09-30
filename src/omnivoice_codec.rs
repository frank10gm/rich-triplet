// =============================================================================
// OmniVoice audio codec
// =============================================================================
//
// The second half of OmniVoice: eight streams of codebook indices in, a 24 kHz
// waveform out. Structurally this is the same family as SNAC -- residual vector
// quantization feeding a stack of transposed convolutions with Snake
// activations -- so it reuses `conv1d.rs` wholesale. The differences are worth
// naming.
//
// | | SNAC 24 kHz | OmniVoice |
// |---|---|---|
// | Codebooks | 3, at strides 4/2/1 | **8, all at the same rate** |
// | Codebook | 4096 x 8 | 1024 x 64 |
// | Upsampling | 8, 8, 4, 2 (512x) | **8, 5, 4, 2, 3 (960x)** |
// | Residual convs | depthwise | **dense** |
// | Weight storage | weight-norm `g` and `v` | already reconstructed |
// | Noise injection | yes, stochastic | **none -- fully deterministic** |
//
// Two of those matter in practice. The codebooks running at one rate means
// there is no stride bookkeeping and no interleave to get wrong: eight indices
// per frame, summed. And no noise block means decoding is deterministic, so a
// given set of codes always produces exactly the same samples -- unlike SNAC,
// where reproducibility had to be bought by suppressing the noise draw.
//
// The dense residual convolutions are why `conv1d_dense` needed an im2col path.
// A 7-tap dense convolution at 512 channels over a sequence upsampled toward
// 96 000 samples is tens of GFLOP per clip.
//
// ## Frame arithmetic
//
// `hop_length` is 960 and the upsampling ratios multiply to exactly that, so
// one frame is 960 samples: 25 Hz at 24 kHz. Eight codes per frame means 200
// tokens per second of audio.
//
// ## Both directions
//
// `OmniCodecDecoder` is the synthesis half and is all that plain generation
// needs -- about 190 of the file's 486 tensors. `OmniCodecEncoder`, at the
// bottom of this file, is the analysis half: the `acoustic_encoder`, the
// `encoder_semantic` stack and a 94 M-parameter HuBERT, which turn reference
// audio into codes for voice cloning. It is loaded only when there is a
// reference, since it is eight times the size of the decoder.

#![allow(dead_code)]

use crate::autograd2::Mat;
use crate::conv1d::{
    conv_transpose1d, conv1d_dense, conv1d_pointwise, snake1d, snake1d_inplace,
};
use crate::gguf_loader::{GgufFile, load_gguf_alpha, load_gguf_conv_weight, load_gguf_vector};
use crate::hubert::{HubertConfig, HubertModel};
use crate::resample::resample;

/// The rate the semantic model runs at, whatever the codec's own is.
pub const SEMANTIC_SAMPLE_RATE: usize = 16000;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct OmniCodecConfig {
    pub sample_rate: usize,
    /// Samples per frame. Equals the product of `upsampling_ratios`.
    pub hop_length: usize,
    /// Quantizer width, and the decoder's input before `fc2`.
    pub latent_dim: usize,
    /// Channels entering the first upsampling block; halves each block.
    pub decoder_dim: usize,
    /// Channels leaving the encoder's input convolution; doubles each block.
    pub encoder_dim: usize,
    /// The decoder's input width after `fc2`, and the acoustic encoder's
    /// output width -- the same number because they are the two ends of one
    /// bottleneck.
    pub decoder_in_dim: usize,
    /// Coarsest first.
    pub upsampling_ratios: Vec<usize>,
    pub n_codebooks: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
}

impl Default for OmniCodecConfig {
    fn default() -> Self {
        OmniCodecConfig {
            sample_rate: 24000,
            hop_length: 960,
            latent_dim: 1024,
            decoder_dim: 1024,
            encoder_dim: 64,
            decoder_in_dim: 256,
            upsampling_ratios: vec![8, 5, 4, 2, 3],
            n_codebooks: 8,
            codebook_size: 1024,
            codebook_dim: 64,
        }
    }
}

impl OmniCodecConfig {
    pub fn defaults() -> Self {
        OmniCodecConfig::default()
    }

    /// Product of `upsampling_ratios`; must equal `hop_length`.
    pub fn upsample_factor(&self) -> usize {
        let mut n = 1usize;
        for &r in &self.upsampling_ratios {
            n *= r;
        }
        n
    }

    /// The semantic path's width: whatever the acoustic path does not fill of
    /// the quantizer's input.
    pub fn semantic_dim(&self) -> usize {
        if self.latent_dim > self.decoder_in_dim { self.latent_dim - self.decoder_in_dim } else { 0 }
    }
}

// =============================================================================
// Layers
// =============================================================================

/// Snake, dense dilated convolution, Snake, pointwise convolution, plus a skip.
///
/// Padding is `3 * dilation` against a kernel of 7, which preserves length
/// exactly, so the skip lines up with no crop.
#[derive(Clone, Debug)]
pub struct OmniResidualUnit {
    pub alpha1: Vec<f32>,
    /// [C, C * 7]
    pub conv1_weight: Mat,
    pub conv1_bias: Vec<f32>,
    pub alpha2: Vec<f32>,
    /// [C, C]
    pub conv2_weight: Mat,
    pub conv2_bias: Vec<f32>,
    pub dilation: usize,
}

impl Default for OmniResidualUnit {
    fn default() -> Self {
        OmniResidualUnit {
            alpha1: Vec::new(),
            conv1_weight: Mat::zeros(0, 0),
            conv1_bias: Vec::new(),
            alpha2: Vec::new(),
            conv2_weight: Mat::zeros(0, 0),
            conv2_bias: Vec::new(),
            dilation: 1,
        }
    }
}

impl OmniResidualUnit {
    pub fn forward(&self, x: &Mat) -> Mat {
        let channels = x.cols;
        let y = snake1d(x, &self.alpha1);

        // Kernel 7 with padding 3*dilation preserves length exactly.
        let mut y = conv1d_dense(
            &y,
            &self.conv1_weight,
            channels,
            7,
            &self.conv1_bias,
            self.dilation,
            3 * self.dilation,
            1,
        );
        debug_assert!(y.rows == x.rows, "omnivoice codec: residual unit changed the sequence length");

        snake1d_inplace(&mut y, &self.alpha2);
        let mut y = conv1d_pointwise(&y, &self.conv2_weight, &self.conv2_bias);

        y.add_assign(x);
        y
    }
}

/// Snake, transposed convolution, then three residual units at dilations
/// 1, 3 and 9.
#[derive(Clone, Debug)]
pub struct OmniDecoderBlock {
    pub alpha: Vec<f32>,
    /// [Cin, Cout * K] -- transposed convolutions store input channels first.
    pub up_weight: Mat,
    /// Per output channel.
    pub up_bias: Vec<f32>,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
    pub output_padding: usize,
    pub units: Vec<OmniResidualUnit>,
}

impl Default for OmniDecoderBlock {
    fn default() -> Self {
        OmniDecoderBlock {
            alpha: Vec::new(),
            up_weight: Mat::zeros(0, 0),
            up_bias: Vec::new(),
            out_channels: 0,
            kernel: 0,
            stride: 0,
            padding: 0,
            output_padding: 0,
            units: Vec::new(),
        }
    }
}

impl OmniDecoderBlock {
    pub fn forward(&self, x: &Mat) -> Mat {
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
        debug_assert!(
            h.rows == x.rows * self.stride,
            "omnivoice codec: block broke the length multiple"
        );

        for unit in &self.units {
            h = unit.forward(&h);
        }
        h
    }
}

/// One level of the residual vector quantizer.
#[derive(Clone, Debug)]
pub struct OmniQuantizerLevel {
    /// [codebook_size, codebook_dim]
    pub codebook: Mat,
    /// [latent_dim, codebook_dim] pointwise.
    pub project_out_weight: Mat,
    pub project_out_bias: Vec<f32>,
    /// [codebook_dim, latent_dim] pointwise. Empty in a decoder-only load;
    /// analysis needs it to get *into* the codebook's space.
    pub project_in_weight: Mat,
    pub project_in_bias: Vec<f32>,
}

impl Default for OmniQuantizerLevel {
    fn default() -> Self {
        OmniQuantizerLevel {
            codebook: Mat::zeros(0, 0),
            project_out_weight: Mat::zeros(0, 0),
            project_out_bias: Vec::new(),
            project_in_weight: Mat::zeros(0, 0),
            project_in_bias: Vec::new(),
        }
    }
}

/// The residual vector quantizer's synthesis half.
///
/// Every codebook runs at the frame rate, so reconstruction is a plain sum with
/// no stride bookkeeping:
///
///     z = sum_i project_out_i(codebook_i[code_i])
#[derive(Clone, Debug, Default)]
pub struct OmniQuantizer {
    pub levels: Vec<OmniQuantizerLevel>,
    pub latent_dim: usize,
}

impl OmniQuantizer {
    /// `codes[i]` holds one index per frame, the same count for every i.
    /// Returns [T, latent_dim].
    pub fn from_codes(&self, codes: &[Vec<u32>]) -> Result<Mat, String> {
        if codes.len() != self.levels.len() {
            return Err(format!(
                "omnivoice codec: expected {} codebooks, got {}",
                self.levels.len(),
                codes.len()
            ));
        }
        if codes.is_empty() || codes[0].is_empty() {
            return Err("omnivoice codec: no codes to decode".into());
        }

        let frames = codes[0].len();
        let mut z = Mat::zeros(frames, self.latent_dim);

        for (i, level) in self.levels.iter().enumerate() {
            let ids = &codes[i];
            // Every codebook runs at the frame rate, so there is no stride to
            // reconcile -- just a length check.
            if ids.len() != frames {
                return Err(format!(
                    "omnivoice codec: codebook {} has {} codes, expected {}",
                    i,
                    ids.len(),
                    frames
                ));
            }

            let dim = level.codebook.cols;
            let mut latents = Mat::zeros(frames, dim);
            for t in 0..frames {
                if ids[t] as usize >= level.codebook.rows {
                    return Err(format!(
                        "omnivoice codec: code {} at codebook {} index {} is out of range \
                         (codebook holds {})",
                        ids[t], i, t, level.codebook.rows
                    ));
                }
                let src = &level.codebook.data[ids[t] as usize * dim..(ids[t] as usize + 1) * dim];
                latents.data[t * dim..(t + 1) * dim].copy_from_slice(src);
            }

            let projected =
                conv1d_pointwise(&latents, &level.project_out_weight, &level.project_out_bias);
            z.add_assign(&projected);
        }

        Ok(z)
    }

    /// The analysis direction: latents to codes, one vector per codebook.
    ///
    /// Residual quantization is greedy and sequential. Level 0 picks the
    /// nearest entry to the latent, its reconstruction is subtracted, and
    /// level 1 codes what is left. So the codebooks are not independent and
    /// cannot be searched in parallel -- each one only ever sees the error the
    /// ones before it could not represent, which is why the later codebooks
    /// carry finer detail and why unmasking them last is the right order.
    pub fn to_codes(&self, latents: &Mat) -> Result<Vec<Vec<u32>>, String> {
        if self.levels.is_empty() {
            return Err("omnivoice codec: the quantizer has no levels".into());
        }
        if latents.cols != self.latent_dim {
            return Err(format!(
                "omnivoice codec: latents are {} wide, expected {}",
                latents.cols, self.latent_dim
            ));
        }
        for level in &self.levels {
            if level.project_in_weight.rows == 0 {
                return Err("omnivoice codec: this quantizer was loaded for synthesis only and has \
                            no project_in"
                    .into());
            }
        }

        let mut codes: Vec<Vec<u32>> = Vec::with_capacity(self.levels.len());

        let mut residual = latents.clone();
        for level in &self.levels {
            let dim = level.codebook.cols;
            let projected =
                conv1d_pointwise(&residual, &level.project_in_weight, &level.project_in_bias);

            // Nearest entry by Euclidean distance. ||x||^2 is the same for every
            // candidate, so only -2 x.e + ||e||^2 decides -- which turns the search
            // into one gemm against the codebook plus a precomputed norm.
            let mut code_norm = vec![0.0f32; level.codebook.rows];
            for (e, slot) in code_norm.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for d in 0..dim {
                    let v = level.codebook.at(e, d);
                    sum += v * v;
                }
                *slot = sum;
            }
            let dots = projected.matmul_bt(&level.codebook);

            let mut chosen = vec![0u32; latents.rows];
            for t in 0..latents.rows {
                let row = &dots.data[t * dots.cols..(t + 1) * dots.cols];
                let mut best = f32::INFINITY;
                let mut best_e = 0u32;
                for e in 0..level.codebook.rows {
                    let d = code_norm[e] - 2.0f32 * row[e];
                    if d < best {
                        best = d;
                        best_e = e as u32;
                    }
                }
                chosen[t] = best_e;
            }

            // Subtract what this level can represent, so the next one codes the
            // error rather than the signal again.
            let mut picked = Mat::zeros(latents.rows, dim);
            for t in 0..latents.rows {
                for d in 0..dim {
                    *picked.at_mut(t, d) = level.codebook.at(chosen[t] as usize, d);
                }
            }
            let reconstructed =
                conv1d_pointwise(&picked, &level.project_out_weight, &level.project_out_bias);
            for i in 0..residual.data.len() {
                residual.data[i] -= reconstructed.data[i];
            }

            codes.push(chosen);
        }
        Ok(codes)
    }
}

// =============================================================================
// Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct OmniCodecDecoder {
    pub config: OmniCodecConfig,
    pub quantizer: OmniQuantizer,

    /// `fc2`: quantizer width down to the decoder's input width.
    pub fc2_weight: Mat,
    pub fc2_bias: Vec<f32>,

    /// `acoustic_decoder.conv1`: [decoder_dim, decoder_in_dim * 7].
    pub in_conv_weight: Mat,
    pub in_conv_bias: Vec<f32>,

    pub blocks: Vec<OmniDecoderBlock>,

    pub out_alpha: Vec<f32>,
    /// `acoustic_decoder.conv2`: [1, C * 7].
    pub out_weight: Mat,
    pub out_bias: Vec<f32>,
}

/// Open the codec GGUF and check that it declares the tokenizer architecture.
fn open_codec_gguf(path: &str) -> Result<GgufFile, String> {
    let gguf = GgufFile::open(path).map_err(|e| e.to_string())?;
    if let Some(arch) = gguf.metadata.get("general.architecture") {
        if let Some(name) = arch.as_str() {
            if name != "omnivoice-tokenizer" {
                return Err(format!(
                    "omnivoice codec: GGUF declares architecture '{}', expected \
                     'omnivoice-tokenizer'",
                    name
                ));
            }
        }
    }
    Ok(gguf)
}

fn conv_weight(gguf: &GgufFile, name: &str, out_channels: usize) -> Result<Mat, String> {
    load_gguf_conv_weight(gguf, name, out_channels).map_err(|e| e.to_string())
}

fn vector(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    load_gguf_vector(gguf, name).map_err(|e| e.to_string())
}

fn alpha(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, String> {
    load_gguf_alpha(gguf, name).map_err(|e| e.to_string())
}

impl OmniCodecDecoder {
    /// Load the decode-side tensors from an `omnivoice-tokenizer` GGUF.
    ///
    /// Ignores every analysis tensor.
    pub fn load(path: &str, cfg: OmniCodecConfig) -> Result<OmniCodecDecoder, String> {
        let gguf = open_codec_gguf(path)?;

        if cfg.upsample_factor() != cfg.hop_length {
            return Err(format!(
                "omnivoice codec: upsampling ratios multiply to {} but hop_length is {}",
                cfg.upsample_factor(),
                cfg.hop_length
            ));
        }

        // ---- quantizer ----
        let mut quantizer = OmniQuantizer { levels: Vec::new(), latent_dim: cfg.latent_dim };
        for i in 0..cfg.n_codebooks {
            let base = format!("quantizer.quantizers.{}", i);
            let mut level = OmniQuantizerLevel::default();

            let embed = conv_weight(&gguf, &format!("{}.codebook.embed", base), cfg.codebook_size)?;
            if embed.cols != cfg.codebook_dim {
                return Err(format!(
                    "omnivoice codec: codebook {} is {}x{}, expected {}x{}",
                    i, embed.rows, embed.cols, cfg.codebook_size, cfg.codebook_dim
                ));
            }
            level.codebook = embed;

            let proj = conv_weight(&gguf, &format!("{}.project_out.weight", base), cfg.latent_dim)?;
            if proj.cols != cfg.codebook_dim {
                return Err(format!(
                    "omnivoice codec: codebook {} project_out is {}x{}, expected {}x{}",
                    i, proj.rows, proj.cols, cfg.latent_dim, cfg.codebook_dim
                ));
            }
            level.project_out_weight = proj;

            level.project_out_bias = vector(&gguf, &format!("{}.project_out.bias", base))?;

            quantizer.levels.push(level);
        }

        // ---- fc2: latent width down to the decoder's input width ----
        let fc2 = conv_weight(&gguf, "fc2.weight", cfg.decoder_in_dim)?;
        if fc2.cols != cfg.latent_dim {
            return Err(format!(
                "omnivoice codec: fc2 is {}x{}, expected {}x{}",
                fc2.rows, fc2.cols, cfg.decoder_in_dim, cfg.latent_dim
            ));
        }
        let fc2_bias = vector(&gguf, "fc2.bias")?;

        // ---- acoustic_decoder ----
        let in_conv = conv_weight(&gguf, "acoustic_decoder.conv1.weight", cfg.decoder_dim)?;
        if in_conv.cols != cfg.decoder_in_dim * 7 {
            return Err(format!(
                "omnivoice codec: acoustic_decoder.conv1 is {}x{}, expected {}x{}",
                in_conv.rows,
                in_conv.cols,
                cfg.decoder_dim,
                cfg.decoder_in_dim * 7
            ));
        }
        let in_bias = vector(&gguf, "acoustic_decoder.conv1.bias")?;

        let mut blocks = Vec::with_capacity(cfg.upsampling_ratios.len());
        let mut channels = cfg.decoder_dim;
        for (i, &stride) in cfg.upsampling_ratios.iter().enumerate() {
            let base = format!("acoustic_decoder.block.{}", i);
            let out_channels = channels / 2;

            let mut block = OmniDecoderBlock {
                stride,
                kernel: 2 * stride,
                padding: stride.div_ceil(2), // ceil(stride / 2)
                output_padding: stride % 2,
                out_channels,
                ..Default::default()
            };

            let a = alpha(&gguf, &format!("{}.snake1.alpha", base))?;
            if a.len() != channels {
                return Err(format!(
                    "omnivoice codec: {}.snake1.alpha has {} channels, expected {}",
                    base,
                    a.len(),
                    channels
                ));
            }
            block.alpha = a;

            // A transposed convolution's weight is [Cin, Cout * K].
            let up = conv_weight(&gguf, &format!("{}.conv_t1.weight", base), channels)?;
            if up.cols != out_channels * block.kernel {
                return Err(format!(
                    "omnivoice codec: {}.conv_t1 is {}x{}, expected {}x{}",
                    base,
                    up.rows,
                    up.cols,
                    channels,
                    out_channels * block.kernel
                ));
            }
            block.up_weight = up;
            let up_bias = vector(&gguf, &format!("{}.conv_t1.bias", base))?;
            if up_bias.len() != out_channels {
                return Err(format!(
                    "omnivoice codec: {}.conv_t1.bias has {} entries, expected {} (bias is per \
                     output channel)",
                    base,
                    up_bias.len(),
                    out_channels
                ));
            }
            block.up_bias = up_bias;

            for u in 0..3 {
                let ub = format!("{}.res_unit{}", base, u + 1);
                let mut unit = OmniResidualUnit {
                    dilation: if u == 0 { 1 } else if u == 1 { 3 } else { 9 },
                    ..Default::default()
                };

                unit.alpha1 = alpha(&gguf, &format!("{}.snake1.alpha", ub))?;
                unit.conv1_weight = conv_weight(&gguf, &format!("{}.conv1.weight", ub), out_channels)?;
                unit.conv1_bias = vector(&gguf, &format!("{}.conv1.bias", ub))?;
                unit.alpha2 = alpha(&gguf, &format!("{}.snake2.alpha", ub))?;
                unit.conv2_weight = conv_weight(&gguf, &format!("{}.conv2.weight", ub), out_channels)?;
                unit.conv2_bias = vector(&gguf, &format!("{}.conv2.bias", ub))?;

                // conv1 is a dense 7-tap, conv2 a pointwise mix. Mixing those up is
                // a shape error rather than a silent one.
                if unit.conv1_weight.cols != out_channels * 7 {
                    return Err(format!(
                        "omnivoice codec: {}.conv1 is {}x{}, expected {}x{}",
                        ub,
                        unit.conv1_weight.rows,
                        unit.conv1_weight.cols,
                        out_channels,
                        out_channels * 7
                    ));
                }
                if unit.conv2_weight.cols != out_channels {
                    return Err(format!(
                        "omnivoice codec: {}.conv2 is not a pointwise {}x{}",
                        ub, out_channels, out_channels
                    ));
                }
                if unit.alpha1.len() != out_channels || unit.alpha2.len() != out_channels {
                    return Err(format!(
                        "omnivoice codec: {} alphas do not match {} channels",
                        ub, out_channels
                    ));
                }

                block.units.push(unit);
            }

            blocks.push(block);
            channels = out_channels;
        }

        // ---- output ----
        let out_alpha = alpha(&gguf, "acoustic_decoder.snake1.alpha")?;
        if out_alpha.len() != channels {
            return Err(format!(
                "omnivoice codec: output alpha has {} channels, expected {}",
                out_alpha.len(),
                channels
            ));
        }

        // One output channel, so GGUF elides the leading dimension entirely.
        let out_w = conv_weight(&gguf, "acoustic_decoder.conv2.weight", 1)?;
        if out_w.cols != channels * 7 {
            return Err(format!(
                "omnivoice codec: acoustic_decoder.conv2 is 1x{}, expected 1x{}",
                out_w.cols,
                channels * 7
            ));
        }
        let out_bias = vector(&gguf, "acoustic_decoder.conv2.bias")?;

        Ok(OmniCodecDecoder {
            config: cfg,
            quantizer,
            fc2_weight: fc2,
            fc2_bias,
            in_conv_weight: in_conv,
            in_conv_bias: in_bias,
            blocks,
            out_alpha,
            out_weight: out_w,
            out_bias,
        })
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.fc2_weight.numel()
            + self.fc2_bias.len()
            + self.in_conv_weight.numel()
            + self.in_conv_bias.len()
            + self.out_alpha.len()
            + self.out_weight.numel()
            + self.out_bias.len();
        for level in &self.quantizer.levels {
            n += level.codebook.numel() + level.project_out_weight.numel() + level.project_out_bias.len();
        }
        for block in &self.blocks {
            n += block.alpha.len() + block.up_weight.numel() + block.up_bias.len();
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
    /// Output length is exactly `frames * hop_length`. Deterministic: there is
    /// no noise injection anywhere in this decoder.
    pub fn decode(&self, codes: &[Vec<u32>]) -> Result<Vec<f32>, String> {
        let z = self.quantizer.from_codes(codes)?;
        let frames = z.rows;

        // fc2 narrows the quantizer's output to the decoder's input width.
        let h = conv1d_pointwise(&z, &self.fc2_weight, &self.fc2_bias);

        let mut h = conv1d_dense(
            &h,
            &self.in_conv_weight,
            self.config.decoder_dim,
            7,
            &self.in_conv_bias,
            1,
            3,
            1,
        );
        if h.rows != frames {
            return Err("omnivoice codec: input convolution changed the frame count".into());
        }

        for block in &self.blocks {
            h = block.forward(&h);
        }

        if h.rows != frames * self.config.hop_length {
            return Err(format!(
                "omnivoice codec: decoder produced {} samples for {} frames, expected {}",
                h.rows,
                frames,
                frames * self.config.hop_length
            ));
        }

        snake1d_inplace(&mut h, &self.out_alpha);
        let mono = conv1d_dense(&h, &self.out_weight, 1, 7, &self.out_bias, 1, 3, 1);
        if mono.cols != 1 {
            return Err("omnivoice codec: output convolution did not collapse to one channel".into());
        }

        // The reference bounds its output with tanh, which is also what keeps the
        // int16 conversion downstream safe without clipping.
        let samples: Vec<f32> = (0..mono.rows).map(|t| mono.at(t, 0).tanh()).collect();
        Ok(samples)
    }
}

// =============================================================================
// Analysis
// =============================================================================
//
// The mirror of the decoder, and the half voice cloning needs: a waveform in,
// eight streams of codebook indices out. It is not a mirror of one stack but of
// two -- the codec quantizes the concatenation of an *acoustic* path and a
// *semantic* one, so that a code carries both how a voice sounds and what it
// said.
//
//   24 kHz samples --> acoustic encoder (DAC) ------> [T, 256] --+
//                  \                                             |--> fc --> RVQ
//                   -> 16 kHz --> HuBERT --> conv stack --> [T, 768] --+
//
// Both arrive at the same 25 Hz frame rate by different arithmetic, and that
// they agree is the load-bearing invariant: the acoustic path divides 24 kHz by
// 960, while the semantic path divides 16 kHz by 320 for 50 Hz and then keeps
// every other frame.

/// One downsampling stage: three residual units, Snake, then a strided
/// convolution that halves the length by `stride` and doubles the channels.
///
/// The residual units run *before* the downsample here, where the decoder runs
/// them after its upsample. That is not symmetry for its own sake -- it keeps
/// the expensive dilated convolutions on the shorter side of the stride in
/// both directions.
#[derive(Clone, Debug)]
pub struct OmniEncoderBlock {
    pub units: Vec<OmniResidualUnit>,
    pub alpha: Vec<f32>,
    /// [Cout, Cin * K]
    pub down_weight: Mat,
    pub down_bias: Vec<f32>,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Default for OmniEncoderBlock {
    fn default() -> Self {
        OmniEncoderBlock {
            units: Vec::new(),
            alpha: Vec::new(),
            down_weight: Mat::zeros(0, 0),
            down_bias: Vec::new(),
            out_channels: 0,
            kernel: 0,
            stride: 1,
            padding: 0,
        }
    }
}

impl OmniEncoderBlock {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut h = x.clone();
        for unit in &self.units {
            h = unit.forward(&h);
        }
        snake1d_inplace(&mut h, &self.alpha);
        conv1d_dense(
            &h,
            &self.down_weight,
            self.out_channels,
            self.kernel,
            &self.down_bias,
            1,
            self.padding,
            self.stride,
        )
    }
}

/// The acoustic half: raw 24 kHz samples to `hidden_size`-wide frames.
#[derive(Clone, Debug)]
pub struct OmniAcousticEncoder {
    /// `acoustic_encoder.conv1`: [encoder_dim, 1 * 7].
    pub in_weight: Mat,
    pub in_bias: Vec<f32>,
    pub blocks: Vec<OmniEncoderBlock>,
    pub out_alpha: Vec<f32>,
    /// `acoustic_encoder.conv2`: [acoustic_dim, C * 3].
    pub out_weight: Mat,
    pub out_bias: Vec<f32>,
}

impl Default for OmniAcousticEncoder {
    fn default() -> Self {
        OmniAcousticEncoder {
            in_weight: Mat::zeros(0, 0),
            in_bias: Vec::new(),
            blocks: Vec::new(),
            out_alpha: Vec::new(),
            out_weight: Mat::zeros(0, 0),
            out_bias: Vec::new(),
        }
    }
}

impl OmniAcousticEncoder {
    /// [T] samples to [T / hop_length, acoustic_dim].
    pub fn forward(&self, samples: &[f32], hop_length: usize) -> Result<Mat, String> {
        if hop_length == 0 || samples.len() % hop_length != 0 {
            return Err(format!(
                "omnivoice codec: {} samples do not divide into frames of {}",
                samples.len(),
                hop_length
            ));
        }
        let frames = samples.len() / hop_length;
        if frames == 0 {
            return Err("omnivoice codec: the reference clip is shorter than one frame".into());
        }

        let h = Mat::new(samples.to_vec(), samples.len(), 1);
        let mut h = conv1d_dense(&h, &self.in_weight, self.in_weight.rows, 7, &self.in_bias, 1, 3, 1);
        debug_assert!(
            h.rows == samples.len(),
            "omnivoice codec: the input convolution changed the length"
        );

        for block in &self.blocks {
            h = block.forward(&h);
        }
        if h.rows != frames {
            return Err(format!(
                "omnivoice codec: the acoustic encoder produced {} frames from {} samples, \
                 expected {}",
                h.rows,
                samples.len(),
                frames
            ));
        }

        snake1d_inplace(&mut h, &self.out_alpha);
        Ok(conv1d_dense(&h, &self.out_weight, self.out_weight.rows, 3, &self.out_bias, 1, 1, 1))
    }
}

/// ELU at alpha 1, the semantic stack's activation.
fn elu_inplace(x: &mut Mat) {
    for v in x.data.iter_mut() {
        *v = if *v > 0.0f32 { *v } else { v.exp_m1() };
    }
}

/// A residual unit in the semantic stack.
///
/// Same shape as the acoustic one and none of the same details: ELU rather
/// than Snake, kernel 3 rather than 7, and no biases at all.
#[derive(Clone, Debug)]
pub struct OmniSemanticResidualUnit {
    /// [C, C * 3]
    pub conv1_weight: Mat,
    /// [C, C]
    pub conv2_weight: Mat,
    pub dilation: usize,
}

impl Default for OmniSemanticResidualUnit {
    fn default() -> Self {
        OmniSemanticResidualUnit {
            conv1_weight: Mat::zeros(0, 0),
            conv2_weight: Mat::zeros(0, 0),
            dilation: 1,
        }
    }
}

impl OmniSemanticResidualUnit {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut h = x.clone();
        elu_inplace(&mut h);
        let mut h = conv1d_dense(
            &h,
            &self.conv1_weight,
            self.conv1_weight.rows,
            3,
            &[],
            self.dilation,
            self.dilation,
            1,
        );
        debug_assert!(
            h.rows == x.rows,
            "omnivoice codec: semantic residual unit changed the length"
        );
        elu_inplace(&mut h);
        let mut h = conv1d_pointwise(&h, &self.conv2_weight, &[]);
        h.add_assign(x);
        h
    }
}

/// Residual units, then a convolution. Both of OmniVoice's blocks run at
/// stride 1, so this stack reshapes the features without resampling them.
#[derive(Clone, Debug)]
pub struct OmniSemanticBlock {
    pub units: Vec<OmniSemanticResidualUnit>,
    /// [Cout, Cin * K]
    pub conv_weight: Mat,
    pub conv_bias: Vec<f32>,
    pub out_channels: usize,
    pub kernel: usize,
    pub stride: usize,
    pub padding: usize,
}

impl Default for OmniSemanticBlock {
    fn default() -> Self {
        OmniSemanticBlock {
            units: Vec::new(),
            conv_weight: Mat::zeros(0, 0),
            conv_bias: Vec::new(),
            out_channels: 0,
            kernel: 3,
            stride: 1,
            padding: 1,
        }
    }
}

impl OmniSemanticBlock {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut h = x.clone();
        for unit in &self.units {
            h = unit.forward(&h);
        }
        conv1d_dense(
            &h,
            &self.conv_weight,
            self.out_channels,
            self.kernel,
            &self.conv_bias,
            1,
            self.padding,
            self.stride,
        )
    }
}

/// The convolutional adapter between HuBERT's features and the quantizer.
#[derive(Clone, Debug)]
pub struct OmniSemanticEncoder {
    /// `encoder_semantic.conv`: [C, C * 3], no bias.
    pub in_weight: Mat,
    pub blocks: Vec<OmniSemanticBlock>,
}

impl Default for OmniSemanticEncoder {
    fn default() -> Self {
        OmniSemanticEncoder { in_weight: Mat::zeros(0, 0), blocks: Vec::new() }
    }
}

impl OmniSemanticEncoder {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut h = conv1d_dense(x, &self.in_weight, self.in_weight.rows, 3, &[], 1, 1, 1);
        for block in &self.blocks {
            h = block.forward(&h);
        }
        h
    }
}

/// Waveform to codes.
#[derive(Clone, Debug)]
pub struct OmniCodecEncoder {
    pub config: OmniCodecConfig,
    pub semantic: HubertModel,
    pub semantic_adapter: OmniSemanticEncoder,
    pub acoustic: OmniAcousticEncoder,

    /// `fc`: the concatenated [acoustic | semantic] width, mixed in place.
    pub fc_weight: Mat,
    pub fc_bias: Vec<f32>,

    pub quantizer: OmniQuantizer,
}

impl OmniCodecEncoder {
    /// Load the analysis-side tensors from an `omnivoice-tokenizer` GGUF.
    ///
    /// Roughly the complement of what `OmniCodecDecoder::load` reads, plus the
    /// codebooks, which both halves need.
    pub fn load(path: &str, cfg: OmniCodecConfig) -> Result<OmniCodecEncoder, String> {
        let gguf = open_codec_gguf(path)?;
        let c = cfg;

        if c.upsample_factor() != c.hop_length {
            return Err(format!(
                "omnivoice codec: upsampling ratios multiply to {} but hop_length is {}",
                c.upsample_factor(),
                c.hop_length
            ));
        }

        // ---- quantizer: codebooks plus both projections ----
        let mut quantizer = OmniQuantizer { levels: Vec::new(), latent_dim: c.latent_dim };
        for i in 0..c.n_codebooks {
            let base = format!("quantizer.quantizers.{}", i);
            let mut level = OmniQuantizerLevel::default();

            let embed = conv_weight(&gguf, &format!("{}.codebook.embed", base), c.codebook_size)?;
            if embed.cols != c.codebook_dim {
                return Err(format!(
                    "omnivoice codec: codebook {} is {}x{}",
                    i, embed.rows, embed.cols
                ));
            }
            level.codebook = embed;

            let pin = conv_weight(&gguf, &format!("{}.project_in.weight", base), c.codebook_dim)?;
            if pin.cols != c.latent_dim {
                return Err(format!(
                    "omnivoice codec: codebook {} project_in is {}x{}",
                    i, pin.rows, pin.cols
                ));
            }
            level.project_in_weight = pin;
            level.project_in_bias = vector(&gguf, &format!("{}.project_in.bias", base))?;

            level.project_out_weight =
                conv_weight(&gguf, &format!("{}.project_out.weight", base), c.latent_dim)?;
            level.project_out_bias = vector(&gguf, &format!("{}.project_out.bias", base))?;

            quantizer.levels.push(level);
        }

        // ---- acoustic encoder ----
        let mut acoustic = OmniAcousticEncoder::default();
        let in_w = conv_weight(&gguf, "acoustic_encoder.conv1.weight", c.encoder_dim)?;
        if in_w.cols != 7 {
            return Err(format!(
                "omnivoice codec: acoustic_encoder.conv1 has {} taps, expected 7 over one input \
                 channel",
                in_w.cols
            ));
        }
        acoustic.in_weight = in_w;
        acoustic.in_bias = vector(&gguf, "acoustic_encoder.conv1.bias")?;

        let mut channels = c.encoder_dim;
        for (i, &stride) in c.upsampling_ratios.iter().enumerate() {
            let base = format!("acoustic_encoder.block.{}", i);
            let mut block = OmniEncoderBlock::default();

            // Three residual units at dilations 1, 3 and 9, all on the block's
            // *input* width -- they run before the downsample.
            for u in 0..3 {
                let ub = format!("{}.res_unit{}", base, u + 1);
                let mut unit = OmniResidualUnit {
                    dilation: if u == 0 { 1 } else if u == 1 { 3 } else { 9 },
                    ..Default::default()
                };

                let a1 = alpha(&gguf, &format!("{}.snake1.alpha", ub))?;
                if a1.len() != channels {
                    return Err(format!(
                        "omnivoice codec: {}.snake1.alpha has {} channels, expected {}",
                        ub,
                        a1.len(),
                        channels
                    ));
                }
                unit.alpha1 = a1;
                unit.conv1_weight = conv_weight(&gguf, &format!("{}.conv1.weight", ub), channels)?;
                unit.conv1_bias = vector(&gguf, &format!("{}.conv1.bias", ub))?;

                unit.alpha2 = alpha(&gguf, &format!("{}.snake2.alpha", ub))?;
                unit.conv2_weight = conv_weight(&gguf, &format!("{}.conv2.weight", ub), channels)?;
                unit.conv2_bias = vector(&gguf, &format!("{}.conv2.bias", ub))?;

                block.units.push(unit);
            }

            block.alpha = alpha(&gguf, &format!("{}.snake1.alpha", base))?;

            // Kernel 2s with padding ceil(s/2) is what turns a multiple of s into
            // exactly that multiple divided by s, for odd and even strides alike.
            block.out_channels = channels * 2;
            block.stride = stride;
            block.kernel = 2 * stride;
            block.padding = stride.div_ceil(2);
            let dw = conv_weight(&gguf, &format!("{}.conv1.weight", base), block.out_channels)?;
            if dw.cols != channels * block.kernel {
                return Err(format!(
                    "omnivoice codec: {}.conv1 is {}x{}, expected {}x{}",
                    base,
                    dw.rows,
                    dw.cols,
                    block.out_channels,
                    channels * block.kernel
                ));
            }
            block.down_weight = dw;
            block.down_bias = vector(&gguf, &format!("{}.conv1.bias", base))?;

            channels = block.out_channels;
            acoustic.blocks.push(block);
        }

        let out_alpha = alpha(&gguf, "acoustic_encoder.snake1.alpha")?;
        if out_alpha.len() != channels {
            return Err(format!(
                "omnivoice codec: acoustic_encoder.snake1.alpha has {} channels, expected {}",
                out_alpha.len(),
                channels
            ));
        }
        acoustic.out_alpha = out_alpha;
        let out_w = conv_weight(&gguf, "acoustic_encoder.conv2.weight", c.decoder_in_dim)?;
        if out_w.cols != channels * 3 {
            return Err(format!(
                "omnivoice codec: acoustic_encoder.conv2 is {}x{}, expected {}x{}",
                out_w.rows,
                out_w.cols,
                c.decoder_in_dim,
                channels * 3
            ));
        }
        acoustic.out_weight = out_w;
        acoustic.out_bias = vector(&gguf, "acoustic_encoder.conv2.bias")?;

        // ---- semantic model and its adapter ----
        let semantic = HubertModel::load(&gguf, HubertConfig::omnivoice_semantic(), "semantic_model")?;
        if semantic.config.hidden_size != c.semantic_dim() {
            return Err(format!(
                "omnivoice codec: the semantic model is {} wide but the quantizer leaves {} for it",
                semantic.config.hidden_size,
                c.semantic_dim()
            ));
        }

        let sem = c.semantic_dim();
        let mut semantic_adapter = OmniSemanticEncoder::default();
        let sem_in = conv_weight(&gguf, "encoder_semantic.conv.weight", sem)?;
        if sem_in.cols != sem * 3 {
            return Err(format!(
                "omnivoice codec: encoder_semantic.conv is {}x{}",
                sem_in.rows, sem_in.cols
            ));
        }
        semantic_adapter.in_weight = sem_in;

        // Both blocks run at stride 1 and keep the width, which is what
        // channel_ratios (1, 1) and strides (1, 1) mean.
        for i in 0..2 {
            let base = format!("encoder_semantic.conv_blocks.{}", i);
            let mut block = OmniSemanticBlock { out_channels: sem, ..Default::default() };

            for u in 0..2 {
                let ub = format!("{}.res_units.{}", base, u);
                let unit = OmniSemanticResidualUnit {
                    dilation: 1,
                    conv1_weight: conv_weight(&gguf, &format!("{}.conv1.weight", ub), sem)?,
                    conv2_weight: conv_weight(&gguf, &format!("{}.conv2.weight", ub), sem)?,
                };
                block.units.push(unit);
            }

            block.conv_weight = conv_weight(&gguf, &format!("{}.conv.weight", base), sem)?;
            block.conv_bias = vector(&gguf, &format!("{}.conv.bias", base))?;

            semantic_adapter.blocks.push(block);
        }

        // ---- fc: mix the two paths ----
        let fc = conv_weight(&gguf, "fc.weight", c.latent_dim)?;
        if fc.cols != c.latent_dim {
            return Err(format!(
                "omnivoice codec: fc is {}x{}, expected a square {}",
                fc.rows, fc.cols, c.latent_dim
            ));
        }
        let fc_bias = vector(&gguf, "fc.bias")?;

        Ok(OmniCodecEncoder {
            config: c,
            semantic,
            semantic_adapter,
            acoustic,
            fc_weight: fc,
            fc_bias,
            quantizer,
        })
    }

    /// Mono 24 kHz samples to `n_codebooks` streams of indices.
    ///
    /// Trailing samples that do not complete a frame are dropped, which is
    /// what the reference does: a partial frame would put the two paths'
    /// lengths out of step and force a padding branch.
    pub fn encode(&self, samples: &[f32]) -> Result<Vec<Vec<u32>>, String> {
        let hop = self.config.hop_length;
        let frames = samples.len() / hop;
        if frames == 0 {
            return Err(format!(
                "omnivoice codec: the clip is shorter than one {}-sample frame",
                hop
            ));
        }
        // Drop the partial frame rather than pad it: a ragged tail would put the
        // acoustic and semantic paths' lengths out of step.
        let trimmed = &samples[..frames * hop];

        // ---- semantic path ----
        let semantic_rate = self.semantic.config.downsample_factor();
        let resampled = resample(trimmed, self.config.sample_rate, SEMANTIC_SAMPLE_RATE);

        // The reference pads by half the semantic hop on each side. It is not a
        // "same" padding of anything -- it is what makes the frame count come out
        // at twice the codec's rate, which the assertion below is the real check on.
        let pad = semantic_rate / 2;
        let mut padded: Vec<f32> = Vec::with_capacity(resampled.len() + 2 * pad);
        padded.extend(std::iter::repeat_n(0.0f32, pad));
        padded.extend_from_slice(&resampled);
        padded.extend(std::iter::repeat_n(0.0f32, pad));
        drop(resampled);

        let hidden = self.semantic.mean_hidden_states(&padded)?;
        if hidden.rows != frames * 2 {
            return Err(format!(
                "omnivoice codec: the semantic model produced {} frames where {} were expected",
                hidden.rows,
                frames * 2
            ));
        }

        // 50 Hz down to the codec's 25 Hz by keeping every other frame -- a plain
        // decimation, not an average.
        let semantic_features = Mat::from_fn(frames, hidden.cols, |r, c| hidden.at(r * 2, c));
        let sem = self.semantic_adapter.forward(&semantic_features);
        if sem.rows != frames {
            return Err("omnivoice codec: the semantic adapter changed the frame count".into());
        }

        // ---- acoustic path ----
        let aco = self.acoustic.forward(trimmed, hop)?;

        // ---- concatenate, mix, quantize ----
        if aco.cols + sem.cols != self.config.latent_dim {
            return Err(format!(
                "omnivoice codec: the two paths are {} and {} wide, which do not fill {}",
                aco.cols, sem.cols, self.config.latent_dim
            ));
        }
        let latent = self.config.latent_dim;
        let mut joined = Mat::zeros(frames, latent);
        for t in 0..frames {
            let row = &mut joined.data[t * latent..(t + 1) * latent];
            // Acoustic first, then semantic: the order fc was trained with.
            for c in 0..aco.cols {
                row[c] = aco.at(t, c);
            }
            for c in 0..sem.cols {
                row[aco.cols + c] = sem.at(t, c);
            }
        }

        let latents = conv1d_pointwise(&joined, &self.fc_weight, &self.fc_bias);
        self.quantizer.to_codes(&latents)
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.semantic.parameter_count();
        let add_mat = |m: &Mat, n: &mut usize| *n += m.data.len();
        let add_vec = |v: &Vec<f32>, n: &mut usize| *n += v.len();

        add_mat(&self.acoustic.in_weight, &mut n);
        add_vec(&self.acoustic.in_bias, &mut n);
        for b in &self.acoustic.blocks {
            add_vec(&b.alpha, &mut n);
            add_mat(&b.down_weight, &mut n);
            add_vec(&b.down_bias, &mut n);
            for u in &b.units {
                add_vec(&u.alpha1, &mut n);
                add_vec(&u.alpha2, &mut n);
                add_mat(&u.conv1_weight, &mut n);
                add_mat(&u.conv2_weight, &mut n);
                add_vec(&u.conv1_bias, &mut n);
                add_vec(&u.conv2_bias, &mut n);
            }
        }
        add_vec(&self.acoustic.out_alpha, &mut n);
        add_mat(&self.acoustic.out_weight, &mut n);
        add_vec(&self.acoustic.out_bias, &mut n);

        add_mat(&self.semantic_adapter.in_weight, &mut n);
        for b in &self.semantic_adapter.blocks {
            add_mat(&b.conv_weight, &mut n);
            add_vec(&b.conv_bias, &mut n);
            for u in &b.units {
                add_mat(&u.conv1_weight, &mut n);
                add_mat(&u.conv2_weight, &mut n);
            }
        }

        add_mat(&self.fc_weight, &mut n);
        add_vec(&self.fc_bias, &mut n);
        for l in &self.quantizer.levels {
            add_mat(&l.codebook, &mut n);
            add_mat(&l.project_in_weight, &mut n);
            add_vec(&l.project_in_bias, &mut n);
            add_mat(&l.project_out_weight, &mut n);
            add_vec(&l.project_out_bias, &mut n);
        }
        n
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conv1d::conv1d_out_len;
    use crate::wav::wave_stats;

    const CODEC_PATH: &str = "models/omnivoice-tokenizer-Q8_0.gguf";

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    fn codec_present() -> bool {
        if std::path::Path::new(CODEC_PATH).exists() {
            return true;
        }
        eprintln!("skip: {} not present", CODEC_PATH);
        false
    }

    /// A quantizer whose codebooks are readable off the output: code `c` of
    /// codebook `i` projects to `c` in column 0 and `i` in column 1.
    fn toy_quantizer(n_codebooks: usize, latent_dim: usize) -> OmniQuantizer {
        let mut q = OmniQuantizer { levels: Vec::new(), latent_dim };
        for i in 0..n_codebooks {
            let mut level = OmniQuantizerLevel {
                codebook: Mat::from_fn(8, 2, |r, c| if c == 0 { r as f32 } else { i as f32 }),
                project_out_weight: Mat::zeros(latent_dim, 2),
                ..Default::default()
            };
            *level.project_out_weight.at_mut(0, 0) = 1.0;
            *level.project_out_weight.at_mut(1, 1) = 1.0;
            q.levels.push(level);
        }
        q
    }

    /// A toy quantizer that can also encode: `project_in` is the identity onto the
    /// first two latent columns, so a latent is its own codebook coordinate.
    fn toy_encoding_quantizer(n_codebooks: usize, latent_dim: usize) -> OmniQuantizer {
        let mut q = toy_quantizer(n_codebooks, latent_dim);
        for level in &mut q.levels {
            level.project_in_weight = Mat::zeros(2, latent_dim);
            *level.project_in_weight.at_mut(0, 0) = 1.0;
            *level.project_in_weight.at_mut(1, 1) = 1.0;
        }
        q
    }

    // =========================================================================
    // Config
    // =========================================================================

    #[test]
    fn omnivoice_codec_upsamples_by_exactly_its_hop_length() {
        let c = OmniCodecConfig::defaults();
        // 8 * 5 * 4 * 2 * 3 == 960, and 24000 / 960 == 25 Hz.
        assert_eq!(c.upsample_factor(), 960);
        assert_eq!(c.upsample_factor(), c.hop_length);
        assert_eq!(c.sample_rate / c.hop_length, 25);
        // Eight codes per frame at 25 Hz is 200 audio tokens a second.
        assert_eq!(c.n_codebooks * (c.sample_rate / c.hop_length), 200);
    }

    #[test]
    fn omnivoice_channel_widths_halve_to_the_output() {
        let c = OmniCodecConfig::defaults();
        let mut ch = c.decoder_dim;
        for _ in 0..c.upsampling_ratios.len() {
            ch /= 2;
        }
        assert_eq!(c.decoder_dim, 1024);
        assert_eq!(ch, 32); // 1024 -> 512 -> 256 -> 128 -> 64 -> 32
    }

    // =========================================================================
    // Quantizer
    // =========================================================================

    #[test]
    fn from_codes_sums_every_codebook_at_one_rate() {
        // Unlike SNAC there are no strides: all codebooks run at the frame rate,
        // so reconstruction is a plain sum and there is no interleave to get wrong.
        let q = toy_quantizer(4, 4);
        let codes = vec![vec![1u32, 2], vec![3, 4], vec![5, 6], vec![7, 0]];
        let z = q.from_codes(&codes).expect("from_codes");
        assert_eq!(z.rows, 2);
        assert_eq!(z.cols, 4);
        // Column 0 is the sum of the code values at that frame.
        assert!(approx(z.at(0, 0), 1.0 + 3.0 + 5.0 + 7.0));
        assert!(approx(z.at(1, 0), 2.0 + 4.0 + 6.0 + 0.0));
        // Column 1 is the sum of the codebook indices, once per frame.
        assert!(approx(z.at(0, 1), 0.0 + 1.0 + 2.0 + 3.0));
        assert!(approx(z.at(1, 1), 6.0));
    }

    #[test]
    fn from_codes_rejects_ragged_codebooks() {
        let q = toy_quantizer(3, 4);
        let z = q.from_codes(&[vec![1, 2], vec![3], vec![5, 6]]);
        let err = z.expect_err("ragged codebooks must be rejected");
        assert!(err.contains("expected 2"));
    }

    #[test]
    fn from_codes_rejects_the_wrong_codebook_count() {
        let q = toy_quantizer(8, 4);
        let err = q.from_codes(&[vec![1], vec![2]]).expect_err("wrong count");
        assert!(err.contains("expected 8 codebooks"));
    }

    #[test]
    fn from_codes_rejects_an_out_of_range_code() {
        let q = toy_quantizer(2, 4);
        let err = q.from_codes(&[vec![1], vec![999]]).expect_err("out of range");
        assert!(err.contains("out of range"));
    }

    #[test]
    fn from_codes_rejects_an_empty_request() {
        let q = toy_quantizer(2, 4);
        assert!(q.from_codes(&[vec![], vec![]]).is_err());
    }

    // =========================================================================
    // Layers
    // =========================================================================

    #[test]
    fn omni_residual_unit_preserves_length_at_every_dilation() {
        const CHANNELS: usize = 8;
        for dilation in [1usize, 3, 9] {
            let unit = OmniResidualUnit {
                dilation,
                alpha1: vec![1.0; CHANNELS],
                alpha2: vec![1.0; CHANNELS],
                conv1_weight: Mat::zeros(CHANNELS, CHANNELS * 7),
                conv2_weight: Mat::zeros(CHANNELS, CHANNELS),
                ..Default::default()
            };

            let x = Mat::ones(48, CHANNELS);
            let out = unit.forward(&x);
            assert_eq!(out.rows, 48);
            assert_eq!(out.cols, CHANNELS);
            // Both convolutions are zeroed, so only the skip survives.
            for &v in &out.data {
                assert!(approx(v, 1.0));
            }
        }
    }

    #[test]
    fn omni_decoder_block_multiplies_length_by_its_stride() {
        // The odd strides matter: OmniVoice upsamples by 5 and 3, where SNAC only
        // ever used powers of two, and the output-padding term is what keeps the
        // multiple exact for those.
        for stride in [2usize, 3, 4, 5, 8] {
            const C_IN: usize = 16;
            let c_out = C_IN / 2;

            let mut block = OmniDecoderBlock {
                stride,
                kernel: 2 * stride,
                padding: stride.div_ceil(2),
                output_padding: stride % 2,
                out_channels: c_out,
                alpha: vec![1.0; C_IN],
                up_weight: Mat::zeros(C_IN, c_out * 2 * stride),
                up_bias: vec![0.0; c_out],
                units: Vec::new(),
            };
            for u in 0..3 {
                block.units.push(OmniResidualUnit {
                    dilation: if u == 0 { 1 } else if u == 1 { 3 } else { 9 },
                    alpha1: vec![1.0; c_out],
                    alpha2: vec![1.0; c_out],
                    conv1_weight: Mat::zeros(c_out, c_out * 7),
                    conv2_weight: Mat::zeros(c_out, c_out),
                    ..Default::default()
                });
            }

            let out = block.forward(&Mat::ones(7, C_IN));
            assert_eq!(out.rows, 7 * stride);
            assert_eq!(out.cols, c_out);
        }
    }

    #[test]
    fn the_full_upsampling_chain_multiplies_by_960() {
        // End to end through the real ratios, with random weights: 5 frames must
        // become exactly 4800 samples. Every padding mistake breaks the multiple.
        let cfg = OmniCodecConfig::defaults();
        let mut t = 5usize;
        for &stride in &cfg.upsampling_ratios {
            t *= stride;
        }
        assert_eq!(t, 5 * 960);
        assert_eq!(t, 4800);
    }

    // =========================================================================
    // The real checkpoint
    // =========================================================================

    #[test]
    fn omni_codec_decoder_loads_the_tokenizer_gguf() {
        if !codec_present() {
            return;
        }
        let d = OmniCodecDecoder::load(CODEC_PATH, OmniCodecConfig::defaults())
            .unwrap_or_else(|e| panic!("{}", e));

        assert_eq!(d.blocks.len(), 5);
        assert_eq!(d.quantizer.levels.len(), 8);

        let want_stride = [8usize, 5, 4, 2, 3];
        let want_out = [512usize, 256, 128, 64, 32];
        for i in 0..5 {
            assert_eq!(d.blocks[i].stride, want_stride[i]);
            assert_eq!(d.blocks[i].kernel, 2 * want_stride[i]);
            assert_eq!(d.blocks[i].out_channels, want_out[i]);
            assert_eq!(d.blocks[i].units.len(), 3);
            // Transposed convolutions are [Cin, Cout * K], biased per output.
            assert_eq!(d.blocks[i].up_weight.cols, want_out[i] * d.blocks[i].kernel);
            assert_eq!(d.blocks[i].up_bias.len(), want_out[i]);
        }

        // Codebooks are 1024 x 64, projected up to the 1024-wide latent.
        for level in &d.quantizer.levels {
            assert_eq!(level.codebook.rows, 1024);
            assert_eq!(level.codebook.cols, 64);
            assert_eq!(level.project_out_weight.rows, 1024);
            assert_eq!(level.project_out_weight.cols, 64);
        }

        assert_eq!(d.out_alpha.len(), 32);
        assert_eq!(d.out_weight.rows, 1);
        assert_eq!(d.out_weight.cols, 32 * 7);

        let params = d.parameter_count();
        assert!(params > 15_000_000, "decoder parameters: {}", params);
        assert!(params < 30_000_000, "decoder parameters: {}", params);
    }

    #[test]
    fn omni_codec_decoder_produces_exactly_frames_times_960_samples() {
        if !codec_present() {
            return;
        }
        let d = OmniCodecDecoder::load(CODEC_PATH, OmniCodecConfig::defaults()).expect("load");

        const FRAMES: usize = 25; // one second
        let mut codes = vec![vec![0u32; FRAMES]; 8];
        for c in 0..8 {
            for t in 0..FRAMES {
                codes[c][t] = ((t * 37 + c * 101) % 1024) as u32;
            }
        }

        let audio = d.decode(&codes).unwrap_or_else(|e| panic!("{}", e));
        assert_eq!(audio.len(), FRAMES * 960);
        assert_eq!(audio.len(), 24000);

        // Arbitrary codes are not speech, so only the bound is asserted -- but the
        // final tanh has to hold it, or the int16 conversion downstream is unsafe.
        let s = wave_stats(&audio);
        assert!(s.in_range(), "{}", s.describe());
    }

    #[test]
    fn omni_codec_decoder_is_deterministic() {
        // There is no noise block anywhere in this decoder, unlike SNAC, so the
        // same codes must give bit-identical samples with no seed involved.
        if !codec_present() {
            return;
        }
        let d = OmniCodecDecoder::load(CODEC_PATH, OmniCodecConfig::defaults()).expect("load");

        let mut codes = vec![vec![0u32; 4]; 8];
        for c in 0..8 {
            for t in 0..4 {
                codes[c][t] = (c * 13 + t) as u32;
            }
        }
        let a = d.decode(&codes).expect("first decode");
        let b = d.decode(&codes).expect("second decode");
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert!(a[i] == b[i]);
        }
    }

    #[test]
    fn omni_codec_decoder_rejects_a_mismatched_config() {
        if !codec_present() {
            return;
        }
        let mut wrong = OmniCodecConfig::defaults();
        wrong.decoder_dim = 1536;
        let err = OmniCodecDecoder::load(CODEC_PATH, wrong).expect_err("mismatched config");
        assert!(err.contains("1536"), "{}", err);
    }

    #[test]
    fn omni_codec_decoder_rejects_ratios_that_disagree_with_hop_length() {
        // The two are stated independently in the checkpoint metadata, and a
        // disagreement means every length downstream would be wrong.
        let mut wrong = OmniCodecConfig::defaults();
        wrong.upsampling_ratios = vec![8, 5, 4, 2]; // 320, not 960
        assert!(OmniCodecDecoder::load("models/does-not-exist.gguf", wrong).is_err());
    }

    // =========================================================================
    // Analysis -- the quantizer
    // =========================================================================

    #[test]
    fn to_codes_picks_the_nearest_codebook_entry() {
        // One codebook of eight entries at (0,0), (1,0) ... (7,0). A latent at
        // 2.4 has to land on entry 2 and one at 2.6 on entry 3.
        let q = toy_encoding_quantizer(1, 4);
        let mut latents = Mat::zeros(4, 4);
        *latents.at_mut(0, 0) = 2.4;
        *latents.at_mut(1, 0) = 2.6;
        *latents.at_mut(2, 0) = -5.0; // below every entry
        *latents.at_mut(3, 0) = 100.0; // above every entry

        let codes = q.to_codes(&latents).expect("to_codes");
        assert_eq!(codes.len(), 1);
        assert_eq!(codes[0], vec![2u32, 3, 0, 7]);
    }

    #[test]
    fn to_codes_quantizes_the_residual_not_the_signal() {
        // Two codebooks over the same eight entries. The first takes the latent,
        // the second takes what is left after the first has been subtracted -- so
        // a latent of exactly 3 leaves nothing and the second codebook picks 0.
        let q = toy_encoding_quantizer(2, 4);
        let mut latents = Mat::zeros(2, 4);
        *latents.at_mut(0, 0) = 3.0;
        *latents.at_mut(1, 0) = 5.5;

        let codes = q.to_codes(&latents).expect("to_codes");
        assert_eq!(codes.len(), 2);
        assert_eq!(codes[0][0], 3);
        assert_eq!(codes[1][0], 0);
        // 5.5 rounds to 5 or 6; whichever it takes, the leftover is half a step,
        // which the second codebook can only round back to nothing.
        assert_eq!(codes[1][1], 0);
    }

    #[test]
    fn from_codes_inverts_to_codes_on_the_codebook_grid() {
        // Latents that sit exactly on an entry survive the round trip, which is
        // the strongest statement a lossy quantizer can make.
        let q = toy_encoding_quantizer(1, 4);
        let mut latents = Mat::zeros(5, 4);
        for t in 0..5 {
            *latents.at_mut(t, 0) = (t + 1) as f32;
        }
        let codes = q.to_codes(&latents).expect("to_codes");
        let back = q.from_codes(&codes).expect("from_codes");
        for t in 0..5 {
            assert!(approx(back.at(t, 0), (t + 1) as f32));
        }
    }

    #[test]
    fn to_codes_refuses_a_quantizer_loaded_for_synthesis_only() {
        // A decoder-only load leaves project_in empty, and guessing it would be
        // worse than failing.
        let q = toy_quantizer(2, 4);
        assert!(q.to_codes(&Mat::zeros(3, 4)).is_err());
    }

    #[test]
    fn to_codes_checks_the_latent_width() {
        let q = toy_encoding_quantizer(1, 4);
        assert!(q.to_codes(&Mat::zeros(3, 5)).is_err());
    }

    // =========================================================================
    // Analysis -- the convolution stacks
    // =========================================================================

    #[test]
    fn an_encoder_block_divides_the_length_by_its_stride() {
        // The mirror of the decoder's invariant, and the one that makes
        // frames * 960 exact in the analysis direction too. Kernel 2s with padding
        // ceil(s/2) is what makes it hold for odd strides as well as even ones.
        for stride in [8usize, 5, 4, 2, 3] {
            let channels = 4usize;
            let out_channels = channels * 2;
            let kernel = 2 * stride;
            let block = OmniEncoderBlock {
                alpha: vec![1.0; channels],
                out_channels,
                stride,
                kernel,
                padding: stride.div_ceil(2),
                down_weight: Mat::from_fn(out_channels, channels * kernel, |r, c| {
                    0.01f32 * ((r + c) % 7) as f32
                }),
                down_bias: vec![0.0; out_channels],
                units: Vec::new(),
            };

            for frames in [1usize, 3, 20] {
                let x = Mat::from_fn(frames * stride, channels, |r, c| {
                    0.001f32 * ((r * 3 + c) % 11) as f32
                });
                let y = block.forward(&x);
                assert_eq!(y.rows, frames);
                assert_eq!(y.cols, block.out_channels);
            }
        }
    }

    #[test]
    fn the_full_downsampling_chain_divides_by_960() {
        let c = OmniCodecConfig::defaults();
        let mut t = 960 * 7;
        for &stride in &c.upsampling_ratios {
            let kernel = 2 * stride;
            let padding = stride.div_ceil(2);
            t = conv1d_out_len(t, kernel, 1, padding, stride);
        }
        assert_eq!(t, 7);
    }

    #[test]
    fn a_semantic_residual_unit_preserves_length() {
        // Kernel 3 at padding 1, unlike the acoustic units' 7 at 3.
        let channels = 6usize;
        let unit = OmniSemanticResidualUnit {
            dilation: 1,
            conv1_weight: Mat::from_fn(channels, channels * 3, |r, c| {
                0.01f32 * ((r + 2 * c) % 5) as f32 - 0.02f32
            }),
            conv2_weight: Mat::from_fn(channels, channels, |r, c| if r == c { 0.5 } else { 0.0 }),
        };

        let x = Mat::from_fn(17, channels, |r, c| 0.05f32 * ((r + c) % 4) as f32);
        let y = unit.forward(&x);
        assert_eq!(y.rows, 17);
        assert_eq!(y.cols, channels);
        for &v in &y.data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn the_semantic_width_is_whatever_the_acoustic_path_leaves() {
        let c = OmniCodecConfig::defaults();
        assert_eq!(c.semantic_dim(), 768);
        assert_eq!(c.decoder_in_dim + c.semantic_dim(), c.latent_dim);
        assert_eq!(c.encoder_dim, 64);
        // The encoder doubles its width per block, ending at 2048 before the
        // projection down to 256.
        let mut ch = c.encoder_dim;
        for _ in 0..c.upsampling_ratios.len() {
            ch *= 2;
        }
        assert_eq!(ch, 2048);
    }

    // =========================================================================
    // Analysis -- the real checkpoint
    // =========================================================================

    #[test]
    fn omni_codec_encoder_loads_the_tokenizer_gguf() {
        if !codec_present() {
            return;
        }
        let e = OmniCodecEncoder::load(CODEC_PATH, OmniCodecConfig::defaults()).expect("load");
        assert_eq!(e.acoustic.blocks.len(), 5);
        assert_eq!(e.semantic_adapter.blocks.len(), 2);
        assert_eq!(e.quantizer.levels.len(), 8);
        for l in &e.quantizer.levels {
            assert_eq!(l.project_in_weight.rows, 64);
            assert_eq!(l.project_in_weight.cols, 1024);
        }
        // The analysis half is bigger than the synthesis half, almost all of it
        // the 94 M-parameter semantic model.
        assert!(e.parameter_count() > 150_000_000);
    }

    #[test]
    fn omni_codec_encoder_emits_one_code_per_codebook_per_frame() {
        if !codec_present() {
            return;
        }
        let cfg = OmniCodecConfig::defaults();
        let e = OmniCodecEncoder::load(CODEC_PATH, cfg.clone()).expect("load");

        // 40 frames, plus a partial one that has to be dropped rather than padded.
        const FRAMES: usize = 40;
        let mut wav = vec![0.0f32; FRAMES * 960 + 137];
        for (i, w) in wav.iter_mut().enumerate() {
            let t = i as f32 / 24000.0f32;
            *w = 0.15f32
                * ((2.0f32 * 3.14159265f32 * 150.0f32 * t).sin()
                    + 0.4f32 * (2.0f32 * 3.14159265f32 * 900.0f32 * t).sin());
        }

        let codes = e.encode(&wav).expect("encode");
        assert_eq!(codes.len(), cfg.n_codebooks);
        for stream in &codes {
            assert_eq!(stream.len(), FRAMES);
            for &c in stream {
                assert!((c as usize) < cfg.codebook_size);
            }
        }
    }

    #[test]
    fn omni_codec_encoder_is_deterministic() {
        if !codec_present() {
            return;
        }
        let e = OmniCodecEncoder::load(CODEC_PATH, OmniCodecConfig::defaults()).expect("load");

        let mut wav = vec![0.0f32; 20 * 960];
        for (i, w) in wav.iter_mut().enumerate() {
            *w = 0.2f32 * (2.0f32 * 3.14159265f32 * 220.0f32 * i as f32 / 24000.0f32).sin();
        }
        let a = e.encode(&wav).expect("first encode");
        let b = e.encode(&wav).expect("second encode");
        assert_eq!(a, b);
    }

    #[test]
    fn omni_codec_encoder_rejects_a_clip_shorter_than_a_frame() {
        if !codec_present() {
            return;
        }
        let e = OmniCodecEncoder::load(CODEC_PATH, OmniCodecConfig::defaults()).expect("load");
        assert!(e.encode(&vec![0.0f32; 500]).is_err());
    }

    #[test]
    fn the_codec_round_trips_a_waveform_through_its_own_codes() {
        if !codec_present() {
            return;
        }
        // The test that makes the analysis path checkable at all without a
        // reference implementation. Decode a set of codes, encode the waveform
        // back, and the indices have to come out close to what went in --
        // exactly for the coarse codebooks and less so for the fine ones, since
        // those code the residual that survives the first few.
        //
        // Chance agreement is one in 1024. Any real fault in the chain -- a
        // normalisation on the wrong axis, the two paths concatenated in the wrong
        // order, a padding off by one -- lands there.
        let cfg = OmniCodecConfig::defaults();
        let d = OmniCodecDecoder::load(CODEC_PATH, cfg.clone()).expect("load decoder");
        let e = OmniCodecEncoder::load(CODEC_PATH, cfg.clone()).expect("load encoder");

        const FRAMES: usize = 30;
        let mut codes = vec![vec![0u32; FRAMES]; cfg.n_codebooks];
        let mut state: u32 = 12345;
        for stream in codes.iter_mut() {
            for slot in stream.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *slot = (state >> 16) % cfg.codebook_size as u32;
            }
        }

        let wav = d.decode(&codes).expect("decode");
        assert_eq!(wav.len(), FRAMES * cfg.hop_length);

        let back = e.encode(&wav).expect("encode");
        assert_eq!(back[0].len(), FRAMES);

        let mut agree = 0usize;
        for t in 0..FRAMES {
            agree += (codes[0][t] == back[0][t]) as usize;
        }
        // Random codes make a waveform the codec was never fit on, so this is a
        // weaker recovery than real audio gives; it is still two orders of
        // magnitude above chance.
        assert!(agree * 4 >= FRAMES);

        // And the resynthesis of the recovered codes has to track the original.
        let again = d.decode(&back).expect("decode again");
        let mut num = 0.0f64;
        let mut da = 0.0f64;
        let mut db = 0.0f64;
        for i in 0..wav.len() {
            num += wav[i] as f64 * again[i] as f64;
            da += wav[i] as f64 * wav[i] as f64;
            db += again[i] as f64 * again[i] as f64;
        }
        assert!(num / (da * db).sqrt() > 0.5);
    }
}
