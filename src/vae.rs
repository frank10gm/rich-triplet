#![allow(dead_code)]
// =============================================================================
// AutoencoderKL -- the 16-channel image VAE, decoder only
// =============================================================================
//
// A diffusion transformer does not emit pixels. It emits latents, and something
// else has to turn those into an image. FLUX emits 16-channel latents at 1/8
// resolution, so this is the second half of that pipeline: latents in, RGB out.
//
// Only the decoder is implemented, for the same reason `snac` is decode only:
// text-to-image never runs the encoder.
//
// ## The stack
//
//   z [h/8 * w/8, 16]
//     -> conv_in            16 -> 512, 3x3
//     -> mid: Resnet(512), Attention(512), Resnet(512)
//     -> up 3: 3x Resnet(512 -> 512), upsample 2x
//     -> up 2: 3x Resnet(512 -> 512), upsample 2x
//     -> up 1: 3x Resnet(512 -> 256), upsample 2x
//     -> up 0: 3x Resnet(256 -> 128)
//     -> GroupNorm, SiLU, conv_out 128 -> 3, 3x3
//
// Note the ordering: diffusers stores the up blocks coarsest-last, so
// `decoder.up_blocks.0` is the *finest* level and runs last. Loading them in
// file order gives a decoder whose channel counts happen to line up for the
// first two levels and then fail an assertion on the third -- which is the good
// case, because reversing only the resnets and not the upsamplers produces a
// blurred image and no error at all.
//
// ## Latents are stored scaled
//
// The VAE was trained on a latent distribution the diffusion model does not
// use directly. Going back the other way needs both constants:
//
//   z = z_model / scaling_factor + shift_factor       (0.3611, 0.1159)
//
// SD 1.x and SDXL have a scaling factor and no shift, so the shift is the one
// that gets dropped when porting from older code. Dropping it does not break
// the image -- it desaturates it and adds a colour cast, which reads as a
// stylistic choice rather than as a bug.
//
// ## Memory
//
// The last upsampling level runs 128 channels at full resolution. At 1024x1024
// that is 537 MB for a single activation, and a residual unit holds three at
// once. `decode_tiled` splits the latent into overlapping tiles and blends the
// results, which caps the working set at the cost of some redundant
// convolution. At 512x512 the whole-image path is fine.

use std::collections::HashMap;

use crate::autograd2::Mat;
use crate::conv2d::{conv2d_dense, conv2d_pointwise, group_norm, group_norm_inplace, silu_inplace, upsample_nearest2d};
use crate::gguf_loader::f16_to_f32;
use crate::transformer3::{parse_safetensors_header, SafeTensor, SafeTensorEntry};

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct VaeConfig {
    /// Latent channels. 16 for the SD3/FLUX autoencoder, 4 for SD 1.x and SDXL.
    pub latent_channels: usize,
    /// Channel width per level, **finest first**, matching diffusers'
    /// `block_out_channels`.
    pub block_out_channels: Vec<usize>,
    /// Residual blocks per level. The decoder uses `layers_per_block + 1`.
    pub layers_per_block: usize,
    pub norm_groups: usize,
    pub norm_eps: f32,
    /// `z = z_model / scaling_factor + shift_factor`.
    pub scaling_factor: f32,
    pub shift_factor: f32,
}

impl Default for VaeConfig {
    fn default() -> Self {
        VaeConfig {
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }
}

impl VaeConfig {
    pub fn flux() -> Self {
        VaeConfig {
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            norm_groups: 32,
            norm_eps: 1e-6,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    /// Spatial factor between latent and pixel, `2^(levels - 1)`.
    pub fn downsample_factor(&self) -> usize {
        if self.block_out_channels.is_empty() { 1 } else { 1usize << (self.block_out_channels.len() - 1) }
    }
}

// =============================================================================
// Layers
// =============================================================================

/// GroupNorm, SiLU, 3x3 conv, GroupNorm, SiLU, 3x3 conv, plus a residual.
///
/// The shortcut is a 1x1 convolution present **only** when the channel count
/// changes. Adding one unconditionally works numerically -- it would just be an
/// identity the loader has no weights for -- so the presence of the tensor is
/// what decides it.
#[derive(Clone, Debug)]
pub struct VaeResnetBlock {
    pub in_channels: usize,
    pub out_channels: usize,

    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub conv1_weight: Mat, // [out, in * 9]
    pub conv1_bias: Vec<f32>,

    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub conv2_weight: Mat, // [out, out * 9]
    pub conv2_bias: Vec<f32>,

    /// Empty when `in_channels == out_channels`.
    pub shortcut_weight: Mat, // [out, in]
    pub shortcut_bias: Vec<f32>,
}

impl Default for VaeResnetBlock {
    fn default() -> Self {
        VaeResnetBlock {
            in_channels: 0,
            out_channels: 0,
            norm1_weight: Vec::new(),
            norm1_bias: Vec::new(),
            conv1_weight: Mat::zeros(0, 0),
            conv1_bias: Vec::new(),
            norm2_weight: Vec::new(),
            norm2_bias: Vec::new(),
            conv2_weight: Mat::zeros(0, 0),
            conv2_bias: Vec::new(),
            shortcut_weight: Mat::zeros(0, 0),
            shortcut_bias: Vec::new(),
        }
    }
}

impl VaeResnetBlock {
    pub fn forward(&self, x: &Mat, h: usize, w: usize, cfg: &VaeConfig) -> Mat {
        debug_assert!(x.cols == self.in_channels, "VaeResnetBlock: channel mismatch");

        let mut hidden = group_norm(x, cfg.norm_groups, &self.norm1_weight, &self.norm1_bias, cfg.norm_eps);
        silu_inplace(&mut hidden);
        hidden = conv2d_dense(&hidden, h, w, &self.conv1_weight, self.out_channels, 3, 3, &self.conv1_bias, 1, 1, 1);

        group_norm_inplace(&mut hidden, cfg.norm_groups, &self.norm2_weight, &self.norm2_bias, cfg.norm_eps);
        silu_inplace(&mut hidden);
        hidden = conv2d_dense(&hidden, h, w, &self.conv2_weight, self.out_channels, 3, 3, &self.conv2_bias, 1, 1, 1);

        // The shortcut exists only where the width changes; everywhere else the
        // residual is the input itself.
        if self.shortcut_weight.rows > 0 {
            let skip = conv2d_pointwise(x, &self.shortcut_weight, &self.shortcut_bias);
            for (d, s) in hidden.data.iter_mut().zip(&skip.data) {
                *d += *s;
            }
        } else {
            for (d, s) in hidden.data.iter_mut().zip(&x.data) {
                *d += *s;
            }
        }
        hidden
    }
}

/// Single-head self-attention over the spatial positions.
///
/// In `conv2d`'s spatial-major layout the input is already one row per
/// position, so this is ordinary attention with no reshaping: GroupNorm, three
/// 1x1 projections, softmax over `h*w` keys at scale `1/sqrt(C)`, a 1x1 output
/// projection and a residual add.
///
/// diffusers stores the projections as `Linear` in current checkpoints and as
/// 1x1 `Conv2d` in older ones. Both flatten to the same [C, C] matrix, so the
/// loader accepts either name and this code does not care.
#[derive(Clone, Debug)]
pub struct VaeAttentionBlock {
    pub channels: usize,

    pub norm_weight: Vec<f32>,
    pub norm_bias: Vec<f32>,
    pub q_weight: Mat, // [C, C]
    pub q_bias: Vec<f32>,
    pub k_weight: Mat,
    pub k_bias: Vec<f32>,
    pub v_weight: Mat,
    pub v_bias: Vec<f32>,
    pub out_weight: Mat,
    pub out_bias: Vec<f32>,
}

impl Default for VaeAttentionBlock {
    fn default() -> Self {
        VaeAttentionBlock {
            channels: 0,
            norm_weight: Vec::new(),
            norm_bias: Vec::new(),
            q_weight: Mat::zeros(0, 0),
            q_bias: Vec::new(),
            k_weight: Mat::zeros(0, 0),
            k_bias: Vec::new(),
            v_weight: Mat::zeros(0, 0),
            v_bias: Vec::new(),
            out_weight: Mat::zeros(0, 0),
            out_bias: Vec::new(),
        }
    }
}

impl VaeAttentionBlock {
    pub fn forward(&self, x: &Mat, cfg: &VaeConfig) -> Mat {
        debug_assert!(x.cols == self.channels, "VaeAttentionBlock: channel mismatch");
        let n = x.rows;
        let channels = self.channels;

        let normed = group_norm(x, cfg.norm_groups, &self.norm_weight, &self.norm_bias, cfg.norm_eps);
        let q = conv2d_pointwise(&normed, &self.q_weight, &self.q_bias);
        let k = conv2d_pointwise(&normed, &self.k_weight, &self.k_bias);
        let v = conv2d_pointwise(&normed, &self.v_weight, &self.v_bias);

        let scale = 1.0f32 / (channels as f32).sqrt();

        // One head over every spatial position, and by far the most expensive
        // thing in the decoder: `n` is 16384 at a 128x128 latent, so the score
        // matrix alone is 268 M entries and the two products are 275 GFLOP.
        //
        // Both products are matmuls -- `Q K^T` and `P V` -- so they belong in
        // BLAS. Written as the obvious triple loop this stage took 76.6 s of a
        // 102 s decode; blocked into `sgemm` calls it is a fraction of that. It
        // is the same lesson as `QLinear`: a scalar loop over a matrix product
        // is never the right answer once the matrix stops being a vector.
        //
        // The blocking is over query rows, because the full score matrix would
        // be 1 GB at 128x128. A block of 256 queries needs `256 * n` floats,
        // which is 16 MB there and comfortably cache-resident.
        let mut out = Mat::zeros(n, channels);
        const QUERY_BLOCK: usize = 256;

        let mut base = 0;
        while base < n {
            let rows = QUERY_BLOCK.min(n - base);

            // Q block against every key: [rows, C] @ [n, C]^T -> [rows, n].
            let q_block = Mat::new(q.data[base * channels..(base + rows) * channels].to_vec(), rows, channels);
            let mut scores = q_block.matmul_bt(&k);

            // Softmax each row in place, folding in the scale.
            for i in 0..rows {
                let row = &mut scores.data[i * n..(i + 1) * n];
                let mut max_score = f32::NEG_INFINITY;
                for j in 0..n {
                    row[j] *= scale;
                    if max_score < row[j] {
                        max_score = row[j];
                    }
                }
                let mut denom = 0.0f32;
                for j in 0..n {
                    row[j] = (row[j] - max_score).exp();
                    denom += row[j];
                }
                let inv = 1.0f32 / denom;
                for j in 0..n {
                    row[j] *= inv;
                }
            }

            // Weighted sum of V: [rows, n] @ [n, C] -> [rows, C].
            let block_out = scores.matmul(&v);
            out.data[base * channels..(base + rows) * channels].copy_from_slice(&block_out.data);
            base += QUERY_BLOCK;
        }

        let mut projected = conv2d_pointwise(&out, &self.out_weight, &self.out_bias);
        for (d, s) in projected.data.iter_mut().zip(&x.data) {
            *d += *s;
        }
        projected
    }
}

/// One decoder level: some residual blocks, then an optional 2x upsample
/// followed by a 3x3 convolution.
#[derive(Clone, Debug)]
pub struct VaeUpBlock {
    pub resnets: Vec<VaeResnetBlock>,
    /// Empty on the finest level, which does not upsample.
    pub upsample_weight: Mat, // [C, C * 9]
    pub upsample_bias: Vec<f32>,
}

impl Default for VaeUpBlock {
    fn default() -> Self {
        VaeUpBlock { resnets: Vec::new(), upsample_weight: Mat::zeros(0, 0), upsample_bias: Vec::new() }
    }
}

impl VaeUpBlock {
    pub fn upsamples(&self) -> bool {
        self.upsample_weight.rows > 0
    }
}

// =============================================================================
// Decoder
// =============================================================================

#[derive(Clone, Debug)]
pub struct VaeDecoder {
    pub cfg: VaeConfig,

    pub conv_in_weight: Mat, // [C_mid, latent * 9]
    pub conv_in_bias: Vec<f32>,

    pub mid_resnet1: VaeResnetBlock,
    pub mid_attn: VaeAttentionBlock,
    pub mid_resnet2: VaeResnetBlock,

    /// Coarsest first -- the reverse of diffusers' storage order.
    pub up_blocks: Vec<VaeUpBlock>,

    pub conv_out_norm_weight: Vec<f32>,
    pub conv_out_norm_bias: Vec<f32>,
    pub conv_out_weight: Mat, // [3, C_fine * 9]
    pub conv_out_bias: Vec<f32>,
}

impl Default for VaeDecoder {
    fn default() -> Self {
        VaeDecoder {
            cfg: VaeConfig::default(),
            conv_in_weight: Mat::zeros(0, 0),
            conv_in_bias: Vec::new(),
            mid_resnet1: VaeResnetBlock::default(),
            mid_attn: VaeAttentionBlock::default(),
            mid_resnet2: VaeResnetBlock::default(),
            up_blocks: Vec::new(),
            conv_out_norm_weight: Vec::new(),
            conv_out_norm_bias: Vec::new(),
            conv_out_weight: Mat::zeros(0, 0),
            conv_out_bias: Vec::new(),
        }
    }
}

// -----------------------------------------------------------------------------
// Loading
// -----------------------------------------------------------------------------

type TensorMap = HashMap<String, SafeTensor>;

/// Read one tensor's bytes and widen them to f32, the way every weight in the
/// decoder is used.
///
/// This is `read_safetensor_from_file` with the BF16 conversion *not* skipped,
/// and with F16 decoded by `gguf_loader::f16_to_f32`. The safetensors reader in
/// `transformer3` always leaves BF16 as raw bits, and its own F16 decode is
/// off by a factor of two on subnormals, so it is not used for the conversion.
pub(crate) fn read_tensor_f32(
    file: &mut std::fs::File,
    data_start: usize,
    entry: &SafeTensorEntry,
) -> Result<SafeTensor, String> {
    use std::io::{Read, Seek, SeekFrom};
    let byte_len = entry.byte_end - entry.byte_start;
    file.seek(SeekFrom::Start((data_start + entry.byte_start) as u64))
        .map_err(|e| format!("safetensors: seek error for {}: {}", entry.name, e))?;
    let mut raw = vec![0u8; byte_len];
    file.read_exact(&mut raw).map_err(|_| format!("safetensors: read error for {}", entry.name))?;

    let data: Vec<f32> = match entry.dtype.as_str() {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "BF16" => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        "F16" => raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        other => return Err(format!("safetensors: unsupported dtype {} for {}", other, entry.name)),
    };
    Ok(SafeTensor { name: entry.name.clone(), shape: entry.shape.clone(), data, bf16_data: None })
}

fn take_mat(map: &mut TensorMap, name: &str, rows: usize, cols: usize) -> Result<Mat, String> {
    let len = match map.get(name) {
        None => return Err(format!("vae: missing tensor {}", name)),
        Some(t) => t.data.len(),
    };
    if len != rows * cols {
        return Err(format!("vae: {} has {} elements, expected {}", name, len, rows * cols));
    }
    let t = map.remove(name).unwrap();
    Ok(Mat::new(t.data, rows, cols))
}

fn take_vec(map: &mut TensorMap, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let have = match map.get(name) {
        None => return Err(format!("vae: missing tensor {}", name)),
        Some(t) => t.data.len(),
    };
    if have != len {
        return Err(format!("vae: {} has {} elements, expected {}", name, have, len));
    }
    Ok(map.remove(name).unwrap().data)
}

/// Which naming convention a checkpoint uses.
///
/// Two are in wide circulation for the same weights. `black-forest-labs` ships
/// `ae.safetensors` under the original LDM names -- `decoder.mid.block_1`,
/// `decoder.up.0.block.0`, `nin_shortcut` -- and diffusers ships
/// `vae/diffusion_pytorch_model.safetensors` under its own --
/// `decoder.mid_block.resnets.0`, `decoder.up_blocks.0.resnets.0`,
/// `conv_shortcut`. Repackaged files put either under either path, so the
/// convention is detected from the tensors present rather than from the
/// filename.
///
/// The difference that matters is not the spelling. **The two number the
/// upsampling levels in opposite directions**: diffusers' `up_blocks.0` is the
/// coarsest and the original's `up.0` is the finest. Loading one as the other
/// lines up for the two 512-channel levels and then fails the channel count on
/// the third -- which is the good case, because reversing the resnets without
/// reversing the upsamplers produces a blurred image and no error at all.
struct VaeNames {
    original: bool,
}

impl VaeNames {
    fn mid_resnet(&self, which: usize) -> String {
        if self.original {
            format!("decoder.mid.block_{}.", which + 1)
        } else {
            format!("decoder.mid_block.resnets.{}.", which)
        }
    }
    fn mid_attn(&self) -> &'static str {
        if self.original { "decoder.mid.attn_1." } else { "decoder.mid_block.attentions.0." }
    }
    fn attn_norm(&self) -> &'static str {
        if self.original { "norm" } else { "group_norm" }
    }
    fn attn_q(&self) -> &'static str {
        if self.original { "q" } else { "to_q" }
    }
    fn attn_k(&self) -> &'static str {
        if self.original { "k" } else { "to_k" }
    }
    fn attn_v(&self) -> &'static str {
        if self.original { "v" } else { "to_v" }
    }
    fn attn_out(&self) -> &'static str {
        if self.original { "proj_out" } else { "to_out.0" }
    }

    /// `exec` counts from the coarsest level, which is the order the decoder
    /// runs them in. The original convention stores them the other way round.
    fn level(&self, exec: usize, n_levels: usize) -> String {
        if self.original {
            format!("decoder.up.{}.", n_levels - 1 - exec)
        } else {
            format!("decoder.up_blocks.{}.", exec)
        }
    }
    fn resnet(&self, level: &str, r: usize) -> String {
        format!("{}{}{}.", level, if self.original { "block." } else { "resnets." }, r)
    }
    fn shortcut(&self) -> &'static str {
        if self.original { "nin_shortcut" } else { "conv_shortcut" }
    }
    fn upsample(&self, level: &str) -> String {
        format!("{}{}", level, if self.original { "upsample.conv" } else { "upsamplers.0.conv" })
    }
    fn norm_out(&self) -> &'static str {
        if self.original { "decoder.norm_out" } else { "decoder.conv_norm_out" }
    }
}

/// One of the attention projections, under whichever name this file uses.
fn take_attn_mat(map: &mut TensorMap, prefix: &str, name: &str, c: usize) -> Result<Mat, String> {
    take_mat(map, &format!("{}{}.weight", prefix, name), c, c)
}

fn take_attn_vec(map: &mut TensorMap, prefix: &str, name: &str, c: usize) -> Result<Vec<f32>, String> {
    take_vec(map, &format!("{}{}.bias", prefix, name), c)
}

fn load_resnet(
    map: &mut TensorMap,
    names: &VaeNames,
    prefix: &str,
    c_in: usize,
    c_out: usize,
) -> Result<VaeResnetBlock, String> {
    let mut b = VaeResnetBlock { in_channels: c_in, out_channels: c_out, ..Default::default() };

    b.norm1_weight = take_vec(map, &format!("{}norm1.weight", prefix), c_in)?;
    b.norm1_bias = take_vec(map, &format!("{}norm1.bias", prefix), c_in)?;
    b.conv1_weight = take_mat(map, &format!("{}conv1.weight", prefix), c_out, c_in * 9)?;
    b.conv1_bias = take_vec(map, &format!("{}conv1.bias", prefix), c_out)?;
    b.norm2_weight = take_vec(map, &format!("{}norm2.weight", prefix), c_out)?;
    b.norm2_bias = take_vec(map, &format!("{}norm2.bias", prefix), c_out)?;
    b.conv2_weight = take_mat(map, &format!("{}conv2.weight", prefix), c_out, c_out * 9)?;
    b.conv2_bias = take_vec(map, &format!("{}conv2.bias", prefix), c_out)?;

    if c_in != c_out {
        b.shortcut_weight = take_mat(map, &format!("{}{}.weight", prefix, names.shortcut()), c_out, c_in)?;
        b.shortcut_bias = take_vec(map, &format!("{}{}.bias", prefix, names.shortcut()), c_out)?;
    }
    Ok(b)
}

impl VaeDecoder {
    /// Load from a diffusers `ae.safetensors` / `diffusion_pytorch_model.safetensors`.
    ///
    /// Tensors are streamed one at a time rather than parsed in bulk: the file
    /// also holds the encoder, which is never used and is half its size.
    pub fn load(path: &str, cfg: VaeConfig) -> Result<VaeDecoder, String> {
        if cfg.block_out_channels.is_empty() {
            return Err("vae: block_out_channels is empty".to_string());
        }

        let mut file = std::fs::File::open(path).map_err(|_| format!("safetensors: cannot open {}", path))?;
        let (data_offset, entries) = parse_safetensors_header(&mut file)?;

        // Stream only what the decoder needs. The file also holds the encoder,
        // which is never used and is a third of its size.
        let mut map = TensorMap::new();
        for e in &entries {
            if !e.name.starts_with("decoder.") {
                continue;
            }
            let t = read_tensor_f32(&mut file, data_offset, e)?;
            map.insert(e.name.clone(), t);
        }
        if map.is_empty() {
            return Err(format!("vae: no decoder.* tensors in {}", path));
        }

        let mut d = VaeDecoder { cfg: cfg.clone(), ..Default::default() };

        let names = VaeNames { original: map.contains_key("decoder.mid.block_1.conv1.weight") };

        let n_levels = cfg.block_out_channels.len();
        let c_coarse = *cfg.block_out_channels.last().unwrap();
        let c_fine = cfg.block_out_channels[0];

        d.conv_in_weight = take_mat(&mut map, "decoder.conv_in.weight", c_coarse, cfg.latent_channels * 9)?;
        d.conv_in_bias = take_vec(&mut map, "decoder.conv_in.bias", c_coarse)?;

        d.mid_resnet1 = load_resnet(&mut map, &names, &names.mid_resnet(0), c_coarse, c_coarse)?;
        d.mid_resnet2 = load_resnet(&mut map, &names, &names.mid_resnet(1), c_coarse, c_coarse)?;

        {
            let p = names.mid_attn();
            let mut a = VaeAttentionBlock { channels: c_coarse, ..Default::default() };
            a.norm_weight = take_vec(&mut map, &format!("{}{}.weight", p, names.attn_norm()), c_coarse)?;
            a.norm_bias = take_vec(&mut map, &format!("{}{}.bias", p, names.attn_norm()), c_coarse)?;
            a.q_weight = take_attn_mat(&mut map, p, names.attn_q(), c_coarse)?;
            a.q_bias = take_attn_vec(&mut map, p, names.attn_q(), c_coarse)?;
            a.k_weight = take_attn_mat(&mut map, p, names.attn_k(), c_coarse)?;
            a.k_bias = take_attn_vec(&mut map, p, names.attn_k(), c_coarse)?;
            a.v_weight = take_attn_mat(&mut map, p, names.attn_v(), c_coarse)?;
            a.v_bias = take_attn_vec(&mut map, p, names.attn_v(), c_coarse)?;
            a.out_weight = take_attn_mat(&mut map, p, names.attn_out(), c_coarse)?;
            a.out_bias = take_attn_vec(&mut map, p, names.attn_out(), c_coarse)?;
            d.mid_attn = a;
        }

        // Walk the levels in execution order -- coarsest first -- and let
        // `names` work out which index that is in this file's convention.
        let mut c_prev = c_coarse;
        for i in 0..n_levels {
            // Execution index `i` is level `n_levels - 1 - i` of the
            // finest-first `block_out_channels`.
            let c_out = cfg.block_out_channels[n_levels - 1 - i];
            let p = names.level(i, n_levels);

            let mut block = VaeUpBlock::default();
            for r in 0..=cfg.layers_per_block {
                let c_in = if r == 0 { c_prev } else { c_out };
                block.resnets.push(load_resnet(&mut map, &names, &names.resnet(&p, r), c_in, c_out)?);
            }
            // Every level upsamples except the finest, which is the last one.
            if i + 1 < n_levels {
                let u = names.upsample(&p);
                block.upsample_weight = take_mat(&mut map, &format!("{}.weight", u), c_out, c_out * 9)?;
                block.upsample_bias = take_vec(&mut map, &format!("{}.bias", u), c_out)?;
            }
            d.up_blocks.push(block);
            c_prev = c_out;
        }

        d.conv_out_norm_weight = take_vec(&mut map, &format!("{}.weight", names.norm_out()), c_fine)?;
        d.conv_out_norm_bias = take_vec(&mut map, &format!("{}.bias", names.norm_out()), c_fine)?;
        d.conv_out_weight = take_mat(&mut map, "decoder.conv_out.weight", 3, c_fine * 9)?;
        d.conv_out_bias = take_vec(&mut map, "decoder.conv_out.bias", 3)?;

        Ok(d)
    }

    // -------------------------------------------------------------------------
    // Decoding
    // -------------------------------------------------------------------------

    /// Decode a latent to pixels in roughly [-1, 1].
    ///
    /// `z` is [lat_h * lat_w, latent_channels] as the diffusion model emits it,
    /// still scaled -- this applies `scaling_factor` and `shift_factor` itself,
    /// so callers must not.
    ///
    /// Returns [(lat_h * f) * (lat_w * f), 3] where `f` is
    /// `cfg.downsample_factor()`.
    pub fn decode(&self, z: &Mat, lat_h: usize, lat_w: usize) -> Result<Mat, String> {
        let cfg = &self.cfg;
        if z.cols != cfg.latent_channels {
            return Err(format!(
                "vae decode: latent has {} channels, expected {}",
                z.cols, cfg.latent_channels
            ));
        }
        if z.rows != lat_h * lat_w {
            return Err(format!("vae decode: rows {} != lat_h*lat_w", z.rows));
        }
        if cfg.block_out_channels.is_empty() {
            return Err("vae decode: block_out_channels is empty".to_string());
        }

        // Undo the training-time rescaling. Both constants, in this order.
        let mut x = z.clone();
        let inv_scale = 1.0f32 / cfg.scaling_factor;
        for v in x.data.iter_mut() {
            *v = *v * inv_scale + cfg.shift_factor;
        }

        let mut h = lat_h;
        let mut w = lat_w;

        x = conv2d_dense(&x, h, w, &self.conv_in_weight, *cfg.block_out_channels.last().unwrap(), 3, 3, &self.conv_in_bias, 1, 1, 1);

        x = self.mid_resnet1.forward(&x, h, w, cfg);
        x = self.mid_attn.forward(&x, cfg);
        x = self.mid_resnet2.forward(&x, h, w, cfg);

        for block in &self.up_blocks {
            for res in &block.resnets {
                x = res.forward(&x, h, w, cfg);
            }
            if block.upsamples() {
                x = upsample_nearest2d(&x, h, w, 2);
                h *= 2;
                w *= 2;
                let c = x.cols;
                x = conv2d_dense(&x, h, w, &block.upsample_weight, c, 3, 3, &block.upsample_bias, 1, 1, 1);
            }
        }

        group_norm_inplace(&mut x, cfg.norm_groups, &self.conv_out_norm_weight, &self.conv_out_norm_bias, cfg.norm_eps);
        silu_inplace(&mut x);
        x = conv2d_dense(&x, h, w, &self.conv_out_weight, 3, 3, 3, &self.conv_out_bias, 1, 1, 1);
        Ok(x)
    }

    /// Decode in overlapping tiles, blending the seams.
    ///
    /// `tile` and `overlap` are in latent pixels (64 and 16 are the usual
    /// values). The whole-image path needs a few gigabytes at 1024x1024; this
    /// caps it at roughly `(tile + overlap)^2 * 64 * channels` bytes per tile.
    ///
    /// The blend is a linear ramp across the overlap. A hard cut leaves a seam
    /// that is invisible in flat regions and obvious across any edge, because
    /// the two tiles' GroupNorm statistics differ -- which is also why the
    /// overlap has to be a real fraction of the tile and not two pixels.
    pub fn decode_tiled(&self, z: &Mat, lat_h: usize, lat_w: usize, tile: usize, overlap: usize) -> Result<Mat, String> {
        if tile == 0 {
            return Err("vae decode_tiled: tile must be > 0".to_string());
        }
        if overlap >= tile {
            return Err("vae decode_tiled: overlap must be smaller than tile".to_string());
        }
        if lat_h <= tile && lat_w <= tile {
            return self.decode(z, lat_h, lat_w);
        }

        let f = self.cfg.downsample_factor();
        let out_h = lat_h * f;
        let out_w = lat_w * f;
        let mut acc = Mat::zeros(out_h * out_w, 3);
        let mut weight_sum = vec![0.0f32; out_h * out_w];

        let step = tile - overlap;
        let mut y0 = 0;
        while y0 < lat_h {
            let th = tile.min(lat_h - y0);
            let mut x0 = 0;
            while x0 < lat_w {
                let tw = tile.min(lat_w - x0);

                let zc = z.cols;
                let mut sub = Mat::zeros(th * tw, zc);
                for y in 0..th {
                    let s0 = ((y0 + y) * lat_w + x0) * zc;
                    sub.data[y * tw * zc..(y + 1) * tw * zc].copy_from_slice(&z.data[s0..s0 + tw * zc]);
                }

                let pixels = self.decode(&sub, th, tw)?;

                // A linear ramp over the overlap, applied only on the edges
                // that actually abut another tile. Ramping an image boundary
                // would fade the picture out at its own edges.
                let ph = th * f;
                let pw = tw * f;
                let ramp = overlap * f;
                let ramp_top = y0 > 0;
                let ramp_left = x0 > 0;
                let ramp_bottom = y0 + th < lat_h;
                let ramp_right = x0 + tw < lat_w;

                for y in 0..ph {
                    let mut wy = 1.0f32;
                    if ramp_top && y < ramp {
                        wy = min_f32(wy, (y + 1) as f32 / (ramp + 1) as f32);
                    }
                    if ramp_bottom && y + ramp >= ph {
                        wy = min_f32(wy, (ph - y) as f32 / (ramp + 1) as f32);
                    }
                    for x in 0..pw {
                        let mut wx = 1.0f32;
                        if ramp_left && x < ramp {
                            wx = min_f32(wx, (x + 1) as f32 / (ramp + 1) as f32);
                        }
                        if ramp_right && x + ramp >= pw {
                            wx = min_f32(wx, (pw - x) as f32 / (ramp + 1) as f32);
                        }
                        let blend = wy * wx;
                        let dst = (y0 * f + y) * out_w + (x0 * f + x);
                        let src = &pixels.data[(y * pw + x) * 3..(y * pw + x) * 3 + 3];
                        let out = &mut acc.data[dst * 3..dst * 3 + 3];
                        for c in 0..3 {
                            out[c] += blend * src[c];
                        }
                        weight_sum[dst] += blend;
                    }
                }
                if x0 + tw >= lat_w {
                    break;
                }
                x0 += step;
            }
            if y0 + th >= lat_h {
                break;
            }
            y0 += step;
        }

        for p in 0..acc.rows {
            let wsum = weight_sum[p];
            if wsum <= 0.0 {
                continue;
            }
            let row = &mut acc.data[p * 3..p * 3 + 3];
            for c in 0..3 {
                row[c] /= wsum;
            }
        }
        Ok(acc)
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.conv_in_weight.numel() + self.conv_in_bias.len();

        let count_resnet = |b: &VaeResnetBlock| {
            b.norm1_weight.len()
                + b.norm1_bias.len()
                + b.conv1_weight.numel()
                + b.conv1_bias.len()
                + b.norm2_weight.len()
                + b.norm2_bias.len()
                + b.conv2_weight.numel()
                + b.conv2_bias.len()
                + b.shortcut_weight.numel()
                + b.shortcut_bias.len()
        };

        n += count_resnet(&self.mid_resnet1) + count_resnet(&self.mid_resnet2);
        let a = &self.mid_attn;
        n += a.norm_weight.len()
            + a.norm_bias.len()
            + a.q_weight.numel()
            + a.q_bias.len()
            + a.k_weight.numel()
            + a.k_bias.len()
            + a.v_weight.numel()
            + a.v_bias.len()
            + a.out_weight.numel()
            + a.out_bias.len();

        for b in &self.up_blocks {
            for r in &b.resnets {
                n += count_resnet(r);
            }
            n += b.upsample_weight.numel() + b.upsample_bias.len();
        }

        n += self.conv_out_norm_weight.len()
            + self.conv_out_norm_bias.len()
            + self.conv_out_weight.numel()
            + self.conv_out_bias.len();
        n
    }
}

/// `std::min(a, b)`: `b` only when it compares strictly below `a`.
#[inline]
fn min_f32(a: f32, b: f32) -> f32 {
    if b < a { b } else { a }
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

    /// Deterministic small signed values, spread by a hash of the index so that
    /// a transposed or mis-strided weight cannot pass on a smooth ramp.
    fn spread(rows: usize, cols: usize, scale: f32, salt: usize) -> Mat {
        Mat::from_fn(rows, cols, |r, c| {
            let i = ((r * 131 + c * 37 + salt * 7919) % 251) as f32;
            ((i * 0.41).sin() * 0.7 + (i * 0.13).cos() * 0.3) * scale
        })
    }

    fn spread_vec(n: usize, scale: f32, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = ((i * 53 + salt * 6151) % 241) as f32;
                ((k * 0.29).sin() * 0.6 + (k * 0.17).cos() * 0.4) * scale
            })
            .collect()
    }

    fn make_resnet(c_in: usize, c_out: usize, salt: usize) -> VaeResnetBlock {
        let mut b = VaeResnetBlock {
            in_channels: c_in,
            out_channels: c_out,
            norm1_weight: vec![1.0; c_in],
            norm1_bias: vec![0.0; c_in],
            conv1_weight: spread(c_out, c_in * 9, 0.05, salt),
            conv1_bias: spread_vec(c_out, 0.02, salt + 1),
            norm2_weight: vec![1.0; c_out],
            norm2_bias: vec![0.0; c_out],
            conv2_weight: spread(c_out, c_out * 9, 0.05, salt + 2),
            conv2_bias: spread_vec(c_out, 0.02, salt + 3),
            ..Default::default()
        };
        if c_in != c_out {
            b.shortcut_weight = spread(c_out, c_in, 0.2, salt + 4);
            b.shortcut_bias = spread_vec(c_out, 0.01, salt + 5);
        }
        b
    }

    /// Build a decoder of the given shape with deterministic synthetic weights.
    ///
    /// Every structural property worth testing -- the level ordering, the
    /// upsample count, the latent rescaling, the tile blending -- is
    /// independent of the weights' values.
    fn make_decoder(cfg: &VaeConfig) -> VaeDecoder {
        let mut d = VaeDecoder { cfg: cfg.clone(), ..Default::default() };

        let n_levels = cfg.block_out_channels.len();
        let c_coarse = *cfg.block_out_channels.last().unwrap();
        let c_fine = cfg.block_out_channels[0];

        d.conv_in_weight = spread(c_coarse, cfg.latent_channels * 9, 0.1, 1);
        d.conv_in_bias = spread_vec(c_coarse, 0.02, 2);

        d.mid_resnet1 = make_resnet(c_coarse, c_coarse, 10);
        d.mid_resnet2 = make_resnet(c_coarse, c_coarse, 20);

        d.mid_attn.channels = c_coarse;
        d.mid_attn.norm_weight = vec![1.0; c_coarse];
        d.mid_attn.norm_bias = vec![0.0; c_coarse];
        d.mid_attn.q_weight = spread(c_coarse, c_coarse, 0.15, 30);
        d.mid_attn.q_bias = spread_vec(c_coarse, 0.01, 31);
        d.mid_attn.k_weight = spread(c_coarse, c_coarse, 0.15, 32);
        d.mid_attn.k_bias = spread_vec(c_coarse, 0.01, 33);
        d.mid_attn.v_weight = spread(c_coarse, c_coarse, 0.15, 34);
        d.mid_attn.v_bias = spread_vec(c_coarse, 0.01, 35);
        d.mid_attn.out_weight = spread(c_coarse, c_coarse, 0.15, 36);
        d.mid_attn.out_bias = spread_vec(c_coarse, 0.01, 37);

        let mut c_prev = c_coarse;
        for i in 0..n_levels {
            let c_out = cfg.block_out_channels[n_levels - 1 - i];
            let mut block = VaeUpBlock::default();
            for r in 0..=cfg.layers_per_block {
                let c_in = if r == 0 { c_prev } else { c_out };
                block.resnets.push(make_resnet(c_in, c_out, 100 + i * 10 + r));
            }
            if i + 1 < n_levels {
                block.upsample_weight = spread(c_out, c_out * 9, 0.05, 200 + i);
                block.upsample_bias = spread_vec(c_out, 0.01, 300 + i);
            }
            d.up_blocks.push(block);
            c_prev = c_out;
        }

        d.conv_out_norm_weight = vec![1.0; c_fine];
        d.conv_out_norm_bias = vec![0.0; c_fine];
        d.conv_out_weight = spread(3, c_fine * 9, 0.2, 400);
        d.conv_out_bias = spread_vec(3, 0.05, 401);
        d
    }

    fn tiny_config() -> VaeConfig {
        VaeConfig {
            latent_channels: 4,
            block_out_channels: vec![8, 16],
            layers_per_block: 1,
            norm_groups: 4,
            norm_eps: 1e-6,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
        }
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    #[test]
    fn the_flux_autoencoder_config_is_the_16_channel_one() {
        let c = VaeConfig::flux();
        assert_eq!(c.latent_channels, 16);
        assert_eq!(c.block_out_channels, vec![128, 256, 512, 512]);
        assert_eq!(c.layers_per_block, 2);
        assert_eq!(c.norm_groups, 32);
        assert!(approx(c.norm_eps, 1e-6, 1e-12));
        // Both rescaling constants. SD 1.x has only the first, which is exactly
        // why the second is the one that goes missing.
        assert!(approx(c.scaling_factor, 0.3611, 1e-6));
        assert!(approx(c.shift_factor, 0.1159, 1e-6));
        assert_eq!(c.downsample_factor(), 8);
    }

    #[test]
    fn downsample_factor_is_one_doubling_per_level_after_the_first() {
        let mut c = VaeConfig::default();
        c.block_out_channels = vec![32];
        assert_eq!(c.downsample_factor(), 1);
        c.block_out_channels = vec![32, 64];
        assert_eq!(c.downsample_factor(), 2);
        c.block_out_channels = vec![32, 64, 128];
        assert_eq!(c.downsample_factor(), 4);
        c.block_out_channels = vec![128, 256, 512, 512];
        assert_eq!(c.downsample_factor(), 8);
    }

    // -------------------------------------------------------------------------
    // Shape
    // -------------------------------------------------------------------------

    #[test]
    fn decode_upsamples_by_exactly_the_configured_factor() {
        let cfg = tiny_config();
        let d = make_decoder(&cfg);
        let z = spread(6 * 5, 4, 1.0, 77);
        let out = d.decode(&z, 6, 5).unwrap();
        assert_eq!(out.cols, 3);
        assert_eq!(out.rows, (6 * 2) * (5 * 2));
    }

    #[test]
    fn a_three_level_decoder_upsamples_by_four() {
        let mut cfg = tiny_config();
        cfg.block_out_channels = vec![8, 8, 16];
        let d = make_decoder(&cfg);
        let z = spread(4 * 4, 4, 1.0, 78);
        let out = d.decode(&z, 4, 4).unwrap();
        assert_eq!(out.rows, (4 * 4) * (4 * 4));
        assert_eq!(cfg.downsample_factor(), 4);
    }

    #[test]
    fn decode_rejects_a_latent_of_the_wrong_shape() {
        let d = make_decoder(&tiny_config());
        assert!(d.decode(&spread(16, 8, 1.0, 1), 4, 4).is_err()); // wrong channels
        assert!(d.decode(&spread(15, 4, 1.0, 1), 4, 4).is_err()); // wrong row count
    }

    #[test]
    fn decode_is_deterministic() {
        let d = make_decoder(&tiny_config());
        let z = spread(4 * 4, 4, 1.0, 5);
        let a = d.decode(&z, 4, 4).unwrap();
        let b = d.decode(&z, 4, 4).unwrap();
        assert_eq!(a.data, b.data);
    }

    #[test]
    fn decode_produces_something_finite_from_a_plausible_latent() {
        let d = make_decoder(&tiny_config());
        let z = spread(8 * 8, 4, 1.0, 9);
        let out = d.decode(&z, 8, 8).unwrap();
        assert!(out.data.iter().all(|v| v.is_finite()));
    }

    // -------------------------------------------------------------------------
    // Latent rescaling
    // -------------------------------------------------------------------------

    #[test]
    fn decode_applies_z_over_scaling_factor_plus_shift_factor() {
        // Two decoders with identical weights, one rescaling and one not.
        // Feeding the second the pre-rescaled latent must give the identical
        // image -- which pins both constants and the order of the two
        // operations.
        let scaled = tiny_config();
        let mut plain = tiny_config();
        plain.scaling_factor = 1.0;
        plain.shift_factor = 0.0;

        let a = make_decoder(&scaled);
        let mut b = make_decoder(&scaled);
        b.cfg = plain;

        let z = spread(5 * 4, 4, 1.0, 11);
        let mut pre = z.clone();
        for v in pre.data.iter_mut() {
            *v = *v / scaled.scaling_factor + scaled.shift_factor;
        }

        let from_scaled = a.decode(&z, 5, 4).unwrap();
        let from_plain = b.decode(&pre, 5, 4).unwrap();
        for i in 0..from_scaled.data.len() {
            assert!(approx(from_scaled.data[i], from_plain.data[i], 1e-4));
        }
    }

    #[test]
    fn dropping_the_shift_factor_changes_the_image() {
        // It does not break it -- it shifts the colour, which is why the
        // omission survives a visual check.
        let with = tiny_config();
        let mut without = tiny_config();
        without.shift_factor = 0.0;

        let a = make_decoder(&with);
        let mut b = make_decoder(&with);
        b.cfg = without;

        let z = spread(4 * 4, 4, 1.0, 13);
        let pa = a.decode(&z, 4, 4).unwrap();
        let pb = b.decode(&z, 4, 4).unwrap();

        let mut max_delta = 0.0f32;
        for i in 0..pa.data.len() {
            max_delta = max_delta.max((pa.data[i] - pb.data[i]).abs());
        }
        assert!(max_delta > 1e-4);
    }

    // -------------------------------------------------------------------------
    // Blocks
    // -------------------------------------------------------------------------

    #[test]
    fn a_resnet_block_carries_a_shortcut_only_when_the_width_changes() {
        let same = make_resnet(8, 8, 1);
        let wider = make_resnet(8, 16, 2);
        assert_eq!(same.shortcut_weight.rows, 0);
        assert_eq!(wider.shortcut_weight.rows, 16);
        assert_eq!(wider.shortcut_weight.cols, 8);
    }

    #[test]
    fn a_zeroed_resnet_block_is_its_own_residual() {
        // With both convolutions zero, the block reduces to the identity plus
        // the convolution biases -- which pins the residual add to the *input*,
        // not to the normalized input.
        let cfg = tiny_config();
        let mut b = make_resnet(8, 8, 3);
        b.conv1_weight = Mat::zeros(8, 8 * 9);
        b.conv2_weight = Mat::zeros(8, 8 * 9);
        b.conv1_bias = vec![0.0; 8];
        b.conv2_bias = vec![0.0; 8];

        let x = spread(4 * 4, 8, 1.0, 4);
        let out = b.forward(&x, 4, 4, &cfg);
        for i in 0..out.data.len() {
            assert!(approx(out.data[i], x.data[i], 1e-5));
        }
    }

    fn zero_attention(c: usize) -> VaeAttentionBlock {
        VaeAttentionBlock {
            channels: c,
            norm_weight: vec![1.0; c],
            norm_bias: vec![0.0; c],
            q_weight: Mat::zeros(c, c),
            q_bias: vec![0.0; c],
            k_weight: Mat::zeros(c, c),
            k_bias: vec![0.0; c],
            v_weight: Mat::zeros(c, c),
            v_bias: vec![0.0; c],
            out_weight: Mat::zeros(c, c),
            out_bias: vec![0.0; c],
        }
    }

    #[test]
    fn a_zeroed_attention_block_is_its_own_residual() {
        let cfg = tiny_config();
        let a = zero_attention(8);
        let x = spread(3 * 3, 8, 1.0, 6);
        let out = a.forward(&x, &cfg);
        for i in 0..out.data.len() {
            assert!(approx(out.data[i], x.data[i], 1e-6));
        }
    }

    #[test]
    fn attention_averages_the_values_when_every_score_is_equal() {
        // Zero q and k make every score zero, so the softmax is uniform and
        // each output row is the mean of v. That pins the softmax axis:
        // averaging over channels instead of positions would give a different,
        // per-row answer.
        let cfg = tiny_config();
        let mut a = zero_attention(8);
        a.v_bias = spread_vec(8, 1.0, 12); // v is a constant per channel
        for i in 0..8 {
            *a.out_weight.at_mut(i, i) = 1.0; // identity projection
        }

        let x = spread(4 * 4, 8, 1.0, 14);
        let out = a.forward(&x, &cfg);
        for r in 0..out.rows {
            for c in 0..8 {
                assert!(approx(out.at(r, c), x.at(r, c) + a.v_bias[c], 1e-5));
            }
        }
    }

    #[test]
    fn attention_mixes_information_across_distant_positions() {
        // Perturbing one corner of the input must change the opposite corner of
        // the output. A convolution cannot do that at this distance; attention
        // must.
        let cfg = tiny_config();
        let d = make_decoder(&cfg);

        let mut z = spread(6 * 6, 4, 1.0, 15);
        let base = d.mid_attn.forward(
            &conv2d_dense(&z, 6, 6, &d.conv_in_weight, 16, 3, 3, &d.conv_in_bias, 1, 1, 1),
            &cfg,
        );

        *z.at_mut(0, 0) += 5.0;
        let poked = d.mid_attn.forward(
            &conv2d_dense(&z, 6, 6, &d.conv_in_weight, 16, 3, 3, &d.conv_in_bias, 1, 1, 1),
            &cfg,
        );

        let far = 6 * 6 - 1;
        let mut delta = 0.0f32;
        for c in 0..16 {
            delta = delta.max((base.at(far, c) - poked.at(far, c)).abs());
        }
        assert!(delta > 1e-5);
    }

    // -------------------------------------------------------------------------
    // Tiling
    // -------------------------------------------------------------------------

    #[test]
    fn decode_tiled_falls_through_to_decode_when_the_latent_fits() {
        let d = make_decoder(&tiny_config());
        let z = spread(6 * 6, 4, 1.0, 21);
        let whole = d.decode(&z, 6, 6).unwrap();
        let tiled = d.decode_tiled(&z, 6, 6, 8, 2).unwrap();
        assert_eq!(whole.data, tiled.data);
    }

    #[test]
    fn decode_tiled_covers_every_output_pixel_exactly_once() {
        // A decoder whose convolutions are all zero emits its output bias
        // everywhere, whatever the tiling. Any pixel the blend weights fail to
        // normalize shows up immediately as a value that is not that bias.
        let cfg = tiny_config();
        let mut d = make_decoder(&cfg);
        d.conv_in_weight = Mat::zeros(16, 4 * 9);
        d.conv_in_bias = vec![0.0; 16];
        for b in d.up_blocks.iter_mut() {
            for r in b.resnets.iter_mut() {
                r.conv1_weight = Mat::zeros(r.out_channels, r.in_channels * 9);
                r.conv2_weight = Mat::zeros(r.out_channels, r.out_channels * 9);
                r.conv1_bias = vec![0.0; r.out_channels];
                r.conv2_bias = vec![0.0; r.out_channels];
                if r.shortcut_weight.rows > 0 {
                    r.shortcut_weight = Mat::zeros(r.out_channels, r.in_channels);
                    r.shortcut_bias = vec![0.0; r.out_channels];
                }
            }
            if b.upsamples() {
                b.upsample_weight = Mat::zeros(b.upsample_weight.rows, b.upsample_weight.cols);
                b.upsample_bias = vec![0.0; b.upsample_bias.len()];
            }
        }
        for m in [&mut d.mid_resnet1, &mut d.mid_resnet2] {
            m.conv1_weight = Mat::zeros(16, 16 * 9);
            m.conv2_weight = Mat::zeros(16, 16 * 9);
            m.conv1_bias = vec![0.0; 16];
            m.conv2_bias = vec![0.0; 16];
        }
        d.mid_attn = zero_attention(16);
        d.conv_out_weight = Mat::zeros(3, 8 * 9);
        d.conv_out_bias = vec![0.25, -0.5, 0.75];

        let z = spread(24 * 20, 4, 1.0, 22);
        let tiled = d.decode_tiled(&z, 24, 20, 8, 3).unwrap();
        assert_eq!(tiled.rows, (24 * 2) * (20 * 2));
        for r in 0..tiled.rows {
            assert!(approx(tiled.at(r, 0), 0.25, 1e-5));
            assert!(approx(tiled.at(r, 1), -0.5, 1e-5));
            assert!(approx(tiled.at(r, 2), 0.75, 1e-5));
        }
    }

    #[test]
    fn decode_tiled_produces_the_right_shape_on_a_non_multiple_latent() {
        let d = make_decoder(&tiny_config());
        let z = spread(23 * 17, 4, 1.0, 23);
        let tiled = d.decode_tiled(&z, 23, 17, 8, 2).unwrap();
        assert_eq!(tiled.rows, (23 * 2) * (17 * 2));
        assert_eq!(tiled.cols, 3);
        assert!(tiled.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn decode_tiled_rejects_an_overlap_that_is_not_smaller_than_the_tile() {
        let d = make_decoder(&tiny_config());
        let z = spread(20 * 20, 4, 1.0, 24);
        assert!(d.decode_tiled(&z, 20, 20, 8, 8).is_err());
        assert!(d.decode_tiled(&z, 20, 20, 0, 0).is_err());
    }

    // -------------------------------------------------------------------------
    // Bookkeeping
    // -------------------------------------------------------------------------

    #[test]
    fn parameter_count_adds_up() {
        let cfg = tiny_config();
        let d = make_decoder(&cfg);

        // conv_in 16 x (4*9) + 16
        let mut expected = 16 * 36 + 16;
        // two mid resnets, 16 -> 16, no shortcut
        let resnet_16 = 16 + 16 + 16 * 144 + 16 + 16 + 16 + 16 * 144 + 16;
        expected += 2 * resnet_16;
        // attention: norm + four 16x16 projections with biases
        expected += 16 + 16 + 4 * (16 * 16 + 16);
        // up level 0: two 16 -> 16 resnets plus a 16 x (16*9) upsample conv
        expected += 2 * resnet_16 + 16 * 144 + 16;
        // up level 1: 16 -> 8 (with shortcut) then 8 -> 8
        expected += 16 + 16 + 8 * 144 + 8 + 8 + 8 + 8 * 72 + 8 + 8 * 16 + 8;
        expected += 8 + 8 + 8 * 72 + 8 + 8 + 8 + 8 * 72 + 8;
        // conv_norm_out + conv_out
        expected += 8 + 8 + 3 * 72 + 3;

        assert_eq!(d.parameter_count(), expected);
    }
}
