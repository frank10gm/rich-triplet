#![allow(dead_code)]
// =============================================================================
// FLUX.1 -- a rectified-flow diffusion transformer
// =============================================================================
//
// The model that turns a prompt into an image, in the sense that it is where
// almost all the parameters and almost all the time go. It does not emit
// pixels: it emits a velocity field over a 16-channel latent, which an Euler
// solver integrates and `vae` decodes.
//
// 11.9B parameters, hidden width 3072, 24 heads of 128.
//
// ## The two block types
//
// **19 double-stream blocks.** Image and text are separate residual streams
// with separate weights, but a single joint attention over the concatenation of
// both. That is the "MM" in MMDiT: the modalities keep their own parameters and
// share only the attention.
//
// **38 single-stream blocks.** One stream over the concatenation, and -- the
// unusual part -- attention and MLP are computed *in parallel* from the same
// modulated input and concatenated before a single output projection, rather
// than run in sequence. It is the ViT-22B trick, and it makes one fused
// `linear1` do the work of a QKV projection and an MLP up-projection at once.
//
// ## Conditioning does not enter through cross-attention
//
// There is no cross-attention anywhere. The prompt enters as *tokens* in the
// text stream, and the timestep and the pooled CLIP vector enter through
// adaptive layer norm: a per-block linear map from the conditioning vector to
// a shift, a scale and a gate for each sub-layer. Every LayerNorm in the model
// is affine-free precisely because the affine part comes from there.
//
// ## Three-axis RoPE
//
// Position is `(t, h, w)` with per-axis head dimensions [16, 56, 56], summing
// to the 128 of a head. Image tokens carry their patch-grid coordinates; text
// tokens carry all zeros, which makes their rotation the identity. The pairs
// are adjacent -- `(x[2i], x[2i+1])` -- not split-half.
//
// ## schnell against dev
//
// schnell is timestep-distilled to four steps *and* guidance-distilled, so
// there is no classifier-free guidance and one forward pass per step. It also
// has no `guidance_in` embedding at all, so loading dev weights into a schnell
// config fails on a missing tensor -- which is the outcome worth having, since
// the reverse (running dev without guidance) silently produces washed-out
// images.

use std::sync::Arc;

use crate::autograd2::{Mat, MatBf16};
use crate::conv2d::{gelu_tanh_inplace, silu_inplace};
use crate::gguf_loader::{gguf_tensor_to_f32, load_gguf_vector, GgufFile, GgufType};
use crate::nn::InitRng;
use crate::qlinear::QLinear;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_double_blocks: usize,
    pub n_single_blocks: usize,
    pub mlp_ratio: f32,
    /// Width of the T5 sequence entering `txt_in`.
    pub context_dim: usize,
    /// Width of the pooled CLIP vector entering `vector_in`.
    pub pooled_dim: usize,
    /// Per-axis RoPE dimensions over (t, h, w). Must sum to `head_dim()`.
    pub axes_dim: Vec<usize>,
    pub rope_theta: f32,
    /// dev has a distilled-guidance embedding; schnell does not.
    pub guidance_embed: bool,
    /// Latent patch size. 2 everywhere in the FLUX family.
    pub patch_size: usize,
    pub qk_norm_eps: f32,
    pub layer_norm_eps: f32,
}

impl Default for FluxConfig {
    fn default() -> Self {
        FluxConfig {
            in_channels: 16,
            hidden_size: 3072,
            n_heads: 24,
            n_double_blocks: 19,
            n_single_blocks: 38,
            mlp_ratio: 4.0,
            context_dim: 4096,
            pooled_dim: 768,
            axes_dim: vec![16, 56, 56],
            rope_theta: 10000.0,
            guidance_embed: false,
            patch_size: 2,
            qk_norm_eps: 1e-6,
            layer_norm_eps: 1e-6,
        }
    }
}

impl FluxConfig {
    pub fn schnell() -> Self {
        FluxConfig { guidance_embed: false, ..Default::default() }
    }

    pub fn dev() -> Self {
        FluxConfig { guidance_embed: true, ..Default::default() }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.n_heads
    }

    /// `in_channels * patch_size^2` -- the width of one packed latent patch.
    pub fn patch_dim(&self) -> usize {
        self.in_channels * self.patch_size * self.patch_size
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f32 * self.mlp_ratio) as usize
    }
}

// =============================================================================
// Patching
// =============================================================================

/// Pack a latent into transformer tokens.
///
/// `z` is [lat_h * lat_w, channels] and the result is
/// [(lat_h/p) * (lat_w/p), channels * p * p], one row per patch.
///
/// The packing order within a patch is **channel-major**:
/// `c (h ph) (w pw) -> (h w) (c ph pw)`. Ordering it the other way is the kind
/// of mistake that survives review, because the model still trains its way to
/// something and the image comes out with a fine 2x2 scramble that reads as
/// noise rather than as a transposition.
pub fn flux_patchify(z: &Mat, lat_h: usize, lat_w: usize, patch: usize) -> Mat {
    debug_assert!(z.rows == lat_h * lat_w, "flux_patchify: rows != lat_h * lat_w");
    debug_assert!(lat_h % patch == 0 && lat_w % patch == 0, "flux_patchify: latent not divisible");

    let channels = z.cols;
    let gh = lat_h / patch;
    let gw = lat_w / patch;
    let width = channels * patch * patch;
    let mut out = Mat::zeros(gh * gw, width);

    for py in 0..gh {
        for px in 0..gw {
            let d0 = (py * gw + px) * width;
            for c in 0..channels {
                for ky in 0..patch {
                    for kx in 0..patch {
                        // Channel-major within the patch: index
                        // `c * patch^2 + ky * patch + kx`.
                        let src = (py * patch + ky) * lat_w + (px * patch + kx);
                        out.data[d0 + c * patch * patch + ky * patch + kx] = z.at(src, c);
                    }
                }
            }
        }
    }
    out
}

/// The inverse of `flux_patchify`.
pub fn flux_unpatchify(tokens: &Mat, lat_h: usize, lat_w: usize, channels: usize, patch: usize) -> Mat {
    let gh = lat_h / patch;
    let gw = lat_w / patch;
    debug_assert!(tokens.rows == gh * gw, "flux_unpatchify: token count != patch grid");
    debug_assert!(tokens.cols == channels * patch * patch, "flux_unpatchify: token width mismatch");

    let mut out = Mat::zeros(lat_h * lat_w, channels);
    for py in 0..gh {
        for px in 0..gw {
            let s0 = (py * gw + px) * tokens.cols;
            for c in 0..channels {
                for ky in 0..patch {
                    for kx in 0..patch {
                        let dst = (py * patch + ky) * lat_w + (px * patch + kx);
                        *out.at_mut(dst, c) = tokens.data[s0 + c * patch * patch + ky * patch + kx];
                    }
                }
            }
        }
    }
    out
}

/// Position ids for the packed tokens: `[n_tokens, 3]` holding (0, y, x).
pub fn flux_image_ids(lat_h: usize, lat_w: usize, patch: usize) -> Mat {
    let gh = lat_h / patch;
    let gw = lat_w / patch;
    let mut ids = Mat::zeros(gh * gw, 3);
    for y in 0..gh {
        for x in 0..gw {
            // Axis 0 stays zero: it exists for video, where it is the frame.
            *ids.at_mut(y * gw + x, 1) = y as f32;
            *ids.at_mut(y * gw + x, 2) = x as f32;
        }
    }
    ids
}

// =============================================================================
// Embeddings
// =============================================================================

/// Sinusoidal timestep embedding, **cosines first**.
///
/// `t` is scaled by `time_factor` (1000 in FLUX) before the frequencies are
/// applied; `max_period` is 10000. The concatenation order is `[cos, sin]`,
/// which is the reverse of the more common convention -- swapping it rotates
/// every conditioning vector by a quarter turn and the model produces coherent
/// images of the wrong timestep.
pub fn flux_timestep_embedding(t: f32, dim: usize, max_period: f32, time_factor: f32) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0f32; dim];
    let scaled = t * time_factor;
    for i in 0..half {
        let freq = (-max_period.ln() * i as f32 / half as f32).exp();
        let arg = scaled * freq;
        // Cosines first. The usual convention is the other way round.
        out[i] = arg.cos();
        out[half + i] = arg.sin();
    }
    out
}

/// Rotate `x` in place by three-axis RoPE.
///
/// `x` is [T, n_heads * head_dim], `ids` is [T, 3]. Each axis owns a
/// contiguous slice of the head dimension, sized by `axes_dim`, and rotates
/// adjacent pairs within it.
pub fn flux_apply_rope(x: &mut Mat, ids: &Mat, n_heads: usize, head_dim: usize, axes_dim: &[usize], theta: f32) {
    debug_assert!(x.rows == ids.rows, "flux_apply_rope: token count mismatch");
    debug_assert!(ids.cols == axes_dim.len(), "flux_apply_rope: id width != axis count");
    debug_assert!(x.cols == n_heads * head_dim, "flux_apply_rope: width != n_heads * head_dim");

    // Precompute one angle per (token, pair). Every head applies the same
    // rotation, so this is computed once and reused 24 times.
    let pairs: usize = axes_dim.iter().map(|d| d / 2).sum();
    debug_assert!(pairs * 2 == head_dim, "flux_apply_rope: axes_dim must sum to head_dim");

    let mut cos_tab = vec![0.0f32; x.rows * pairs];
    let mut sin_tab = vec![0.0f32; x.rows * pairs];
    for t in 0..x.rows {
        let mut p = 0usize;
        for (a, &dim) in axes_dim.iter().enumerate() {
            let pos = ids.at(t, a);
            for i in 0..dim / 2 {
                // theta^(-2i/dim), matching `rope()`'s `arange(0, dim, 2)/dim`.
                //
                // Computed in f64. The reference does the same, and it is not
                // fussiness: `theta` is 10000 and the exponent is a ratio, so a
                // single-precision `pow` puts a relative error into every
                // frequency, which the position then multiplies up.
                let exponent = (2 * i) as f64 / dim as f64;
                let omega = 1.0 / (theta as f64).powf(exponent);
                let angle = pos as f64 * omega;
                cos_tab[t * pairs + p] = angle.cos() as f32;
                sin_tab[t * pairs + p] = angle.sin() as f32;
                p += 1;
            }
        }
    }

    let cols = x.cols;
    for t in 0..x.rows {
        let row = &mut x.data[t * cols..(t + 1) * cols];
        for h in 0..n_heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            for p in 0..pairs {
                // Adjacent pairs, not split halves.
                let a = head[2 * p];
                let b = head[2 * p + 1];
                let c = cos_tab[t * pairs + p];
                let s = sin_tab[t * pairs + p];
                head[2 * p] = a * c - b * s;
                head[2 * p + 1] = a * s + b * c;
            }
        }
    }
}

// =============================================================================
// Layers
// =============================================================================

/// One `(shift, scale, gate)` triple out of a modulation projection.
#[derive(Clone, Debug, Default)]
pub struct FluxModulation {
    pub shift: Vec<f32>,
    pub scale: Vec<f32>,
    pub gate: Vec<f32>,
}

/// Per-head RMSNorm applied to Q and K before rotation.
#[derive(Clone, Debug, Default)]
pub struct FluxQkNorm {
    pub query_scale: Vec<f32>, // [head_dim]
    pub key_scale: Vec<f32>,
}

#[derive(Clone, Default)]
pub struct FluxDoubleBlock {
    pub img_mod: QLinear, // hidden -> 6 * hidden
    pub img_qkv: QLinear, // hidden -> 3 * hidden
    pub img_proj: QLinear,
    pub img_mlp_in: QLinear,
    pub img_mlp_out: QLinear,
    pub img_norm: FluxQkNorm,

    pub txt_mod: QLinear,
    pub txt_qkv: QLinear,
    pub txt_proj: QLinear,
    pub txt_mlp_in: QLinear,
    pub txt_mlp_out: QLinear,
    pub txt_norm: FluxQkNorm,
}

#[derive(Clone, Default)]
pub struct FluxSingleBlock {
    pub modulation: QLinear, // hidden -> 3 * hidden
    /// hidden -> 3 * hidden + mlp_hidden, one projection doing QKV and the MLP
    /// up-projection together.
    pub linear1: QLinear,
    /// hidden + mlp_hidden -> hidden
    pub linear2: QLinear,
    pub norm: FluxQkNorm,
}

// =============================================================================
// Internals
// =============================================================================

/// LayerNorm with no affine parameters. Every norm in FLUX is one of these:
/// the scale and shift arrive from the modulation path instead.
fn layer_norm_noaffine(x: &Mat, eps: f32) -> Mat {
    let cols = x.cols;
    let mut out = Mat::zeros(x.rows, cols);
    let n = cols as f64;
    for r in 0..x.rows {
        let src = &x.data[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for c in 0..cols {
            sum += src[c] as f64;
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for c in 0..cols {
            let d = src[c] as f64 - mean;
            var += d * d;
        }
        let inv = (1.0 / (var / n + eps as f64).sqrt()) as f32;
        let mean_f = mean as f32;
        let dst = &mut out.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dst[c] = (src[c] - mean_f) * inv;
        }
    }
    out
}

/// `(1 + scale) * x + shift`, broadcast over rows.
fn modulate_inplace(x: &mut Mat, shift: &[f32], scale: &[f32]) {
    debug_assert!(shift.len() == x.cols && scale.len() == x.cols, "modulate: width mismatch");
    let cols = x.cols;
    for r in 0..x.rows {
        let row = &mut x.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] = (1.0 + scale[c]) * row[c] + shift[c];
        }
    }
}

/// `dst += gate * src`, broadcast over rows.
fn gated_add_inplace(dst: &mut Mat, src: &Mat, gate: &[f32]) {
    debug_assert!(dst.rows == src.rows && dst.cols == src.cols, "gated_add: shape mismatch");
    let cols = dst.cols;
    for r in 0..dst.rows {
        let d = &mut dst.data[r * cols..(r + 1) * cols];
        let s = &src.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            d[c] += gate[c] * s[c];
        }
    }
}

/// Split a `[1, k * hidden]` modulation projection into its triples.
fn split_modulation(m: &Mat, hidden: usize, triples: usize) -> Vec<FluxModulation> {
    debug_assert!(m.rows == 1 && m.cols == triples * 3 * hidden, "split_modulation: width mismatch");
    let src = &m.data[..m.cols];
    (0..triples)
        .map(|t| {
            let base = t * 3 * hidden;
            // Order within a triple is shift, scale, gate.
            FluxModulation {
                shift: src[base..base + hidden].to_vec(),
                scale: src[base + hidden..base + 2 * hidden].to_vec(),
                gate: src[base + 2 * hidden..base + 3 * hidden].to_vec(),
            }
        })
        .collect()
}

/// Per-head RMSNorm over the head dimension, in place.
fn head_rms_norm_inplace(x: &mut Mat, n_heads: usize, head_dim: usize, scale: &[f32], eps: f32) {
    debug_assert!(scale.len() == head_dim, "head_rms_norm: scale length != head_dim");
    let cols = x.cols;
    for r in 0..x.rows {
        let row = &mut x.data[r * cols..(r + 1) * cols];
        for h in 0..n_heads {
            let head = &mut row[h * head_dim..(h + 1) * head_dim];
            let mut sq = 0.0f64;
            for c in 0..head_dim {
                sq += head[c] as f64 * head[c] as f64;
            }
            let inv = (1.0 / (sq / head_dim as f64 + eps as f64).sqrt()) as f32;
            for c in 0..head_dim {
                head[c] = head[c] * inv * scale[c];
            }
        }
    }
}

/// Multi-head scaled dot-product attention with no mask.
///
/// Every token attends to every other one -- there is nothing causal about a
/// diffusion transformer, and the text tokens are as visible to the image as
/// the image is to itself.
fn attention(q: &Mat, k: &Mat, v: &Mat, n_heads: usize, head_dim: usize) -> Mat {
    let t = q.rows;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let width = n_heads * head_dim;
    let mut out = Mat::zeros(t, width);
    let mut scores = vec![0.0f32; t];

    for h in 0..n_heads {
        let off = h * head_dim;
        for i in 0..t {
            let qi = &q.data[i * q.cols + off..i * q.cols + off + head_dim];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..t {
                let kj = &k.data[j * k.cols + off..j * k.cols + off + head_dim];
                let mut acc = 0.0f32;
                for c in 0..head_dim {
                    acc += qi[c] * kj[c];
                }
                scores[j] = acc * scale;
                if max_score < scores[j] {
                    max_score = scores[j];
                }
            }
            let mut denom = 0.0f32;
            for j in 0..t {
                scores[j] = (scores[j] - max_score).exp();
                denom += scores[j];
            }
            let inv = 1.0f32 / denom;
            let dst = &mut out.data[i * width + off..i * width + off + head_dim];
            for j in 0..t {
                let weight = scores[j] * inv;
                let vj = &v.data[j * v.cols + off..j * v.cols + off + head_dim];
                for c in 0..head_dim {
                    dst[c] += weight * vj[c];
                }
            }
        }
    }
    out
}

/// Take columns `[from, to)` of every row.
fn slice_cols(x: &Mat, from: usize, to: usize) -> Mat {
    let w = to - from;
    let mut out = Mat::zeros(x.rows, w);
    for r in 0..x.rows {
        out.data[r * w..(r + 1) * w].copy_from_slice(&x.data[r * x.cols + from..r * x.cols + to]);
    }
    out
}

/// Stack `a` on top of `b`. Both must be the same width.
fn concat_rows(a: &Mat, b: &Mat) -> Mat {
    debug_assert!(a.cols == b.cols, "concat_rows: width mismatch");
    let mut data = Vec::with_capacity(a.data.len() + b.data.len());
    data.extend_from_slice(&a.data);
    data.extend_from_slice(&b.data);
    Mat::new(data, a.rows + b.rows, a.cols)
}

/// Take rows `[from, to)`.
fn slice_rows(x: &Mat, from: usize, to: usize) -> Mat {
    Mat::new(x.data[from * x.cols..to * x.cols].to_vec(), to - from, x.cols)
}

/// Join two matrices side by side.
fn concat_cols(a: &Mat, b: &Mat) -> Mat {
    debug_assert!(a.rows == b.rows, "concat_cols: row mismatch");
    let w = a.cols + b.cols;
    let mut out = Mat::zeros(a.rows, w);
    for r in 0..a.rows {
        let dst = &mut out.data[r * w..(r + 1) * w];
        dst[..a.cols].copy_from_slice(&a.data[r * a.cols..(r + 1) * a.cols]);
        dst[a.cols..].copy_from_slice(&b.data[r * b.cols..(r + 1) * b.cols]);
    }
    out
}

// =============================================================================
// Loading
// =============================================================================

fn io_err(e: std::io::Error) -> String {
    e.to_string()
}

fn load_linear(gguf: &GgufFile, name: &str, out_features: usize, in_features: usize, with_bias: bool) -> Result<QLinear, String> {
    let wname = format!("{}.weight", name);
    let idx = gguf.find_tensor(&wname).ok_or_else(|| format!("flux: missing tensor {}.weight", name))?;
    let info = &gguf.tensor_info[idx];
    if info.n_elements() != out_features * in_features {
        return Err(format!(
            "flux: {}.weight has {} elements, expected {}",
            name,
            info.n_elements(),
            out_features * in_features
        ));
    }

    let mut bias = Vec::new();
    if with_bias {
        let b = load_gguf_vector(gguf, &format!("{}.bias", name)).map_err(io_err)?;
        if b.len() != out_features {
            return Err(format!("flux: {}.bias has {} elements, expected {}", name, b.len(), out_features));
        }
        bias = b;
    }

    match info.gguf_type {
        GgufType::Q4K => {
            let q = gguf.decode_q4k_to_q4kmat(idx).map_err(io_err)?;
            Ok(QLinear::from_q4k(q, out_features, in_features, bias))
        }
        GgufType::Bf16 => {
            let bits = gguf.decode_bf16(idx).map_err(io_err)?;
            Ok(QLinear::from_bf16(MatBf16 { data: Arc::new(bits), rows: out_features, cols: in_features }, bias))
        }
        GgufType::F32 | GgufType::F16 => {
            // Kept at full precision. Widening f16 to f32 is exact, and folding
            // it down to bfloat instead would throw away two mantissa bits for
            // no saving worth having -- there are four such tensors in FLUX,
            // 100 MB between them, and one of them is the final layer's
            // modulation, which sets the scale of every output channel.
            let f = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
            Ok(QLinear::from_f32(Mat::new(f, out_features, in_features), bias))
        }
        _ => {
            // Q6_K, Q8_0 and Q5_K land here: decode to f32 and fold to BF16,
            // which halves the resident cost and throws away less than the
            // source format already did.
            let f = gguf_tensor_to_f32(gguf, idx).map_err(io_err)?;
            let m = Mat::new(f, out_features, in_features);
            Ok(QLinear::from_bf16(m.to_bf16(), bias))
        }
    }
}

fn load_scale(gguf: &GgufFile, name: &str, len: usize) -> Result<Vec<f32>, String> {
    let v = load_gguf_vector(gguf, name).map_err(io_err)?;
    if v.len() != len {
        return Err(format!("flux: {} has {} elements, expected {}", name, v.len(), len));
    }
    Ok(v)
}

/// Check that the RoPE axes tile the head dimension in whole pairs.
///
/// Each axis rotates adjacent pairs inside its own slice, so an odd axis width
/// leaves one dimension of that slice unrotated and pushes every later axis off
/// its frequencies. The result is a model that runs, produces finite numbers,
/// and attends to the wrong positions.
pub fn flux_check_axes(cfg: &FluxConfig) -> Result<(), String> {
    let mut axes_sum = 0usize;
    for &d in &cfg.axes_dim {
        // Each axis rotates adjacent pairs within its own slice, so an odd
        // width would leave one dimension of that slice unrotated and shift
        // every later axis off its own frequencies. The published FLUX axes --
        // 16, 56, 56 -- are all even for exactly this reason.
        if d % 2 != 0 {
            return Err(format!("flux: RoPE axis dimension {} is not even", d));
        }
        axes_sum += d;
    }
    if axes_sum != cfg.head_dim() {
        return Err(format!("flux: axes_dim sums to {}, expected head_dim {}", axes_sum, cfg.head_dim()));
    }
    Ok(())
}

// =============================================================================
// Model
// =============================================================================

#[derive(Clone)]
pub struct FluxModel {
    pub cfg: FluxConfig,

    pub img_in: QLinear,      // patch_dim -> hidden
    pub txt_in: QLinear,      // context_dim -> hidden
    pub time_in_1: QLinear,   // 256 -> hidden
    pub time_in_2: QLinear,   // hidden -> hidden
    pub vector_in_1: QLinear, // pooled_dim -> hidden
    pub vector_in_2: QLinear,
    /// dev only; empty on schnell.
    pub guidance_in_1: QLinear,
    pub guidance_in_2: QLinear,

    pub double_blocks: Vec<FluxDoubleBlock>,
    pub single_blocks: Vec<FluxSingleBlock>,

    pub final_mod: QLinear,    // hidden -> 2 * hidden
    pub final_linear: QLinear, // hidden -> patch_dim
}

impl Default for FluxModel {
    fn default() -> Self {
        FluxModel {
            cfg: FluxConfig::default(),
            img_in: QLinear::default(),
            txt_in: QLinear::default(),
            time_in_1: QLinear::default(),
            time_in_2: QLinear::default(),
            vector_in_1: QLinear::default(),
            vector_in_2: QLinear::default(),
            guidance_in_1: QLinear::default(),
            guidance_in_2: QLinear::default(),
            double_blocks: Vec::new(),
            single_blocks: Vec::new(),
            final_mod: QLinear::default(),
            final_linear: QLinear::default(),
        }
    }
}

impl FluxModel {
    /// Load from a GGUF checkpoint using the original FLUX tensor names, which
    /// is what the community Q4_K/Q5_K/Q8_0 conversions carry.
    pub fn load_gguf(path: &str, cfg: FluxConfig) -> Result<FluxModel, String> {
        flux_check_axes(&cfg)?;

        let gguf = GgufFile::open(path).map_err(|e| format!("gguf: cannot open {}: {}", path, e))?;

        let mut m = FluxModel { cfg: cfg.clone(), ..Default::default() };
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let mlp_hidden = cfg.mlp_hidden();

        m.img_in = load_linear(&gguf, "img_in", hidden, cfg.patch_dim(), true)?;
        m.txt_in = load_linear(&gguf, "txt_in", hidden, cfg.context_dim, true)?;
        m.time_in_1 = load_linear(&gguf, "time_in.in_layer", hidden, 256, true)?;
        m.time_in_2 = load_linear(&gguf, "time_in.out_layer", hidden, hidden, true)?;
        m.vector_in_1 = load_linear(&gguf, "vector_in.in_layer", hidden, cfg.pooled_dim, true)?;
        m.vector_in_2 = load_linear(&gguf, "vector_in.out_layer", hidden, hidden, true)?;

        if cfg.guidance_embed {
            m.guidance_in_1 = load_linear(&gguf, "guidance_in.in_layer", hidden, 256, true)?;
            m.guidance_in_2 = load_linear(&gguf, "guidance_in.out_layer", hidden, hidden, true)?;
        }

        m.double_blocks.reserve(cfg.n_double_blocks);
        for i in 0..cfg.n_double_blocks {
            let p = format!("double_blocks.{}.", i);
            let mut b = FluxDoubleBlock::default();
            b.img_mod = load_linear(&gguf, &format!("{p}img_mod.lin"), 6 * hidden, hidden, true)?;
            b.img_qkv = load_linear(&gguf, &format!("{p}img_attn.qkv"), 3 * hidden, hidden, true)?;
            b.img_proj = load_linear(&gguf, &format!("{p}img_attn.proj"), hidden, hidden, true)?;
            b.img_mlp_in = load_linear(&gguf, &format!("{p}img_mlp.0"), mlp_hidden, hidden, true)?;
            b.img_mlp_out = load_linear(&gguf, &format!("{p}img_mlp.2"), hidden, mlp_hidden, true)?;
            b.img_norm.query_scale = load_scale(&gguf, &format!("{p}img_attn.norm.query_norm.scale"), head_dim)?;
            b.img_norm.key_scale = load_scale(&gguf, &format!("{p}img_attn.norm.key_norm.scale"), head_dim)?;

            b.txt_mod = load_linear(&gguf, &format!("{p}txt_mod.lin"), 6 * hidden, hidden, true)?;
            b.txt_qkv = load_linear(&gguf, &format!("{p}txt_attn.qkv"), 3 * hidden, hidden, true)?;
            b.txt_proj = load_linear(&gguf, &format!("{p}txt_attn.proj"), hidden, hidden, true)?;
            b.txt_mlp_in = load_linear(&gguf, &format!("{p}txt_mlp.0"), mlp_hidden, hidden, true)?;
            b.txt_mlp_out = load_linear(&gguf, &format!("{p}txt_mlp.2"), hidden, mlp_hidden, true)?;
            b.txt_norm.query_scale = load_scale(&gguf, &format!("{p}txt_attn.norm.query_norm.scale"), head_dim)?;
            b.txt_norm.key_scale = load_scale(&gguf, &format!("{p}txt_attn.norm.key_norm.scale"), head_dim)?;

            m.double_blocks.push(b);
        }

        m.single_blocks.reserve(cfg.n_single_blocks);
        for i in 0..cfg.n_single_blocks {
            let p = format!("single_blocks.{}.", i);
            let mut b = FluxSingleBlock::default();
            b.modulation = load_linear(&gguf, &format!("{p}modulation.lin"), 3 * hidden, hidden, true)?;
            b.linear1 = load_linear(&gguf, &format!("{p}linear1"), 3 * hidden + mlp_hidden, hidden, true)?;
            b.linear2 = load_linear(&gguf, &format!("{p}linear2"), hidden, hidden + mlp_hidden, true)?;
            b.norm.query_scale = load_scale(&gguf, &format!("{p}norm.query_norm.scale"), head_dim)?;
            b.norm.key_scale = load_scale(&gguf, &format!("{p}norm.key_norm.scale"), head_dim)?;
            m.single_blocks.push(b);
        }

        m.final_mod = load_linear(&gguf, "final_layer.adaLN_modulation.1", 2 * hidden, hidden, true)?;
        m.final_linear = load_linear(&gguf, "final_layer.linear", cfg.patch_dim(), hidden, true)?;

        Ok(m)
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    /// One velocity prediction.
    ///
    /// `latent` is [lat_h * lat_w, in_channels] -- unpacked, as the sampler
    /// holds it. `context` is the T5 sequence [T_txt, context_dim] and `pooled`
    /// is the CLIP vector. `timestep` runs from 1 down to 0. `guidance` is
    /// ignored unless the config has `guidance_embed` (pass 0 otherwise).
    ///
    /// Returns a velocity of the same shape as `latent`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        latent: &Mat,
        lat_h: usize,
        lat_w: usize,
        context: &Mat,
        pooled: &[f32],
        timestep: f32,
        guidance: f32,
    ) -> Result<Mat, String> {
        let cfg = &self.cfg;
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let patch = cfg.patch_size;

        if latent.cols != cfg.in_channels {
            return Err(format!(
                "flux forward: latent has {} channels, expected {}",
                latent.cols, cfg.in_channels
            ));
        }
        if latent.rows != lat_h * lat_w {
            return Err("flux forward: latent rows != lat_h * lat_w".to_string());
        }
        if lat_h % patch != 0 || lat_w % patch != 0 {
            return Err("flux forward: latent dimensions must be multiples of the patch size".to_string());
        }
        if context.cols != cfg.context_dim {
            return Err(format!(
                "flux forward: context width {} != context_dim {}",
                context.cols, cfg.context_dim
            ));
        }
        if pooled.len() != cfg.pooled_dim {
            return Err(format!(
                "flux forward: pooled vector has {} entries, expected {}",
                pooled.len(),
                cfg.pooled_dim
            ));
        }
        flux_check_axes(cfg)?;

        // --- Conditioning vector ---------------------------------------------
        let t_emb = flux_timestep_embedding(timestep, 256, 10000.0, 1000.0);
        let mut vec = self.time_in_2.forward(&{
            let mut h = self.time_in_1.forward(&Mat::new(t_emb, 1, 256));
            silu_inplace(&mut h);
            h
        });

        if cfg.guidance_embed {
            let g_emb = flux_timestep_embedding(guidance, 256, 10000.0, 1000.0);
            let g = self.guidance_in_2.forward(&{
                let mut h = self.guidance_in_1.forward(&Mat::new(g_emb, 1, 256));
                silu_inplace(&mut h);
                h
            });
            for c in 0..hidden {
                vec.data[c] += g.data[c];
            }
        }

        {
            let y = Mat::new(pooled.to_vec(), 1, cfg.pooled_dim);
            let p = self.vector_in_2.forward(&{
                let mut h = self.vector_in_1.forward(&y);
                silu_inplace(&mut h);
                h
            });
            for c in 0..hidden {
                vec.data[c] += p.data[c];
            }
        }

        // The modulation projections all read `silu(vec)`, not `vec`.
        let mut mod_input = vec.clone();
        silu_inplace(&mut mod_input);

        // Debug hook: dump intermediates for the parity harness. Off unless the
        // environment asks -- the branch costs one environment lookup per
        // forward pass, which is beneath measurement next to 12B parameters.
        let dump_dir = std::env::var("RT_FLUX_DUMP").ok();
        let dump = |tag: &str, m: &Mat| {
            if let Some(dir) = &dump_dir {
                let bytes: Vec<u8> = m.data.iter().flat_map(|v| v.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{}/{}.bin", dir, tag), bytes);
            }
        };
        dump("vec", &vec);

        // --- Streams ---------------------------------------------------------
        let mut img = self.img_in.forward(&flux_patchify(latent, lat_h, lat_w, patch));
        let mut txt = self.txt_in.forward(context);
        dump("img_in", &img);
        dump("txt_in", &txt);

        let img_ids = flux_image_ids(lat_h, lat_w, patch);
        // Text positions are all zero, so their rotation is the identity. That
        // is deliberate: the prompt has no place in the image's coordinate
        // system.
        let txt_ids = Mat::zeros(txt.rows, 3);
        let ids = concat_rows(&txt_ids, &img_ids);
        let n_txt = txt.rows;

        // --- Double-stream blocks --------------------------------------------
        for (bi, b) in self.double_blocks.iter().enumerate() {
            let img_mod = split_modulation(&b.img_mod.forward(&mod_input), hidden, 2);
            let txt_mod = split_modulation(&b.txt_mod.forward(&mod_input), hidden, 2);

            let mut img_m = layer_norm_noaffine(&img, cfg.layer_norm_eps);
            modulate_inplace(&mut img_m, &img_mod[0].shift, &img_mod[0].scale);
            let img_qkv = b.img_qkv.forward(&img_m);

            let mut txt_m = layer_norm_noaffine(&txt, cfg.layer_norm_eps);
            modulate_inplace(&mut txt_m, &txt_mod[0].shift, &txt_mod[0].scale);
            let txt_qkv = b.txt_qkv.forward(&txt_m);

            // The fused projection is laid out [q | k | v], each head-major.
            let mut img_q = slice_cols(&img_qkv, 0, hidden);
            let mut img_k = slice_cols(&img_qkv, hidden, 2 * hidden);
            let img_v = slice_cols(&img_qkv, 2 * hidden, 3 * hidden);
            head_rms_norm_inplace(&mut img_q, cfg.n_heads, head_dim, &b.img_norm.query_scale, cfg.qk_norm_eps);
            head_rms_norm_inplace(&mut img_k, cfg.n_heads, head_dim, &b.img_norm.key_scale, cfg.qk_norm_eps);

            let mut q_txt = slice_cols(&txt_qkv, 0, hidden);
            let mut k_txt = slice_cols(&txt_qkv, hidden, 2 * hidden);
            let v_txt = slice_cols(&txt_qkv, 2 * hidden, 3 * hidden);
            head_rms_norm_inplace(&mut q_txt, cfg.n_heads, head_dim, &b.txt_norm.query_scale, cfg.qk_norm_eps);
            head_rms_norm_inplace(&mut k_txt, cfg.n_heads, head_dim, &b.txt_norm.key_scale, cfg.qk_norm_eps);

            // Text first, then image -- the order the position ids were built in.
            let mut q = concat_rows(&q_txt, &img_q);
            let mut k = concat_rows(&k_txt, &img_k);
            let v = concat_rows(&v_txt, &img_v);
            flux_apply_rope(&mut q, &ids, cfg.n_heads, head_dim, &cfg.axes_dim, cfg.rope_theta);
            flux_apply_rope(&mut k, &ids, cfg.n_heads, head_dim, &cfg.axes_dim, cfg.rope_theta);

            let attn = attention(&q, &k, &v, cfg.n_heads, head_dim);
            let txt_attn = slice_rows(&attn, 0, n_txt);
            let img_attn = slice_rows(&attn, n_txt, attn.rows);

            gated_add_inplace(&mut img, &b.img_proj.forward(&img_attn), &img_mod[0].gate);
            {
                let mut h = layer_norm_noaffine(&img, cfg.layer_norm_eps);
                modulate_inplace(&mut h, &img_mod[1].shift, &img_mod[1].scale);
                let mut f = b.img_mlp_in.forward(&h);
                gelu_tanh_inplace(&mut f);
                gated_add_inplace(&mut img, &b.img_mlp_out.forward(&f), &img_mod[1].gate);
            }

            gated_add_inplace(&mut txt, &b.txt_proj.forward(&txt_attn), &txt_mod[0].gate);
            {
                let mut h = layer_norm_noaffine(&txt, cfg.layer_norm_eps);
                modulate_inplace(&mut h, &txt_mod[1].shift, &txt_mod[1].scale);
                let mut f = b.txt_mlp_in.forward(&h);
                gelu_tanh_inplace(&mut f);
                gated_add_inplace(&mut txt, &b.txt_mlp_out.forward(&f), &txt_mod[1].gate);
            }
            if dump_dir.is_some() && bi == 0 {
                dump("block0_img", &img);
                dump("block0_txt", &txt);
            }
        }

        if dump_dir.is_some() {
            dump("after_double", &concat_rows(&txt, &img));
        }

        // --- Single-stream blocks --------------------------------------------
        let mut x = concat_rows(&txt, &img);
        let mlp_hidden = cfg.mlp_hidden();

        for b in &self.single_blocks {
            let m = split_modulation(&b.modulation.forward(&mod_input), hidden, 1);

            let mut x_mod = layer_norm_noaffine(&x, cfg.layer_norm_eps);
            modulate_inplace(&mut x_mod, &m[0].shift, &m[0].scale);

            // One projection produces QKV and the MLP's up-projection together.
            let fused = b.linear1.forward(&x_mod);
            let mut q = slice_cols(&fused, 0, hidden);
            let mut k = slice_cols(&fused, hidden, 2 * hidden);
            let v = slice_cols(&fused, 2 * hidden, 3 * hidden);
            let mut mlp = slice_cols(&fused, 3 * hidden, 3 * hidden + mlp_hidden);

            head_rms_norm_inplace(&mut q, cfg.n_heads, head_dim, &b.norm.query_scale, cfg.qk_norm_eps);
            head_rms_norm_inplace(&mut k, cfg.n_heads, head_dim, &b.norm.key_scale, cfg.qk_norm_eps);
            flux_apply_rope(&mut q, &ids, cfg.n_heads, head_dim, &cfg.axes_dim, cfg.rope_theta);
            flux_apply_rope(&mut k, &ids, cfg.n_heads, head_dim, &cfg.axes_dim, cfg.rope_theta);

            let attn = attention(&q, &k, &v, cfg.n_heads, head_dim);
            gelu_tanh_inplace(&mut mlp);

            // Attention and MLP run in parallel from the same input and are
            // joined before a single output projection, rather than in
            // sequence.
            let joined = concat_cols(&attn, &mlp);
            gated_add_inplace(&mut x, &b.linear2.forward(&joined), &m[0].gate);
        }

        // --- Final layer -----------------------------------------------------
        let out_img = slice_rows(&x, n_txt, x.rows);
        let fm = self.final_mod.forward(&mod_input);
        // Two values here, not three, and in the order shift then scale.
        let shift = &fm.data[0..hidden];
        let scale = &fm.data[hidden..2 * hidden];

        let mut normed = layer_norm_noaffine(&out_img, cfg.layer_norm_eps);
        modulate_inplace(&mut normed, shift, scale);
        let tokens = self.final_linear.forward(&normed);

        Ok(flux_unpatchify(&tokens, lat_h, lat_w, cfg.in_channels, patch))
    }

    pub fn parameter_count(&self) -> usize {
        let count = |l: &QLinear| l.out_features * l.in_features + l.bias.len();
        let mut n = count(&self.img_in)
            + count(&self.txt_in)
            + count(&self.time_in_1)
            + count(&self.time_in_2)
            + count(&self.vector_in_1)
            + count(&self.vector_in_2)
            + count(&self.guidance_in_1)
            + count(&self.guidance_in_2)
            + count(&self.final_mod)
            + count(&self.final_linear);
        for b in &self.double_blocks {
            n += count(&b.img_mod)
                + count(&b.img_qkv)
                + count(&b.img_proj)
                + count(&b.img_mlp_in)
                + count(&b.img_mlp_out)
                + b.img_norm.query_scale.len()
                + b.img_norm.key_scale.len();
            n += count(&b.txt_mod)
                + count(&b.txt_qkv)
                + count(&b.txt_proj)
                + count(&b.txt_mlp_in)
                + count(&b.txt_mlp_out)
                + b.txt_norm.query_scale.len()
                + b.txt_norm.key_scale.len();
        }
        for b in &self.single_blocks {
            n += count(&b.modulation)
                + count(&b.linear1)
                + count(&b.linear2)
                + b.norm.query_scale.len()
                + b.norm.key_scale.len();
        }
        n
    }

    pub fn weight_bytes(&self) -> usize {
        let mut n = self.img_in.size_bytes()
            + self.txt_in.size_bytes()
            + self.time_in_1.size_bytes()
            + self.time_in_2.size_bytes()
            + self.vector_in_1.size_bytes()
            + self.vector_in_2.size_bytes()
            + self.guidance_in_1.size_bytes()
            + self.guidance_in_2.size_bytes()
            + self.final_mod.size_bytes()
            + self.final_linear.size_bytes();
        for b in &self.double_blocks {
            n += b.img_mod.size_bytes()
                + b.img_qkv.size_bytes()
                + b.img_proj.size_bytes()
                + b.img_mlp_in.size_bytes()
                + b.img_mlp_out.size_bytes();
            n += b.txt_mod.size_bytes()
                + b.txt_qkv.size_bytes()
                + b.txt_proj.size_bytes()
                + b.txt_mlp_in.size_bytes()
                + b.txt_mlp_out.size_bytes();
        }
        for b in &self.single_blocks {
            n += b.modulation.size_bytes() + b.linear1.size_bytes() + b.linear2.size_bytes();
        }
        n
    }

    pub fn free_weights(&mut self) {
        for l in [
            &mut self.img_in,
            &mut self.txt_in,
            &mut self.time_in_1,
            &mut self.time_in_2,
            &mut self.vector_in_1,
            &mut self.vector_in_2,
            &mut self.guidance_in_1,
            &mut self.guidance_in_2,
            &mut self.final_mod,
            &mut self.final_linear,
        ] {
            l.free_weight();
        }
        self.double_blocks.clear();
        self.single_blocks.clear();
    }
}

// =============================================================================
// Sampling
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct FluxSampleParams {
    pub width: usize,
    pub height: usize,
    pub steps: usize,
    pub seed: u64,
    /// Ignored unless the config has `guidance_embed`.
    pub guidance: f32,
    /// Timestep shift. schnell uses 1.0, which is no shift at all; dev shifts
    /// as a function of sequence length.
    pub shift: f32,
}

impl Default for FluxSampleParams {
    fn default() -> Self {
        FluxSampleParams { width: 1024, height: 1024, steps: 4, seed: 0, guidance: 3.5, shift: 1.0 }
    }
}

/// The rectified-flow schedule: `steps + 1` sigmas running from 1 down to 0.
///
/// With `shift == 1` this is a plain linear spacing, which is what schnell
/// wants. Larger values push the samples toward the noisy end, where a
/// many-step sampler needs the resolution.
pub fn flux_schedule(steps: usize, shift: f32) -> Vec<f32> {
    (0..=steps)
        .map(|i| {
            let t = 1.0f32 - i as f32 / steps as f32;
            // The shift reparameterises the schedule toward the noisy end. At
            // shift == 1 it is the identity, which is what schnell wants.
            if shift == 1.0 { t } else { (shift * t) / (1.0 + (shift - 1.0) * t) }
        })
        .collect()
}

/// Sample a latent by Euler integration of the velocity field.
///
/// Returns [lat_h * lat_w, in_channels], still in the model's scaling -- hand
/// it straight to `VaeDecoder::decode`, which applies the rescaling itself.
pub fn flux_sample(model: &FluxModel, context: &Mat, pooled: &[f32], params: &FluxSampleParams) -> Result<Mat, String> {
    let cfg = &model.cfg;
    let factor = 8usize; // the VAE's spatial compression
    if params.width % (factor * cfg.patch_size) != 0 || params.height % (factor * cfg.patch_size) != 0 {
        return Err(format!(
            "flux sample: width and height must be multiples of {}",
            factor * cfg.patch_size
        ));
    }
    if params.steps == 0 {
        return Err("flux sample: steps must be > 0".to_string());
    }

    let lat_h = params.height / factor;
    let lat_w = params.width / factor;

    // Gaussian noise at sigma 1, which is where the schedule starts.
    let mut rng = InitRng::new(params.seed);
    let mut x = Mat::zeros(lat_h * lat_w, cfg.in_channels);
    for v in x.data.iter_mut() {
        *v = rng.next_normal();
    }

    let sigmas = flux_schedule(params.steps, params.shift);
    for i in 0..params.steps {
        let velocity = model.forward(&x, lat_h, lat_w, context, pooled, sigmas[i], params.guidance)?;
        // Euler: one step of `dx = v dt` along a schedule that runs downward,
        // so the increment is negative.
        let dt = sigmas[i + 1] - sigmas[i];
        for (xv, vv) in x.data.iter_mut().zip(&velocity.data) {
            *xv += dt * *vv;
        }
    }
    Ok(x)
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

    fn spread_vec(n: usize, scale: f32, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = ((i * 53 + salt * 6151) % 241) as f32;
                ((k * 0.29).sin() * 0.6 + (k * 0.17).cos() * 0.4) * scale
            })
            .collect()
    }

    fn linear(out: usize, inp: usize, salt: usize) -> QLinear {
        QLinear::from_f32(spread(out, inp, 0.1, salt), spread_vec(out, 0.02, salt + 1))
    }

    /// A FLUX of the real shape but a hundredth of the width. Every structural
    /// property under test -- the patch order, the stream split, the RoPE axes,
    /// the modulation arithmetic -- is independent of the widths.
    fn tiny_config() -> FluxConfig {
        FluxConfig {
            in_channels: 4,
            hidden_size: 32,
            n_heads: 4, // head_dim 8
            n_double_blocks: 2,
            n_single_blocks: 2,
            mlp_ratio: 2.0, // mlp_hidden 64
            context_dim: 12,
            pooled_dim: 6,
            axes_dim: vec![2, 2, 4], // even, and sums to head_dim 8
            guidance_embed: false,
            patch_size: 2,
            ..Default::default()
        }
    }

    fn make_model(cfg: &FluxConfig) -> FluxModel {
        let mut m = FluxModel { cfg: cfg.clone(), ..Default::default() };
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let mlp = cfg.mlp_hidden();

        m.img_in = linear(hidden, cfg.patch_dim(), 1);
        m.txt_in = linear(hidden, cfg.context_dim, 3);
        m.time_in_1 = linear(hidden, 256, 5);
        m.time_in_2 = linear(hidden, hidden, 7);
        m.vector_in_1 = linear(hidden, cfg.pooled_dim, 9);
        m.vector_in_2 = linear(hidden, hidden, 11);
        if cfg.guidance_embed {
            m.guidance_in_1 = linear(hidden, 256, 13);
            m.guidance_in_2 = linear(hidden, hidden, 15);
        }

        for i in 0..cfg.n_double_blocks {
            let b = FluxDoubleBlock {
                img_mod: linear(6 * hidden, hidden, 20 + i * 20),
                img_qkv: linear(3 * hidden, hidden, 22 + i * 20),
                img_proj: linear(hidden, hidden, 24 + i * 20),
                img_mlp_in: linear(mlp, hidden, 26 + i * 20),
                img_mlp_out: linear(hidden, mlp, 28 + i * 20),
                img_norm: FluxQkNorm { query_scale: vec![1.0; head_dim], key_scale: vec![1.0; head_dim] },
                txt_mod: linear(6 * hidden, hidden, 30 + i * 20),
                txt_qkv: linear(3 * hidden, hidden, 32 + i * 20),
                txt_proj: linear(hidden, hidden, 34 + i * 20),
                txt_mlp_in: linear(mlp, hidden, 36 + i * 20),
                txt_mlp_out: linear(hidden, mlp, 38 + i * 20),
                txt_norm: FluxQkNorm { query_scale: vec![1.0; head_dim], key_scale: vec![1.0; head_dim] },
            };
            m.double_blocks.push(b);
        }

        for i in 0..cfg.n_single_blocks {
            m.single_blocks.push(FluxSingleBlock {
                modulation: linear(3 * hidden, hidden, 60 + i * 10),
                linear1: linear(3 * hidden + mlp, hidden, 62 + i * 10),
                linear2: linear(hidden, hidden + mlp, 64 + i * 10),
                norm: FluxQkNorm { query_scale: vec![1.0; head_dim], key_scale: vec![1.0; head_dim] },
            });
        }

        m.final_mod = linear(2 * hidden, hidden, 90);
        m.final_linear = linear(cfg.patch_dim(), hidden, 92);
        m
    }

    fn max_delta(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    #[test]
    fn schnell_and_dev_differ_only_in_the_guidance_embedding() {
        let s = FluxConfig::schnell();
        let d = FluxConfig::dev();
        assert!(!s.guidance_embed);
        assert!(d.guidance_embed);
        assert_eq!(s.hidden_size, d.hidden_size);
        assert_eq!(s.n_double_blocks, d.n_double_blocks);
        assert_eq!(s.n_single_blocks, d.n_single_blocks);
    }

    #[test]
    fn the_flux_shape_constants_line_up() {
        let c = FluxConfig::schnell();
        assert_eq!(c.hidden_size, 3072);
        assert_eq!(c.n_heads, 24);
        assert_eq!(c.head_dim(), 128);
        assert_eq!(c.n_double_blocks, 19);
        assert_eq!(c.n_single_blocks, 38);
        assert_eq!(c.mlp_hidden(), 12288);
        // 16 channels through a 2x2 patch is a 64-wide token.
        assert_eq!(c.patch_dim(), 64);
        assert_eq!(c.context_dim, 4096); // T5-XXL
        assert_eq!(c.pooled_dim, 768); // CLIP-L
        // The RoPE axes must fill a head exactly.
        assert_eq!(c.axes_dim.iter().sum::<usize>(), c.head_dim());
        assert_eq!(c.axes_dim, vec![16, 56, 56]);
    }

    // -------------------------------------------------------------------------
    // Patching
    // -------------------------------------------------------------------------

    #[test]
    fn patchify_and_unpatchify_are_inverses() {
        let z = spread(8 * 6, 4, 1.0, 100);
        let tokens = flux_patchify(&z, 8, 6, 2);
        assert_eq!(tokens.rows, 4 * 3);
        assert_eq!(tokens.cols, 4 * 4);
        let back = flux_unpatchify(&tokens, 8, 6, 4, 2);
        assert_eq!(back.rows, z.rows);
        assert_eq!(back.cols, z.cols);
        for i in 0..z.data.len() {
            assert!(approx(back.data[i], z.data[i], 1e-7));
        }
    }

    #[test]
    fn patchify_packs_channel_major_within_a_patch() {
        // A latent whose value encodes its own (channel, y, x) makes the packing
        // order readable directly.
        let (h, w, ch) = (4usize, 4usize, 2usize);
        let mut z = Mat::zeros(h * w, ch);
        for y in 0..h {
            for x in 0..w {
                for c in 0..ch {
                    *z.at_mut(y * w + x, c) = (c * 100 + y * 10 + x) as f32;
                }
            }
        }
        let tokens = flux_patchify(&z, h, w, 2);
        // Patch (0, 0) holds channel 0's four pixels first, then channel 1's.
        let p0 = &tokens.data[0..tokens.cols];
        assert!(approx(p0[0], 0.0, 1e-6)); // c0 (0,0)
        assert!(approx(p0[1], 1.0, 1e-6)); // c0 (0,1)
        assert!(approx(p0[2], 10.0, 1e-6)); // c0 (1,0)
        assert!(approx(p0[3], 11.0, 1e-6)); // c0 (1,1)
        assert!(approx(p0[4], 100.0, 1e-6)); // c1 (0,0)
        assert!(approx(p0[7], 111.0, 1e-6)); // c1 (1,1)

        // Patch (0, 1) is two pixels to the right.
        let p1 = &tokens.data[tokens.cols..2 * tokens.cols];
        assert!(approx(p1[0], 2.0, 1e-6));
        assert!(approx(p1[3], 13.0, 1e-6));
    }

    #[test]
    fn patchify_is_not_pixel_major() {
        // The plausible alternative ordering -- pixel-major within the patch --
        // agrees with the real one only when there is a single channel.
        let z = spread(4 * 4, 3, 1.0, 101);
        let tokens = flux_patchify(&z, 4, 4, 2);
        // Under channel-major, columns 0..3 are all channel 0. Under pixel-major
        // they would be pixel (0,0)'s three channels then pixel (0,1)'s first.
        assert!(approx(tokens.at(0, 0), z.at(0, 0), 1e-7));
        assert!(approx(tokens.at(0, 1), z.at(1, 0), 1e-7));
        assert!(!approx(tokens.at(0, 1), z.at(0, 1), 1e-7));
    }

    #[test]
    fn image_ids_carry_the_patch_grid_coordinates_on_axes_1_and_2() {
        let ids = flux_image_ids(8, 6, 2);
        assert_eq!(ids.rows, 4 * 3);
        assert_eq!(ids.cols, 3);
        for y in 0..4 {
            for x in 0..3 {
                let r = y * 3 + x;
                assert!(approx(ids.at(r, 0), 0.0, 1e-7)); // the video axis
                assert!(approx(ids.at(r, 1), y as f32, 1e-7));
                assert!(approx(ids.at(r, 2), x as f32, 1e-7));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Timestep embedding
    // -------------------------------------------------------------------------

    #[test]
    fn the_timestep_embedding_puts_cosines_first() {
        let e = flux_timestep_embedding(0.0, 8, 10000.0, 1000.0);
        assert_eq!(e.len(), 8);
        // At t = 0 every argument is zero, so the cosine half is all ones and
        // the sine half is all zeros. Swapping the halves is immediately
        // visible.
        for i in 0..4 {
            assert!(approx(e[i], 1.0, 1e-6));
            assert!(approx(e[4 + i], 0.0, 1e-6));
        }
    }

    #[test]
    fn the_timestep_embedding_scales_t_by_1000() {
        // The lowest frequency is 1, so its argument is exactly `1000 * t`.
        let e = flux_timestep_embedding(0.001, 8, 10000.0, 1000.0);
        assert!(approx(e[0], 1.0f32.cos(), 1e-5));
        assert!(approx(e[4], 1.0f32.sin(), 1e-5));
    }

    #[test]
    fn the_timestep_embedding_separates_nearby_timesteps() {
        let a = flux_timestep_embedding(0.25, 256, 10000.0, 1000.0);
        let b = flux_timestep_embedding(0.26, 256, 10000.0, 1000.0);
        assert!(max_delta(&a, &b) > 0.1);
    }

    // -------------------------------------------------------------------------
    // RoPE
    // -------------------------------------------------------------------------

    // The odd {2, 3, 3} axes in the next two tests are what the CPP tests pass.
    // They do not fill the head, which `flux_apply_rope` asserts in debug
    // builds -- CPP runs its tests with that assertion compiled out (Release,
    // NDEBUG), so they only run here with debug assertions off.
    #[test]
    #[cfg_attr(debug_assertions, ignore = "odd axes trip the debug assertion, as CPP's assert would")]
    fn rope_leaves_zero_positions_untouched() {
        // Text tokens carry all-zero ids, so their rotation must be the
        // identity. If it is not, the prompt is rotated as though it sat at the
        // image's origin and the two streams stop agreeing on what position
        // means.
        let original = spread(5, 4 * 8, 1.0, 200);
        let mut x = original.clone();
        let ids = Mat::zeros(5, 3);
        flux_apply_rope(&mut x, &ids, 4, 8, &[2, 3, 3], 10000.0);
        for i in 0..x.data.len() {
            assert!(approx(x.data[i], original.data[i], 1e-6));
        }
    }

    #[test]
    #[cfg_attr(debug_assertions, ignore = "odd axes trip the debug assertion, as CPP's assert would")]
    fn rope_preserves_the_norm_of_every_pair() {
        let mut x = spread(6, 4 * 8, 1.0, 201);
        let before = x.clone();
        let mut ids = Mat::zeros(6, 3);
        for r in 0..6 {
            *ids.at_mut(r, 1) = r as f32;
            *ids.at_mut(r, 2) = (r * 2) as f32;
        }
        flux_apply_rope(&mut x, &ids, 4, 8, &[2, 3, 3], 10000.0);

        for r in 0..6 {
            for h in 0..4 {
                for p in 0..4 {
                    let at = h * 8 + 2 * p;
                    let n0 = before.at(r, at) * before.at(r, at) + before.at(r, at + 1) * before.at(r, at + 1);
                    let n1 = x.at(r, at) * x.at(r, at) + x.at(r, at + 1) * x.at(r, at + 1);
                    assert!(approx(n0, n1, 1e-4));
                }
            }
        }
    }

    #[test]
    fn rope_rotates_adjacent_pairs_not_split_halves() {
        // One head, one axis, dimension 2: a single pair at a known angle.
        let mut x = Mat::new(vec![1.0, 0.0], 1, 2);
        let ids = Mat::new(vec![1.0], 1, 1);
        flux_apply_rope(&mut x, &ids, 1, 2, &[2], 10000.0);
        // omega = theta^0 = 1, so the angle is exactly the position.
        assert!(approx(x.at(0, 0), 1.0f32.cos(), 1e-6));
        assert!(approx(x.at(0, 1), 1.0f32.sin(), 1e-6));
    }

    #[test]
    fn each_rope_axis_rotates_its_own_slice_of_the_head() {
        // Moving only the h coordinate must leave the w slice alone.
        let head_dim = 8;
        let axes = [2usize, 2, 4];
        let mut a = spread(1, head_dim, 1.0, 202);
        let mut b = a.clone();
        let ids_a = Mat::zeros(1, 3);
        let mut ids_b = Mat::zeros(1, 3);
        *ids_b.at_mut(0, 1) = 3.0; // move h only

        flux_apply_rope(&mut a, &ids_a, 1, head_dim, &axes, 10000.0);
        flux_apply_rope(&mut b, &ids_b, 1, head_dim, &axes, 10000.0);

        // Axis 0 owns dims 0..1, axis 1 owns 2..3, axis 2 owns 4..7. Only axis
        // 1's slice may change.
        assert!(approx(a.at(0, 0), b.at(0, 0), 1e-6));
        assert!(approx(a.at(0, 1), b.at(0, 1), 1e-6));
        assert!(!approx(a.at(0, 2), b.at(0, 2), 1e-5));
        assert!(!approx(a.at(0, 3), b.at(0, 3), 1e-5));
        for d in 4..8 {
            assert!(approx(a.at(0, d), b.at(0, d), 1e-6));
        }
    }

    #[test]
    fn check_axes_rejects_axes_that_do_not_tile_the_head() {
        let mut c = tiny_config();
        assert!(flux_check_axes(&c).is_ok());

        // An odd axis leaves a dimension unrotated and shifts everything after it.
        c.axes_dim = vec![2, 3, 3];
        assert!(flux_check_axes(&c).is_err());

        // And the axes still have to fill the head exactly.
        c.axes_dim = vec![2, 2, 2];
        assert!(flux_check_axes(&c).is_err());

        assert!(flux_check_axes(&FluxConfig::schnell()).is_ok());
        assert!(flux_check_axes(&FluxConfig::dev()).is_ok());
    }

    // -------------------------------------------------------------------------
    // Schedule
    // -------------------------------------------------------------------------

    #[test]
    fn the_schedule_runs_from_1_to_0_with_one_more_entry_than_steps() {
        let s = flux_schedule(4, 1.0);
        assert_eq!(s.len(), 5);
        assert!(approx(s[0], 1.0, 1e-6));
        assert!(approx(*s.last().unwrap(), 0.0, 1e-6));
        for i in 0..s.len() - 1 {
            assert!(s[i] > s[i + 1]);
        }
    }

    #[test]
    fn a_shift_of_1_is_a_linear_schedule() {
        let s = flux_schedule(4, 1.0);
        assert!(approx(s[0], 1.0, 1e-6));
        assert!(approx(s[1], 0.75, 1e-6));
        assert!(approx(s[2], 0.5, 1e-6));
        assert!(approx(s[3], 0.25, 1e-6));
        assert!(approx(s[4], 0.0, 1e-6));
    }

    #[test]
    fn a_shift_above_1_pushes_the_schedule_toward_the_noisy_end() {
        let plain = flux_schedule(8, 1.0);
        let shifted = flux_schedule(8, 3.0);
        assert!(approx(shifted[0], 1.0, 1e-6));
        assert!(approx(*shifted.last().unwrap(), 0.0, 1e-6));
        // Every interior sigma sits higher, so more steps are spent at high noise.
        for i in 1..plain.len() - 1 {
            assert!(shifted[i] > plain[i]);
        }
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    #[test]
    fn the_model_returns_a_velocity_the_shape_of_its_latent() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let z = spread(8 * 6, 4, 1.0, 300);
        let ctx = spread(5, cfg.context_dim, 1.0, 301);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 302);

        let v = m.forward(&z, 8, 6, &ctx, &pooled, 0.75, 0.0).unwrap();
        assert_eq!(v.rows, z.rows);
        assert_eq!(v.cols, z.cols);
        assert!(v.data.iter().all(|f| f.is_finite()));
    }

    #[test]
    fn the_model_rejects_shapes_it_cannot_patch() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let ctx = spread(5, cfg.context_dim, 1.0, 303);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 304);

        // Odd latent dimensions cannot be split into 2x2 patches.
        assert!(m.forward(&spread(7 * 6, 4, 1.0, 1), 7, 6, &ctx, &pooled, 0.5, 0.0).is_err());
        // Wrong channel count.
        assert!(m.forward(&spread(8 * 6, 5, 1.0, 1), 8, 6, &ctx, &pooled, 0.5, 0.0).is_err());
        // Wrong context width.
        assert!(m.forward(&spread(8 * 6, 4, 1.0, 1), 8, 6, &spread(5, 7, 1.0, 1), &pooled, 0.5, 0.0).is_err());
        // Wrong pooled width.
        assert!(m.forward(&spread(8 * 6, 4, 1.0, 1), 8, 6, &ctx, &[0.0; 3], 0.5, 0.0).is_err());
    }

    #[test]
    fn the_velocity_depends_on_the_timestep() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let z = spread(4 * 4, 4, 1.0, 305);
        let ctx = spread(3, cfg.context_dim, 1.0, 306);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 307);

        let a = m.forward(&z, 4, 4, &ctx, &pooled, 0.9, 0.0).unwrap();
        let b = m.forward(&z, 4, 4, &ctx, &pooled, 0.1, 0.0).unwrap();
        assert!(max_delta(&a.data, &b.data) > 1e-5);
    }

    #[test]
    fn the_velocity_depends_on_the_prompt_and_on_the_pooled_vector_separately() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let z = spread(4 * 4, 4, 1.0, 308);
        let ctx = spread(3, cfg.context_dim, 1.0, 309);
        let ctx2 = spread(3, cfg.context_dim, 1.0, 310);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 311);
        let pooled2 = spread_vec(cfg.pooled_dim, 1.0, 312);

        let base = m.forward(&z, 4, 4, &ctx, &pooled, 0.5, 0.0).unwrap();
        let other_text = m.forward(&z, 4, 4, &ctx2, &pooled, 0.5, 0.0).unwrap();
        let other_pooled = m.forward(&z, 4, 4, &ctx, &pooled2, 0.5, 0.0).unwrap();

        assert!(max_delta(&base.data, &other_text.data) > 1e-5);
        assert!(max_delta(&base.data, &other_pooled.data) > 1e-5);
    }

    #[test]
    fn the_image_stream_is_spatially_aware() {
        // Perturbing one patch must change a distant patch's velocity. Only the
        // joint attention can carry that, so this fails if the streams never
        // meet.
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let mut z = spread(6 * 6, 4, 1.0, 313);
        let ctx = spread(3, cfg.context_dim, 1.0, 314);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 315);

        let base = m.forward(&z, 6, 6, &ctx, &pooled, 0.5, 0.0).unwrap();
        *z.at_mut(0, 0) += 3.0;
        let poked = m.forward(&z, 6, 6, &ctx, &pooled, 0.5, 0.0).unwrap();

        let far = 6 * 6 - 1;
        let mut delta = 0.0f32;
        for c in 0..4 {
            delta = delta.max((base.at(far, c) - poked.at(far, c)).abs());
        }
        assert!(delta > 1e-6);
    }

    #[test]
    fn the_guidance_embedding_is_used_only_when_the_config_asks_for_it() {
        let mut cfg = tiny_config();
        cfg.guidance_embed = true;
        let m = make_model(&cfg);
        let z = spread(4 * 4, 4, 1.0, 316);
        let ctx = spread(3, cfg.context_dim, 1.0, 317);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 318);

        let a = m.forward(&z, 4, 4, &ctx, &pooled, 0.5, 1.0).unwrap();
        let b = m.forward(&z, 4, 4, &ctx, &pooled, 0.5, 7.0).unwrap();
        assert!(max_delta(&a.data, &b.data) > 1e-5);

        // schnell has no guidance path, so the argument is inert there.
        let s = make_model(&tiny_config());
        let c = s.forward(&z, 4, 4, &ctx, &pooled, 0.5, 1.0).unwrap();
        let d = s.forward(&z, 4, 4, &ctx, &pooled, 0.5, 7.0).unwrap();
        assert_eq!(c.data, d.data);
    }

    // -------------------------------------------------------------------------
    // Sampling
    // -------------------------------------------------------------------------

    #[test]
    fn sampling_returns_a_latent_of_the_right_shape() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let ctx = spread(3, cfg.context_dim, 1.0, 400);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 401);

        let p = FluxSampleParams { width: 64, height: 32, steps: 2, seed: 7, ..Default::default() };
        let z = flux_sample(&m, &ctx, &pooled, &p).unwrap();
        assert_eq!(z.rows, (32 / 8) * (64 / 8));
        assert_eq!(z.cols, cfg.in_channels);
        assert!(z.data.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn sampling_is_reproducible_from_its_seed() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let ctx = spread(3, cfg.context_dim, 1.0, 402);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 403);

        let mut p = FluxSampleParams { width: 32, height: 32, steps: 2, seed: 99, ..Default::default() };
        let a = flux_sample(&m, &ctx, &pooled, &p).unwrap();
        let b = flux_sample(&m, &ctx, &pooled, &p).unwrap();
        assert_eq!(a.data, b.data);

        p.seed = 100;
        let c = flux_sample(&m, &ctx, &pooled, &p).unwrap();
        assert_ne!(a.data, c.data);
    }

    #[test]
    fn sampling_rejects_dimensions_that_do_not_divide() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let ctx = spread(3, cfg.context_dim, 1.0, 404);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 405);

        let mut p = FluxSampleParams { steps: 1, width: 20, height: 32, ..Default::default() }; // 20 is not a multiple of 16
        assert!(flux_sample(&m, &ctx, &pooled, &p).is_err());

        p.width = 32;
        p.steps = 0;
        assert!(flux_sample(&m, &ctx, &pooled, &p).is_err());
    }

    #[test]
    fn a_zero_velocity_model_leaves_the_noise_untouched() {
        // Zeroing the final projection makes every velocity zero, so Euler
        // integration must return exactly the initial noise. That pins the
        // integration to `x += dt * v` and nothing else.
        let cfg = tiny_config();
        let mut m = make_model(&cfg);
        m.final_linear = QLinear::from_f32(Mat::zeros(cfg.patch_dim(), cfg.hidden_size), vec![0.0; cfg.patch_dim()]);

        let ctx = spread(3, cfg.context_dim, 1.0, 406);
        let pooled = spread_vec(cfg.pooled_dim, 1.0, 407);

        let p = FluxSampleParams { width: 32, height: 32, steps: 4, seed: 5, ..Default::default() };
        let z = flux_sample(&m, &ctx, &pooled, &p).unwrap();

        let mut rng = InitRng::new(5);
        for &v in &z.data {
            assert!(approx(v, rng.next_normal(), 1e-6));
        }
    }

    // -------------------------------------------------------------------------
    // Bookkeeping
    // -------------------------------------------------------------------------

    fn lin(out: usize, inp: usize) -> usize {
        out * inp + out
    }

    #[test]
    fn parameter_count_adds_up() {
        let cfg = tiny_config();
        let m = make_model(&cfg);
        let h = cfg.hidden_size;
        let mlp = cfg.mlp_hidden();
        let hd = cfg.head_dim();

        let mut expected = lin(h, cfg.patch_dim())
            + lin(h, cfg.context_dim)
            + lin(h, 256)
            + lin(h, h)
            + lin(h, cfg.pooled_dim)
            + lin(h, h)
            + lin(2 * h, h)
            + lin(cfg.patch_dim(), h);
        let one_stream = lin(6 * h, h) + lin(3 * h, h) + lin(h, h) + lin(mlp, h) + lin(h, mlp) + 2 * hd;
        expected += cfg.n_double_blocks * 2 * one_stream;
        expected += cfg.n_single_blocks * (lin(3 * h, h) + lin(3 * h + mlp, h) + lin(h, h + mlp) + 2 * hd);

        assert_eq!(m.parameter_count(), expected);
    }

    #[test]
    fn the_real_flux_config_comes_out_near_twelve_billion_parameters() {
        // A shape check on the config rather than on loaded weights: the block
        // counts and widths multiply out to the published size, so a typo in
        // any one of them shows up here.
        let c = FluxConfig::schnell();
        let h = c.hidden_size;
        let mlp = c.mlp_hidden();

        let mut n = lin(h, c.patch_dim())
            + lin(h, c.context_dim)
            + lin(h, 256)
            + lin(h, h)
            + lin(h, c.pooled_dim)
            + lin(h, h)
            + lin(2 * h, h)
            + lin(c.patch_dim(), h);
        n += c.n_double_blocks
            * 2
            * (lin(6 * h, h) + lin(3 * h, h) + lin(h, h) + lin(mlp, h) + lin(h, mlp) + 2 * c.head_dim());
        n += c.n_single_blocks * (lin(3 * h, h) + lin(3 * h + mlp, h) + lin(h, h + mlp) + 2 * c.head_dim());

        assert!(n > 11_000_000_000);
        assert!(n < 13_000_000_000);
    }
}
