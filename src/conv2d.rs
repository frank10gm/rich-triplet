#![allow(dead_code)]
// =============================================================================
// 2-D convolution primitives -- for image autoencoders
// =============================================================================
//
// `conv1d` exists because neural audio codecs are built out of convolutions.
// Image autoencoders are built out of the same operator one dimension up, so
// this is that file with a second spatial axis threaded through it. The
// three-path structure (pointwise fast path, direct loop, tiled im2col + gemm)
// is deliberately identical -- the reasoning that picked it there applies
// unchanged here, only more so, since the problems are larger.
//
// ## Activation layout
//
// Activations are `Mat` in **spatial-major** order -- `[H * W, channels]`, one
// row per pixel, row-major in the pixel index so that `p = y * W + x`. PyTorch
// stores images channel-major, `[channels, H, W]`. Spatial-major is the
// two-dimensional reading of the rule the rest of this project follows: a `Mat`
// row is always one token, one timestep, or -- here -- one pixel.
//
// The layout is not a matter of taste. It is what makes the rest cheap:
//
//   * A 1x1 convolution is literally `x @ weight^T`, so `Mat::matmul_bt` and
//     BLAS apply with no repacking. A VAE decoder is roughly half 1x1
//     convolutions once the residual shortcuts are counted.
//   * The mid-block's self-attention wants one row per spatial position, which
//     is what this already is -- the attention code written for transformers
//     applies to an image with no reshaping at all.
//   * A diffusion transformer's patchify step is a regrouping of rows, not a
//     transpose.
//
// It costs something in exactly one place: `group_norm` reduces over channels
// **and** space together, so its reduction is strided rather than contiguous.
// That is a handful of extra lines, paid once, against a repack on every
// convolution in the other layout.
//
// Because `Mat` is flat, `h` and `w` travel as explicit arguments. The
// invariant `x.rows == h * w` is asserted everywhere it is assumed.
//
// ## Weight layout
//
// As in `conv1d`, every weight is a `Mat` whose row-major bytes are
// **exactly** the PyTorch flat buffer, so loading is a wrap rather than a
// repack:
//
// | PyTorch parameter    | shape              | `Mat` passed here        |
// |----------------------|--------------------|--------------------------|
// | `Conv2d.weight`      | [Cout, Cin, KH, KW]| [Cout, Cin * KH * KW]    |
// | `Conv2d.weight` (1x1)| [Cout, Cin, 1, 1]  | [Cout, Cin]              |
//
// The innermost index is the kernel's x axis, then its y axis, then the input
// channel -- so column `ci * KH * KW + ky * KW + kx`. im2col gathers into that
// same order, which is why the gemm needs neither operand repacked.
//
// ## Output sizes
//
//   conv2d:  out = floor((in + 2*padding - dilation*(kernel - 1) - 1) / stride) + 1
//
// applied independently per axis. A VAE decoder holds every 3x3 convolution at
// `padding = 1, stride = 1`, which makes that collapse to `out == in`; the
// size is asserted to be preserved whenever those parameters line up, because
// a decoder that silently shrinks by two pixels per layer still produces an
// image and the mistake shows up only as a crop.

use crate::autograd2::Mat;

/// Add a per-output-channel bias to every row, if there is one.
fn add_bias_rows(out: &mut Mat, bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    debug_assert!(bias.len() == out.cols, "conv2d: bias length != output channels");
    let cols = out.cols;
    for r in 0..out.rows {
        let row = &mut out.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += bias[c];
        }
    }
}

// =============================================================================
// Output sizes
// =============================================================================

/// Output extent of one axis of a 2-D convolution. Returns 0 when the kernel
/// does not fit, which callers should treat as an error rather than an empty
/// result.
pub fn conv2d_out_size(input: usize, kernel: usize, dilation: usize, padding: usize, stride: usize) -> usize {
    debug_assert!(kernel >= 1, "conv2d_out_size: kernel must be >= 1");
    debug_assert!(dilation >= 1, "conv2d_out_size: dilation must be >= 1");
    debug_assert!(stride >= 1, "conv2d_out_size: stride must be >= 1");
    let reach = dilation * (kernel - 1);
    let padded = input + 2 * padding;
    if padded > reach { (padded - reach - 1) / stride + 1 } else { 0 }
}

// =============================================================================
// Convolutions
// =============================================================================

/// Kernel-size-1 convolution: `out[p] = weight @ x[p] + bias`.
///
/// `x` is [H*W, Cin], `weight` is [Cout, Cin], output is [H*W, Cout]. This is a
/// dense channel mix with no sliding window, so it goes straight to
/// `Mat::matmul_bt` and picks up BLAS.
///
/// `bias` may be empty, meaning no bias.
pub fn conv2d_pointwise(x: &Mat, weight: &Mat, bias: &[f32]) -> Mat {
    debug_assert!(x.cols == weight.cols, "conv2d_pointwise: x.cols != weight.cols (Cin mismatch)");
    let mut out = x.matmul_bt(weight);
    add_bias_rows(&mut out, bias);
    out
}

/// Dense (`groups == 1`) 2-D convolution with zero padding.
///
/// `x` is [h*w, Cin], `weight` is [Cout, Cin * kernel_h * kernel_w], output is
/// [out_h * out_w, Cout] where the extents come from `conv2d_out_size`.
///
/// Three paths, picked by shape, mirroring `conv1d_dense`:
///
///   * A 1x1 kernel with no dilation, padding or stride delegates to
///     `conv2d_pointwise`, which is a plain matmul.
///   * Small problems use a direct accumulation loop and no scratch buffer.
///   * Everything else goes through im2col: gather the `Cin * KH * KW`
///     receptive field of each output pixel into a row, then one `matmul_bt`
///     against the weight.
///
/// The tiling in the third path is load-bearing rather than an optimisation. A
/// VAE decoder's last level runs 128 channels of 3x3 over 1024x1024 pixels; the
/// full patch matrix would be `1048576 * 1152` floats, 4.8 GB. Tiles of a few
/// megabytes turn that into a sequence of cache-friendly gemms.
///
/// `bias` may be empty, meaning no bias.
#[allow(clippy::too_many_arguments)]
pub fn conv2d_dense(
    x: &Mat,
    h: usize,
    w: usize,
    weight: &Mat,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    bias: &[f32],
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Mat {
    debug_assert!(x.rows == h * w, "conv2d_dense: x.rows != h * w");
    debug_assert!(weight.rows == out_channels, "conv2d_dense: weight.rows != out_channels");
    debug_assert!(
        weight.cols == x.cols * kernel_h * kernel_w,
        "conv2d_dense: weight.cols != Cin * KH * KW"
    );

    if kernel_h == 1 && kernel_w == 1 && dilation == 1 && padding == 0 && stride == 1 {
        return conv2d_pointwise(x, weight, bias);
    }

    let c_in = x.cols;
    let out_h = conv2d_out_size(h, kernel_h, dilation, padding, stride);
    let out_w = conv2d_out_size(w, kernel_w, dilation, padding, stride);
    let n_out = out_h * out_w;

    // The "same" case a decoder lives in. Getting this wrong crops the image by
    // a couple of pixels per layer, which is invisible in isolation.
    debug_assert!(
        !(stride == 1 && dilation == 1 && kernel_h == 2 * padding + 1 && kernel_w == 2 * padding + 1)
            || (out_h == h && out_w == w),
        "conv2d_dense: odd kernel with matching padding must preserve size"
    );

    if n_out == 0 {
        return Mat::zeros(0, out_channels);
    }

    let patch = c_in * kernel_h * kernel_w;

    // im2col + gemm once the problem is big enough to pay for the scratch
    // buffer, on the same threshold as the 1-D version.
    const GEMM_THRESHOLD: usize = 1 << 16; // output elements x taps
    if n_out * out_channels * patch >= GEMM_THRESHOLD {
        let mut out = Mat::zeros(n_out, out_channels);

        // Tile over output pixels so the patch matrix stays a few megabytes
        // rather than `n_out * patch` floats -- gigabytes at decoder shapes.
        let tile = ((1usize << 21) / patch.max(1)).max(1);
        let mut cols = Mat::zeros(tile.min(n_out), patch);

        let mut base = 0;
        while base < n_out {
            let rows = tile.min(n_out - base);
            if cols.rows != rows {
                cols = Mat::zeros(rows, patch);
            } else {
                cols.data.fill(0.0);
            }
            for r in 0..rows {
                let p = base + r;
                let oy = p / out_w;
                let ox = p % out_w;
                let crow = &mut cols.data[r * patch..(r + 1) * patch];
                for ky in 0..kernel_h {
                    let shifted_y = oy * stride + ky * dilation;
                    if shifted_y < padding {
                        continue; // top zero pad
                    }
                    let sy = shifted_y - padding;
                    if sy >= h {
                        continue; // bottom zero pad
                    }
                    for kx in 0..kernel_w {
                        let shifted_x = ox * stride + kx * dilation;
                        if shifted_x < padding {
                            continue; // left zero pad
                        }
                        let sx = shifted_x - padding;
                        if sx >= w {
                            continue; // right zero pad
                        }
                        let xrow = &x.data[(sy * w + sx) * c_in..(sy * w + sx + 1) * c_in];
                        let tap = ky * kernel_w + kx;
                        // Column `ci * KH * KW + ky * KW + kx` mirrors the
                        // weight's layout, so the gemm repacks neither operand.
                        for ci in 0..c_in {
                            crow[ci * kernel_h * kernel_w + tap] = xrow[ci];
                        }
                    }
                }
            }
            let block = cols.matmul_bt(weight);
            out.data[base * out_channels..(base + rows) * out_channels]
                .copy_from_slice(&block.data[..rows * out_channels]);
            base += tile;
        }
        add_bias_rows(&mut out, bias);
        return out;
    }

    let mut out = Mat::zeros(n_out, out_channels);
    for oy in 0..out_h {
        for ox in 0..out_w {
            let o0 = (oy * out_w + ox) * out_channels;
            for ky in 0..kernel_h {
                let shifted_y = oy * stride + ky * dilation;
                if shifted_y < padding {
                    continue;
                }
                let sy = shifted_y - padding;
                if sy >= h {
                    continue;
                }
                for kx in 0..kernel_w {
                    let shifted_x = ox * stride + kx * dilation;
                    if shifted_x < padding {
                        continue;
                    }
                    let sx = shifted_x - padding;
                    if sx >= w {
                        continue;
                    }
                    let xrow = &x.data[(sy * w + sx) * c_in..(sy * w + sx + 1) * c_in];
                    let tap = ky * kernel_w + kx;
                    for co in 0..out_channels {
                        let wrow = &weight.data[co * weight.cols..(co + 1) * weight.cols];
                        let mut acc = 0.0f32;
                        for ci in 0..c_in {
                            acc += xrow[ci] * wrow[ci * kernel_h * kernel_w + tap];
                        }
                        out.data[o0 + co] += acc;
                    }
                }
            }
        }
    }

    add_bias_rows(&mut out, bias);
    out
}

// =============================================================================
// Resampling
// =============================================================================

/// Nearest-neighbour upsample by an integer factor in both axes.
///
/// `x` is [h*w, C], output is [(h*factor) * (w*factor), C].
///
/// This is the VAE decoder's only upsampling operator: it upsamples and *then*
/// convolves, rather than using a transposed convolution. Substituting a
/// transposed convolution with the same weights is a plausible-looking change
/// that produces checkerboard artefacts -- a regular high-frequency texture
/// that reads as heavy JPEG compression rather than as a bug.
///
/// The output is a pure repeat: each input pixel becomes a `factor x factor`
/// block. Interpolating instead softens every edge in the image by a fraction
/// of a pixel, which is invisible per layer and cumulative over four of them.
pub fn upsample_nearest2d(x: &Mat, h: usize, w: usize, factor: usize) -> Mat {
    debug_assert!(x.rows == h * w, "upsample_nearest2d: x.rows != h * w");
    debug_assert!(factor >= 1, "upsample_nearest2d: factor must be >= 1");
    if factor == 1 {
        return x.clone();
    }

    let out_h = h * factor;
    let out_w = w * factor;
    let channels = x.cols;
    let mut out = Mat::zeros(out_h * out_w, channels);

    for oy in 0..out_h {
        let sy = oy / factor;
        for ox in 0..out_w {
            let sx = ox / factor;
            let src = &x.data[(sy * w + sx) * channels..(sy * w + sx + 1) * channels];
            let d0 = (oy * out_w + ox) * channels;
            out.data[d0..d0 + channels].copy_from_slice(src);
        }
    }
    out
}

// =============================================================================
// Normalization and activations
// =============================================================================

/// In-place `group_norm`, for the long chains in a decoder where the
/// intermediate is dead immediately afterwards.
pub fn group_norm_inplace(x: &mut Mat, groups: usize, weight: &[f32], bias: &[f32], eps: f32) {
    let channels = x.cols;
    debug_assert!(groups >= 1, "group_norm: groups must be >= 1");
    debug_assert!(channels % groups == 0, "group_norm: channels not divisible by groups");
    debug_assert!(weight.is_empty() || weight.len() == channels, "group_norm: weight length != C");
    debug_assert!(bias.is_empty() || bias.len() == channels, "group_norm: bias length != C");
    if x.rows == 0 {
        return;
    }

    let per_group = channels / groups;
    let n = x.rows * per_group;
    let count = n as f64;

    for g in 0..groups {
        let lo = g * per_group;

        // Two passes rather than sum-of-squares. A decoder's last level reduces
        // over sixteen million values, and the one-pass form loses the variance
        // in the cancellation when the mean is far from zero.
        let mut sum = 0.0f64;
        for r in 0..x.rows {
            let row = &x.data[r * channels..(r + 1) * channels];
            for c in 0..per_group {
                sum += row[lo + c] as f64;
            }
        }
        let mean = sum / count;

        let mut sq = 0.0f64;
        for r in 0..x.rows {
            let row = &x.data[r * channels..(r + 1) * channels];
            for c in 0..per_group {
                let d = row[lo + c] as f64 - mean;
                sq += d * d;
            }
        }
        // Biased variance, as PyTorch uses for normalization layers.
        let inv_std = (1.0 / (sq / count + eps as f64).sqrt()) as f32;
        let mean_f = mean as f32;

        for r in 0..x.rows {
            let row = &mut x.data[r * channels..(r + 1) * channels];
            for c in 0..per_group {
                let ch = lo + c;
                let mut v = (row[ch] - mean_f) * inv_std;
                if !weight.is_empty() {
                    v *= weight[ch];
                }
                if !bias.is_empty() {
                    v += bias[ch];
                }
                row[ch] = v;
            }
        }
    }
}

/// GroupNorm over spatial-major activations.
///
/// `x` is [h*w, C]. The channels are split into `groups` contiguous blocks and
/// each block is normalized over its channels **and every spatial position**
/// jointly -- one mean and one variance per group, not per pixel.
///
/// That joint reduction is the whole content of the operator and the one thing
/// easy to get wrong. Normalizing each row independently is LayerNorm; it runs,
/// it is numerically well behaved, and it produces an image whose local
/// contrast is subtly flattened everywhere. There is no way to see it without a
/// reference, so the tests check it directly: a tensor that is constant within
/// each row but varies across rows must **not** normalize to zeros.
///
/// `weight` and `bias` are per channel and may be empty, meaning no affine.
///
/// `eps` is 1e-6 in the diffusers VAE. The far more common 1e-5 is close
/// enough to look right and far enough to shift contrast measurably once four
/// levels have compounded it.
pub fn group_norm(x: &Mat, groups: usize, weight: &[f32], bias: &[f32], eps: f32) -> Mat {
    let mut out = x.clone();
    group_norm_inplace(&mut out, groups, weight, bias, eps);
    out
}

/// `x * sigmoid(x)`, in place. The VAE decoder's activation throughout.
pub fn silu_inplace(x: &mut Mat) {
    for v in x.data.iter_mut() {
        *v = *v / (1.0 + (-*v).exp());
    }
}

/// `x * sigmoid(1.702 * x)`, in place.
///
/// CLIP's activation. It is a cheap approximation of GELU that predates the
/// tanh one, and CLIP-L was trained with it -- substituting either exact GELU
/// or the tanh approximation shifts the text embedding enough to change which
/// image a prompt produces.
pub fn quick_gelu_inplace(x: &mut Mat) {
    for v in x.data.iter_mut() {
        *v = *v / (1.0 + (-1.702f32 * *v).exp());
    }
}

/// `0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))`, in place.
///
/// The tanh approximation, which is what T5 v1.1 and FLUX both use.
pub fn gelu_tanh_inplace(x: &mut Mat) {
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    for v in x.data.iter_mut() {
        let c = *v * *v * *v;
        *v = 0.5 * *v * (1.0 + (SQRT_2_OVER_PI * (*v + 0.044715 * c)).tanh());
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute comparison with an explicit tolerance. A fixed 1e-3 would be
    /// far too loose for checking an activation against its own definition.
    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    /// Relative comparison, for the same reason the conv1d tests need one:
    /// im2col + gemm and the direct loop sum the same products in different
    /// orders.
    fn approx_rel(a: f32, b: f32, tol: f32) -> bool {
        let scale = 1.0f32.max(a.abs()).max(b.abs());
        (a - b).abs() / scale < tol
    }

    fn mats_close(a: &Mat, b: &Mat, tol: f32) -> bool {
        if a.rows != b.rows || a.cols != b.cols {
            return false;
        }
        a.data.iter().zip(&b.data).all(|(&x, &y)| approx_rel(x, y, tol))
    }

    /// Fill a matrix with a deterministic spread of small signed values.
    fn ramp(rows: usize, cols: usize, scale: f32, shift: f32) -> Mat {
        Mat::from_fn(rows, cols, |r, c| (r * cols + c) as f32 * scale - shift)
    }

    fn ramp_default(rows: usize, cols: usize) -> Mat {
        ramp(rows, cols, 0.01, 0.5)
    }

    /// A spread that is not monotone, so a transposed spatial index cannot pass
    /// by accident on a smoothly varying input.
    fn wobble(rows: usize, cols: usize) -> Mat {
        Mat::from_fn(rows, cols, |r, c| {
            let i = (r * 31 + c * 17) as f32;
            (i * 0.37).sin() * 0.8 + (i * 0.11).cos() * 0.3
        })
    }

    /// Reference dense 2-D convolution, written out index by index.
    #[allow(clippy::too_many_arguments)]
    fn ref_conv2d(
        x: &Mat,
        h: usize,
        w: usize,
        weight: &Mat,
        c_out: usize,
        kh: usize,
        kw: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Mat {
        let c_in = x.cols;
        let out_h = conv2d_out_size(h, kh, dilation, padding, stride);
        let out_w = conv2d_out_size(w, kw, dilation, padding, stride);
        let mut out = Mat::zeros(out_h * out_w, c_out);
        for oy in 0..out_h {
            for ox in 0..out_w {
                for co in 0..c_out {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let sy = (oy * stride + ky * dilation) as i64 - padding as i64;
                                let sx = (ox * stride + kx * dilation) as i64 - padding as i64;
                                if sy < 0 || sy >= h as i64 || sx < 0 || sx >= w as i64 {
                                    continue;
                                }
                                let p = sy as usize * w + sx as usize;
                                acc += x.at(p, ci) * weight.at(co, ci * kh * kw + ky * kw + kx);
                            }
                        }
                    }
                    *out.at_mut(oy * out_w + ox, co) = acc;
                }
            }
        }
        out
    }

    /// Reference GroupNorm, reducing over channels and space in two explicit
    /// passes.
    fn ref_group_norm(x: &Mat, groups: usize, weight: &[f32], bias: &[f32], eps: f32) -> Mat {
        let c = x.cols;
        let per = c / groups;
        let mut out = x.clone();
        for g in 0..groups {
            let mut sum = 0.0f64;
            let mut n = 0usize;
            for r in 0..x.rows {
                for k in 0..per {
                    sum += x.at(r, g * per + k) as f64;
                    n += 1;
                }
            }
            let mean = sum / n as f64;
            let mut var = 0.0f64;
            for r in 0..x.rows {
                for k in 0..per {
                    let d = x.at(r, g * per + k) as f64 - mean;
                    var += d * d;
                }
            }
            var /= n as f64;
            for r in 0..x.rows {
                for k in 0..per {
                    let ch = g * per + k;
                    let mut v = (x.at(r, ch) as f64 - mean) / (var + eps as f64).sqrt();
                    if !weight.is_empty() {
                        v *= weight[ch] as f64;
                    }
                    if !bias.is_empty() {
                        v += bias[ch] as f64;
                    }
                    *out.at_mut(r, ch) = v as f32;
                }
            }
        }
        out
    }

    // -------------------------------------------------------------------------
    // Output sizes
    // -------------------------------------------------------------------------

    #[test]
    fn conv2d_out_size_matches_pytorch_formula() {
        // "same" padding for odd kernels
        assert_eq!(conv2d_out_size(32, 3, 1, 1, 1), 32);
        assert_eq!(conv2d_out_size(32, 5, 1, 2, 1), 32);
        assert_eq!(conv2d_out_size(32, 1, 1, 0, 1), 32);
        // valid padding shrinks by the reach
        assert_eq!(conv2d_out_size(32, 3, 1, 0, 1), 30);
        // stride halves, rounding up
        assert_eq!(conv2d_out_size(32, 3, 1, 1, 2), 16);
        assert_eq!(conv2d_out_size(33, 3, 1, 1, 2), 17);
        // dilation extends the reach
        assert_eq!(conv2d_out_size(32, 3, 2, 0, 1), 28);
        // a kernel that does not fit reports zero rather than wrapping
        assert_eq!(conv2d_out_size(2, 5, 1, 0, 1), 0);
    }

    // -------------------------------------------------------------------------
    // Pointwise
    // -------------------------------------------------------------------------

    #[test]
    fn pointwise_is_exactly_a_matmul_against_the_transposed_weight() {
        let x = ramp_default(6 * 5, 8);
        let w = ramp(4, 8, 0.03, 0.2);
        let expected = x.matmul_bt(&w);
        let got = conv2d_pointwise(&x, &w, &[]);
        assert!(mats_close(&got, &expected, 1e-6));
    }

    #[test]
    fn one_by_one_dense_takes_the_pointwise_path_and_agrees() {
        let x = wobble(7 * 9, 6);
        let w = ramp(5, 6, 0.05, 0.3);
        let bias = [0.1f32, -0.2, 0.3, -0.4, 0.5];
        let got = conv2d_dense(&x, 7, 9, &w, 5, 1, 1, &bias, 1, 0, 1);
        let expected = conv2d_pointwise(&x, &w, &bias);
        assert_eq!(got.rows, 63);
        assert_eq!(got.cols, 5);
        assert!(mats_close(&got, &expected, 1e-6));
    }

    // -------------------------------------------------------------------------
    // Dense convolution, against the index-by-index reference
    // -------------------------------------------------------------------------

    #[test]
    fn three_by_three_padding_one_preserves_the_spatial_extent() {
        let (h, w) = (9, 7);
        let x = wobble(h * w, 4);
        let weight = ramp(6, 4 * 3 * 3, 0.004, 0.07);
        let got = conv2d_dense(&x, h, w, &weight, 6, 3, 3, &[], 1, 1, 1);
        assert_eq!(got.rows, h * w);
        assert_eq!(got.cols, 6);
        assert!(mats_close(&got, &ref_conv2d(&x, h, w, &weight, 6, 3, 3, 1, 1, 1), 1e-5));
    }

    #[test]
    fn both_dense_paths_agree_with_the_reference() {
        // Small enough for the direct loop, then the same convolution at a size
        // that crosses the im2col threshold. Both must match the reference.
        for &(h, w, c_in, c_out) in &[(5usize, 4usize, 3usize, 2usize), (28, 24, 16, 24)] {
            let x = wobble(h * w, c_in);
            let weight = ramp(c_out, c_in * 3 * 3, 0.002, 0.05);
            let got = conv2d_dense(&x, h, w, &weight, c_out, 3, 3, &[], 1, 1, 1);
            let expected = ref_conv2d(&x, h, w, &weight, c_out, 3, 3, 1, 1, 1);
            assert!(mats_close(&got, &expected, 1e-5));
        }
    }

    #[test]
    fn asymmetric_height_and_width_do_not_transpose_the_spatial_index() {
        // h != w and a non-symmetric input: any swap of the two axes changes the
        // answer, which a square test would silently accept.
        let (h, w) = (11, 4);
        let x = wobble(h * w, 3);
        let weight = ramp(2, 3 * 3 * 3, 0.01, 0.15);
        let got = conv2d_dense(&x, h, w, &weight, 2, 3, 3, &[], 1, 1, 1);
        let expected = ref_conv2d(&x, h, w, &weight, 2, 3, 3, 1, 1, 1);
        assert_eq!(got.rows, h * w);
        assert!(mats_close(&got, &expected, 1e-5));

        // The transposed problem is genuinely different.
        let swapped = conv2d_dense(&x, w, h, &weight, 2, 3, 3, &[], 1, 1, 1);
        assert!(!mats_close(&got, &swapped, 1e-5));
    }

    #[test]
    fn non_square_kernel_keeps_its_axes_straight() {
        let (h, w) = (8, 9);
        let x = wobble(h * w, 3);
        let weight = ramp(4, 3 * 1 * 5, 0.01, 0.2);
        let got = conv2d_dense(&x, h, w, &weight, 4, 1, 5, &[], 1, 0, 1);
        assert_eq!(got.rows, conv2d_out_size(h, 1, 1, 0, 1) * conv2d_out_size(w, 5, 1, 0, 1));
        assert!(mats_close(&got, &ref_conv2d(&x, h, w, &weight, 4, 1, 5, 1, 0, 1), 1e-5));
    }

    #[test]
    fn stride_two_halves_both_axes() {
        let (h, w) = (16, 12);
        let x = wobble(h * w, 8);
        let weight = ramp(8, 8 * 3 * 3, 0.003, 0.05);
        let got = conv2d_dense(&x, h, w, &weight, 8, 3, 3, &[], 2, 1, 1);
        assert_eq!(got.rows, 8 * 6);
        assert!(mats_close(&got, &ref_conv2d(&x, h, w, &weight, 8, 3, 3, 2, 1, 1), 1e-5));
    }

    #[test]
    fn dilation_spreads_the_taps_without_changing_the_output_layout() {
        let (h, w) = (12, 10);
        let x = wobble(h * w, 4);
        let weight = ramp(4, 4 * 3 * 3, 0.006, 0.1);
        let got = conv2d_dense(&x, h, w, &weight, 4, 3, 3, &[], 1, 2, 2);
        assert_eq!(got.rows, h * w); // dilation 2, padding 2, kernel 3 is also "same"
        assert!(mats_close(&got, &ref_conv2d(&x, h, w, &weight, 4, 3, 3, 1, 2, 2), 1e-5));
    }

    #[test]
    fn padding_wider_than_the_input_still_only_reads_real_pixels() {
        let (h, w) = (2, 3);
        let x = wobble(h * w, 2);
        let weight = ramp(3, 2 * 5 * 5, 0.02, 0.25);
        let got = conv2d_dense(&x, h, w, &weight, 3, 5, 5, &[], 1, 4, 1);
        assert_eq!(got.rows, conv2d_out_size(h, 5, 1, 4, 1) * conv2d_out_size(w, 5, 1, 4, 1));
        assert!(mats_close(&got, &ref_conv2d(&x, h, w, &weight, 3, 5, 5, 1, 4, 1), 1e-5));
    }

    #[test]
    fn bias_is_added_once_per_output_channel() {
        let (h, w) = (6, 6);
        let x = wobble(h * w, 3);
        let weight = ramp(4, 3 * 3 * 3, 0.01, 0.1);
        let bias = [1.0f32, -2.0, 0.5, 3.0];
        let plain = conv2d_dense(&x, h, w, &weight, 4, 3, 3, &[], 1, 1, 1);
        let biased = conv2d_dense(&x, h, w, &weight, 4, 3, 3, &bias, 1, 1, 1);
        for r in 0..plain.rows {
            for c in 0..plain.cols {
                assert!(approx(biased.at(r, c), plain.at(r, c) + bias[c], 1e-5));
            }
        }
    }

    #[test]
    fn a_delta_kernel_reproduces_the_input_shifted() {
        // One input channel, one output channel, a 3x3 kernel that is zero
        // except for the top-left tap. The result is the input translated by
        // one pixel in each axis -- which pins the sign of the padding offset.
        let (h, w) = (5, 4);
        let x = wobble(h * w, 1);
        let mut weight = Mat::zeros(1, 9);
        *weight.at_mut(0, 0) = 1.0; // ky = 0, kx = 0
        let got = conv2d_dense(&x, h, w, &weight, 1, 3, 3, &[], 1, 1, 1);
        assert_eq!(got.rows, h * w);
        for oy in 0..h {
            for ox in 0..w {
                // out[oy][ox] reads x[oy - 1][ox - 1]
                let expected = if oy == 0 || ox == 0 { 0.0 } else { x.at((oy - 1) * w + (ox - 1), 0) };
                assert!(approx(got.at(oy * w + ox, 0), expected, 1e-6));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Upsampling
    // -------------------------------------------------------------------------

    #[test]
    fn upsample_nearest2d_repeats_each_pixel_into_a_block() {
        let (h, w) = (3, 4);
        let x = wobble(h * w, 5);
        let up = upsample_nearest2d(&x, h, w, 2);
        assert_eq!(up.rows, (h * 2) * (w * 2));
        assert_eq!(up.cols, 5);
        for oy in 0..h * 2 {
            for ox in 0..w * 2 {
                for c in 0..5 {
                    assert!(approx(up.at(oy * (w * 2) + ox, c), x.at((oy / 2) * w + (ox / 2), c), 1e-7));
                }
            }
        }
    }

    #[test]
    fn upsample_nearest2d_by_one_is_the_identity() {
        let x = wobble(4 * 4, 3);
        assert!(mats_close(&upsample_nearest2d(&x, 4, 4, 1), &x, 1e-7));
    }

    // -------------------------------------------------------------------------
    // GroupNorm
    // -------------------------------------------------------------------------

    #[test]
    fn group_norm_agrees_with_a_two_pass_reference() {
        let x = wobble(8 * 6, 16);
        let weight: Vec<f32> = (0..16).map(|i| 0.5 + 0.1 * i as f32).collect();
        let bias: Vec<f32> = (0..16).map(|i| -0.2 + 0.05 * i as f32).collect();
        let got = group_norm(&x, 4, &weight, &bias, 1e-6);
        let expected = ref_group_norm(&x, 4, &weight, &bias, 1e-6);
        assert!(mats_close(&got, &expected, 1e-5));
    }

    #[test]
    fn group_norm_statistics_span_space_not_just_channels() {
        // Every row is constant across its channels but the rows differ.
        // LayerNorm maps this to all zeros. GroupNorm must not: the variance it
        // sees is the variance *between* rows, which is nonzero.
        let mut x = Mat::zeros(6, 4);
        for r in 0..x.rows {
            for c in 0..x.cols {
                *x.at_mut(r, c) = r as f32 - 2.5;
            }
        }
        let got = group_norm(&x, 1, &[], &[], 1e-6);
        let all_zero = got.data.iter().all(|v| v.abs() <= 1e-3);
        assert!(!all_zero);
        assert!(mats_close(&got, &ref_group_norm(&x, 1, &[], &[], 1e-6), 1e-5));
    }

    #[test]
    fn group_norm_with_one_group_normalizes_everything_jointly() {
        let x = wobble(5 * 5, 8);
        let got = group_norm(&x, 1, &[], &[], 1e-6);
        let mut sum = 0.0f64;
        let mut sq = 0.0f64;
        for &v in &got.data {
            sum += v as f64;
            sq += v as f64 * v as f64;
        }
        let n = got.data.len() as f64;
        assert!((sum / n).abs() < 1e-4);
        assert!((sq / n - 1.0).abs() < 1e-3);
    }

    #[test]
    fn group_norm_with_one_group_per_channel_normalizes_each_channel_over_space() {
        let channels = 6;
        let x = wobble(7 * 3, channels);
        let got = group_norm(&x, channels, &[], &[], 1e-6);
        for c in 0..channels {
            let mut sum = 0.0f64;
            let mut sq = 0.0f64;
            for r in 0..got.rows {
                sum += got.at(r, c) as f64;
                sq += got.at(r, c) as f64 * got.at(r, c) as f64;
            }
            let n = got.rows as f64;
            assert!((sum / n).abs() < 1e-4);
            assert!((sq / n - 1.0).abs() < 1e-3);
        }
    }

    #[test]
    fn group_norm_groups_do_not_leak_into_each_other() {
        // Scaling one group must leave the other's output untouched.
        let mut x = wobble(4 * 4, 8);
        let base = group_norm(&x, 2, &[], &[], 1e-6);
        for r in 0..x.rows {
            for c in 4..8 {
                *x.at_mut(r, c) *= 10.0;
            }
        }
        let perturbed = group_norm(&x, 2, &[], &[], 1e-6);
        for r in 0..x.rows {
            for c in 0..4 {
                assert!(approx(base.at(r, c), perturbed.at(r, c), 1e-5));
            }
        }
    }

    #[test]
    fn group_norm_affine_parameters_are_per_channel() {
        let x = wobble(4 * 4, 4);
        let weight = [2.0f32; 4];
        let bias = [1.0f32; 4];
        let plain = group_norm(&x, 2, &[], &[], 1e-6);
        let affine = group_norm(&x, 2, &weight, &bias, 1e-6);
        for i in 0..plain.data.len() {
            assert!(approx(affine.data[i], plain.data[i] * 2.0 + 1.0, 1e-5));
        }
    }

    #[test]
    fn group_norm_eps_is_not_negligible_at_the_default() {
        // 1e-6 against 1e-5 is a visible difference on a low-variance group,
        // which is why the default is spelled out rather than inherited.
        let mut x = Mat::zeros(4, 4);
        for r in 0..4 {
            for c in 0..4 {
                *x.at_mut(r, c) = 1e-3 * (r * 4 + c) as f32;
            }
        }
        let a = group_norm(&x, 1, &[], &[], 1e-6);
        let b = group_norm(&x, 1, &[], &[], 1e-5);
        assert!(!mats_close(&a, &b, 1e-4));
    }

    // -------------------------------------------------------------------------
    // Activations
    // -------------------------------------------------------------------------

    #[test]
    fn silu_quick_gelu_and_gelu_tanh_match_their_definitions() {
        let xs = vec![-3.0f32, -1.0, -0.25, 0.0, 0.25, 1.0, 3.0];
        let m = Mat::new(xs.clone(), 1, xs.len());

        let mut silu = m.clone();
        silu_inplace(&mut silu);
        let mut qg = m.clone();
        quick_gelu_inplace(&mut qg);
        let mut gt = m.clone();
        gelu_tanh_inplace(&mut gt);

        for (i, &x) in xs.iter().enumerate() {
            assert!(approx(silu.data[i], x / (1.0 + (-x).exp()), 1e-6));
            assert!(approx(qg.data[i], x / (1.0 + (-1.702f32 * x).exp()), 1e-6));
            let inner = 0.7978845608028654f32 * (x + 0.044715 * x * x * x);
            assert!(approx(gt.data[i], 0.5 * x * (1.0 + inner.tanh()), 1e-6));
        }
    }

    #[test]
    fn quick_gelu_is_not_interchangeable_with_gelu_tanh() {
        // They differ by up to a few percent in the middle of the range, which
        // is enough to move a CLIP text embedding.
        let mut a = Mat::new(vec![1.0], 1, 1);
        let mut b = a.clone();
        quick_gelu_inplace(&mut a);
        gelu_tanh_inplace(&mut b);
        assert!((a.data[0] - b.data[0]).abs() > 1e-3);
    }

    #[test]
    fn gelu_tanh_stays_finite_where_the_cubic_term_explodes() {
        // `x^3` at x = 27 puts the tanh argument near 724. The CPU's `tanh`
        // saturates there correctly; the GPU kernel has to clamp to get the
        // same answer, and this pins what that answer is.
        let vals = [-40.0f32, -27.0, 0.0, 27.0, 40.0];
        let mut m = Mat::from_fn(1, 5, |_, c| vals[c]);
        let before = m.clone();
        gelu_tanh_inplace(&mut m);
        for &v in &m.data {
            assert!(v.is_finite());
        }
        // Far from zero GELU is the identity on the positive side and zero on
        // the negative one, to well within a float.
        assert!(approx(m.at(0, 0), 0.0, 1e-6));
        assert!(approx(m.at(0, 1), 0.0, 1e-6));
        assert!(approx(m.at(0, 2), 0.0, 1e-6));
        assert!(approx(m.at(0, 3), before.at(0, 3), 1e-4));
        assert!(approx(m.at(0, 4), before.at(0, 4), 1e-4));
    }
}
