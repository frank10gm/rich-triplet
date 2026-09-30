// =============================================================================
// 1-D convolution primitives -- for neural audio codec decoders
// =============================================================================
//
// The transformer stack needs no convolution beyond Qwen 3.5's causal depthwise
// conv1d, which is specialised into `Qwen35DeltaNet`. Neural audio codecs
// (SNAC, DAC, Encodec) are built almost entirely out of convolutions, so they
// need the general forms: dilated, grouped, and transposed.
//
// ## Activation layout
//
// Activations are `Mat` in **time-major** order -- `[T, channels]`, one row per
// frame. PyTorch stores audio activations channel-major, `[channels, T]`. Time-
// major is what the rest of this project uses (a `Mat` row is always one token
// or one timestep), and it makes a kernel-size-1 convolution literally a
// matmul against the weight, so `Mat::matmul_bt` and BLAS apply unchanged.
//
// ## Weight layout
//
// Every weight is a `Mat` whose row-major bytes are **exactly** the PyTorch
// flat buffer, so loading is a wrap rather than a repack:
//
// | PyTorch parameter          | shape             | `Mat` passed here     |
// |----------------------------|-------------------|-----------------------|
// | `Conv1d.weight`            | [Cout, Cin, K]    | [Cout, Cin * K]       |
// | `Conv1d.weight` (k=1)      | [Cout, Cin, 1]    | [Cout, Cin]           |
// | `Conv1d.weight` (depthwise)| [C, 1, K]         | [C, K]                |
// | `ConvTranspose1d.weight`   | **[Cin, Cout, K]**| [Cin, Cout * K]       |
//
// Note the transposed convolution stores **input** channels first. That is not
// a typo carried over from `Conv1d`: it is PyTorch's actual layout, and it also
// decides which axis `weight_norm` normalizes over -- see
// `weight_norm_combine`.
//
// ## Output lengths
//
//   conv1d:           T_out = T + 2*padding - dilation * (K - 1)
//   conv_transpose1d: T_out = (T - 1) * stride - 2*padding + K + output_padding
//
// A codec decoder block picks `K = 2*stride`, `padding = ceil(stride/2)` and
// `output_padding = stride % 2`, which makes the second formula collapse to
// exactly `T * stride` for every stride, even or odd:
//
//   even s:  (T-1)s - 2(s/2)     + 2s + 0 = Ts - s - s     + 2s     = Ts
//   odd  s:  (T-1)s - 2((s+1)/2) + 2s + 1 = Ts - s - s - 1 + 2s + 1 = Ts
//
// That exact-multiple property is the cheapest end-to-end check a codec
// decoder has, so `conv_transpose1d` asserts it whenever those three
// parameters line up. Debugging a decoder by ear is miserable; a length
// assertion fires immediately.

#![allow(dead_code)]

use crate::autograd2::Mat;

/// Add a per-output-channel bias to every row, if there is one.
fn add_bias_rows(out: &mut Mat, bias: &[f32]) {
    if bias.is_empty() {
        return;
    }
    assert!(bias.len() == out.cols, "conv1d: bias length != output channels");
    let cols = out.cols;
    for row in out.data.chunks_exact_mut(cols.max(1)) {
        for c in 0..cols {
            row[c] += bias[c];
        }
    }
}

/// Row `r` of `m` as a slice.
#[inline]
fn row(m: &Mat, r: usize) -> &[f32] {
    &m.data[r * m.cols..(r + 1) * m.cols]
}

/// Row `r` of `m` as a mutable slice.
#[inline]
fn row_mut(m: &mut Mat, r: usize) -> &mut [f32] {
    let cols = m.cols;
    &mut m.data[r * cols..(r + 1) * cols]
}

// =============================================================================
// Output lengths
// =============================================================================

/// Output length of a 1-D convolution. Returns 0 when the kernel does not fit,
/// which callers should treat as an error rather than an empty result.
///
/// Pass `stride = 1` for an unstrided convolution.
pub fn conv1d_out_len(
    t_in: usize,
    kernel: usize,
    dilation: usize,
    padding: usize,
    stride: usize,
) -> usize {
    assert!(kernel >= 1, "conv1d_out_len: kernel must be >= 1");
    assert!(dilation >= 1, "conv1d_out_len: dilation must be >= 1");
    assert!(stride >= 1, "conv1d_out_len: stride must be >= 1");
    // PyTorch's formula: floor((T + 2p - d*(k-1) - 1) / s) + 1.
    let reach = dilation * (kernel - 1);
    let padded = t_in + 2 * padding;
    if padded > reach {
        (padded - reach - 1) / stride + 1
    } else {
        0
    }
}

/// Output length of a 1-D transposed convolution.
pub fn conv_transpose1d_out_len(
    t_in: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> usize {
    assert!(kernel >= 1, "conv_transpose1d_out_len: kernel must be >= 1");
    assert!(stride >= 1, "conv_transpose1d_out_len: stride must be >= 1");
    if t_in == 0 {
        return 0;
    }
    let grown = (t_in - 1) * stride + kernel + output_padding;
    let trim = 2 * padding;
    if grown > trim { grown - trim } else { 0 }
}

// =============================================================================
// Convolutions
// =============================================================================

/// Kernel-size-1 convolution: `out[t] = weight @ x[t] + bias`.
///
/// This is a dense matmul, not a sliding window -- `x [T, Cin]` against
/// `weight [Cout, Cin]` is exactly `x @ weight^T`, so it goes straight to
/// `Mat::matmul_bt` and picks up BLAS. Codec decoders use k=1 convolutions
/// for every channel mix (the residual units' second conv, the noise block,
/// the quantizer's `out_proj`), so this is the hot one.
///
/// `bias` may be empty, meaning no bias.
pub fn conv1d_pointwise(x: &Mat, weight: &Mat, bias: &[f32]) -> Mat {
    assert!(
        x.cols == weight.cols,
        "conv1d_pointwise: x.cols != weight.cols (Cin mismatch)"
    );
    // x [T, Cin] @ weight^T [Cin, Cout] -- a kernel-size-1 convolution is a
    // dense channel mix with no sliding window at all.
    let mut out = x.matmul_bt(weight);
    add_bias_rows(&mut out, bias);
    out
}

/// Depthwise convolution -- `groups == channels`, so each channel is filtered
/// independently and no channel mixing happens.
///
/// `x` is [T, C], `weight` is [C, K], output is [T_out, C]. Zero padding.
///
/// `bias` may be empty, meaning no bias.
pub fn conv1d_depthwise(
    x: &Mat,
    weight: &Mat,
    bias: &[f32],
    dilation: usize,
    padding: usize,
) -> Mat {
    let channels = x.cols;
    let kernel = weight.cols;
    assert!(
        weight.rows == channels,
        "conv1d_depthwise: weight.rows != x.cols (channel mismatch)"
    );

    let t_out = conv1d_out_len(x.rows, kernel, dilation, padding, 1);
    let mut out = Mat::zeros(t_out, channels);

    // Tap-major loop order. `weight` is [C, K], so a single tap's coefficients
    // are strided by K; hoisting them into a contiguous scratch column once per
    // tap leaves the inner loop reading three contiguous arrays, which
    // vectorizes. The accumulation is order-independent, so this is free.
    let mut tap = vec![0.0f32; channels];
    for k in 0..kernel {
        for c in 0..channels {
            tap[c] = weight.at(c, k);
        }
        for t in 0..t_out {
            let shifted = t + k * dilation;
            if shifted < padding {
                continue; // left zero pad
            }
            let src = shifted - padding;
            if src >= x.rows {
                continue; // right zero pad
            }
            let xrow = row(x, src);
            let orow = row_mut(&mut out, t);
            for c in 0..channels {
                orow[c] += xrow[c] * tap[c];
            }
        }
    }

    add_bias_rows(&mut out, bias);
    out
}

/// Dense (`groups == 1`) convolution.
///
/// `x` is [T, Cin], `weight` is [Cout, Cin * K], output is [T_out, Cout].
///
/// Three paths, picked by shape:
///
///   * `kernel == 1` with no dilation or padding delegates to
///     `conv1d_pointwise`, which is a plain matmul.
///   * Small problems use a direct accumulation loop, which avoids the scratch
///     buffer entirely.
///   * Everything else goes through im2col: gather the `Cin * K` receptive
///     field of each output position into a row, then one `matmul_bt` against
///     the weight. That turns the convolution into a gemm and hands it to
///     BLAS.
///
/// The third path is not an optimisation so much as a requirement. A SNAC
/// decoder only needs a dense wide kernel for its final `[64 -> 1, k=7]`
/// projection, but OmniVoice's residual units are dense 7-taps at up to 512
/// channels running over a sequence that has already been upsampled toward
/// 96 000 samples -- tens of GFLOP per clip. The direct loop would take tens of
/// seconds where a gemm takes well under one.
///
/// im2col is materialised in row tiles rather than all at once, since the full
/// matrix would be `T_out * Cin * K` floats -- hundreds of megabytes at those
/// shapes.
///
/// `stride` skips output positions -- an encoder downsamples with it, and it is
/// free, since a strided convolution simply evaluates fewer of the same dot
/// products. Pass `stride = 1` for an unstrided convolution.
///
/// `bias` may be empty, meaning no bias.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_dense(
    x: &Mat,
    weight: &Mat,
    out_channels: usize,
    kernel: usize,
    bias: &[f32],
    dilation: usize,
    padding: usize,
    stride: usize,
) -> Mat {
    assert!(
        weight.rows == out_channels,
        "conv1d_dense: weight.rows != out_channels"
    );
    assert!(
        weight.cols == x.cols * kernel,
        "conv1d_dense: weight.cols != Cin * kernel"
    );

    if kernel == 1 && dilation == 1 && padding == 0 && stride == 1 {
        return conv1d_pointwise(x, weight, bias);
    }

    let c_in = x.cols;
    let t_out = conv1d_out_len(x.rows, kernel, dilation, padding, stride);

    // im2col + gemm once the problem is big enough to pay for the scratch
    // buffer. The threshold is deliberately low: below it the direct loop is
    // within noise, above it the gemm wins by orders of magnitude.
    const GEMM_THRESHOLD: usize = 1 << 16; // output elements x taps
    if t_out * out_channels * c_in * kernel >= GEMM_THRESHOLD {
        let mut out = Mat::zeros(t_out, out_channels);
        // Tile over output positions so the patch matrix stays cache-sized
        // rather than `t_out * c_in * kernel` floats.
        let patch = c_in * kernel;
        let tile = ((1usize << 21) / patch.max(1)).max(1);
        let mut cols = Mat::zeros(tile.min(t_out), patch);

        let mut base = 0;
        while base < t_out {
            let rows = tile.min(t_out - base);
            if cols.rows != rows {
                cols = Mat::zeros(rows, patch);
            } else {
                cols.data.fill(0.0);
            }
            for r in 0..rows {
                let crow = row_mut(&mut cols, r);
                for k in 0..kernel {
                    let shifted = (base + r) * stride + k * dilation;
                    if shifted < padding {
                        continue;
                    }
                    let src = shifted - padding;
                    if src >= x.rows {
                        continue;
                    }
                    let xrow = row(x, src);
                    // Column `ci * kernel + k` mirrors the weight's layout, so
                    // the gemm needs no repacking of either operand.
                    for ci in 0..c_in {
                        crow[ci * kernel + k] = xrow[ci];
                    }
                }
            }
            let block = cols.matmul_bt(weight);
            for r in 0..rows {
                row_mut(&mut out, base + r).copy_from_slice(row(&block, r));
            }
            base += tile;
        }
        add_bias_rows(&mut out, bias);
        return out;
    }

    let mut out = Mat::zeros(t_out, out_channels);
    for t in 0..t_out {
        for k in 0..kernel {
            let shifted = t * stride + k * dilation;
            if shifted < padding {
                continue;
            }
            let src = shifted - padding;
            if src >= x.rows {
                continue;
            }
            let xrow = row(x, src);
            for co in 0..out_channels {
                let wrow = row(weight, co);
                let mut acc = 0.0f32;
                for ci in 0..c_in {
                    acc += xrow[ci] * wrow[ci * kernel + k];
                }
                out.data[t * out_channels + co] += acc;
            }
        }
    }

    add_bias_rows(&mut out, bias);
    out
}

/// Grouped convolution -- the general case between dense and depthwise.
///
/// `x` is [T, Cin], `weight` is [Cout, (Cin / groups) * K], output is
/// [T_out, Cout]. Group `g` maps its slice of the input channels to its slice
/// of the output ones and to nothing else, so `groups == 1` is
/// `conv1d_dense` and `groups == Cin == Cout` is `conv1d_depthwise`.
///
/// This exists for exactly one layer: HuBERT's positional convolution is
/// `Conv1d(768, 768, kernel=128, groups=16)`, which is neither of the two
/// special cases. It runs each group through `conv1d_dense`, which is the
/// right trade at 16 groups and the wrong one at 768 -- hence keeping the
/// depthwise routine.
///
/// `bias` may be empty, meaning no bias.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_grouped(
    x: &Mat,
    weight: &Mat,
    out_channels: usize,
    kernel: usize,
    bias: &[f32],
    groups: usize,
    dilation: usize,
    padding: usize,
    stride: usize,
) -> Mat {
    assert!(groups >= 1, "conv1d_grouped: groups must be >= 1");
    assert!(x.cols % groups == 0, "conv1d_grouped: groups must divide Cin");
    assert!(
        out_channels % groups == 0,
        "conv1d_grouped: groups must divide Cout"
    );
    assert!(
        weight.rows == out_channels,
        "conv1d_grouped: weight.rows != out_channels"
    );
    assert!(
        weight.cols == (x.cols / groups) * kernel,
        "conv1d_grouped: weight.cols != (Cin / groups) * kernel"
    );

    if groups == 1 {
        return conv1d_dense(x, weight, out_channels, kernel, bias, dilation, padding, stride);
    }

    let in_per = x.cols / groups;
    let out_per = out_channels / groups;
    let t_out = conv1d_out_len(x.rows, kernel, dilation, padding, stride);
    let mut out = Mat::zeros(t_out, out_channels);

    // Each group is an independent dense convolution over its own channel
    // slice, so gather the slice, convolve, and scatter the result back. The
    // gathers are what a native grouped kernel would avoid, and at 16 groups
    // they are a rounding error against the gemm.
    for g in 0..groups {
        let xg = Mat::from_fn(x.rows, in_per, |r, c| x.at(r, g * in_per + c));
        let wg = Mat::from_fn(out_per, weight.cols, |r, c| weight.at(g * out_per + r, c));
        let og = conv1d_dense(&xg, &wg, out_per, kernel, &[], dilation, padding, stride);
        for r in 0..t_out {
            for c in 0..out_per {
                *out.at_mut(r, g * out_per + c) = og.at(r, c);
            }
        }
    }

    add_bias_rows(&mut out, bias);
    out
}

/// Transposed (fractionally-strided) convolution -- the upsampling operator.
///
/// `x` is [T, Cin], `weight` is [Cin, Cout * K] (PyTorch's input-channels-first
/// layout, flat), output is [T_out, Cout].
///
/// Implemented in **scatter** form:
///
///   out[t * stride - padding + k, co] += sum_ci x[t, ci] * w[ci, co, k]
///
/// which needs no kernel flip. The gather form -- reading the output position
/// and pulling from the input -- computes the same thing only with the kernel
/// reversed, and getting that backwards yields a time-reversed impulse
/// response: audio that sounds smeared rather than obviously broken. Scatter
/// keeps the indexing honest.
///
/// The `sum_ci` is hoisted into one gemm: `x [T, Cin] @ weight [Cin, Cout*K]`
/// gives every `(t, co, k)` product at once, and the scatter-add then places
/// column `co * K + k` of row `t` at output position `t * stride - padding + k`.
/// The weight needs no repacking because [Cin, Cout*K] row-major already *is*
/// the [Cin, Cout, K] flat buffer.
///
/// `bias` may be empty, meaning no bias. It is per **output** channel, so its
/// length is `out_channels`, not `x.cols`.
#[allow(clippy::too_many_arguments)]
pub fn conv_transpose1d(
    x: &Mat,
    weight: &Mat,
    out_channels: usize,
    kernel: usize,
    bias: &[f32],
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> Mat {
    assert!(
        weight.rows == x.cols,
        "conv_transpose1d: weight.rows != x.cols (Cin mismatch)"
    );
    assert!(
        weight.cols == out_channels * kernel,
        "conv_transpose1d: weight.cols != out_channels * kernel"
    );
    assert!(stride >= 1, "conv_transpose1d: stride must be >= 1");
    assert!(
        bias.is_empty() || bias.len() == out_channels,
        "conv_transpose1d: bias length != out_channels"
    );

    let t_out = conv_transpose1d_out_len(x.rows, kernel, stride, padding, output_padding);

    // A codec decoder block picks kernel = 2*stride, padding = ceil(stride/2),
    // output_padding = stride % 2, which makes the output exactly `stride`
    // times longer. Every padding or output-padding mistake breaks that
    // multiple, so check it here: a failed assertion beats bisecting a
    // 4-block decoder by listening to the result.
    if kernel == 2 * stride && padding == (stride + 1) / 2 && output_padding == stride % 2 {
        assert!(
            t_out == x.rows * stride,
            "conv_transpose1d: codec block must multiply length by exactly stride"
        );
    }

    // Every (t, co, k) product in one gemm. `weight` is [Cin, Cout*K], so
    // column `co * K + k` of the result holds the contribution that tap `k` of
    // output channel `co` makes from input frame `t`.
    let prod = x.matmul(weight);

    let mut out = Mat::zeros(t_out, out_channels);
    for t in 0..x.rows {
        let prow = row(&prod, t);
        for k in 0..kernel {
            // Scatter form: input frame t writes to t*stride + k, then padding
            // trims from the front. No kernel flip -- see the header.
            let placed = t * stride + k;
            if placed < padding {
                continue;
            }
            let dst = placed - padding;
            if dst >= t_out {
                continue;
            }
            let orow = row_mut(&mut out, dst);
            for co in 0..out_channels {
                orow[co] += prow[co * kernel + k];
            }
        }
    }

    add_bias_rows(&mut out, bias);
    out
}

// =============================================================================
// Activations
// =============================================================================

/// In-place `snake1d`, for the long activation chains in a decoder where the
/// intermediate is dead immediately afterwards.
pub fn snake1d_inplace(x: &mut Mat, alpha: &[f32]) {
    assert!(alpha.len() == x.cols, "snake1d: alpha length != x.cols");

    // Reciprocal hoisted per channel, and the epsilon added before inverting
    // rather than used as a guard afterwards -- trained alphas come close
    // enough to zero that the two differ.
    let recip: Vec<f32> = alpha.iter().map(|&a| 1.0f32 / (a + 1e-9f32)).collect();

    let cols = x.cols;
    for row in x.data.chunks_exact_mut(cols.max(1)) {
        for c in 0..cols {
            let s = (alpha[c] * row[c]).sin();
            row[c] += s * s * recip[c];
        }
    }
}

/// Snake: `x + sin^2(alpha * x) / (alpha + 1e-9)`, with a learned per-channel
/// `alpha`.
///
/// A periodic activation. Unlike GELU or SiLU it does not saturate, which is
/// what lets a codec decoder reproduce the periodic structure of voiced speech
/// -- the inductive bias the function exists for.
///
/// The epsilon sits **inside** the reciprocal (`1 / (alpha + 1e-9)`, not
/// `1 / alpha` guarded afterwards), matching the reference; trained `alpha`
/// values pass close enough to zero for the difference to show.
///
/// `x` is [T, C] and `alpha` has length C. Do not substitute BigVGAN's
/// two-parameter `snake_beta` -- SNAC uses the single-alpha form.
pub fn snake1d(x: &Mat, alpha: &[f32]) -> Mat {
    let mut out = x.clone();
    snake1d_inplace(&mut out, alpha);
    out
}

// =============================================================================
// Weight normalization
// =============================================================================

/// Reconstruct a weight-normalized parameter: `W = g * v / ||v||`.
///
/// `torch.nn.utils.parametrizations.weight_norm` stores the parameter as a
/// magnitude `g` (`parametrizations.weight.original0`) and a direction `v`
/// (`...original1`), and reconstructs `W = g * v / ||v||` with the norm taken
/// over every axis **except** axis 0 of the stored tensor.
///
/// Which axis that is depends on the layer, and this is a real trap:
///
/// | layer              | stored `v` shape  | `g` shape    | norm is per   |
/// |--------------------|-------------------|--------------|---------------|
/// | `Conv1d`           | [Cout, Cin, K]    | [Cout, 1, 1] | output channel|
/// | `ConvTranspose1d`  | [Cin, Cout, K]    | [Cin, 1, 1]  | **input** ch. |
///
/// So a transposed convolution normalizes per input channel while every other
/// convolution in the same model normalizes per output channel. Taking axis 0
/// of the *stored* tensor -- rather than assuming "output channels" -- gets
/// both right with no special case, which is why this takes a flat `v` and
/// derives the group count from `g.len()`.
///
/// Returns the reconstructed flat weight, same length and order as `v`.
pub fn weight_norm_combine(g: &[f32], v: &[f32]) -> Vec<f32> {
    assert!(!g.is_empty(), "weight_norm_combine: g is empty");
    assert!(!v.is_empty(), "weight_norm_combine: v is empty");
    assert!(
        v.len() % g.len() == 0,
        "weight_norm_combine: v length not divisible by g length"
    );

    // Axis 0 of the *stored* tensor, whatever that axis means for this layer:
    // output channels for Conv1d, input channels for ConvTranspose1d.
    let groups = g.len();
    let inner = v.len() / groups;

    let mut out = vec![0.0f32; v.len()];
    for i in 0..groups {
        let vr = &v[i * inner..(i + 1) * inner];
        let mut sq = 0.0f32;
        for &value in vr {
            sq += value * value;
        }
        let len = sq.sqrt();
        let scale = if len > 0.0 { g[i] / len } else { 0.0 };
        let orow = &mut out[i * inner..(i + 1) * inner];
        for j in 0..inner {
            orow[j] = vr[j] * scale;
        }
    }
    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::InitRng;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    /// Relative comparison, for results large enough that an absolute tolerance
    /// says nothing.
    ///
    /// im2col + gemm and the direct accumulation loop sum the same products in
    /// different orders, so they agree to f32 epsilon and not further. At values in
    /// the thousands that is millivolts of absolute difference and entirely
    /// expected; only the relative error is meaningful.
    fn approx_rel(a: f32, b: f32) -> bool {
        let tol = 1e-5f32;
        let scale = 1.0f32.max(a.abs()).max(b.abs());
        (a - b).abs() / scale < tol
    }

    /// Fill a matrix with a deterministic spread of small signed values.
    fn ramp(rows: usize, cols: usize, scale: f32, shift: f32) -> Mat {
        Mat::from_fn(rows, cols, |r, c| (r * cols + c) as f32 * scale - shift)
    }

    fn ramp_default(rows: usize, cols: usize) -> Mat {
        ramp(rows, cols, 0.01, 0.5)
    }

    /// Reference dense convolution, written out index by index.
    fn ref_conv1d_dense(
        x: &Mat,
        w: &Mat,
        c_out: usize,
        kernel: usize,
        dilation: usize,
        padding: usize,
    ) -> Mat {
        let c_in = x.cols;
        let t_out = x.rows + 2 * padding - dilation * (kernel - 1);
        let mut out = Mat::zeros(t_out, c_out);
        for t in 0..t_out {
            for co in 0..c_out {
                let mut acc = 0.0f32;
                for ci in 0..c_in {
                    for k in 0..kernel {
                        let src = (t + k * dilation) as i64 - padding as i64;
                        if src < 0 || src >= x.rows as i64 {
                            continue;
                        }
                        acc += x.at(src as usize, ci) * w.at(co, ci * kernel + k);
                    }
                }
                *out.at_mut(t, co) = acc;
            }
        }
        out
    }

    /// Reference transposed convolution in scatter form, written out index by index.
    fn ref_conv_transpose1d(
        x: &Mat,
        w: &Mat,
        c_out: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> Mat {
        let c_in = x.cols;
        let t_out = (x.rows - 1) * stride + kernel + output_padding - 2 * padding;
        let mut out = Mat::zeros(t_out, c_out);
        for t in 0..x.rows {
            for k in 0..kernel {
                let dst = (t * stride + k) as i64 - padding as i64;
                if dst < 0 || dst >= t_out as i64 {
                    continue;
                }
                for co in 0..c_out {
                    let mut acc = 0.0f32;
                    for ci in 0..c_in {
                        acc += x.at(t, ci) * w.at(ci, co * kernel + k);
                    }
                    *out.at_mut(dst as usize, co) += acc;
                }
            }
        }
        out
    }

    /// Reference strided grouped convolution, written the obvious way.
    #[allow(clippy::too_many_arguments)]
    fn ref_conv1d_grouped(
        x: &Mat,
        w: &Mat,
        c_out: usize,
        kernel: usize,
        groups: usize,
        dilation: usize,
        padding: usize,
        stride: usize,
    ) -> Mat {
        let in_per = x.cols / groups;
        let out_per = c_out / groups;
        let reach = dilation * (kernel - 1);
        let t_out = (x.rows + 2 * padding - reach - 1) / stride + 1;
        let mut out = Mat::zeros(t_out, c_out);
        for t in 0..t_out {
            for co in 0..c_out {
                let g = co / out_per;
                let mut acc = 0.0f32;
                for ci in 0..in_per {
                    for k in 0..kernel {
                        let src = (t * stride + k * dilation) as i64 - padding as i64;
                        if src < 0 || src >= x.rows as i64 {
                            continue;
                        }
                        acc += x.at(src as usize, g * in_per + ci) * w.at(co, ci * kernel + k);
                    }
                }
                *out.at_mut(t, co) = acc;
            }
        }
        out
    }

    // =========================================================================
    // Output lengths
    // =========================================================================

    #[test]
    fn conv1d_out_len_matches_pytorch_stride_1_formula() {
        assert_eq!(conv1d_out_len(10, 1, 1, 0, 1), 10);
        assert_eq!(conv1d_out_len(10, 7, 1, 0, 1), 4);
        assert_eq!(conv1d_out_len(10, 7, 1, 3, 1), 10); // pad = (k-1)/2 preserves length
        assert_eq!(conv1d_out_len(10, 3, 2, 2, 1), 10); // pad = d*(k-1)/2 preserves length
    }

    #[test]
    fn conv1d_out_len_preserves_length_for_every_snac_residual_dilation() {
        // ResidualUnit: kernel 7, padding = ((7-1)*dilation)/2 = 3*dilation.
        // Exact length preservation is what makes the reference's centre-crop
        // branch dead code, so the whole decoder can assume it.
        for dilation in [1usize, 3, 9] {
            assert_eq!(conv1d_out_len(64, 7, dilation, 3 * dilation, 1), 64);
        }
    }

    #[test]
    fn conv1d_out_len_reports_0_when_the_kernel_does_not_fit() {
        assert_eq!(conv1d_out_len(4, 7, 1, 0, 1), 0);
        assert_eq!(conv1d_out_len(6, 7, 1, 0, 1), 0); // padded == reach, not one more
    }

    #[test]
    fn conv_transpose1d_out_len_collapses_to_t_times_stride_for_codec_blocks() {
        // kernel = 2*stride, padding = ceil(stride/2), output_padding = stride % 2.
        for stride in [1usize, 2, 3, 4, 8] {
            let kernel = 2 * stride;
            let padding = (stride + 1) / 2;
            let output_padding = stride % 2;
            for t in [1usize, 4, 37] {
                assert_eq!(
                    conv_transpose1d_out_len(t, kernel, stride, padding, output_padding),
                    t * stride
                );
            }
        }
    }

    #[test]
    fn conv_transpose1d_out_len_matches_pytorch_outside_the_codec_form() {
        assert_eq!(conv_transpose1d_out_len(4, 3, 1, 0, 0), 6); // (4-1)*1 + 3
        assert_eq!(conv_transpose1d_out_len(4, 4, 2, 0, 0), 10); // (4-1)*2 + 4
        assert_eq!(conv_transpose1d_out_len(0, 4, 2, 0, 0), 0);
    }

    // =========================================================================
    // Pointwise
    // =========================================================================

    #[test]
    fn conv1d_pointwise_is_a_dense_channel_mix() {
        // x = [1 2], weight = [[1 0], [0 1], [1 1]] -> out = [1 2 3]
        let x = Mat::new(vec![1.0, 2.0], 1, 2);
        let w = Mat::new(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2);
        let out = conv1d_pointwise(&x, &w, &[]);
        assert_eq!(out.rows, 1);
        assert_eq!(out.cols, 3);
        assert!(approx(out.at(0, 0), 1.0));
        assert!(approx(out.at(0, 1), 2.0));
        assert!(approx(out.at(0, 2), 3.0));
    }

    #[test]
    fn conv1d_pointwise_adds_a_per_output_channel_bias() {
        let x = Mat::new(vec![1.0, 2.0], 1, 2);
        let w = Mat::new(vec![1.0, 0.0, 0.0, 1.0], 2, 2);
        let bias = [10.0f32, -10.0];
        let out = conv1d_pointwise(&x, &w, &bias);
        assert!(approx(out.at(0, 0), 11.0));
        assert!(approx(out.at(0, 1), -8.0));
    }

    #[test]
    fn conv1d_dense_delegates_to_pointwise_at_kernel_1() {
        let x = ramp_default(9, 5);
        let w = ramp(7, 5, 0.03, 0.2);
        let bias = [0.1f32, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7];
        let dense = conv1d_dense(&x, &w, 7, 1, &bias, 1, 0, 1);
        let point = conv1d_pointwise(&x, &w, &bias);
        assert_eq!(dense.rows, point.rows);
        assert_eq!(dense.cols, point.cols);
        for i in 0..dense.data.len() {
            assert!(approx(dense.data[i], point.data[i]));
        }
    }

    // =========================================================================
    // Depthwise
    // =========================================================================

    #[test]
    fn conv1d_depthwise_does_not_mix_channels() {
        // Channel 0 gets an identity tap, channel 1 a doubling tap. If the two
        // ever mixed, the outputs would not stay proportional to their own input.
        let x = Mat::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2);
        let w = Mat::new(vec![1.0, 2.0], 2, 1); // [C=2, K=1]
        let out = conv1d_depthwise(&x, &w, &[], 1, 0);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 2);
        for t in 0..3 {
            assert!(approx(out.at(t, 0), x.at(t, 0)));
            assert!(approx(out.at(t, 1), x.at(t, 1) * 2.0));
        }
    }

    #[test]
    fn conv1d_depthwise_matches_a_per_channel_dense_convolution() {
        let channels = 6;
        let kernel = 7;
        for dilation in [1usize, 3, 9] {
            let padding = 3 * dilation;
            let x = ramp(24, channels, 0.017, 0.4);
            let w = ramp(channels, kernel, 0.031, 0.3);
            let got = conv1d_depthwise(&x, &w, &[], dilation, padding);
            assert_eq!(got.rows, 24);
            assert_eq!(got.cols, channels);

            // Same thing as a dense convolution with a block-diagonal weight.
            let mut dense_w = Mat::zeros(channels, channels * kernel);
            for c in 0..channels {
                for k in 0..kernel {
                    *dense_w.at_mut(c, c * kernel + k) = w.at(c, k);
                }
            }
            let want = ref_conv1d_dense(&x, &dense_w, channels, kernel, dilation, padding);
            for i in 0..got.data.len() {
                assert!(approx(got.data[i], want.data[i]));
            }
        }
    }

    // =========================================================================
    // Dense
    // =========================================================================

    #[test]
    fn conv1d_dense_matches_the_reference_loop() {
        let x = ramp(20, 4, 0.023, 0.35);
        let w = ramp(3, 4 * 7, 0.011, 0.25);
        let bias = [0.5f32, -0.25, 0.125];
        let got = conv1d_dense(&x, &w, 3, 7, &bias, 1, 3, 1);
        let mut want = ref_conv1d_dense(&x, &w, 3, 7, 1, 3);
        for r in 0..want.rows {
            for c in 0..want.cols {
                *want.at_mut(r, c) += bias[c];
            }
        }
        assert_eq!(got.rows, want.rows);
        assert_eq!(got.cols, want.cols);
        for i in 0..got.data.len() {
            assert!(approx(got.data[i], want.data[i]));
        }
    }

    #[test]
    fn conv1d_dense_shapes_match_snacs_final_projection() {
        // decoder.model.7: WNConv1d(64 -> 1, kernel 7, padding 3).
        let x = ramp(128, 64, 0.001, 0.06);
        let w = ramp(1, 64 * 7, 0.0005, 0.01);
        let bias = [0.02f32];
        let out = conv1d_dense(&x, &w, 1, 7, &bias, 1, 3, 1);
        assert_eq!(out.rows, 128);
        assert_eq!(out.cols, 1);
    }

    // =========================================================================
    // Transposed
    // =========================================================================

    #[test]
    fn conv_transpose1d_lays_an_impulse_response_down_unflipped() {
        // A single input frame of 1.0 must reproduce the kernel in forward order.
        // The gather formulation of the same operation would emit it reversed, so
        // this is the test that catches a flipped kernel.
        let kernel = 3;
        let x = Mat::new(vec![1.0], 1, 1);
        let w = Mat::new(vec![7.0, 8.0, 9.0], 1, kernel); // [Cin=1, Cout*K = 1*3]
        let out = conv_transpose1d(&x, &w, 1, kernel, &[], 1, 0, 0);
        assert_eq!(out.rows, 3);
        assert_eq!(out.cols, 1);
        assert!(approx(out.at(0, 0), 7.0));
        assert!(approx(out.at(1, 0), 8.0));
        assert!(approx(out.at(2, 0), 9.0));
    }

    #[test]
    fn conv_transpose1d_overlaps_strided_impulses_additively() {
        // Two frames, stride 2, kernel 4: frame 0 writes positions 0..3 and frame 1
        // writes 2..5, so 2 and 3 receive a contribution from both.
        let kernel = 4;
        let x = Mat::new(vec![1.0, 1.0], 2, 1);
        let w = Mat::new(vec![1.0, 2.0, 4.0, 8.0], 1, kernel);
        let out = conv_transpose1d(&x, &w, 1, kernel, &[], 2, 0, 0);
        assert_eq!(out.rows, 6);
        assert!(approx(out.at(0, 0), 1.0));
        assert!(approx(out.at(1, 0), 2.0));
        assert!(approx(out.at(2, 0), 4.0 + 1.0));
        assert!(approx(out.at(3, 0), 8.0 + 2.0));
        assert!(approx(out.at(4, 0), 4.0));
        assert!(approx(out.at(5, 0), 8.0));
    }

    #[test]
    fn conv_transpose1d_matches_the_reference_loop() {
        for stride in [2usize, 4, 8] {
            let kernel = 2 * stride;
            let padding = (stride + 1) / 2;
            let c_in = 6;
            let c_out = 3;
            let x = ramp(5, c_in, 0.019, 0.3);
            let w = ramp(c_in, c_out * kernel, 0.007, 0.2);
            let got = conv_transpose1d(&x, &w, c_out, kernel, &[], stride, padding, stride % 2);
            let want = ref_conv_transpose1d(&x, &w, c_out, kernel, stride, padding, stride % 2);
            assert_eq!(got.rows, want.rows);
            assert_eq!(got.cols, want.cols);
            for i in 0..got.data.len() {
                assert!(approx(got.data[i], want.data[i]));
            }
        }
    }

    #[test]
    fn conv_transpose1d_biases_per_output_channel_not_per_input() {
        // Cin != Cout for every SNAC upsampling block, so a bias applied on the
        // wrong axis is a length mismatch rather than a silent error.
        let stride = 2;
        let kernel = 2 * stride;
        let x = Mat::zeros(3, 4); // Cin = 4
        let w = Mat::zeros(4, 2 * kernel);
        let bias = [1.0f32, -1.0]; // Cout = 2
        let out = conv_transpose1d(&x, &w, 2, kernel, &bias, stride, (stride + 1) / 2, stride % 2);
        assert_eq!(out.rows, 6);
        assert_eq!(out.cols, 2);
        for t in 0..out.rows {
            assert!(approx(out.at(t, 0), 1.0));
            assert!(approx(out.at(t, 1), -1.0));
        }
    }

    #[test]
    fn snac_decoder_rates_upsample_4_code_frames_to_exactly_2048_samples() {
        // The end-to-end length invariant: SNAC 24 kHz uses decoder_rates
        // [8, 8, 4, 2] for a total of 512x, and one Orpheus 7-token frame carries
        // 4 frames at the finest codebook rate -- so 4 * 512 = 2048 samples, which
        // is 85.33 ms at 24 kHz. Everything about the frame accounting downstream
        // rests on this, so pin it here with real convolutions rather than
        // arithmetic.
        let mut rng = InitRng::new(7);
        let mut h = Mat::from_fn(4, 32, |_, _| 0.0);
        for v in h.data.iter_mut() {
            *v = rng.next_normal() * 0.1;
        }

        let mut channels = 32;
        for stride in [8usize, 8, 4, 2] {
            let kernel = 2 * stride;
            let out_channels = channels / 2;
            let mut w = Mat::zeros(channels, out_channels * kernel);
            for v in w.data.iter_mut() {
                *v = rng.next_normal() * 0.05;
            }
            h = conv_transpose1d(&h, &w, out_channels, kernel, &[], stride, (stride + 1) / 2, stride % 2);
            channels = out_channels;

            // Each residual unit must preserve length exactly, at every dilation.
            for dilation in [1usize, 3, 9] {
                let before = h.rows;
                let mut dw = Mat::zeros(channels, 7);
                for v in dw.data.iter_mut() {
                    *v = rng.next_normal() * 0.1;
                }
                let filtered = conv1d_depthwise(&h, &dw, &[], dilation, 3 * dilation);
                assert_eq!(filtered.rows, before);
            }
        }

        assert_eq!(h.rows, 4 * 512);
        assert_eq!(h.rows, 2048);
        assert_eq!(channels, 2);
    }

    // =========================================================================
    // Snake
    // =========================================================================

    #[test]
    fn snake1d_matches_x_plus_sin2_alpha_x_over_alpha() {
        let x = Mat::new(vec![0.5, -1.25, 2.0, 0.0], 2, 2);
        let alpha = [1.0f32, 0.5];
        let out = snake1d(&x, &alpha);
        for r in 0..2 {
            for c in 0..2 {
                let a = alpha[c];
                let s = (a * x.at(r, c)).sin();
                assert!(approx(out.at(r, c), x.at(r, c) + s * s / (a + 1e-9)));
            }
        }
    }

    #[test]
    fn snake1d_applies_alpha_per_channel() {
        // Same value in both channels, different alpha -> different output. A
        // scalar alpha, or one indexed by row, would make these equal.
        let x = Mat::new(vec![1.0, 1.0], 1, 2);
        let alpha = [1.0f32, 2.0];
        let out = snake1d(&x, &alpha);
        assert!(!approx(out.at(0, 0), out.at(0, 1)));
    }

    #[test]
    fn snake1d_survives_alpha_at_and_below_zero() {
        // Trained alphas pass close to zero; the epsilon lives inside the
        // reciprocal so this must stay finite rather than blowing up.
        let x = Mat::new(vec![1.0, 1.0, 1.0], 1, 3);
        let alpha = [0.0f32, -1.0, 1e-8];
        let out = snake1d(&x, &alpha);
        for c in 0..3 {
            assert!(out.at(0, c).is_finite());
        }
        // alpha == 0 gives sin(0) == 0 exactly, so the value passes through.
        assert!(approx(out.at(0, 0), 1.0));
        // sin is odd and squared, so negating alpha only changes the divisor sign.
        assert!(approx(out.at(0, 1), 1.0 - 1.0f32.sin() * 1.0f32.sin()));
    }

    #[test]
    fn snake1d_inplace_agrees_with_the_copying_form() {
        let x = ramp(7, 5, 0.07, 0.9);
        let alpha = [0.5f32, 1.0, 1.5, -0.5, 2.0];
        let copied = snake1d(&x, &alpha);
        let mut in_place = x.clone();
        snake1d_inplace(&mut in_place, &alpha);
        for i in 0..copied.data.len() {
            assert!(approx(copied.data[i], in_place.data[i]));
        }
    }

    // =========================================================================
    // Weight normalization
    // =========================================================================

    #[test]
    fn weight_norm_combine_rescales_each_group_to_its_magnitude() {
        // ||[3, 4]|| == 5, so g == 2 scales it by 2/5.
        let w = weight_norm_combine(&[2.0], &[3.0, 4.0]);
        assert_eq!(w.len(), 2);
        assert!(approx(w[0], 1.2));
        assert!(approx(w[1], 1.6));
    }

    #[test]
    fn weight_norm_combine_normalizes_every_group_independently() {
        let w = weight_norm_combine(&[1.0, 10.0], &[3.0, 4.0, 0.0, 5.0]);
        assert!(approx(w[0], 0.6));
        assert!(approx(w[1], 0.8));
        assert!(approx(w[2], 0.0));
        assert!(approx(w[3], 10.0));
    }

    #[test]
    fn weight_norm_combine_groups_by_axis_0_of_the_stored_tensor() {
        // The trap: Conv1d stores [Cout, Cin, K] and ConvTranspose1d stores
        // [Cin, Cout, K], so the same flat buffer normalizes over a different axis
        // depending on the layer. Deriving the group count from g rather than
        // assuming "output channels" handles both -- these two calls share a `v`
        // and must give genuinely different answers.
        let v = [1.0f32; 6];

        // Conv1d-style: g per output channel, 2 groups of 3.
        let as_conv = weight_norm_combine(&[1.0, 1.0], &v);
        // ConvTranspose1d-style: g per input channel, 3 groups of 2.
        let as_convt = weight_norm_combine(&[1.0, 1.0, 1.0], &v);

        assert_eq!(as_conv.len(), v.len());
        assert_eq!(as_convt.len(), v.len());
        // 1/sqrt(3) vs 1/sqrt(2) -- picking the wrong axis is a real numeric error,
        // not a relabelling.
        assert!(approx(as_conv[0], 1.0 / 3.0f32.sqrt()));
        assert!(approx(as_convt[0], 1.0 / 2.0f32.sqrt()));
    }

    #[test]
    fn weight_norm_combine_handles_snacs_transposed_conv_shapes() {
        // decoder.model.2.block.1: v is [1024, 512, 16] and g is [1024, 1, 1], so
        // the magnitude is per input channel. Scaled down here, same structure.
        let c_in = 8;
        let c_out = 4;
        let kernel = 16;
        let g: Vec<f32> = (0..c_in).map(|i| 1.0 + i as f32).collect();
        let mut v = vec![0.0f32; c_in * c_out * kernel];
        for i in 0..c_in {
            v[i * c_out * kernel] = 2.0; // one nonzero per group, norm == 2
        }
        let w = weight_norm_combine(&g, &v);
        assert_eq!(w.len(), v.len());
        for i in 0..c_in {
            // g[i] * 2 / 2 == g[i]
            assert!(approx(w[i * c_out * kernel], g[i]));
        }
    }

    #[test]
    fn weight_norm_combine_yields_zeros_for_a_zero_direction() {
        let w = weight_norm_combine(&[5.0], &[0.0, 0.0, 0.0]);
        for value in w {
            assert_eq!(value, 0.0);
            assert!(value.is_finite());
        }
    }

    #[test]
    fn conv1d_dense_im2col_path_matches_the_direct_loop() {
        // Big enough to cross the gemm threshold, so this compares the two
        // implementations against each other rather than re-testing one of them.
        let c_in = 24;
        let c_out = 24;
        let kernel = 7;
        let x = ramp(600, c_in, 0.0013, 0.4);
        let w = ramp(c_out, c_in * kernel, 0.0007, 0.15);
        let bias: Vec<f32> = (0..c_out).map(|i| 0.01 * i as f32 - 0.1).collect();

        let got = conv1d_dense(&x, &w, c_out, kernel, &bias, 1, 3, 1);
        let mut want = ref_conv1d_dense(&x, &w, c_out, kernel, 1, 3);
        for r in 0..want.rows {
            for c in 0..want.cols {
                *want.at_mut(r, c) += bias[c];
            }
        }
        assert_eq!(got.rows, 600);
        assert_eq!(got.cols, c_out);
        for i in 0..got.data.len() {
            assert!(approx_rel(got.data[i], want.data[i]));
        }
    }

    #[test]
    fn conv1d_dense_im2col_is_correct_across_tile_boundaries() {
        // The patch matrix is built in row tiles, so a position whose receptive
        // field straddles a tile edge is the interesting case: the gather reads
        // from `x`, not from the previous tile, so it must not depend on tiling at
        // all. Dilation 3 widens the field to 18 positions to make that bite.
        let c_in = 16;
        let c_out = 8;
        let kernel = 7;
        let dilation = 3;
        let x = ramp(2000, c_in, 0.0011, 0.5);
        let w = ramp(c_out, c_in * kernel, 0.0009, 0.2);

        let got = conv1d_dense(&x, &w, c_out, kernel, &[], dilation, 3 * dilation, 1);
        let want = ref_conv1d_dense(&x, &w, c_out, kernel, dilation, 3 * dilation);
        assert_eq!(got.rows, want.rows);
        assert_eq!(got.rows, 2000);
        for i in 0..got.data.len() {
            assert!(approx_rel(got.data[i], want.data[i]));
        }
    }

    #[test]
    fn conv1d_dense_handles_omnivoices_residual_unit_shape() {
        // acoustic_decoder.block.N.res_unitN.conv1: dense 7-tap, 512 channels.
        // These run after the sequence has been upsampled, which is why the gemm
        // path exists at all.
        let x = ramp(1024, 64, 0.0005, 0.02);
        let w = ramp(64, 64 * 7, 0.0003, 0.01);
        let out = conv1d_dense(&x, &w, 64, 7, &[], 1, 3, 1);
        assert_eq!(out.rows, 1024);
        assert_eq!(out.cols, 64);
        for v in &out.data {
            assert!(v.is_finite());
        }
    }

    // =========================================================================
    // Stride and groups
    // =========================================================================

    #[test]
    fn conv1d_out_len_follows_pytorch_when_strided() {
        // floor((T + 2p - d*(k-1) - 1) / s) + 1.
        assert_eq!(conv1d_out_len(100, 7, 1, 3, 1), 100);
        assert_eq!(conv1d_out_len(100, 4, 1, 1, 2), 50);
        assert_eq!(conv1d_out_len(100, 16, 1, 4, 8), 12);
        // The odd-stride case the codec's last encoder block hits: kernel 6,
        // stride 3, padding 2 turns 3n samples into exactly n.
        assert_eq!(conv1d_out_len(30, 6, 1, 2, 3), 10);
        assert_eq!(conv1d_out_len(300, 6, 1, 2, 3), 100);
    }

    #[test]
    fn conv1d_dense_strides_without_changing_the_arithmetic() {
        let x = ramp(200, 12, 0.003, 0.4);
        let w = ramp(20, 12 * 5, 0.001, 0.2);
        for stride in [1usize, 2, 3, 5] {
            let got = conv1d_dense(&x, &w, 20, 5, &[], 1, 2, stride);
            let want = ref_conv1d_grouped(&x, &w, 20, 5, 1, 1, 2, stride);
            assert_eq!(got.rows, want.rows);
            for i in 0..got.data.len() {
                assert!(approx(got.data[i], want.data[i]));
            }
        }
    }

    #[test]
    fn a_strided_convolution_is_a_subsample_of_an_unstrided_one() {
        // Nothing about the dot products changes with stride; only how many of
        // them are evaluated. This catches an off-by-one in the input offset that
        // a same-shape comparison would not.
        let x = ramp(120, 8, 0.005, 0.3);
        let w = ramp(6, 8 * 3, 0.002, 0.1);
        let full = conv1d_dense(&x, &w, 6, 3, &[], 1, 1, 1);
        let strided = conv1d_dense(&x, &w, 6, 3, &[], 1, 1, 4);
        for t in 0..strided.rows {
            for c in 0..6 {
                assert!(approx(strided.at(t, c), full.at(t * 4, c)));
            }
        }
    }

    #[test]
    fn conv1d_grouped_matches_a_reference_loop() {
        let groups = 4;
        let x = ramp(80, 16, 0.004, 0.35);
        let w = ramp(24, (16 / groups) * 3, 0.0015, 0.12);
        let bias: Vec<f32> = (0..24).map(|i| 0.01 * i as f32 - 0.1).collect();

        let got = conv1d_grouped(&x, &w, 24, 3, &bias, groups, 1, 1, 2);
        let mut want = ref_conv1d_grouped(&x, &w, 24, 3, groups, 1, 1, 2);
        for r in 0..want.rows {
            for c in 0..want.cols {
                *want.at_mut(r, c) += bias[c];
            }
        }
        assert_eq!(got.rows, want.rows);
        for i in 0..got.data.len() {
            assert!(approx(got.data[i], want.data[i]));
        }
    }

    #[test]
    fn conv1d_grouped_degenerates_to_the_two_special_cases() {
        let x = ramp(60, 8, 0.004, 0.3);

        // groups == 1 is a dense convolution.
        let dense_w = ramp(8, 8 * 3, 0.002, 0.1);
        let a = conv1d_grouped(&x, &dense_w, 8, 3, &[], 1, 1, 1, 1);
        let b = conv1d_dense(&x, &dense_w, 8, 3, &[], 1, 1, 1);
        assert_eq!(a.data.len(), b.data.len());
        for i in 0..a.data.len() {
            assert!(approx(a.data[i], b.data[i]));
        }

        // groups == channels is a depthwise one. `conv1d_depthwise` stores [C, K],
        // which is the same buffer a grouped weight of [C, 1 * K] holds.
        let dw = ramp(8, 3, 0.003, 0.2);
        let c = conv1d_grouped(&x, &dw, 8, 3, &[], 8, 1, 1, 1);
        let d = conv1d_depthwise(&x, &dw, &[], 1, 1);
        assert_eq!(c.data.len(), d.data.len());
        for i in 0..c.data.len() {
            assert!(approx(c.data[i], d.data[i]));
        }
    }

    #[test]
    fn conv1d_grouped_keeps_the_groups_apart() {
        // The point of grouping: changing an input channel must not move an output
        // channel in another group. A dense convolution would fail this.
        let groups = 2;
        let mut x = ramp(40, 4, 0.01, 0.5);
        let w = ramp(4, (4 / groups) * 3, 0.005, 0.25);

        let before = conv1d_grouped(&x, &w, 4, 3, &[], groups, 1, 1, 1);
        *x.at_mut(20, 0) += 1.0; // group 0's input
        let after = conv1d_grouped(&x, &w, 4, 3, &[], groups, 1, 1, 1);

        let mut group0_moved = false;
        for t in 0..before.rows {
            for c in 0..2 {
                group0_moved = group0_moved || before.at(t, c) != after.at(t, c);
            }
            for c in 2..4 {
                assert_eq!(before.at(t, c), after.at(t, c));
            }
        }
        assert!(group0_moved);
    }
}
