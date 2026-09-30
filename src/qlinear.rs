#![allow(dead_code)]
// =============================================================================
// QLinear -- an inference-only linear layer over whichever weight format loaded
// =============================================================================
//
// `Linear2` in `nn2.rs` is the trainable layer: it carries `TensorNode`
// weights so gradients flow, and inference paths reach it by wrapping their
// activations in `TensorNode::leaf` and unwrapping the result. That works, and
// it is what the language models do -- but it also builds an autodiff graph
// that inference never walks, and some of those nodes hold reference cycles
// that plain refcounting cannot collect.
//
// The diffusion stack never trains, so it uses this instead: one weight, three
// possible storages, `Mat` in and `Mat` out, no graph at any point.
//
//   f32   -- Mat::matmul_bt, the VAE and anything small
//   BF16  -- MatBf16::matmul_by_t, half the memory and lossless
//   Q4_K  -- Q4KMat::matmul_q4k_t_exact, what a 12B transformer has to use
//
// The weight is always [out_features, in_features] -- PyTorch's `Linear.weight`
// layout, so loading is a wrap -- and the multiply is always against its
// transpose.

use crate::autograd2::{Mat, MatBf16, Q4KMat};
use crate::nn2::mark_pages_reusable;

#[derive(Clone)]
pub struct QLinear {
    pub in_features: usize,
    pub out_features: usize,

    /// Exactly one of these is populated once a weight is loaded.
    pub f32: Mat,
    pub bf16: Option<MatBf16>,
    pub q4k: Option<Q4KMat>,

    /// Empty means no bias, which is the common case in this stack: T5 has no
    /// biases at all, and FLUX has them only on its projections.
    pub bias: Vec<f32>,
}

impl Default for QLinear {
    fn default() -> Self {
        QLinear { in_features: 0, out_features: 0, f32: Mat::zeros(0, 0), bf16: None, q4k: None, bias: Vec::new() }
    }
}

/// Add a row vector to every row of `x`, in place. Empty `b` is a no-op.
pub fn add_bias_rows(x: &mut Mat, b: &[f32]) {
    if b.is_empty() {
        return;
    }
    debug_assert!(b.len() == x.cols, "QLinear: bias length != out_features");
    let cols = x.cols;
    for r in 0..x.rows {
        let row = &mut x.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            row[c] += b[c];
        }
    }
}

impl QLinear {
    pub fn from_f32(weight: Mat, bias: Vec<f32>) -> Self {
        QLinear { out_features: weight.rows, in_features: weight.cols, f32: weight, bias, ..Default::default() }
    }

    pub fn from_bf16(weight: MatBf16, bias: Vec<f32>) -> Self {
        QLinear { out_features: weight.rows, in_features: weight.cols, bf16: Some(weight), bias, ..Default::default() }
    }

    pub fn from_q4k(weight: Q4KMat, out_features: usize, in_features: usize, bias: Vec<f32>) -> Self {
        QLinear { out_features, in_features, q4k: Some(weight), bias, ..Default::default() }
    }

    /// True once a weight of any format is present.
    pub fn loaded(&self) -> bool {
        self.q4k.is_some() || self.bf16.is_some() || self.f32.rows > 0
    }

    /// `x [T, in] -> [T, out]`, plus the bias if there is one.
    pub fn forward(&self, x: &Mat) -> Mat {
        debug_assert!(x.cols == self.in_features, "QLinear: input width != in_features");

        // Priority matches the memory cost of the format: whichever compact
        // form was loaded is the one that exists, and f32 is the fallback.
        //
        // Q4_K takes the chunked-sgemm path, not `matmul_q4k_t`. That one has a
        // fused NEON path for the single row a decode step asks for and a plain
        // triple loop for everything else, which is the right trade for a
        // language model and the wrong one here: this stack never has a batch
        // of one. A 256-token T5 prompt through 24 layers is a couple of
        // teraflops, and the scalar loop turns a minute of work into an hour of
        // it. The chunked path dequantizes a chunk of weight rows at a time and
        // hands each chunk to sgemm, for about 8 MB of scratch.
        //
        // `_exact` rather than `_blas` because of the other half of that trade:
        // the batch-of-one shortcut quantizes the activation to int8, and the
        // only batch-of-one matmuls in a diffusion transformer are the
        // modulation projections, whose shift, scale and gate multiply every
        // token and every channel of the block.
        let mut out = if let Some(q) = &self.q4k {
            q.matmul_q4k_t_exact(x)
        } else if let Some(b) = &self.bf16 {
            b.matmul_by_t(x)
        } else {
            x.matmul_bt(&self.f32)
        };
        add_bias_rows(&mut out, &self.bias);
        out
    }

    /// Bytes held by the weight, whichever format it is in.
    pub fn size_bytes(&self) -> usize {
        if let Some(q) = &self.q4k {
            return q.size_bytes();
        }
        if let Some(b) = &self.bf16 {
            return b.size_bytes();
        }
        self.f32.numel() * std::mem::size_of::<f32>()
    }

    /// Release the weight. For after a GPU upload, which holds its own copy.
    pub fn free_weight(&mut self) {
        self.q4k = None;
        self.bf16 = None;
        mark_pages_reusable(&self.f32.data);
        self.f32 = Mat::zeros(0, 0);
    }
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

    #[test]
    fn agrees_across_its_weight_formats() {
        let w = spread(6, 8, 1.0, 5);
        let b = spread_vec(6, 0.5, 6);
        let x = spread(4, 8, 1.0, 7);

        let f = QLinear::from_f32(w.clone(), b.clone());
        let h = QLinear::from_bf16(w.to_bf16(), b);

        let rf = f.forward(&x);
        let rh = h.forward(&x);
        assert_eq!(rf.rows, 4);
        assert_eq!(rf.cols, 6);
        for i in 0..rf.data.len() {
            // BF16 keeps 8 mantissa bits, so agreement is to about 1%.
            assert!(approx(rf.data[i], rh.data[i], 0.05));
        }
    }

    #[test]
    fn applies_its_bias_once_per_row() {
        let w = spread(3, 5, 1.0, 8);
        let b = vec![1.0f32, -2.0, 0.5];
        let x = spread(4, 5, 1.0, 9);
        let plain = QLinear::from_f32(w.clone(), Vec::new()).forward(&x);
        let biased = QLinear::from_f32(w, b.clone()).forward(&x);
        for r in 0..4 {
            for c in 0..3 {
                assert!(approx(biased.at(r, c), plain.at(r, c) + b[c], 1e-5));
            }
        }
    }

    #[test]
    fn reports_whether_a_weight_is_present() {
        let empty = QLinear::default();
        assert!(!empty.loaded());
        let mut loaded = QLinear::from_f32(spread(2, 2, 1.0, 1), Vec::new());
        assert!(loaded.loaded());
        loaded.free_weight();
        assert!(!loaded.loaded());
    }
}
