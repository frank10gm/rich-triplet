/// # Tensor-Level Neural Network Layers (nn2)
///
/// Same layers as nn.rs, but now each layer operates on whole matrices
/// via TensorNode instead of vectors of scalar Values.
///
/// ## What changes
///
/// Before (scalar):
///   Linear::forward(&self, input: &[Value]) -> Vec<Value>
///   → Manual dot-product loop: for each output neuron j, sum_i(w[j][i]*x[i])
///   → Creates out_features * in_features * 2 + out_features nodes per call
///
/// After (tensor):
///   Linear::forward(&self, input: &TensorNode) -> TensorNode
///   → Single matmul: input @ weight.T + bias
///   → Creates 2 nodes total (matmul + add_bias)
///
/// The same math — one matrix multiplication instead of millions of scalar ops.
///
/// ## Parameter storage
///
/// Each parameter is a `TensorNode::leaf` — no backward function, just
/// raw data and a gradient accumulator. The optimizer reads `.grad()` and
/// writes `.set_data()` directly on these leaf nodes.

use crate::autograd2::{TensorNode, Mat, MatBf16, Q4Mat};
use crate::nn::InitRng; // reuse the RNG from Phase 2

// =============================================================================
// Trait: Module2
// =============================================================================

pub trait Module2 {
    /// All learnable parameters of this module.
    fn parameters(&self) -> Vec<TensorNode>;

    /// Zero all parameter gradients.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

// =============================================================================
// Trait: Trainable — anything train2() can optimise
// =============================================================================

/// A model that can be trained with `train2()`.
///
/// Both `Gpt2` (transformer2) and `GptOssModel` (transformer3) implement
/// this trait, so the same training loop works for both architectures.
pub trait Trainable: Module2 {
    /// Compute logits: token_ids → [T, vocab_size]
    fn forward_tokens(&self, token_ids: &[usize]) -> TensorNode;

    /// Compute cross-entropy loss: scalar TensorNode with backward wired.
    fn loss_tokens(&self, token_ids: &[usize], targets: &[usize]) -> TensorNode;

    /// Compute mean cross-entropy loss over a batch of B sequences.
    ///
    /// Runs `loss_tokens` for each sequence, averages the scalar loss values,
    /// then calls `backward()` on each individual loss scaled by 1/B so that
    /// gradients accumulate into the shared parameters correctly.
    ///
    /// Returns a plain scalar `TensorNode` (leaf) holding the mean loss value.
    /// Callers must NOT call `.backward()` on the returned node — backward has
    /// already been triggered internally.
    fn loss_batch_tokens(&self, batch: &[(&[usize], &[usize])]) -> TensorNode {
        let b = batch.len();
        assert!(b > 0, "loss_batch_tokens: empty batch");

        // Forward + backward for each sequence, accumulating grads / B.
        let mut total_loss = 0.0f32;
        for (inp, tgt) in batch {
            let loss_node = self.loss_tokens(inp, tgt);
            let val = loss_node.data().at(0, 0);
            total_loss += val;

            // Scale the upstream gradient by 1/B so the accumulated gradient
            // across all B sequences equals the mean-batch gradient.
            let upstream = Mat::new(vec![1.0 / b as f32], 1, 1);
            loss_node.set_grad(upstream);
            loss_node.backward();
        }

        TensorNode::leaf(Mat::new(vec![total_loss / b as f32], 1, 1))
    }
}

// =============================================================================
// Linear layer
// =============================================================================
//
// output = input @ weight.T + bias
//
// Shapes:
//   input:   [T, in_features]    (T = sequence length / batch size)
//   weight:  [out_features, in_features]
//   weight.T:[in_features, out_features]
//   bias:    [1, out_features]
//   output:  [T, out_features]
//
// Why weight.T?  We store weight as [out, in] (each row = one output neuron's
// weights) because that's how gradient math works out cleanly (dW = input.T @ dOut).
// To compute output we need [in, out] so we transpose.

pub struct Linear2 {
    /// [out_features, in_features]
    pub weight: TensorNode,
    /// [1, out_features]
    pub bias: TensorNode,
    pub in_features: usize,
    pub out_features: usize,
    /// INT4 quantized weight, set by `quantize()`.
    /// When present, forward uses `matmul_q4_t` instead of the f32 weight.
    pub q4_weight: Option<Q4Mat>,
    /// BF16 weight storage (inference-only). When present and q4_weight is
    /// absent, forward dequantizes on-the-fly: 2× less RAM than f32, lossless.
    pub bf16_weight: Option<MatBf16>,
}

impl Linear2 {
    pub fn new(in_features: usize, out_features: usize, rng: &mut InitRng) -> Self {
        let w_data = Mat::new(rng.normal_vec(out_features * in_features, 0.02),
                              out_features, in_features);
        let b_data = Mat::zeros(1, out_features);
        Linear2 {
            weight: TensorNode::leaf(w_data),
            bias:   TensorNode::leaf(b_data),
            in_features,
            out_features,
            q4_weight: None,
            bf16_weight: None,
        }
    }

    /// Create a Linear layer without bias (bias is fixed at zero, not a parameter).
    ///
    /// Used by architectures like Gemma 3 that have no bias in projection layers.
    pub fn new_no_bias(in_features: usize, out_features: usize, rng: &mut InitRng) -> Self {
        let w_data = Mat::new(rng.normal_vec(out_features * in_features, 0.02),
                              out_features, in_features);
        let b_data = Mat::zeros(1, out_features);
        Linear2 {
            weight: TensorNode::leaf(w_data),
            bias:   TensorNode::leaf(b_data),
            in_features,
            out_features,
            q4_weight: None,
            bf16_weight: None,
        }
    }

    /// Quantize the weight matrix to 4-bit and store it.
    ///
    /// After calling this, `forward()` uses the INT4 path (`matmul_q4_t`)
    /// instead of the f32 matmul — halving the memory footprint and avoiding
    /// the dequantize allocation on every call.
    ///
    /// The f32 weight tensor is kept unchanged so gradients still flow for
    /// any further training.
    pub fn quantize(&mut self) {
        self.q4_weight = Some(Q4Mat::quantize(&self.weight.data().clone()));
    }

    /// Quantize to INT4 and free the f32 weight (inference-only).
    ///
    /// Replaces the f32 weight with a zero-sized placeholder to reclaim RAM.
    /// Do NOT call this if you need backward passes (training).
    pub fn quantize_and_free_f32(&mut self) {
        self.q4_weight = Some(Q4Mat::quantize(&self.weight.data().clone()));
        // Replace f32 weight with an empty mat to free ~4× memory.
        self.weight.set_data(Mat::zeros(0, 0));
    }

    /// Quantize BF16 weight to INT4 and free all float storage (inference-only).
    ///
    /// Use this when the weight was loaded via `load_bf16`. Dequantizes BF16→f32
    /// in one pass, then quantizes to Q4, then frees both the BF16 and f32 copies.
    /// Falls back to `quantize_and_free_f32` if no BF16 weight is present.
    pub fn quantize_bf16_and_free(&mut self) {
        if let Some(ref bf16) = self.bf16_weight {
            let f32_mat = bf16.to_f32();
            self.q4_weight = Some(Q4Mat::quantize(&f32_mat));
            self.bf16_weight = None;
            self.weight.set_data(Mat::zeros(0, 0));
        } else {
            self.quantize_and_free_f32();
        }
    }

    /// input: [T, in_features]  →  output: [T, out_features]
    pub fn forward(&self, input: &TensorNode) -> TensorNode {
        self.fused_linear(input)
    }

    /// Fused linear: output = input @ weight.T + bias
    /// Store weights as BF16 for inference-only use (lossless, 2× less RAM).
    ///
    /// Clears the f32 TensorNode weight so it doesn't consume memory.
    /// Do NOT call this if you need backward passes.
    pub fn load_bf16(&mut self, bits: Vec<u16>, rows: usize, cols: usize) {
        self.bf16_weight = Some(MatBf16 { data: bits, rows, cols });
        self.weight.set_data(Mat::zeros(0, 0));
    }

    /// Keeps weight directly in the graph so its gradient is tracked.
    ///
    /// Priority: INT4 (q4_weight) > BF16 (bf16_weight) > f32 (weight).
    fn fused_linear(&self, input: &TensorNode) -> TensorNode {
        // We implement linear as a single custom node that tracks both
        // input and weight — avoiding a separate transpose node.
        let x = input.data().clone();
        let b = self.bias.data().clone();

        // Forward matmul: INT4 > BF16 > f32.
        let out_data = if let Some(ref q4) = self.q4_weight {
            assert_eq!(x.cols, q4.cols,
                "Linear (q4): input cols {} != weight cols {}", x.cols, q4.cols);
            #[cfg(feature = "metal")]
            let mut o = crate::metal_ops::metal_matmul_q4_t(&x, q4);
            #[cfg(not(feature = "metal"))]
            let mut o = q4.matmul_q4_t(&x);
            for r in 0..o.rows { for c in 0..o.cols { *o.at_mut(r, c) += b.at(0, c); } }
            o
        } else if let Some(ref bf16) = self.bf16_weight {
            // Dequantize BF16 → f32 on the fly (one bit-shift per element).
            let w = bf16.to_f32();
            assert_eq!(x.cols, w.cols,
                "Linear (bf16): input cols {} != weight cols {}", x.cols, w.cols);
            // Use BLAS transB to avoid allocating the [in, out] transpose matrix.
            #[cfg(feature = "blas")]
            let mut o = x.matmul_bt(&w);
            #[cfg(not(feature = "blas"))]
            let mut o = x.matmul(&w.transpose());
            for r in 0..o.rows { for c in 0..o.cols { *o.at_mut(r, c) += b.at(0, c); } }
            o
        } else {
            let w = self.weight.data().clone();
            assert_eq!(x.cols, w.cols,
                "Linear: input cols {} != weight cols {}", x.cols, w.cols);
            let wt = w.transpose();
            let mut o = x.matmul(&wt);
            for r in 0..o.rows { for c in 0..o.cols { *o.at_mut(r, c) += b.at(0, c); } }
            o
        };

        let out = TensorNode::leaf(out_data);

        let input_c  = input.clone();
        let weight_c = self.weight.clone();
        let bias_c   = self.bias.clone();
        let out_c    = out.clone();

        out.set_backward(
            Box::new(move || {
                let dout = out_c.grad().clone();    // [T, out] — clone drops the Ref
                let x    = input_c.data().clone();  // [T, in]
                let w    = weight_c.data().clone();  // [out, in]

                // dInput = dOut @ W         : [T, in]
                let dinput = dout.matmul(&w);
                let new_ig = input_c.grad().clone().add(&dinput); // clone releases Ref
                input_c.set_grad(new_ig);

                // dW = dOut.T @ input       : [out, in]  (= dOut.T @ X)
                let dw = dout.transpose().matmul(&x);
                let new_wg = weight_c.grad().clone().add(&dw);
                weight_c.set_grad(new_wg);

                // d_bias = sum_rows(dOut)   : [1, out]
                let db = dout.sum_rows();
                let new_bg = bias_c.grad().clone().add(&db);
                bias_c.set_grad(new_bg);
            }),
            vec![input.clone(), self.weight.clone(), self.bias.clone()],
        );
        out
    }
}

impl Module2 for Linear2 {
    fn parameters(&self) -> Vec<TensorNode> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

// =============================================================================
// LayerNorm2
// =============================================================================

pub struct LayerNorm2 {
    pub gamma: TensorNode,   // [1, d_model]
    pub beta:  TensorNode,   // [1, d_model]
    pub d_model: usize,
}

impl LayerNorm2 {
    pub fn new(d_model: usize) -> Self {
        LayerNorm2 {
            gamma:   TensorNode::leaf(Mat::ones(1, d_model)),
            beta:    TensorNode::leaf(Mat::zeros(1, d_model)),
            d_model,
        }
    }

    /// x: [T, d_model]  →  normalized: [T, d_model]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        x.layer_norm(&self.gamma, &self.beta)
    }
}

impl Module2 for LayerNorm2 {
    fn parameters(&self) -> Vec<TensorNode> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

// =============================================================================
// MLP2 — Feed-Forward Network
// =============================================================================
//
// x → Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)
//
// Now the two matmuls are the dominant operations — and each is a single node
// instead of d_model * 4*d_model scalar multiply-accumulate chains.

pub struct Mlp2 {
    pub fc1: Linear2,
    pub fc2: Linear2,
}

impl Mlp2 {
    pub fn new(d_model: usize, rng: &mut InitRng) -> Self {
        let hidden = 4 * d_model;
        Mlp2 {
            fc1: Linear2::new(d_model, hidden, rng),
            fc2: Linear2::new(hidden, d_model, rng),
        }
    }

    /// x: [T, d_model]  →  output: [T, d_model]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let h = self.fc1.forward(x).gelu();
        self.fc2.forward(&h)
    }
}

impl Module2 for Mlp2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.fc1.parameters();
        p.extend(self.fc2.parameters());
        p
    }
}

// =============================================================================
// RmsNorm2 — RMS Layer Normalization (used by GPT-OSS, LLaMA, Mistral)
// =============================================================================
//
// RMSNorm(x) = x / RMS(x) * gamma
//
// Simpler than LayerNorm: no mean subtraction, no beta term.
// Empirically just as effective, slightly faster.

pub struct RmsNorm2 {
    pub gamma: TensorNode,  // [1, d_model]  — initialized to ones
    pub d_model: usize,
    pub eps: f32,
}

impl RmsNorm2 {
    pub fn new(d_model: usize) -> Self {
        RmsNorm2 {
            gamma: TensorNode::leaf(Mat::ones(1, d_model)),
            d_model,
            eps: 1e-5,
        }
    }

    pub fn new_with_eps(d_model: usize, eps: f32) -> Self {
        RmsNorm2 {
            gamma: TensorNode::leaf(Mat::ones(1, d_model)),
            d_model,
            eps,
        }
    }

    /// x: [T, d_model]  →  normalized: [T, d_model]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        x.rms_norm(&self.gamma, self.eps)
    }

    /// Gemma3 variant: applies `(1 + gamma)` scaling instead of `gamma`.
    /// Used by all norm layers in Gemma3 (gamma is stored as zeros-init, trained offset).
    pub fn forward_gemma3(&self, x: &TensorNode) -> TensorNode {
        x.rms_norm_gemma3(&self.gamma, self.eps)
    }
}

impl Module2 for RmsNorm2 {
    fn parameters(&self) -> Vec<TensorNode> {
        vec![self.gamma.clone()]
    }
}

// =============================================================================
// SwiGluMlp2 — SwiGLU Feed-Forward Network (used by GPT-OSS, LLaMA, PaLM)
// =============================================================================
//
// Standard FFN (GPT-2):  x → Linear → GELU → Linear
//
// SwiGLU FFN (GPT-OSS):
//   gate  = x @ W_gate    [T, intermediate_size]
//   up    = x @ W_up      [T, intermediate_size]
//   hidden = SiLU(gate) * up   (element-wise; SiLU = x*sigmoid(x))
//   out   = hidden @ W_down    [T, d_model]
//
// The "gating" mechanism (SiLU(gate) * up) allows the network to selectively
// suppress or amplify each feature dimension — more expressive than a single
// activation function.
//
// Note: no bias in projections (matches GPT-OSS config: attention_bias=true
// only for attention, not for FFN).

pub struct SwiGluMlp2 {
    pub gate_proj: Linear2,   // d_model → intermediate_size
    pub up_proj:   Linear2,   // d_model → intermediate_size
    pub down_proj: Linear2,   // intermediate_size → d_model
    /// Clamp the gate pre-activation to [-clamp, clamp] before SiLU.
    /// GPT-OSS uses 7.0; set to f32::INFINITY to disable (default).
    pub swiglu_clamp: f32,
}

impl SwiGluMlp2 {
    pub fn new(d_model: usize, intermediate_size: usize, rng: &mut InitRng) -> Self {
        SwiGluMlp2 {
            gate_proj: Linear2::new(d_model, intermediate_size, rng),
            up_proj:   Linear2::new(d_model, intermediate_size, rng),
            down_proj: Linear2::new(intermediate_size, d_model, rng),
            swiglu_clamp: f32::INFINITY,
        }
    }

    pub fn new_with_clamp(d_model: usize, intermediate_size: usize, clamp: f32, rng: &mut InitRng) -> Self {
        let mut mlp = Self::new(d_model, intermediate_size, rng);
        mlp.swiglu_clamp = clamp;
        mlp
    }

    /// x: [T, d_model]  →  output: [T, d_model]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let gate_pre = self.gate_proj.forward(x);
        // Apply clamp before SiLU when swiglu_clamp is finite (GPT-OSS uses 7.0).
        let gate = if self.swiglu_clamp.is_finite() {
            gate_pre.clamp(-self.swiglu_clamp, self.swiglu_clamp).silu()
        } else {
            gate_pre.silu()
        };
        let up     = self.up_proj.forward(x);
        let hidden = gate.mul_elem_node(&up);
        self.down_proj.forward(&hidden)
    }
}

impl Module2 for SwiGluMlp2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.gate_proj.parameters();
        p.extend(self.up_proj.parameters());
        p.extend(self.down_proj.parameters());
        p
    }
}

// =============================================================================
// Dropout2
// =============================================================================
//
// ## What dropout does
//
// During training, each element is independently set to zero with probability p
// (the "drop rate"), and the surviving elements are scaled up by 1/(1-p) so
// the expected value is unchanged.  During inference, nothing is dropped.
//
// ## Why it works
//
// By randomly disabling neurons, dropout:
//   - Forces the network to learn redundant representations
//   - Acts as an ensemble: each mini-batch trains a different sub-network
//   - Reduces co-adaptation of neurons → better generalisation
//
// ## p choices
//
//   p = 0.0  → no dropout (same as not using it)
//   p = 0.1  → mild; good for residual streams in transformers
//   p = 0.5  → aggressive; common in fully-connected classifiers
//
// ## Implementation note
//
// The backward rule is simple:
//   Forward:  mask = Bernoulli(1-p); y = x * mask / (1-p)
//   Backward: dx = dy * mask / (1-p)   (same mask reused)
//
// The mask is generated fresh for each forward call using a simple LCG RNG
// seeded by an atomic counter, giving different masks per step without
// requiring the caller to manage a PRNG state.
//
// ## Usage
//
//   let drop = Dropout2::new(0.1);  // drop 10% of activations
//   let y = drop.forward(&x, true); // training=true
//   let y = drop.forward(&x, false); // inference — identity

pub struct Dropout2 {
    /// Drop probability (0 = no dropout, 1 = drop everything)
    pub p: f32,
    /// LCG seed counter — advanced atomically each forward call
    seed: std::sync::atomic::AtomicU64,
}

impl Dropout2 {
    pub fn new(p: f32) -> Self {
        assert!(p >= 0.0 && p < 1.0, "dropout p must be in [0, 1)");
        Dropout2 { p, seed: std::sync::atomic::AtomicU64::new(12345) }
    }

    /// x: [T, D]  →  output: [T, D]
    ///
    /// training=true:   apply random mask
    /// training=false:  identity (return x unchanged)
    pub fn forward(&self, x: &TensorNode, training: bool) -> TensorNode {
        if !training || self.p == 0.0 {
            return x.clone();
        }

        let keep_prob = 1.0 - self.p;
        let scale = 1.0 / keep_prob;

        // Generate mask using LCG — one call per element
        let seed0 = self.seed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let xd = x.data().clone();
        let mask = Mat::from_fn(xd.rows, xd.cols, |r, c| {
            // Per-element LCG: mix seed with position
            let s = seed0
                .wrapping_add((r * xd.cols + c) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 33) as f32 / (1u64 << 31) as f32;
            if u > self.p { scale } else { 0.0 }
        });

        // out = x * mask
        let out_data = Mat::from_fn(xd.rows, xd.cols, |r, c| xd.at(r, c) * mask.at(r, c));
        let out = TensorNode::leaf(out_data);

        let x_c = x.clone();
        let out_c = out.clone();
        let mask_c = mask;

        out.set_backward(Box::new(move || {
            // dx = dout * mask (same mask as forward)
            let dout = out_c.grad().clone();
            let mut dx = x_c.grad().clone();
            for i in 0..dx.data.len() {
                dx.data[i] += dout.data[i] * mask_c.data[i];
            }
            x_c.set_grad(dx);
            x_c.call_backward_fn();
        }), vec![x.clone()]);

        out
    }
}

impl Module2 for Dropout2 {
    /// Dropout has no learnable parameters.
    fn parameters(&self) -> Vec<TensorNode> { vec![] }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nn::InitRng;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-3 }

    // Numerical gradient helper — perturb element [r,c] of `param` and measure loss change
    fn numerical_grad_param<F: Fn() -> f32>(
        param: &TensorNode,
        r: usize, c: usize,
        loss_fn: &F,
    ) -> f32 {
        let h = 1e-3f32;
        let orig = param.data().at(r, c);
        let mut plus_data = param.data().clone();
        *plus_data.at_mut(r, c) = orig + h;
        param.set_data(plus_data);
        let lp = loss_fn();

        let mut minus_data = param.data().clone();
        *minus_data.at_mut(r, c) = orig - h;
        param.set_data(minus_data);
        let lm = loss_fn();

        // Restore
        let mut orig_data = param.data().clone();
        *orig_data.at_mut(r, c) = orig;
        param.set_data(orig_data);

        (lp - lm) / (2.0 * h)
    }

    // --- Linear2 ---

    #[test]
    fn test_linear2_output_shape() {
        let mut rng = InitRng::new(0);
        let layer = Linear2::new(4, 6, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(3, 4)); // [T=3, in=4]
        let out = layer.forward(&x);
        assert_eq!((out.data().rows, out.data().cols), (3, 6));
    }

    #[test]
    fn test_linear2_bias_zero_init() {
        let mut rng = InitRng::new(0);
        let layer = Linear2::new(4, 3, &mut rng);
        let b = layer.bias.data();
        assert!(b.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_linear2_weight_grad() {
        // Check dW numerically for a single-row input
        let mut rng = InitRng::new(42);
        let layer = Linear2::new(3, 2, &mut rng);
        let x_data = Mat::new(vec![1.0, -0.5, 0.3], 1, 3);

        // Compute analytical gradient
        let x = TensorNode::leaf(x_data.clone());
        let out = layer.fused_linear(&x);
        out.seed_grad_ones();
        out.call_backward_fn();
        let analytical = layer.weight.grad().clone();

        // Compute numerical gradient for weight[0,0]
        let layer2 = Linear2::new(3, 2, &mut rng);
        // Copy weights
        layer2.weight.set_data(layer.weight.data().clone());
        layer2.bias.set_data(layer.bias.data().clone());

        let num = numerical_grad_param(&layer2.weight, 0, 0, &|| {
            let x_n = TensorNode::leaf(x_data.clone());
            let out = layer2.fused_linear(&x_n);
            out.data().data.iter().sum::<f32>()
        });

        assert!(approx(analytical.at(0, 0), num),
            "dW[0,0]: analytical={:.4} numerical={:.4}", analytical.at(0,0), num);
    }

    #[test]
    fn test_linear2_param_count() {
        let mut rng = InitRng::new(0);
        let layer = Linear2::new(4, 3, &mut rng);
        // 2 nodes: weight [3,4] and bias [1,3]
        assert_eq!(layer.parameters().len(), 2);
    }

    // --- LayerNorm2 ---

    #[test]
    fn test_layernorm2_output_mean_zero() {
        let ln = LayerNorm2::new(4);
        let x = TensorNode::leaf(Mat::new(vec![1.,2.,3.,4.], 1, 4));
        let out = ln.forward(&x);
        let mean = out.data().data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "LN mean should be 0, got {}", mean);
    }

    #[test]
    fn test_layernorm2_output_std_one() {
        let ln = LayerNorm2::new(4);
        let x = TensorNode::leaf(Mat::new(vec![1.,2.,3.,4.], 1, 4));
        let out = ln.forward(&x);
        let vals: Vec<f32> = out.data().data.clone();
        let mean = vals.iter().sum::<f32>() / 4.0;
        let std = (vals.iter().map(|&v| (v-mean).powi(2)).sum::<f32>() / 4.0).sqrt();
        assert!((std - 1.0).abs() < 1e-4, "LN std should be 1, got {}", std);
    }

    // --- Dropout2 ---

    #[test]
    fn test_dropout_inference_is_identity() {
        let drop = Dropout2::new(0.5);
        let x = TensorNode::leaf(Mat::from_fn(4, 8, |r, c| (r * 8 + c) as f32));
        let out = drop.forward(&x, false);
        let xd = x.data();
        let od = out.data();
        for r in 0..4 { for c in 0..8 {
            assert_eq!(xd.at(r, c), od.at(r, c), "inference dropout must be identity");
        }}
    }

    #[test]
    fn test_dropout_training_zeros_some_elements() {
        let drop = Dropout2::new(0.5);
        let x = TensorNode::leaf(Mat::from_fn(8, 16, |_, _| 1.0));
        let out = drop.forward(&x, true);
        let zeros = out.data().data.iter().filter(|&&v| v == 0.0).count();
        // With p=0.5 and 128 elements, expect roughly 64 zeros.
        // Accept anything in [20, 108] — very wide to avoid flakiness.
        assert!(zeros > 20 && zeros < 108,
            "expected ~50% zeros with p=0.5, got {}/128", zeros);
    }

    #[test]
    fn test_dropout_zero_p_is_identity() {
        let drop = Dropout2::new(0.0);
        let x = TensorNode::leaf(Mat::from_fn(3, 4, |r, c| (r * 4 + c) as f32 * 0.1));
        let out = drop.forward(&x, true); // even in training, p=0 → identity
        let xd = x.data(); let od = out.data();
        for r in 0..3 { for c in 0..4 {
            assert_eq!(xd.at(r, c), od.at(r, c));
        }}
    }

    #[test]
    fn test_dropout_scales_surviving_elements() {
        // With p=0.5, surviving elements should be scaled by 2.0
        let drop = Dropout2::new(0.5);
        let x = TensorNode::leaf(Mat::from_fn(4, 4, |_, _| 1.0));
        let out = drop.forward(&x, true);
        for &v in &out.data().data {
            assert!(v == 0.0 || (v - 2.0).abs() < 1e-5,
                "dropout output should be 0 or 2 (scale=1/(1-0.5)), got {}", v);
        }
    }

    #[test]
    fn test_dropout_backward_finite() {
        let drop = Dropout2::new(0.3);
        let x = TensorNode::leaf(Mat::from_fn(2, 4, |r, c| (r * 4 + c) as f32 * 0.5));
        let out = drop.forward(&x, true);
        let sum_val: f32 = out.data().data.iter().sum();
        let loss = TensorNode::leaf(Mat::new(vec![sum_val], 1, 1));
        let out_c = out.clone();
        loss.set_backward(Box::new(move || {
            let ones = Mat::ones(out_c.data().rows, out_c.data().cols);
            out_c.set_grad(ones);
            out_c.call_backward_fn();
        }), vec![out]);
        loss.backward();
        assert!(x.grad().data.iter().all(|v| v.is_finite()),
            "dropout backward should produce finite gradients");
    }

    // --- Mlp2 ---

    #[test]
    fn test_mlp2_output_shape() {
        let mut rng = InitRng::new(0);
        let mlp = Mlp2::new(8, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(5, 8)); // [T=5, d=8]
        let out = mlp.forward(&x);
        assert_eq!((out.data().rows, out.data().cols), (5, 8));
    }

    #[test]
    fn test_mlp2_backward_finite() {
        // Check that backward runs and all gradients are finite
        let mut rng = InitRng::new(3);
        let mlp = Mlp2::new(4, &mut rng);
        let x = TensorNode::leaf(Mat::new(
            vec![0.5, -0.3, 1.2, -0.8], 1, 4
        ));

        let out = mlp.forward(&x);

        // Make a scalar loss = sum(out), then backward
        let sum_val = out.data().sum();
        let loss = TensorNode::leaf(Mat::new(vec![sum_val], 1, 1));
        let out_c = out.clone();
        let (or, oc) = { let d = out.data(); (d.rows, d.cols) };
        loss.set_backward(
            Box::new(move || {
                let new_g = out_c.grad().clone().add(&Mat::ones(or, oc));
                out_c.set_grad(new_g);
            }),
            vec![out],
        );
        loss.backward();

        for p in mlp.parameters() {
            let g = p.grad();
            assert!(g.data.iter().all(|x| x.is_finite()),
                "gradient should be finite");
        }
    }
}
