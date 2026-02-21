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

use crate::autograd2::{TensorNode, Mat};
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
        }
    }

    /// input: [T, in_features]  →  output: [T, out_features]
    pub fn forward(&self, input: &TensorNode) -> TensorNode {
        self.fused_linear(input)
    }

    /// Fused linear: output = input @ weight.T + bias
    /// Keeps weight directly in the graph so its gradient is tracked.
    fn fused_linear(&self, input: &TensorNode) -> TensorNode {
        // We implement linear as a single custom node that tracks both
        // input and weight — avoiding a separate transpose node.
        let x = input.data().clone();
        let w = self.weight.data().clone();
        let b = self.bias.data().clone();

        assert_eq!(x.cols, w.cols,
            "Linear: input cols {} != weight cols {}", x.cols, w.cols);

        // out = x @ w.T + b  : [T, out]
        let out_data = {
            let wt = w.transpose();
            let mut o = x.matmul(&wt);
            // add bias broadcast
            for r in 0..o.rows {
                for c in 0..o.cols {
                    *o.at_mut(r, c) += b.at(0, c);
                }
            }
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
