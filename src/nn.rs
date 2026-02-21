/// # Neural Network Layers
///
/// Now that we have autograd (the differentiation engine) and tensors
/// (the data structure), we can build the actual neural network components.
///
/// Every layer in an LLM is composed of three primitives:
///
///   1. **Linear** (also called Dense or Fully Connected)
///      output = input @ weight.T + bias
///      This is a learned linear transformation. All meaning in the network
///      comes from the weight matrices.
///
///   2. **Activation functions**
///      Non-linear functions applied element-wise after linear layers.
///      Without them, stacking linear layers is pointless — they'd collapse
///      into a single linear transformation. Non-linearity gives the network
///      expressive power.
///
///   3. **LayerNorm**
///      Normalizes the activations to have mean≈0 and std≈1, then rescales.
///      Critical for training stability — without it, deep networks diverge.
///
/// ## The module system
///
/// We define a `Module` trait — anything that:
///   - Has parameters (weights, biases)
///   - Can do a forward pass
///   - Can zero all its gradients
///
/// This mirrors PyTorch's `nn.Module`.

use crate::autograd::Value;

// =============================================================================
// Module trait — the interface every layer must implement
// =============================================================================

pub trait Module {
    /// All learnable parameters of this module (weights + biases).
    /// The optimizer will iterate over these to apply gradient updates.
    fn parameters(&self) -> Vec<Value>;

    /// Zero all parameter gradients.
    /// Call this before every backward pass — otherwise gradients accumulate
    /// across batches (which is sometimes intentional, but usually not).
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

// =============================================================================
// Weight initialization
// =============================================================================
//
// The starting values of weights matter enormously.
//
// ## Why not zeros?
//   If all weights are identical, all neurons compute the same thing and
//   learn the same gradient — they never differentiate. This is called
//   the "symmetry problem".
//
// ## Why not large random values?
//   Large weights cause large activations, which cause exploding gradients
//   during backprop. The network diverges immediately.
//
// ## Xavier / Glorot initialization (what we use)
//   Sample from Uniform(-limit, +limit) where limit = sqrt(6 / (fan_in + fan_out))
//   This keeps the variance of activations roughly constant across layers,
//   preventing both explosion and vanishing.
//
// ## Kaiming / He initialization (used with ReLU)
//   Sample from Normal(0, sqrt(2 / fan_in))
//   Compensates for the fact that ReLU kills half the neurons (negative side).
//
// For our transformer we use a simple scaled normal: N(0, 0.02)
// This is the same initialization used in the original GPT-2 paper.

/// Simple LCG for weight initialization — deterministic given a seed.
pub struct InitRng {
    state: u64,
}

impl InitRng {
    pub fn new(seed: u64) -> Self {
        InitRng { state: seed.wrapping_add(1) }
    }

    /// Next random f32 in (0, 1)
    pub fn next_f32(&mut self) -> f32 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use upper 32 bits (better quality than lower bits in LCG)
        let bits = (self.state >> 32) as u32;
        (bits as f32) / (u32::MAX as f32)
    }

    /// Box-Muller transform: two uniform samples → one standard normal sample
    /// N(0,1): most values within [-3, 3], 68% within [-1, 1]
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().max(1e-7); // avoid log(0)
        let u2 = self.next_f32();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f32::consts::PI * u2;
        r * theta.cos()
    }

    /// Sample n values from N(0, std)
    pub fn normal_vec(&mut self, n: usize, std: f32) -> Vec<f32> {
        (0..n).map(|_| self.next_normal() * std).collect()
    }
}

// =============================================================================
// Linear layer
// =============================================================================
//
// The fundamental building block of every neural network.
//
// ## What it computes
//
//   For input x of shape [in_features]:
//     output[j] = sum_i(weight[j,i] * x[i]) + bias[j]
//
//   In matrix form for a batch of inputs [batch, in_features]:
//     output = input @ weight.T + bias    shape: [batch, out_features]
//
// ## Parameters
//   weight: [out_features, in_features]  — the transformation matrix
//   bias:   [out_features]               — one offset per output neuron
//
// ## Why is it called "linear"?
//   output is a linear function of input: f(ax + by) = a·f(x) + b·f(y)
//   The bias makes it "affine" technically, but the name "linear" stuck.
//
// ## In a transformer
//   - Query/Key/Value projections are linear layers
//   - The feed-forward blocks are two linear layers with GELU in between
//   - The output projection (logits) is a linear layer

pub struct Linear {
    /// weight[j][i] = connection strength from input i to output j
    pub weight: Vec<Vec<Value>>, // [out_features][in_features]

    /// bias[j] = offset added to output j
    pub bias: Vec<Value>, // [out_features]

    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, rng: &mut InitRng) -> Self {
        // GPT-2 weight init: N(0, 0.02)
        let std = 0.02f32;

        let weight = (0..out_features)
            .map(|_| {
                rng.normal_vec(in_features, std)
                    .into_iter()
                    .map(Value::new)
                    .collect()
            })
            .collect();

        let bias = (0..out_features)
            .map(|_| Value::new(0.0)) // biases start at zero
            .collect();

        Linear { weight, bias, in_features, out_features }
    }

    /// Forward pass for a single input vector of length `in_features`.
    ///
    /// Returns a vector of length `out_features`.
    ///
    /// For each output neuron j:
    ///   out[j] = dot(weight[j], input) + bias[j]
    ///          = sum_i(weight[j][i] * input[i]) + bias[j]
    pub fn forward(&self, input: &[Value]) -> Vec<Value> {
        assert_eq!(
            input.len(),
            self.in_features,
            "Linear forward: input length {} != in_features {}",
            input.len(),
            self.in_features
        );

        (0..self.out_features)
            .map(|j| {
                // dot product: sum_i(w[j][i] * x[i])
                let dot = input
                    .iter()
                    .zip(self.weight[j].iter())
                    .map(|(x, w)| w.mul(x))
                    .reduce(|acc, v| acc.add(&v))
                    .unwrap();

                dot.add(&self.bias[j])
            })
            .collect()
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Value> {
        let mut params = Vec::new();
        for row in &self.weight {
            params.extend(row.iter().cloned());
        }
        params.extend(self.bias.iter().cloned());
        params
    }
}

// =============================================================================
// Activation functions
// =============================================================================
//
// Linear layers stack into a useless identity without non-linearities.
// Activations introduce the non-linearity that lets the network approximate
// any function (the Universal Approximation Theorem).
//
// ## ReLU — Rectified Linear Unit
//   f(x) = max(0, x)
//   Simple, fast, but can cause "dead neurons" (gradient = 0 forever if x < 0).
//   Used in older models (ResNet, early transformers).
//
// ## GELU — Gaussian Error Linear Unit
//   f(x) = x · Φ(x)   where Φ is the standard normal CDF
//   Approximation: x · σ(1.702 · x)   (sigmoid-based, what we implement)
//
//   GELU is the standard in modern transformers (BERT, GPT-2, GPT-3, GPT-4).
//   Unlike ReLU it never fully gates the gradient — the transition is smooth,
//   which helps gradients flow and avoids dead neurons.
//
// ## Why does the shape of the activation matter?
//   ReLU: hard gate at 0, gradient is exactly 0 or 1
//   GELU: soft gate, gradient varies smoothly — better for deep nets

/// Apply ReLU to every element of a vector.
pub fn relu(xs: &[Value]) -> Vec<Value> {
    xs.iter().map(|x| x.relu()).collect()
}

/// Apply GELU to every element of a vector.
///
/// GELU(x) = x * sigmoid(1.702 * x)
///
/// We build this from our existing autograd primitives:
///   sigmoid(z) = 1 / (1 + exp(-z)) = exp(z) / (1 + exp(z))
///   GELU(x) = x * sigmoid(1.702 * x)
///
/// The constant 1.702 comes from approximating Φ(x) with a sigmoid.
/// OpenAI uses this exact approximation in GPT-2.
pub fn gelu(xs: &[Value]) -> Vec<Value> {
    xs.iter()
        .map(|x| {
            // z = 1.702 * x
            let scale = Value::new(1.702);
            let z = scale.mul(x);

            // sigmoid(z) = 1 / (1 + exp(-z))
            let neg_z = z.neg();
            let exp_neg_z = neg_z.exp();
            let one = Value::new(1.0);
            let denom = one.add(&exp_neg_z);
            let sigmoid_z = one.div(&denom);

            // gelu(x) = x * sigmoid(1.702 * x)
            x.mul(&sigmoid_z)
        })
        .collect()
}

/// Apply softmax to a vector of logits, returning a probability distribution.
///
/// softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
///
/// We subtract max(x) for numerical stability — this doesn't change the result
/// mathematically but prevents overflow in exp().
///
/// This is used in:
///   1. Attention: converting raw scores to attention weights
///   2. Output head: converting final hidden state to token probabilities
pub fn softmax(xs: &[Value]) -> Vec<Value> {
    // Find max value (as f32, not tracked — just for numerical stability)
    let max_val = xs.iter()
        .map(|x| x.val())
        .fold(f32::NEG_INFINITY, f32::max);

    // Shift and exponentiate
    let max_node = Value::new(max_val);
    let exps: Vec<Value> = xs.iter()
        .map(|x| x.sub(&max_node).exp())
        .collect();

    // Sum of exps
    let sum = exps.iter()
        .cloned()
        .reduce(|acc, v| acc.add(&v))
        .unwrap();

    // Normalize
    exps.iter().map(|e| e.div(&sum)).collect()
}

// =============================================================================
// LayerNorm — Layer Normalization
// =============================================================================
//
// ## The problem it solves
//
// As activations flow through many layers, their scale can drift wildly —
// either exploding (huge values, unstable gradients) or vanishing (near-zero
// values, no learning signal). This makes deep networks very hard to train.
//
// ## What LayerNorm does
//
// For each token's feature vector x of dimension d:
//
//   1. Compute mean:  μ = (1/d) * sum(x)
//   2. Compute std:   σ = sqrt((1/d) * sum((x - μ)²) + ε)
//   3. Normalize:     x̂ = (x - μ) / σ
//   4. Scale+shift:   y = γ * x̂ + β
//
// γ (gamma) and β (beta) are learnable parameters — the network can choose
// how much to scale/shift after normalization. This lets the network "undo"
// the normalization if needed.
//
// ε (epsilon) is a tiny constant (~1e-5) that prevents division by zero
// when the variance is very small.
//
// ## LayerNorm vs BatchNorm
//
// BatchNorm normalizes across the batch dimension (needs a large batch).
// LayerNorm normalizes across the feature dimension (works for any batch size,
// even batch size 1 at inference). This makes it ideal for language models
// where sequence lengths vary.
//
// ## Where it appears in a transformer
//
//   x → LayerNorm → Attention → + → LayerNorm → FFN → +
//         (pre-norm)                    (pre-norm)
//
// Modern transformers use "pre-norm" (normalize before each sub-block).
// The original "Attention is All You Need" used post-norm; pre-norm
// trains more stably and is what GPT-2 onward uses.

pub struct LayerNorm {
    /// Learnable scale parameter γ (gamma), one per feature dimension.
    /// Initialized to 1 — starts as pure normalization.
    pub gamma: Vec<Value>,

    /// Learnable shift parameter β (beta), one per feature dimension.
    /// Initialized to 0 — starts with no shift.
    pub beta: Vec<Value>,

    /// Small constant for numerical stability (prevents division by zero).
    eps: f32,

    pub d_model: usize,
}

impl LayerNorm {
    pub fn new(d_model: usize) -> Self {
        LayerNorm {
            gamma: (0..d_model).map(|_| Value::new(1.0)).collect(),
            beta:  (0..d_model).map(|_| Value::new(0.0)).collect(),
            eps: 1e-5,
            d_model,
        }
    }

    /// Normalize a single feature vector of length `d_model`.
    pub fn forward(&self, xs: &[Value]) -> Vec<Value> {
        assert_eq!(xs.len(), self.d_model);
        let n = self.d_model as f32;

        // Step 1: mean = (1/d) * sum(x)
        let sum = xs.iter()
            .cloned()
            .reduce(|acc, v| acc.add(&v))
            .unwrap();
        let mean = sum.mul(&Value::new(1.0 / n));

        // Step 2: variance = (1/d) * sum((x - mean)²)
        let var_sum = xs.iter()
            .map(|x| {
                let diff = x.sub(&mean);
                diff.mul(&diff.clone()) // (x - mean)²
            })
            .reduce(|acc, v| acc.add(&v))
            .unwrap();
        let variance = var_sum.mul(&Value::new(1.0 / n));

        // Step 3: std = sqrt(variance + eps)
        let std = variance.add(&Value::new(self.eps)).pow(0.5);

        // Step 4: normalize, then scale and shift
        xs.iter()
            .zip(self.gamma.iter())
            .zip(self.beta.iter())
            .map(|((x, gamma), beta)| {
                // x_hat = (x - mean) / std
                let x_hat = x.sub(&mean).div(&std);
                // y = gamma * x_hat + beta
                gamma.mul(&x_hat).add(beta)
            })
            .collect()
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Value> {
        let mut params = self.gamma.clone();
        params.extend(self.beta.iter().cloned());
        params
    }
}

// =============================================================================
// MLP — Multi-Layer Perceptron (the Feed-Forward block in a transformer)
// =============================================================================
//
// Every transformer layer has two sub-components:
//   1. Multi-Head Self-Attention   (we build this in Phase 3)
//   2. Feed-Forward Network (FFN) = this MLP
//
// The FFN is:
//   x → Linear(d_model → 4*d_model) → GELU → Linear(4*d_model → d_model)
//
// The expansion factor 4 is a hyperparameter chosen empirically — it gives
// the network enough "workspace" to perform complex transformations before
// projecting back down. GPT-2, GPT-3, LLaMA all use 4x expansion.
//
// Why do we need FFN if we have attention?
//   Attention is great at routing information between positions.
//   FFN is great at transforming the information at each position.
//   Together they give: communication + computation.

pub struct Mlp {
    pub fc1: Linear,   // d_model → 4 * d_model
    pub fc2: Linear,   // 4 * d_model → d_model
}

impl Mlp {
    pub fn new(d_model: usize, rng: &mut InitRng) -> Self {
        let hidden = 4 * d_model;
        Mlp {
            fc1: Linear::new(d_model, hidden, rng),
            fc2: Linear::new(hidden, d_model, rng),
        }
    }

    /// x → fc1 → GELU → fc2 → output (same shape as input)
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let h = gelu(&self.fc1.forward(x));
        self.fc2.forward(&h)
    }
}

impl Module for Mlp {
    fn parameters(&self) -> Vec<Value> {
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

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    // --- InitRng ---

    #[test]
    fn test_rng_normal_mean_std() {
        // 10_000 samples from N(0,1) should have mean≈0 and std≈1
        let mut rng = InitRng::new(42);
        let samples: Vec<f32> = (0..10_000).map(|_| rng.next_normal()).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let std = (samples.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / samples.len() as f32)
            .sqrt();
        assert!(mean.abs() < 0.05, "mean should be ≈0, got {:.4}", mean);
        assert!((std - 1.0).abs() < 0.05, "std should be ≈1, got {:.4}", std);
    }

    // --- Linear ---

    #[test]
    fn test_linear_output_shape() {
        let mut rng = InitRng::new(0);
        let layer = Linear::new(3, 5, &mut rng);
        let input: Vec<Value> = vec![1.0, 2.0, 3.0].into_iter().map(Value::new).collect();
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_linear_bias_starts_zero() {
        // All biases should start at 0
        let mut rng = InitRng::new(0);
        let layer = Linear::new(4, 3, &mut rng);
        for b in &layer.bias {
            assert!(approx(b.val(), 0.0));
        }
    }

    #[test]
    fn test_linear_backward() {
        // Simple 1-in, 1-out linear: output = w * x + b
        // dL/dw = x, dL/db = 1
        let mut rng = InitRng::new(1);
        let layer = Linear::new(1, 1, &mut rng);

        // Override weight and bias for predictable test
        let w = Value::new(3.0);
        let b = Value::new(0.5);
        let x = Value::new(2.0);

        let out = w.mul(&x).add(&b); // 3*2 + 0.5 = 6.5
        out.backward();

        assert!(approx(w.grad(), 2.0), "dL/dw = x = 2, got {}", w.grad());
        assert!(approx(b.grad(), 1.0), "dL/db = 1, got {}", b.grad());
        assert!(approx(x.grad(), 3.0), "dL/dx = w = 3, got {}", x.grad());
    }

    #[test]
    fn test_linear_param_count() {
        let mut rng = InitRng::new(0);
        let layer = Linear::new(4, 3, &mut rng);
        // weight: 3 * 4 = 12, bias: 3 → total 15
        assert_eq!(layer.parameters().len(), 15);
    }

    // --- Activations ---

    #[test]
    fn test_relu_values() {
        let xs: Vec<Value> = vec![-2.0, 0.0, 3.0].into_iter().map(Value::new).collect();
        let out = relu(&xs);
        assert!(approx(out[0].val(), 0.0));
        assert!(approx(out[1].val(), 0.0));
        assert!(approx(out[2].val(), 3.0));
    }

    #[test]
    fn test_gelu_positive_input() {
        // GELU(x) ≈ x for large positive x (sigmoid(1.702*x) → 1)
        let xs: Vec<Value> = vec![5.0].into_iter().map(Value::new).collect();
        let out = gelu(&xs);
        assert!(
            (out[0].val() - 5.0).abs() < 0.01,
            "GELU(5) ≈ 5, got {}",
            out[0].val()
        );
    }

    #[test]
    fn test_gelu_near_zero() {
        // GELU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        let xs: Vec<Value> = vec![0.0].into_iter().map(Value::new).collect();
        let out = gelu(&xs);
        assert!(approx(out[0].val(), 0.0));
    }

    #[test]
    fn test_gelu_negative_small() {
        // GELU(-1) should be a small negative number (not zero like ReLU)
        let xs: Vec<Value> = vec![-1.0].into_iter().map(Value::new).collect();
        let out = gelu(&xs);
        // GELU(-1) ≈ -0.159
        assert!(out[0].val() < 0.0, "GELU(-1) should be negative, got {}", out[0].val());
        assert!(out[0].val() > -0.5, "GELU(-1) should be small negative");
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let xs: Vec<Value> = vec![1.0, 2.0, 3.0].into_iter().map(Value::new).collect();
        let probs = softmax(&xs);
        let total: f32 = probs.iter().map(|v| v.val()).sum();
        assert!(approx(total, 1.0), "softmax should sum to 1, got {}", total);
    }

    #[test]
    fn test_softmax_max_wins() {
        // The largest input should produce the largest probability
        let xs: Vec<Value> = vec![1.0, 5.0, 2.0].into_iter().map(Value::new).collect();
        let probs = softmax(&xs);
        let vals: Vec<f32> = probs.iter().map(|v| v.val()).collect();
        assert!(vals[1] > vals[0] && vals[1] > vals[2]);
    }

    // --- LayerNorm ---

    #[test]
    fn test_layernorm_output_mean_zero() {
        let ln = LayerNorm::new(4);
        let xs: Vec<Value> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(Value::new).collect();
        let out = ln.forward(&xs);
        let mean = out.iter().map(|v| v.val()).sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "LayerNorm output should have mean≈0, got {}", mean);
    }

    #[test]
    fn test_layernorm_output_std_one() {
        let ln = LayerNorm::new(4);
        let xs: Vec<Value> = vec![1.0, 2.0, 3.0, 4.0].into_iter().map(Value::new).collect();
        let out = ln.forward(&xs);
        let mean = out.iter().map(|v| v.val()).sum::<f32>() / 4.0;
        let std = (out.iter().map(|v| (v.val() - mean).powi(2)).sum::<f32>() / 4.0).sqrt();
        assert!((std - 1.0).abs() < 1e-4, "LayerNorm output should have std≈1, got {}", std);
    }

    #[test]
    fn test_layernorm_constant_input() {
        // All identical inputs → variance = 0, output should all be beta = 0
        let ln = LayerNorm::new(3);
        let xs: Vec<Value> = vec![7.0, 7.0, 7.0].into_iter().map(Value::new).collect();
        let out = ln.forward(&xs);
        // x_hat = (7 - 7) / sqrt(0 + eps) = 0, y = gamma*0 + beta = 0
        for v in &out {
            assert!(v.val().abs() < 1e-3, "Expected ≈0, got {}", v.val());
        }
    }

    #[test]
    fn test_layernorm_param_count() {
        let ln = LayerNorm::new(8);
        // gamma (8) + beta (8) = 16
        assert_eq!(ln.parameters().len(), 16);
    }

    // --- MLP ---

    #[test]
    fn test_mlp_output_shape() {
        let mut rng = InitRng::new(0);
        let mlp = Mlp::new(8, &mut rng);
        let x: Vec<Value> = (0..8).map(|i| Value::new(i as f32 * 0.1)).collect();
        let out = mlp.forward(&x);
        assert_eq!(out.len(), 8, "MLP output should match input dimension");
    }

    #[test]
    fn test_mlp_param_count() {
        let mut rng = InitRng::new(0);
        let mlp = Mlp::new(4, &mut rng);
        // fc1: 4->16: weight 64 + bias 16 = 80
        // fc2: 16->4: weight 64 + bias 4  = 68
        // total = 148
        assert_eq!(mlp.parameters().len(), 148);
    }

    #[test]
    fn test_mlp_backward_runs() {
        // Check that backward doesn't panic and produces finite gradients
        let mut rng = InitRng::new(7);
        let mlp = Mlp::new(4, &mut rng);
        let x: Vec<Value> = vec![0.5, -0.3, 1.2, -0.8].into_iter().map(Value::new).collect();
        let out = mlp.forward(&x);
        // Sum all outputs as a scalar loss
        let loss = out.iter().cloned().reduce(|a, b| a.add(&b)).unwrap();
        loss.backward();
        for p in mlp.parameters() {
            assert!(p.grad().is_finite(), "gradient should be finite, got {}", p.grad());
        }
    }
}
