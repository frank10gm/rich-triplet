/// # Tensor-Level Automatic Differentiation
///
/// ## Why the scalar engine is slow
///
/// In `autograd.rs`, every single number — every element of every weight matrix,
/// every activation — is its own `Value` node. A Linear layer with shape [64, 64]
/// creates 4096 nodes just for the weights, plus thousands more for intermediate
/// computations. The backward pass must visit every one.
///
/// For our tiny model with d_model=32, a single forward pass creates roughly:
///   - Embedding lookup:    32 * 16 = 512 nodes
///   - Per block (×2):
///       LayerNorm:         32 * 16 * ~8 ops = ~4096 nodes
///       Q,K,V projections: 32*32*16 * 3   = ~49,152 nodes
///       Attention scores:  16*16*32        = ~8,192 nodes
///       MLP:               32*128*16 * 2   = ~131,072 nodes
///   Total: ~400,000+ nodes per forward pass
///
/// ## The tensor engine: one node per operation
///
/// Instead of one node per number, we have one node per *operation*.
/// A matmul of [16,32] × [32,32] is a single node — not 16*32*32 = 16,384 nodes.
/// The entire model graph shrinks to ~50 nodes:
///
///   embed → ln → q_proj → k_proj → v_proj → scores → softmax →
///   attn_out → o_proj → residual → ln → fc1 → gelu → fc2 → residual →
///   ln_final → lm_head → loss
///
/// ## What changes in the backward rules
///
/// Scalar rule:        d(a*b)/da = b             (a single number)
/// Tensor rule:        d(A@B)/dA = dOut @ B.T    (a whole matrix)
///
/// The math is the same chain rule — we're just doing it on entire matrices
/// at once instead of element by element. This is called a
/// "vector-Jacobian product" (VJP) or "cotangent" in JAX terminology.
///
/// ## The key backward rules we need
///
/// ### matmul: C = A @ B  (shapes [M,K] × [K,N] → [M,N])
///   dA = dC @ B.T       (shape [M,K])
///   dB = A.T @ dC       (shape [K,N])
///
///   Intuition: dA[i,k] = sum_j dC[i,j] * B[k,j]
///              = (dC @ B.T)[i,k]  ← one matmul instead of M*K*N scalar ops
///
/// ### add: C = A + B  (element-wise, same shape)
///   dA = dC
///   dB = dC
///
/// ### element-wise mul: C = A * B
///   dA = dC * B
///   dB = dC * A
///
/// ### softmax: S = softmax(X)  (applied row-wise to [T, V])
///   dX[i] = S[i] * (dS[i] - dot(dS[i], S[i]))
///   (Jacobian of softmax contracted with upstream gradient)
///
/// ### layernorm: Y = (X - μ) / σ * γ + β
///   Closed-form gradient using stored μ, σ, X̂ from forward pass.
///   dγ = sum(dY * X̂, dim=0)
///   dβ = sum(dY, dim=0)
///   dX = (1/σ) * (dY*γ - mean(dY*γ) - X̂*mean(dY*γ*X̂))
///
/// ### cross-entropy loss: L = mean(-log(softmax(logits)[targets]))
///   Combined with softmax for numerical stability:
///   d(logits)[t, v] = (softmax(logits)[t, v] - one_hot(targets[t], v)) / T

use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;

// =============================================================================
// The Tensor type — re-exported from tensor.rs but with grad operations added
// =============================================================================
//
// We reuse our existing Tensor for data storage, but add all the gradient
// math here as free functions operating on flat Vec<f32>.

/// A flat 2-D matrix stored in row-major order.
/// We use this instead of the full Tensor struct to keep things focused.
#[derive(Clone, Debug)]
pub struct Mat {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    pub fn new(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols,
            "Mat::new: data len {} != rows*cols {}*{}={}", data.len(), rows, cols, rows*cols);
        Mat { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Mat { data: vec![1.0; rows * cols], rows, cols }
    }

    pub fn from_fn<F: Fn(usize, usize) -> f32>(rows: usize, cols: usize, f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows { for c in 0..cols { data.push(f(r, c)); } }
        Mat { data, rows, cols }
    }

    #[inline] pub fn at(&self, r: usize, c: usize) -> f32 { self.data[r * self.cols + c] }
    #[inline] pub fn at_mut(&mut self, r: usize, c: usize) -> &mut f32 { &mut self.data[r * self.cols + c] }

    pub fn numel(&self) -> usize { self.rows * self.cols }

    // -------------------------------------------------------------------------
    // Core linear algebra — these are the hot paths
    // -------------------------------------------------------------------------

    /// C = A @ B    [M,K] × [K,N] → [M,N]
    ///
    /// This single function replaces M*N*K scalar multiply-add operations
    /// in the old engine. On modern CPUs the compiler can auto-vectorize the
    /// inner loop with SIMD instructions.
    pub fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows,
            "matmul shape mismatch: [{},{}] × [{},{}]", self.rows, self.cols, b.rows, b.cols);
        let (m, k, n) = (self.rows, self.cols, b.cols);
        let mut out = Mat::zeros(m, n);
        for i in 0..m {
            for p in 0..k {
                let a_ip = self.at(i, p);
                for j in 0..n {
                    *out.at_mut(i, j) += a_ip * b.at(p, j);
                }
            }
        }
        out
    }

    /// A.T — transpose: [M,N] → [N,M]
    pub fn transpose(&self) -> Mat {
        Mat::from_fn(self.cols, self.rows, |r, c| self.at(c, r))
    }

    /// Element-wise addition (same shape).
    pub fn add(&self, other: &Mat) -> Mat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Mat::new(self.data.iter().zip(&other.data).map(|(a,b)| a+b).collect(),
                 self.rows, self.cols)
    }

    /// In-place element-wise addition: self += other
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) { *a += b; }
    }

    /// Element-wise multiplication (same shape).
    pub fn mul_elem(&self, other: &Mat) -> Mat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Mat::new(self.data.iter().zip(&other.data).map(|(a,b)| a*b).collect(),
                 self.rows, self.cols)
    }

    /// Scale every element by a scalar.
    pub fn scale(&self, s: f32) -> Mat {
        Mat::new(self.data.iter().map(|x| x * s).collect(), self.rows, self.cols)
    }

    /// Element-wise map.
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Mat {
        Mat::new(self.data.iter().map(|&x| f(x)).collect(), self.rows, self.cols)
    }

    /// Sum all elements.
    pub fn sum(&self) -> f32 { self.data.iter().sum() }

    /// Sum along rows → shape [1, cols].
    /// out[c] = sum_r self[r,c]
    pub fn sum_rows(&self) -> Mat {
        let mut out = Mat::zeros(1, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                *out.at_mut(0, c) += self.at(r, c);
            }
        }
        out
    }

    /// Row-wise mean → shape [rows, 1].
    pub fn row_mean(&self) -> Mat {
        let inv_n = 1.0 / self.cols as f32;
        Mat::from_fn(self.rows, 1, |r, _| {
            self.data[r * self.cols..(r+1) * self.cols].iter().sum::<f32>() * inv_n
        })
    }

    /// Broadcast-add a [rows,1] column vector to each column of self.
    pub fn add_col_broadcast(&self, col: &Mat) -> Mat {
        assert_eq!(col.cols, 1);
        assert_eq!(col.rows, self.rows);
        Mat::from_fn(self.rows, self.cols, |r, c| self.at(r, c) + col.at(r, 0))
    }

    /// Broadcast-subtract a [rows,1] column vector from each column.
    pub fn sub_col_broadcast(&self, col: &Mat) -> Mat {
        assert_eq!(col.cols, 1);
        assert_eq!(col.rows, self.rows);
        Mat::from_fn(self.rows, self.cols, |r, c| self.at(r, c) - col.at(r, 0))
    }

    /// Broadcast-multiply element-wise by a [rows,1] column vector.
    pub fn mul_col_broadcast(&self, col: &Mat) -> Mat {
        assert_eq!(col.cols, 1);
        assert_eq!(col.rows, self.rows);
        Mat::from_fn(self.rows, self.cols, |r, c| self.at(r, c) * col.at(r, 0))
    }

    /// Broadcast-multiply element-wise by a [1,cols] row vector.
    pub fn mul_row_broadcast(&self, row: &Mat) -> Mat {
        assert_eq!(row.rows, 1);
        assert_eq!(row.cols, self.cols);
        Mat::from_fn(self.rows, self.cols, |r, c| self.at(r, c) * row.at(0, c))
    }

    /// L2 norm of all elements: sqrt(sum(x^2))
    pub fn norm(&self) -> f32 {
        self.data.iter().map(|x| x * x).sum::<f32>().sqrt()
    }
}

// =============================================================================
// TensorNode — one node in the computation graph, holding a full matrix
// =============================================================================

struct NodeData {
    /// The forward-pass value (the matrix this node computed)
    pub data: Mat,

    /// The gradient of the loss w.r.t. this matrix.
    /// Same shape as `data`. Accumulated during backward.
    pub grad: Mat,

    /// How to propagate `grad` back to this node's inputs.
    /// None for leaf nodes (parameters, inputs).
    backward_fn: Option<Box<dyn Fn()>>,

    /// Input nodes (used to build topological order).
    prev: Vec<TensorNode>,
}

/// A reference-counted, interior-mutable node in the tensor computation graph.
///
/// The Rc<RefCell<>> pattern is the same as scalar autograd — it lets nodes
/// reference each other and accumulate gradients mutably through shared refs.
#[derive(Clone)]
pub struct TensorNode(Rc<RefCell<NodeData>>);

impl TensorNode {
    /// Create a leaf node (a parameter or input — no backward function).
    pub fn leaf(data: Mat) -> Self {
        let rows = data.rows;
        let cols = data.cols;
        TensorNode(Rc::new(RefCell::new(NodeData {
            grad: Mat::zeros(rows, cols),
            data,
            backward_fn: None,
            prev: vec![],
        })))
    }

    /// Read the forward data.
    pub fn data(&self) -> std::cell::Ref<Mat> {
        std::cell::Ref::map(self.0.borrow(), |n| &n.data)
    }

    /// Read the gradient.
    pub fn grad(&self) -> std::cell::Ref<Mat> {
        std::cell::Ref::map(self.0.borrow(), |n| &n.grad)
    }

    /// Zero the gradient tensor (call before each backward pass).
    pub fn zero_grad(&self) {
        let mut inner = self.0.borrow_mut();
        let (r, c) = (inner.grad.rows, inner.grad.cols);
        inner.grad = Mat::zeros(r, c);
    }

    /// Directly set the data (used by optimizer to apply updates).
    pub fn set_data(&self, data: Mat) {
        self.0.borrow_mut().data = data;
    }

    /// Directly set gradient values (used for gradient clipping).
    pub fn set_grad(&self, grad: Mat) {
        self.0.borrow_mut().grad = grad;
    }

    /// Register a backward function and predecessor nodes on an existing leaf.
    ///
    /// Used by layers in nn2 that need to build custom fused backward nodes
    /// without accessing the private inner field directly.
    pub fn set_backward(&self, f: Box<dyn Fn()>, prev: Vec<TensorNode>) {
        let mut inner = self.0.borrow_mut();
        inner.backward_fn = Some(f);
        inner.prev = prev;
    }

    /// Read gradient via the unsafe fn-pointer trick (same as backward()),
    /// so callers outside this module can call a node's backward_fn without
    /// holding the RefCell borrow.
    pub fn call_backward_fn(&self) {
        let fn_ptr = {
            let inner = self.0.borrow();
            inner.backward_fn.as_ref().map(|f| unsafe {
                &*(f.as_ref() as *const dyn Fn())
            })
        };
        if let Some(f) = fn_ptr { f(); }
    }

    /// Seed this node's gradient (set grad = ones of same shape).
    /// Used in tests to kick off a manual backward.
    pub fn seed_grad_ones(&self) {
        let mut inner = self.0.borrow_mut();
        let (r, c) = (inner.data.rows, inner.data.cols);
        inner.grad = Mat::ones(r, c);
    }

    // =========================================================================
    // Operations — each returns a new node and registers the backward rule
    // =========================================================================
    //
    // Pattern for every op:
    //   1. Compute the output matrix (forward)
    //   2. Create the output node
    //   3. Register backward_fn that accumulates gradient into input nodes
    //   4. Record inputs in prev for topological sort

    /// C = A @ B    Matrix multiply: [M,K] × [K,N] → [M,N]
    ///
    /// Backward:
    ///   dA += dC @ B.T    [M,K] = [M,N] @ [N,K]
    ///   dB += A.T @ dC    [K,N] = [K,M] @ [M,N]
    ///
    /// Why these formulas?
    ///   C[i,j] = sum_k A[i,k]*B[k,j]
    ///   dL/dA[i,k] = sum_j dL/dC[i,j] * B[k,j] = (dC @ B.T)[i,k]
    ///   dL/dB[k,j] = sum_i dL/dC[i,j] * A[i,k] = (A.T @ dC)[k,j]
    pub fn matmul(&self, b: &TensorNode) -> TensorNode {
        let out_data = self.data().matmul(&b.data());
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let b_c = b.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let b_data = b_c.0.borrow().data.clone();
            let a_data = self_c.0.borrow().data.clone();

            // dA += dC @ B.T
            let da = dout.matmul(&b_data.transpose());
            self_c.0.borrow_mut().grad.add_assign(&da);

            // dB += A.T @ dC
            let db = a_data.transpose().matmul(&dout);
            b_c.0.borrow_mut().grad.add_assign(&db);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), b.clone()];
        out
    }

    /// C = A + B    Element-wise add (same shape).
    ///
    /// Backward: dA += dC,  dB += dC
    /// (gradient passes through unchanged to both inputs)
    pub fn add(&self, b: &TensorNode) -> TensorNode {
        let out_data = self.data().add(&b.data());
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let b_c = b.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            self_c.0.borrow_mut().grad.add_assign(&dout);
            b_c.0.borrow_mut().grad.add_assign(&dout);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), b.clone()];
        out
    }

    /// C = A + bias_row   Broadcast-add a [1, cols] bias to every row of A.
    ///
    /// This is the bias addition in a Linear layer.
    ///
    /// Backward:
    ///   dA    += dC                     (same shape, pass through)
    ///   d_bias += sum_rows(dC)          (sum over rows — bias is shared across rows)
    ///
    /// Why sum over rows for d_bias?
    ///   bias[j] is added to every row of A, so it participates in every row's
    ///   contribution to the loss. The total gradient is the sum of all those
    ///   contributions.
    pub fn add_bias(&self, bias: &TensorNode) -> TensorNode {
        assert_eq!(bias.data().rows, 1);
        assert_eq!(bias.data().cols, self.data().cols);

        // Broadcast: add bias row to every row of self
        let a = self.data().clone();
        let b = bias.data().clone();
        let out_data = Mat::from_fn(a.rows, a.cols, |r, c| a.at(r, c) + b.at(0, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let bias_c = bias.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            // dA += dC
            self_c.0.borrow_mut().grad.add_assign(&dout);
            // d_bias += sum over rows of dC → shape [1, cols]
            let db = dout.sum_rows();
            bias_c.0.borrow_mut().grad.add_assign(&db);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), bias.clone()];
        out
    }

    /// Element-wise GELU activation applied to every element.
    ///
    /// GELU(x) = x * sigmoid(1.702 * x)
    ///
    /// Backward: d(GELU(x))/dx via the chain rule.
    /// We store the forward input for use in backward.
    ///
    /// d/dx [x * σ(1.702x)]
    ///   = σ(1.702x) + x * σ(1.702x) * (1 - σ(1.702x)) * 1.702
    ///   = σ(z) + x * σ(z) * (1-σ(z)) * 1.702    where z = 1.702*x
    pub fn gelu(&self) -> TensorNode {
        let x = self.data().clone();

        // Precompute sigmoid values for reuse in backward
        let sigmoid_vals = Mat::from_fn(x.rows, x.cols, |r, c| {
            let z = 1.702 * x.at(r, c);
            1.0 / (1.0 + (-z).exp())
        });

        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| {
            x.at(r, c) * sigmoid_vals.at(r, c)
        });
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();

            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let xv = x_data.at(r, c);
                let z  = 1.702 * xv;
                let s  = 1.0 / (1.0 + (-z).exp()); // sigmoid(z)
                let dgelu_dx = s + xv * s * (1.0 - s) * 1.702;
                dout.at(r, c) * dgelu_dx
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Row-wise softmax: each row of [T, V] is converted to a probability distribution.
    ///
    /// S[t, v] = exp(X[t,v] - max_v) / sum_v exp(X[t,v] - max_v)
    ///
    /// Backward (Jacobian-vector product of softmax):
    ///
    ///   For each row t:
    ///     dX[t] = S[t] * (dS[t] - dot(dS[t], S[t]))
    ///
    ///   Derivation:
    ///     ∂S[t,i]/∂X[t,j] = S[t,i] * (δ_ij - S[t,j])  (Jacobian of softmax)
    ///     dL/dX[t,j] = sum_i dL/dS[t,i] * ∂S[t,i]/∂X[t,j]
    ///                = sum_i dS[t,i] * S[t,i] * (δ_ij - S[t,j])
    ///                = dS[t,j]*S[t,j] - S[t,j] * sum_i(dS[t,i]*S[t,i])
    ///                = S[t,j] * (dS[t,j] - dot(dS[t], S[t]))
    pub fn softmax(&self) -> TensorNode {
        let x = self.data().clone();
        let t = x.rows;
        let v = x.cols;

        // Numerically stable: subtract row max before exp
        let mut s_data = Mat::zeros(t, v);
        for r in 0..t {
            let row_max = (0..v).map(|c| x.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
            let mut row_sum = 0.0f32;
            for c in 0..v {
                let e = (x.at(r, c) - row_max).exp();
                *s_data.at_mut(r, c) = e;
                row_sum += e;
            }
            for c in 0..v {
                *s_data.at_mut(r, c) /= row_sum;
            }
        }

        let out = TensorNode::leaf(s_data);
        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let ds = out_c.0.borrow().grad.clone();   // upstream gradient
            let s  = out_c.0.borrow().data.clone();   // forward softmax values
            let (t, v) = (s.rows, s.cols);

            let mut dx = Mat::zeros(t, v);
            for r in 0..t {
                // dot(dS[r], S[r])  — scalar
                let dot: f32 = (0..v).map(|c| ds.at(r, c) * s.at(r, c)).sum();
                for c in 0..v {
                    // dX[r,c] = S[r,c] * (dS[r,c] - dot)
                    *dx.at_mut(r, c) = s.at(r, c) * (ds.at(r, c) - dot);
                }
            }
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Layer Normalization — normalizes each row independently.
    ///
    /// Y = (X - μ) / σ * γ + β
    ///
    /// where μ and σ are computed per-row (per token), and γ,β are
    /// learned [1, d_model] vectors (shared across all tokens/rows).
    ///
    /// Backward (full derivation):
    ///
    ///   Let X̂ = (X - μ) / σ   (the normalized X, stored from forward)
    ///       D = dY * γ          (upstream scaled by gamma)
    ///       N = d_model
    ///
    ///   dγ = sum_rows(dY * X̂)          (gradient for gamma)
    ///   dβ = sum_rows(dY)               (gradient for beta)
    ///   dX = (1/σ) * (D - mean(D) - X̂ * mean(D * X̂))
    ///
    ///   This formula comes from differentiating through the normalization.
    ///   The mean() subtractions remove the contributions from the mean and
    ///   variance computations (they depend on X too).
    pub fn layer_norm(&self, gamma: &TensorNode, beta: &TensorNode) -> TensorNode {
        let x = self.data().clone();
        let g = gamma.data().clone();
        let b = beta.data().clone();
        let (t, d) = (x.rows, x.cols);
        let eps = 1e-5f32;
        let inv_d = 1.0 / d as f32;

        // Forward: compute mean and variance per row, then normalize
        let mut mean  = vec![0.0f32; t];
        let mut var   = vec![0.0f32; t];
        let mut x_hat = Mat::zeros(t, d);

        for r in 0..t {
            mean[r] = (0..d).map(|c| x.at(r, c)).sum::<f32>() * inv_d;
            var[r]  = (0..d).map(|c| (x.at(r, c) - mean[r]).powi(2)).sum::<f32>() * inv_d;
            let inv_std = 1.0 / (var[r] + eps).sqrt();
            for c in 0..d {
                *x_hat.at_mut(r, c) = (x.at(r, c) - mean[r]) * inv_std;
            }
        }

        // Y = gamma * X̂ + beta  (broadcast gamma/beta across rows)
        let out_data = Mat::from_fn(t, d, |r, c| {
            x_hat.at(r, c) * g.at(0, c) + b.at(0, c)
        });
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let gamma_c = gamma.clone();
        let beta_c  = beta.clone();
        let out_c   = out.clone();
        let x_hat_stored = x_hat.clone(); // need X̂ in backward
        let var_stored   = var.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dy = out_c.0.borrow().grad.clone();
            let g  = gamma_c.0.borrow().data.clone();

            // dγ = sum_rows(dY * X̂)   shape [1, d]
            let mut dg = Mat::zeros(1, d);
            for c in 0..d {
                for r in 0..t { *dg.at_mut(0, c) += dy.at(r, c) * x_hat_stored.at(r, c); }
            }
            gamma_c.0.borrow_mut().grad.add_assign(&dg);

            // dβ = sum_rows(dY)        shape [1, d]
            let db = dy.sum_rows();
            beta_c.0.borrow_mut().grad.add_assign(&db);

            // dX per row
            let mut dx = Mat::zeros(t, d);
            for r in 0..t {
                let inv_std = 1.0 / (var_stored[r] + eps).sqrt();
                // D[r] = dY[r] * gamma   (element-wise)
                let d_row: Vec<f32> = (0..d).map(|c| dy.at(r, c) * g.at(0, c)).collect();
                let mean_d   = d_row.iter().sum::<f32>() * inv_d;
                let mean_dxh = d_row.iter().enumerate()
                    .map(|(c, &dv)| dv * x_hat_stored.at(r, c))
                    .sum::<f32>() * inv_d;
                for c in 0..d {
                    *dx.at_mut(r, c) = inv_std * (d_row[c] - mean_d
                        - x_hat_stored.at(r, c) * mean_dxh);
                }
            }
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), gamma.clone(), beta.clone()];
        out
    }

    /// Causal self-attention: given Q,K,V matrices, compute attended output.
    ///
    /// This fuses score computation + causal masking + softmax + weighted sum
    /// into one node. Fewer nodes in the graph = faster backward traversal.
    ///
    /// Forward:
    ///   scores = Q @ K.T / sqrt(d_head)         [T, T]
    ///   scores[i,j] = -1e9 for j > i            (causal mask)
    ///   weights = softmax(scores)                [T, T]
    ///   output  = weights @ V                    [T, d_head]
    ///
    /// Backward:
    ///   d_weights = dOut @ V.T                   [T, T]
    ///   dV        = weights.T @ dOut             [T, d_head]
    ///   d_scores  = softmax_backward(d_weights)  [T, T]  (causal positions only)
    ///   dQ        = d_scores @ K / sqrt(d_head)  [T, d_head]
    ///   dK        = d_scores.T @ Q / sqrt(d_head)[T, d_head]
    pub fn causal_attention(q: &TensorNode, k: &TensorNode, v: &TensorNode, d_head: usize)
        -> TensorNode
    {
        let q_d = q.data().clone();
        let k_d = k.data().clone();
        let v_d = v.data().clone();
        let t = q_d.rows;
        let scale = 1.0 / (d_head as f32).sqrt();

        // scores = Q @ K.T * scale  [T, T]
        let mut scores = q_d.matmul(&k_d.transpose()).scale(scale);

        // Apply causal mask
        for i in 0..t {
            for j in (i+1)..t {
                *scores.at_mut(i, j) = -1e9;
            }
        }

        // weights = softmax(scores) row-wise  [T, T]
        let weights = {
            let mut w = Mat::zeros(t, t);
            for r in 0..t {
                let row_max = (0..t).map(|c| scores.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0f32;
                for c in 0..t {
                    let e = (scores.at(r, c) - row_max).exp();
                    *w.at_mut(r, c) = e;
                    row_sum += e;
                }
                for c in 0..t { *w.at_mut(r, c) /= row_sum; }
            }
            w
        };

        // output = weights @ V  [T, d_head]
        let out_data = weights.matmul(&v_d);
        let out = TensorNode::leaf(out_data);

        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        let out_c = out.clone();
        let weights_stored = weights;

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let w  = &weights_stored;
            let q_d = q_c.0.borrow().data.clone();
            let k_d = k_c.0.borrow().data.clone();
            let v_d = v_c.0.borrow().data.clone();

            // dV = W.T @ dOut  [T, d_head]
            let dv = w.transpose().matmul(&dout);
            v_c.0.borrow_mut().grad.add_assign(&dv);

            // d_weights = dOut @ V.T  [T, T]
            let dw = dout.matmul(&v_d.transpose());

            // Backward through causal softmax: d_scores[i,j] = 0 for j>i
            let mut dscores = Mat::zeros(t, t);
            for r in 0..t {
                // Only positions [0..=r] were unmasked
                let dot: f32 = (0..=r).map(|c| dw.at(r, c) * w.at(r, c)).sum();
                for c in 0..=r {
                    *dscores.at_mut(r, c) = w.at(r, c) * (dw.at(r, c) - dot);
                }
            }
            let dscores = dscores.scale(scale);

            // dQ += dscores @ K  [T, d_head]
            let dq = dscores.matmul(&k_d);
            q_c.0.borrow_mut().grad.add_assign(&dq);

            // dK += dscores.T @ Q  [T, d_head]
            let dk = dscores.transpose().matmul(&q_d);
            k_c.0.borrow_mut().grad.add_assign(&dk);
        }));
        out.0.borrow_mut().prev = vec![q.clone(), k.clone(), v.clone()];
        out
    }

    // =========================================================================
    // Backward pass
    // =========================================================================

    /// Run the full backward pass from this node (must be a scalar loss).
    ///
    /// The algorithm is identical to scalar autograd:
    ///   1. Build topological order of all ancestor nodes
    ///   2. Seed this node's gradient with ones (shape [1,1] for a scalar loss)
    ///   3. Walk in reverse topological order, calling each backward_fn
    ///
    /// The `unsafe` block is the same trick as scalar autograd: we need to call
    /// backward_fn while NOT holding the RefCell borrow, because backward_fn
    /// itself will borrow other nodes (including potentially this one's inputs).
    /// We extract a raw pointer to the Fn before releasing the borrow.
    pub fn backward(&self) {
        // Build topological order
        let mut topo: Vec<TensorNode> = Vec::new();
        let mut visited: HashSet<*const RefCell<NodeData>> = HashSet::new();

        fn build(v: &TensorNode, topo: &mut Vec<TensorNode>,
                 visited: &mut HashSet<*const RefCell<NodeData>>) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.contains(&ptr) { return; }
            visited.insert(ptr);
            let prev = v.0.borrow().prev.clone();
            for p in &prev { build(p, topo, visited); }
            topo.push(v.clone());
        }
        build(self, &mut topo, &mut visited);

        // Seed: gradient of loss w.r.t. itself = 1
        {
            let mut inner = self.0.borrow_mut();
            assert!(inner.data.numel() == 1,
                "backward() must be called on a scalar (1×1 matrix), got shape [{},{}]",
                inner.data.rows, inner.data.cols);
            inner.grad = Mat::ones(1, 1);
        }

        // Walk in reverse topological order, calling each backward_fn.
        // We must release the borrow before calling the fn (it borrows input nodes).
        for node in topo.iter().rev() {
            let fn_ptr = {
                let inner = node.0.borrow();
                inner.backward_fn.as_ref().map(|f| {
                    // Safety: f is owned by node (stored in the Rc), which stays
                    // alive for the duration of this loop. We drop the borrow
                    // before calling, so no aliasing occurs.
                    unsafe { &*(f.as_ref() as *const dyn Fn()) }
                })
            };
            if let Some(f) = fn_ptr { f(); }
        }
    }
}

impl std::fmt::Debug for TensorNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.borrow();
        write!(f, "TensorNode(shape=[{},{}], grad_norm={:.4})",
               inner.data.rows, inner.data.cols, inner.grad.norm())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-3 }

    // Numerical gradient check: perturb each element of mat by h, measure output change.
    fn numerical_grad<F: Fn(&Mat) -> f32>(f: &F, mat: &Mat) -> Mat {
        let h = 1e-3f32;
        Mat::from_fn(mat.rows, mat.cols, |r, c| {
            let mut plus = mat.clone();
            *plus.at_mut(r, c) += h;
            let mut minus = mat.clone();
            *minus.at_mut(r, c) -= h;
            (f(&plus) - f(&minus)) / (2.0 * h)
        })
    }

    // --- Mat basics ---

    #[test]
    fn test_matmul_shape() {
        let a = Mat::zeros(3, 4);
        let b = Mat::zeros(4, 5);
        let c = a.matmul(&b);
        assert_eq!((c.rows, c.cols), (3, 5));
    }

    #[test]
    fn test_matmul_values() {
        // [1 2; 3 4] @ [5; 6] = [1*5+2*6; 3*5+4*6] = [17; 39]
        let a = Mat::new(vec![1.,2.,3.,4.], 2, 2);
        let b = Mat::new(vec![5.,6.], 2, 1);
        let c = a.matmul(&b);
        assert!(approx(c.at(0,0), 17.0));
        assert!(approx(c.at(1,0), 39.0));
    }

    #[test]
    fn test_transpose() {
        let a = Mat::new(vec![1.,2.,3.,4.,5.,6.], 2, 3);
        let at = a.transpose();
        assert_eq!((at.rows, at.cols), (3, 2));
        assert!(approx(at.at(0,0), 1.0));
        assert!(approx(at.at(1,0), 2.0));
        assert!(approx(at.at(0,1), 4.0));
    }

    // Helper: call a node's backward_fn without holding a RefCell borrow.
    fn call_backward(node: &TensorNode) {
        let fn_ptr = {
            let inner = node.0.borrow();
            inner.backward_fn.as_ref().map(|f| unsafe { &*(f.as_ref() as *const dyn Fn()) })
        };
        if let Some(f) = fn_ptr { f(); }
    }

    // --- TensorNode backward rules (verified numerically) ---

    #[test]
    fn test_matmul_grad_a() {
        // Loss = sum(A @ B). Check dA numerically.
        let a_data = Mat::new(vec![1.,2.,3.,4.,5.,6.], 2, 3);
        let b_data = Mat::new(vec![0.1,0.2, 0.3,0.4, 0.5,0.6], 3, 2);

        let num = numerical_grad(&|a| {
            let a_n = TensorNode::leaf(a.clone());
            let b_n = TensorNode::leaf(b_data.clone());
            let c = a_n.matmul(&b_n);
            c.data().data.iter().sum::<f32>()
        }, &a_data);

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.matmul(&b);
        // Read shape before borrowing mutably
        let (cr, cc) = { let d = c.data(); (d.rows, d.cols) };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        { let inner = c.0.borrow();
          let fn_ptr = inner.backward_fn.as_ref().map(|f| unsafe { &*(f.as_ref() as *const dyn Fn()) });
          drop(inner);
          if let Some(f) = fn_ptr { f(); }
        }

        let ag = a.grad().clone();
        for r in 0..ag.rows {
            for col in 0..ag.cols {
                assert!(approx(ag.at(r, col), num.at(r, col)),
                    "dA[{},{}]: analytical={:.4} numerical={:.4}", r, col, ag.at(r,col), num.at(r,col));
            }
        }
    }

    #[test]
    fn test_matmul_grad_b() {
        let a_data = Mat::new(vec![1.,2.,3.,4.,5.,6.], 2, 3);
        let b_data = Mat::new(vec![0.1,0.2, 0.3,0.4, 0.5,0.6], 3, 2);

        let num = numerical_grad(&|b| {
            let a_n = TensorNode::leaf(a_data.clone());
            let b_n = TensorNode::leaf(b.clone());
            let c = a_n.matmul(&b_n);
            c.data().data.iter().sum::<f32>()
        }, &b_data);

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.matmul(&b);
        let (cr, cc) = { let d = c.data(); (d.rows, d.cols) };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        call_backward(&c);

        let bg = b.grad().clone();
        for r in 0..bg.rows {
            for col in 0..bg.cols {
                assert!(approx(bg.at(r, col), num.at(r, col)),
                    "dB[{},{}]: analytical={:.4} numerical={:.4}", r, col, bg.at(r,col), num.at(r,col));
            }
        }
    }

    #[test]
    fn test_add_bias_grad() {
        let a_data = Mat::new(vec![1.,2.,3.,4.,5.,6.], 3, 2);
        let b_data = Mat::new(vec![0.5, -0.5], 1, 2);

        let num_b = numerical_grad(&|b| {
            let a_n = TensorNode::leaf(a_data.clone());
            let b_n = TensorNode::leaf(b.clone());
            let c = a_n.add_bias(&b_n);
            c.data().data.iter().sum::<f32>()
        }, &b_data);

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.add_bias(&b);
        let (cr, cc) = { let d = c.data(); (d.rows, d.cols) };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        call_backward(&c);

        let bg = b.grad().clone();
        for col in 0..2 {
            assert!(approx(bg.at(0, col), num_b.at(0, col)),
                "d_bias[{}]: analytical={:.4} numerical={:.4}", col, bg.at(0,col), num_b.at(0,col));
        }
    }

    #[test]
    fn test_gelu_grad() {
        let x_data = Mat::new(vec![-1.0, 0.0, 0.5, 2.0], 1, 4);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            let g = xn.gelu();
            g.data().data.iter().sum::<f32>()
        }, &x_data);

        let x = TensorNode::leaf(x_data);
        let g = x.gelu();
        g.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&g);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(approx(xg.at(0,c), num.at(0,c)),
                "GELU grad[{}]: analytical={:.4} numerical={:.4}", c, xg.at(0,c), num.at(0,c));
        }
    }

    #[test]
    fn test_softmax_probabilities() {
        let x = TensorNode::leaf(Mat::new(vec![1.,2.,3., 4.,5.,6.], 2, 3));
        let s = x.softmax();
        // Each row must sum to 1
        for r in 0..2 {
            let row_sum: f32 = (0..3).map(|c| s.data().at(r,c)).sum();
            assert!(approx(row_sum, 1.0), "row {} sum = {}", r, row_sum);
        }
    }

    #[test]
    fn test_softmax_grad() {
        let x_data = Mat::new(vec![1.0, 2.0, 0.5], 1, 3);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            let s = xn.softmax();
            // Loss = sum(s * weights) with fixed weights to get non-trivial grad
            s.data().data.iter().enumerate().map(|(i, &v)| v * (i+1) as f32).sum::<f32>()
        }, &x_data);

        let x = TensorNode::leaf(x_data);
        let s = x.softmax();
        // Seed: dLoss/dS[i] = i+1
        let ds = Mat::new(vec![1.0, 2.0, 3.0], 1, 3);
        s.0.borrow_mut().grad = ds;
        call_backward(&s);

        let xg = x.grad().clone();
        for c in 0..3 {
            assert!(approx(xg.at(0,c), num.at(0,c)),
                "softmax grad[{}]: analytical={:.4} numerical={:.4}", c, xg.at(0,c), num.at(0,c));
        }
    }

    #[test]
    fn test_layer_norm_output_mean_zero() {
        let x = TensorNode::leaf(Mat::new(vec![1.,2.,3.,4.], 1, 4));
        let gamma = TensorNode::leaf(Mat::ones(1, 4));
        let beta  = TensorNode::leaf(Mat::zeros(1, 4));
        let out = x.layer_norm(&gamma, &beta);
        let mean = out.data().data.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "LN output mean should be 0, got {}", mean);
    }

    #[test]
    fn test_layer_norm_grad_x() {
        let x_data  = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data  = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);
        let b_data  = Mat::zeros(1, 4);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            let gn = TensorNode::leaf(g_data.clone());
            let bn = TensorNode::leaf(b_data.clone());
            let out = xn.layer_norm(&gn, &bn);
            out.data().data.iter().sum::<f32>()
        }, &x_data);

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let b = TensorNode::leaf(b_data);
        let out = x.layer_norm(&g, &b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(approx(xg.at(0,c), num.at(0,c)),
                "LN dX[{}]: analytical={:.4} numerical={:.4}", c, xg.at(0,c), num.at(0,c));
        }
    }

    #[test]
    fn test_layer_norm_grad_gamma() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);
        let b_data = Mat::zeros(1, 4);

        let num = numerical_grad(&|g| {
            let xn = TensorNode::leaf(x_data.clone());
            let gn = TensorNode::leaf(g.clone());
            let bn = TensorNode::leaf(b_data.clone());
            let out = xn.layer_norm(&gn, &bn);
            out.data().data.iter().sum::<f32>()
        }, &g_data);

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let b = TensorNode::leaf(b_data);
        let out = x.layer_norm(&g, &b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let gg = g.grad().clone();
        for c in 0..4 {
            assert!(approx(gg.at(0,c), num.at(0,c)),
                "LN d_gamma[{}]: analytical={:.4} numerical={:.4}", c, gg.at(0,c), num.at(0,c));
        }
    }

    #[test]
    fn test_causal_attention_output_shape() {
        let t = 4; let d = 8;
        let q = TensorNode::leaf(Mat::zeros(t, d));
        let k = TensorNode::leaf(Mat::zeros(t, d));
        let v = TensorNode::leaf(Mat::zeros(t, d));
        let out = TensorNode::causal_attention(&q, &k, &v, d);
        assert_eq!((out.data().rows, out.data().cols), (t, d));
    }

    #[test]
    fn test_causal_attention_grad_v() {
        let t = 3; let d = 4;
        let q_data = Mat::from_fn(t, d, |r,c| (r*d+c) as f32 * 0.1);
        let k_data = Mat::from_fn(t, d, |r,c| (r*d+c) as f32 * 0.05);
        let v_data = Mat::from_fn(t, d, |r,c| (r*d+c) as f32 * 0.07);

        let num = numerical_grad(&|v| {
            let qn = TensorNode::leaf(q_data.clone());
            let kn = TensorNode::leaf(k_data.clone());
            let vn = TensorNode::leaf(v.clone());
            let out = TensorNode::causal_attention(&qn, &kn, &vn, d);
            out.data().data.iter().sum::<f32>()
        }, &v_data);

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::causal_attention(&q, &k, &v, d);
        out.0.borrow_mut().grad = Mat::ones(t, d);
        call_backward(&out);

        let vg = v.grad().clone();
        for r in 0..t { for c in 0..d {
            assert!(approx(vg.at(r,c), num.at(r,c)),
                "attn dV[{},{}]: analytical={:.4} numerical={:.4}", r, c, vg.at(r,c), num.at(r,c));
        }}
    }

    #[test]
    fn test_full_backward_via_backward_method() {
        // Build a small compute graph and call .backward() end-to-end.
        // We add a sum-to-scalar node so backward() can be called directly.
        let a = TensorNode::leaf(Mat::new(vec![1.,2.,3.,4.], 2, 2));
        let b = TensorNode::leaf(Mat::new(vec![0.5,0.5,0.5,0.5], 2, 2));
        let c = a.matmul(&b);    // [2,2]
        let g = c.gelu();        // [2,2]

        // Reduce to scalar: loss = sum(g) implemented as sum node
        let sum_val = g.data().sum();
        let loss = TensorNode::leaf(Mat::new(vec![sum_val], 1, 1));
        // Wire loss.backward_fn to propagate ones back into g
        let g_c = g.clone();
        let (gr, gc) = (g.data().rows, g.data().cols);
        loss.0.borrow_mut().backward_fn = Some(Box::new(move || {
            g_c.0.borrow_mut().grad.add_assign(&Mat::ones(gr, gc));
        }));
        loss.0.borrow_mut().prev = vec![g];

        loss.backward();

        // a.grad should be finite and non-zero
        let ag = a.grad().clone();
        assert!(ag.data.iter().all(|x| x.is_finite()), "gradient should be finite");
        assert!(ag.data.iter().any(|x| x.abs() > 1e-6), "gradient should be non-zero");
    }
}
