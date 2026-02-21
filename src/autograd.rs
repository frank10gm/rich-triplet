/// # Autograd — Automatic Differentiation
///
/// This is the engine that makes neural networks learn.
///
/// ## The problem
///
/// We want to minimize a loss function L(weights).
/// To do that we use gradient descent:
///
///   weight -= learning_rate * dL/d(weight)
///
/// The problem: a transformer has millions of weights, each contributing
/// to the loss through a long chain of operations. Computing dL/d(weight)
/// by hand for every weight is impossible.
///
/// ## The solution: the chain rule + a computation graph
///
/// If L = f(g(h(x))), then by the chain rule:
///
///   dL/dx = dL/df · df/dg · dg/dh · dh/dx
///
/// Autograd automates this. It works in two passes:
///
///   Forward pass:  compute the output, but also RECORD every operation
///                  in a "computation graph" (who produced what from what)
///
///   Backward pass: starting from the loss (gradient = 1.0), walk the
///                  graph in reverse and accumulate gradients using the
///                  chain rule at each node
///
/// This is called "reverse-mode automatic differentiation" or "backprop".
///
/// ## Our design: Value nodes
///
/// Each `Value` wraps a scalar f32 and records:
///   - its current value
///   - its gradient (dL/d(this_value)), accumulated during backward
///   - the backward function: how to push gradient to its inputs
///
/// Example trace for z = x * y:
///
///   x = Value(3.0)    grad starts at 0
///   y = Value(4.0)    grad starts at 0
///   z = x * y         z.data = 12.0
///                      z._backward = || { x.grad += z.grad * y.data
///                                         y.grad += z.grad * x.data }
///
///   z.grad = 1.0      (seed the backward pass)
///   z.backward()      → x.grad = 1.0 * 4.0 = 4.0
///                        y.grad = 1.0 * 3.0 = 3.0
///
/// ## Why scalars and not tensors?
///
/// Karpathy's micrograd does this with scalars for clarity.
/// We do the same — it makes every derivative rule crystal clear.
/// In Phase 3 we'll extend to tensors once the concept is solid.
///
/// ## Topological sort
///
/// The backward pass must visit nodes in reverse topological order —
/// a node must receive ALL incoming gradients before it propagates
/// backward. We compute this order once during `backward()`.

use std::cell::RefCell;
use std::rc::Rc;
use std::collections::HashSet;

// =============================================================================
// ValueData — the internal state of one node in the computation graph
// =============================================================================

struct ValueData {
    /// The scalar value computed in the forward pass
    pub val: f32,

    /// Accumulated gradient dL/d(this), filled in during backward pass
    pub grad: f32,

    /// Human-readable label for debugging (e.g. "w1", "loss")
    pub label: String,

    /// The backward function: how to propagate gradient to this node's inputs.
    /// It's a closure that captures references to the input nodes.
    /// `None` for leaf nodes (inputs, weights) — they have no inputs to propagate to.
    backward_fn: Option<Box<dyn Fn()>>,

    /// The nodes that were used to produce this value (direct inputs).
    /// Used to build the topological order for the backward pass.
    prev: Vec<Value>,
}

// =============================================================================
// Value — a reference-counted, interior-mutable node
// =============================================================================
//
// We wrap ValueData in Rc<RefCell<...>> because:
//
//   Rc<T>       — multiple owners (a value can be used in multiple operations)
//   RefCell<T>  — interior mutability (we mutate .grad during backward,
//                 even through shared references)
//
// This is the standard Rust pattern for graph-like structures where nodes
// need to reference each other mutably.

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueData>>);

impl Value {
    /// Create a new leaf node (no inputs — a weight or input value).
    pub fn new(val: f32) -> Self {
        Value(Rc::new(RefCell::new(ValueData {
            val,
            grad: 0.0,
            label: String::new(),
            backward_fn: None,
            prev: vec![],
        })))
    }

    /// Create a new leaf node with a label (for debugging).
    pub fn with_label(val: f32, label: &str) -> Self {
        let v = Self::new(val);
        v.0.borrow_mut().label = label.to_string();
        v
    }

    /// Read the current scalar value.
    pub fn val(&self) -> f32 {
        self.0.borrow().val
    }

    /// Read the current gradient.
    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    /// Zero out the gradient (call before each backward pass).
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }

    /// Attach a label for debugging.
    pub fn label(&self, s: &str) -> Self {
        self.0.borrow_mut().label = s.to_string();
        self.clone()
    }

    // -------------------------------------------------------------------------
    // Operations — each records how to differentiate itself
    // -------------------------------------------------------------------------

    /// z = x + y
    ///
    /// Derivative rule: dL/dx = dL/dz · 1   (gradient passes through unchanged)
    ///                  dL/dy = dL/dz · 1
    ///
    /// Addition is the "gradient router" — it distributes the upstream gradient
    /// equally to both inputs. This is why residual connections in transformers
    /// are so powerful: gradients flow through them unimpeded.
    pub fn add(&self, other: &Value) -> Value {
        let out_val = self.val() + other.val();

        let self_clone = self.clone();
        let other_clone = other.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // dL/d(self) += dL/dz * 1
                self_clone.0.borrow_mut().grad += upstream;
                // dL/d(other) += dL/dz * 1
                other_clone.0.borrow_mut().grad += upstream;
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];
        out
    }

    /// z = x * y
    ///
    /// Derivative rule: dL/dx = dL/dz · y   (multiply by the OTHER value)
    ///                  dL/dy = dL/dz · x
    ///
    /// Intuition: if y is large, then x has a big influence on z,
    /// so x's gradient should be amplified by y.
    pub fn mul(&self, other: &Value) -> Value {
        let out_val = self.val() * other.val();

        let self_clone = self.clone();
        let other_clone = other.clone();
        let self_val = self.val();
        let other_val = other.val();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                self_clone.0.borrow_mut().grad  += upstream * other_val;
                other_clone.0.borrow_mut().grad += upstream * self_val;
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];
        out
    }

    /// z = x^n  (power with a constant exponent)
    ///
    /// Derivative rule: dL/dx = dL/dz · n · x^(n-1)
    ///
    /// Special cases we'll use:
    ///   pow(-1)  → reciprocal  1/x
    ///   pow(2)   → square      x²
    pub fn pow(&self, n: f32) -> Value {
        let x = self.val();
        let out_val = x.powf(n);

        let self_clone = self.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // d(x^n)/dx = n * x^(n-1)
                self_clone.0.borrow_mut().grad += upstream * n * x.powf(n - 1.0);
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// z = exp(x)
    ///
    /// Derivative rule: dL/dx = dL/dz · exp(x)
    ///
    /// The magical property of e^x: its derivative is itself!
    /// This makes exp very common in neural networks (softmax, GELU…).
    pub fn exp(&self) -> Value {
        let x = self.val();
        let out_val = x.exp();

        let self_clone = self.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // d(exp(x))/dx = exp(x) = out_val
                self_clone.0.borrow_mut().grad += upstream * out_val;
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// z = ln(x)  (natural logarithm)
    ///
    /// Derivative rule: dL/dx = dL/dz · 1/x
    ///
    /// Used in cross-entropy loss: L = -log(p_correct)
    pub fn ln(&self) -> Value {
        let x = self.val();
        assert!(x > 0.0, "ln of non-positive value: {}", x);
        let out_val = x.ln();

        let self_clone = self.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // d(ln(x))/dx = 1/x
                self_clone.0.borrow_mut().grad += upstream * (1.0 / x);
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// z = relu(x) = max(0, x)
    ///
    /// Derivative rule: dL/dx = dL/dz · (1 if x > 0 else 0)
    ///
    /// ReLU "gates" the gradient — if the forward value was negative,
    /// the gradient is completely blocked (the neuron is "dead" for that input).
    /// This is the most common activation in older networks.
    /// Transformers mostly use GELU instead (we'll add that later).
    pub fn relu(&self) -> Value {
        let x = self.val();
        let out_val = x.max(0.0);

        let self_clone = self.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // Gradient passes through only if forward value was > 0
                self_clone.0.borrow_mut().grad += upstream * if x > 0.0 { 1.0 } else { 0.0 };
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// z = tanh(x)
    ///
    /// Derivative rule: dL/dx = dL/dz · (1 - tanh(x)²)
    ///
    /// Range: (-1, 1). Used in older RNNs (LSTMs).
    /// We include it because it demonstrates the "squashing" activations.
    pub fn tanh(&self) -> Value {
        let x = self.val();
        let t = x.tanh();
        let out_val = t;

        let self_clone = self.clone();
        let out = Self::new(out_val);

        {
            let out_weak = out.clone();
            out.0.borrow_mut().backward_fn = Some(Box::new(move || {
                let upstream = out_weak.0.borrow().grad;
                // d(tanh(x))/dx = 1 - tanh(x)^2
                self_clone.0.borrow_mut().grad += upstream * (1.0 - t * t);
            }));
        }

        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    // -------------------------------------------------------------------------
    // Convenience: subtraction and negation in terms of add/mul
    // -------------------------------------------------------------------------

    pub fn neg(&self) -> Value {
        self.mul(&Value::new(-1.0))
    }

    pub fn sub(&self, other: &Value) -> Value {
        self.add(&other.neg())
    }

    pub fn div(&self, other: &Value) -> Value {
        self.mul(&other.pow(-1.0))
    }

    // -------------------------------------------------------------------------
    // Backward pass
    // -------------------------------------------------------------------------

    /// Trigger the full backward pass from this node (usually the loss).
    ///
    /// Algorithm:
    ///   1. Build the topological order of all ancestor nodes
    ///   2. Seed this node's gradient to 1.0 (dL/dL = 1)
    ///   3. Visit nodes in reverse topological order, calling each backward_fn
    pub fn backward(&self) {
        // Build topological order via DFS
        let mut topo: Vec<Value> = Vec::new();
        let mut visited: HashSet<*const RefCell<ValueData>> = HashSet::new();

        fn build_topo(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const RefCell<ValueData>>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            // Visit all inputs first (they come earlier in topological order)
            let prev = v.0.borrow().prev.clone();
            for child in &prev {
                build_topo(child, topo, visited);
            }
            topo.push(v.clone());
        }

        build_topo(self, &mut topo, &mut visited);

        // Seed: dL/dL = 1.0
        self.0.borrow_mut().grad = 1.0;

        // Walk in reverse topological order — from output back to inputs
        for node in topo.iter().rev() {
            let backward_fn = node.0.borrow().backward_fn.as_ref().map(|f| {
                // We can't call f while borrowing node, so extract the fn pointer
                // by re-borrowing. We store backward_fn as Option<Box<dyn Fn()>>
                // which is 'static — safe to call outside the borrow.
                unsafe {
                    // Safety: we immediately call this and don't keep the ref
                    let ptr = f.as_ref() as *const dyn Fn();
                    &*ptr
                }
            });
            if let Some(f) = backward_fn {
                f();
            }
        }
    }
}

// Custom Debug so we can print Value nicely
impl std::fmt::Debug for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.borrow();
        write!(
            f,
            "Value(val={:.4}, grad={:.4}, label={:?})",
            inner.val, inner.grad, inner.label
        )
    }
}

// =============================================================================
// Tests — these are the best way to understand what autograd does
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// Verify a gradient numerically using finite differences, in f64 for precision.
    ///
    /// The numerical gradient is: (f(x+h) - f(x-h)) / (2h)
    /// This is the definition of the derivative, computed approximately.
    /// We use f64 here so the numerical estimate is precise enough to match
    /// the f32 analytical result — f32 finite differences are too noisy.
    fn numerical_gradient<F: Fn(f64) -> f64>(f: F, x: f32) -> f32 {
        let h = 1e-5f64;
        let xd = x as f64;
        ((f(xd + h) - f(xd - h)) / (2.0 * h)) as f32
    }

    #[test]
    fn test_add_forward() {
        let x = Value::new(3.0);
        let y = Value::new(4.0);
        let z = x.add(&y);
        assert!(approx_eq(z.val(), 7.0));
    }

    #[test]
    fn test_add_backward() {
        // z = x + y, dz/dx = 1, dz/dy = 1
        let x = Value::new(3.0);
        let y = Value::new(4.0);
        let z = x.add(&y);
        z.backward();
        assert!(approx_eq(x.grad(), 1.0), "dz/dx = 1, got {}", x.grad());
        assert!(approx_eq(y.grad(), 1.0), "dz/dy = 1, got {}", y.grad());
    }

    #[test]
    fn test_mul_backward() {
        // z = x * y
        // dz/dx = y = 4.0
        // dz/dy = x = 3.0
        let x = Value::new(3.0);
        let y = Value::new(4.0);
        let z = x.mul(&y);
        z.backward();
        assert!(approx_eq(x.grad(), 4.0), "dz/dx = y = 4, got {}", x.grad());
        assert!(approx_eq(y.grad(), 3.0), "dz/dy = x = 3, got {}", y.grad());
    }

    #[test]
    fn test_chain_rule() {
        // L = (x + y) * z
        // dL/dx = z = 5
        // dL/dy = z = 5
        // dL/dz = x + y = 7
        let x = Value::new(3.0);
        let y = Value::new(4.0);
        let z = Value::new(5.0);
        let s = x.add(&y);
        let l = s.mul(&z);
        l.backward();

        assert!(approx_eq(x.grad(), 5.0), "dL/dx = {}", x.grad());
        assert!(approx_eq(y.grad(), 5.0), "dL/dy = {}", y.grad());
        assert!(approx_eq(z.grad(), 7.0), "dL/dz = {}", z.grad());
    }

    #[test]
    fn test_exp_backward() {
        // z = exp(x) at x=2
        // dz/dx = exp(x) = exp(2) ≈ 7.389
        let x_val = 2.0f32;
        let x = Value::new(x_val);
        let z = x.exp();
        z.backward();

        let expected = numerical_gradient(|v: f64| v.exp(), x_val);
        assert!(
            approx_eq(x.grad(), expected),
            "d(exp(x))/dx: got {:.4}, expected {:.4}",
            x.grad(),
            expected
        );
    }

    #[test]
    fn test_ln_backward() {
        let x_val = 3.0f32;
        let x = Value::new(x_val);
        let z = x.ln();
        z.backward();

        let expected = numerical_gradient(|v: f64| v.ln(), x_val);
        assert!(
            approx_eq(x.grad(), expected),
            "d(ln(x))/dx: got {:.4}, expected {:.4}",
            x.grad(),
            expected
        );
    }

    #[test]
    fn test_relu_positive() {
        // x = 3.0 > 0 → relu(x) = 3, gradient passes through = 1
        let x = Value::new(3.0);
        let z = x.relu();
        z.backward();
        assert!(approx_eq(x.grad(), 1.0));
    }

    #[test]
    fn test_relu_negative() {
        // x = -2.0 < 0 → relu(x) = 0, gradient is blocked = 0
        let x = Value::new(-2.0);
        let z = x.relu();
        z.backward();
        assert!(approx_eq(x.grad(), 0.0));
    }

    #[test]
    fn test_pow_backward() {
        // z = x^3 at x=2 → dz/dx = 3*x^2 = 12
        let x_val = 2.0f32;
        let x = Value::new(x_val);
        let z = x.pow(3.0);
        z.backward();

        let expected = numerical_gradient(|v: f64| v.powf(3.0), x_val);
        assert!(
            approx_eq(x.grad(), expected),
            "d(x^3)/dx: got {:.4}, expected {:.4}",
            x.grad(),
            expected
        );
    }

    #[test]
    fn test_tanh_backward() {
        let x_val = 0.5f32;
        let x = Value::new(x_val);
        let z = x.tanh();
        z.backward();

        let expected = numerical_gradient(|v| v.tanh(), x_val);
        assert!(
            approx_eq(x.grad(), expected),
            "d(tanh)/dx: got {:.4}, expected {:.4}",
            x.grad(),
            expected
        );
    }

    #[test]
    fn test_simple_neuron() {
        // Simulate one neuron: output = relu(w*x + b)
        // x=2, w=0.5, b=-1
        // forward: w*x = 1.0, +b = 0.0, relu = 0.0
        // Since relu output = 0, gradient is blocked: dL/dw = 0, dL/db = 0
        let x = Value::new(2.0);
        let w = Value::with_label(0.5, "w");
        let b = Value::with_label(-1.0, "b");

        let wx = w.mul(&x);
        let pre = wx.add(&b);
        let out = pre.relu();
        out.backward();

        assert!(approx_eq(w.grad(), 0.0)); // blocked by relu at 0
        assert!(approx_eq(b.grad(), 0.0));
    }

    #[test]
    fn test_neuron_active() {
        // output = relu(w*x + b), x=2, w=1, b=0.5
        // forward: w*x = 2.0, +b = 2.5, relu = 2.5  (active!)
        // dL/dw = dL/d(out) * d(out)/d(pre) * d(pre)/dw
        //       = 1 * 1 * x = 2.0
        // dL/db = 1 * 1 * 1 = 1.0
        let x = Value::new(2.0);
        let w = Value::with_label(1.0, "w");
        let b = Value::with_label(0.5, "b");

        let wx = w.mul(&x);
        let pre = wx.add(&b);
        let out = pre.relu();
        out.backward();

        assert!(approx_eq(w.grad(), 2.0), "dL/dw = x = 2, got {}", w.grad());
        assert!(approx_eq(b.grad(), 1.0), "dL/db = 1, got {}", b.grad());
    }

    #[test]
    fn test_gradient_accumulation() {
        // When a value is used in multiple operations, gradients accumulate.
        // L = x * x  (same x used twice)
        // dL/dx = 2x = 4.0 (when x=2)
        let x = Value::new(2.0);
        let z = x.mul(&x); // x is used as both left and right
        z.backward();
        assert!(approx_eq(x.grad(), 4.0), "d(x²)/dx = 2x = 4, got {}", x.grad());
    }

    #[test]
    fn test_mini_loss() {
        // Cross-entropy-like loss for a single prediction:
        // L = -ln(softmax) ≈ -ln(exp(s) / sum_exp)
        //
        // Simplified: L = -ln(p) where p = exp(s) / (exp(s) + exp(0))
        // s = 2.0 (score for the correct class)
        let s = Value::new(2.0);
        let exp_s = s.exp();
        let exp_0 = Value::new(1.0f32.exp()); // competitor score = 1.0
        let sum = exp_s.add(&exp_0);
        let p = exp_s.div(&sum);
        let loss = p.ln().neg();
        loss.backward();

        // Numerical check
        let num_grad = numerical_gradient(|v: f64| {
            let es = v.exp();
            let e0 = 1.0f64.exp();
            let prob = es / (es + e0);
            -prob.ln()
        }, 2.0);

        assert!(
            approx_eq(s.grad(), num_grad),
            "loss gradient: got {:.4}, expected {:.4}",
            s.grad(),
            num_grad
        );
    }
}
