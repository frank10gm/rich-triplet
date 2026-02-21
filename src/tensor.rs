/// # Tensor — the fundamental data structure of every neural network
///
/// A Tensor is simply a multi-dimensional array of floating-point numbers.
/// You can think of it as a generalization:
///   - a scalar  (single number)  is a 0-D tensor   shape: []
///   - a vector  (list)           is a 1-D tensor   shape: [n]
///   - a matrix  (table)          is a 2-D tensor   shape: [rows, cols]
///   - a 3-D tensor               is a "cube"       shape: [depth, rows, cols]
///   - and so on…
///
/// In an LLM, virtually everything is a tensor:
///   - a batch of tokens          shape: [batch_size, sequence_length]
///   - word embeddings            shape: [vocab_size, embedding_dim]
///   - attention weights          shape: [batch, heads, seq, seq]
///   - the final logits           shape: [batch, seq, vocab_size]
///
/// ## Memory layout: row-major (C order)
///
/// We store all values in a flat Vec<f32>. For a 2D tensor of shape [3, 4]:
///
///   [ row0col0, row0col1, row0col2, row0col3,
///     row1col0, row1col1, row1col2, row1col3,
///     row2col0, row2col1, row2col2, row2col3 ]
///
/// To access element at (row=i, col=j) we compute the *flat index*:
///   index = i * 4 + j      (i * num_cols + j)
///
/// For a 3D tensor of shape [D, R, C]:
///   index = d * (R*C) + r * C + c
///
/// This generalizes: each dimension has a "stride" — how many elements
/// you skip in the flat array when you move one step in that dimension.
///   strides for shape [D, R, C] = [R*C, C, 1]

#[derive(Debug, Clone)]
pub struct Tensor {
    /// The actual numbers, stored flat in row-major order
    pub data: Vec<f32>,

    /// The shape describes how many dimensions there are and their sizes.
    /// e.g. [2, 3] means 2 rows, 3 columns → 6 total elements
    pub shape: Vec<usize>,

    /// Strides: how many elements to skip per dimension in the flat array.
    /// Computed automatically from shape. For shape [2, 3]: strides = [3, 1]
    pub strides: Vec<usize>,
}

impl Tensor {
    /// Create a new Tensor from raw data and a shape.
    ///
    /// # Panics
    /// Panics if the data length doesn't match the product of the shape.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected_len: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected_len,
            "data length {} does not match shape {:?} (expected {} elements)",
            data.len(),
            shape,
            expected_len
        );

        let strides = Self::compute_strides(&shape);
        Tensor { data, shape, strides }
    }

    /// Compute row-major strides from a shape.
    ///
    /// For shape [A, B, C]:
    ///   strides[2] = 1         (moving 1 step along dim 2 skips 1 element)
    ///   strides[1] = C         (moving 1 step along dim 1 skips C elements)
    ///   strides[0] = B*C       (moving 1 step along dim 0 skips B*C elements)
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let ndim = shape.len();
        let mut strides = vec![1usize; ndim];
        // Start from the second-to-last dimension and go backward
        for i in (0..ndim.saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![0.0f32; n], shape)
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self::new(vec![1.0f32; n], shape)
    }

    /// Create a 1D tensor from a range [0, n)  — useful for position indices.
    pub fn arange(n: usize) -> Self {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Self::new(data, vec![n])
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Convert a multi-dimensional index into a flat index using strides.
    ///
    /// Example: tensor of shape [3, 4], access (1, 2):
    ///   flat = 1 * strides[0] + 2 * strides[1] = 1*4 + 2*1 = 6
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.ndim(), "wrong number of indices");
        indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum()
    }

    /// Read a single element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f32 {
        let flat = self.flat_index(indices);
        self.data[flat]
    }

    /// Write a single element by multi-dimensional index.
    pub fn set(&mut self, indices: &[usize], value: f32) {
        let flat = self.flat_index(indices);
        self.data[flat] = value;
    }

    // -------------------------------------------------------------------------
    // Element-wise operations
    // -------------------------------------------------------------------------
    // These operations apply the same function to every element independently.
    // They are the simplest operations, but used constantly (activations, etc.)

    /// Element-wise addition: C[i] = A[i] + B[i]
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for add");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise subtraction: C[i] = A[i] - B[i]
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for sub");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Element-wise multiplication (Hadamard product): C[i] = A[i] * B[i]
    /// Note: this is NOT matrix multiplication. That's `matmul` below.
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch for mul");
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Scale every element by a scalar constant: B[i] = A[i] * s
    pub fn scale(&self, s: f32) -> Tensor {
        let data = self.data.iter().map(|&x| x * s).collect();
        Tensor::new(data, self.shape.clone())
    }

    /// Apply a function to every element: B[i] = f(A[i])
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Tensor {
        let data = self.data.iter().map(|&x| f(x)).collect();
        Tensor::new(data, self.shape.clone())
    }

    // -------------------------------------------------------------------------
    // Reduction operations
    // -------------------------------------------------------------------------
    // These collapse one or more dimensions into a single value.

    /// Sum all elements into a single scalar.
    pub fn sum_all(&self) -> f32 {
        self.data.iter().sum()
    }

    /// Sum along the last dimension.
    ///
    /// For a 2D tensor of shape [M, N], returns a tensor of shape [M]:
    ///   out[i] = sum over j of in[i, j]
    ///
    /// This is used e.g. in softmax normalization.
    pub fn sum_last_dim(&self) -> Tensor {
        assert!(self.ndim() >= 1);
        let last = self.shape[self.ndim() - 1]; // size of last dimension
        let outer: usize = self.data.len() / last; // product of all other dims

        let data = (0..outer)
            .map(|i| self.data[i * last..(i + 1) * last].iter().sum())
            .collect();

        // Shape drops the last dimension
        let new_shape = self.shape[..self.ndim() - 1].to_vec();
        // Edge case: if result is 0-D, make it [1]
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        Tensor::new(data, new_shape)
    }

    /// Max value along the last dimension (used in softmax for numerical stability).
    pub fn max_last_dim(&self) -> Tensor {
        assert!(self.ndim() >= 1);
        let last = self.shape[self.ndim() - 1];
        let outer = self.data.len() / last;

        let data = (0..outer)
            .map(|i| {
                self.data[i * last..(i + 1) * last]
                    .iter()
                    .cloned()
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();

        let new_shape = self.shape[..self.ndim() - 1].to_vec();
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        Tensor::new(data, new_shape)
    }

    // -------------------------------------------------------------------------
    // Matrix Multiplication — the most important operation in deep learning
    // -------------------------------------------------------------------------
    //
    // Matrix multiplication of A (shape [M, K]) and B (shape [K, N])
    // produces C (shape [M, N]) where:
    //
    //   C[i, j] = sum over k of (A[i, k] * B[k, j])
    //
    // Every linear layer in a neural network is a matmul:
    //   output = input @ weight  (+ bias)
    //
    // In attention:
    //   scores = queries @ keys.T    (T = transpose)
    //   context = scores @ values
    //
    // This naive O(M*N*K) implementation is correct but not optimized.
    // Real frameworks use BLAS, GPU kernels, or tiled algorithms.
    // For learning purposes, clarity beats performance.

    /// 2D matrix multiplication: A [M, K] × B [K, N] → C [M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.ndim(), 2, "matmul requires 2D tensors");
        assert_eq!(other.ndim(), 2, "matmul requires 2D tensors");

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        assert_eq!(
            k, k2,
            "matmul inner dimensions must match: [{}, {}] x [{}, {}]",
            m, k, k2, n
        );

        let mut result = vec![0.0f32; m * n];

        // Triple loop: i = row of A, j = col of B, p = shared inner dimension
        for i in 0..m {
            for p in 0..k {
                // a_val is fixed for inner j loop — this ordering improves cache usage
                let a_val = self.data[i * k + p];
                for j in 0..n {
                    result[i * n + j] += a_val * other.data[p * n + j];
                }
            }
        }

        Tensor::new(result, vec![m, n])
    }

    /// Transpose a 2D tensor: swap rows and columns.
    ///
    /// A [M, N] → A.T [N, M]
    ///
    /// In attention we compute: scores = Q @ K^T
    /// meaning we need to transpose the key matrix before multiplying.
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.ndim(), 2, "transpose requires a 2D tensor");
        let m = self.shape[0];
        let n = self.shape[1];

        let mut result = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                // A[i,j] goes to A.T[j,i]
                result[j * m + i] = self.data[i * n + j];
            }
        }
        Tensor::new(result, vec![n, m])
    }

    // -------------------------------------------------------------------------
    // Shape manipulation
    // -------------------------------------------------------------------------

    /// Reshape the tensor to a new shape (data is unchanged, only view changes).
    ///
    /// The total number of elements must remain the same.
    /// e.g. shape [2, 6] can be reshaped to [3, 4] or [12] or [2, 2, 3]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let new_n: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            new_n,
            "reshape: element count must stay the same ({} != {})",
            self.numel(),
            new_n
        );
        Tensor::new(self.data.clone(), new_shape)
    }

    /// Print a human-readable summary of the tensor.
    pub fn print_info(&self, name: &str) {
        println!(
            "{}: shape={:?}, numel={}, first_few={:?}",
            name,
            self.shape,
            self.numel(),
            &self.data[..self.data.len().min(6)]
        );
    }
}

// -------------------------------------------------------------------------
// Softmax — the activation at the heart of attention
// -------------------------------------------------------------------------
//
// Softmax converts a vector of raw scores ("logits") into a probability
// distribution (all values in [0,1] that sum to 1).
//
//   softmax(x)[i] = exp(x[i]) / sum_j(exp(x[j]))
//
// Why do we subtract the max first?
//   exp(large_number) overflows to infinity in float32.
//   Subtracting max(x) before exp is mathematically equivalent
//   (the constant cancels out) but keeps values numerically stable.
//
//   softmax(x)[i] = exp(x[i] - max(x)) / sum_j(exp(x[j] - max(x)))
//
// In the transformer, softmax is applied to attention scores to produce
// "attention weights" — how much each token should attend to each other.

pub fn softmax(x: &Tensor) -> Tensor {
    // Subtract max for numerical stability (along last dim)
    let max = x.max_last_dim();

    // Broadcast max back to x's shape and subtract
    let last_dim = x.shape[x.ndim() - 1];
    let outer = x.numel() / last_dim;

    let mut shifted = x.data.clone();
    for i in 0..outer {
        let m = max.data[i];
        for j in 0..last_dim {
            shifted[i * last_dim + j] -= m;
        }
    }
    let shifted_tensor = Tensor::new(shifted, x.shape.clone());

    // exp of every element
    let exp_x = shifted_tensor.map(|v| v.exp());

    // Sum of exps along last dim
    let sum_exp = exp_x.sum_last_dim();

    // Divide each element by the sum of its row
    let mut result = exp_x.data.clone();
    for i in 0..outer {
        let s = sum_exp.data[i];
        for j in 0..last_dim {
            result[i * last_dim + j] /= s;
        }
    }

    Tensor::new(result, x.shape.clone())
}

// -------------------------------------------------------------------------
// Tests — read these as worked examples to understand the operations
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strides_2d() {
        // shape [3, 4] → strides should be [4, 1]
        // meaning: to move 1 row forward, skip 4 elements
        //          to move 1 col forward, skip 1 element
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.strides, vec![4, 1]);
    }

    #[test]
    fn test_strides_3d() {
        // shape [2, 3, 4] → strides should be [12, 4, 1]
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.strides, vec![12, 4, 1]);
    }

    #[test]
    fn test_flat_index() {
        // For shape [3, 4], element (1, 2) should be at flat index 6
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.flat_index(&[1, 2]), 6); // 1*4 + 2*1 = 6
    }

    #[test]
    fn test_get_set() {
        let mut t = Tensor::zeros(vec![2, 3]);
        t.set(&[1, 2], 42.0);
        assert_eq!(t.get(&[1, 2]), 42.0);
        assert_eq!(t.get(&[0, 0]), 0.0); // others unchanged
    }

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_matmul() {
        // [2, 3] × [3, 2] → [2, 2]
        //
        //  A = |1 2 3|    B = |7  8 |
        //      |4 5 6|        |9  10|
        //                     |11 12|
        //
        //  C[0,0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
        //  C[0,1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
        //  C[1,0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
        //  C[1,1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154

        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::new(
            vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            vec![3, 2],
        );
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_transpose() {
        // |1 2 3|  →  |1 4|
        // |4 5 6|      |2 5|
        //              |3 6|
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let at = a.transpose();
        assert_eq!(at.shape, vec![3, 2]);
        assert_eq!(at.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let s = softmax(&x);
        let total: f32 = s.data.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "softmax must sum to 1, got {}", total);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values should not produce NaN or infinity
        let x = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![1, 3]);
        let s = softmax(&x);
        for &v in &s.data {
            assert!(v.is_finite(), "softmax produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_reshape() {
        let a = Tensor::arange(12);
        let b = a.reshape(vec![3, 4]);
        assert_eq!(b.shape, vec![3, 4]);
        assert_eq!(b.data, a.data); // same data, different view
    }
}
