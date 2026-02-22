/// # NDArray — N-dimensional array for batched tensor operations
///
/// This module provides an N-dimensional float array that complements the
/// existing 2-D `Mat` type. It is the foundation for Stage 2 (batched
/// training with 3-D/4-D tensors).
///
/// ## Design
///
/// Storage is a flat `Arc<Vec<f32>>` in row-major (C) order. Views produced
/// by `reshape`, `permute`, and `slice` share the same allocation via `Arc`
/// and describe their layout through `shape`, `strides`, and `offset`.
///
/// ```
/// // Shape [2, 3, 4] has C-order strides [12, 4, 1]:
/// //   element [i, j, k] lives at offset + i*12 + j*4 + k*1
/// ```
///
/// ## Relation to Mat
///
/// `Mat` (rows, cols) maps to `NDArray` with shape `[rows, cols]` and strides
/// `[cols, 1]`.  Use `NDArray::from_mat` / `NDArray::into_mat` to convert.
/// The existing `TensorNode` / autograd2 pipeline is unchanged at this stage.
///
/// ## Key operations
///
/// - **Views** (zero-copy): `reshape`, `permute`, `slice`, `expand_dims`
/// - **Elementwise**: `add`, `mul`, `scale`, `add_scalar`, `map`
/// - **Reductions**: `reduce_sum`, `softmax`
/// - **Batched matmul**: `bmm` — multiplies last two dims, broadcasts over leading dims

use std::sync::Arc;
use crate::autograd2::Mat;

// =============================================================================
// Helper: C-order strides
// =============================================================================

/// Compute row-major (C-order) strides for a given shape.
///
/// `strides[i] = product(shape[i+1..])`, so the last dimension is always 1.
///
/// ```
/// assert_eq!(c_strides(&[2,3,4]), vec![12, 4, 1]);
/// ```
fn c_strides(shape: &[usize]) -> Vec<usize> {
    let n = shape.len();
    if n == 0 { return vec![]; }
    let mut strides = vec![1usize; n];
    for i in (0..n - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// =============================================================================
// Helper: insert a value into a Vec at a given position
// =============================================================================

fn insert_at(v: &[usize], axis: usize, val: usize) -> Vec<usize> {
    let mut out = Vec::with_capacity(v.len() + 1);
    out.extend_from_slice(&v[..axis]);
    out.push(val);
    out.extend_from_slice(&v[axis..]);
    out
}

// =============================================================================
// IndexIter — multi-dimensional index iterator in C-order
// =============================================================================

/// Iterates over all multi-indices of a given shape in row-major (C) order.
///
/// Empty shapes yield zero items. Scalar shape `[]` yields one empty Vec.
pub struct IndexIter {
    shape:   Vec<usize>,
    current: Vec<usize>,
    done:    bool,
}

impl IndexIter {
    pub fn new(shape: &[usize]) -> Self {
        if shape.is_empty() {
            // Scalar: one iteration with empty index
            return IndexIter { shape: vec![], current: vec![], done: false };
        }
        let done = shape.iter().any(|&d| d == 0);
        IndexIter { shape: shape.to_vec(), current: vec![0; shape.len()], done }
    }
}

impl Iterator for IndexIter {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        if self.done { return None; }
        let result = self.current.clone();

        if self.shape.is_empty() {
            // Scalar: only one step
            self.done = true;
            return Some(result);
        }

        // Increment last index, carry over
        let n = self.shape.len();
        let mut dim = n - 1;
        loop {
            self.current[dim] += 1;
            if self.current[dim] < self.shape[dim] {
                break;
            }
            self.current[dim] = 0;
            if dim == 0 {
                self.done = true;
                break;
            }
            dim -= 1;
        }
        Some(result)
    }
}

// =============================================================================
// NDArray
// =============================================================================

/// N-dimensional float array in row-major (C) order.
///
/// Views (`reshape`, `permute`, `slice`) share the underlying `Arc<Vec<f32>>`
/// without copying. Mutation via `at_mut` triggers copy-on-write.
#[derive(Clone, Debug)]
pub struct NDArray {
    /// Flat storage shared across views.
    pub data: Arc<Vec<f32>>,
    /// Logical shape, e.g. `[batch, heads, seq, d_head]`.
    pub shape: Vec<usize>,
    /// Element strides (not byte strides). `strides[i]` is how many elements
    /// to skip when index `i` increases by 1.
    pub strides: Vec<usize>,
    /// Element offset into `data` for the first element of this view.
    pub offset: usize,
}

impl NDArray {
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// All-zero array.
    pub fn zeros(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        NDArray {
            data:    Arc::new(vec![0.0; n]),
            shape:   shape.to_vec(),
            strides: c_strides(shape),
            offset:  0,
        }
    }

    /// All-one array.
    pub fn ones(shape: &[usize]) -> Self {
        let n = shape.iter().product();
        NDArray {
            data:    Arc::new(vec![1.0; n]),
            shape:   shape.to_vec(),
            strides: c_strides(shape),
            offset:  0,
        }
    }

    /// Build from a closure `f(multi_index) -> f32`, iterating in C-order.
    pub fn from_fn(shape: &[usize], f: impl Fn(&[usize]) -> f32) -> Self {
        let mut data = Vec::with_capacity(shape.iter().product());
        for idx in IndexIter::new(shape) {
            data.push(f(&idx));
        }
        NDArray {
            data:    Arc::new(data),
            shape:   shape.to_vec(),
            strides: c_strides(shape),
            offset:  0,
        }
    }

    /// Build from a flat `Vec<f32>` with a given shape.
    ///
    /// Panics if `data.len() != shape.iter().product()`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n,
            "from_vec: data length {} != shape product {}", data.len(), n);
        NDArray {
            data:    Arc::new(data),
            shape:   shape.to_vec(),
            strides: c_strides(shape),
            offset:  0,
        }
    }

    /// Convert a `Mat` (2-D) to an `NDArray` with shape `[rows, cols]`.
    pub fn from_mat(m: &Mat) -> Self {
        NDArray {
            data:    Arc::new(m.data.clone()),
            shape:   vec![m.rows, m.cols],
            strides: vec![m.cols, 1],
            offset:  0,
        }
    }

    /// Convert a 2-D `NDArray` to a `Mat`.
    ///
    /// Panics if `ndim() != 2`.
    pub fn into_mat(&self) -> Mat {
        assert_eq!(self.ndim(), 2,
            "into_mat: requires 2-D array, got shape {:?}", self.shape);
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for j in 0..cols {
                data.push(self.at(&[i, j]));
            }
        }
        Mat::new(data, rows, cols)
    }

    // -------------------------------------------------------------------------
    // Shape / size accessors
    // -------------------------------------------------------------------------

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize { self.shape.len() }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize { self.shape.iter().product() }

    // -------------------------------------------------------------------------
    // Indexing
    // -------------------------------------------------------------------------

    /// Compute the flat index in `data` for the given multi-index.
    #[inline]
    pub fn flat_index(&self, idx: &[usize]) -> usize {
        debug_assert_eq!(idx.len(), self.ndim(),
            "flat_index: idx len {} != ndim {}", idx.len(), self.ndim());
        self.offset + idx.iter().zip(self.strides.iter()).map(|(&i, &s)| i * s).sum::<usize>()
    }

    /// Read element at a multi-index.
    #[inline]
    pub fn at(&self, idx: &[usize]) -> f32 {
        self.data[self.flat_index(idx)]
    }

    /// Write element at a multi-index (copy-on-write if the `Arc` is shared).
    #[inline]
    pub fn at_mut(&mut self, idx: &[usize]) -> &mut f32 {
        let fi = self.flat_index(idx);
        &mut Arc::make_mut(&mut self.data)[fi]
    }

    // -------------------------------------------------------------------------
    // Contiguity
    // -------------------------------------------------------------------------

    /// True when the array has C-order strides and `offset == 0`.
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && self.strides == c_strides(&self.shape)
    }

    /// Return a contiguous, offset-0 copy.  If already contiguous, shares data via Arc.
    pub fn contiguous(&self) -> NDArray {
        if self.is_contiguous() {
            return NDArray {
                data:    Arc::clone(&self.data),
                shape:   self.shape.clone(),
                strides: self.strides.clone(),
                offset:  0,
            };
        }
        let mut data = Vec::with_capacity(self.numel());
        for idx in IndexIter::new(&self.shape) {
            data.push(self.at(&idx));
        }
        NDArray {
            data:    Arc::new(data),
            shape:   self.shape.clone(),
            strides: c_strides(&self.shape),
            offset:  0,
        }
    }

    /// Always returns an owned, contiguous copy (same as `contiguous()`).
    pub fn clone_owned(&self) -> NDArray { self.contiguous() }

    // -------------------------------------------------------------------------
    // Views (zero-copy)
    // -------------------------------------------------------------------------

    /// Reshape to a new shape without copying.
    ///
    /// The total number of elements must be unchanged.
    /// If the array is not contiguous, it is made contiguous first (one allocation).
    pub fn reshape(&self, new_shape: &[usize]) -> NDArray {
        let new_n: usize = new_shape.iter().product();
        assert_eq!(new_n, self.numel(),
            "reshape: new shape {:?} has {} elements, expected {}", new_shape, new_n, self.numel());

        if !self.is_contiguous() {
            return self.contiguous().reshape(new_shape);
        }

        NDArray {
            data:    Arc::clone(&self.data),
            shape:   new_shape.to_vec(),
            strides: c_strides(new_shape),
            offset:  self.offset,
        }
    }

    /// Permute axes without copying.
    ///
    /// `axes` must be a permutation of `0..ndim()`.
    /// The result is typically non-contiguous.
    pub fn permute(&self, axes: &[usize]) -> NDArray {
        assert_eq!(axes.len(), self.ndim(),
            "permute: axes len {} != ndim {}", axes.len(), self.ndim());

        // Validate it's a valid permutation
        let mut seen = vec![false; self.ndim()];
        for &a in axes {
            assert!(a < self.ndim(), "permute: axis {} out of range", a);
            assert!(!seen[a], "permute: duplicate axis {}", a);
            seen[a] = true;
        }

        let new_shape:   Vec<usize> = axes.iter().map(|&a| self.shape[a]).collect();
        let new_strides: Vec<usize> = axes.iter().map(|&a| self.strides[a]).collect();

        NDArray {
            data:    Arc::clone(&self.data),
            shape:   new_shape,
            strides: new_strides,
            offset:  self.offset,
        }
    }

    /// Fix one axis at a given index, returning an array with one fewer dimension.
    ///
    /// Example: `x.slice(0, 2)` on shape `[B, T, D]` returns shape `[T, D]`
    /// pointing into batch element 2.
    pub fn slice(&self, axis: usize, index: usize) -> NDArray {
        assert!(axis < self.ndim(),
            "slice: axis {} out of range for ndim {}", axis, self.ndim());
        assert!(index < self.shape[axis],
            "slice: index {} out of range for dim size {}", index, self.shape[axis]);

        let new_offset  = self.offset + index * self.strides[axis];
        let new_shape:   Vec<usize> = self.shape.iter().enumerate()
            .filter(|&(i, _)| i != axis).map(|(_, &d)| d).collect();
        let new_strides: Vec<usize> = self.strides.iter().enumerate()
            .filter(|&(i, _)| i != axis).map(|(_, &s)| s).collect();

        NDArray {
            data:    Arc::clone(&self.data),
            shape:   new_shape,
            strides: new_strides,
            offset:  new_offset,
        }
    }

    /// Insert a size-1 dimension at position `axis`.
    ///
    /// `axis` may be 0..=ndim() (including at the end).
    pub fn expand_dims(&self, axis: usize) -> NDArray {
        assert!(axis <= self.ndim(),
            "expand_dims: axis {} out of range for ndim {}", axis, self.ndim());

        let mut new_shape   = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.insert(axis, 1);
        new_strides.insert(axis, 0); // stride for a size-1 dim is irrelevant; use 0

        NDArray {
            data:    Arc::clone(&self.data),
            shape:   new_shape,
            strides: new_strides,
            offset:  self.offset,
        }
    }

    // -------------------------------------------------------------------------
    // Elementwise operations
    // -------------------------------------------------------------------------

    /// Element-wise addition. Shapes must match exactly.
    pub fn add(&self, other: &NDArray) -> NDArray {
        assert_eq!(self.shape, other.shape,
            "add: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        let mut out = NDArray::zeros(&self.shape);
        for idx in IndexIter::new(&self.shape) {
            *out.at_mut(&idx) = self.at(&idx) + other.at(&idx);
        }
        out
    }

    /// Element-wise multiplication. Shapes must match exactly.
    pub fn mul(&self, other: &NDArray) -> NDArray {
        assert_eq!(self.shape, other.shape,
            "mul: shape mismatch {:?} vs {:?}", self.shape, other.shape);
        let mut out = NDArray::zeros(&self.shape);
        for idx in IndexIter::new(&self.shape) {
            *out.at_mut(&idx) = self.at(&idx) * other.at(&idx);
        }
        out
    }

    /// Multiply all elements by a scalar.
    pub fn scale(&self, s: f32) -> NDArray {
        let mut out = NDArray::zeros(&self.shape);
        for idx in IndexIter::new(&self.shape) {
            *out.at_mut(&idx) = self.at(&idx) * s;
        }
        out
    }

    /// Add a scalar to all elements.
    pub fn add_scalar(&self, s: f32) -> NDArray {
        let mut out = NDArray::zeros(&self.shape);
        for idx in IndexIter::new(&self.shape) {
            *out.at_mut(&idx) = self.at(&idx) + s;
        }
        out
    }

    /// Apply a function to every element.
    pub fn map(&self, f: impl Fn(f32) -> f32) -> NDArray {
        let mut out = NDArray::zeros(&self.shape);
        for idx in IndexIter::new(&self.shape) {
            *out.at_mut(&idx) = f(self.at(&idx));
        }
        out
    }

    // -------------------------------------------------------------------------
    // Reductions
    // -------------------------------------------------------------------------

    /// Sum over one axis, removing that dimension from the output.
    ///
    /// Example: `[B, T, D].reduce_sum(1)` → `[B, D]`.
    pub fn reduce_sum(&self, axis: usize) -> NDArray {
        assert!(axis < self.ndim(),
            "reduce_sum: axis {} out of range for ndim {}", axis, self.ndim());

        let mut out_shape = self.shape.clone();
        out_shape.remove(axis);
        let mut out = NDArray::zeros(&out_shape);

        for idx in IndexIter::new(&self.shape) {
            let mut out_idx = idx.clone();
            out_idx.remove(axis);
            *out.at_mut(&out_idx) += self.at(&idx);
        }
        out
    }

    /// Numerically stable softmax along one axis.
    ///
    /// Example: `[B, T, V].softmax(2)` applies softmax over the vocabulary dim.
    pub fn softmax(&self, axis: usize) -> NDArray {
        assert!(axis < self.ndim(),
            "softmax: axis {} out of range for ndim {}", axis, self.ndim());

        let mut out = self.clone_owned();
        let n = self.shape[axis];

        // Build the shape of all "fiber prefix" indices (every dim except `axis`)
        let prefix_shape: Vec<usize> = self.shape.iter().enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &d)| d)
            .collect();

        for prefix in IndexIter::new(&prefix_shape) {
            // Pass 1: find max for numerical stability
            let mut max_v = f32::NEG_INFINITY;
            for k in 0..n {
                let idx = insert_at(&prefix, axis, k);
                let v = out.at(&idx);
                if v > max_v { max_v = v; }
            }

            // Pass 2: exp(x - max) and accumulate sum
            let mut sum = 0.0f32;
            for k in 0..n {
                let idx = insert_at(&prefix, axis, k);
                let e = (out.at(&idx) - max_v).exp();
                *out.at_mut(&idx) = e;
                sum += e;
            }

            // Pass 3: normalize
            for k in 0..n {
                let idx = insert_at(&prefix, axis, k);
                *out.at_mut(&idx) /= sum;
            }
        }
        out
    }

    // -------------------------------------------------------------------------
    // Batched matrix multiply
    // -------------------------------------------------------------------------

    /// Batched matrix multiply: `C[..., M, N] = A[..., M, K] @ B[..., K, N]`.
    ///
    /// Operates on the last two dimensions of each array. All leading (batch)
    /// dimensions must match exactly.
    ///
    /// Example shapes:
    /// - 2-D: `[M, K] @ [K, N]` → `[M, N]`
    /// - 3-D: `[B, M, K] @ [B, K, N]` → `[B, M, N]`
    /// - 4-D: `[B, H, T, D] @ [B, H, D, T]` → `[B, H, T, T]`
    pub fn bmm(&self, other: &NDArray) -> NDArray {
        let ndim = self.ndim();
        assert!(ndim >= 2, "bmm: requires at least 2-D arrays");
        assert_eq!(other.ndim(), ndim,
            "bmm: ndim mismatch {} vs {}", ndim, other.ndim());

        let m = self.shape[ndim - 2];
        let k = self.shape[ndim - 1];
        let k2 = other.shape[ndim - 2];
        let n  = other.shape[ndim - 1];

        assert_eq!(k, k2,
            "bmm: inner dims don't match: A has K={} but B has K={}", k, k2);

        let batch_shape = &self.shape[..ndim - 2];
        assert_eq!(batch_shape, &other.shape[..ndim - 2],
            "bmm: batch shape mismatch {:?} vs {:?}", batch_shape, &other.shape[..ndim - 2]);

        let mut out_shape = batch_shape.to_vec();
        out_shape.push(m);
        out_shape.push(n);

        // ------------------------------------------------------------------
        // Metal batched path: extract each batch slice as a contiguous Mat,
        // dispatch all slices in one GPU call, pack results back.
        // Only used when the metal feature is active and the per-slice
        // problem size justifies GPU dispatch.
        // ------------------------------------------------------------------
        #[cfg(feature = "metal")]
        {
            const METAL_BMM_THRESHOLD: usize = 32_768;
            let batch_size: usize = batch_shape.iter().product::<usize>().max(1);
            if m * k * n >= METAL_BMM_THRESHOLD {
                // Make both arrays contiguous so the data slice is reliable.
                let a_cont = self.contiguous();
                let b_cont = other.contiguous();

                // Build (Mat_A, Mat_B) pairs for each batch index.
                let pairs_owned: Vec<(Mat, Mat)> = IndexIter::new(batch_shape)
                    .map(|batch_idx| {
                        // Flat offset of slice start in contiguous storage.
                        let a_off: usize = batch_idx.iter().zip(a_cont.strides.iter())
                            .map(|(i, s)| i * s).sum::<usize>() + a_cont.offset;
                        let b_off: usize = batch_idx.iter().zip(b_cont.strides.iter())
                            .map(|(i, s)| i * s).sum::<usize>() + b_cont.offset;

                        let a_slice = &a_cont.data[a_off .. a_off + m * k];
                        let b_slice = &b_cont.data[b_off .. b_off + k * n];

                        (Mat::new(a_slice.to_vec(), m, k),
                         Mat::new(b_slice.to_vec(), k, n))
                    })
                    .collect();

                let refs: Vec<(&Mat, &Mat)> = pairs_owned.iter()
                    .map(|(a, b)| (a, b))
                    .collect();

                let results = crate::metal_ops::metal_matmul_batched(&refs);

                // Pack the B output Mats back into a single NDArray.
                let total = batch_size * m * n;
                let mut flat = Vec::with_capacity(total);
                for mat in &results {
                    flat.extend_from_slice(&mat.data);
                }
                return NDArray::from_vec(flat, &out_shape);
            }
        }

        // ------------------------------------------------------------------
        // CPU fallback: scalar loop over every (batch_idx, i, p, j).
        // ------------------------------------------------------------------
        let mut out = NDArray::zeros(&out_shape);

        for batch_idx in IndexIter::new(batch_shape) {
            for i in 0..m {
                for p in 0..k {
                    let a_idx: Vec<usize> = batch_idx.iter().cloned()
                        .chain([i, p]).collect();
                    let a_val = self.at(&a_idx);

                    for j in 0..n {
                        let b_idx: Vec<usize> = batch_idx.iter().cloned()
                            .chain([p, j]).collect();
                        let o_idx: Vec<usize> = batch_idx.iter().cloned()
                            .chain([i, j]).collect();
                        *out.at_mut(&o_idx) += a_val * other.at(&b_idx);
                    }
                }
            }
        }
        out
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // c_strides helper
    // -------------------------------------------------------------------------

    #[test]
    fn test_c_strides_1d() {
        assert_eq!(c_strides(&[5]), vec![1]);
    }

    #[test]
    fn test_c_strides_2d() {
        assert_eq!(c_strides(&[3, 4]), vec![4, 1]);
    }

    #[test]
    fn test_c_strides_3d() {
        assert_eq!(c_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn test_c_strides_4d() {
        assert_eq!(c_strides(&[2, 3, 4, 5]), vec![60, 20, 5, 1]);
    }

    #[test]
    fn test_c_strides_empty() {
        assert_eq!(c_strides(&[]), vec![]);
    }

    // -------------------------------------------------------------------------
    // IndexIter
    // -------------------------------------------------------------------------

    #[test]
    fn test_index_iter_2d() {
        let idxs: Vec<Vec<usize>> = IndexIter::new(&[2, 3]).collect();
        assert_eq!(idxs, vec![
            vec![0,0], vec![0,1], vec![0,2],
            vec![1,0], vec![1,1], vec![1,2],
        ]);
    }

    #[test]
    fn test_index_iter_1d() {
        let idxs: Vec<Vec<usize>> = IndexIter::new(&[4]).collect();
        assert_eq!(idxs, vec![vec![0], vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_index_iter_scalar() {
        let idxs: Vec<Vec<usize>> = IndexIter::new(&[]).collect();
        assert_eq!(idxs, vec![vec![]]);
    }

    #[test]
    fn test_index_iter_zero_dim() {
        let idxs: Vec<Vec<usize>> = IndexIter::new(&[0, 3]).collect();
        assert!(idxs.is_empty());
    }

    #[test]
    fn test_index_iter_count() {
        assert_eq!(IndexIter::new(&[2, 3, 4]).count(), 24);
    }

    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    #[test]
    fn test_zeros_shape() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        assert_eq!(nd.shape, vec![2, 3, 4]);
        assert_eq!(nd.strides, vec![12, 4, 1]);
        assert_eq!(nd.numel(), 24);
    }

    #[test]
    fn test_zeros_values() {
        let nd = NDArray::zeros(&[3, 4]);
        for idx in IndexIter::new(&[3, 4]) {
            assert_eq!(nd.at(&idx), 0.0);
        }
    }

    #[test]
    fn test_ones_values() {
        let nd = NDArray::ones(&[2, 3]);
        for idx in IndexIter::new(&[2, 3]) {
            assert_eq!(nd.at(&idx), 1.0);
        }
    }

    #[test]
    fn test_from_fn() {
        let nd = NDArray::from_fn(&[2, 3], |idx| (idx[0] * 3 + idx[1]) as f32);
        assert_eq!(nd.at(&[0, 0]), 0.0);
        assert_eq!(nd.at(&[1, 2]), 5.0);
        assert_eq!(nd.at(&[0, 2]), 2.0);
    }

    #[test]
    fn test_from_vec_correct() {
        let nd = NDArray::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(nd.at(&[0, 0]), 1.0);
        assert_eq!(nd.at(&[1, 2]), 6.0);
    }

    #[test]
    #[should_panic(expected = "from_vec")]
    fn test_from_vec_wrong_len_panics() {
        NDArray::from_vec(vec![1.0, 2.0], &[2, 3]);
    }

    // -------------------------------------------------------------------------
    // Flat index
    // -------------------------------------------------------------------------

    #[test]
    fn test_flat_index_2d() {
        let nd = NDArray::zeros(&[3, 4]);
        // [1, 2] → 1*4 + 2 = 6
        assert_eq!(nd.flat_index(&[1, 2]), 6);
    }

    #[test]
    fn test_flat_index_3d() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        // [1, 2, 3] → 1*12 + 2*4 + 3 = 23
        assert_eq!(nd.flat_index(&[1, 2, 3]), 23);
    }

    // -------------------------------------------------------------------------
    // at / at_mut
    // -------------------------------------------------------------------------

    #[test]
    fn test_at_mut_write_read() {
        let mut nd = NDArray::zeros(&[3, 4]);
        *nd.at_mut(&[2, 3]) = 99.0;
        assert_eq!(nd.at(&[2, 3]), 99.0);
        assert_eq!(nd.at(&[0, 0]), 0.0);
    }

    // -------------------------------------------------------------------------
    // Mat interop
    // -------------------------------------------------------------------------

    #[test]
    fn test_from_mat_shape() {
        let m = Mat::from_fn(3, 4, |r, c| (r * 4 + c) as f32);
        let nd = NDArray::from_mat(&m);
        assert_eq!(nd.shape, vec![3, 4]);
        assert_eq!(nd.strides, vec![4, 1]);
        assert_eq!(nd.offset, 0);
    }

    #[test]
    fn test_from_mat_values() {
        let m = Mat::from_fn(3, 4, |r, c| (r * 4 + c) as f32);
        let nd = NDArray::from_mat(&m);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(nd.at(&[r, c]), m.at(r, c));
            }
        }
    }

    #[test]
    fn test_into_mat_round_trip() {
        let m = Mat::from_fn(3, 4, |r, c| (r * 4 + c) as f32);
        let nd = NDArray::from_mat(&m);
        let m2 = nd.into_mat();
        assert_eq!(m.rows, m2.rows);
        assert_eq!(m.cols, m2.cols);
        assert_eq!(m.data, m2.data);
    }

    #[test]
    #[should_panic(expected = "into_mat")]
    fn test_into_mat_wrong_ndim_panics() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        nd.into_mat();
    }

    // -------------------------------------------------------------------------
    // Contiguity
    // -------------------------------------------------------------------------

    #[test]
    fn test_is_contiguous_fresh() {
        let nd = NDArray::zeros(&[2, 3]);
        assert!(nd.is_contiguous());
    }

    #[test]
    fn test_contiguous_preserves_values() {
        let nd = NDArray::from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f32);
        let c = nd.permute(&[1, 0]); // non-contiguous
        assert!(!c.is_contiguous());
        let cc = c.contiguous();
        assert!(cc.is_contiguous());
        for i in 0..4 {
            for j in 0..3 {
                assert_eq!(cc.at(&[i, j]), nd.at(&[j, i]));
            }
        }
    }

    // -------------------------------------------------------------------------
    // Reshape
    // -------------------------------------------------------------------------

    #[test]
    fn test_reshape_shape() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        let r = nd.reshape(&[6, 4]);
        assert_eq!(r.shape, vec![6, 4]);
        assert_eq!(r.strides, vec![4, 1]);
    }

    #[test]
    fn test_reshape_values() {
        let nd = NDArray::from_fn(&[24], |idx| idx[0] as f32);
        let r = nd.reshape(&[2, 3, 4]);
        // element [1,2,3] → flat index 1*12 + 2*4 + 3 = 23
        assert_eq!(r.at(&[1, 2, 3]), 23.0);
    }

    #[test]
    fn test_reshape_flat_to_2d() {
        let nd = NDArray::from_fn(&[6], |idx| idx[0] as f32);
        let r = nd.reshape(&[2, 3]);
        assert_eq!(r.at(&[0, 0]), 0.0);
        assert_eq!(r.at(&[1, 2]), 5.0);
    }

    #[test]
    #[should_panic(expected = "reshape")]
    fn test_reshape_wrong_size_panics() {
        let nd = NDArray::zeros(&[2, 3]);
        nd.reshape(&[5]);
    }

    #[test]
    fn test_reshape_after_permute() {
        // permute → non-contiguous; reshape forces a copy first
        let nd = NDArray::from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f32);
        let p = nd.permute(&[1, 0]); // [4, 3]
        let r = p.reshape(&[12]);     // must make contiguous first
        assert_eq!(r.shape, vec![12]);
        assert_eq!(r.numel(), 12);
        // Verify values are transposed order: p[i,j] = nd[j,i]
        // flat order of p: p[0,0], p[0,1], p[0,2], p[1,0], ...
        // p[0,0] = nd[0,0] = 0; p[0,1] = nd[1,0] = 4; p[0,2] = nd[2,0] = 8
        assert_eq!(r.at(&[0]), 0.0);
        assert_eq!(r.at(&[1]), 4.0);
        assert_eq!(r.at(&[2]), 8.0);
    }

    // -------------------------------------------------------------------------
    // Permute
    // -------------------------------------------------------------------------

    #[test]
    fn test_permute_2d_transpose_shape() {
        let nd = NDArray::zeros(&[3, 4]);
        let t = nd.permute(&[1, 0]);
        assert_eq!(t.shape, vec![4, 3]);
    }

    #[test]
    fn test_permute_2d_transpose_values() {
        let nd = NDArray::from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f32);
        let t = nd.permute(&[1, 0]);
        for r in 0..3 {
            for c in 0..4 {
                assert_eq!(t.at(&[c, r]), nd.at(&[r, c]));
            }
        }
    }

    #[test]
    fn test_permute_3d_shape() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        let p = nd.permute(&[2, 0, 1]);
        assert_eq!(p.shape, vec![4, 2, 3]);
    }

    #[test]
    fn test_permute_identity() {
        let nd = NDArray::zeros(&[2, 3, 4]);
        let p = nd.permute(&[0, 1, 2]);
        assert_eq!(p.shape, nd.shape);
        assert_eq!(p.strides, nd.strides);
    }

    #[test]
    fn test_permute_not_contiguous() {
        let nd = NDArray::zeros(&[3, 4]);
        let p = nd.permute(&[1, 0]);
        assert!(!p.is_contiguous());
    }

    #[test]
    #[should_panic(expected = "duplicate axis")]
    fn test_permute_duplicate_axis_panics() {
        let nd = NDArray::zeros(&[3, 4]);
        nd.permute(&[0, 0]);
    }

    // -------------------------------------------------------------------------
    // Slice
    // -------------------------------------------------------------------------

    #[test]
    fn test_slice_removes_dim() {
        let nd = NDArray::zeros(&[4, 3, 2]);
        let s = nd.slice(0, 2);
        assert_eq!(s.shape, vec![3, 2]);
    }

    #[test]
    fn test_slice_values() {
        let nd = NDArray::from_fn(&[4, 3, 2], |idx| (idx[0] * 6 + idx[1] * 2 + idx[2]) as f32);
        let s = nd.slice(0, 2); // batch element 2
        assert_eq!(s.at(&[1, 0]), nd.at(&[2, 1, 0]));
        assert_eq!(s.at(&[2, 1]), nd.at(&[2, 2, 1]));
    }

    #[test]
    fn test_slice_middle_dim() {
        let nd = NDArray::from_fn(&[2, 5, 3], |idx| idx[1] as f32);
        let s = nd.slice(1, 3);
        assert_eq!(s.shape, vec![2, 3]);
        // All values in this slice should have the original dim-1 == 3
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(s.at(&[i, j]), 3.0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn test_slice_out_of_bounds_panics() {
        let nd = NDArray::zeros(&[4, 3]);
        nd.slice(0, 10);
    }

    // -------------------------------------------------------------------------
    // expand_dims
    // -------------------------------------------------------------------------

    #[test]
    fn test_expand_dims_front() {
        let nd = NDArray::zeros(&[3, 4]);
        let e = nd.expand_dims(0);
        assert_eq!(e.shape, vec![1, 3, 4]);
    }

    #[test]
    fn test_expand_dims_middle() {
        let nd = NDArray::zeros(&[3, 4]);
        let e = nd.expand_dims(1);
        assert_eq!(e.shape, vec![3, 1, 4]);
    }

    #[test]
    fn test_expand_dims_back() {
        let nd = NDArray::zeros(&[3, 4]);
        let e = nd.expand_dims(2);
        assert_eq!(e.shape, vec![3, 4, 1]);
    }

    #[test]
    fn test_expand_dims_values_preserved() {
        let nd = NDArray::from_fn(&[3], |idx| idx[0] as f32);
        let e = nd.expand_dims(0); // [1, 3]
        assert_eq!(e.at(&[0, 0]), 0.0);
        assert_eq!(e.at(&[0, 2]), 2.0);
    }

    // -------------------------------------------------------------------------
    // Elementwise ops
    // -------------------------------------------------------------------------

    #[test]
    fn test_add() {
        let a = NDArray::ones(&[2, 3]);
        let b = NDArray::ones(&[2, 3]).scale(2.0);
        let c = a.add(&b);
        for idx in IndexIter::new(&[2, 3]) {
            assert_eq!(c.at(&idx), 3.0);
        }
    }

    #[test]
    #[should_panic(expected = "add: shape mismatch")]
    fn test_add_shape_mismatch_panics() {
        let a = NDArray::zeros(&[2, 3]);
        let b = NDArray::zeros(&[3, 2]);
        a.add(&b);
    }

    #[test]
    fn test_mul() {
        let a = NDArray::ones(&[2, 3]).scale(3.0);
        let b = NDArray::ones(&[2, 3]).scale(4.0);
        let c = a.mul(&b);
        assert_eq!(c.at(&[0, 0]), 12.0);
        assert_eq!(c.at(&[1, 2]), 12.0);
    }

    #[test]
    fn test_scale() {
        let a = NDArray::from_fn(&[3], |idx| idx[0] as f32);
        let s = a.scale(2.0);
        assert_eq!(s.at(&[0]), 0.0);
        assert_eq!(s.at(&[1]), 2.0);
        assert_eq!(s.at(&[2]), 4.0);
    }

    #[test]
    fn test_add_scalar() {
        let a = NDArray::from_fn(&[3], |idx| idx[0] as f32);
        let s = a.add_scalar(10.0);
        assert_eq!(s.at(&[0]), 10.0);
        assert_eq!(s.at(&[2]), 12.0);
    }

    #[test]
    fn test_map() {
        let a = NDArray::from_fn(&[3], |idx| idx[0] as f32);
        let b = a.map(|x| x * x);
        assert_eq!(b.at(&[0]), 0.0);
        assert_eq!(b.at(&[1]), 1.0);
        assert_eq!(b.at(&[2]), 4.0);
    }

    // -------------------------------------------------------------------------
    // reduce_sum
    // -------------------------------------------------------------------------

    #[test]
    fn test_reduce_sum_axis0() {
        let nd = NDArray::ones(&[3, 4]);
        let r = nd.reduce_sum(0);
        assert_eq!(r.shape, vec![4]);
        for c in 0..4 {
            assert!((r.at(&[c]) - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduce_sum_axis1() {
        let nd = NDArray::ones(&[3, 4]);
        let r = nd.reduce_sum(1);
        assert_eq!(r.shape, vec![3]);
        for row in 0..3 {
            assert!((r.at(&[row]) - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduce_sum_3d_middle() {
        let nd = NDArray::ones(&[2, 3, 4]);
        let r = nd.reduce_sum(1);
        assert_eq!(r.shape, vec![2, 4]);
        for i in 0..2 {
            for j in 0..4 {
                assert!((r.at(&[i, j]) - 3.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_reduce_sum_values() {
        // nd[i,j] = i*3 + j; sum over j (axis=1) for row i = sum(i*3+0, i*3+1, i*3+2) = 3*i*3+3 = 9i+3
        let nd = NDArray::from_fn(&[4, 3], |idx| (idx[0] * 3 + idx[1]) as f32);
        let r = nd.reduce_sum(1);
        for i in 0..4 {
            let expected = (9 * i + 3) as f32;
            assert!((r.at(&[i]) - expected).abs() < 1e-5,
                "row {}: expected {}, got {}", i, expected, r.at(&[i]));
        }
    }

    // -------------------------------------------------------------------------
    // softmax
    // -------------------------------------------------------------------------

    #[test]
    fn test_softmax_sums_to_one() {
        let nd = NDArray::from_fn(&[3, 4], |idx| (idx[0] * 4 + idx[1]) as f32);
        let s = nd.softmax(1);
        for r in 0..3 {
            let total: f32 = (0..4).map(|c| s.at(&[r, c])).sum();
            assert!((total - 1.0).abs() < 1e-6, "row {} sum = {}", r, total);
        }
    }

    #[test]
    fn test_softmax_uniform_input() {
        let nd = NDArray::ones(&[2, 4]);
        let s = nd.softmax(1);
        for i in 0..2 {
            for j in 0..4 {
                assert!((s.at(&[i, j]) - 0.25).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_softmax_axis0() {
        let nd = NDArray::ones(&[4, 3]);
        let s = nd.softmax(0);
        for c in 0..3 {
            let col_sum: f32 = (0..4).map(|r| s.at(&[r, c])).sum();
            assert!((col_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax_large_values_stable() {
        // Should not overflow / produce NaN due to max subtraction
        let nd = NDArray::from_fn(&[1, 4], |idx| idx[1] as f32 * 1000.0);
        let s = nd.softmax(1);
        // All probability should be concentrated on the last element
        assert!(s.at(&[0, 3]) > 0.999, "expected ~1.0, got {}", s.at(&[0, 3]));
        assert!(s.at(&[0, 0]).is_finite());
    }

    #[test]
    fn test_softmax_3d_last_axis() {
        let nd = NDArray::ones(&[2, 3, 8]);
        let s = nd.softmax(2);
        assert_eq!(s.shape, vec![2, 3, 8]);
        for i in 0..2 {
            for j in 0..3 {
                let total: f32 = (0..8).map(|k| s.at(&[i, j, k])).sum();
                assert!((total - 1.0).abs() < 1e-6);
            }
        }
    }

    // -------------------------------------------------------------------------
    // bmm
    // -------------------------------------------------------------------------

    #[test]
    fn test_bmm_2d_shape() {
        let a = NDArray::zeros(&[3, 4]);
        let b = NDArray::zeros(&[4, 5]);
        let c = a.bmm(&b);
        assert_eq!(c.shape, vec![3, 5]);
    }

    #[test]
    fn test_bmm_2d_values() {
        // Identity-like: a=[3,4] all ones, b=[4,5] all ones → c=[3,5] all 4.0
        let a = NDArray::ones(&[3, 4]);
        let b = NDArray::ones(&[4, 5]);
        let c = a.bmm(&b);
        for idx in IndexIter::new(&[3, 5]) {
            assert!((c.at(&idx) - 4.0).abs() < 1e-5, "at {:?}: {}", idx, c.at(&idx));
        }
    }

    #[test]
    fn test_bmm_3d_shape() {
        let a = NDArray::zeros(&[2, 3, 4]);
        let b = NDArray::zeros(&[2, 4, 5]);
        let c = a.bmm(&b);
        assert_eq!(c.shape, vec![2, 3, 5]);
    }

    #[test]
    fn test_bmm_3d_values() {
        // Each element of C[b,i,j] = sum_k A[b,i,k]*B[b,k,j] = K * 1.0 * 1.0 = K
        let k = 4usize;
        let a = NDArray::ones(&[2, 3, k]);
        let b = NDArray::ones(&[2, k, 5]);
        let c = a.bmm(&b);
        assert_eq!(c.shape, vec![2, 3, 5]);
        for idx in IndexIter::new(&[2, 3, 5]) {
            assert!((c.at(&idx) - k as f32).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bmm_4d_shape() {
        // [B, H, T, D] @ [B, H, D, T] → [B, H, T, T]  (attention scores)
        let (b, h, t, d) = (2, 4, 8, 16);
        let q = NDArray::zeros(&[b, h, t, d]);
        let k = NDArray::zeros(&[b, h, d, t]);
        let scores = q.bmm(&k);
        assert_eq!(scores.shape, vec![b, h, t, t]);
    }

    #[test]
    fn test_bmm_4d_values() {
        let (b, h, t, d) = (1, 2, 4, 8);
        let q = NDArray::ones(&[b, h, t, d]).scale(0.1);
        let k = NDArray::ones(&[b, h, d, t]).scale(0.1);
        let scores = q.bmm(&k);
        // each element = D * 0.1 * 0.1 = D * 0.01
        let expected = d as f32 * 0.01;
        for idx in IndexIter::new(&[b, h, t, t]) {
            assert!((scores.at(&idx) - expected).abs() < 1e-4,
                "at {:?}: expected {}, got {}", idx, expected, scores.at(&idx));
        }
    }

    #[test]
    fn test_bmm_vs_mat_matmul() {
        let m = Mat::from_fn(3, 4, |r, c| (r * 4 + c) as f32);
        let n = Mat::from_fn(4, 5, |r, c| (r * 5 + c) as f32);
        let c_mat = m.matmul(&n);

        let a_nd = NDArray::from_mat(&m);
        let b_nd = NDArray::from_mat(&n);
        let c_nd = a_nd.bmm(&b_nd);

        assert_eq!(c_nd.shape, vec![3, 5]);
        for r in 0..3 {
            for c in 0..5 {
                assert!((c_nd.at(&[r, c]) - c_mat.at(r, c)).abs() < 1e-4,
                    "mismatch at [{r},{c}]: nd={} mat={}", c_nd.at(&[r,c]), c_mat.at(r,c));
            }
        }
    }

    #[test]
    #[should_panic(expected = "inner dims don't match")]
    fn test_bmm_inner_dim_mismatch_panics() {
        let a = NDArray::zeros(&[3, 4]);
        let b = NDArray::zeros(&[5, 3]); // 4 != 5
        a.bmm(&b);
    }

    #[test]
    #[should_panic(expected = "batch shape mismatch")]
    fn test_bmm_batch_dim_mismatch_panics() {
        let a = NDArray::zeros(&[2, 3, 4]);
        let b = NDArray::zeros(&[3, 4, 5]); // batch 2 != 3
        a.bmm(&b);
    }

    // -------------------------------------------------------------------------
    // Attention head pattern (integration)
    // -------------------------------------------------------------------------

    #[test]
    fn test_attention_head_bmm_pattern() {
        // Simulate what batched attention does:
        // Q, K: [B, H, T, D_head]
        // scores = Q.bmm(K.permute([0,1,3,2])) → [B, H, T, T]
        let (b, h, t, d) = (1, 4, 8, 16);
        let q = NDArray::from_fn(&[b, h, t, d], |idx| idx[3] as f32 * 0.01);
        let k = NDArray::from_fn(&[b, h, t, d], |idx| idx[3] as f32 * 0.01);

        let k_t = k.permute(&[0, 1, 3, 2]); // [B, H, D, T]
        assert_eq!(k_t.shape, vec![b, h, d, t]);

        let scores = q.bmm(&k_t);
        assert_eq!(scores.shape, vec![b, h, t, t]);

        // Each score[b,h,i,j] = sum_d Q[b,h,i,d]*K[b,h,j,d]
        // = sum_d (d*0.01)^2 = 0.0001 * sum(0^2..15^2) = 0.0001 * 1240 = 0.124
        let expected: f32 = (0..d).map(|dv| (dv as f32 * 0.01).powi(2)).sum();
        for bi in 0..b { for hi in 0..h { for ti in 0..t { for tj in 0..t {
            let got = scores.at(&[bi, hi, ti, tj]);
            assert!((got - expected).abs() < 1e-4,
                "scores[{bi},{hi},{ti},{tj}] = {got}, expected {expected}");
        }}}}
    }
}
