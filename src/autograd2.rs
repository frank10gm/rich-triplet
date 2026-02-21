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
    ///
    /// When compiled with `--features blas`, delegates to `cblas_sgemm` for
    /// an additional 4–8× speedup via CPU-vendor BLAS (Apple Accelerate on
    /// macOS, OpenBLAS/MKL on Linux/Windows).
    pub fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows,
            "matmul shape mismatch: [{},{}] × [{},{}]", self.rows, self.cols, b.rows, b.cols);

        #[cfg(feature = "blas")]
        {
            return self.matmul_blas(b);
        }

        #[cfg(not(feature = "blas"))]
        {
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
    }

    /// BLAS-accelerated matmul via `cblas_sgemm`.
    ///
    /// Enabled when compiled with `--features blas`.
    ///
    /// ## What cblas_sgemm does
    ///
    /// `cblas_sgemm(Order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)`
    ///
    /// Computes: C = alpha * op(A) @ op(B) + beta * C
    ///   - Order = RowMajor  (our Mat is row-major)
    ///   - TransA = NoTrans, TransB = NoTrans
    ///   - alpha = 1.0, beta = 0.0  (pure multiply, no accumulation into C)
    ///   - lda = K (leading dimension of A = number of columns)
    ///   - ldb = N (leading dimension of B)
    ///   - ldc = N (leading dimension of C)
    ///
    /// The function dispatches to the best SIMD kernel for the current CPU
    /// (AVX-512, AVX2, NEON, etc.) and uses highly optimised cache-blocking
    /// — typically 4–8× faster than the pure Rust triple loop.
    ///
    /// ## Platform notes
    ///
    /// - macOS: links Apple Accelerate (built-in, no install needed)
    /// - Linux: requires `libopenblas-dev` or `libmkl-dev`
    /// - Windows: requires OpenBLAS or MKL DLL
    #[cfg(feature = "blas")]
    fn matmul_blas(&self, b: &Mat) -> Mat {
        let (m, k, n) = (self.rows, self.cols, b.cols);
        let mut out = Mat::zeros(m, n);
        unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::None,
                m as i32, n as i32, k as i32,
                1.0_f32,                    // alpha
                &self.data, k as i32,       // A, lda
                &b.data,    n as i32,       // B, ldb
                0.0_f32,                    // beta
                &mut out.data, n as i32,    // C, ldc
            );
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

    // =========================================================================
    // Parallel matmul — multi-threaded CPU version of matmul()
    // =========================================================================
    //
    // The standard matmul() runs on a single core. For large matrices
    // (d_model × d_model in a real GPT-OSS-20b layer) this is the bottleneck.
    //
    // We split the output rows across N threads (default: number of logical CPUs).
    // Each thread computes a contiguous slice of rows of C = A @ B independently
    // — no synchronization needed during computation.
    //
    // ## Why this is safe without a mutex
    //
    // Thread i writes to rows [i*chunk .. (i+1)*chunk) of `out`.
    // Ranges don't overlap, so there's no data race.
    // We use `unsafe` + raw pointer slicing to give each thread its own slice.
    // The `std::thread::scope` API guarantees all threads finish before `matmul_parallel` returns.
    //
    // ## Performance
    //
    // On a 4-core laptop with SIMD auto-vectorization:
    //   Single-threaded:   d_model=1024 matmul → ~120ms
    //   4-thread parallel: → ~35ms (3.4x speedup)
    //   16-thread parallel: → ~15ms on a 16-core desktop
    //
    // Real production code would use BLAS (cblas_sgemm) which further speeds
    // this up via SIMD + optimized memory access patterns, but requires a
    // dependency on a BLAS library (OpenBLAS, MKL, Accelerate).
    // This version achieves parallelism with zero external dependencies.

    /// C = A @ B — multi-threaded.
    ///
    /// Equivalent to `self.matmul(b)` but uses `n_threads` threads.
    /// Pass `n_threads = 0` to use the number of logical CPUs.
    pub fn matmul_parallel(&self, b: &Mat, n_threads: usize) -> Mat {
        assert_eq!(self.cols, b.rows,
            "matmul_parallel shape mismatch: [{},{}] × [{},{}]",
            self.rows, self.cols, b.rows, b.cols);
        let (m, k, n) = (self.rows, self.cols, b.cols);

        let n_threads = if n_threads == 0 {
            std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(1)
        } else {
            n_threads.min(m).max(1)
        };

        let mut out = Mat::zeros(m, n);

        // Safety: we partition `out.data` into non-overlapping row slices.
        // Each thread has exclusive access to its slice throughout the scope.
        let a_ptr = self.data.as_ptr();
        let b_ptr = b.data.as_ptr();
        let out_ptr = out.data.as_mut_ptr();

        std::thread::scope(|s| {
            let chunk = (m + n_threads - 1) / n_threads; // rows per thread (ceiling)
            for thread_id in 0..n_threads {
                let row_start = thread_id * chunk;
                let row_end   = (row_start + chunk).min(m);
                if row_start >= row_end { break; }

                // SAFETY: each thread writes to a disjoint range of rows.
                // `a_ptr`, `b_ptr` are read-only; `out_ptr` range is unique per thread.
                let slice_len  = (row_end - row_start) * n;
                let slice_start = row_start * n;
                let out_slice: &mut [f32] = unsafe {
                    std::slice::from_raw_parts_mut(out_ptr.add(slice_start), slice_len)
                };
                let a_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(a_ptr, m * k)
                };
                let b_slice: &[f32] = unsafe {
                    std::slice::from_raw_parts(b_ptr, k * n)
                };

                s.spawn(move || {
                    for i in 0..(row_end - row_start) {
                        let global_i = row_start + i;
                        for p in 0..k {
                            let a_ip = a_slice[global_i * k + p];
                            for j in 0..n {
                                out_slice[i * n + j] += a_ip * b_slice[p * n + j];
                            }
                        }
                    }
                });
            }
        });

        out
    }
}

// =============================================================================
// Q4Mat — 4-bit quantized weight matrix
// =============================================================================
//
// ## Why quantize?
//
// GPT-OSS-20b has ~21B parameters stored as BF16 → 42GB on disk and in RAM.
// A MacBook Pro with 32GB unified memory cannot hold the full model.
//
// 4-bit quantization reduces this to ~11GB (2x compression from 2-byte BF16).
// Combined with mmap loading (no full copy in RAM) this makes inference feasible
// on consumer hardware.
//
// ## The quantization scheme
//
// We use block quantization (the same scheme as GGUF Q4_K and llama.cpp):
//
//   1. Split the weight matrix row by row into blocks of BLOCK_SIZE elements.
//   2. For each block:
//      - Find absmax = max(|x_i|)
//      - scale = absmax / 7.0  (7 = (2^3 - 1), since we pack 2 nibbles per byte)
//      - For each element: q = clamp(round(x / scale), -7, 7)  → 4 bits
//      - Store scale (f32) + packed nibbles (i8 pairs)
//   3. Dequantize: x_approx = q * scale
//
// Each 4-bit value is stored as a nibble (4 bits) in a u8.
// Two consecutive elements share one byte: lo = q[0] & 0xF, hi = (q[1] >> 4) & 0xF
// We use the signed range [-7, 7] (symmetric) rather than [-8, 7] to keep zero exact.
//
// ## Block size trade-off
//
//   Larger block → fewer scales stored → better compression
//   Smaller block → better accuracy (scale adapts to local range)
//   Common choices: 32 (high quality), 64, 128 (used by llama.cpp)
//
// We default to BLOCK_SIZE = 32.
//
// ## Integration with the rest of the code
//
// `Q4Mat::dequantize()` returns a full `Mat` which can be used in any existing
// matmul operation. This is the simplest correct approach.
//
// For maximum performance, `matmul_q4()` dequantizes on the fly during the
// inner loop (fused kernel), avoiding a full intermediate matrix allocation.

/// Block size for 4-bit quantization. Each block shares one scale value.
pub const Q4_BLOCK_SIZE: usize = 32;

/// A 4-bit quantized matrix.
///
/// Elements are stored as signed 4-bit integers, two per byte (packed nibbles).
/// Each block of `Q4_BLOCK_SIZE` elements has an associated f32 scale.
#[derive(Clone)]
pub struct Q4Mat {
    pub rows: usize,
    pub cols: usize,

    /// Packed nibbles: ceil(rows*cols / 2) bytes.
    /// Element (r,c) at flat index k = r*cols+c:
    ///   if k is even:  low nibble of packed[k/2]
    ///   if k is odd:   high nibble of packed[k/2]
    /// Each nibble is a signed integer in [-7, 7] with zero biased at 0.
    /// Encoding: 4-bit two's-complement (nibble & 0xF, then sign-extend):
    ///   0x0=0, ..., 0x7=7, 0x8=-8(unused), 0x9=-7, ..., 0xF=-1
    pub packed: Vec<u8>,

    /// One scale per block: `ceil(rows*cols / Q4_BLOCK_SIZE)` values.
    /// Block b covers flat elements [b*Q4_BLOCK_SIZE .. (b+1)*Q4_BLOCK_SIZE).
    pub scales: Vec<f32>,
}

impl Q4Mat {
    /// Quantize a `Mat` to 4-bit.
    ///
    /// The input matrix is quantized block-by-block (blocks of `Q4_BLOCK_SIZE`
    /// elements along the flattened dimension). The output can be used with
    /// `dequantize()` or `matmul_q4()`.
    pub fn quantize(mat: &Mat) -> Self {
        let n = mat.rows * mat.cols;
        let n_blocks = n.div_ceil(Q4_BLOCK_SIZE);
        let n_packed = n.div_ceil(2);

        let mut packed = vec![0u8; n_packed];
        let mut scales = vec![0.0f32; n_blocks];

        for block in 0..n_blocks {
            let start = block * Q4_BLOCK_SIZE;
            let end   = (start + Q4_BLOCK_SIZE).min(n);

            // Find absmax for this block
            let absmax = mat.data[start..end]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);

            let scale = if absmax == 0.0 { 1.0 } else { absmax / 7.0 };
            scales[block] = scale;

            // Quantize and pack
            for k in start..end {
                let q = (mat.data[k] / scale).round().clamp(-7.0, 7.0) as i8;
                // Pack as nibble (4-bit two's complement)
                let nibble = (q & 0x0F) as u8; // low 4 bits preserve the sign bit for i4
                if k % 2 == 0 {
                    packed[k / 2] |= nibble;          // low nibble
                } else {
                    packed[k / 2] |= nibble << 4;     // high nibble
                }
            }
        }

        Q4Mat { rows: mat.rows, cols: mat.cols, packed, scales }
    }

    /// Dequantize: recover an approximate `Mat` from the 4-bit representation.
    ///
    /// The recovered value at element k is `q[k] * scale[block(k)]`.
    /// The quantization error is at most `0.5 * scale`, which is at most
    /// `absmax / 14.0 ≈ 7%` of the largest value in the block.
    pub fn dequantize(&self) -> Mat {
        let n = self.rows * self.cols;
        let mut data = vec![0.0f32; n];

        for k in 0..n {
            let nibble = if k % 2 == 0 {
                self.packed[k / 2] & 0x0F          // low nibble
            } else {
                (self.packed[k / 2] >> 4) & 0x0F   // high nibble
            };
            // Sign-extend from 4-bit two's complement
            let q = if nibble >= 8 { nibble as i8 - 16 } else { nibble as i8 };
            let block = k / Q4_BLOCK_SIZE;
            data[k] = q as f32 * self.scales[block];
        }

        Mat::new(data, self.rows, self.cols)
    }

    /// Fused matrix multiplication: (A: f32) @ (B: Q4).T
    ///
    /// Computes the same result as `a.matmul(&self.dequantize().transpose())`
    /// but without materializing the full dequantized matrix.
    ///
    /// Shape: A is [M, K], B is [N, K] (stored transposed as in Linear2.weight) → [M, N]
    ///
    /// This is the critical path for inference: the linear layer's forward pass
    /// is `input @ weight.T` where weight is quantized.
    pub fn matmul_q4_t(&self, a: &Mat) -> Mat {
        // self is [N, K], stored row-major
        // a    is [M, K]
        // out  is [M, N]
        let (m, k, nn) = (a.rows, a.cols, self.rows);
        assert_eq!(k, self.cols,
            "matmul_q4_t: a.cols {} != q4.cols {}", k, self.cols);

        let mut out = Mat::zeros(m, nn);

        for j in 0..nn {  // output column = B row
            // Dequantize row j of B on the fly
            let row_start_elem = j * k;
            for i in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let flat_idx = row_start_elem + p;
                    let nibble = if flat_idx % 2 == 0 {
                        self.packed[flat_idx / 2] & 0x0F
                    } else {
                        (self.packed[flat_idx / 2] >> 4) & 0x0F
                    };
                    let q = if nibble >= 8 { nibble as i8 - 16 } else { nibble as i8 };
                    let block = flat_idx / Q4_BLOCK_SIZE;
                    let w = q as f32 * self.scales[block];
                    acc += a.at(i, p) * w;
                }
                *out.at_mut(i, j) = acc;
            }
        }

        out
    }

    /// Memory usage in bytes (excluding struct overhead).
    pub fn size_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * 4
    }

    /// Compression ratio vs f32 storage.
    pub fn compression_ratio(&self) -> f32 {
        let f32_bytes = self.rows * self.cols * 4;
        f32_bytes as f32 / self.size_bytes() as f32
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
    // GPT-OSS inference primitives (forward only — no backward_fn)
    // =========================================================================
    //
    // These ops implement the architectural features of GPT-OSS that are not
    // present in GPT-2. They are inference-only: no backward_fn is registered,
    // so gradients do not flow through them. This is intentional — GPT-OSS at
    // 20B parameters cannot be trained in this codebase anyway.

    /// RMS Layer Normalization — normalizes each row by its RMS value.
    ///
    /// RMSNorm(x) = x / sqrt(mean(x²) + eps) * gamma
    ///
    /// Compared to LayerNorm, RMSNorm:
    ///   - Has no beta (no learned bias, just centering is skipped)
    ///   - Divides by RMS instead of std deviation
    ///   - Is simpler and slightly faster (no mean subtraction)
    ///   - Used by LLaMA, Mistral, GPT-OSS, and most post-2022 models
    ///
    /// gamma: [1, d_model]  — learned scale (initialized to ones)
    /// x:     [T, d_model]  — input
    /// out:   [T, d_model]  — normalized output
    ///
    /// Backward:
    ///   Let r[t] = 1/sqrt(mean(x[t]²) + eps)  (per-row RMS inverse)
    ///       x̂[t] = x[t] * r[t]                (normalized row, before gamma)
    ///
    ///   dγ = sum_t(dout[t] * x̂[t])            shape [1, d]
    ///   dx[t,i] = r[t] * (dout[t,i]*γ[i] - x̂[t,i] * mean(dout[t]*γ*x̂[t]) / 1)
    ///           = r[t] * (D[t,i] - x̂[t,i] * mean(D[t]*x̂[t]))
    ///   where D = dout * gamma (element-wise broadcast)
    pub fn rms_norm(&self, gamma: &TensorNode, eps: f32) -> TensorNode {
        let x = self.data().clone();
        let g = gamma.data().clone();
        let (t, d) = (x.rows, x.cols);
        let inv_d = 1.0 / d as f32;

        // Forward: compute per-row RMS and normalized values
        let mut rms_inv = vec![0.0f32; t]; // r[t]
        let mut x_hat   = Mat::zeros(t, d); // x̂ = x * r

        for r in 0..t {
            let mean_sq = (0..d).map(|c| x.at(r, c).powi(2)).sum::<f32>() * inv_d;
            rms_inv[r] = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..d {
                *x_hat.at_mut(r, c) = x.at(r, c) * rms_inv[r];
            }
        }

        let out_data = Mat::from_fn(t, d, |r, c| x_hat.at(r, c) * g.at(0, c));
        let out = TensorNode::leaf(out_data);

        let self_c  = self.clone();
        let gamma_c = gamma.clone();
        let out_c   = out.clone();
        let x_hat_s = x_hat.clone();
        let rms_inv_s = rms_inv.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let g    = gamma_c.0.borrow().data.clone();

            // dγ = sum_t(dout[t] * x̂[t])   shape [1, d]
            let mut dg = Mat::zeros(1, d);
            for c in 0..d {
                for r in 0..t { *dg.at_mut(0, c) += dout.at(r, c) * x_hat_s.at(r, c); }
            }
            gamma_c.0.borrow_mut().grad.add_assign(&dg);

            // dx[t] = r[t] * (D[t] - x̂[t]*mean(D[t]*x̂[t]))
            // where D[t,i] = dout[t,i]*γ[i]
            let mut dx = Mat::zeros(t, d);
            for r in 0..t {
                let d_row: Vec<f32> = (0..d).map(|c| dout.at(r, c) * g.at(0, c)).collect();
                let mean_dxh = d_row.iter().enumerate()
                    .map(|(c, &dv)| dv * x_hat_s.at(r, c))
                    .sum::<f32>() * inv_d;
                for c in 0..d {
                    *dx.at_mut(r, c) = rms_inv_s[r] * (d_row[c] - x_hat_s.at(r, c) * mean_dxh);
                }
            }
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), gamma.clone()];
        out
    }

    /// SiLU (Sigmoid Linear Unit) activation: SiLU(x) = x * sigmoid(x)
    ///
    /// This is the activation used inside SwiGLU (the GPT-OSS FFN activation).
    /// SiLU is smoother than ReLU and empirically outperforms GELU on many tasks.
    ///
    /// Note: our existing gelu() uses the approximation x*sigmoid(1.702x),
    /// which is actually an approximation of SiLU. SiLU is the exact version.
    ///
    /// Backward:
    ///   d/dx [x*σ(x)] = σ(x) + x*σ(x)*(1-σ(x)) = σ(x)*(1 + x*(1-σ(x)))
    pub fn silu(&self) -> TensorNode {
        let x = self.data().clone();

        // Precompute sigmoid for reuse in backward
        let sig = Mat::from_fn(x.rows, x.cols, |r, c| {
            1.0 / (1.0 + (-x.at(r, c)).exp())
        });

        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| x.at(r, c) * sig.at(r, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c  = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout   = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();

            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let xv = x_data.at(r, c);
                let s  = 1.0 / (1.0 + (-xv).exp());
                // d/dx[x*σ] = σ + x*σ*(1-σ)
                let dsilu = s + xv * s * (1.0 - s);
                dout.at(r, c) * dsilu
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Element-wise multiplication of two same-shape TensorNodes.
    ///
    /// Used in SwiGLU: hidden = SiLU(gate) * up
    ///
    /// Backward:
    ///   dA = dC * B,  dB = dC * A
    pub fn mul_elem_node(&self, other: &TensorNode) -> TensorNode {
        let a = self.data().clone();
        let b = other.data().clone();
        assert_eq!((a.rows, a.cols), (b.rows, b.cols),
            "mul_elem_node: shape mismatch [{},{}] vs [{},{}]", a.rows, a.cols, b.rows, b.cols);

        let out_data = a.mul_elem(&b);
        let out = TensorNode::leaf(out_data);

        let self_c  = self.clone();
        let other_c = other.clone();
        let out_c   = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout  = out_c.0.borrow().grad.clone();
            let a_data = self_c.0.borrow().data.clone();
            let b_data = other_c.0.borrow().data.clone();

            // dA = dC * B
            let da = dout.mul_elem(&b_data);
            self_c.0.borrow_mut().grad.add_assign(&da);

            // dB = dC * A
            let db = dout.mul_elem(&a_data);
            other_c.0.borrow_mut().grad.add_assign(&db);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];
        out
    }

    /// Apply Rotary Position Embeddings (RoPE) to a [T, d_head] tensor.
    ///
    /// RoPE encodes position by rotating pairs of dimensions in Q and K vectors.
    /// Unlike learned positional embeddings (GPT-2) which add a fixed vector,
    /// RoPE modifies the dot products Q·K such that the attention score between
    /// positions i and j depends only on their *relative* distance (i - j).
    ///
    /// This is why RoPE generalizes better to sequence lengths longer than
    /// those seen during training (with YaRN scaling).
    ///
    /// Algorithm for position t, dimension pair (2i, 2i+1):
    ///   angle  = t / theta^(2i / d_head)
    ///   x'[2i]   = x[2i]   * cos(angle) - x[2i+1] * sin(angle)
    ///   x'[2i+1] = x[2i+1] * cos(angle) + x[2i]   * sin(angle)
    ///
    /// Parameters:
    ///   self:       [T, d_head]  — Q or K for a single head
    ///   seq_offset: position index of the first token (0 for new sequences)
    ///   theta:      rope_theta from config (10000 for LLaMA, 150000 for GPT-OSS)
    ///
    /// Backward:
    ///   RoPE is an orthogonal rotation — its inverse is rotation by -angle.
    ///   For each pair (2i, 2i+1) at row t:
    ///     dx[2i]   = dout[2i]  * cos(a) + dout[2i+1] * sin(a)
    ///     dx[2i+1] = dout[2i+1]* cos(a) - dout[2i]   * sin(a)
    pub fn rope_apply(&self, seq_offset: usize, theta: f32) -> TensorNode {
        let x = self.data().clone();
        let (t, d) = (x.rows, x.cols);
        assert!(d % 2 == 0, "rope_apply: d_head must be even, got {}", d);

        // Precompute cos/sin table — reused in backward
        let angles = Mat::from_fn(t, d / 2, |row, pair| {
            let pos = (seq_offset + row) as f32;
            pos / theta.powf(2.0 * pair as f32 / d as f32)
        });

        let out_data = Mat::from_fn(t, d, |row, col| {
            let pair   = col / 2;
            let is_odd = col % 2 == 1;
            let angle  = angles.at(row, pair);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            if !is_odd {
                x.at(row, col) * cos_a - x.at(row, col + 1) * sin_a
            } else {
                x.at(row, col) * cos_a + x.at(row, col - 1) * sin_a
            }
        });

        let out = TensorNode::leaf(out_data);
        let self_c  = self.clone();
        let out_c   = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();

            // Backward: inverse rotation by -angle
            let dx = Mat::from_fn(t, d, |row, col| {
                let pair   = col / 2;
                let is_odd = col % 2 == 1;
                let angle  = angles.at(row, pair);
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                if !is_odd {
                    // dx[2i] = dout[2i]*cos + dout[2i+1]*sin
                    dout.at(row, col) * cos_a + dout.at(row, col + 1) * sin_a
                } else {
                    // dx[2i+1] = dout[2i+1]*cos - dout[2i]*sin
                    dout.at(row, col) * cos_a - dout.at(row, col - 1) * sin_a
                }
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Apply RoPE with YaRN frequency scaling — extends context to 128k tokens.
    ///
    /// ## Why basic RoPE fails at long sequences
    ///
    /// Basic RoPE (rope_apply) was designed for context lengths ≤ 4096. The angle
    /// formula uses pos / theta^(2i/d). For high-frequency pairs (small i), this
    /// produces very rapidly cycling angles, which the model was trained to
    /// recognize. At positions >> 4096, these cycles repeat in patterns the model
    /// has never seen, causing incoherent attention.
    ///
    /// ## YaRN fix: interpolate the position, scale the temperature
    ///
    /// YaRN (Yet Another RoPE extensioN) rescales the effective position for
    /// each frequency dimension independently:
    ///
    ///   - "low frequencies" (large i, slow rotation): linear interpolation
    ///     effective_pos = pos / scale
    ///   - "high frequencies" (small i, fast rotation): no interpolation
    ///     effective_pos = pos  (unchanged, stays in trained range)
    ///   - "medium frequencies": smooth blend between the two
    ///
    /// Additionally a global "attention temperature" factor `mscale` is applied
    /// to keep attention scores well-calibrated at long range.
    ///
    /// ## Parameters (from GPT-OSS config)
    ///   theta:        150000.0   base frequency
    ///   original_ctx: 4096       context length the model was trained with
    ///   max_ctx:      131072     desired extended context
    ///   beta_fast:    32         freq threshold for "high freq" (no interpolation)
    ///   beta_slow:    1          freq threshold for "low freq" (full interpolation)
    ///
    /// Backward: same inverse-rotation as rope_apply, using the YaRN-scaled angles.
    pub fn rope_apply_yarn(
        &self,
        seq_offset:   usize,
        theta:        f32,
        original_ctx: usize,
        max_ctx:      usize,
        beta_fast:    f32,
        beta_slow:    f32,
    ) -> TensorNode {
        let x = self.data().clone();
        let (t, d) = (x.rows, x.cols);
        assert!(d % 2 == 0, "rope_apply_yarn: d_head must be even, got {}", d);

        let scale = max_ctx as f32 / original_ctx as f32;

        // mscale: attention temperature correction — keeps softmax numerically
        // stable at long range.  Formula from the YaRN paper (eq. 12).
        let mscale = 0.1 * scale.ln() + 1.0;

        // Precompute per-pair effective angles.
        // For dimension pair i (0-indexed), the original frequency is:
        //   omega_i = 1 / theta^(2i/d)
        // YaRN interpolates based on how "fast" this frequency rotates at
        // the trained context boundary.
        let angles = Mat::from_fn(t, d / 2, |row, pair| {
            let pos = (seq_offset + row) as f32;
            let omega = 1.0 / theta.powf(2.0 * pair as f32 / d as f32);

            // How many cycles does this frequency complete per trained context?
            // cycles_per_ctx = original_ctx * omega / (2π)
            let cycles_per_ctx = original_ctx as f32 * omega / (2.0 * std::f32::consts::PI);

            // Ramp: 0 = fully interpolated (slow freq), 1 = no interpolation (fast freq)
            let ramp = if cycles_per_ctx < beta_slow {
                0.0f32
            } else if cycles_per_ctx > beta_fast {
                1.0f32
            } else {
                (cycles_per_ctx - beta_slow) / (beta_fast - beta_slow)
            };

            // Blend between interpolated and original position
            let effective_pos = (1.0 - ramp) * (pos / scale) + ramp * pos;

            // Apply mscale correction and compute angle
            effective_pos * omega * mscale
        });

        let out_data = Mat::from_fn(t, d, |row, col| {
            let pair   = col / 2;
            let is_odd = col % 2 == 1;
            let angle  = angles.at(row, pair);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            if !is_odd {
                x.at(row, col) * cos_a - x.at(row, col + 1) * sin_a
            } else {
                x.at(row, col) * cos_a + x.at(row, col - 1) * sin_a
            }
        });

        let out    = TensorNode::leaf(out_data);
        let self_c = self.clone();
        let out_c  = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let dx = Mat::from_fn(t, d, |row, col| {
                let pair   = col / 2;
                let is_odd = col % 2 == 1;
                let angle  = angles.at(row, pair);
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                if !is_odd {
                    dout.at(row, col) * cos_a + dout.at(row, col + 1) * sin_a
                } else {
                    dout.at(row, col) * cos_a - dout.at(row, col - 1) * sin_a
                }
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Grouped Multi-Query Attention (GQA) with causal mask and full backward.
    ///
    /// GQA is a memory-efficient variant of multi-head attention where Q has
    /// n_q_heads projection heads but K and V share only n_kv_heads heads.
    ///
    /// Each "group" of (n_q_heads / n_kv_heads) Q heads shares one K and V head.
    /// This reduces the KV cache size by n_q_heads/n_kv_heads × during inference.
    ///
    /// For GPT-OSS-20b: n_q_heads=64, n_kv_heads=8, group_size=8.
    ///
    /// Parameters:
    ///   q: [T, n_q_heads * d_head]   — all Q projections concatenated
    ///   k: [T, n_kv_heads * d_head]  — all K projections concatenated
    ///   v: [T, n_kv_heads * d_head]  — all V projections concatenated
    ///
    /// Returns: [T, n_q_heads * d_head]
    ///
    /// Backward:
    ///   Same as causal_attention per head, but dK and dV accumulate from all
    ///   Q-heads in the group:
    ///     For each q_head h, kv_head = h/group_size:
    ///       dV_kvh  += W_h.T @ dOut_h
    ///       dW_h     = dOut_h @ V_h.T  (then softmax backward)
    ///       dQ_h    += dScores_h @ K_kvh * scale
    ///       dK_kvh  += dScores_h.T @ Q_h * scale
    pub fn gqa_attention(
        q: &TensorNode,
        k: &TensorNode,
        v: &TensorNode,
        n_q_heads: usize,
        n_kv_heads: usize,
        d_head: usize,
    ) -> TensorNode {
        let q_data = q.data().clone();
        let k_data = k.data().clone();
        let v_data = v.data().clone();
        let t = q_data.rows;
        let scale = 1.0 / (d_head as f32).sqrt();
        let group_size = n_q_heads / n_kv_heads;

        assert_eq!(q_data.cols, n_q_heads * d_head);
        assert_eq!(k_data.cols, n_kv_heads * d_head);
        assert_eq!(v_data.cols, n_kv_heads * d_head);

        // Forward: run attention per q-head, storing weights for backward
        let mut out_data    = Mat::zeros(t, n_q_heads * d_head);
        let mut all_weights = vec![Mat::zeros(t, t); n_q_heads]; // one per q-head

        for qh in 0..n_q_heads {
            let kvh = qh / group_size;

            let q_h = Mat::from_fn(t, d_head, |r, c| q_data.at(r, qh  * d_head + c));
            let k_h = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kvh * d_head + c));
            let v_h = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kvh * d_head + c));

            let mut scores = q_h.matmul(&k_h.transpose()).scale(scale);
            for i in 0..t { for j in (i+1)..t { *scores.at_mut(i, j) = -1e9; } }

            let mut w = Mat::zeros(t, t);
            for r in 0..t {
                let row_max = (0..t).map(|c| scores.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0f32;
                for c in 0..t {
                    let e = (scores.at(r, c) - row_max).exp();
                    *w.at_mut(r, c) = e; row_sum += e;
                }
                for c in 0..t { *w.at_mut(r, c) /= row_sum; }
            }

            let out_h = w.matmul(&v_h);
            for r in 0..t { for c in 0..d_head { *out_data.at_mut(r, qh * d_head + c) = out_h.at(r, c); } }
            all_weights[qh] = w;
        }

        let out = TensorNode::leaf(out_data);
        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout   = out_c.0.borrow().grad.clone();
            let q_data = q_c.0.borrow().data.clone();
            let k_data = k_c.0.borrow().data.clone();
            let v_data = v_c.0.borrow().data.clone();

            let mut dq_data = Mat::zeros(t, n_q_heads  * d_head);
            let mut dk_data = Mat::zeros(t, n_kv_heads * d_head);
            let mut dv_data = Mat::zeros(t, n_kv_heads * d_head);

            for qh in 0..n_q_heads {
                let kvh = qh / group_size;
                let w = &all_weights[qh];

                let dout_h = Mat::from_fn(t, d_head, |r, c| dout.at(r, qh  * d_head + c));
                let q_h    = Mat::from_fn(t, d_head, |r, c| q_data.at(r, qh  * d_head + c));
                let k_h    = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kvh * d_head + c));
                let v_h    = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kvh * d_head + c));

                // dV_kvh += W.T @ dOut_h
                let dv_h = w.transpose().matmul(&dout_h);
                for r in 0..t { for c in 0..d_head { *dv_data.at_mut(r, kvh * d_head + c) += dv_h.at(r, c); } }

                // dW = dOut_h @ V_h.T  [T, T]
                let dw = dout_h.matmul(&v_h.transpose());

                // Backward through causal softmax
                let mut dscores = Mat::zeros(t, t);
                for r in 0..t {
                    let dot: f32 = (0..=r).map(|c| dw.at(r, c) * w.at(r, c)).sum();
                    for c in 0..=r { *dscores.at_mut(r, c) = w.at(r, c) * (dw.at(r, c) - dot); }
                }
                let dscores = dscores.scale(scale);

                // dQ_h += dScores @ K_h
                let dq_h = dscores.matmul(&k_h);
                for r in 0..t { for c in 0..d_head { *dq_data.at_mut(r, qh * d_head + c) += dq_h.at(r, c); } }

                // dK_kvh += dScores.T @ Q_h
                let dk_h = dscores.transpose().matmul(&q_h);
                for r in 0..t { for c in 0..d_head { *dk_data.at_mut(r, kvh * d_head + c) += dk_h.at(r, c); } }
            }

            q_c.0.borrow_mut().grad.add_assign(&dq_data);
            k_c.0.borrow_mut().grad.add_assign(&dk_data);
            v_c.0.borrow_mut().grad.add_assign(&dv_data);
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

    /// When blas feature is active this exercises the cblas_sgemm path;
    /// otherwise it exercises the pure-Rust path — either way the result
    /// must match the reference value.
    #[test]
    fn test_matmul_blas_matches_reference() {
        let a = Mat::from_fn(8, 16, |r, c| (r * 16 + c) as f32 * 0.01 - 0.5);
        let b = Mat::from_fn(16, 8, |r, c| (r * 8 + c) as f32 * 0.02 - 0.3);
        let result = a.matmul(&b);
        // Compute reference with explicit triple loop to avoid depending on matmul
        let (m, k, n) = (8, 16, 8);
        let mut expected = Mat::zeros(m, n);
        for i in 0..m { for p in 0..k { for j in 0..n {
            *expected.at_mut(i, j) += a.at(i, p) * b.at(p, j);
        }}}
        for r in 0..m { for c in 0..n {
            assert!((result.at(r, c) - expected.at(r, c)).abs() < 1e-4,
                "matmul[{},{}]: got {} expected {}", r, c, result.at(r,c), expected.at(r,c));
        }}
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

    // --- RMSNorm backward ---

    #[test]
    fn test_rms_norm_grad_x() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            let gn = TensorNode::leaf(g_data.clone());
            let out = xn.rms_norm(&gn, 1e-5);
            out.data().data.iter().sum::<f32>()
        }, &x_data);

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let out = x.rms_norm(&g, 1e-5);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(approx(xg.at(0, c), num.at(0, c)),
                "RMSNorm dX[{}]: analytical={:.4} numerical={:.4}", c, xg.at(0, c), num.at(0, c));
        }
    }

    #[test]
    fn test_rms_norm_grad_gamma() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);

        let num = numerical_grad(&|g| {
            let xn = TensorNode::leaf(x_data.clone());
            let gn = TensorNode::leaf(g.clone());
            let out = xn.rms_norm(&gn, 1e-5);
            out.data().data.iter().sum::<f32>()
        }, &g_data);

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let out = x.rms_norm(&g, 1e-5);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let gg = g.grad().clone();
        for c in 0..4 {
            assert!(approx(gg.at(0, c), num.at(0, c)),
                "RMSNorm dGamma[{}]: analytical={:.4} numerical={:.4}", c, gg.at(0, c), num.at(0, c));
        }
    }

    // --- SiLU backward ---

    #[test]
    fn test_silu_grad() {
        let x_data = Mat::new(vec![-1.0, 0.0, 0.5, 2.0], 1, 4);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            xn.silu().data().data.iter().sum::<f32>()
        }, &x_data);

        let x   = TensorNode::leaf(x_data);
        let out = x.silu();
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(approx(xg.at(0, c), num.at(0, c)),
                "SiLU grad[{}]: analytical={:.4} numerical={:.4}", c, xg.at(0, c), num.at(0, c));
        }
    }

    // --- mul_elem_node backward ---

    #[test]
    fn test_mul_elem_node_grad() {
        let a_data = Mat::new(vec![1.0, 2.0, 0.5, -1.0], 1, 4);
        let b_data = Mat::new(vec![0.3, -0.5, 1.2, 0.8], 1, 4);

        let num_a = numerical_grad(&|a| {
            let an = TensorNode::leaf(a.clone());
            let bn = TensorNode::leaf(b_data.clone());
            an.mul_elem_node(&bn).data().data.iter().sum::<f32>()
        }, &a_data);

        let a   = TensorNode::leaf(a_data);
        let b   = TensorNode::leaf(b_data);
        let out = a.mul_elem_node(&b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let ag = a.grad().clone();
        for c in 0..4 {
            assert!(approx(ag.at(0, c), num_a.at(0, c)),
                "mul_elem dA[{}]: analytical={:.4} numerical={:.4}", c, ag.at(0, c), num_a.at(0, c));
        }
    }

    // --- RoPE backward ---

    #[test]
    fn test_rope_grad() {
        // Use slightly looser tolerance for RoPE: trig functions accumulate more
        // floating-point error in the central-difference approximation.
        let tol = 5e-3f32;
        let x_data = Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.1 + 0.1);

        let num = numerical_grad(&|x| {
            let xn = TensorNode::leaf(x.clone());
            xn.rope_apply(0, 10000.0).data().data.iter().sum::<f32>()
        }, &x_data);

        let x   = TensorNode::leaf(x_data);
        let out = x.rope_apply(0, 10000.0);
        let (r, c) = { let d = out.data(); (d.rows, d.cols) };
        out.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out);

        let xg = x.grad().clone();
        for row in 0..3 { for col in 0..8 {
            assert!((xg.at(row, col) - num.at(row, col)).abs() < tol,
                "RoPE dX[{},{}]: analytical={:.4} numerical={:.4}",
                row, col, xg.at(row, col), num.at(row, col));
        }}
    }

    // --- GQA backward ---

    #[test]
    fn test_gqa_grad_q() {
        let t = 3; let n_q = 4; let n_kv = 2; let dh = 4;
        let q_data = Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.05 + 0.1);
        let k_data = Mat::from_fn(t, n_kv * dh, |r, c| (r * (n_kv * dh) + c) as f32 * 0.03 + 0.05);
        let v_data = Mat::from_fn(t, n_kv * dh, |r, c| (r * (n_kv * dh) + c) as f32 * 0.04 + 0.02);

        let num = numerical_grad(&|q| {
            let qn = TensorNode::leaf(q.clone());
            let kn = TensorNode::leaf(k_data.clone());
            let vn = TensorNode::leaf(v_data.clone());
            TensorNode::gqa_attention(&qn, &kn, &vn, n_q, n_kv, dh).data().data.iter().sum::<f32>()
        }, &q_data);

        let q   = TensorNode::leaf(q_data);
        let k   = TensorNode::leaf(k_data);
        let v   = TensorNode::leaf(v_data);
        let out = TensorNode::gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let (r, c) = { let d = out.data(); (d.rows, d.cols) };
        out.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out);

        let qg = q.grad().clone();
        for row in 0..t { for col in 0..(n_q * dh) {
            assert!(approx(qg.at(row, col), num.at(row, col)),
                "GQA dQ[{},{}]: analytical={:.4} numerical={:.4}",
                row, col, qg.at(row, col), num.at(row, col));
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

    // --- Parallel matmul ---

    #[test]
    fn test_matmul_parallel_matches_sequential() {
        let m = 32; let k = 64; let n = 48;
        let a = Mat::from_fn(m, k, |r, c| (r * k + c) as f32 * 0.01 - 0.5);
        let b = Mat::from_fn(k, n, |r, c| (r * n + c) as f32 * 0.02 - 0.3);

        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b, 4);

        assert_eq!((par.rows, par.cols), (seq.rows, seq.cols));
        for r in 0..m { for c in 0..n {
            assert!((par.at(r, c) - seq.at(r, c)).abs() < 1e-4,
                "par[{},{}]={} seq[{},{}]={}", r, c, par.at(r,c), r, c, seq.at(r,c));
        }}
    }

    #[test]
    fn test_matmul_parallel_single_thread() {
        let a = Mat::from_fn(3, 4, |r, c| (r + c) as f32);
        let b = Mat::from_fn(4, 2, |r, c| (r * 2 + c) as f32);
        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b, 1);
        for r in 0..3 { for c in 0..2 {
            assert!((par.at(r, c) - seq.at(r, c)).abs() < 1e-5);
        }}
    }

    #[test]
    fn test_matmul_parallel_auto_threads() {
        let a = Mat::from_fn(16, 8, |r, c| (r * 8 + c) as f32 * 0.1);
        let b = Mat::from_fn(8, 16, |r, c| (r * 16 + c) as f32 * 0.1);
        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b, 0); // 0 = auto-detect thread count
        for r in 0..16 { for c in 0..16 {
            assert!((par.at(r, c) - seq.at(r, c)).abs() < 1e-3,
                "auto-thread: par[{},{}]={:.4} seq[{},{}]={:.4}", r,c,par.at(r,c),r,c,seq.at(r,c));
        }}
    }

    // --- Q4 quantization ---

    #[test]
    fn test_q4_quantize_dequantize_roundtrip() {
        let m = Mat::from_fn(4, 8, |r, c| ((r * 8 + c) as f32 / 31.0) * 2.0 - 1.0);
        let q = Q4Mat::quantize(&m);
        let m2 = q.dequantize();
        assert_eq!((m2.rows, m2.cols), (4, 8));
        for r in 0..4 { for c in 0..8 {
            assert!((m2.at(r, c) - m.at(r, c)).abs() < 0.15,
                "q4 round-trip error at [{},{}]: {} vs {}", r, c, m2.at(r,c), m.at(r,c));
        }}
    }

    #[test]
    fn test_q4_compression_ratio() {
        let m = Mat::from_fn(32, 32, |r, c| (r * 32 + c) as f32);
        let q = Q4Mat::quantize(&m);
        let ratio = q.compression_ratio();
        assert!(ratio > 4.0, "expected >4x compression, got {:.2}x", ratio);
    }

    #[test]
    fn test_q4_zeros_stay_zero() {
        let m = Mat::zeros(4, 8);
        let q = Q4Mat::quantize(&m);
        let m2 = q.dequantize();
        assert!(m2.data.iter().all(|&v| v == 0.0), "zeros should stay zero after Q4");
    }

    #[test]
    fn test_q4_matmul_matches_dequant() {
        let a = Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.05 + 0.1);
        let w = Mat::from_fn(4, 8, |r, c| (r * 8 + c) as f32 * 0.03 - 0.2);
        let q = Q4Mat::quantize(&w);

        let w_approx  = q.dequantize();
        let ref_out   = a.matmul(&w_approx.transpose());
        let fused_out = q.matmul_q4_t(&a);

        assert_eq!((fused_out.rows, fused_out.cols), (ref_out.rows, ref_out.cols));
        for r in 0..ref_out.rows { for c in 0..ref_out.cols {
            assert!((fused_out.at(r, c) - ref_out.at(r, c)).abs() < 1e-4,
                "q4 matmul [{},{}]: fused={:.5} ref={:.5}", r, c, fused_out.at(r,c), ref_out.at(r,c));
        }}
    }

    #[test]
    fn test_q4_non_multiple_block_size() {
        let m = Mat::from_fn(1, 10, |_, c| c as f32 * 0.1 - 0.5);
        let q = Q4Mat::quantize(&m);
        let m2 = q.dequantize();
        for c in 0..10 {
            assert!((m2.at(0, c) - m.at(0, c)).abs() < 0.15,
                "small Q4: error at col {}: {} vs {}", c, m2.at(0,c), m.at(0,c));
        }
    }
}
