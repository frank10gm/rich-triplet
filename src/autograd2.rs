use crate::ndarray::NDArray;
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
use std::collections::HashSet;
use std::rc::Rc;

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
        assert_eq!(
            data.len(),
            rows * cols,
            "Mat::new: data len {} != rows*cols {}*{}={}",
            data.len(),
            rows,
            cols,
            rows * cols
        );
        Mat { data, rows, cols }
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn ones(rows: usize, cols: usize) -> Self {
        Mat {
            data: vec![1.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn from_fn<F: Fn(usize, usize) -> f32>(rows: usize, cols: usize, f: F) -> Self {
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                data.push(f(r, c));
            }
        }
        Mat { data, rows, cols }
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        self.data[r * self.cols + c]
    }
    #[inline]
    pub fn at_mut(&mut self, r: usize, c: usize) -> &mut f32 {
        &mut self.data[r * self.cols + c]
    }

    pub fn numel(&self) -> usize {
        self.rows * self.cols
    }

    // -------------------------------------------------------------------------
    // Core linear algebra — these are the hot paths
    // -------------------------------------------------------------------------

    /// C = A @ B    [M,K] × [K,N] → [M,N]
    ///
    /// This single function replaces M*N*K scalar multiply-add operations
    /// in the old engine. On modern CPUs the compiler can auto-vectorize the
    /// inner loop with SIMD instructions.
    ///
    /// Dispatch priority (highest first):
    ///   1. `--features blas`     → `cblas_sgemm` (4–8× vs pure Rust)
    ///   2. `--features metal`    → Apple Metal GPU compute shader (Apple Silicon)
    ///   3. `--features parallel` → multi-threaded pure Rust (≈N_CPU × speedup)
    ///   4. default               → single-threaded pure Rust
    ///
    /// The `parallel` feature is useful when BLAS is unavailable (e.g. no
    /// Apple Accelerate / OpenBLAS installed).  When both `blas` and `parallel`
    /// are active, `blas` wins because BLAS is already multi-threaded internally.
    pub fn matmul(&self, b: &Mat) -> Mat {
        assert_eq!(
            self.cols, b.rows,
            "matmul shape mismatch: [{},{}] × [{},{}]",
            self.rows, self.cols, b.rows, b.cols
        );

        #[cfg(feature = "blas")]
        {
            return self.matmul_blas(b);
        }

        #[cfg(all(feature = "metal", not(feature = "blas")))]
        {
            return crate::metal_ops::metal_matmul(self, b);
        }

        #[cfg(all(feature = "parallel", not(feature = "blas"), not(feature = "metal")))]
        {
            return self.matmul_parallel(b, 0);
        }

        #[cfg(not(any(feature = "blas", feature = "metal", feature = "parallel")))]
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
                m as i32,
                n as i32,
                k as i32,
                1.0_f32, // alpha
                &self.data,
                k as i32, // A, lda
                &b.data,
                n as i32, // B, ldb
                0.0_f32,  // beta
                &mut out.data,
                n as i32, // C, ldc
            );
        }
        out
    }

    /// BLAS matmul where B is used transposed: computes `self @ b^T`.
    ///
    /// Equivalent to `self.matmul(&b.transpose())` but avoids the transpose
    /// allocation. Used by `fused_linear` with BF16 weights where the weight
    /// is stored as [out, in] and we need x @ W^T.
    #[cfg(feature = "blas")]
    pub fn matmul_bt(&self, b: &Mat) -> Mat {
        // self: [M, K],  b: [N, K]  →  out: [M, N]
        let (m, k, n) = (self.rows, self.cols, b.rows);
        assert_eq!(
            k, b.cols,
            "matmul_bt: [{},{}] × [{},{}]^T shape mismatch",
            self.rows, self.cols, b.rows, b.cols
        );
        let mut out = Mat::zeros(m, n);
        unsafe {
            cblas::sgemm(
                cblas::Layout::RowMajor,
                cblas::Transpose::None,
                cblas::Transpose::Ordinary,
                m as i32,
                n as i32,
                k as i32,
                1.0_f32,
                &self.data,
                k as i32,
                &b.data,
                k as i32, // ldb = K (B is [N,K] row-major)
                0.0_f32,
                &mut out.data,
                n as i32,
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
        Mat::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a + b)
                .collect(),
            self.rows,
            self.cols,
        )
    }

    /// In-place element-wise addition: self += other
    pub fn add_assign(&mut self, other: &Mat) {
        assert_eq!(self.data.len(), other.data.len());
        for (a, b) in self.data.iter_mut().zip(&other.data) {
            *a += b;
        }
    }

    /// Element-wise multiplication (same shape).
    pub fn mul_elem(&self, other: &Mat) -> Mat {
        assert_eq!((self.rows, self.cols), (other.rows, other.cols));
        Mat::new(
            self.data
                .iter()
                .zip(&other.data)
                .map(|(a, b)| a * b)
                .collect(),
            self.rows,
            self.cols,
        )
    }

    /// Scale every element by a scalar.
    pub fn scale(&self, s: f32) -> Mat {
        Mat::new(
            self.data.iter().map(|x| x * s).collect(),
            self.rows,
            self.cols,
        )
    }

    /// Element-wise map.
    pub fn map<F: Fn(f32) -> f32>(&self, f: F) -> Mat {
        Mat::new(
            self.data.iter().map(|&x| f(x)).collect(),
            self.rows,
            self.cols,
        )
    }

    /// Sum all elements.
    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

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
            self.data[r * self.cols..(r + 1) * self.cols]
                .iter()
                .sum::<f32>()
                * inv_n
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
        assert_eq!(
            self.cols, b.rows,
            "matmul_parallel shape mismatch: [{},{}] × [{},{}]",
            self.rows, self.cols, b.rows, b.cols
        );
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
                let row_end = (row_start + chunk).min(m);
                if row_start >= row_end {
                    break;
                }

                // SAFETY: each thread writes to a disjoint range of rows.
                // `a_ptr`, `b_ptr` are read-only; `out_ptr` range is unique per thread.
                let slice_len = (row_end - row_start) * n;
                let slice_start = row_start * n;
                let out_slice: &mut [f32] =
                    unsafe { std::slice::from_raw_parts_mut(out_ptr.add(slice_start), slice_len) };
                let a_slice: &[f32] = unsafe { std::slice::from_raw_parts(a_ptr, m * k) };
                let b_slice: &[f32] = unsafe { std::slice::from_raw_parts(b_ptr, k * n) };

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
// BF16 (bfloat16) storage — half the memory, nearly lossless
// =============================================================================
//
// ## What is BF16?
//
// BF16 (Brain Float 16) is a 16-bit floating-point format with:
//   - 1 sign bit
//   - 8 exponent bits  (same as f32 — same range: ~1e-38 to ~3e38)
//   - 7 mantissa bits  (vs 23 for f32 — lower precision)
//
// Because the exponent range equals f32, BF16 ↔ f32 conversion is trivial:
//   f32_bits   = bf16_bits << 16          (zero-pad the lower 16 bits)
//   bf16_bits  = (f32_bits >> 16) as u16  (truncate the lower 16 bits)
//
// ## Why use BF16?
//
// All major open-weight models (LLaMA, Gemma, Mistral, GPT-OSS) ship weights
// as BF16:
//   • Half the disk space and RAM of f32
//   • Full exponent range → no clipping issues during fine-tuning
//   • Supported natively by NVIDIA Ampere / Hopper, Apple M3+, Google TPUs
//
// In this CPU-only codebase we store weights as BF16 but dequantize to f32
// on the fly for computation (same strategy as Q4, but simpler).
//
// `MatBf16` provides a compact storage format.  `Mat::from_bf16()` converts
// a `MatBf16` to `Mat` for use in matmul.  `Mat::to_bf16()` converts back.
//
// ## Precision loss
//
// BF16 has only 7 mantissa bits (2.3 decimal digits of precision, vs 7.2 for f32).
// This is acceptable for weights but not for accumulators (loss, gradients) —
// those should stay as f32.

/// A matrix stored in BF16 (bfloat16) format for compact in-memory storage.
///
/// Use `Mat::from_bf16` to convert to f32 for computation.
/// Use `Mat::to_bf16` to convert an f32 `Mat` back to `MatBf16`.
///
/// `data` is reference-counted so that weight-tied tensors (embed_tokens and
/// lm_head) can share the same allocation without cloning 1.3 GB of BF16 bits.
#[derive(Clone)]
pub struct MatBf16 {
    pub data: std::sync::Arc<Vec<u16>>, // bf16 bits, one u16 per element
    pub rows: usize,
    pub cols: usize,
}

impl MatBf16 {
    /// Convert a BF16-packed u16 to f32.
    ///
    /// Shift the 16 bits into the upper half of a u32 (f32's bit layout):
    /// the sign and exponent fields are identical, mantissa is zero-padded.
    #[inline]
    pub fn bf16_to_f32(bits: u16) -> f32 {
        f32::from_bits((bits as u32) << 16)
    }

    /// Convert an f32 to BF16 bits (round-to-nearest, ties-to-even).
    ///
    /// For NaN/Inf, we preserve the special value.
    /// For finite values: shift right by 16, with rounding.
    #[inline]
    pub fn f32_to_bf16(v: f32) -> u16 {
        let bits = v.to_bits();
        if v.is_nan() {
            // Propagate NaN, ensure mantissa bit is set
            return ((bits >> 16) as u16) | 0x0040;
        }
        // Round-to-nearest-even: look at the truncated bits
        let rounding_bias = 0x7fff_u32 + ((bits >> 16) & 1);
        let rounded = bits.wrapping_add(rounding_bias);
        (rounded >> 16) as u16
    }

    /// Convert this `MatBf16` to a full f32 `Mat`.
    pub fn to_f32(&self) -> Mat {
        Mat {
            data: self.data.iter().map(|&b| Self::bf16_to_f32(b)).collect(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Size in bytes (2 bytes per element).
    pub fn size_bytes(&self) -> usize {
        self.data.len() * 2
    }

    /// Compression ratio vs f32 (always 2.0×).
    pub fn compression_ratio(&self) -> f32 {
        2.0
    }

    #[inline]
    pub fn at(&self, r: usize, c: usize) -> f32 {
        Self::bf16_to_f32(self.data[r * self.cols + c])
    }

    /// Compute `a [M, K] @ self^T` where `self` is [N, K] BF16.
    ///
    /// **Decode path (M ≤ 4)** — dequantises a chunk of BF16 weight rows at a
    /// time into a scratch buffer, then calls a single `sgemm` per chunk.
    /// This reduces BLAS call overhead from N calls (e.g. 262 K for lm_head)
    /// to N/CHUNK calls (e.g. 256), which eliminates the dominant per-call
    /// overhead (~1 µs × 262 K = ~262 ms per token).  Chunk size targets
    /// ~8 MB of scratch (fits comfortably in Apple-Silicon L2 cache).
    ///
    /// **Prefill path (M > 4)** — dequantises the full N×K matrix once and
    /// delegates to an accelerated sgemm, which amortises the cost across M
    /// query rows.
    pub fn matmul_by_t(&self, a: &Mat) -> Mat {
        let m = a.rows;
        let k = a.cols;
        let n = self.rows;
        assert_eq!(
            k, self.cols,
            "matmul_by_t BF16: input cols {} != weight cols {}",
            k, self.cols
        );

        // GEMV fast path for decode (M=1): fused BF16 dot + multi-threading.
        // Bypasses dequant-to-scratch + sgemm entirely — reads BF16 directly.
        //
        // Note: Metal GPU GEMV was tested but is slower than CPU NEON on Apple
        // Silicon due to shared memory bandwidth and per-dispatch overhead (~0.5ms).
        // See metal_ops::metal_gemv_bf16_t for the GPU kernel (kept for potential
        // future graph-based execution).
        if m == 1 {
            return self.gemv_mt(a);
        }

        // Prefill: full dequant then an accelerated GEMM.
        if m > 4 {
            let w = self.to_f32(); // [N, K] f32

            #[cfg(feature = "blas")]
            {
                return a.matmul_bt(&w);
            }
            #[cfg(not(feature = "blas"))]
            {
                return a.matmul(&w.transpose());
            }
        }

        // ── Decode path (M ≤ 4): chunked SGEMM ───────────────────────────
        // Dequantise CHUNK weight rows → f32, call sgemm once per chunk.
        // The output-slice trick: pass &mut out.data[j0..] with ldc = n so
        // sgemm writes C[i, j] directly to out[i, j0 + j] via row stride n.
        #[cfg(feature = "blas")]
        {
            // Target ~8 MB of scratch: chunk = 8 MB / (K × 4 bytes), clamped.
            let chunk: usize = ((8 * 1024 * 1024) / (k * 4)).max(64).min(n);
            let mut out = Mat::zeros(m, n);
            let mut chunk_buf = vec![0.0f32; chunk * k];

            let mut j0 = 0usize;
            while j0 < n {
                let j1 = (j0 + chunk).min(n);
                let actual = j1 - j0;

                // Dequantise BF16 rows j0..j1 into chunk_buf[0..actual*k].
                for ji in 0..actual {
                    let src = (j0 + ji) * k;
                    let dst = ji * k;
                    for c in 0..k {
                        chunk_buf[dst + c] = Self::bf16_to_f32(self.data[src + c]);
                    }
                }

                // SGEMM: out[m, actual] = a[m, k] × chunk_buf[actual, k]^T
                unsafe {
                    cblas::sgemm(
                        cblas::Layout::RowMajor,
                        cblas::Transpose::None,
                        cblas::Transpose::Ordinary,
                        m as i32,
                        actual as i32,
                        k as i32,
                        1.0f32,
                        &a.data,
                        k as i32,
                        &chunk_buf[..actual * k],
                        k as i32,
                        0.0f32,
                        &mut out.data[j0..],
                        n as i32,
                    );
                }
                j0 = j1;
            }
            return out;
        }

        // Non-BLAS fallback: scalar row-by-row.
        #[cfg(not(feature = "blas"))]
        {
            let mut out = Mat::zeros(m, n);
            let mut row_f32 = vec![0.0f32; k];
            for j in 0..n {
                let base = j * k;
                for c in 0..k {
                    row_f32[c] = Self::bf16_to_f32(self.data[base + c]);
                }
                for i in 0..m {
                    let mut dot = 0.0f32;
                    for c in 0..k {
                        dot += a.data[i * k + c] * row_f32[c];
                    }
                    *out.at_mut(i, j) = dot;
                }
            }
            return out;
        }

        // Unreachable — one of the cfg branches above always returns.
        #[allow(unreachable_code)]
        Mat::zeros(m, n)
    }

    // =========================================================================
    // Fused GEMV — M=1 decode fast path
    // =========================================================================

    /// Fused BF16 dot product for a single weight row against activation vector.
    /// On aarch64, uses NEON intrinsics for ~4× throughput over scalar code.
    #[inline]
    fn dot_row(&self, row: usize, a: &[f32]) -> f32 {
        let k = self.cols;
        let base = row * k;

        #[cfg(target_arch = "aarch64")]
        {
            // SAFETY: NEON is always available on aarch64.
            // `base + k <= self.data.len()` by construction (row < self.rows).
            // `a.len() >= k` is guaranteed by the caller.
            unsafe {
                Self::dot_bf16_neon(
                    self.data.as_ptr().add(base),
                    a.as_ptr(),
                    k,
                )
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let bf16 = &*self.data;
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += Self::bf16_to_f32(bf16[base + p]) * a[p];
            }
            acc
        }
    }

    /// NEON-accelerated BF16 dot product.
    ///
    /// Processes 16 elements per iteration: load u16, zero-extend to u32,
    /// shift left 16 → f32 bit pattern, FMA with activation.
    /// 4 independent accumulators hide FMA latency.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_bf16_neon(bf16_ptr: *const u16, a_ptr: *const f32, k: usize) -> f32 {
        use std::arch::aarch64::*;

        unsafe {
            let mut acc0 = vdupq_n_f32(0.0);
            let mut acc1 = vdupq_n_f32(0.0);
            let mut acc2 = vdupq_n_f32(0.0);
            let mut acc3 = vdupq_n_f32(0.0);

            let mut p = 0usize;

            // Main loop: 16 elements per iteration.
            while p + 16 <= k {
                let b0 = vld1_u16(bf16_ptr.add(p));
                let b1 = vld1_u16(bf16_ptr.add(p + 4));
                let b2 = vld1_u16(bf16_ptr.add(p + 8));
                let b3 = vld1_u16(bf16_ptr.add(p + 12));

                let f0 = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(b0)));
                let f1 = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(b1)));
                let f2 = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(b2)));
                let f3 = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(b3)));

                acc0 = vfmaq_f32(acc0, f0, vld1q_f32(a_ptr.add(p)));
                acc1 = vfmaq_f32(acc1, f1, vld1q_f32(a_ptr.add(p + 4)));
                acc2 = vfmaq_f32(acc2, f2, vld1q_f32(a_ptr.add(p + 8)));
                acc3 = vfmaq_f32(acc3, f3, vld1q_f32(a_ptr.add(p + 12)));

                p += 16;
            }

            // Tail: 4 elements at a time.
            while p + 4 <= k {
                let b = vld1_u16(bf16_ptr.add(p));
                let f = vreinterpretq_f32_u32(vshlq_n_u32::<16>(vmovl_u16(b)));
                acc0 = vfmaq_f32(acc0, f, vld1q_f32(a_ptr.add(p)));
                p += 4;
            }

            // Reduce 4 accumulators → scalar.
            acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
            let mut result = vaddvq_f32(acc0);

            // Scalar remainder.
            while p < k {
                result += f32::from_bits((*bf16_ptr.add(p) as u32) << 16) * *a_ptr.add(p);
                p += 1;
            }

            result
        }
    }

    /// Multi-threaded GEMV: `a[1,K] @ self^T[N,K] → [1,N]`.
    ///
    /// Each thread computes dot products for a contiguous range of output
    /// neurons, reading BF16 weights directly — no scratch buffer needed.
    fn gemv_mt(&self, a: &Mat) -> Mat {
        let k = self.cols;
        let n = self.rows;
        debug_assert_eq!(a.rows, 1);
        debug_assert_eq!(a.cols, k);

        let mut out = Mat::zeros(1, n);

        let hw_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let n_threads = hw_threads.min(n / 128).max(1);

        if n_threads > 1 {
            let out_ptr = out.data.as_mut_ptr();
            let a_data = &a.data[..k];

            std::thread::scope(|s| {
                let chunk = n.div_ceil(n_threads);
                for tid in 0..n_threads {
                    let j0 = tid * chunk;
                    let j1 = ((tid + 1) * chunk).min(n);
                    if j0 >= n {
                        break;
                    }

                    // SAFETY: each thread writes to a disjoint range [j0..j1).
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(out_ptr.add(j0), j1 - j0)
                    };

                    s.spawn(move || {
                        for j in 0..(j1 - j0) {
                            out_slice[j] = self.dot_row(j0 + j, a_data);
                        }
                    });
                }
            });
        } else {
            let a_data = &a.data[..k];
            for j in 0..n {
                out.data[j] = self.dot_row(j, a_data);
            }
        }

        out
    }
}

// =============================================================================
// Q4KMat — Q4_K (GGUF Q4_K_M) native packed storage
// =============================================================================
//
// Stores weights in the raw Q4_K block layout (144 bytes per 256 elements):
//   bytes[0..2]   — f16 `d`    (super-block scale for the 8 sub-block scales)
//   bytes[2..4]   — f16 `dmin` (super-block scale for the 8 sub-block mins)
//   bytes[4..16]  — 12 packed 6-bit scale/min pairs
//   bytes[16..144]— 128 bytes of 4-bit quants (4 chunks × 32 bytes × 8 nibbles)
//
// Dequantization on-the-fly avoids storing BF16/f32 copies at rest, saving
// ~3.5× RAM compared to BF16 (144/256 ≈ 0.56 bytes/elem vs 2 bytes/elem).
//
// The matmul path mirrors Q4Mat: chunked SGEMM for decode (M ≤ 4),
// full dequant + single SGEMM for prefill (M > 4).

/// Q4_K native packed weight matrix.
///
/// `blocks` holds the raw GGUF Q4_K bytes in row-major block order.
/// Each super-block of 256 elements takes exactly 144 bytes.
/// Dimensions follow our convention: `rows` = out_features, `cols` = in_features.
/// Pre-quantized Q8 activation data for SDOT-accelerated Q4K dot products.
/// Activations are quantized to int8 per 32-element sub-block.
struct Q8Activation {
    /// Quantized activation values (length = K, padded to multiple of 32).
    q8: Vec<i8>,
    /// Per-32-element scale: actual_value ≈ q8[i] * scales[i/32].
    scales: Vec<f32>,
    /// Per-32-element float sum of original activations (for min subtraction).
    sums: Vec<f32>,
}

#[derive(Clone)]
pub struct Q4KMat {
    pub rows: usize,
    pub cols: usize,
    /// Raw Q4_K block bytes: `n_blocks * 144` bytes where
    /// `n_blocks = ceil(rows * cols / 256)`.
    pub blocks: Vec<u8>,
}

impl Q4KMat {
    /// Convert IEEE 754 half-precision (f16) bits to f32.
    /// Q4_K super-block headers store d and dmin as f16, not bfloat16.
    #[inline(always)]
    fn f16_to_f32(bits: u16) -> f32 {
        let sign     = ((bits >> 15) & 1) as u32;
        let exponent = ((bits >> 10) & 0x1F) as u32;
        let mantissa = (bits & 0x3FF) as u32;
        let f32_bits = if exponent == 0 {
            // Denormal: ±0.0 if mantissa==0, else denormal
            if mantissa == 0 { sign << 31 }
            else {
                // Normalise the denormal (matches gguf_loader::f16_to_f32)
                let mut m = mantissa;
                let mut e = 0u32;
                while (m & 0x400) == 0 { m <<= 1; e += 1; }
                m &= 0x3FF;
                (sign << 31) | ((127 - 14 - e) << 23) | (m << 13)
            }
        } else if exponent == 0x1F {
            // Inf / NaN
            (sign << 31) | (0xFF << 23) | (mantissa << 13)
        } else {
            (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13)
        };
        f32::from_bits(f32_bits)
    }

    /// Extract the 6-bit scale and 6-bit min for sub-block pair `j` (0..8)
    /// from the 12-byte scales array of a Q4_K super-block.
    #[inline(always)]
    fn scale_min(sc: &[u8], j: usize) -> (f32, f32) {
        let (sv, mv) = if j < 4 {
            (sc[j] & 0x3F, sc[j + 4] & 0x3F)
        } else {
            (
                (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4),
                (sc[j + 4] >> 4)   | ((sc[j + 0] >> 6) << 4),
            )
        };
        (sv as f32, mv as f32)
    }

    /// Convert f32 to IEEE 754 half-precision (f16) bits.
    /// Inverse of `f16_to_f32`. Rounds to nearest, ties to even.
    #[inline(always)]
    fn f32_to_f16(val: f32) -> u16 {
        let bits = val.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;

        if exp == 0xFF {
            // Inf / NaN
            return (sign | 0x7C00 | (mant >> 13) as u32) as u16;
        }
        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1F {
            // Overflow → Inf
            return (sign | 0x7C00) as u16;
        }
        if new_exp <= 0 {
            // Denormal or zero
            if new_exp < -10 {
                return sign as u16; // too small
            }
            let m = mant | 0x800000;
            let shift = (1 - new_exp) as u32 + 13;
            return (sign | (m >> shift) as u32) as u16;
        }
        (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
    }

    /// Pack 6-bit scale and min values into the 12-byte scales array.
    /// `sv[0..8]` are 6-bit scale values, `mv[0..8]` are 6-bit min values.
    fn pack_scales(sv: &[u8; 8], mv: &[u8; 8]) -> [u8; 12] {
        let mut sc = [0u8; 12];
        // Indices 0..3: low 6 bits of sv[0..3] and mv[0..3], high 2 bits
        // stored in sv[4..7] and mv[4..7] high positions.
        for j in 0..4 {
            sc[j]     = (sv[j] & 0x3F) | ((sv[j + 4] & 0x30) << 2);  // bits 4-5 of sv[j+4] → bits 6-7
            sc[j + 4] = (mv[j] & 0x3F) | ((mv[j + 4] & 0x30) << 2);  // bits 4-5 of mv[j+4] → bits 6-7
        }
        // Indices 8..11: low 4 bits of sv[4..7] in low nibble, low 4 bits of mv[4..7] in high nibble
        for j in 0..4 {
            sc[8 + j] = (sv[j + 4] & 0x0F) | ((mv[j + 4] & 0x0F) << 4);
        }
        sc
    }

    /// Quantize an f32 matrix to Q4_K format.
    ///
    /// `mat` must have cols divisible by 256. Each super-block of 256 elements
    /// is quantized into 144 bytes following the GGML Q4_K layout.
    pub fn quantize(mat: &Mat) -> Self {
        let (rows, cols) = (mat.rows, mat.cols);
        assert_eq!(cols % 256, 0, "Q4KMat::quantize: cols must be multiple of 256, got {}", cols);
        let n_blocks_per_row = cols / 256;
        let total_blocks = rows * n_blocks_per_row;
        let mut blocks = vec![0u8; total_blocks * 144];

        let n_threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4);
        let rows_per_thread = rows.div_ceil(n_threads);

        std::thread::scope(|s| {
            for (chunk_idx, block_chunk) in blocks.chunks_mut(rows_per_thread * n_blocks_per_row * 144).enumerate() {
                let row_start = chunk_idx * rows_per_thread;
                let row_end = (row_start + rows_per_thread).min(rows);
                let mat_data = &mat.data;
                s.spawn(move || {
                    for row in row_start..row_end {
                        let row_data = &mat_data[row * cols..(row + 1) * cols];
                        for b in 0..n_blocks_per_row {
                            let block_data = &row_data[b * 256..(b + 1) * 256];
                            let boff = ((row - row_start) * n_blocks_per_row + b) * 144;

                            Self::quantize_block(block_data, &mut block_chunk[boff..boff + 144]);
                        }
                    }
                });
            }
        });

        Q4KMat { rows, cols, blocks }
    }

    /// Quantize a single 256-element block into 144 bytes of Q4K format.
    fn quantize_block(block_data: &[f32], out: &mut [u8]) {
        // Step 1: compute per-sub-block scale and min (8 sub-blocks of 32 elements)
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];
        for sb in 0..8 {
            let sub = &block_data[sb * 32..(sb + 1) * 32];
            let mut smin = f32::INFINITY;
            let mut smax = f32::NEG_INFINITY;
            for &v in sub {
                if v < smin { smin = v; }
                if v > smax { smax = v; }
            }
            if smin >= 0.0 {
                sub_mins[sb] = 0.0;
                sub_scales[sb] = if smax > 0.0 { smax / 15.0 } else { 0.0 };
            } else {
                sub_mins[sb] = smin;
                sub_scales[sb] = if smax > smin { (smax - smin) / 15.0 } else { 0.0 };
            }
        }

        // Step 2: find super-block d and dmin
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().map(|m| -m).fold(0.0f32, f32::max);
        let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
        let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

        // Step 3: quantize sub-block scales and mins to 6-bit
        let mut sv = [0u8; 8];
        let mut mv = [0u8; 8];
        if d > 0.0 {
            let inv_d = 1.0 / d;
            for j in 0..8 {
                sv[j] = ((sub_scales[j] * inv_d + 0.5) as u8).min(63);
            }
        }
        if dmin > 0.0 {
            let inv_dmin = 1.0 / dmin;
            for j in 0..8 {
                mv[j] = (((-sub_mins[j]) * inv_dmin + 0.5) as u8).min(63);
            }
        }

        // Step 4: store d and dmin as f16
        let d_f16 = Self::f32_to_f16(d);
        let dmin_f16 = Self::f32_to_f16(dmin);
        out[0..2].copy_from_slice(&d_f16.to_le_bytes());
        out[2..4].copy_from_slice(&dmin_f16.to_le_bytes());

        // Step 5: pack 6-bit scales/mins into 12-byte array
        let sc = Self::pack_scales(&sv, &mv);
        out[4..16].copy_from_slice(&sc);

        // Step 6: quantize values into 4-bit nibbles
        let d_val = Self::f16_to_f32(d_f16);
        let dmin_val = Self::f16_to_f32(dmin_f16);
        for chunk in 0..4 {
            let scale1 = d_val * sv[chunk * 2] as f32;
            let min1   = dmin_val * mv[chunk * 2] as f32;
            let scale2 = d_val * sv[chunk * 2 + 1] as f32;
            let min2   = dmin_val * mv[chunk * 2 + 1] as f32;

            let lo_data = &block_data[chunk * 64..chunk * 64 + 32];
            let hi_data = &block_data[chunk * 64 + 32..chunk * 64 + 64];

            for l in 0..32 {
                let q_lo = if scale1 > 0.0 {
                    (((lo_data[l] + min1) / scale1 + 0.5) as u8).min(15)
                } else { 0 };
                let q_hi = if scale2 > 0.0 {
                    (((hi_data[l] + min2) / scale2 + 0.5) as u8).min(15)
                } else { 0 };
                out[16 + chunk * 32 + l] = q_lo | (q_hi << 4);
            }
        }
    }

    /// Quantize a BF16 matrix to Q4K without materializing the full f32 matrix.
    ///
    /// Uses a per-row f32 scratch buffer (~10 KB for cols=2560) instead of
    /// allocating rows×cols×4 bytes (~2.7 GB for lm_head).
    pub fn quantize_from_bf16(bf16: &crate::autograd2::MatBf16) -> Self {
        let (rows, cols) = (bf16.rows, bf16.cols);
        assert_eq!(cols % 256, 0, "Q4KMat::quantize_from_bf16: cols must be multiple of 256, got {}", cols);
        let n_blocks_per_row = cols / 256;
        let total_blocks = rows * n_blocks_per_row;
        let mut blocks = vec![0u8; total_blocks * 144];

        let n_threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4);
        let rows_per_thread = rows.div_ceil(n_threads);

        std::thread::scope(|s| {
            for (chunk_idx, block_chunk) in blocks.chunks_mut(rows_per_thread * n_blocks_per_row * 144).enumerate() {
                let row_start = chunk_idx * rows_per_thread;
                let row_end = (row_start + rows_per_thread).min(rows);
                let bf16_data = &bf16.data;
                s.spawn(move || {
                    let mut row_buf = vec![0.0f32; cols];
                    for row in row_start..row_end {
                        // Convert one row BF16→f32
                        let src = &bf16_data[row * cols..(row + 1) * cols];
                        for (o, &b) in row_buf.iter_mut().zip(src.iter()) {
                            *o = MatBf16::bf16_to_f32(b);
                        }
                        // Quantize row into Q4K blocks
                        for b in 0..n_blocks_per_row {
                            let block_data = &row_buf[b * 256..(b + 1) * 256];
                            let boff = ((row - row_start) * n_blocks_per_row + b) * 144;
                            Self::quantize_block(block_data, &mut block_chunk[boff..boff + 144]);
                        }
                    }
                });
            }
        });

        Q4KMat { rows, cols, blocks }
    }

    /// Convert a Q4_0 matrix (Q4Mat) to Q4K format.
    /// Row-by-row dequant → requant with ~10 KB scratch per thread.
    pub fn from_q4mat(q4: &Q4Mat) -> Self {
        let (rows, cols) = (q4.rows, q4.cols);
        assert_eq!(cols % 256, 0, "Q4KMat::from_q4mat: cols must be multiple of 256, got {}", cols);
        let n_blocks_per_row = cols / 256;
        let total_blocks = rows * n_blocks_per_row;
        let mut blocks = vec![0u8; total_blocks * 144];

        let n_threads = std::thread::available_parallelism().map(|t| t.get()).unwrap_or(4);
        let rows_per_thread = rows.div_ceil(n_threads);

        std::thread::scope(|s| {
            for (chunk_idx, block_chunk) in blocks.chunks_mut(rows_per_thread * n_blocks_per_row * 144).enumerate() {
                let row_start = chunk_idx * rows_per_thread;
                let row_end = (row_start + rows_per_thread).min(rows);
                s.spawn(move || {
                    let mut row_buf = vec![0.0f32; cols];
                    for row in row_start..row_end {
                        q4.dequantize_row_into(row, &mut row_buf);
                        for b in 0..n_blocks_per_row {
                            let block_data = &row_buf[b * 256..(b + 1) * 256];
                            let boff = ((row - row_start) * n_blocks_per_row + b) * 144;
                            Self::quantize_block(block_data, &mut block_chunk[boff..boff + 144]);
                        }
                    }
                });
            }
        });

        Q4KMat { rows, cols, blocks }
    }

    /// Dequantize row `row_idx` (in [0, rows)) into `buf` (length >= cols).
    ///
    /// `cols` must be a multiple of 256 (true for all Gemma3 weight matrices),
    /// so each row starts exactly on a super-block boundary.
    pub fn dequantize_row_into(&self, row_idx: usize, buf: &mut [f32]) {
        let k = self.cols;
        debug_assert_eq!(k % 256, 0, "cols must be a multiple of 256");
        let n_blocks = k / 256;
        let base_block = row_idx * n_blocks;

        for b in 0..n_blocks {
            let boff = (base_block + b) * 144;
            let d_bits    = u16::from_le_bytes([self.blocks[boff],     self.blocks[boff + 1]]);
            let dmin_bits = u16::from_le_bytes([self.blocks[boff + 2], self.blocks[boff + 3]]);
            let d    = Self::f16_to_f32(d_bits);
            let dmin = Self::f16_to_f32(dmin_bits);
            let sc   = &self.blocks[boff + 4..boff + 16];
            let qs   = &self.blocks[boff + 16..boff + 144];

            // 4 chunks of 64 elements; chunk c uses qs[c*32..(c+1)*32]:
            //   low  nibbles → sub-block 2c   (scale_min index 2c)
            //   high nibbles → sub-block 2c+1 (scale_min index 2c+1)
            let out = &mut buf[b * 256..(b + 1) * 256];
            for chunk in 0..4usize {
                let (sv1, mv1) = Self::scale_min(sc, chunk * 2);
                let (sv2, mv2) = Self::scale_min(sc, chunk * 2 + 1);
                let scale1 = d * sv1; let min1 = dmin * mv1;
                let scale2 = d * sv2; let min2 = dmin * mv2;
                let q = &qs[chunk * 32..(chunk + 1) * 32];
                let (lo, hi) = out[chunk * 64..chunk * 64 + 64].split_at_mut(32);

                #[cfg(target_arch = "aarch64")]
                unsafe {
                    Self::dequant_chunk_neon(q.as_ptr(), lo.as_mut_ptr(), hi.as_mut_ptr(),
                                            scale1, min1, scale2, min2);
                }

                #[cfg(not(target_arch = "aarch64"))]
                for l in 0..32 {
                    lo[l] = scale1 * (q[l] & 0x0F) as f32 - min1;
                    hi[l] = scale2 * (q[l] >> 4)   as f32 - min2;
                }
            }
        }
    }

    /// NEON-accelerated Q4_K dequant for one 32-byte chunk (64 f32 outputs).
    ///
    /// Processes 16 packed bytes at a time:
    ///   - extract low nibbles (AND 0x0F) → 16 u8
    ///   - extract high nibbles (SHR 4)   → 16 u8
    ///   - widen u8 → u16 → u32 → f32 in groups of 4
    ///   - scale and subtract min: w = scale * nibble_f32 - min
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dequant_chunk_neon(
        q_ptr: *const u8,
        lo_ptr: *mut f32,
        hi_ptr: *mut f32,
        scale1: f32,
        min1: f32,
        scale2: f32,
        min2: f32,
    ) {
        use std::arch::aarch64::*;

        unsafe {
            let mask_0f = vdupq_n_u8(0x0F);
            let s1 = vdupq_n_f32(scale1);
            let m1 = vdupq_n_f32(min1);
            let s2 = vdupq_n_f32(scale2);
            let m2 = vdupq_n_f32(min2);

            // Two passes of 16 bytes each cover all 32 bytes.
            for half in 0..2u32 {
                let off = (half * 16) as usize;
                let raw = vld1q_u8(q_ptr.add(off));
                let lo_nib = vandq_u8(raw, mask_0f);
                let hi_nib = vshrq_n_u8::<4>(raw);

                // ── Low nibbles → lo_ptr[off..off+16] ────────────────────
                let lo8a = vmovl_u8(vget_low_u8(lo_nib)); // 8 × u16
                let g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8a)));
                let g1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8a)));
                let lo8b = vmovl_u8(vget_high_u8(lo_nib));
                let g2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8b)));
                let g3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8b)));

                vst1q_f32(lo_ptr.add(off),      vsubq_f32(vmulq_f32(s1, g0), m1));
                vst1q_f32(lo_ptr.add(off + 4),   vsubq_f32(vmulq_f32(s1, g1), m1));
                vst1q_f32(lo_ptr.add(off + 8),   vsubq_f32(vmulq_f32(s1, g2), m1));
                vst1q_f32(lo_ptr.add(off + 12),  vsubq_f32(vmulq_f32(s1, g3), m1));

                // ── High nibbles → hi_ptr[off..off+16] ───────────────────
                let hi8a = vmovl_u8(vget_low_u8(hi_nib));
                let h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8a)));
                let h1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8a)));
                let hi8b = vmovl_u8(vget_high_u8(hi_nib));
                let h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8b)));
                let h3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8b)));

                vst1q_f32(hi_ptr.add(off),      vsubq_f32(vmulq_f32(s2, h0), m2));
                vst1q_f32(hi_ptr.add(off + 4),   vsubq_f32(vmulq_f32(s2, h1), m2));
                vst1q_f32(hi_ptr.add(off + 8),   vsubq_f32(vmulq_f32(s2, h2), m2));
                vst1q_f32(hi_ptr.add(off + 12),  vsubq_f32(vmulq_f32(s2, h3), m2));
            }
        }
    }

    /// Memory usage in bytes.
    pub fn size_bytes(&self) -> usize {
        self.blocks.len()
    }

    /// Scalar matmul: `a [M, K] @ self^T [N, K] → [M, N]`.
    pub fn matmul_q4k_t(&self, a: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, self.rows);
        assert_eq!(k, self.cols, "matmul_q4k_t: a.cols {} != q4k.cols {}", k, self.cols);
        // GEMV fast path for decode (M=1): fused NEON dot + multi-threading.
        if m == 1 {
            return self.gemv_mt(a);
        }
        let mut out = Mat::zeros(m, n);
        let mut row_buf = vec![0.0f32; k];
        for j in 0..n {
            self.dequantize_row_into(j, &mut row_buf);
            for i in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    acc += a.data[i * k + p] * row_buf[p];
                }
                *out.at_mut(i, j) = acc;
            }
        }
        out
    }

    /// BLAS-accelerated matmul: `a [M, K] @ self^T [N, K] → [M, N]`.
    ///
    /// Chunked SGEMM with ~8 MB scratch (prefill and decode).
    #[cfg(feature = "blas")]
    pub fn matmul_q4k_t_blas(&self, a: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, self.rows);
        assert_eq!(k, self.cols, "matmul_q4k_t_blas: a.cols {} != q4k.cols {}", k, self.cols);

        // GEMV fast path for decode (M=1): fused NEON dot + multi-threading.
        if m == 1 {
            return self.gemv_mt(a);
        }

        // Chunked SGEMM for both prefill and decode.
        //
        // Split output neurons across threads, each with its own scratch buffer.
        // Each thread dequants + sgemms its range independently.
        // Apple Accelerate sgemm is thread-safe; output ranges are disjoint.
        let n_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let chunk: usize = ((8 * 1024 * 1024) / (k * 4)).max(64).min(n);
        let mut out = Mat::zeros(m, n);

        // Only parallelise when the matmul is large enough that thread-spawn
        // overhead (~200 µs for 8 threads) is small relative to the dequant work.
        // gate/up/down projections (N=10240) qualify; k/v (N=1024) do not.
        if n_threads > 1 && n >= 4096 {
            let out_ptr = out.data.as_mut_ptr();
            let a_ptr = a.data.as_ptr();
            let a_len = a.data.len();

            std::thread::scope(|s| {
                let rows_per_thread = n.div_ceil(n_threads);
                for tid in 0..n_threads {
                    let j_start = tid * rows_per_thread;
                    let j_end = ((tid + 1) * rows_per_thread).min(n);
                    if j_start >= n {
                        break;
                    }

                    // SAFETY: each thread writes to disjoint column ranges of `out`.
                    // `a_slice` is read-only and shared across all threads.
                    // `out_slice` covers [j_start .. m*n) — overlapping in raw range
                    // but each thread's sgemm only writes to its own column band.
                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(
                            out_ptr.add(j_start),
                            m * n - j_start,
                        )
                    };
                    let a_slice = unsafe {
                        std::slice::from_raw_parts(a_ptr, a_len)
                    };

                    s.spawn(move || {
                        let range_n = j_end - j_start;
                        let local_chunk = chunk.min(range_n);
                        let mut chunk_buf = vec![0.0f32; local_chunk * k];

                        let mut j0 = j_start;
                        while j0 < j_end {
                            let j1 = (j0 + local_chunk).min(j_end);
                            let actual = j1 - j0;

                            for ji in 0..actual {
                                self.dequantize_row_into(
                                    j0 + ji,
                                    &mut chunk_buf[ji * k..(ji + 1) * k],
                                );
                            }

                            unsafe {
                                cblas::sgemm(
                                    cblas::Layout::RowMajor,
                                    cblas::Transpose::None,
                                    cblas::Transpose::Ordinary,
                                    m as i32,
                                    actual as i32,
                                    k as i32,
                                    1.0f32,
                                    a_slice,
                                    k as i32,
                                    &chunk_buf[..actual * k],
                                    k as i32,
                                    0.0f32,
                                    &mut out_slice[j0 - j_start..],
                                    n as i32,
                                );
                            }
                            j0 = j1;
                        }
                    });
                }
            });
        } else {
            // Single-threaded fallback.
            let mut chunk_buf = vec![0.0f32; chunk * k];
            let mut j0 = 0usize;
            while j0 < n {
                let j1 = (j0 + chunk).min(n);
                let actual = j1 - j0;
                for ji in 0..actual {
                    self.dequantize_row_into(j0 + ji, &mut chunk_buf[ji * k..(ji + 1) * k]);
                }
                unsafe {
                    cblas::sgemm(
                        cblas::Layout::RowMajor,
                        cblas::Transpose::None,
                        cblas::Transpose::Ordinary,
                        m as i32,
                        actual as i32,
                        k as i32,
                        1.0f32,
                        &a.data,
                        k as i32,
                        &chunk_buf[..actual * k],
                        k as i32,
                        0.0f32,
                        &mut out.data[j0..],
                        n as i32,
                    );
                }
                j0 = j1;
            }
        }
        out
    }

    // =========================================================================
    // Fused GEMV — M=1 decode fast path
    // =========================================================================

    /// Fused Q4_K dot product for a single weight row against activation vector.
    ///
    /// Uses the integer-accumulation trick: accumulate `nibble * activation`
    /// and `sum(activation)` separately, then apply scale/min at the sub-block
    /// level (2 multiplies per 32 elements instead of per element).
    #[inline]
    fn dot_row(&self, row_idx: usize, a: &[f32]) -> f32 {
        let k = self.cols;
        let n_blocks = k / 256;
        let base_block = row_idx * n_blocks;

        #[cfg(target_arch = "aarch64")]
        {
            unsafe {
                Self::dot_row_neon(
                    self.blocks.as_ptr(),
                    a.as_ptr(),
                    base_block,
                    n_blocks,
                )
            }
        }

        #[cfg(not(target_arch = "aarch64"))]
        {
            let mut acc = 0.0f32;
            for b in 0..n_blocks {
                let boff = (base_block + b) * 144;
                let d = Self::f16_to_f32(u16::from_le_bytes([
                    self.blocks[boff],
                    self.blocks[boff + 1],
                ]));
                let dmin = Self::f16_to_f32(u16::from_le_bytes([
                    self.blocks[boff + 2],
                    self.blocks[boff + 3],
                ]));
                let sc = &self.blocks[boff + 4..boff + 16];
                let qs = &self.blocks[boff + 16..boff + 144];
                let a_base = b * 256;

                for chunk in 0..4usize {
                    let (sv1, mv1) = Self::scale_min(sc, chunk * 2);
                    let (sv2, mv2) = Self::scale_min(sc, chunk * 2 + 1);
                    let scale1 = d * sv1;
                    let min1 = dmin * mv1;
                    let scale2 = d * sv2;
                    let min2 = dmin * mv2;
                    let q = &qs[chunk * 32..(chunk + 1) * 32];
                    let a_off = a_base + chunk * 64;

                    let mut dot_lo = 0.0f32;
                    let mut dot_hi = 0.0f32;
                    let mut sum_a_lo = 0.0f32;
                    let mut sum_a_hi = 0.0f32;

                    for l in 0..32 {
                        let a_lo = a[a_off + l];
                        dot_lo += (q[l] & 0x0F) as f32 * a_lo;
                        sum_a_lo += a_lo;

                        let a_hi = a[a_off + 32 + l];
                        dot_hi += (q[l] >> 4) as f32 * a_hi;
                        sum_a_hi += a_hi;
                    }

                    acc += scale1 * dot_lo - min1 * sum_a_lo
                         + scale2 * dot_hi - min2 * sum_a_hi;
                }
            }
            acc
        }
    }

    /// NEON-accelerated fused Q4_K dot product for one row.
    ///
    /// For each 32-byte chunk (64 elements), processes 16 packed nibbles at a
    /// time: extracts lo/hi nibbles into u8x16, widens to f32x4 groups, FMA
    /// with activation vector, and separately accumulates sum(activation) for
    /// the min-subtraction term.
    ///
    /// 4 independent accumulators per sub-block hide FMA latency.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_row_neon(
        blocks_ptr: *const u8,
        a_ptr: *const f32,
        base_block: usize,
        n_blocks: usize,
    ) -> f32 {
        use std::arch::aarch64::*;

        unsafe {
            let mask_0f = vdupq_n_u8(0x0F);
            let mut total = 0.0f32;

            for b in 0..n_blocks {
                let boff = (base_block + b) * 144;
                let bp = blocks_ptr.add(boff);

                // Read f16 scale (d) and f16 min (dmin) from header
                let d = Self::f16_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
                let dmin = Self::f16_to_f32(u16::from_le_bytes([*bp.add(2), *bp.add(3)]));

                let sc_ptr = bp.add(4);
                let qs_ptr = bp.add(16);
                let a_base = b * 256;

                for chunk in 0..4usize {
                    // Decode 6-bit scale/min for the two sub-blocks in this chunk
                    let sc = std::slice::from_raw_parts(sc_ptr, 12);
                    let (sv1, mv1) = Self::scale_min(sc, chunk * 2);
                    let (sv2, mv2) = Self::scale_min(sc, chunk * 2 + 1);
                    let scale1 = d * sv1;
                    let min1 = dmin * mv1;
                    let scale2 = d * sv2;
                    let min2 = dmin * mv2;

                    let q = qs_ptr.add(chunk * 32);
                    let a_lo_ptr = a_ptr.add(a_base + chunk * 64);
                    let a_hi_ptr = a_ptr.add(a_base + chunk * 64 + 32);

                    // Accumulators: dot(nibble, activation) and sum(activation)
                    let mut dot_lo = vdupq_n_f32(0.0);
                    let mut dot_hi = vdupq_n_f32(0.0);
                    let mut sum_lo = vdupq_n_f32(0.0);
                    let mut sum_hi = vdupq_n_f32(0.0);

                    // Process 32 bytes = 16+16 in two passes of 16
                    for half in 0..2u32 {
                        let off = (half * 16) as usize;
                        let raw = vld1q_u8(q.add(off));
                        let lo_nib = vandq_u8(raw, mask_0f);
                        let hi_nib = vshrq_n_u8::<4>(raw);

                        // Low nibbles → 4 groups of f32x4, dot with activation
                        let lo8a = vmovl_u8(vget_low_u8(lo_nib));
                        let g0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8a)));
                        let g1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8a)));
                        let lo8b = vmovl_u8(vget_high_u8(lo_nib));
                        let g2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo8b)));
                        let g3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(lo8b)));

                        let a0 = vld1q_f32(a_lo_ptr.add(off));
                        let a1 = vld1q_f32(a_lo_ptr.add(off + 4));
                        let a2 = vld1q_f32(a_lo_ptr.add(off + 8));
                        let a3 = vld1q_f32(a_lo_ptr.add(off + 12));

                        dot_lo = vfmaq_f32(dot_lo, g0, a0);
                        dot_lo = vfmaq_f32(dot_lo, g1, a1);
                        dot_lo = vfmaq_f32(dot_lo, g2, a2);
                        dot_lo = vfmaq_f32(dot_lo, g3, a3);

                        sum_lo = vaddq_f32(sum_lo, a0);
                        sum_lo = vaddq_f32(sum_lo, a1);
                        sum_lo = vaddq_f32(sum_lo, a2);
                        sum_lo = vaddq_f32(sum_lo, a3);

                        // High nibbles → 4 groups of f32x4, dot with activation
                        let hi8a = vmovl_u8(vget_low_u8(hi_nib));
                        let h0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8a)));
                        let h1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8a)));
                        let hi8b = vmovl_u8(vget_high_u8(hi_nib));
                        let h2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi8b)));
                        let h3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(hi8b)));

                        let b0 = vld1q_f32(a_hi_ptr.add(off));
                        let b1 = vld1q_f32(a_hi_ptr.add(off + 4));
                        let b2 = vld1q_f32(a_hi_ptr.add(off + 8));
                        let b3 = vld1q_f32(a_hi_ptr.add(off + 12));

                        dot_hi = vfmaq_f32(dot_hi, h0, b0);
                        dot_hi = vfmaq_f32(dot_hi, h1, b1);
                        dot_hi = vfmaq_f32(dot_hi, h2, b2);
                        dot_hi = vfmaq_f32(dot_hi, h3, b3);

                        sum_hi = vaddq_f32(sum_hi, b0);
                        sum_hi = vaddq_f32(sum_hi, b1);
                        sum_hi = vaddq_f32(sum_hi, b2);
                        sum_hi = vaddq_f32(sum_hi, b3);
                    }

                    // Reduce: scale*dot - min*sum for each sub-block
                    total += scale1 * vaddvq_f32(dot_lo) - min1 * vaddvq_f32(sum_lo)
                           + scale2 * vaddvq_f32(dot_hi) - min2 * vaddvq_f32(sum_hi);
                }
            }

            total
        }
    }

    /// Quantize an f32 activation vector to Q8 (int8 per 32-element block).
    fn quantize_activation_q8(a: &[f32]) -> Q8Activation {
        let k = a.len();
        let n_blocks = k.div_ceil(32);
        let mut q8 = vec![0i8; n_blocks * 32];
        let mut scales = vec![0.0f32; n_blocks];
        let mut sums = vec![0.0f32; n_blocks];

        for blk in 0..n_blocks {
            let start = blk * 32;
            let end = (start + 32).min(k);

            let mut max_abs = 0.0f32;
            let mut sum = 0.0f32;
            for i in start..end {
                max_abs = max_abs.max(a[i].abs());
                sum += a[i];
            }
            sums[blk] = sum;

            if max_abs == 0.0 {
                scales[blk] = 0.0;
                continue;
            }

            let scale = max_abs / 127.0;
            let inv_scale = 1.0 / scale;
            scales[blk] = scale;

            for i in start..end {
                q8[i] = (a[i] * inv_scale).round().clamp(-128.0, 127.0) as i8;
            }
        }

        Q8Activation { q8, scales, sums }
    }

    /// SDOT-accelerated Q4K × Q8 dot product for one row (inline asm).
    ///
    /// Uses ARM SDOT instruction to compute 4×(4×i8→i32) per cycle,
    /// avoiding the expensive u8→u16→u32→f32 widening chain of dot_row_neon.
    #[cfg(target_arch = "aarch64")]
    #[inline]
    unsafe fn dot_row_q8(
        blocks_ptr: *const u8,
        q8: &Q8Activation,
        base_block: usize,
        n_blocks: usize,
    ) -> f32 {
        use std::arch::aarch64::*;

        unsafe {
            let mask_0f = vdupq_n_u8(0x0F);
            let mut total = 0.0f32;

            for b in 0..n_blocks {
                let boff = (base_block + b) * 144;
                let bp = blocks_ptr.add(boff);

                let d = Self::f16_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
                let dmin = Self::f16_to_f32(u16::from_le_bytes([*bp.add(2), *bp.add(3)]));

                let sc_ptr = bp.add(4);
                let qs_ptr = bp.add(16);
                let a_base = b * 256; // offset into q8 arrays

                for chunk in 0..4usize {
                    let sc = std::slice::from_raw_parts(sc_ptr, 12);
                    let (sv1, mv1) = Self::scale_min(sc, chunk * 2);
                    let (sv2, mv2) = Self::scale_min(sc, chunk * 2 + 1);

                    let q = qs_ptr.add(chunk * 32);
                    let q8_lo_off = a_base + chunk * 64;
                    let q8_hi_off = a_base + chunk * 64 + 32;
                    let q8_lo_blk = q8_lo_off / 32; // scale/sum block index
                    let q8_hi_blk = q8_hi_off / 32;

                    // Integer dot products using SDOT
                    let mut isum_lo = vdupq_n_s32(0);
                    let mut isum_hi = vdupq_n_s32(0);

                    // Process 32 packed bytes in two passes of 16
                    for half in 0..2u32 {
                        let off = (half * 16) as usize;
                        let raw = vld1q_u8(q.add(off));
                        let lo_nib = vreinterpretq_s8_u8(vandq_u8(raw, mask_0f));
                        let hi_nib = vreinterpretq_s8_u8(vshrq_n_u8::<4>(raw));

                        // Load Q8-quantized activations
                        let a_lo = vld1q_s8(q8.q8.as_ptr().add(q8_lo_off + off));
                        let a_hi = vld1q_s8(q8.q8.as_ptr().add(q8_hi_off + off));

                        // SDOT: isum += dot4(nibbles, q8_activations) per lane
                        // Using inline asm since vdotq_s32 is nightly-only
                        std::arch::asm!(
                            "sdot {isum_lo:v}.4s, {lo_nib:v}.16b, {a_lo:v}.16b",
                            "sdot {isum_hi:v}.4s, {hi_nib:v}.16b, {a_hi:v}.16b",
                            isum_lo = inout(vreg) isum_lo,
                            isum_hi = inout(vreg) isum_hi,
                            lo_nib = in(vreg) lo_nib,
                            hi_nib = in(vreg) hi_nib,
                            a_lo = in(vreg) a_lo,
                            a_hi = in(vreg) a_hi,
                            options(pure, nomem, nostack),
                        );
                    }

                    // Reduce i32x4 → scalar
                    let int_dot_lo = vaddvq_s32(isum_lo) as f32;
                    let int_dot_hi = vaddvq_s32(isum_hi) as f32;

                    // Apply scales:
                    // acc += d * sv * q8_scale * int_dot - dmin * mv * sum_a
                    let q8_scale_lo = q8.scales[q8_lo_blk];
                    let q8_scale_hi = q8.scales[q8_hi_blk];
                    let sum_a_lo = q8.sums[q8_lo_blk];
                    let sum_a_hi = q8.sums[q8_hi_blk];

                    total += d * sv1 * q8_scale_lo * int_dot_lo - dmin * mv1 * sum_a_lo
                           + d * sv2 * q8_scale_hi * int_dot_hi - dmin * mv2 * sum_a_hi;
                }
            }

            total
        }
    }

    /// Multi-threaded GEMV: `a[1,K] @ self^T[N,K] → [1,N]`.
    ///
    /// Each thread computes fused dot products for a contiguous range of
    /// output neurons, reading Q4_K blocks directly — no dequant scratch buffer.
    fn gemv_mt(&self, a: &Mat) -> Mat {
        let n = self.rows;
        let k = self.cols;
        debug_assert_eq!(a.rows, 1);
        debug_assert_eq!(a.cols, k);

        let mut out = Mat::zeros(1, n);

        let hw_threads = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        let n_threads = hw_threads.min(n / 128).max(1);

        // Quantize activation to Q8 once (shared across all threads).
        #[cfg(target_arch = "aarch64")]
        let q8 = Self::quantize_activation_q8(&a.data[..k]);

        if n_threads > 1 {
            let out_ptr = out.data.as_mut_ptr();

            std::thread::scope(|s| {
                let chunk = n.div_ceil(n_threads);
                for tid in 0..n_threads {
                    let j0 = tid * chunk;
                    let j1 = ((tid + 1) * chunk).min(n);
                    if j0 >= n {
                        break;
                    }

                    let out_slice = unsafe {
                        std::slice::from_raw_parts_mut(out_ptr.add(j0), j1 - j0)
                    };

                    #[cfg(target_arch = "aarch64")]
                    let q8_ref = &q8;

                    s.spawn(move || {
                        let n_blocks = k / 256;
                        for j in 0..(j1 - j0) {
                            let base_block = (j0 + j) * n_blocks;
                            #[cfg(target_arch = "aarch64")]
                            {
                                out_slice[j] = unsafe {
                                    Self::dot_row_q8(
                                        self.blocks.as_ptr(),
                                        q8_ref,
                                        base_block,
                                        n_blocks,
                                    )
                                };
                            }
                            #[cfg(not(target_arch = "aarch64"))]
                            {
                                out_slice[j] = self.dot_row(j0 + j, &a.data[..k]);
                            }
                        }
                    });
                }
            });
        } else {
            let n_blocks = k / 256;
            for j in 0..n {
                let base_block = j * n_blocks;
                #[cfg(target_arch = "aarch64")]
                {
                    out.data[j] = unsafe {
                        Self::dot_row_q8(
                            self.blocks.as_ptr(),
                            &q8,
                            base_block,
                            n_blocks,
                        )
                    };
                }
                #[cfg(not(target_arch = "aarch64"))]
                {
                    out.data[j] = self.dot_row(j, &a.data[..k]);
                }
            }
        }

        out
    }
}

impl Mat {
    /// Convert this `Mat` to compact BF16 storage.
    pub fn to_bf16(&self) -> MatBf16 {
        MatBf16 {
            data: std::sync::Arc::new(self.data.iter().map(|&v| MatBf16::f32_to_bf16(v)).collect()),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// Construct a `Mat` from a `MatBf16` (dequantize on load).
    pub fn from_bf16(src: &MatBf16) -> Mat {
        src.to_f32()
    }

    /// Build a `Mat` from raw BF16 bytes (as stored in .safetensors BF16 shards).
    ///
    /// `bytes` must be `rows * cols * 2` bytes, little-endian BF16.
    pub fn from_bf16_bytes(bytes: &[u8], rows: usize, cols: usize) -> Self {
        assert_eq!(
            bytes.len(),
            rows * cols * 2,
            "from_bf16_bytes: expected {} bytes, got {}",
            rows * cols * 2,
            bytes.len()
        );
        let data: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|c| {
                let bits = u16::from_le_bytes([c[0], c[1]]);
                MatBf16::bf16_to_f32(bits)
            })
            .collect();
        Mat { data, rows, cols }
    }
}

// =============================================================================
// Gradient checkpointing
// =============================================================================
//
// ## The memory problem
//
// During backpropagation, every intermediate activation (every TensorNode
// created in the forward pass) must be kept in memory until its backward_fn
// is called.  For a GPT model with L layers, T tokens, and d_model dimensions,
// this is roughly:
//
//   L × T × d_model × (several matrices per layer) × 4 bytes
//
// For GPT-OSS-20b with L=24, T=2048, d_model=2880:
//   24 × 2048 × 2880 × ~10 × 4 bytes ≈ 5.6 GB
//
// That's on top of the weights themselves.
//
// ## The solution: recomputation
//
// Gradient checkpointing (also called "activation checkpointing" or
// "rematerialisation") trades compute for memory:
//
//   - During the forward pass, only store activations at "checkpoints"
//     (typically the input to each transformer block).
//   - During the backward pass, recompute the forward pass for each
//     segment from its checkpoint to recover the activations needed for
//     the gradient.
//
// This reduces activation memory from O(L × T × d) to O(√L × T × d)
// with only a 33% increase in compute (each layer is computed twice).
//
// ## Implementation
//
// We implement the simplest useful form: per-block checkpointing.
//
// `CheckpointedBlock` wraps a `TransformerBlock2` (or any `Module2`-shaped
// layer) and provides a `forward_checkpointed` method that:
//   1. Saves only the input tensor to the block (not all intermediate activations)
//   2. During backward, re-runs the block's forward to recover needed activations
//      then computes the block's backward
//
// ## How to use
//
//   let blocks: Vec<CheckpointedBlock> = model.blocks.iter()
//       .map(|b| CheckpointedBlock::new(b))
//       .collect();
//   for block in &blocks {
//       x = block.forward_checkpointed(&x);
//   }
//   loss.backward(); // block inputs recomputed as needed

use std::sync::Arc;

/// A wrapper that implements gradient checkpointing for any function
/// `f: TensorNode → TensorNode`.
///
/// Only the input is stored; the forward pass is re-run during backward
/// to recover intermediate activations.
pub struct Checkpoint<F>
where
    F: Fn(&TensorNode) -> TensorNode,
{
    f: Arc<F>,
}

impl<F> Checkpoint<F>
where
    F: Fn(&TensorNode) -> TensorNode + 'static,
{
    pub fn new(f: F) -> Self {
        Checkpoint { f: Arc::new(f) }
    }

    /// Run `f(x)`, but register a backward that re-computes the forward
    /// before propagating gradients.
    ///
    /// ## Memory behaviour
    ///
    /// Forward: saves only `x` (the input).  All intermediate nodes created
    ///   inside `f` are dropped immediately.
    ///
    /// Backward: re-runs `f(x)` to re-create the intermediate graph, runs
    ///   that graph's backward pass, then accumulates the gradient into `x`.
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        // Compute forward: get the output value but drop the full graph
        let out_data = (self.f)(x).data().clone();
        let out = TensorNode::leaf(out_data);

        let x_c = x.clone();
        let f_c = Arc::clone(&self.f);
        let out_c = out.clone();

        out.set_backward(
            Box::new(move || {
                // Re-run forward to rebuild the intermediate graph
                // Zero the re-created input's grad so we accumulate correctly
                x_c.zero_grad();
                let recomputed = f_c(&x_c);

                // Seed with the gradient that flowed back to `out`
                recomputed.set_grad(out_c.grad().clone());
                // Run backward through the recomputed graph
                recomputed.call_backward_fn();

                // The gradient now lives in x_c.grad (accumulated by recomputed backward)
            }),
            vec![x.clone()],
        );

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
            let end = (start + Q4_BLOCK_SIZE).min(n);

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
                    packed[k / 2] |= nibble; // low nibble
                } else {
                    packed[k / 2] |= nibble << 4; // high nibble
                }
            }
        }

        Q4Mat {
            rows: mat.rows,
            cols: mat.cols,
            packed,
            scales,
        }
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
                self.packed[k / 2] & 0x0F // low nibble
            } else {
                (self.packed[k / 2] >> 4) & 0x0F // high nibble
            };
            // Sign-extend from 4-bit two's complement
            let q = if nibble >= 8 {
                nibble as i8 - 16
            } else {
                nibble as i8
            };
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
        assert_eq!(
            k, self.cols,
            "matmul_q4_t: a.cols {} != q4.cols {}",
            k, self.cols
        );

        let mut out = Mat::zeros(m, nn);

        for j in 0..nn {
            // output column = B row
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
                    let q = if nibble >= 8 {
                        nibble as i8 - 16
                    } else {
                        nibble as i8
                    };
                    let block = flat_idx / Q4_BLOCK_SIZE;
                    let w = q as f32 * self.scales[block];
                    acc += a.at(i, p) * w;
                }
                *out.at_mut(i, j) = acc;
            }
        }

        out
    }

    /// Dequantize row `j` of this Q4Mat into the provided f32 buffer.
    /// `buf` must have length >= self.cols.
    #[inline]
    fn dequantize_row_into(&self, j: usize, buf: &mut [f32]) {
        let k = self.cols;
        let row_start = j * k;
        let mut block = row_start / Q4_BLOCK_SIZE;
        let mut block_end = (block + 1) * Q4_BLOCK_SIZE;

        for p in 0..k {
            let flat = row_start + p;
            if flat >= block_end {
                block += 1;
                block_end += Q4_BLOCK_SIZE;
            }
            let nibble = if flat & 1 == 0 {
                self.packed[flat >> 1] & 0x0F
            } else {
                (self.packed[flat >> 1] >> 4) & 0x0F
            };
            let q = if nibble >= 8 {
                nibble as i8 - 16
            } else {
                nibble as i8
            };
            buf[p] = q as f32 * self.scales[block];
        }
    }

    /// BLAS-accelerated matmul: A [M,K] @ Q4^T [N,K] → [M,N].
    ///
    /// Dispatch strategy:
    ///
    /// * **Decode path (M ≤ 4)** — dequantizes CHUNK rows of Q4 at a time into
    ///   a scratch buffer and calls a single `cblas_sgemm` per chunk.  This
    ///   reduces BLAS call overhead from N calls (e.g. 262 K for lm_head) to
    ///   N/CHUNK calls (e.g. 256 with CHUNK=1024).  Per-call overhead of ~1 µs
    ///   means 262 K sdot calls ≈ 262 ms wasted per token; chunked sgemm
    ///   reduces that to ~2–3 ms.  Chunk size targets ~8 MB scratch (fits in
    ///   Apple-Silicon L2 cache).
    ///
    /// * **Prefill path (M > 4)** — dequantizes the entire weight matrix once
    ///   into an N×K f32 buffer and calls a single `cblas_sgemm`.  The extra
    ///   N×K allocation is justified because `sgemm` can exploit multi-level
    ///   cache blocking across both M and N, giving far better throughput than
    ///   M×N individual `sdot` calls when M is large.
    #[cfg(feature = "blas")]
    pub fn matmul_q4_t_blas(&self, a: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, self.rows);
        assert_eq!(
            k, self.cols,
            "matmul_q4_t_blas: a.cols {} != q4.cols {}",
            k, self.cols
        );

        // Prefill: dequantize the full weight matrix and call sgemm once.
        if m > 4 {
            let w = self.dequantize(); // [N, K]
            return a.matmul_bt(&w); // sgemm: A[M,K] @ W^T[K,N] → [M,N]
        }

        // ── Decode path (M ≤ 4): chunked SGEMM ───────────────────────────
        // Dequantise CHUNK weight rows into a contiguous f32 scratch buffer,
        // then call sgemm once for the chunk.  The output-slice trick:
        // pass &mut out.data[j0..] with ldc = n so sgemm writes
        //   C[i, j]  →  out.data[j0 + i*n + j]  =  out[i, j0+j].  ✓
        //
        // Target ~8 MB of scratch (fits in Apple Silicon L2).
        let chunk: usize = ((8 * 1024 * 1024) / (k * 4)).max(64).min(n);
        let mut out = Mat::zeros(m, n);
        let mut chunk_buf = vec![0.0f32; chunk * k];

        let mut j0 = 0usize;
        while j0 < n {
            let j1 = (j0 + chunk).min(n);
            let actual = j1 - j0;

            // Dequantise rows j0..j1 into chunk_buf[0..actual*k].
            for ji in 0..actual {
                self.dequantize_row_into(j0 + ji, &mut chunk_buf[ji * k..(ji + 1) * k]);
            }

            // SGEMM: out[m, actual] = a[m, k] × chunk_buf[actual, k]^T
            unsafe {
                cblas::sgemm(
                    cblas::Layout::RowMajor,
                    cblas::Transpose::None,
                    cblas::Transpose::Ordinary,
                    m as i32,
                    actual as i32,
                    k as i32,
                    1.0f32,
                    &a.data,
                    k as i32,
                    &chunk_buf[..actual * k],
                    k as i32,
                    0.0f32,
                    &mut out.data[j0..],
                    n as i32,
                );
            }
            j0 = j1;
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
// Q8Mat — 8-bit symmetric block quantization
// =============================================================================
//
// ## Why Q8 when Q4 already exists?
//
// Q4 achieves 8× compression (4 bits instead of 32), but with 7 distinct
// quantization levels per block it can introduce noticeable accuracy loss on
// models with narrow weight distributions.
//
// Q8 uses 8-bit signed integers (range -127..127), giving 255 distinct levels.
// Quantization error is at most 0.4% of the block max (vs ~7% for Q4).
// Memory is 4× smaller than f32 (still 2× larger than Q4).
//
// Typical use case:
//   - KV cache quantization (low error tolerance)
//   - Activations (also low error tolerance)
//   - Small models where Q4 is too lossy
//
// ## Format
//
//   packed[k] = round(data[k] / scale[block(k)])  — one i8 per element
//   scale[b]  = absmax(block b) / 127.0
//
// ## Compression vs f32
//
//   Q8: 1 byte/element + 4 bytes/block (scales)
//   f32: 4 bytes/element
//   Ratio ≈ 4× (exact ratio depends on block size)

/// Block size for Q8 quantization.
/// Larger blocks → fewer scale values → higher compression, lower accuracy.
const Q8_BLOCK_SIZE: usize = 64;

/// 8-bit symmetric block-quantized matrix.
pub struct Q8Mat {
    pub rows: usize,
    pub cols: usize,
    /// One i8 per element, in row-major order.
    pub packed: Vec<i8>,
    /// One f32 scale per block of `Q8_BLOCK_SIZE` elements.
    pub scales: Vec<f32>,
}

impl Q8Mat {
    /// Quantize a `Mat` to 8-bit symmetric block quantization.
    pub fn quantize(mat: &Mat) -> Self {
        let n = mat.rows * mat.cols;
        let n_blocks = n.div_ceil(Q8_BLOCK_SIZE);
        let mut packed = vec![0i8; n];
        let mut scales = vec![0.0f32; n_blocks];

        for block in 0..n_blocks {
            let start = block * Q8_BLOCK_SIZE;
            let end = (start + Q8_BLOCK_SIZE).min(n);

            let absmax = mat.data[start..end]
                .iter()
                .map(|x| x.abs())
                .fold(0.0f32, f32::max);

            let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
            scales[block] = scale;

            for k in start..end {
                packed[k] = (mat.data[k] / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }

        Q8Mat {
            rows: mat.rows,
            cols: mat.cols,
            packed,
            scales,
        }
    }

    /// Dequantize: recover an approximate `Mat`.
    ///
    /// Quantization error per element ≤ 0.5 * scale ≤ absmax/254.
    pub fn dequantize(&self) -> Mat {
        let n = self.rows * self.cols;
        let data: Vec<f32> = (0..n)
            .map(|k| {
                let block = k / Q8_BLOCK_SIZE;
                self.packed[k] as f32 * self.scales[block]
            })
            .collect();
        Mat::new(data, self.rows, self.cols)
    }

    /// Fused matrix multiplication: (A: f32) @ (B: Q8).T
    ///
    /// Shape: A is [M, K], B is [N, K] → [M, N]
    ///
    /// Dequantizes B row-by-row on the fly (no full materialization).
    pub fn matmul_q8_t(&self, a: &Mat) -> Mat {
        let (m, k, nn) = (a.rows, a.cols, self.rows);
        assert_eq!(
            k, self.cols,
            "matmul_q8_t: a.cols {} != q8.cols {}",
            k, self.cols
        );

        let mut out = Mat::zeros(m, nn);
        for j in 0..nn {
            let row_start = j * k;
            for i in 0..m {
                let mut acc = 0.0f32;
                for p in 0..k {
                    let flat = row_start + p;
                    let block = flat / Q8_BLOCK_SIZE;
                    let w = self.packed[flat] as f32 * self.scales[block];
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
// Checkpoint save/load — binary format for TensorNode weights
// =============================================================================
//
// ## Format
//
// A checkpoint file stores a flat sequence of tensors in binary:
//
//   [magic: u32 = 0x4358504B "CXPK"]
//   [version: u32 = 1]
//   [n_tensors: u32]
//   For each tensor:
//     [name_len: u32]
//     [name: name_len bytes, UTF-8]
//     [rows: u32]
//     [cols: u32]
//     [data: rows*cols f32 in little-endian]
//
// ## Usage
//
//   save_checkpoint("model.ckpt", &[("embed", &embed_node), ("lm_head.w", &lm_node)]).unwrap();
//   let ckpt = load_checkpoint("model.ckpt").unwrap();
//   for (name, mat) in &ckpt { ... }

const CKPT_MAGIC: u32 = 0x4358504B; // "CXPK"
const CKPT_VERSION: u32 = 1;

/// Save a list of named tensors to a binary checkpoint file.
///
/// The tensors are identified by name, so the order does not have to match
/// the loading order.
pub fn save_checkpoint(path: &str, tensors: &[(&str, &TensorNode)]) -> Result<(), String> {
    use std::io::Write;
    let mut buf: Vec<u8> = Vec::new();

    // Header
    buf.extend_from_slice(&CKPT_MAGIC.to_le_bytes());
    buf.extend_from_slice(&CKPT_VERSION.to_le_bytes());
    buf.extend_from_slice(&(tensors.len() as u32).to_le_bytes());

    for (name, node) in tensors {
        let name_bytes = name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        let data = node.data();
        buf.extend_from_slice(&(data.rows as u32).to_le_bytes());
        buf.extend_from_slice(&(data.cols as u32).to_le_bytes());
        for &v in &data.data {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    let mut file = std::fs::File::create(path)
        .map_err(|e| format!("save_checkpoint: cannot create {}: {}", path, e))?;
    file.write_all(&buf)
        .map_err(|e| format!("save_checkpoint: write failed: {}", e))?;
    Ok(())
}

/// Load a checkpoint file, returning a list of (name, Mat) pairs.
///
/// The caller is responsible for matching names to model parameters.
pub fn load_checkpoint(path: &str) -> Result<Vec<(String, Mat)>, String> {
    let bytes =
        std::fs::read(path).map_err(|e| format!("load_checkpoint: cannot read {}: {}", path, e))?;

    let mut pos = 0usize;

    let read_u32 = |b: &[u8], p: &mut usize| -> Result<u32, String> {
        if *p + 4 > b.len() {
            return Err("unexpected EOF reading u32".to_string());
        }
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        Ok(v)
    };
    let read_f32 = |b: &[u8], p: &mut usize| -> Result<f32, String> {
        if *p + 4 > b.len() {
            return Err("unexpected EOF reading f32".to_string());
        }
        let v = f32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        Ok(v)
    };

    let magic = read_u32(&bytes, &mut pos)?;
    let version = read_u32(&bytes, &mut pos)?;
    if magic != CKPT_MAGIC {
        return Err(format!(
            "load_checkpoint: bad magic 0x{:08X} (expected 0x{:08X})",
            magic, CKPT_MAGIC
        ));
    }
    if version != CKPT_VERSION {
        return Err(format!("load_checkpoint: unsupported version {}", version));
    }

    let n_tensors = read_u32(&bytes, &mut pos)? as usize;
    let mut tensors = Vec::with_capacity(n_tensors);

    for _ in 0..n_tensors {
        let name_len = read_u32(&bytes, &mut pos)? as usize;
        if pos + name_len > bytes.len() {
            return Err("unexpected EOF reading name".to_string());
        }
        let name = std::str::from_utf8(&bytes[pos..pos + name_len])
            .map_err(|e| format!("invalid UTF-8 name: {}", e))?
            .to_string();
        pos += name_len;

        let rows = read_u32(&bytes, &mut pos)? as usize;
        let cols = read_u32(&bytes, &mut pos)? as usize;
        let n_elem = rows * cols;
        let mut data = Vec::with_capacity(n_elem);
        for _ in 0..n_elem {
            data.push(read_f32(&bytes, &mut pos)?);
        }
        tensors.push((name, Mat::new(data, rows, cols)));
    }

    Ok(tensors)
}

/// Restore model parameters from a checkpoint file.
///
/// Loads the checkpoint at `path` and applies each tensor to the matching
/// parameter in `params` by position (index order must match the order used
/// when saving).  Shape mismatches are reported as errors.
///
/// ## Example
/// ```ignore
/// let params = model.parameters();
/// restore_checkpoint("model.ckpt", &params)?;
/// ```
pub fn restore_checkpoint(path: &str, params: &[TensorNode]) -> Result<(), String> {
    let tensors = load_checkpoint(path)?;
    if tensors.len() != params.len() {
        return Err(format!(
            "restore_checkpoint: checkpoint has {} tensors but model has {} parameters",
            tensors.len(),
            params.len()
        ));
    }
    for (i, ((name, mat), param)) in tensors.iter().zip(params.iter()).enumerate() {
        let p_data = param.data();
        if mat.rows != p_data.rows || mat.cols != p_data.cols {
            return Err(format!(
                "restore_checkpoint: tensor {} ('{}') shape [{},{}] does not match parameter shape [{},{}]",
                i, name, mat.rows, mat.cols, p_data.rows, p_data.cols
            ));
        }
        drop(p_data);
        param.set_data(mat.clone());
    }
    Ok(())
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
            inner
                .backward_fn
                .as_ref()
                .map(|f| unsafe { &*(f.as_ref() as *const dyn Fn()) })
        };
        if let Some(f) = fn_ptr {
            f();
        }
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

        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| x.at(r, c) * sigmoid_vals.at(r, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();

            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let xv = x_data.at(r, c);
                let z = 1.702 * xv;
                let s = 1.0 / (1.0 + (-z).exp()); // sigmoid(z)
                let dgelu_dx = s + xv * s * (1.0 - s) * 1.702;
                dout.at(r, c) * dgelu_dx
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// GELU with the PyTorch tanh approximation (used by Gemma 3):
    ///   gelu_tanh(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    ///
    /// This is the standard `gelu_pytorch_tanh` activation.
    pub fn gelu_tanh(&self) -> TensorNode {
        let x = self.data().clone();
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654; // sqrt(2/pi)
        const COEFF: f32 = 0.044715;

        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| {
            let xv = x.at(r, c);
            let inner = SQRT_2_OVER_PI * (xv + COEFF * xv * xv * xv);
            xv * 0.5 * (1.0 + inner.tanh())
        });
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();
            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let xv = x_data.at(r, c);
                let x3 = xv * xv * xv;
                let inner = SQRT_2_OVER_PI * (xv + COEFF * x3);
                let t = inner.tanh();
                let sech2 = 1.0 - t * t; // sech^2
                let dg = 0.5 * (1.0 + t)
                    + xv * 0.5 * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * COEFF * xv * xv);
                dout.at(r, c) * dg
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
            let ds = out_c.0.borrow().grad.clone(); // upstream gradient
            let s = out_c.0.borrow().data.clone(); // forward softmax values
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
        let mut mean = vec![0.0f32; t];
        let mut var = vec![0.0f32; t];
        let mut x_hat = Mat::zeros(t, d);

        for r in 0..t {
            mean[r] = (0..d).map(|c| x.at(r, c)).sum::<f32>() * inv_d;
            var[r] = (0..d).map(|c| (x.at(r, c) - mean[r]).powi(2)).sum::<f32>() * inv_d;
            let inv_std = 1.0 / (var[r] + eps).sqrt();
            for c in 0..d {
                *x_hat.at_mut(r, c) = (x.at(r, c) - mean[r]) * inv_std;
            }
        }

        // Y = gamma * X̂ + beta  (broadcast gamma/beta across rows)
        let out_data = Mat::from_fn(t, d, |r, c| x_hat.at(r, c) * g.at(0, c) + b.at(0, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let gamma_c = gamma.clone();
        let beta_c = beta.clone();
        let out_c = out.clone();
        let x_hat_stored = x_hat.clone(); // need X̂ in backward
        let var_stored = var.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dy = out_c.0.borrow().grad.clone();
            let g = gamma_c.0.borrow().data.clone();

            // dγ = sum_rows(dY * X̂)   shape [1, d]
            let mut dg = Mat::zeros(1, d);
            for c in 0..d {
                for r in 0..t {
                    *dg.at_mut(0, c) += dy.at(r, c) * x_hat_stored.at(r, c);
                }
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
                let mean_d = d_row.iter().sum::<f32>() * inv_d;
                let mean_dxh = d_row
                    .iter()
                    .enumerate()
                    .map(|(c, &dv)| dv * x_hat_stored.at(r, c))
                    .sum::<f32>()
                    * inv_d;
                for c in 0..d {
                    *dx.at_mut(r, c) =
                        inv_std * (d_row[c] - mean_d - x_hat_stored.at(r, c) * mean_dxh);
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
    pub fn causal_attention(
        q: &TensorNode,
        k: &TensorNode,
        v: &TensorNode,
        d_head: usize,
    ) -> TensorNode {
        let q_d = q.data().clone();
        let k_d = k.data().clone();
        let v_d = v.data().clone();
        let t = q_d.rows;
        let scale = 1.0 / (d_head as f32).sqrt();

        // scores = Q @ K.T * scale  [T, T]
        let mut scores = q_d.matmul(&k_d.transpose()).scale(scale);

        // Apply causal mask
        for i in 0..t {
            for j in (i + 1)..t {
                *scores.at_mut(i, j) = -1e9;
            }
        }

        // weights = softmax(scores) row-wise  [T, T]
        let weights = {
            let mut w = Mat::zeros(t, t);
            for r in 0..t {
                let row_max = (0..t)
                    .map(|c| scores.at(r, c))
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0f32;
                for c in 0..t {
                    let e = (scores.at(r, c) - row_max).exp();
                    *w.at_mut(r, c) = e;
                    row_sum += e;
                }
                for c in 0..t {
                    *w.at_mut(r, c) /= row_sum;
                }
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
            let w = &weights_stored;
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
        let mut x_hat = Mat::zeros(t, d); // x̂ = x * r

        for r in 0..t {
            let mean_sq = (0..d).map(|c| x.at(r, c).powi(2)).sum::<f32>() * inv_d;
            rms_inv[r] = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..d {
                *x_hat.at_mut(r, c) = x.at(r, c) * rms_inv[r];
            }
        }

        let out_data = Mat::from_fn(t, d, |r, c| x_hat.at(r, c) * g.at(0, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let gamma_c = gamma.clone();
        let out_c = out.clone();
        let x_hat_s = x_hat.clone();
        let rms_inv_s = rms_inv.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let g = gamma_c.0.borrow().data.clone();

            // dγ = sum_t(dout[t] * x̂[t])   shape [1, d]
            let mut dg = Mat::zeros(1, d);
            for c in 0..d {
                for r in 0..t {
                    *dg.at_mut(0, c) += dout.at(r, c) * x_hat_s.at(r, c);
                }
            }
            gamma_c.0.borrow_mut().grad.add_assign(&dg);

            // dx[t] = r[t] * (D[t] - x̂[t]*mean(D[t]*x̂[t]))
            // where D[t,i] = dout[t,i]*γ[i]
            let mut dx = Mat::zeros(t, d);
            for r in 0..t {
                let d_row: Vec<f32> = (0..d).map(|c| dout.at(r, c) * g.at(0, c)).collect();
                let mean_dxh = d_row
                    .iter()
                    .enumerate()
                    .map(|(c, &dv)| dv * x_hat_s.at(r, c))
                    .sum::<f32>()
                    * inv_d;
                for c in 0..d {
                    *dx.at_mut(r, c) = rms_inv_s[r] * (d_row[c] - x_hat_s.at(r, c) * mean_dxh);
                }
            }
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone(), gamma.clone()];
        out
    }

    /// Gemma3 RMS Layer Normalization — uses `(1 + gamma)` scaling.
    ///
    /// Gemma3 stores gamma initialized to **zeros** (not ones) and applies
    /// the formula:
    ///   RMSNorm_gemma3(x) = x / sqrt(mean(x²) + eps) * (1 + gamma)
    ///
    /// This differs from the standard `rms_norm` which applies just `gamma`.
    /// All norm layers in Gemma3 (input_layernorm, post_attention_layernorm,
    /// pre_feedforward_layernorm, post_feedforward_layernorm, q_norm, k_norm,
    /// and the final norm) use this formula.
    pub fn rms_norm_gemma3(&self, gamma: &TensorNode, eps: f32) -> TensorNode {
        let x = self.data().clone();
        let g = gamma.data().clone();
        let (t, d) = (x.rows, x.cols);
        let inv_d = 1.0 / d as f32;

        let mut rms_inv = vec![0.0f32; t];
        let mut x_hat = Mat::zeros(t, d);

        for r in 0..t {
            let mean_sq = (0..d).map(|c| x.at(r, c).powi(2)).sum::<f32>() * inv_d;
            rms_inv[r] = 1.0 / (mean_sq + eps).sqrt();
            for c in 0..d {
                *x_hat.at_mut(r, c) = x.at(r, c) * rms_inv[r];
            }
        }

        // (1 + gamma) scaling
        let out_data = Mat::from_fn(t, d, |r, c| x_hat.at(r, c) * (1.0 + g.at(0, c)));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let gamma_c = gamma.clone();
        let out_c = out.clone();
        let x_hat_s = x_hat.clone();
        let rms_inv_s = rms_inv.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let g = gamma_c.0.borrow().data.clone();

            // dγ = sum_t(dout[t] * x̂[t])  (same as rms_norm since d/dgamma of (1+g)*x̂ = x̂)
            let mut dg = Mat::zeros(1, d);
            for c in 0..d {
                for r in 0..t {
                    *dg.at_mut(0, c) += dout.at(r, c) * x_hat_s.at(r, c);
                }
            }
            gamma_c.0.borrow_mut().grad.add_assign(&dg);

            // dx uses (1 + gamma) as the effective scale
            let mut dx = Mat::zeros(t, d);
            for r in 0..t {
                let d_row: Vec<f32> = (0..d).map(|c| dout.at(r, c) * (1.0 + g.at(0, c))).collect();
                let mean_dxh = d_row
                    .iter()
                    .enumerate()
                    .map(|(c, &dv)| dv * x_hat_s.at(r, c))
                    .sum::<f32>()
                    * inv_d;
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
        let sig = Mat::from_fn(x.rows, x.cols, |r, c| 1.0 / (1.0 + (-x.at(r, c)).exp()));

        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| x.at(r, c) * sig.at(r, c));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();

            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let xv = x_data.at(r, c);
                let s = 1.0 / (1.0 + (-xv).exp());
                // d/dx[x*σ] = σ + x*σ*(1-σ)
                let dsilu = s + xv * s * (1.0 - s);
                dout.at(r, c) * dsilu
            });
            self_c.0.borrow_mut().grad.add_assign(&dx);
        }));
        out.0.borrow_mut().prev = vec![self.clone()];
        out
    }

    /// Element-wise clamp to [min_val, max_val].
    ///
    /// Used in SwiGLU with `swiglu_limit=7.0` (GPT-OSS config): clamps the gate
    /// pre-activation before SiLU to prevent saturation in early training.
    ///
    /// Backward: gradient passes through where input was in range, zero otherwise.
    pub fn clamp(&self, min_val: f32, max_val: f32) -> TensorNode {
        let x = self.data().clone();
        let out_data = Mat::from_fn(x.rows, x.cols, |r, c| x.at(r, c).clamp(min_val, max_val));
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let x_data = self_c.0.borrow().data.clone();
            let dx = Mat::from_fn(x_data.rows, x_data.cols, |r, c| {
                let v = x_data.at(r, c);
                if v > min_val && v < max_val {
                    dout.at(r, c)
                } else {
                    0.0
                }
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
        assert_eq!(
            (a.rows, a.cols),
            (b.rows, b.cols),
            "mul_elem_node: shape mismatch [{},{}] vs [{},{}]",
            a.rows,
            a.cols,
            b.rows,
            b.cols
        );

        let out_data = a.mul_elem(&b);
        let out = TensorNode::leaf(out_data);

        let self_c = self.clone();
        let other_c = other.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
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
            let pair = col / 2;
            let is_odd = col % 2 == 1;
            let angle = angles.at(row, pair);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            if !is_odd {
                x.at(row, col) * cos_a - x.at(row, col + 1) * sin_a
            } else {
                x.at(row, col) * cos_a + x.at(row, col - 1) * sin_a
            }
        });

        let out = TensorNode::leaf(out_data);
        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();

            // Backward: inverse rotation by -angle
            let dx = Mat::from_fn(t, d, |row, col| {
                let pair = col / 2;
                let is_odd = col % 2 == 1;
                let angle = angles.at(row, pair);
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
        seq_offset: usize,
        theta: f32,
        original_ctx: usize,
        max_ctx: usize,
        beta_fast: f32,
        beta_slow: f32,
    ) -> TensorNode {
        let x = self.data().clone();
        let (t, d) = (x.rows, x.cols);
        assert!(
            d % 2 == 0,
            "rope_apply_yarn: d_head must be even, got {}",
            d
        );

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
            let pair = col / 2;
            let is_odd = col % 2 == 1;
            let angle = angles.at(row, pair);
            let (cos_a, sin_a) = (angle.cos(), angle.sin());
            if !is_odd {
                x.at(row, col) * cos_a - x.at(row, col + 1) * sin_a
            } else {
                x.at(row, col) * cos_a + x.at(row, col - 1) * sin_a
            }
        });

        let out = TensorNode::leaf(out_data);
        let self_c = self.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let dx = Mat::from_fn(t, d, |row, col| {
                let pair = col / 2;
                let is_odd = col % 2 == 1;
                let angle = angles.at(row, pair);
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
        let mut out_data = Mat::zeros(t, n_q_heads * d_head);
        let mut all_weights = vec![Mat::zeros(t, t); n_q_heads]; // one per q-head

        for qh in 0..n_q_heads {
            let kvh = qh / group_size;

            let q_h = Mat::from_fn(t, d_head, |r, c| q_data.at(r, qh * d_head + c));
            let k_h = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kvh * d_head + c));
            let v_h = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kvh * d_head + c));

            let mut scores = q_h.matmul(&k_h.transpose()).scale(scale);
            for i in 0..t {
                for j in (i + 1)..t {
                    *scores.at_mut(i, j) = -1e9;
                }
            }

            let mut w = Mat::zeros(t, t);
            for r in 0..t {
                let row_max = (0..t)
                    .map(|c| scores.at(r, c))
                    .fold(f32::NEG_INFINITY, f32::max);
                let mut row_sum = 0.0f32;
                for c in 0..t {
                    let e = (scores.at(r, c) - row_max).exp();
                    *w.at_mut(r, c) = e;
                    row_sum += e;
                }
                for c in 0..t {
                    *w.at_mut(r, c) /= row_sum;
                }
            }

            let out_h = w.matmul(&v_h);
            for r in 0..t {
                for c in 0..d_head {
                    *out_data.at_mut(r, qh * d_head + c) = out_h.at(r, c);
                }
            }
            all_weights[qh] = w;
        }

        let out = TensorNode::leaf(out_data);
        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        let out_c = out.clone();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone();
            let q_data = q_c.0.borrow().data.clone();
            let k_data = k_c.0.borrow().data.clone();
            let v_data = v_c.0.borrow().data.clone();

            let mut dq_data = Mat::zeros(t, n_q_heads * d_head);
            let mut dk_data = Mat::zeros(t, n_kv_heads * d_head);
            let mut dv_data = Mat::zeros(t, n_kv_heads * d_head);

            for qh in 0..n_q_heads {
                let kvh = qh / group_size;
                let w = &all_weights[qh];

                let dout_h = Mat::from_fn(t, d_head, |r, c| dout.at(r, qh * d_head + c));
                let q_h = Mat::from_fn(t, d_head, |r, c| q_data.at(r, qh * d_head + c));
                let k_h = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kvh * d_head + c));
                let v_h = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kvh * d_head + c));

                // dV_kvh += W.T @ dOut_h
                let dv_h = w.transpose().matmul(&dout_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dv_data.at_mut(r, kvh * d_head + c) += dv_h.at(r, c);
                    }
                }

                // dW = dOut_h @ V_h.T  [T, T]
                let dw = dout_h.matmul(&v_h.transpose());

                // Backward through causal softmax
                let mut dscores = Mat::zeros(t, t);
                for r in 0..t {
                    let dot: f32 = (0..=r).map(|c| dw.at(r, c) * w.at(r, c)).sum();
                    for c in 0..=r {
                        *dscores.at_mut(r, c) = w.at(r, c) * (dw.at(r, c) - dot);
                    }
                }
                let dscores = dscores.scale(scale);

                // dQ_h += dScores @ K_h
                let dq_h = dscores.matmul(&k_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dq_data.at_mut(r, qh * d_head + c) += dq_h.at(r, c);
                    }
                }

                // dK_kvh += dScores.T @ Q_h
                let dk_h = dscores.transpose().matmul(&q_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dk_data.at_mut(r, kvh * d_head + c) += dk_h.at(r, c);
                    }
                }
            }

            q_c.0.borrow_mut().grad.add_assign(&dq_data);
            k_c.0.borrow_mut().grad.add_assign(&dk_data);
            v_c.0.borrow_mut().grad.add_assign(&dv_data);
        }));
        out.0.borrow_mut().prev = vec![q.clone(), k.clone(), v.clone()];
        out
    }

    /// Batched GQA attention — same semantics as `gqa_attention` but uses
    /// `NDArray::bmm` for the score and output matmuls instead of a per-head loop.
    ///
    /// ## Forward
    ///
    /// 1. Reshape Q `[T, n_q*D]` → `[n_q, T, D]`
    ///    Reshape K/V `[T, n_kv*D]` → `[n_kv, T, D]`
    /// 2. Expand K/V heads by the group factor: `[n_q, T, D]`
    /// 3. scores `[n_q, T, T]` = Q `[n_q, T, D]` @ K.permute(0,2,1) `[n_q, D, T]` * scale
    /// 4. Apply causal mask (upper-triangular = -inf)
    /// 5. Softmax over last axis
    /// 6. out `[n_q, T, D]` = weights `[n_q, T, T]` @ V `[n_q, T, D]`
    /// 7. Reshape back to `[T, n_q*D]`
    ///
    /// ## Backward
    ///
    /// Same math as `gqa_attention` backward but expressed with batched matmuls:
    ///   dV   = W.T @ dOut         (per head)
    ///   dW   = dOut @ V.T         (per head)
    ///   dS   = W * (dW - diag(W @ dW.T) broadcast) * scale  (softmax VJP, causal)
    ///   dQ   = dS @ K             (per head)
    ///   dK  += dS.T @ Q           (per head, accumulated over q-heads in same kv group)
    pub fn batched_gqa_attention(
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
        let scale = 1.0_f32 / (d_head as f32).sqrt();
        let group_size = n_q_heads / n_kv_heads;

        assert_eq!(q_data.cols, n_q_heads * d_head);
        assert_eq!(k_data.cols, n_kv_heads * d_head);
        assert_eq!(v_data.cols, n_kv_heads * d_head);

        // ---- reshape: [T, H*D] → [H, T, D] ----
        // NDArray uses row-major; we have data laid out as T rows of H*D.
        // reshape [T, H*D] → [T, H, D] then permute [1,0,2] → [H, T, D].
        let q_nd = NDArray::from_mat(&q_data)
            .reshape(&[t, n_q_heads, d_head])
            .permute(&[1, 0, 2]); // [n_q,  T, D]
        let k_nd = NDArray::from_mat(&k_data)
            .reshape(&[t, n_kv_heads, d_head])
            .permute(&[1, 0, 2]); // [n_kv, T, D]
        let v_nd = NDArray::from_mat(&v_data)
            .reshape(&[t, n_kv_heads, d_head])
            .permute(&[1, 0, 2]); // [n_kv, T, D]

        // ---- expand KV heads to match Q heads ----
        // Each KV head serves `group_size` Q heads.
        // Build [n_q, T, D] by repeating each KV head `group_size` times.
        let k_exp = NDArray::from_fn(&[n_q_heads, t, d_head], |idx| {
            let kvh = idx[0] / group_size;
            k_nd.at(&[kvh, idx[1], idx[2]])
        });
        let v_exp = NDArray::from_fn(&[n_q_heads, t, d_head], |idx| {
            let kvh = idx[0] / group_size;
            v_nd.at(&[kvh, idx[1], idx[2]])
        });

        // ---- scores: [n_q, T, T] = Q [n_q,T,D] @ K^T [n_q,D,T] * scale ----
        let k_t = k_exp.permute(&[0, 2, 1]); // [n_q, D, T]
        let mut scores = q_nd.bmm(&k_t).scale(scale); // [n_q, T, T]

        // ---- causal mask: scores[h, i, j] = -1e9 for j > i ----
        for h in 0..n_q_heads {
            for i in 0..t {
                for j in (i + 1)..t {
                    *scores.at_mut(&[h, i, j]) = -1e9;
                }
            }
        }

        // ---- softmax over last axis ----
        let weights = scores.softmax(2); // [n_q, T, T]

        // ---- output: [n_q, T, D] = weights @ V ----
        let out_nd = weights.bmm(&v_exp); // [n_q, T, D]

        // ---- reshape back: [n_q, T, D] → [T, n_q, D] → [T, n_q*D] ----
        let out_mat = out_nd
            .permute(&[1, 0, 2]) // [T, n_q, D]
            .reshape(&[t, n_q_heads * d_head])
            .into_mat();

        let out = TensorNode::leaf(out_mat);
        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        let out_c = out.clone();

        // Store attention weights for backward (one [T,T] per q-head)
        // We extract them from the NDArray into Vec<Mat> for the closure.
        let all_weights: Vec<Mat> = (0..n_q_heads)
            .map(|h| Mat::from_fn(t, t, |r, c| weights.at(&[h, r, c])))
            .collect();

        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone(); // [T, n_q*D]
            let q_data = q_c.0.borrow().data.clone();
            let k_data = k_c.0.borrow().data.clone();
            let v_data = v_c.0.borrow().data.clone();

            let mut dq_data = Mat::zeros(t, n_q_heads * d_head);
            let mut dk_data = Mat::zeros(t, n_kv_heads * d_head);
            let mut dv_data = Mat::zeros(t, n_kv_heads * d_head);

            for qh in 0..n_q_heads {
                let kvh = qh / group_size;
                let w = &all_weights[qh];

                let dout_h = Mat::from_fn(t, d_head, |r, c| dout.at(r, qh * d_head + c));
                let q_h = Mat::from_fn(t, d_head, |r, c| q_data.at(r, qh * d_head + c));
                let k_h = Mat::from_fn(t, d_head, |r, c| k_data.at(r, kvh * d_head + c));
                let v_h = Mat::from_fn(t, d_head, |r, c| v_data.at(r, kvh * d_head + c));

                // dV += W.T @ dOut_h
                let dv_h = w.transpose().matmul(&dout_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dv_data.at_mut(r, kvh * d_head + c) += dv_h.at(r, c);
                    }
                }

                // dW = dOut_h @ V_h.T
                let dw = dout_h.matmul(&v_h.transpose());

                // Softmax backward (causal)
                let mut dscores = Mat::zeros(t, t);
                for r in 0..t {
                    let dot: f32 = (0..=r).map(|c| dw.at(r, c) * w.at(r, c)).sum();
                    for c in 0..=r {
                        *dscores.at_mut(r, c) = w.at(r, c) * (dw.at(r, c) - dot);
                    }
                }
                let dscores = dscores.scale(scale);

                // dQ_h += dScores @ K_h
                let dq_h = dscores.matmul(&k_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dq_data.at_mut(r, qh * d_head + c) += dq_h.at(r, c);
                    }
                }

                // dK_kvh += dScores.T @ Q_h
                let dk_h = dscores.transpose().matmul(&q_h);
                for r in 0..t {
                    for c in 0..d_head {
                        *dk_data.at_mut(r, kvh * d_head + c) += dk_h.at(r, c);
                    }
                }
            }

            q_c.0.borrow_mut().grad.add_assign(&dq_data);
            k_c.0.borrow_mut().grad.add_assign(&dk_data);
            v_c.0.borrow_mut().grad.add_assign(&dv_data);
        }));

        out.0.borrow_mut().prev = vec![q.clone(), k.clone(), v.clone()];
        out
    }

    /// Flash Attention — causal self-attention with O(T) memory instead of O(T²).
    ///
    /// ## Motivation
    ///
    /// Standard causal attention (`causal_attention`) materialises the full
    /// `[T, T]` score matrix in memory.  For T = 2048 and f32 that is 16 MB
    /// *per head* — and the backward pass needs it again.  At T = 8192 it is
    /// 256 MB per head, making long-context inference impractical on CPU.
    ///
    /// Flash Attention tiles the computation: it processes the sequence in
    /// BLOCK_Q × BLOCK_K chunks, accumulating the output with an online
    /// softmax normaliser so the full T×T matrix is never instantiated.
    ///
    /// ## Algorithm (Dao et al. 2022, Algorithm 1)
    ///
    /// For each query tile `q_block` of size Br:
    ///   - Maintain a running max `m` and normaliser `l` (both shape [Br])
    ///   - For each key/value tile `k_block, v_block` of size Bc:
    ///       s = q_block @ k_block.T * scale          [Br, Bc]
    ///       apply causal mask: s[i,j] = -inf if j > global_j
    ///       m_new = max(m, rowmax(s))
    ///       p    = exp(s - m_new)                    [Br, Bc] — unnormalised
    ///       l_new = exp(m - m_new) * l + rowsum(p)
    ///       acc  = diag(exp(m - m_new)) * acc + p @ v_block
    ///       m, l = m_new, l_new
    ///   - out_block = acc / l                        [Br, d_head]
    ///
    /// Memory: O(T * d_head) for the output + O(Br + Bc) temporaries.
    /// Compute: identical to standard attention (same number of multiplications).
    ///
    /// ## Backward
    ///
    /// Flash Attention backward requires recomputing the softmax weights from
    /// the stored normaliser (l, m) instead of storing the full weight matrix.
    /// This is ~2× more compute but O(T) memory.
    ///
    /// We store the per-query-tile l and m vectors (O(T) total) for the backward.
    ///
    /// ## Parameters
    ///
    /// - `q`: `[T, d_head]` — queries
    /// - `k`: `[T, d_head]` — keys
    /// - `v`: `[T, d_head]` — values
    /// - `d_head`: head dimension (used for scaling)
    ///
    /// Returns `[T, d_head]`.
    ///
    /// ## Tile sizes
    ///
    /// `BLOCK_R` (query tile) and `BLOCK_C` (key/value tile) default to 64.
    /// For small T they automatically reduce to T so the algorithm stays correct.
    pub fn flash_attention(
        q: &TensorNode,
        k: &TensorNode,
        v: &TensorNode,
        d_head: usize,
    ) -> TensorNode {
        const BLOCK_R: usize = 64;
        const BLOCK_C: usize = 64;

        let q_data = q.data().clone();
        let k_data = k.data().clone();
        let v_data = v.data().clone();
        let t = q_data.rows;
        let scale = 1.0_f32 / (d_head as f32).sqrt();

        assert_eq!(q_data.cols, d_head);
        assert_eq!(k_data.cols, d_head);
        assert_eq!(v_data.cols, d_head);

        // Output accumulator and online-softmax state stored per row.
        let mut out_data = Mat::zeros(t, d_head);
        // l[i] = running normaliser for row i (sum of exp weights)
        // m[i] = running max for row i
        let mut l_global = vec![0.0f32; t];
        let mut m_global = vec![f32::NEG_INFINITY; t];

        // Tile loop — q rows in chunks of BLOCK_R
        let mut q_start = 0;
        while q_start < t {
            let q_end = (q_start + BLOCK_R).min(t);
            let br = q_end - q_start;

            // Per-block accumulator and softmax state
            let mut acc = vec![0.0f32; br * d_head]; // [br, d_head]
            let mut m_blk = vec![f32::NEG_INFINITY; br];
            let mut l_blk = vec![0.0f32; br];

            // Inner loop — kv rows in chunks of BLOCK_C
            // Causal: only kv positions ≤ current q position can attend.
            // The latest q position in this tile is (q_end - 1), so we only
            // need kv tiles up to that position.
            let mut kv_start = 0;
            while kv_start < q_end {
                let kv_end = (kv_start + BLOCK_C).min(t).min(q_end);
                let bc = kv_end - kv_start;

                // Compute score tile: s[qi, ki] = scale * Q[q_start+qi] · K[kv_start+ki]
                // Shape: [br, bc]
                let mut s = vec![0.0f32; br * bc];
                for qi in 0..br {
                    let global_qi = q_start + qi;
                    for ki in 0..bc {
                        let global_ki = kv_start + ki;
                        // Causal mask: future keys get -inf
                        if global_ki > global_qi {
                            s[qi * bc + ki] = f32::NEG_INFINITY;
                            continue;
                        }
                        let mut dot = 0.0f32;
                        for d in 0..d_head {
                            dot += q_data.at(global_qi, d) * k_data.at(global_ki, d);
                        }
                        s[qi * bc + ki] = dot * scale;
                    }
                }

                // Online softmax update per query row
                for qi in 0..br {
                    // Row max over this tile
                    let tile_max = (0..bc)
                        .map(|ki| s[qi * bc + ki])
                        .fold(f32::NEG_INFINITY, f32::max);

                    let m_new = m_blk[qi].max(tile_max);

                    // Rescale existing accumulator by exp(m_old - m_new)
                    let rescale = (m_blk[qi] - m_new).exp();
                    for d in 0..d_head {
                        acc[qi * d_head + d] *= rescale;
                    }
                    l_blk[qi] *= rescale;

                    // Accumulate this tile: p[ki] = exp(s[qi,ki] - m_new)
                    for ki in 0..bc {
                        let p = (s[qi * bc + ki] - m_new).exp();
                        l_blk[qi] += p;
                        let global_ki = kv_start + ki;
                        for d in 0..d_head {
                            acc[qi * d_head + d] += p * v_data.at(global_ki, d);
                        }
                    }
                    m_blk[qi] = m_new;
                }

                kv_start += BLOCK_C;
            }

            // Write normalised output for this query tile
            for qi in 0..br {
                let global_qi = q_start + qi;
                let inv_l = 1.0 / l_blk[qi];
                for d in 0..d_head {
                    *out_data.at_mut(global_qi, d) = acc[qi * d_head + d] * inv_l;
                }
                l_global[global_qi] = l_blk[qi];
                m_global[global_qi] = m_blk[qi];
            }

            q_start += BLOCK_R;
        }

        let out = TensorNode::leaf(out_data);
        let q_c = q.clone();
        let k_c = k.clone();
        let v_c = v.clone();
        let out_c = out.clone();

        // Backward: recompute softmax weights from stored (l, m) and propagate
        // gradients.  Memory: O(T) — no T×T matrix stored.
        out.0.borrow_mut().backward_fn = Some(Box::new(move || {
            let dout = out_c.0.borrow().grad.clone(); // [T, d_head]
            let q_d = q_c.0.borrow().data.clone();
            let k_d = k_c.0.borrow().data.clone();
            let v_d = v_c.0.borrow().data.clone();
            let out_d = out_c.0.borrow().data.clone(); // [T, d_head] — final output

            let mut dq = Mat::zeros(t, d_head);
            let mut dk = Mat::zeros(t, d_head);
            let mut dv = Mat::zeros(t, d_head);

            // For each query tile, recompute the softmax weights and propagate.
            let mut q_start = 0;
            while q_start < t {
                let q_end = (q_start + BLOCK_R).min(t);
                let br = q_end - q_start;

                let mut kv_start = 0;
                while kv_start < q_end {
                    let kv_end = (kv_start + BLOCK_C).min(t).min(q_end);
                    let bc = kv_end - kv_start;

                    // Recompute score tile and softmax weights p[qi, ki]
                    let mut p = vec![0.0f32; br * bc];
                    for qi in 0..br {
                        let global_qi = q_start + qi;
                        let m_i = m_global[global_qi];
                        for ki in 0..bc {
                            let global_ki = kv_start + ki;
                            if global_ki > global_qi {
                                p[qi * bc + ki] = 0.0;
                                continue;
                            }
                            let mut dot = 0.0f32;
                            for d in 0..d_head {
                                dot += q_d.at(global_qi, d) * k_d.at(global_ki, d);
                            }
                            p[qi * bc + ki] = (dot * scale - m_i).exp() / l_global[global_qi];
                        }
                    }

                    // dV += P.T @ dOut_tile    [bc, d_head]
                    for ki in 0..bc {
                        let global_ki = kv_start + ki;
                        for d in 0..d_head {
                            let mut sum = 0.0f32;
                            for qi in 0..br {
                                sum += p[qi * bc + ki] * dout.at(q_start + qi, d);
                            }
                            *dv.at_mut(global_ki, d) += sum;
                        }
                    }

                    // dP = dOut_tile @ V.T    [br, bc]
                    let mut dp = vec![0.0f32; br * bc];
                    for qi in 0..br {
                        for ki in 0..bc {
                            let global_ki = kv_start + ki;
                            let mut sum = 0.0f32;
                            for d in 0..d_head {
                                sum += dout.at(q_start + qi, d) * v_d.at(global_ki, d);
                            }
                            dp[qi * bc + ki] = sum;
                        }
                    }

                    // Softmax backward: dS[qi,ki] = P[qi,ki] * (dP[qi,ki] - Di)
                    // where Di = sum_j(P[qi,j] * dP[qi,j])  (dot product of row)
                    // = sum_j(P[qi,j] * (dOut[qi] · V[j]))
                    // = dOut[qi] · out[qi]  (since out = sum_j P*V)
                    let mut ds = vec![0.0f32; br * bc];
                    for qi in 0..br {
                        let global_qi = q_start + qi;
                        // Di = dOut[qi] · out[qi]
                        let di: f32 = (0..d_head)
                            .map(|d| dout.at(global_qi, d) * out_d.at(global_qi, d))
                            .sum();
                        for ki in 0..bc {
                            if kv_start + ki > global_qi {
                                continue;
                            }
                            ds[qi * bc + ki] = p[qi * bc + ki] * (dp[qi * bc + ki] - di) * scale;
                        }
                    }

                    // dQ += dS @ K_tile    [br, d_head]
                    for qi in 0..br {
                        let global_qi = q_start + qi;
                        for d in 0..d_head {
                            let mut sum = 0.0f32;
                            for ki in 0..bc {
                                sum += ds[qi * bc + ki] * k_d.at(kv_start + ki, d);
                            }
                            *dq.at_mut(global_qi, d) += sum;
                        }
                    }

                    // dK += dS.T @ Q_tile  [bc, d_head]
                    for ki in 0..bc {
                        let global_ki = kv_start + ki;
                        for d in 0..d_head {
                            let mut sum = 0.0f32;
                            for qi in 0..br {
                                sum += ds[qi * bc + ki] * q_d.at(q_start + qi, d);
                            }
                            *dk.at_mut(global_ki, d) += sum;
                        }
                    }

                    kv_start += BLOCK_C;
                }

                q_start += BLOCK_R;
            }

            q_c.0.borrow_mut().grad.add_assign(&dq);
            k_c.0.borrow_mut().grad.add_assign(&dk);
            v_c.0.borrow_mut().grad.add_assign(&dv);
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

        fn build(
            v: &TensorNode,
            topo: &mut Vec<TensorNode>,
            visited: &mut HashSet<*const RefCell<NodeData>>,
        ) {
            let ptr = Rc::as_ptr(&v.0);
            if visited.contains(&ptr) {
                return;
            }
            visited.insert(ptr);
            let prev = v.0.borrow().prev.clone();
            for p in &prev {
                build(p, topo, visited);
            }
            topo.push(v.clone());
        }
        build(self, &mut topo, &mut visited);

        // Seed: gradient of loss w.r.t. itself = 1
        {
            let mut inner = self.0.borrow_mut();
            assert!(
                inner.data.numel() == 1,
                "backward() must be called on a scalar (1×1 matrix), got shape [{},{}]",
                inner.data.rows,
                inner.data.cols
            );
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
            if let Some(f) = fn_ptr {
                f();
            }
        }
    }
}

impl std::fmt::Debug for TensorNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.0.borrow();
        write!(
            f,
            "TensorNode(shape=[{},{}], grad_norm={:.4})",
            inner.data.rows,
            inner.data.cols,
            inner.grad.norm()
        )
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

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
        let a = Mat::new(vec![1., 2., 3., 4.], 2, 2);
        let b = Mat::new(vec![5., 6.], 2, 1);
        let c = a.matmul(&b);
        assert!(approx(c.at(0, 0), 17.0));
        assert!(approx(c.at(1, 0), 39.0));
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
        for i in 0..m {
            for p in 0..k {
                for j in 0..n {
                    *expected.at_mut(i, j) += a.at(i, p) * b.at(p, j);
                }
            }
        }
        for r in 0..m {
            for c in 0..n {
                assert!(
                    (result.at(r, c) - expected.at(r, c)).abs() < 1e-4,
                    "matmul[{},{}]: got {} expected {}",
                    r,
                    c,
                    result.at(r, c),
                    expected.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_transpose() {
        let a = Mat::new(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let at = a.transpose();
        assert_eq!((at.rows, at.cols), (3, 2));
        assert!(approx(at.at(0, 0), 1.0));
        assert!(approx(at.at(1, 0), 2.0));
        assert!(approx(at.at(0, 1), 4.0));
    }

    // Helper: call a node's backward_fn without holding a RefCell borrow.
    fn call_backward(node: &TensorNode) {
        let fn_ptr = {
            let inner = node.0.borrow();
            inner
                .backward_fn
                .as_ref()
                .map(|f| unsafe { &*(f.as_ref() as *const dyn Fn()) })
        };
        if let Some(f) = fn_ptr {
            f();
        }
    }

    // --- TensorNode backward rules (verified numerically) ---

    #[test]
    fn test_matmul_grad_a() {
        // Loss = sum(A @ B). Check dA numerically.
        let a_data = Mat::new(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let b_data = Mat::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3, 2);

        let num = numerical_grad(
            &|a| {
                let a_n = TensorNode::leaf(a.clone());
                let b_n = TensorNode::leaf(b_data.clone());
                let c = a_n.matmul(&b_n);
                c.data().data.iter().sum::<f32>()
            },
            &a_data,
        );

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.matmul(&b);
        // Read shape before borrowing mutably
        let (cr, cc) = {
            let d = c.data();
            (d.rows, d.cols)
        };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        {
            let inner = c.0.borrow();
            let fn_ptr = inner
                .backward_fn
                .as_ref()
                .map(|f| unsafe { &*(f.as_ref() as *const dyn Fn()) });
            drop(inner);
            if let Some(f) = fn_ptr {
                f();
            }
        }

        let ag = a.grad().clone();
        for r in 0..ag.rows {
            for col in 0..ag.cols {
                assert!(
                    approx(ag.at(r, col), num.at(r, col)),
                    "dA[{},{}]: analytical={:.4} numerical={:.4}",
                    r,
                    col,
                    ag.at(r, col),
                    num.at(r, col)
                );
            }
        }
    }

    #[test]
    fn test_matmul_grad_b() {
        let a_data = Mat::new(vec![1., 2., 3., 4., 5., 6.], 2, 3);
        let b_data = Mat::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], 3, 2);

        let num = numerical_grad(
            &|b| {
                let a_n = TensorNode::leaf(a_data.clone());
                let b_n = TensorNode::leaf(b.clone());
                let c = a_n.matmul(&b_n);
                c.data().data.iter().sum::<f32>()
            },
            &b_data,
        );

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.matmul(&b);
        let (cr, cc) = {
            let d = c.data();
            (d.rows, d.cols)
        };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        call_backward(&c);

        let bg = b.grad().clone();
        for r in 0..bg.rows {
            for col in 0..bg.cols {
                assert!(
                    approx(bg.at(r, col), num.at(r, col)),
                    "dB[{},{}]: analytical={:.4} numerical={:.4}",
                    r,
                    col,
                    bg.at(r, col),
                    num.at(r, col)
                );
            }
        }
    }

    #[test]
    fn test_add_bias_grad() {
        let a_data = Mat::new(vec![1., 2., 3., 4., 5., 6.], 3, 2);
        let b_data = Mat::new(vec![0.5, -0.5], 1, 2);

        let num_b = numerical_grad(
            &|b| {
                let a_n = TensorNode::leaf(a_data.clone());
                let b_n = TensorNode::leaf(b.clone());
                let c = a_n.add_bias(&b_n);
                c.data().data.iter().sum::<f32>()
            },
            &b_data,
        );

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let c = a.add_bias(&b);
        let (cr, cc) = {
            let d = c.data();
            (d.rows, d.cols)
        };
        c.0.borrow_mut().grad = Mat::ones(cr, cc);
        call_backward(&c);

        let bg = b.grad().clone();
        for col in 0..2 {
            assert!(
                approx(bg.at(0, col), num_b.at(0, col)),
                "d_bias[{}]: analytical={:.4} numerical={:.4}",
                col,
                bg.at(0, col),
                num_b.at(0, col)
            );
        }
    }

    #[test]
    fn test_gelu_grad() {
        let x_data = Mat::new(vec![-1.0, 0.0, 0.5, 2.0], 1, 4);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                let g = xn.gelu();
                g.data().data.iter().sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let g = x.gelu();
        g.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&g);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(
                approx(xg.at(0, c), num.at(0, c)),
                "GELU grad[{}]: analytical={:.4} numerical={:.4}",
                c,
                xg.at(0, c),
                num.at(0, c)
            );
        }
    }

    #[test]
    fn test_softmax_probabilities() {
        let x = TensorNode::leaf(Mat::new(vec![1., 2., 3., 4., 5., 6.], 2, 3));
        let s = x.softmax();
        // Each row must sum to 1
        for r in 0..2 {
            let row_sum: f32 = (0..3).map(|c| s.data().at(r, c)).sum();
            assert!(approx(row_sum, 1.0), "row {} sum = {}", r, row_sum);
        }
    }

    #[test]
    fn test_softmax_grad() {
        let x_data = Mat::new(vec![1.0, 2.0, 0.5], 1, 3);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                let s = xn.softmax();
                // Loss = sum(s * weights) with fixed weights to get non-trivial grad
                s.data()
                    .data
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v * (i + 1) as f32)
                    .sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let s = x.softmax();
        // Seed: dLoss/dS[i] = i+1
        let ds = Mat::new(vec![1.0, 2.0, 3.0], 1, 3);
        s.0.borrow_mut().grad = ds;
        call_backward(&s);

        let xg = x.grad().clone();
        for c in 0..3 {
            assert!(
                approx(xg.at(0, c), num.at(0, c)),
                "softmax grad[{}]: analytical={:.4} numerical={:.4}",
                c,
                xg.at(0, c),
                num.at(0, c)
            );
        }
    }

    #[test]
    fn test_layer_norm_output_mean_zero() {
        let x = TensorNode::leaf(Mat::new(vec![1., 2., 3., 4.], 1, 4));
        let gamma = TensorNode::leaf(Mat::ones(1, 4));
        let beta = TensorNode::leaf(Mat::zeros(1, 4));
        let out = x.layer_norm(&gamma, &beta);
        let mean = out.data().data.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-5,
            "LN output mean should be 0, got {}",
            mean
        );
    }

    #[test]
    fn test_layer_norm_grad_x() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);
        let b_data = Mat::zeros(1, 4);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                let gn = TensorNode::leaf(g_data.clone());
                let bn = TensorNode::leaf(b_data.clone());
                let out = xn.layer_norm(&gn, &bn);
                out.data().data.iter().sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let b = TensorNode::leaf(b_data);
        let out = x.layer_norm(&g, &b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(
                approx(xg.at(0, c), num.at(0, c)),
                "LN dX[{}]: analytical={:.4} numerical={:.4}",
                c,
                xg.at(0, c),
                num.at(0, c)
            );
        }
    }

    #[test]
    fn test_layer_norm_grad_gamma() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);
        let b_data = Mat::zeros(1, 4);

        let num = numerical_grad(
            &|g| {
                let xn = TensorNode::leaf(x_data.clone());
                let gn = TensorNode::leaf(g.clone());
                let bn = TensorNode::leaf(b_data.clone());
                let out = xn.layer_norm(&gn, &bn);
                out.data().data.iter().sum::<f32>()
            },
            &g_data,
        );

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let b = TensorNode::leaf(b_data);
        let out = x.layer_norm(&g, &b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let gg = g.grad().clone();
        for c in 0..4 {
            assert!(
                approx(gg.at(0, c), num.at(0, c)),
                "LN d_gamma[{}]: analytical={:.4} numerical={:.4}",
                c,
                gg.at(0, c),
                num.at(0, c)
            );
        }
    }

    #[test]
    fn test_causal_attention_output_shape() {
        let t = 4;
        let d = 8;
        let q = TensorNode::leaf(Mat::zeros(t, d));
        let k = TensorNode::leaf(Mat::zeros(t, d));
        let v = TensorNode::leaf(Mat::zeros(t, d));
        let out = TensorNode::causal_attention(&q, &k, &v, d);
        assert_eq!((out.data().rows, out.data().cols), (t, d));
    }

    #[test]
    fn test_causal_attention_grad_v() {
        let t = 3;
        let d = 4;
        let q_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1);
        let k_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05);
        let v_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07);

        let num = numerical_grad(
            &|v| {
                let qn = TensorNode::leaf(q_data.clone());
                let kn = TensorNode::leaf(k_data.clone());
                let vn = TensorNode::leaf(v.clone());
                let out = TensorNode::causal_attention(&qn, &kn, &vn, d);
                out.data().data.iter().sum::<f32>()
            },
            &v_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::causal_attention(&q, &k, &v, d);
        out.0.borrow_mut().grad = Mat::ones(t, d);
        call_backward(&out);

        let vg = v.grad().clone();
        for r in 0..t {
            for c in 0..d {
                assert!(
                    approx(vg.at(r, c), num.at(r, c)),
                    "attn dV[{},{}]: analytical={:.4} numerical={:.4}",
                    r,
                    c,
                    vg.at(r, c),
                    num.at(r, c)
                );
            }
        }
    }

    // --- Flash Attention ---

    #[test]
    fn test_flash_attention_output_shape() {
        let t = 5;
        let d = 8;
        let q = TensorNode::leaf(Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1));
        let k = TensorNode::leaf(Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05));
        let v = TensorNode::leaf(Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07));
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        assert_eq!((out.data().rows, out.data().cols), (t, d));
    }

    #[test]
    fn test_flash_attention_matches_causal_attention() {
        // Flash and standard attention must produce identical outputs.
        let t = 8;
        let d = 16;
        let q_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1 - 0.5);
        let k_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05 + 0.1);
        let v_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07 - 0.2);

        let q1 = TensorNode::leaf(q_data.clone());
        let k1 = TensorNode::leaf(k_data.clone());
        let v1 = TensorNode::leaf(v_data.clone());
        let ref_out = TensorNode::causal_attention(&q1, &k1, &v1, d);

        let q2 = TensorNode::leaf(q_data);
        let k2 = TensorNode::leaf(k_data);
        let v2 = TensorNode::leaf(v_data);
        let flash_out = TensorNode::flash_attention(&q2, &k2, &v2, d);

        let r = ref_out.data().clone();
        let f = flash_out.data().clone();
        for row in 0..t {
            for col in 0..d {
                let diff = (r.at(row, col) - f.at(row, col)).abs();
                assert!(
                    diff < 1e-4,
                    "flash vs causal [{row},{col}]: flash={:.5} ref={:.5}",
                    f.at(row, col),
                    r.at(row, col)
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_causal_first_token() {
        // Token 0 can only attend to itself — output must equal v[0] exactly.
        let t = 4;
        let d = 4;
        let q = TensorNode::leaf(Mat::from_fn(t, d, |_, _| 1.0));
        let k = TensorNode::leaf(Mat::from_fn(t, d, |_, _| 1.0));
        let v = TensorNode::leaf(Mat::from_fn(t, d, |r, c| (r * d + c) as f32));
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        // Row 0: softmax over only position 0 → weight=1 → output = v[0]
        for c in 0..d {
            assert!(
                (out.data().at(0, c) - v.data().at(0, c)).abs() < 1e-4,
                "flash first token col {c}: got {} expected {}",
                out.data().at(0, c),
                v.data().at(0, c)
            );
        }
    }

    #[test]
    fn test_flash_attention_output_finite() {
        let t = 16;
        let d = 32;
        let q = TensorNode::leaf(Mat::from_fn(t, d, |r, c| ((r + c) as f32) * 0.01));
        let k = TensorNode::leaf(Mat::from_fn(t, d, |r, c| ((r * d + c) as f32) * 0.01 - 0.5));
        let v = TensorNode::leaf(Mat::from_fn(t, d, |r, c| ((r + c) as f32) * 0.02));
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        assert!(
            out.data().data.iter().all(|x| x.is_finite()),
            "flash output has NaN/Inf"
        );
    }

    #[test]
    fn test_flash_attention_grad_v() {
        // Numerical gradient check for dV
        let t = 4;
        let d = 4;
        let q_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1);
        let k_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05);
        let v_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07);

        let num = numerical_grad(
            &|v| {
                let qn = TensorNode::leaf(q_data.clone());
                let kn = TensorNode::leaf(k_data.clone());
                let vn = TensorNode::leaf(v.clone());
                TensorNode::flash_attention(&qn, &kn, &vn, d)
                    .data()
                    .data
                    .iter()
                    .sum::<f32>()
            },
            &v_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        out.0.borrow_mut().grad = Mat::ones(t, d);
        call_backward(&out);

        let vg = v.grad().clone();
        for r in 0..t {
            for c in 0..d {
                assert!(
                    approx(vg.at(r, c), num.at(r, c)),
                    "flash dV[{r},{c}]: analytical={:.4} numerical={:.4}",
                    vg.at(r, c),
                    num.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_grad_q() {
        let t = 4;
        let d = 4;
        let q_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1 + 0.1);
        let k_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05);
        let v_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07);

        let num = numerical_grad(
            &|q| {
                let qn = TensorNode::leaf(q.clone());
                let kn = TensorNode::leaf(k_data.clone());
                let vn = TensorNode::leaf(v_data.clone());
                TensorNode::flash_attention(&qn, &kn, &vn, d)
                    .data()
                    .data
                    .iter()
                    .sum::<f32>()
            },
            &q_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        out.0.borrow_mut().grad = Mat::ones(t, d);
        call_backward(&out);

        let qg = q.grad().clone();
        for r in 0..t {
            for c in 0..d {
                assert!(
                    approx(qg.at(r, c), num.at(r, c)),
                    "flash dQ[{r},{c}]: analytical={:.4} numerical={:.4}",
                    qg.at(r, c),
                    num.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_grad_k() {
        let t = 4;
        let d = 4;
        let q_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.1 + 0.1);
        let k_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.05 + 0.05);
        let v_data = Mat::from_fn(t, d, |r, c| (r * d + c) as f32 * 0.07);

        let num = numerical_grad(
            &|k| {
                let qn = TensorNode::leaf(q_data.clone());
                let kn = TensorNode::leaf(k.clone());
                let vn = TensorNode::leaf(v_data.clone());
                TensorNode::flash_attention(&qn, &kn, &vn, d)
                    .data()
                    .data
                    .iter()
                    .sum::<f32>()
            },
            &k_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::flash_attention(&q, &k, &v, d);
        out.0.borrow_mut().grad = Mat::ones(t, d);
        call_backward(&out);

        let kg = k.grad().clone();
        for r in 0..t {
            for c in 0..d {
                assert!(
                    approx(kg.at(r, c), num.at(r, c)),
                    "flash dK[{r},{c}]: analytical={:.4} numerical={:.4}",
                    kg.at(r, c),
                    num.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_flash_attention_large_t_matches_causal() {
        // T=128 (> BLOCK_SIZE=64) — tests multi-tile correctness
        let t = 128;
        let d = 16;
        let q_data = Mat::from_fn(t, d, |r, c| ((r * d + c) as f32 * 0.01).sin());
        let k_data = Mat::from_fn(t, d, |r, c| ((r * d + c) as f32 * 0.01).cos());
        let v_data = Mat::from_fn(t, d, |r, c| (r as f32 * 0.1 - c as f32 * 0.05));

        let q1 = TensorNode::leaf(q_data.clone());
        let k1 = TensorNode::leaf(k_data.clone());
        let v1 = TensorNode::leaf(v_data.clone());
        let ref_out = TensorNode::causal_attention(&q1, &k1, &v1, d);

        let q2 = TensorNode::leaf(q_data);
        let k2 = TensorNode::leaf(k_data);
        let v2 = TensorNode::leaf(v_data);
        let flash_out = TensorNode::flash_attention(&q2, &k2, &v2, d);

        let r = ref_out.data().clone();
        let f = flash_out.data().clone();
        for row in 0..t {
            for col in 0..d {
                let diff = (r.at(row, col) - f.at(row, col)).abs();
                assert!(
                    diff < 1e-3,
                    "large T flash vs causal [{row},{col}]: flash={:.5} ref={:.5}",
                    f.at(row, col),
                    r.at(row, col)
                );
            }
        }
    }

    // --- RMSNorm backward ---

    #[test]
    fn test_rms_norm_grad_x() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                let gn = TensorNode::leaf(g_data.clone());
                let out = xn.rms_norm(&gn, 1e-5);
                out.data().data.iter().sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let out = x.rms_norm(&g, 1e-5);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(
                approx(xg.at(0, c), num.at(0, c)),
                "RMSNorm dX[{}]: analytical={:.4} numerical={:.4}",
                c,
                xg.at(0, c),
                num.at(0, c)
            );
        }
    }

    #[test]
    fn test_rms_norm_grad_gamma() {
        let x_data = Mat::new(vec![0.5, -0.3, 1.2, -0.8], 1, 4);
        let g_data = Mat::new(vec![1.0, 0.8, 1.2, 0.9], 1, 4);

        let num = numerical_grad(
            &|g| {
                let xn = TensorNode::leaf(x_data.clone());
                let gn = TensorNode::leaf(g.clone());
                let out = xn.rms_norm(&gn, 1e-5);
                out.data().data.iter().sum::<f32>()
            },
            &g_data,
        );

        let x = TensorNode::leaf(x_data);
        let g = TensorNode::leaf(g_data);
        let out = x.rms_norm(&g, 1e-5);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let gg = g.grad().clone();
        for c in 0..4 {
            assert!(
                approx(gg.at(0, c), num.at(0, c)),
                "RMSNorm dGamma[{}]: analytical={:.4} numerical={:.4}",
                c,
                gg.at(0, c),
                num.at(0, c)
            );
        }
    }

    // --- SiLU backward ---

    #[test]
    fn test_silu_grad() {
        let x_data = Mat::new(vec![-1.0, 0.0, 0.5, 2.0], 1, 4);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                xn.silu().data().data.iter().sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let out = x.silu();
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let xg = x.grad().clone();
        for c in 0..4 {
            assert!(
                approx(xg.at(0, c), num.at(0, c)),
                "SiLU grad[{}]: analytical={:.4} numerical={:.4}",
                c,
                xg.at(0, c),
                num.at(0, c)
            );
        }
    }

    // --- mul_elem_node backward ---

    #[test]
    fn test_mul_elem_node_grad() {
        let a_data = Mat::new(vec![1.0, 2.0, 0.5, -1.0], 1, 4);
        let b_data = Mat::new(vec![0.3, -0.5, 1.2, 0.8], 1, 4);

        let num_a = numerical_grad(
            &|a| {
                let an = TensorNode::leaf(a.clone());
                let bn = TensorNode::leaf(b_data.clone());
                an.mul_elem_node(&bn).data().data.iter().sum::<f32>()
            },
            &a_data,
        );

        let a = TensorNode::leaf(a_data);
        let b = TensorNode::leaf(b_data);
        let out = a.mul_elem_node(&b);
        out.0.borrow_mut().grad = Mat::ones(1, 4);
        call_backward(&out);

        let ag = a.grad().clone();
        for c in 0..4 {
            assert!(
                approx(ag.at(0, c), num_a.at(0, c)),
                "mul_elem dA[{}]: analytical={:.4} numerical={:.4}",
                c,
                ag.at(0, c),
                num_a.at(0, c)
            );
        }
    }

    // --- RoPE backward ---

    #[test]
    fn test_rope_grad() {
        // Use slightly looser tolerance for RoPE: trig functions accumulate more
        // floating-point error in the central-difference approximation.
        let tol = 5e-3f32;
        let x_data = Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.1 + 0.1);

        let num = numerical_grad(
            &|x| {
                let xn = TensorNode::leaf(x.clone());
                xn.rope_apply(0, 10000.0).data().data.iter().sum::<f32>()
            },
            &x_data,
        );

        let x = TensorNode::leaf(x_data);
        let out = x.rope_apply(0, 10000.0);
        let (r, c) = {
            let d = out.data();
            (d.rows, d.cols)
        };
        out.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out);

        let xg = x.grad().clone();
        for row in 0..3 {
            for col in 0..8 {
                assert!(
                    (xg.at(row, col) - num.at(row, col)).abs() < tol,
                    "RoPE dX[{},{}]: analytical={:.4} numerical={:.4}",
                    row,
                    col,
                    xg.at(row, col),
                    num.at(row, col)
                );
            }
        }
    }

    // --- GQA backward ---

    #[test]
    fn test_gqa_grad_q() {
        let t = 3;
        let n_q = 4;
        let n_kv = 2;
        let dh = 4;
        let q_data = Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.05 + 0.1);
        let k_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.03 + 0.05
        });
        let v_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.04 + 0.02
        });

        let num = numerical_grad(
            &|q| {
                let qn = TensorNode::leaf(q.clone());
                let kn = TensorNode::leaf(k_data.clone());
                let vn = TensorNode::leaf(v_data.clone());
                TensorNode::gqa_attention(&qn, &kn, &vn, n_q, n_kv, dh)
                    .data()
                    .data
                    .iter()
                    .sum::<f32>()
            },
            &q_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let (r, c) = {
            let d = out.data();
            (d.rows, d.cols)
        };
        out.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out);

        let qg = q.grad().clone();
        for row in 0..t {
            for col in 0..(n_q * dh) {
                assert!(
                    approx(qg.at(row, col), num.at(row, col)),
                    "GQA dQ[{},{}]: analytical={:.4} numerical={:.4}",
                    row,
                    col,
                    qg.at(row, col),
                    num.at(row, col)
                );
            }
        }
    }

    #[test]
    fn test_full_backward_via_backward_method() {
        // Build a small compute graph and call .backward() end-to-end.
        // We add a sum-to-scalar node so backward() can be called directly.
        let a = TensorNode::leaf(Mat::new(vec![1., 2., 3., 4.], 2, 2));
        let b = TensorNode::leaf(Mat::new(vec![0.5, 0.5, 0.5, 0.5], 2, 2));
        let c = a.matmul(&b); // [2,2]
        let g = c.gelu(); // [2,2]

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
        assert!(
            ag.data.iter().all(|x| x.is_finite()),
            "gradient should be finite"
        );
        assert!(
            ag.data.iter().any(|x| x.abs() > 1e-6),
            "gradient should be non-zero"
        );
    }

    // --- Parallel matmul ---

    #[test]
    fn test_matmul_parallel_matches_sequential() {
        let m = 32;
        let k = 64;
        let n = 48;
        let a = Mat::from_fn(m, k, |r, c| (r * k + c) as f32 * 0.01 - 0.5);
        let b = Mat::from_fn(k, n, |r, c| (r * n + c) as f32 * 0.02 - 0.3);

        // Compare parallel vs single-thread directly (not via matmul() dispatch,
        // which may route to Metal GPU with different fp rounding).
        let seq = a.matmul_parallel(&b, 1);
        let par = a.matmul_parallel(&b, 4);

        assert_eq!((par.rows, par.cols), (seq.rows, seq.cols));
        for r in 0..m {
            for c in 0..n {
                assert!(
                    (par.at(r, c) - seq.at(r, c)).abs() < 1e-4,
                    "par[{},{}]={} seq[{},{}]={}",
                    r,
                    c,
                    par.at(r, c),
                    r,
                    c,
                    seq.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_matmul_parallel_single_thread() {
        let a = Mat::from_fn(3, 4, |r, c| (r + c) as f32);
        let b = Mat::from_fn(4, 2, |r, c| (r * 2 + c) as f32);
        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b, 1);
        for r in 0..3 {
            for c in 0..2 {
                assert!((par.at(r, c) - seq.at(r, c)).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_matmul_parallel_auto_threads() {
        let a = Mat::from_fn(16, 8, |r, c| (r * 8 + c) as f32 * 0.1);
        let b = Mat::from_fn(8, 16, |r, c| (r * 16 + c) as f32 * 0.1);
        let seq = a.matmul(&b);
        let par = a.matmul_parallel(&b, 0); // 0 = auto-detect thread count
        for r in 0..16 {
            for c in 0..16 {
                assert!(
                    (par.at(r, c) - seq.at(r, c)).abs() < 1e-3,
                    "auto-thread: par[{},{}]={:.4} seq[{},{}]={:.4}",
                    r,
                    c,
                    par.at(r, c),
                    r,
                    c,
                    seq.at(r, c)
                );
            }
        }
    }

    // --- Q4 quantization ---

    #[test]
    fn test_q4_quantize_dequantize_roundtrip() {
        let m = Mat::from_fn(4, 8, |r, c| ((r * 8 + c) as f32 / 31.0) * 2.0 - 1.0);
        let q = Q4Mat::quantize(&m);
        let m2 = q.dequantize();
        assert_eq!((m2.rows, m2.cols), (4, 8));
        for r in 0..4 {
            for c in 0..8 {
                assert!(
                    (m2.at(r, c) - m.at(r, c)).abs() < 0.15,
                    "q4 round-trip error at [{},{}]: {} vs {}",
                    r,
                    c,
                    m2.at(r, c),
                    m.at(r, c)
                );
            }
        }
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
        assert!(
            m2.data.iter().all(|&v| v == 0.0),
            "zeros should stay zero after Q4"
        );
    }

    #[test]
    fn test_q4_matmul_matches_dequant() {
        let a = Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.05 + 0.1);
        let w = Mat::from_fn(4, 8, |r, c| (r * 8 + c) as f32 * 0.03 - 0.2);
        let q = Q4Mat::quantize(&w);

        let w_approx = q.dequantize();
        let ref_out = a.matmul(&w_approx.transpose());
        let fused_out = q.matmul_q4_t(&a);

        assert_eq!(
            (fused_out.rows, fused_out.cols),
            (ref_out.rows, ref_out.cols)
        );
        for r in 0..ref_out.rows {
            for c in 0..ref_out.cols {
                assert!(
                    (fused_out.at(r, c) - ref_out.at(r, c)).abs() < 1e-4,
                    "q4 matmul [{},{}]: fused={:.5} ref={:.5}",
                    r,
                    c,
                    fused_out.at(r, c),
                    ref_out.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_q4_non_multiple_block_size() {
        let m = Mat::from_fn(1, 10, |_, c| c as f32 * 0.1 - 0.5);
        let q = Q4Mat::quantize(&m);
        let m2 = q.dequantize();
        for c in 0..10 {
            assert!(
                (m2.at(0, c) - m.at(0, c)).abs() < 0.15,
                "small Q4: error at col {}: {} vs {}",
                c,
                m2.at(0, c),
                m.at(0, c)
            );
        }
    }

    // -------------------------------------------------------------------------
    // Q4K quantize round-trip tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_q4k_quantize_dequantize_roundtrip() {
        // Create a matrix with varied values; cols must be multiple of 256
        let m = Mat::from_fn(4, 256, |r, c| {
            ((r * 256 + c) as f32 / 1024.0) * 2.0 - 1.0
        });
        let q = Q4KMat::quantize(&m);

        // Verify block count
        let expected_blocks = 4; // 4 rows * (256/256) blocks per row
        assert_eq!(q.blocks.len(), expected_blocks * 144);
        assert_eq!(q.rows, 4);
        assert_eq!(q.cols, 256);

        // Dequantize and check round-trip error
        let mut buf = vec![0.0f32; 256];
        let mut max_err = 0.0f32;
        for r in 0..4 {
            q.dequantize_row_into(r, &mut buf);
            for c in 0..256 {
                let err = (buf[c] - m.at(r, c)).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 0.15,
            "Q4K round-trip max error {} should be < 0.15",
            max_err
        );
    }

    #[test]
    fn test_q4k_quantize_zeros() {
        let m = Mat::zeros(2, 256);
        let q = Q4KMat::quantize(&m);
        let mut buf = vec![0.0f32; 256];
        for r in 0..2 {
            q.dequantize_row_into(r, &mut buf);
            for c in 0..256 {
                assert!(
                    buf[c].abs() < 1e-6,
                    "Q4K zeros: row {} col {} got {}",
                    r, c, buf[c]
                );
            }
        }
    }

    #[test]
    fn test_q4k_quantize_large_matrix() {
        // Simulate lm_head-like dimensions (smaller scale)
        let m = Mat::from_fn(16, 512, |r, c| {
            ((r * 512 + c) as f32 * 0.001).sin()
        });
        let q = Q4KMat::quantize(&m);
        assert_eq!(q.rows, 16);
        assert_eq!(q.cols, 512);

        let mut buf = vec![0.0f32; 512];
        let mut max_err = 0.0f32;
        for r in 0..16 {
            q.dequantize_row_into(r, &mut buf);
            for c in 0..512 {
                let err = (buf[c] - m.at(r, c)).abs();
                max_err = max_err.max(err);
            }
        }
        assert!(
            max_err < 0.15,
            "Q4K large matrix round-trip max error {} should be < 0.15",
            max_err
        );
    }

    #[test]
    fn test_f16_roundtrip() {
        // Test the f32→f16→f32 round-trip
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.001, 100.0] {
            let bits = Q4KMat::f32_to_f16(v);
            let back = Q4KMat::f16_to_f32(bits);
            let err = (back - v).abs();
            let tol = v.abs() * 0.002 + 1e-6; // ~0.1% relative tolerance
            assert!(
                err < tol,
                "f16 round-trip for {}: got {} (err {})",
                v, back, err
            );
        }
    }

    // -------------------------------------------------------------------------
    // BF16 tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_bf16_roundtrip_values() {
        for v in [0.0f32, 1.0, -1.0, 2.0, 0.5, 16.0, -8.0, 0.125] {
            let bits = MatBf16::f32_to_bf16(v);
            let back = MatBf16::bf16_to_f32(bits);
            assert!(
                (back - v).abs() < 1e-2,
                "bf16 roundtrip: {} → bits={:#06x} → {}",
                v,
                bits,
                back
            );
        }
    }

    #[test]
    fn test_bf16_precision_loss_small() {
        let v = 3.14159f32;
        let bits = MatBf16::f32_to_bf16(v);
        let back = MatBf16::bf16_to_f32(bits);
        assert!((back - v).abs() < 0.01, "bf16 precision: {} → {}", v, back);
    }

    #[test]
    fn test_mat_to_bf16_shape() {
        let m = Mat::from_fn(4, 8, |r, c| (r * 8 + c) as f32 * 0.1);
        let bf = m.to_bf16();
        assert_eq!((bf.rows, bf.cols), (4, 8));
        assert_eq!(bf.data.len(), 32);
    }

    #[test]
    fn test_mat_bf16_roundtrip() {
        let m = Mat::from_fn(3, 5, |r, c| (r as f32 - 1.5) * (c as f32 + 0.5));
        let bf = m.to_bf16();
        let back = bf.to_f32();
        assert_eq!((back.rows, back.cols), (m.rows, m.cols));
        for r in 0..m.rows {
            for c in 0..m.cols {
                assert!(
                    (back.at(r, c) - m.at(r, c)).abs() < 0.05,
                    "bf16 Mat roundtrip [{},{}]: {} → {}",
                    r,
                    c,
                    m.at(r, c),
                    back.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_mat_from_bf16_bytes() {
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let mut bytes = Vec::new();
        for &v in &values {
            let bits = MatBf16::f32_to_bf16(v);
            bytes.extend_from_slice(&bits.to_le_bytes());
        }
        let m = Mat::from_bf16_bytes(&bytes, 2, 2);
        assert_eq!((m.rows, m.cols), (2, 2));
        for (i, &v) in values.iter().enumerate() {
            let r = i / 2;
            let c = i % 2;
            assert!(
                (m.at(r, c) - v).abs() < 0.01,
                "from_bf16_bytes [{},{}]: expected {} got {}",
                r,
                c,
                v,
                m.at(r, c)
            );
        }
    }

    #[test]
    fn test_bf16_compression_ratio() {
        let m = Mat::from_fn(8, 8, |_, _| 1.0f32);
        let bf = m.to_bf16();
        assert_eq!(bf.compression_ratio(), 2.0);
        assert_eq!(bf.size_bytes(), m.numel() * 2);
    }

    #[test]
    fn test_bf16_matmul_by_t_decode_matches_reference() {
        // Decode path: M=1, N large (simulates lm_head with small batch).
        // matmul_by_t must give the same result as to_f32() + matmul — the
        // reference uses bf16_w.to_f32() (not the original f32 weights) so
        // that any BF16 quantisation error is shared and only the matmul
        // algorithm is being tested.
        let n = 128; // vocab-like dimension
        let k = 16; // hidden-like dimension
        let weight_f32 = Mat::from_fn(n, k, |r, c| ((r * k + c) as f32) * 0.01 - 0.3);
        let bf16_w = weight_f32.to_bf16();
        let input = Mat::from_fn(1, k, |_, c| c as f32 * 0.05 - 0.1);

        // Reference: naive path using the same BF16-dequantised weights.
        let w_f32 = bf16_w.to_f32();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&w_f32);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&w_f32.transpose());

        // Fast decode path via matmul_by_t (should be numerically identical).
        let result = bf16_w.matmul_by_t(&input);

        assert_eq!((result.rows, result.cols), (1, n));
        for j in 0..n {
            assert!(
                (result.at(0, j) - reference.at(0, j)).abs() < 1e-4,
                "matmul_by_t decode [{j}]: got {:.6} expected {:.6}",
                result.at(0, j),
                reference.at(0, j),
            );
        }
    }

    #[test]
    fn test_bf16_matmul_by_t_prefill_matches_reference() {
        // Prefill path: M>4.  Output must match the naive to_f32() + sgemm path.
        // Again use bf16_w.to_f32() as the reference so that BF16 rounding is
        // shared and only the matmul dispatch is tested.
        let n = 32;
        let k = 16;
        let m = 8; // multiple query rows → triggers the sgemm branch
        let weight_f32 = Mat::from_fn(n, k, |r, c| ((r + c) as f32) * 0.05 - 0.5);
        let bf16_w = weight_f32.to_bf16();
        let input = Mat::from_fn(m, k, |r, c| (r as f32 - 0.5) * (c as f32 * 0.1 + 0.1));

        let w_f32 = bf16_w.to_f32();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&w_f32);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&w_f32.transpose());

        let result = bf16_w.matmul_by_t(&input);

        assert_eq!((result.rows, result.cols), (m, n));
        for r in 0..m {
            for c in 0..n {
                assert!(
                    (result.at(r, c) - reference.at(r, c)).abs() < 1e-4,
                    "matmul_by_t prefill [{r},{c}]: got {:.6} expected {:.6}",
                    result.at(r, c),
                    reference.at(r, c),
                );
            }
        }
    }

    /// Test that the chunked-SGEMM decode path (triggered when N > chunk size)
    /// gives the same result as the reference full-dequant + sgemm path.
    #[test]
    fn test_bf16_matmul_by_t_decode_large_n_chunked() {
        // N=4096 is large enough to span multiple chunks (chunk ≈ 8MB/(K*4)).
        // For K=32: chunk = 8MB/128 = 65536, so N=4096 fits in one chunk but
        // exercises the while-loop path.  For K=512: chunk = 8MB/2048 = 4096
        // which triggers exactly one chunk of full size.
        // Use K=64 so chunk = 8MB/256 = 32768 > 4096, meaning a single loop
        // iteration covers all N — this validates the strided-ldc write path.
        let n = 4096;
        let k = 64;
        let weight_f32 = Mat::from_fn(n, k, |r, c| ((r * k + c) as f32) * 0.001 - 0.5);
        let bf16_w = weight_f32.to_bf16();
        let input = Mat::from_fn(1, k, |_, c| c as f32 * 0.05 - 0.1);

        // Reference via full dequant (matches what matmul_by_t prefill would do).
        let w_f32 = bf16_w.to_f32();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&w_f32);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&w_f32.transpose());

        let result = bf16_w.matmul_by_t(&input);

        assert_eq!((result.rows, result.cols), (1, n));
        for j in 0..n {
            assert!(
                (result.at(0, j) - reference.at(0, j)).abs() < 5e-3,
                "chunked BF16 decode [{j}]: got {:.6} expected {:.6}",
                result.at(0, j),
                reference.at(0, j),
            );
        }
    }

    /// Test the multi-chunk path: N large enough to require multiple chunks.
    /// With K=2048: chunk = 8MB/(2048*4) = 1024; use N=4096 → 4 chunks.
    #[test]
    fn test_bf16_matmul_by_t_decode_multi_chunk() {
        let k = 2048;
        let n = 4096; // 4 chunks of 1024
        let weight_f32 = Mat::from_fn(n, k, |r, c| {
            let v = (r as f32 * 0.001) + (c as f32 * 0.0001);
            v - 0.5
        });
        let bf16_w = weight_f32.to_bf16();
        let input = Mat::from_fn(1, k, |_, c| (c as f32) * (1.0 / k as f32) - 0.5);

        let w_f32 = bf16_w.to_f32();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&w_f32);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&w_f32.transpose());

        let result = bf16_w.matmul_by_t(&input);

        assert_eq!((result.rows, result.cols), (1, n));
        let mut max_err = 0.0f32;
        for j in 0..n {
            let err = (result.at(0, j) - reference.at(0, j)).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 1e-2,
            "multi-chunk BF16 decode max error {max_err:.2e} (threshold 1e-2)"
        );
    }

    /// Same test for Q4 chunked decode: N > chunk triggers the while-loop path.
    #[test]
    fn test_q4_matmul_decode_large_n_chunked() {
        // K=256 → chunk = 8MB/1024 = 8192; use N=4096 (one chunk).
        let n = 4096;
        let k = 256;
        let weight_f32 = Mat::from_fn(n, k, |r, c| ((r + c) as f32) * 0.005 - 0.5);
        let q4w = Q4Mat::quantize(&weight_f32);
        let input = Mat::from_fn(1, k, |_, c| (c as f32) * 0.01 - 0.5);

        // Reference: dequant the whole Q4 mat and use the standard matmul.
        let dq = q4w.dequantize();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&dq);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&dq.transpose());

        #[cfg(feature = "blas")]
        let result = q4w.matmul_q4_t_blas(&input);
        #[cfg(not(feature = "blas"))]
        let result = q4w.matmul_q4_t(&input);

        assert_eq!((result.rows, result.cols), (1, n));
        let mut max_err = 0.0f32;
        for j in 0..n {
            let err = (result.at(0, j) - reference.at(0, j)).abs();
            if err > max_err {
                max_err = err;
            }
        }
        // Q4 has quantisation error; tolerate up to ~5 % of absmax.
        assert!(
            max_err < 0.05,
            "chunked Q4 decode large-N max error {max_err:.4} (threshold 0.05)"
        );
    }

    /// Multi-chunk Q4 test: K=2048 → chunk=1024; N=4096 → 4 chunks.
    #[test]
    fn test_q4_matmul_decode_multi_chunk() {
        let k = 2048;
        let n = 4096;
        let weight_f32 = Mat::from_fn(n, k, |r, c| {
            ((r * k + c) as f32) * (1.0 / (n * k) as f32) * 2.0 - 1.0
        });
        let q4w = Q4Mat::quantize(&weight_f32);
        let input = Mat::from_fn(1, k, |_, c| (c as f32) / (k as f32) - 0.5);

        let dq = q4w.dequantize();
        #[cfg(feature = "blas")]
        let reference = input.matmul_bt(&dq);
        #[cfg(not(feature = "blas"))]
        let reference = input.matmul(&dq.transpose());

        #[cfg(feature = "blas")]
        let result = q4w.matmul_q4_t_blas(&input);
        #[cfg(not(feature = "blas"))]
        let result = q4w.matmul_q4_t(&input);

        assert_eq!((result.rows, result.cols), (1, n));
        let mut max_err = 0.0f32;
        for j in 0..n {
            let err = (result.at(0, j) - reference.at(0, j)).abs();
            if err > max_err {
                max_err = err;
            }
        }
        assert!(
            max_err < 0.05,
            "multi-chunk Q4 decode max error {max_err:.4} (threshold 0.05)"
        );
    }

    #[test]
    fn test_bf16_special_values() {
        let zero = MatBf16::bf16_to_f32(MatBf16::f32_to_bf16(0.0));
        assert_eq!(zero, 0.0);
        let inf = MatBf16::bf16_to_f32(MatBf16::f32_to_bf16(f32::INFINITY));
        assert!(inf.is_infinite() && inf > 0.0);
        let neg_inf = MatBf16::bf16_to_f32(MatBf16::f32_to_bf16(f32::NEG_INFINITY));
        assert!(neg_inf.is_infinite() && neg_inf < 0.0);
    }

    // -------------------------------------------------------------------------
    // Gradient checkpointing tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_checkpoint_forward_correct() {
        let x = TensorNode::leaf(Mat::from_fn(4, 4, |r, c| (r * 4 + c) as f32 * 0.1));
        let w = TensorNode::leaf(Mat::from_fn(4, 4, |r, c| if r == c { 1.0 } else { 0.0 }));
        let w_c = w.clone();
        let chk = Checkpoint::new(move |x: &TensorNode| x.matmul(&w_c));
        let out_chk = chk.forward(&x);
        let out_ref = x.matmul(&w);
        let od = out_chk.data();
        let rd = out_ref.data();
        for r in 0..4 {
            for c in 0..4 {
                assert!(
                    (od.at(r, c) - rd.at(r, c)).abs() < 1e-5,
                    "checkpoint output mismatch [{},{}]",
                    r,
                    c
                );
            }
        }
    }

    #[test]
    fn test_checkpoint_backward_gradient_flows() {
        let x_data = Mat::from_fn(2, 4, |r, c| (r * 4 + c) as f32 * 0.5 + 0.1);
        let w_data = Mat::from_fn(4, 4, |_, _| 0.25);
        let x = TensorNode::leaf(x_data);
        let w = TensorNode::leaf(w_data);
        let w_c = w.clone();
        let chk = Checkpoint::new(move |x: &TensorNode| x.matmul(&w_c));
        let out = chk.forward(&x);
        // Build a loss: sum of all out elements
        let sum_val: f32 = out.data().data.iter().sum();
        let loss = TensorNode::leaf(Mat::new(vec![sum_val], 1, 1));
        let out_c = out.clone();
        loss.set_backward(
            Box::new(move || {
                let ones = Mat::ones(out_c.data().rows, out_c.data().cols);
                out_c.set_grad(ones);
                out_c.call_backward_fn();
            }),
            vec![out],
        );
        loss.backward();
        let gx = x.grad();
        assert!(
            gx.data.iter().any(|&v| v.abs() > 1e-6),
            "expected non-zero gradient through checkpoint"
        );
    }

    #[test]
    fn test_checkpoint_saves_only_input_as_prev() {
        // The output of Checkpoint::forward has exactly 1 prev (the input x),
        // not the full internal forward graph.
        let x = TensorNode::leaf(Mat::from_fn(2, 2, |r, c| (r * 2 + c) as f32));
        let chk = Checkpoint::new(|x: &TensorNode| {
            let t1 = x.gelu();
            t1.gelu()
        });
        let out = chk.forward(&x);
        assert_eq!(
            out.0.borrow().prev.len(),
            1,
            "checkpointed output should have exactly 1 prev"
        );
    }

    // -------------------------------------------------------------------------
    // Q8 quantization tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_q8_quantize_shape_preserved() {
        let m = Mat::from_fn(8, 16, |r, c| (r * 16 + c) as f32 * 0.1 - 1.0);
        let q = Q8Mat::quantize(&m);
        assert_eq!(q.rows, 8);
        assert_eq!(q.cols, 16);
        assert_eq!(q.packed.len(), 8 * 16);
    }

    #[test]
    fn test_q8_roundtrip_accurate() {
        // Small matrix: deq(quant(x)) should closely approximate x
        let m = Mat::from_fn(4, 8, |r, c| (r as f32 - 1.5) * (c as f32 + 0.5));
        let q = Q8Mat::quantize(&m);
        let back = q.dequantize();
        assert_eq!((back.rows, back.cols), (m.rows, m.cols));
        for r in 0..m.rows {
            for c in 0..m.cols {
                let err = (back.at(r, c) - m.at(r, c)).abs();
                // Q8 max error = absmax/254 per block
                let absmax = m.data.iter().map(|&v| v.abs()).fold(0.0f32, f32::max);
                let max_err = absmax / 254.0 + 1e-5;
                assert!(
                    err <= max_err,
                    "Q8 roundtrip [{r},{c}]: orig={} deq={} err={:.4} max={:.4}",
                    m.at(r, c),
                    back.at(r, c),
                    err,
                    max_err
                );
            }
        }
    }

    #[test]
    fn test_q8_size_bytes_less_than_f32() {
        let m = Mat::from_fn(32, 64, |r, c| (r * 64 + c) as f32 * 0.01);
        let q = Q8Mat::quantize(&m);
        let f32_bytes = 32 * 64 * 4;
        assert!(
            q.size_bytes() < f32_bytes,
            "Q8 must use less memory than f32: {} vs {}",
            q.size_bytes(),
            f32_bytes
        );
    }

    #[test]
    fn test_q8_compression_ratio() {
        // 32*64 = 2048 elements, 1 byte each = 2048 bytes packed
        // + ceil(2048/64)=32 scale values * 4 = 128 bytes → total 2176 bytes
        // f32: 2048 * 4 = 8192 bytes → ratio ≈ 3.77
        let m = Mat::from_fn(32, 64, |_, _| 1.0);
        let q = Q8Mat::quantize(&m);
        assert!(
            q.compression_ratio() > 2.0,
            "Q8 should compress by at least 2×, got {:.2}×",
            q.compression_ratio()
        );
    }

    #[test]
    fn test_q8_matmul_matches_f32() {
        // Q8 matmul must closely approximate f32 matmul
        let a = Mat::from_fn(3, 8, |r, c| (r * 8 + c) as f32 * 0.1 - 1.0);
        let b = Mat::from_fn(4, 8, |r, c| (r * 8 + c) as f32 * 0.05 - 0.5);
        let q = Q8Mat::quantize(&b);

        let exact = a.matmul(&b.transpose());
        let approx = q.matmul_q8_t(&a);

        assert_eq!((approx.rows, approx.cols), (3, 4));
        for r in 0..3 {
            for c in 0..4 {
                let err = (approx.at(r, c) - exact.at(r, c)).abs();
                assert!(
                    err < 0.1,
                    "Q8 matmul [{r},{c}]: exact={:.4} approx={:.4} err={:.4}",
                    exact.at(r, c),
                    approx.at(r, c),
                    err
                );
            }
        }
    }

    #[test]
    fn test_q8_zero_matrix() {
        let m = Mat::zeros(4, 4);
        let q = Q8Mat::quantize(&m);
        let back = q.dequantize();
        for &v in &back.data {
            assert_eq!(v, 0.0, "zero matrix should roundtrip to zero");
        }
    }

    // -------------------------------------------------------------------------
    // Checkpoint save/load tests
    // -------------------------------------------------------------------------

    #[test]
    fn test_save_load_checkpoint_roundtrip() {
        let tmp = std::env::temp_dir().join("test_ckpt_roundtrip.bin");
        let path = tmp.to_str().unwrap();

        let t1 = TensorNode::leaf(Mat::new(vec![1.0f32, 2.0, 3.0, 4.0], 2, 2));
        let t2 = TensorNode::leaf(Mat::new(vec![5.0f32, 6.0], 1, 2));

        save_checkpoint(path, &[("layer.w", &t1), ("layer.b", &t2)])
            .expect("save_checkpoint failed");

        let loaded = load_checkpoint(path).expect("load_checkpoint failed");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "layer.w");
        assert_eq!(loaded[1].0, "layer.b");

        let m1 = &loaded[0].1;
        assert_eq!((m1.rows, m1.cols), (2, 2));
        assert_eq!(m1.at(0, 0), 1.0);
        assert_eq!(m1.at(1, 1), 4.0);

        let m2 = &loaded[1].1;
        assert_eq!((m2.rows, m2.cols), (1, 2));
        assert_eq!(m2.at(0, 0), 5.0);
        assert_eq!(m2.at(0, 1), 6.0);
    }

    #[test]
    fn test_save_load_checkpoint_large_tensor() {
        let tmp = std::env::temp_dir().join("test_ckpt_large.bin");
        let path = tmp.to_str().unwrap();
        let rows = 64usize;
        let cols = 128usize;
        let data: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.001 - 0.5).collect();
        let node = TensorNode::leaf(Mat::new(data.clone(), rows, cols));

        save_checkpoint(path, &[("big", &node)]).expect("save failed");
        let loaded = load_checkpoint(path).expect("load failed");
        let m = &loaded[0].1;
        assert_eq!((m.rows, m.cols), (rows, cols));
        for (i, (&orig, &got)) in data.iter().zip(m.data.iter()).enumerate() {
            assert_eq!(orig, got, "data mismatch at index {}", i);
        }
    }

    #[test]
    fn test_load_checkpoint_bad_magic_returns_err() {
        let tmp = std::env::temp_dir().join("test_ckpt_bad.bin");
        let path = tmp.to_str().unwrap();
        // Write garbage
        std::fs::write(path, b"BADM\x01\x00\x00\x00\x00\x00\x00\x00").unwrap();
        let res = load_checkpoint(path);
        assert!(res.is_err(), "bad magic should return Err");
    }

    #[test]
    fn test_checkpoint_name_survives_roundtrip() {
        let tmp = std::env::temp_dir().join("test_ckpt_name.bin");
        let path = tmp.to_str().unwrap();
        let node = TensorNode::leaf(Mat::ones(1, 4));
        save_checkpoint(path, &[("model.layers.0.attn.q_proj.weight", &node)]).unwrap();
        let loaded = load_checkpoint(path).unwrap();
        assert_eq!(loaded[0].0, "model.layers.0.attn.q_proj.weight");
    }

    // =========================================================================
    // batched_gqa_attention — correctness vs gqa_attention
    // =========================================================================

    #[test]
    fn test_batched_gqa_output_matches_gqa() {
        // batched_gqa_attention must produce identical forward output to gqa_attention.
        let t = 5;
        let n_q = 4;
        let n_kv = 2;
        let dh = 8;
        let q_data = Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.01 + 0.1);
        let k_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.02 + 0.05
        });
        let v_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.015 + 0.03
        });

        let q1 = TensorNode::leaf(q_data.clone());
        let k1 = TensorNode::leaf(k_data.clone());
        let v1 = TensorNode::leaf(v_data.clone());
        let out_ref = TensorNode::gqa_attention(&q1, &k1, &v1, n_q, n_kv, dh);

        let q2 = TensorNode::leaf(q_data);
        let k2 = TensorNode::leaf(k_data);
        let v2 = TensorNode::leaf(v_data);
        let out_bat = TensorNode::batched_gqa_attention(&q2, &k2, &v2, n_q, n_kv, dh);

        let ref_d = out_ref.data().clone();
        let bat_d = out_bat.data().clone();
        assert_eq!((ref_d.rows, ref_d.cols), (bat_d.rows, bat_d.cols));
        for r in 0..t {
            for c in 0..(n_q * dh) {
                assert!(
                    (ref_d.at(r, c) - bat_d.at(r, c)).abs() < 1e-4,
                    "output mismatch at [{r},{c}]: ref={:.5} bat={:.5}",
                    ref_d.at(r, c),
                    bat_d.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_batched_gqa_mha_output_matches_gqa() {
        // n_q == n_kv (standard MHA case): must still match.
        let t = 4;
        let n = 3;
        let dh = 6;
        let q_data = Mat::from_fn(t, n * dh, |r, c| (r + c) as f32 * 0.03 + 0.02);
        let k_data = q_data.clone();
        let v_data = Mat::from_fn(t, n * dh, |r, c| ((r * (n * dh) + c) as f32 + 1.0) * 0.02);

        let out_ref = TensorNode::gqa_attention(
            &TensorNode::leaf(q_data.clone()),
            &TensorNode::leaf(k_data.clone()),
            &TensorNode::leaf(v_data.clone()),
            n,
            n,
            dh,
        );
        let out_bat = TensorNode::batched_gqa_attention(
            &TensorNode::leaf(q_data.clone()),
            &TensorNode::leaf(k_data.clone()),
            &TensorNode::leaf(v_data.clone()),
            n,
            n,
            dh,
        );

        let ref_d = out_ref.data().clone();
        let bat_d = out_bat.data().clone();
        for r in 0..t {
            for c in 0..(n * dh) {
                assert!(
                    (ref_d.at(r, c) - bat_d.at(r, c)).abs() < 1e-4,
                    "MHA mismatch at [{r},{c}]: ref={:.5} bat={:.5}",
                    ref_d.at(r, c),
                    bat_d.at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_batched_gqa_output_shape() {
        let t = 6;
        let n_q = 4;
        let n_kv = 2;
        let dh = 8;
        let q = TensorNode::leaf(Mat::zeros(t, n_q * dh));
        let k = TensorNode::leaf(Mat::zeros(t, n_kv * dh));
        let v = TensorNode::leaf(Mat::zeros(t, n_kv * dh));
        let out = TensorNode::batched_gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (t, n_q * dh));
    }

    #[test]
    fn test_batched_gqa_causal_first_token() {
        // Token 0 can only attend to itself — output[0] must equal V[0..dh] exactly
        // (softmax over a single position → weight=1.0).
        let t = 4;
        let n_q = 2;
        let n_kv = 2;
        let dh = 4;
        let q = TensorNode::leaf(Mat::ones(t, n_q * dh));
        let k = TensorNode::leaf(Mat::ones(t, n_kv * dh));
        let v = TensorNode::leaf(Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c + 1) as f32
        }));
        let out = TensorNode::batched_gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let d = out.data();
        // For each query head h, token 0 must match v[0, h*dh .. (h+1)*dh]
        for h in 0..n_q {
            let kvh = h; // group_size=1 for n_q==n_kv
            for c in 0..dh {
                let expected = v.data().at(0, kvh * dh + c);
                let got = d.at(0, h * dh + c);
                assert!(
                    (got - expected).abs() < 1e-4,
                    "head {h}, col {c}: expected {expected:.4} got {got:.4}"
                );
            }
        }
    }

    #[test]
    fn test_batched_gqa_grad_v() {
        // Gradient check for dV: batched vs numerical.
        let t = 3;
        let n_q = 2;
        let n_kv = 1;
        let dh = 4;
        let q_data = Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.05 + 0.1);
        let k_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.03 + 0.05
        });
        let v_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.04 + 0.02
        });

        let num = numerical_grad(
            &|v| {
                let qn = TensorNode::leaf(q_data.clone());
                let kn = TensorNode::leaf(k_data.clone());
                let vn = TensorNode::leaf(v.clone());
                TensorNode::batched_gqa_attention(&qn, &kn, &vn, n_q, n_kv, dh)
                    .data()
                    .data
                    .iter()
                    .sum::<f32>()
            },
            &v_data,
        );

        let q = TensorNode::leaf(q_data);
        let k = TensorNode::leaf(k_data);
        let v = TensorNode::leaf(v_data);
        let out = TensorNode::batched_gqa_attention(&q, &k, &v, n_q, n_kv, dh);
        let (r, c) = {
            let d = out.data();
            (d.rows, d.cols)
        };
        out.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out);

        let vg = v.grad().clone();
        for row in 0..t {
            for col in 0..(n_kv * dh) {
                assert!(
                    approx(vg.at(row, col), num.at(row, col)),
                    "batched dV[{row},{col}]: analytical={:.4} numerical={:.4}",
                    vg.at(row, col),
                    num.at(row, col)
                );
            }
        }
    }

    #[test]
    fn test_batched_gqa_grad_matches_gqa_grad() {
        // Gradient of batched_gqa must match gradient of gqa_attention exactly.
        let t = 3;
        let n_q = 4;
        let n_kv = 2;
        let dh = 4;
        let q_data = Mat::from_fn(t, n_q * dh, |r, c| (r * (n_q * dh) + c) as f32 * 0.05 + 0.1);
        let k_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.03 + 0.05
        });
        let v_data = Mat::from_fn(t, n_kv * dh, |r, c| {
            (r * (n_kv * dh) + c) as f32 * 0.04 + 0.02
        });

        // Reference: gqa_attention
        let q1 = TensorNode::leaf(q_data.clone());
        let k1 = TensorNode::leaf(k_data.clone());
        let v1 = TensorNode::leaf(v_data.clone());
        let out1 = TensorNode::gqa_attention(&q1, &k1, &v1, n_q, n_kv, dh);
        let (r, c) = {
            let d = out1.data();
            (d.rows, d.cols)
        };
        out1.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out1);

        // Batched: batched_gqa_attention
        let q2 = TensorNode::leaf(q_data);
        let k2 = TensorNode::leaf(k_data);
        let v2 = TensorNode::leaf(v_data);
        let out2 = TensorNode::batched_gqa_attention(&q2, &k2, &v2, n_q, n_kv, dh);
        out2.0.borrow_mut().grad = Mat::ones(r, c);
        call_backward(&out2);

        let dq1 = q1.grad().clone();
        let dq2 = q2.grad().clone();
        let dk1 = k1.grad().clone();
        let dk2 = k2.grad().clone();
        let dv1 = v1.grad().clone();
        let dv2 = v2.grad().clone();

        for row in 0..t {
            for col in 0..(n_q * dh) {
                assert!(
                    approx(dq1.at(row, col), dq2.at(row, col)),
                    "dQ mismatch [{row},{col}]: ref={:.4} bat={:.4}",
                    dq1.at(row, col),
                    dq2.at(row, col)
                );
            }
        }
        for row in 0..t {
            for col in 0..(n_kv * dh) {
                assert!(
                    approx(dk1.at(row, col), dk2.at(row, col)),
                    "dK mismatch [{row},{col}]: ref={:.4} bat={:.4}",
                    dk1.at(row, col),
                    dk2.at(row, col)
                );
                assert!(
                    approx(dv1.at(row, col), dv2.at(row, col)),
                    "dV mismatch [{row},{col}]: ref={:.4} bat={:.4}",
                    dv1.at(row, col),
                    dv2.at(row, col)
                );
            }
        }
    }
}
