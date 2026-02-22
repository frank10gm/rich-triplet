/// # Metal GPU matrix multiplication
///
/// Provides two GPU-accelerated entry points:
/// * `metal_matmul(a, b) -> Mat`  — single `C = A @ B`
/// * `metal_matmul_batched(pairs) -> Vec<Mat>` — B independent `C_i = A_i @ B_i`
///   in one GPU dispatch (batch parallelism along the Z grid axis).
///
/// ## Feature gate
///
/// This module is compiled only when `--features metal` is passed. The caller
/// (`autograd2.rs`) inserts a `#[cfg(feature = "metal")]` branch in `Mat::matmul`
/// before the BLAS / parallel / scalar fallbacks.
///
/// ## Design
///
/// * One `MetalContext` is initialised lazily per thread via a `std::cell::OnceCell`.
///   Calling `MTLCreateSystemDefaultDevice` is expensive (~1 ms); we pay it once.
/// * All Metal objects (device, queue, pipelines) live inside `MetalContext` for the
///   lifetime of the thread.
/// * Buffers are created fresh on every call with `MTLResourceStorageModeShared`
///   (Apple Silicon unified memory — no explicit synchronisation needed).
/// * The MSL kernels are compiled from source at first use, cached in the context.
///
/// ## MSL kernel (`matmul_tiled`)
///
/// Uses 16×16 threadgroup tiles and `threadgroup float` shared memory to amortise
/// global memory reads. Each tile of A and B is loaded once into fast threadgroup
/// memory and reused by all 16 threads in the row/column.
///
/// ## MSL kernel (`matmul_tiled_batched`)
///
/// Same tiled algorithm but the grid has a Z dimension equal to the batch size B.
/// Each threadgroup in slice `tgid.z = b` reads from `A[b*M*K..]` and writes to
/// `C[b*M*N..]`. All B matmuls run concurrently on the GPU.
///
/// ## CPU threshold
///
/// For small matrices the Metal overhead (buffer allocation, command encoding,
/// GPU wake-up) dominates. We fall back to the scalar CPU path when
/// `M * K * N < METAL_THRESHOLD` (default 32768 = 32³).

#[cfg(feature = "metal")]
mod inner {
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::NSString;
    use objc2_metal::{
        MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLCreateSystemDefaultDevice,
        MTLDevice, MTLFunction, MTLLibrary, MTLResourceOptions, MTLSize,
    };
    use std::cell::OnceCell;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use crate::autograd2::Mat;

    // -------------------------------------------------------------------------
    // Minimum ops before using Metal (M*K*N flops).
    // Below this the GPU wake-up cost exceeds compute savings.
    // -------------------------------------------------------------------------
    const METAL_THRESHOLD: usize = 32_768; // 32³

    // -------------------------------------------------------------------------
    // MSL source — tiled matmul with shared memory
    //
    // TILE_SIZE must match the threadgroup size (16×16) used at dispatch.
    // Each thread computes one output element C[row,col].
    //
    // Algorithm:
    //   1. The K dimension is split into (K / TILE_SIZE) tiles.
    //   2. Each tile loads a 16×16 block of A and B into fast threadgroup memory.
    //   3. All 256 threads in the threadgroup accumulate the partial dot product
    //      from that tile, then synchronise before loading the next.
    //
    // Boundary handling: out-of-range loads read 0.0 so partial tiles work
    // correctly when M, K, or N are not multiples of TILE_SIZE.
    // -------------------------------------------------------------------------
    const MATMUL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define TS 16u

// Single matrix multiply: C[M,N] = A[M,K] @ B[K,N]
kernel void matmul_tiled(
    device const float* A  [[ buffer(0) ]],
    device const float* B  [[ buffer(1) ]],
    device       float* C  [[ buffer(2) ]],
    constant     uint&  M  [[ buffer(3) ]],
    constant     uint&  K  [[ buffer(4) ]],
    constant     uint&  N  [[ buffer(5) ]],
    uint2 tgid [[ threadgroup_position_in_grid ]],
    uint2 tid  [[ thread_position_in_threadgroup ]])
{
    uint row = tgid.y * TS + tid.y;
    uint col = tgid.x * TS + tid.x;

    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    float acc = 0.0f;
    uint n_tiles = (K + TS - 1u) / TS;

    for (uint t = 0u; t < n_tiles; t++) {
        uint a_col = t * TS + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        uint b_row = t * TS + tid.y;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0u; k < TS; k++)
            acc += As[tid.y][k] * Bs[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// Batched matrix multiply: C[b,M,N] = A[b,M,K] @ B[b,K,N]
//
// All B pairs have the same M, K, N.  The matrices are packed contiguously:
//   A_flat[b * M*K  ..  (b+1) * M*K  - 1]
//   B_flat[b * K*N  ..  (b+1) * K*N  - 1]
//   C_flat[b * M*N  ..  (b+1) * M*N  - 1]
//
// The grid Z dimension equals the batch size B; tgid.z selects the slice.
kernel void matmul_tiled_batched(
    device const float* A  [[ buffer(0) ]],
    device const float* B  [[ buffer(1) ]],
    device       float* C  [[ buffer(2) ]],
    constant     uint&  M  [[ buffer(3) ]],
    constant     uint&  K  [[ buffer(4) ]],
    constant     uint&  N  [[ buffer(5) ]],
    uint3 tgid [[ threadgroup_position_in_grid ]],
    uint3 tid  [[ thread_position_in_threadgroup ]])
{
    uint b   = tgid.z;
    uint row = tgid.y * TS + tid.y;
    uint col = tgid.x * TS + tid.x;

    // Offset into the packed batch buffers for slice b
    device const float* Ab = A + b * M * K;
    device const float* Bb = B + b * K * N;
    device       float* Cb = C + b * M * N;

    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    float acc = 0.0f;
    uint n_tiles = (K + TS - 1u) / TS;

    for (uint t = 0u; t < n_tiles; t++) {
        uint a_col = t * TS + tid.x;
        As[tid.y][tid.x] = (row < M && a_col < K) ? Ab[row * K + a_col] : 0.0f;

        uint b_row = t * TS + tid.y;
        Bs[tid.y][tid.x] = (b_row < K && col < N) ? Bb[b_row * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0u; k < TS; k++)
            acc += As[tid.y][k] * Bs[k][tid.x];

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N)
        Cb[row * N + col] = acc;
}
"#;

    // -------------------------------------------------------------------------
    // Cached Metal objects — one per thread
    // -------------------------------------------------------------------------
    pub struct MetalContext {
        pub device:            Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue:             Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pub pipeline:          Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pub pipeline_batched:  Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    thread_local! {
        static METAL_CTX: OnceCell<MetalContext> = OnceCell::new();
    }

    fn init_context() -> MetalContext {
        let device = MTLCreateSystemDefaultDevice()
            .expect("Metal: no GPU device found");

        let queue = device
            .newCommandQueue()
            .expect("Metal: could not create command queue");

        // Compile both kernels from the same source string.
        let source = NSString::from_str(MATMUL_MSL);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Metal: MSL compilation failed");

        let make_pipeline = |name: &str| {
            let fn_name = NSString::from_str(name);
            let func: Retained<ProtocolObject<dyn MTLFunction>> = library
                .newFunctionWithName(&fn_name)
                .unwrap_or_else(|| panic!("Metal: function '{}' not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|_| panic!("Metal: could not create pipeline for '{}'", name))
        };

        let pipeline         = make_pipeline("matmul_tiled");
        let pipeline_batched = make_pipeline("matmul_tiled_batched");

        MetalContext { device, queue, pipeline, pipeline_batched }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    /// Upload a &[f32] slice to a new MTLBuffer (StorageModeShared).
    fn upload(device: &ProtocolObject<dyn MTLDevice>, data: &[f32])
        -> Retained<ProtocolObject<dyn MTLBuffer>>
    {
        let byte_len = data.len() * 4;
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    byte_len,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: buffer allocation failed")
        }
    }

    /// Allocate an empty output MTLBuffer (StorageModeShared).
    fn alloc_output(device: &ProtocolObject<dyn MTLDevice>, elems: usize)
        -> Retained<ProtocolObject<dyn MTLBuffer>>
    {
        device
            .newBufferWithLength_options(
                elems * 4,
                MTLResourceOptions::StorageModeShared,
            )
            .expect("Metal: output buffer allocation failed")
    }

    /// Upload a single u32 as a 4-byte constant buffer.
    fn upload_u32(device: &ProtocolObject<dyn MTLDevice>, val: u32)
        -> Retained<ProtocolObject<dyn MTLBuffer>>
    {
        let bytes = val.to_ne_bytes();
        let ptr = NonNull::new(bytes.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    4,
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: constant buffer allocation failed")
        }
    }

    // -------------------------------------------------------------------------
    // Public entry point
    // -------------------------------------------------------------------------

    /// GPU-accelerated `C = A @ B`  ([M,K] × [K,N] → [M,N]).
    ///
    /// Falls back to CPU scalar loop when the problem is too small to benefit.
    pub fn metal_matmul(a: &Mat, b: &Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, b.cols);

        // Small-matrix fast path — avoid Metal overhead.
        if m * k * n < METAL_THRESHOLD {
            let mut out = Mat::zeros(m, n);
            for i in 0..m {
                for p in 0..k {
                    let a_ip = a.at(i, p);
                    for j in 0..n {
                        *out.at_mut(i, j) += a_ip * b.at(p, j);
                    }
                }
            }
            return out;
        }

        METAL_CTX.with(|cell| {
            let ctx = cell.get_or_init(init_context);

            // Upload inputs and allocate output.
            let buf_a = upload(&ctx.device, &a.data);
            let buf_b = upload(&ctx.device, &b.data);
            let buf_c = alloc_output(&ctx.device, m * n);
            let buf_m = upload_u32(&ctx.device, m as u32);
            let buf_k = upload_u32(&ctx.device, k as u32);
            let buf_n = upload_u32(&ctx.device, n as u32);

            // Encode and dispatch.
            let cmd_buf = ctx.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");

            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            encoder.setComputePipelineState(&ctx.pipeline);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&buf_m), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&buf_k), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&buf_n), 0, 5);
            }

            // Threadgroup 16×16 matches TS in the MSL kernel.
            // Grid is the number of 16×16 threadgroups needed to cover M×N.
            let tg_size = MTLSize { width: 16, height: 16, depth: 1 };
            let grid_size = MTLSize {
                width:  (n + 15) / 16,
                height: (m + 15) / 16,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            // Read output back from shared-memory buffer.
            let ptr = buf_c.contents().as_ptr() as *const f32;
            let out_data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(ptr, m * n).to_vec()
            };

            Mat::new(out_data, m, n)
        })
    }

    // -------------------------------------------------------------------------
    // Batched entry point
    // -------------------------------------------------------------------------

    /// GPU-accelerated batched `C_i = A_i @ B_i` for i in 0..B.
    ///
    /// All pairs must have the same shape: A[M,K], B[K,N].
    /// The B matmuls are dispatched in a single Metal call with the batch
    /// dimension mapped to the Z axis of the compute grid, so all B slices
    /// run concurrently on the GPU.
    ///
    /// Falls back to B sequential `metal_matmul` calls if any of the
    /// following is true:
    /// * `pairs` is empty
    /// * batch size is 1 (no benefit over the single kernel)
    /// * problem size per slice is below `METAL_THRESHOLD`
    pub fn metal_matmul_batched(pairs: &[(&Mat, &Mat)]) -> Vec<Mat> {
        if pairs.is_empty() { return vec![]; }
        if pairs.len() == 1 {
            return vec![metal_matmul(pairs[0].0, pairs[0].1)];
        }

        let (m, k) = (pairs[0].0.rows, pairs[0].0.cols);
        let n      =  pairs[0].1.cols;

        // Validate all pairs share the same shape.
        for (i, (a, b)) in pairs.iter().enumerate() {
            assert_eq!((a.rows, a.cols), (m, k),
                "metal_matmul_batched: pair {} A shape [{},{}] != [{},{}]",
                i, a.rows, a.cols, m, k);
            assert_eq!((b.rows, b.cols), (k, n),
                "metal_matmul_batched: pair {} B shape [{},{}] != [{},{}]",
                i, b.rows, b.cols, k, n);
        }

        let batch = pairs.len();

        // Small problem — fall back to sequential single-kernel calls.
        if m * k * n < METAL_THRESHOLD {
            return pairs.iter().map(|(a, b)| metal_matmul(a, b)).collect();
        }

        METAL_CTX.with(|cell| {
            let ctx = cell.get_or_init(init_context);

            // Pack all A slices contiguously, then all B slices.
            let a_flat: Vec<f32> = pairs.iter().flat_map(|(a, _)| a.data.iter().cloned()).collect();
            let b_flat: Vec<f32> = pairs.iter().flat_map(|(_, b)| b.data.iter().cloned()).collect();

            let buf_a = upload(&ctx.device, &a_flat);
            let buf_b = upload(&ctx.device, &b_flat);
            let buf_c = alloc_output(&ctx.device, batch * m * n);
            let buf_m = upload_u32(&ctx.device, m as u32);
            let buf_k = upload_u32(&ctx.device, k as u32);
            let buf_n = upload_u32(&ctx.device, n as u32);

            let cmd_buf = ctx.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");

            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            encoder.setComputePipelineState(&ctx.pipeline_batched);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buf_a), 0, 0);
                encoder.setBuffer_offset_atIndex(Some(&buf_b), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&buf_c), 0, 2);
                encoder.setBuffer_offset_atIndex(Some(&buf_m), 0, 3);
                encoder.setBuffer_offset_atIndex(Some(&buf_k), 0, 4);
                encoder.setBuffer_offset_atIndex(Some(&buf_n), 0, 5);
            }

            // XY grid: 16×16 tiles over [M, N].  Z grid: batch size B.
            let tg_size = MTLSize { width: 16, height: 16, depth: 1 };
            let grid_size = MTLSize {
                width:  (n + 15) / 16,
                height: (m + 15) / 16,
                depth:  batch,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            // Unpack the packed output buffer into B separate Mat values.
            let ptr = buf_c.contents().as_ptr() as *const f32;
            let out_flat: &[f32] = unsafe { std::slice::from_raw_parts(ptr, batch * m * n) };

            (0..batch).map(|b| {
                let start = b * m * n;
                Mat::new(out_flat[start..start + m * n].to_vec(), m, n)
            }).collect()
        })
    }

    // -------------------------------------------------------------------------
    // Tests
    // -------------------------------------------------------------------------
    #[cfg(test)]
    mod tests {
        use super::*;

        fn small_matmul_cpu(a: &Mat, b: &Mat) -> Mat {
            let (m, k, n) = (a.rows, a.cols, b.cols);
            let mut out = Mat::zeros(m, n);
            for i in 0..m { for p in 0..k { for j in 0..n {
                *out.at_mut(i, j) += a.at(i, p) * b.at(p, j);
            }}}
            out
        }

        #[test]
        fn test_metal_matmul_identity() {
            // C = A @ I should equal A
            let a = Mat::from_fn(4, 4, |r, c| if r == c { 1.0 } else { 0.5 });
            let i = Mat::from_fn(4, 4, |r, c| if r == c { 1.0 } else { 0.0 });
            let got = metal_matmul(&a, &i);
            assert_eq!(got.rows, 4);
            assert_eq!(got.cols, 4);
            for r in 0..4 { for c in 0..4 {
                assert!((got.at(r, c) - a.at(r, c)).abs() < 1e-5,
                    "identity mismatch at [{r},{c}]");
            }}
        }

        #[test]
        fn test_metal_matmul_known_result() {
            // 2×3 × 3×2 with known values
            let a = Mat::new(vec![1.0, 2.0, 3.0,
                                  4.0, 5.0, 6.0], 2, 3);
            let b = Mat::new(vec![7.0,  8.0,
                                  9.0,  10.0,
                                  11.0, 12.0], 3, 2);
            let got = metal_matmul(&a, &b);
            // row0: 1*7+2*9+3*11=58,  1*8+2*10+3*12=64
            // row1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
            assert!((got.at(0, 0) - 58.0).abs() < 1e-4);
            assert!((got.at(0, 1) - 64.0).abs() < 1e-4);
            assert!((got.at(1, 0) - 139.0).abs() < 1e-4);
            assert!((got.at(1, 1) - 154.0).abs() < 1e-4);
        }

        #[test]
        fn test_metal_matmul_matches_cpu() {
            // Larger random-ish matrix, compare GPU vs CPU
            let m = 64; let k = 32; let n = 48;
            let a = Mat::from_fn(m, k, |r, c| ((r * k + c) as f32) * 0.01);
            let b = Mat::from_fn(k, n, |r, c| ((r * n + c) as f32) * 0.01 - 0.5);
            let gpu = metal_matmul(&a, &b);
            let cpu = small_matmul_cpu(&a, &b);
            assert_eq!(gpu.rows, m); assert_eq!(gpu.cols, n);
            for r in 0..m { for c in 0..n {
                let diff = (gpu.at(r, c) - cpu.at(r, c)).abs();
                assert!(diff < 1e-3,
                    "GPU/CPU mismatch at [{r},{c}]: gpu={} cpu={}", gpu.at(r,c), cpu.at(r,c));
            }}
        }

        #[test]
        fn test_metal_matmul_non_square() {
            let a = Mat::from_fn(3, 7, |r, c| (r + c) as f32);
            let b = Mat::from_fn(7, 5, |r, c| (r * c) as f32 * 0.1);
            let gpu = metal_matmul(&a, &b);
            let cpu = small_matmul_cpu(&a, &b);
            assert_eq!(gpu.rows, 3); assert_eq!(gpu.cols, 5);
            for r in 0..3 { for c in 0..5 {
                let diff = (gpu.at(r, c) - cpu.at(r, c)).abs();
                assert!(diff < 1e-3);
            }}
        }

        #[test]
        fn test_metal_matmul_small_uses_cpu_path() {
            // 4×4 × 4×4 = 256 ops < METAL_THRESHOLD=32768 → uses CPU fast path
            // Result must still be correct
            let a = Mat::from_fn(4, 4, |r, _c| (r + 1) as f32);
            let b = Mat::from_fn(4, 4, |_r, c| (c + 1) as f32);
            let got = metal_matmul(&a, &b);
            let cpu = small_matmul_cpu(&a, &b);
            for r in 0..4 { for c in 0..4 {
                assert!((got.at(r, c) - cpu.at(r, c)).abs() < 1e-5);
            }}
        }

        // --- Batched kernel tests ---

        #[test]
        fn test_metal_matmul_batched_matches_single() {
            // Each C_i = A_i @ B_i should equal individual metal_matmul results.
            let m = 32; let k = 16; let n = 24;
            let pairs: Vec<(Mat, Mat)> = (0..4).map(|b| {
                let a = Mat::from_fn(m, k, |r, c| ((b * 100 + r * k + c) as f32) * 0.01);
                let bm = Mat::from_fn(k, n, |r, c| ((b * 50  + r * n + c) as f32) * 0.01 - 0.5);
                (a, bm)
            }).collect();

            let refs: Vec<(&Mat, &Mat)> = pairs.iter().map(|(a, b)| (a, b)).collect();
            let batched = metal_matmul_batched(&refs);

            for (i, ((a, b), got)) in pairs.iter().zip(batched.iter()).enumerate() {
                let expected = small_matmul_cpu(a, b);
                assert_eq!(got.rows, m); assert_eq!(got.cols, n);
                for r in 0..m { for c in 0..n {
                    let diff = (got.at(r, c) - expected.at(r, c)).abs();
                    assert!(diff < 1e-3,
                        "batch {i} [{r},{c}]: got {} expected {}", got.at(r,c), expected.at(r,c));
                }}
            }
        }

        #[test]
        fn test_metal_matmul_batched_single_pair_delegates() {
            // batch=1 delegates to single-kernel path, still correct
            let a = Mat::from_fn(8, 8, |r, c| (r + c) as f32 * 0.1);
            let b = Mat::from_fn(8, 8, |r, c| (r * c) as f32 * 0.05 + 0.1);
            let result = metal_matmul_batched(&[(&a, &b)]);
            assert_eq!(result.len(), 1);
            let expected = small_matmul_cpu(&a, &b);
            for r in 0..8 { for c in 0..8 {
                assert!((result[0].at(r, c) - expected.at(r, c)).abs() < 1e-3);
            }}
        }

        #[test]
        fn test_metal_matmul_batched_empty() {
            let result = metal_matmul_batched(&[]);
            assert_eq!(result.len(), 0);
        }

        #[test]
        fn test_metal_matmul_batched_non_square() {
            // Verify non-square matrices work across a batch
            let m = 5; let k = 7; let n = 3;
            let pairs: Vec<(Mat, Mat)> = (0..3).map(|b| {
                let a = Mat::from_fn(m, k, |r, c| (b * 10 + r + c) as f32);
                let bm = Mat::from_fn(k, n, |r, c| (r * c + b + 1) as f32 * 0.1);
                (a, bm)
            }).collect();
            let refs: Vec<(&Mat, &Mat)> = pairs.iter().map(|(a, b)| (a, b)).collect();
            let batched = metal_matmul_batched(&refs);
            assert_eq!(batched.len(), 3);
            for (i, ((a, b), got)) in pairs.iter().zip(batched.iter()).enumerate() {
                let expected = small_matmul_cpu(a, b);
                for r in 0..m { for c in 0..n {
                    let diff = (got.at(r, c) - expected.at(r, c)).abs();
                    assert!(diff < 1e-2, "batch {i} [{r},{c}]: {:.4} vs {:.4}",
                        got.at(r,c), expected.at(r,c));
                }}
            }
        }
    }
}

#[cfg(feature = "metal")]
pub use inner::{metal_matmul, metal_matmul_batched};
