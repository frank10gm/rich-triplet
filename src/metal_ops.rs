/// # Metal GPU matrix multiplication
///
/// Provides `metal_matmul(a, b) -> Mat` accelerated via Apple Metal compute shaders.
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
/// * All Metal objects (device, queue, pipeline) live inside `MetalContext` for the
///   lifetime of the thread.
/// * Buffers are created fresh on every call with `MTLResourceStorageModeShared`
///   (Apple Silicon unified memory — no explicit synchronisation needed).
/// * The MSL kernel is compiled from source at first use, also cached in the context.
///
/// ## MSL kernel (`matmul_f32`)
///
/// ```metal
/// kernel void matmul_f32(
///     device const float *A  [[buffer(0)]],
///     device const float *B  [[buffer(1)]],
///     device       float *C  [[buffer(2)]],
///     constant     uint  &M  [[buffer(3)]],
///     constant     uint  &K  [[buffer(4)]],
///     constant     uint  &N  [[buffer(5)]],
///     uint2 gid [[thread_position_in_grid]])
/// {
///     uint row = gid.y, col = gid.x;
///     if (row >= M || col >= N) return;
///     float acc = 0.0;
///     for (uint k = 0; k < K; k++)
///         acc += A[row * K + k] * B[k * N + col];
///     C[row * N + col] = acc;
/// }
/// ```
///
/// Threadgroup size is 16×16; grid is ceil(N/16) × ceil(M/16).
///
/// ## CPU threshold
///
/// For small matrices the Metal overhead (buffer allocation, command encoding,
/// GPU wake-up) dominates. We fall back to the scalar CPU path when
/// `M * K * N < METAL_THRESHOLD` (default 32768 = 32×32×32).

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
    // MSL source
    // -------------------------------------------------------------------------
    const MATMUL_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

kernel void matmul_f32(
    device const float* A  [[ buffer(0) ]],
    device const float* B  [[ buffer(1) ]],
    device       float* C  [[ buffer(2) ]],
    constant     uint&  M  [[ buffer(3) ]],
    constant     uint&  K  [[ buffer(4) ]],
    constant     uint&  N  [[ buffer(5) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++)
        acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
}
"#;

    // -------------------------------------------------------------------------
    // Cached Metal objects — one per thread
    // -------------------------------------------------------------------------
    pub struct MetalContext {
        pub device:   Retained<ProtocolObject<dyn MTLDevice>>,
        pub queue:    Retained<ProtocolObject<dyn MTLCommandQueue>>,
        pub pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
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

        // Compile the MSL kernel from source.
        let source = NSString::from_str(MATMUL_MSL);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .expect("Metal: MSL compilation failed");

        let fn_name = NSString::from_str("matmul_f32");
        let func: Retained<ProtocolObject<dyn MTLFunction>> = library
            .newFunctionWithName(&fn_name)
            .expect("Metal: function 'matmul_f32' not found");

        let pipeline = device
            .newComputePipelineStateWithFunction_error(&func)
            .expect("Metal: could not create compute pipeline");

        MetalContext { device, queue, pipeline }
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

            // Threadgroup 16×16; grid covers the full M×N output.
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
    }
}

#[cfg(feature = "metal")]
pub use inner::metal_matmul;
