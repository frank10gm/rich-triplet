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
    use std::cell::{OnceCell, RefCell};
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::ptr::NonNull;

    use crate::autograd2::{Mat, MatBf16, Q4KMat};

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
    // MSL source — Q4 fused dequantize + matmul
    //
    // Computes C[M,N] = A[M,K] @ dequant(B_q4[N,K])^T
    // where B_q4 is stored as block-wise 4-bit symmetric quantization:
    //   - packed[]: 2 nibbles per byte, row-major over [N, K]
    //   - scales[]: one f32 per BLOCK_SIZE elements (flat index)
    //
    // Each thread computes one output element C[row,col].
    // Uses threadgroup shared memory to cache a tile of A.
    // The B (Q4) tile is dequantized on-the-fly from nibbles.
    // -------------------------------------------------------------------------
    // Q4 GEMV kernel — optimised for M=1 (single-token decode).
    //
    // Grid: one threadgroup per output element (row i, col j).
    //   tgid.x = j  (which weight row / output neuron)
    //   tgid.y = i  (which input row; usually 0 for decode)
    // Threadgroup: TG_K threads, each handling K/TG_K elements of the dot product.
    // A simd_sum reduction collapses the partial sums to a single value.
    //
    // For M=1, K=2560, N=10240 (gate_proj):
    //   - 10240 threadgroups × 128 threads = 1.3M threads dispatched
    //   - Each thread reads K/128 = 20 nibble pairs → very low register pressure
    //   - Full GPU utilisation even for batch size 1
    // TG_K must equal SIMD_SIZE (32) so that simd_sum covers the whole threadgroup.
    // One threadgroup per output element; 32 threads stride over K.
    const MATMUL_Q4_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define Q4_BLOCK_SIZE 32u
#define TG_K          32u    // must equal simd_size (32 on all Apple GPUs)

kernel void matmul_q4_t(
    device const float* A       [[ buffer(0) ]],
    device const uchar* packed  [[ buffer(1) ]],
    device const float* scales  [[ buffer(2) ]],
    device       float* C       [[ buffer(3) ]],
    constant     uint&  M       [[ buffer(4) ]],
    constant     uint&  K       [[ buffer(5) ]],
    constant     uint&  N       [[ buffer(6) ]],
    uint3 tgid   [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid.x;   // output neuron (weight row)
    uint i = tgid.y;   // input row (0 for single-token decode)
    if (j >= N || i >= M) return;

    float acc = 0.0f;
    uint row_start = j * K;

    // Each of the 32 threads covers K/32 elements, strided
    for (uint p = lid; p < K; p += TG_K) {
        uint  flat   = row_start + p;
        uchar byte_v = packed[flat >> 1u];
        uchar nibble = (flat & 1u) == 0u ? (byte_v & 0x0Fu) : ((byte_v >> 4u) & 0x0Fu);
        int   q      = (nibble >= 8u) ? (int(nibble) - 16) : int(nibble);
        float w      = float(q) * scales[flat / Q4_BLOCK_SIZE];
        acc += A[i * K + p] * w;
    }

    // simd_sum reduces all 32 lanes (one full SIMD group = one threadgroup)
    acc = simd_sum(acc);

    if (lid == 0)
        C[i * N + j] = acc;
}
"#;

    // -------------------------------------------------------------------------
    // MSL source — Q4_K fused dequantize + GEMV
    //
    // Computes C[1,N] = A[1,K] @ dequant(Q4K[N,K])^T
    // where Q4K is stored as super-blocks of 256 elements (144 bytes each):
    //   [0..2)    f16 d       (super-block scale)
    //   [2..4)    f16 dmin    (super-block min)
    //   [4..16)   12 bytes    (packed 6-bit scale/min for 8 sub-blocks)
    //   [16..144) 128 bytes   (packed 4-bit nibbles, 2 per byte)
    //
    // One threadgroup (32 threads = one SIMD group) per output element.
    // Each thread strides over super-blocks, dequantizes on-the-fly.
    // -------------------------------------------------------------------------
    const GEMV_Q4K_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define Q4K_BLOCK_BYTES 144u
#define Q4K_BLOCK_ELEMS 256u
#define TG_K 32u

// Extract 6-bit scale and min for sub-block j (0..8) from 12-byte array.
inline float2 scale_min(device const uchar* sc, uint j) {
    float sv, mv;
    if (j < 4u) {
        sv = float(sc[j] & 0x3Fu);
        mv = float(sc[j + 4u] & 0x3Fu);
    } else {
        sv = float((sc[j + 4u] & 0x0Fu) | ((sc[j - 4u] >> 6u) << 4u));
        mv = float((sc[j + 4u] >> 4u)   | ((sc[j]      >> 6u) << 4u));
    }
    return float2(sv, mv);
}

kernel void gemv_q4k_t(
    device const float* A         [[ buffer(0) ]],
    device const uchar* blocks    [[ buffer(1) ]],
    device       float* C         [[ buffer(2) ]],
    constant     uint&  K         [[ buffer(3) ]],
    constant     uint&  N         [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    uint n_sb = K / Q4K_BLOCK_ELEMS;
    float acc = 0.0f;

    for (uint b = lid; b < n_sb; b += TG_K) {
        uint boff = (j * n_sb + b) * Q4K_BLOCK_BYTES;
        device const uchar* bp = blocks + boff;

        ushort d_bits    = ushort(bp[0]) | (ushort(bp[1]) << 8u);
        ushort dmin_bits = ushort(bp[2]) | (ushort(bp[3]) << 8u);
        float d    = float(as_type<half>(d_bits));
        float dmin = float(as_type<half>(dmin_bits));

        device const uchar* sc = bp + 4u;
        device const uchar* qs = bp + 16u;
        uint a_base = b * Q4K_BLOCK_ELEMS;

        for (uint chunk = 0u; chunk < 4u; chunk++) {
            float2 sm1 = scale_min(sc, chunk * 2u);
            float2 sm2 = scale_min(sc, chunk * 2u + 1u);
            float scale1 = d * sm1.x;  float min1 = dmin * sm1.y;
            float scale2 = d * sm2.x;  float min2 = dmin * sm2.y;

            device const uchar* q = qs + chunk * 32u;
            uint a_off = a_base + chunk * 64u;

            float dot_lo = 0.0f, dot_hi = 0.0f;
            float sum_lo = 0.0f, sum_hi = 0.0f;

            for (uint l = 0u; l < 32u; l++) {
                float a_lo = A[a_off + l];
                float a_hi = A[a_off + 32u + l];
                dot_lo += float(q[l] & 0x0Fu) * a_lo;
                dot_hi += float(q[l] >> 4u)   * a_hi;
                sum_lo += a_lo;
                sum_hi += a_hi;
            }

            acc += scale1 * dot_lo - min1 * sum_lo
                 + scale2 * dot_hi - min2 * sum_hi;
        }
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
}
"#;

    // -------------------------------------------------------------------------
    // MSL source — BF16 GEMV
    //
    // Computes C[1,N] = A[1,K] @ BF16[N,K]^T
    // where BF16 weights are stored as u16 (upper 16 bits of f32).
    // -------------------------------------------------------------------------
    const GEMV_BF16_MSL: &str = r#"
#include <metal_stdlib>
using namespace metal;

#define TG_K_BF16 32u

kernel void gemv_bf16_t(
    device const float*  A      [[ buffer(0) ]],
    device const ushort* W      [[ buffer(1) ]],
    device       float*  C      [[ buffer(2) ]],
    constant     uint&   K      [[ buffer(3) ]],
    constant     uint&   N      [[ buffer(4) ]],
    uint  tgid_x [[ threadgroup_position_in_grid ]],
    uint  lid    [[ thread_index_in_threadgroup ]])
{
    uint j = tgid_x;
    if (j >= N) return;

    float acc = 0.0f;
    uint row_start = j * K;

    for (uint p = lid; p < K; p += TG_K_BF16) {
        ushort bits = W[row_start + p];
        float w = as_type<float>(uint(bits) << 16u);
        acc += A[p] * w;
    }

    acc = simd_sum(acc);

    if (lid == 0u)
        C[j] = acc;
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
        pub pipeline_q4:       Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pub pipeline_q4k:      Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        pub pipeline_bf16:     Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        /// Persistent weight buffer cache. Key: (data_ptr, byte_len).
        pub weight_cache:      HashMap<(usize, usize), Retained<ProtocolObject<dyn MTLBuffer>>>,
        /// Pre-allocated scratch buffers for GEMV to avoid per-call allocation.
        /// activation: max K f32 values; output: max N f32 values; dims: 2 u32s.
        pub scratch_act:       Retained<ProtocolObject<dyn MTLBuffer>>,
        pub scratch_out:       Retained<ProtocolObject<dyn MTLBuffer>>,
        pub scratch_dims:      Retained<ProtocolObject<dyn MTLBuffer>>,
        pub scratch_act_cap:   usize,  // capacity in f32 elements
        pub scratch_out_cap:   usize,  // capacity in f32 elements
    }

    thread_local! {
        static METAL_CTX: OnceCell<RefCell<MetalContext>> = OnceCell::new();
    }

    fn init_context() -> MetalContext {
        let device = MTLCreateSystemDefaultDevice()
            .expect("Metal: no GPU device found");

        let queue = device
            .newCommandQueue()
            .expect("Metal: could not create command queue");

        let make_pipeline_from_src = |src: &str, name: &str| {
            let source = NSString::from_str(src);
            let library = device
                .newLibraryWithSource_options_error(&source, None)
                .unwrap_or_else(|_| panic!("Metal: MSL compilation failed for '{}'", name));
            let fn_name = NSString::from_str(name);
            let func: Retained<ProtocolObject<dyn MTLFunction>> = library
                .newFunctionWithName(&fn_name)
                .unwrap_or_else(|| panic!("Metal: function '{}' not found", name));
            device
                .newComputePipelineStateWithFunction_error(&func)
                .unwrap_or_else(|_| panic!("Metal: could not create pipeline for '{}'", name))
        };

        let pipeline         = make_pipeline_from_src(MATMUL_MSL, "matmul_tiled");
        let pipeline_batched = make_pipeline_from_src(MATMUL_MSL, "matmul_tiled_batched");
        let pipeline_q4      = make_pipeline_from_src(MATMUL_Q4_MSL, "matmul_q4_t");
        let pipeline_q4k     = make_pipeline_from_src(GEMV_Q4K_MSL, "gemv_q4k_t");
        let pipeline_bf16    = make_pipeline_from_src(GEMV_BF16_MSL, "gemv_bf16_t");

        // Pre-allocate scratch buffers for GEMV fast path.
        // These sizes cover all Gemma3-4b dimensions; they'll be grown if needed.
        let act_cap = 10240;  // max K (down_proj input)
        let out_cap = 262144; // max N (lm_head)
        let scratch_act = device.newBufferWithLength_options(
            act_cap * 4, MTLResourceOptions::StorageModeShared,
        ).expect("Metal: scratch_act allocation failed");
        let scratch_out = device.newBufferWithLength_options(
            out_cap * 4, MTLResourceOptions::StorageModeShared,
        ).expect("Metal: scratch_out allocation failed");
        // dims: [K, N] as two u32s
        let scratch_dims = device.newBufferWithLength_options(
            8, MTLResourceOptions::StorageModeShared,
        ).expect("Metal: scratch_dims allocation failed");

        MetalContext {
            device, queue, pipeline, pipeline_batched, pipeline_q4,
            pipeline_q4k, pipeline_bf16,
            weight_cache: HashMap::new(),
            scratch_act, scratch_out, scratch_dims,
            scratch_act_cap: act_cap, scratch_out_cap: out_cap,
        }
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

    /// Upload a &[u8] slice to a new MTLBuffer.
    fn upload_bytes(device: &ProtocolObject<dyn MTLDevice>, data: &[u8])
        -> Retained<ProtocolObject<dyn MTLBuffer>>
    {
        let ptr = NonNull::new(data.as_ptr() as *mut c_void).unwrap();
        unsafe {
            device
                .newBufferWithBytes_length_options(
                    ptr,
                    data.len(),
                    MTLResourceOptions::StorageModeShared,
                )
                .expect("Metal: buffer allocation failed")
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
            let ctx_cell = cell.get_or_init(|| RefCell::new(init_context()));
            let ctx = ctx_cell.borrow();

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
            let ctx_cell = cell.get_or_init(|| RefCell::new(init_context()));
            let ctx = ctx_cell.borrow();

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
    // Q4 matmul entry point
    // -------------------------------------------------------------------------

    /// GPU-accelerated `C = A @ dequant(Q4)^T`  ([M,K] × [N,K] → [M,N]).
    ///
    /// `q4` is stored row-major as [N, K] with nibble-packed weights.
    /// Falls back to CPU `matmul_q4_t` when the problem is too small.
    pub fn metal_matmul_q4_t(a: &Mat, q4: &crate::autograd2::Q4Mat) -> Mat {
        let (m, k, n) = (a.rows, a.cols, q4.rows);
        assert_eq!(k, q4.cols,
            "metal_matmul_q4_t: a.cols {} != q4.cols {}", k, q4.cols);

        if m * k * n < METAL_THRESHOLD {
            return q4.matmul_q4_t(a);
        }

        METAL_CTX.with(|cell| {
            let ctx_cell = cell.get_or_init(|| RefCell::new(init_context()));
            let ctx = ctx_cell.borrow();

            let buf_a      = upload(&ctx.device, &a.data);
            let buf_packed = upload_bytes(&ctx.device, &q4.packed);
            let buf_scales = upload(&ctx.device, &q4.scales);
            let buf_c      = alloc_output(&ctx.device, m * n);
            let buf_m      = upload_u32(&ctx.device, m as u32);
            let buf_k      = upload_u32(&ctx.device, k as u32);
            let buf_n      = upload_u32(&ctx.device, n as u32);

            let cmd_buf = ctx.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            encoder.setComputePipelineState(&ctx.pipeline_q4);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&buf_a),      0, 0);
                encoder.setBuffer_offset_atIndex(Some(&buf_packed),  0, 1);
                encoder.setBuffer_offset_atIndex(Some(&buf_scales),  0, 2);
                encoder.setBuffer_offset_atIndex(Some(&buf_c),       0, 3);
                encoder.setBuffer_offset_atIndex(Some(&buf_m),       0, 4);
                encoder.setBuffer_offset_atIndex(Some(&buf_k),       0, 5);
                encoder.setBuffer_offset_atIndex(Some(&buf_n),       0, 6);
            }

            // One threadgroup per output element; 32 threads = one SIMD group.
            let tg_size   = MTLSize { width: 32, height: 1, depth: 1 };
            let grid_size = MTLSize { width: n,  height: m, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            let ptr = buf_c.contents().as_ptr() as *const f32;
            let out_data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(ptr, m * n).to_vec()
            };
            Mat::new(out_data, m, n)
        })
    }

    // -------------------------------------------------------------------------
    // Q4K GEMV entry point
    // -------------------------------------------------------------------------

    /// Ensure scratch_act can hold `k` f32 values, growing if needed.
    fn ensure_scratch_act(ctx: &mut MetalContext, k: usize) {
        if k > ctx.scratch_act_cap {
            ctx.scratch_act = ctx.device.newBufferWithLength_options(
                k * 4, MTLResourceOptions::StorageModeShared,
            ).expect("Metal: scratch_act realloc failed");
            ctx.scratch_act_cap = k;
        }
    }

    /// Ensure scratch_out can hold `n` f32 values, growing if needed.
    fn ensure_scratch_out(ctx: &mut MetalContext, n: usize) {
        if n > ctx.scratch_out_cap {
            ctx.scratch_out = ctx.device.newBufferWithLength_options(
                n * 4, MTLResourceOptions::StorageModeShared,
            ).expect("Metal: scratch_out realloc failed");
            ctx.scratch_out_cap = n;
        }
    }

    /// GPU-accelerated Q4K GEMV: `C[1,N] = A[1,K] @ dequant(Q4K[N,K])^T`.
    ///
    /// Weight blocks are cached as persistent Metal buffers on first call.
    /// Uses pre-allocated scratch buffers to avoid per-call allocation.
    pub fn metal_gemv_q4k_t(a: &Mat, q4k: &Q4KMat) -> Mat {
        debug_assert_eq!(a.rows, 1, "metal_gemv_q4k_t: only M=1 supported");
        let k = a.cols;
        let n = q4k.rows;
        assert_eq!(k, q4k.cols,
            "metal_gemv_q4k_t: a.cols {} != q4k.cols {}", k, q4k.cols);

        METAL_CTX.with(|cell| {
            let ctx_cell = cell.get_or_init(|| RefCell::new(init_context()));
            let mut ctx = ctx_cell.borrow_mut();

            // Ensure weight buffer is in cache (upload once, reuse forever).
            let cache_key = (q4k.blocks.as_ptr() as usize, q4k.blocks.len());
            if !ctx.weight_cache.contains_key(&cache_key) {
                let buf = upload_bytes(&ctx.device, &q4k.blocks);
                ctx.weight_cache.insert(cache_key, buf);
            }

            // Write activation into pre-allocated scratch buffer (memcpy, no alloc).
            ensure_scratch_act(&mut ctx, k);
            ensure_scratch_out(&mut ctx, n);
            unsafe {
                let act_ptr = ctx.scratch_act.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(a.data.as_ptr(), act_ptr, k);
                let dims_ptr = ctx.scratch_dims.contents().as_ptr() as *mut u32;
                *dims_ptr = k as u32;
                *dims_ptr.add(1) = n as u32;
            }

            let cmd_buf = ctx.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            encoder.setComputePipelineState(&ctx.pipeline_q4k);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_act),              0, 0);
                encoder.setBuffer_offset_atIndex(Some(&ctx.weight_cache[&cache_key]), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_out),              0, 2);
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_dims),             0, 3);  // K
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_dims),             4, 4);  // N (offset 4 bytes)
            }

            let tg_size   = MTLSize { width: 32, height: 1, depth: 1 };
            let grid_size = MTLSize { width: n,  height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            // Read output directly from scratch_out (no allocation, just copy to Vec).
            let ptr = ctx.scratch_out.contents().as_ptr() as *const f32;
            let out_data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(ptr, n).to_vec()
            };
            Mat::new(out_data, 1, n)
        })
    }

    // -------------------------------------------------------------------------
    // BF16 GEMV entry point
    // -------------------------------------------------------------------------

    /// Upload a &[u16] slice as bytes to a new MTLBuffer.
    fn upload_u16(device: &ProtocolObject<dyn MTLDevice>, data: &[u16])
        -> Retained<ProtocolObject<dyn MTLBuffer>>
    {
        let byte_len = data.len() * 2;
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

    /// GPU-accelerated BF16 GEMV: `C[1,N] = A[1,K] @ BF16[N,K]^T`.
    ///
    /// Weight data is cached as a persistent Metal buffer on first call.
    /// Uses pre-allocated scratch buffers to avoid per-call allocation.
    pub fn metal_gemv_bf16_t(a: &Mat, bf16: &MatBf16) -> Mat {
        debug_assert_eq!(a.rows, 1, "metal_gemv_bf16_t: only M=1 supported");
        let k = a.cols;
        let n = bf16.rows;
        assert_eq!(k, bf16.cols,
            "metal_gemv_bf16_t: a.cols {} != bf16.cols {}", k, bf16.cols);

        METAL_CTX.with(|cell| {
            let ctx_cell = cell.get_or_init(|| RefCell::new(init_context()));
            let mut ctx = ctx_cell.borrow_mut();

            // Ensure weight buffer is in cache.
            let cache_key = (bf16.data.as_ptr() as usize, bf16.data.len() * 2);
            if !ctx.weight_cache.contains_key(&cache_key) {
                let buf = upload_u16(&ctx.device, &bf16.data);
                ctx.weight_cache.insert(cache_key, buf);
            }

            ensure_scratch_act(&mut ctx, k);
            ensure_scratch_out(&mut ctx, n);
            unsafe {
                let act_ptr = ctx.scratch_act.contents().as_ptr() as *mut f32;
                std::ptr::copy_nonoverlapping(a.data.as_ptr(), act_ptr, k);
                let dims_ptr = ctx.scratch_dims.contents().as_ptr() as *mut u32;
                *dims_ptr = k as u32;
                *dims_ptr.add(1) = n as u32;
            }

            let cmd_buf = ctx.queue
                .commandBuffer()
                .expect("Metal: commandBuffer() failed");
            let encoder = cmd_buf
                .computeCommandEncoder()
                .expect("Metal: computeCommandEncoder() failed");

            encoder.setComputePipelineState(&ctx.pipeline_bf16);

            unsafe {
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_act),              0, 0);
                encoder.setBuffer_offset_atIndex(Some(&ctx.weight_cache[&cache_key]), 0, 1);
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_out),              0, 2);
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_dims),             0, 3);  // K
                encoder.setBuffer_offset_atIndex(Some(&ctx.scratch_dims),             4, 4);  // N (offset 4 bytes)
            }

            let tg_size   = MTLSize { width: 32, height: 1, depth: 1 };
            let grid_size = MTLSize { width: n,  height: 1, depth: 1 };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(grid_size, tg_size);
            encoder.endEncoding();

            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            let ptr = ctx.scratch_out.contents().as_ptr() as *const f32;
            let out_data: Vec<f32> = unsafe {
                std::slice::from_raw_parts(ptr, n).to_vec()
            };
            Mat::new(out_data, 1, n)
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

        // --- Q4K GEMV tests ---

        /// Convert f32 to IEEE 754 f16 bits (truncation, good enough for tests).
        fn f32_to_f16_bits(val: f32) -> u16 {
            let bits = val.to_bits();
            let sign = (bits >> 31) & 1;
            let exp  = ((bits >> 23) & 0xFF) as i32;
            let mant = bits & 0x7FFFFF;
            if exp == 0 { return (sign << 15) as u16; }
            let new_exp = exp - 127 + 15;
            if new_exp <= 0 { return (sign << 15) as u16; }
            if new_exp >= 31 { return ((sign << 15) | 0x7C00) as u16; }
            let new_mant = mant >> 13;
            ((sign << 15) | (new_exp as u32) << 10 | new_mant) as u16
        }

        /// Build a Q4KMat with known values for testing.
        fn make_test_q4k(n_rows: usize, k: usize) -> Q4KMat {
            assert_eq!(k % 256, 0);
            let n_blocks_per_row = k / 256;
            let total_blocks = n_rows * n_blocks_per_row;
            let mut blocks = vec![0u8; total_blocks * 144];

            for j in 0..n_rows {
                for b in 0..n_blocks_per_row {
                    let boff = (j * n_blocks_per_row + b) * 144;
                    // d = 0.1, dmin = 0.05 as f16
                    let d_bits = f32_to_f16_bits(0.1);
                    let dmin_bits = f32_to_f16_bits(0.05);
                    blocks[boff]   = d_bits as u8;
                    blocks[boff+1] = (d_bits >> 8) as u8;
                    blocks[boff+2] = dmin_bits as u8;
                    blocks[boff+3] = (dmin_bits >> 8) as u8;

                    // Scales: sub-blocks 0-3: scale=2, min=1
                    for i in 0..4 { blocks[boff + 4 + i] = 2; }
                    for i in 4..8 { blocks[boff + 4 + i] = 1; }
                    // Sub-blocks 4-7: set to 0
                    for i in 8..12 { blocks[boff + 4 + i] = 0; }

                    // Nibbles: all 0x55 (low=5, high=5)
                    for i in 0..128 {
                        blocks[boff + 16 + i] = 0x55;
                    }
                }
            }

            Q4KMat { rows: n_rows, cols: k, blocks }
        }

        #[test]
        fn test_metal_gemv_q4k_matches_cpu() {
            // Small Q4K: 4 rows, K=256 (one super-block per row)
            let k = 256;
            let n = 4;
            let q4k = make_test_q4k(n, k);
            let a = Mat::from_fn(1, k, |_, c| (c as f32 + 1.0) * 0.01);

            // CPU reference: dequantize each row and compute dot product
            let mut expected = Mat::zeros(1, n);
            for j in 0..n {
                let mut row = vec![0.0f32; k];
                q4k.dequantize_row_into(j, &mut row);
                let dot: f32 = row.iter().zip(a.data.iter()).map(|(w, x)| w * x).sum();
                expected.data[j] = dot;
            }

            let got = metal_gemv_q4k_t(&a, &q4k);
            assert_eq!(got.rows, 1);
            assert_eq!(got.cols, n);
            for j in 0..n {
                let diff = (got.data[j] - expected.data[j]).abs();
                let rel = diff / (expected.data[j].abs() + 1e-6);
                assert!(rel < 1e-3,
                    "Q4K GEMV mismatch at col {j}: got {:.6} expected {:.6} (rel {:.6})",
                    got.data[j], expected.data[j], rel);
            }
        }

        #[test]
        fn test_metal_gemv_q4k_larger() {
            // Realistic-ish: 64 rows, K=512
            let k = 512;
            let n = 64;
            let q4k = make_test_q4k(n, k);
            let a = Mat::from_fn(1, k, |_, c| ((c % 7) as f32 - 3.0) * 0.1);

            let mut expected = Mat::zeros(1, n);
            for j in 0..n {
                let mut row = vec![0.0f32; k];
                q4k.dequantize_row_into(j, &mut row);
                let dot: f32 = row.iter().zip(a.data.iter()).map(|(w, x)| w * x).sum();
                expected.data[j] = dot;
            }

            let got = metal_gemv_q4k_t(&a, &q4k);
            for j in 0..n {
                let diff = (got.data[j] - expected.data[j]).abs();
                let rel = diff / (expected.data[j].abs() + 1e-6);
                assert!(rel < 1e-3,
                    "Q4K GEMV mismatch at col {j}: got {:.6} expected {:.6}",
                    got.data[j], expected.data[j]);
            }
        }

        #[test]
        fn test_metal_gemv_q4k_cache_reuse() {
            // Call twice with same Q4KMat — second call uses cached buffer
            let k = 256;
            let n = 4;
            let q4k = make_test_q4k(n, k);
            let a1 = Mat::from_fn(1, k, |_, c| c as f32 * 0.01);
            let a2 = Mat::from_fn(1, k, |_, c| (k - c) as f32 * 0.01);

            let r1 = metal_gemv_q4k_t(&a1, &q4k);
            let r2 = metal_gemv_q4k_t(&a2, &q4k);

            // Results should differ (different activations)
            assert!(r1.data[0] != r2.data[0] || r1.data[1] != r2.data[1],
                "cache reuse: results should differ for different activations");

            // Both should match CPU reference
            for (a, r) in [(&a1, &r1), (&a2, &r2)] {
                for j in 0..n {
                    let mut row = vec![0.0f32; k];
                    q4k.dequantize_row_into(j, &mut row);
                    let dot: f32 = row.iter().zip(a.data.iter()).map(|(w, x)| w * x).sum();
                    let diff = (r.data[j] - dot).abs();
                    assert!(diff / (dot.abs() + 1e-6) < 1e-3);
                }
            }
        }

        // --- BF16 GEMV tests ---

        #[test]
        fn test_metal_gemv_bf16_matches_cpu() {
            let k = 64;
            let n = 8;
            // Create BF16 weight matrix [N, K]
            let w_f32: Vec<f32> = (0..n*k).map(|i| ((i % 17) as f32 - 8.0) * 0.1).collect();
            let w_bf16: Vec<u16> = w_f32.iter().map(|&v| MatBf16::f32_to_bf16(v)).collect();
            let bf16 = MatBf16 {
                data: std::sync::Arc::new(w_bf16),
                rows: n,
                cols: k,
            };
            let a = Mat::from_fn(1, k, |_, c| (c as f32 + 1.0) * 0.05);

            // CPU reference: dequant BF16 to f32, dot product
            let mut expected = Mat::zeros(1, n);
            for j in 0..n {
                let mut dot = 0.0f32;
                for p in 0..k {
                    let w = MatBf16::bf16_to_f32(bf16.data[j * k + p]);
                    dot += w * a.data[p];
                }
                expected.data[j] = dot;
            }

            let got = metal_gemv_bf16_t(&a, &bf16);
            assert_eq!(got.rows, 1);
            assert_eq!(got.cols, n);
            for j in 0..n {
                let diff = (got.data[j] - expected.data[j]).abs();
                assert!(diff < 1e-2,
                    "BF16 GEMV mismatch at col {j}: got {:.6} expected {:.6}",
                    got.data[j], expected.data[j]);
            }
        }

        #[test]
        fn test_metal_gemv_bf16_larger() {
            let k = 256;
            let n = 128;
            let w_f32: Vec<f32> = (0..n*k).map(|i| ((i % 31) as f32 - 15.0) * 0.01).collect();
            let w_bf16: Vec<u16> = w_f32.iter().map(|&v| MatBf16::f32_to_bf16(v)).collect();
            let bf16 = MatBf16 {
                data: std::sync::Arc::new(w_bf16),
                rows: n,
                cols: k,
            };
            let a = Mat::from_fn(1, k, |_, c| ((c % 13) as f32 - 6.0) * 0.1);

            let mut expected = Mat::zeros(1, n);
            for j in 0..n {
                let mut dot = 0.0f32;
                for p in 0..k {
                    let w = MatBf16::bf16_to_f32(bf16.data[j * k + p]);
                    dot += w * a.data[p];
                }
                expected.data[j] = dot;
            }

            let got = metal_gemv_bf16_t(&a, &bf16);
            for j in 0..n {
                let diff = (got.data[j] - expected.data[j]).abs();
                assert!(diff < 1e-2,
                    "BF16 GEMV mismatch at col {j}: got {:.6} expected {:.6}",
                    got.data[j], expected.data[j]);
            }
        }
    }
}

#[cfg(feature = "metal")]
pub use inner::{metal_matmul, metal_matmul_batched, metal_matmul_q4_t,
                metal_gemv_q4k_t, metal_gemv_bf16_t};
