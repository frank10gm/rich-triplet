/// GGUF file format reader.
///
/// Implements a minimal subset of the GGUF spec (v2/v3) sufficient to load
/// Gemma 3 4B Q4_0 weights from HuggingFace GGUF files.
///
/// ## GGUF binary layout
///
/// ```text
/// [magic: 4 bytes "GGUF"]
/// [version: u32le]
/// [n_tensors: u64le]
/// [n_kv: u64le]
/// [kv pairs: n_kv × (key_str, value_type, value)]
/// [tensor_infos: n_tensors × (name_str, n_dims, dims[], type, offset)]
/// [padding to alignment]
/// [tensor data: raw bytes at tensor_info offsets]
/// ```
///
/// ## Q4_0 block format (18 bytes per 32 elements)
///
/// ```text
/// [scale: f16le (2 bytes)]
/// [nibbles: 16 bytes — 32 × 4-bit unsigned values, low nibble first]
/// ```
///
/// Dequant: `value[i] = scale * (nibble[i] - 8)`
///
/// ## GGUF metadata value types
/// 0=u8, 1=i8, 2=u16, 3=i16, 4=u32, 5=i32, 6=f32, 7=bool,
/// 8=string, 9=array, 10=u64, 11=i64, 12=f64

use std::collections::HashMap;
use std::io::{self, Read, Seek, SeekFrom};
use crate::autograd2::{Q4KMat, Q4Mat};

// ---------------------------------------------------------------------------
// Tensor type enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgufType {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2K  = 10,
    Q3K  = 11,
    Q4K  = 12,
    Q5K  = 13,
    Q6K  = 14,
    Q8K  = 15,
    Iq2Xxs = 16,
    Iq2Xs  = 17,
    Iq3Xxs = 18,
    Iq1S   = 19,
    Iq4Nl  = 20,
    Iq3S   = 21,
    Iq2S   = 22,
    Iq4Xs  = 23,
    I8   = 24,
    I16  = 25,
    I32  = 26,
    I64  = 27,
    F64  = 28,
    Iq1M = 29,
    Bf16 = 30,
    Unknown,
}

impl GgufType {
    fn from_u32(v: u32) -> Self {
        match v {
            0  => Self::F32,
            1  => Self::F16,
            2  => Self::Q4_0,
            3  => Self::Q4_1,
            6  => Self::Q5_0,
            7  => Self::Q5_1,
            8  => Self::Q8_0,
            9  => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            16 => Self::Iq2Xxs,
            17 => Self::Iq2Xs,
            18 => Self::Iq3Xxs,
            19 => Self::Iq1S,
            20 => Self::Iq4Nl,
            21 => Self::Iq3S,
            22 => Self::Iq2S,
            23 => Self::Iq4Xs,
            24 => Self::I8,
            25 => Self::I16,
            26 => Self::I32,
            27 => Self::I64,
            28 => Self::F64,
            29 => Self::Iq1M,
            30 => Self::Bf16,
            _  => Self::Unknown,
        }
    }

    /// Bytes per block and elements per block for computing total tensor bytes.
    /// Returns `(bytes_per_block, elems_per_block)`.
    pub fn block_info(self) -> Option<(usize, usize)> {
        match self {
            Self::F32  => Some((4,  1)),
            Self::F16  => Some((2,  1)),
            Self::Bf16 => Some((2,  1)),
            Self::Q4_0 => Some((18, 32)),  // 2 bytes f16 scale + 16 bytes nibbles
            Self::Q8_0 => Some((34, 32)),  // 4 bytes f32 scale + 32 bytes i8
            Self::Q4K  => Some((144, 256)), // Q4_K super-block
            Self::Q6K  => Some((210, 256)), // Q6_K super-block
            _          => None,
        }
    }

    /// Total bytes for `n_elements` elements of this type.
    pub fn byte_size(self, n_elements: usize) -> Option<usize> {
        let (bpb, epb) = self.block_info()?;
        let n_blocks = (n_elements + epb - 1) / epb;
        Some(n_blocks * bpb)
    }
}

// ---------------------------------------------------------------------------
// Tensor info (from the header section)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct GgufTensorInfo {
    pub name:        String,
    pub shape:       Vec<usize>,   // e.g. [out, in] for weight matrices
    pub gguf_type:   GgufType,
    pub data_offset: u64,          // byte offset from start of tensor data section
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

pub struct GgufFile {
    pub metadata:    HashMap<String, GgufMetaValue>,
    pub tensor_info: Vec<GgufTensorInfo>,
    /// Byte offset in the file where the tensor data section starts.
    pub data_start:  u64,
    pub file_path:   String,
}

impl GgufFile {
    /// Open and parse the header (metadata + tensor info) of a GGUF file.
    /// Does NOT read tensor data into memory.
    pub fn open(path: &str) -> io::Result<Self> {
        let mut f = std::fs::File::open(path)?;

        // --- magic ---
        let mut magic = [0u8; 4];
        f.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }

        // --- version ---
        let version = read_u32le(&mut f)?;
        if version < 2 || version > 3 {
            return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unsupported GGUF version {version}")));
        }

        // --- counts ---
        let n_tensors = read_u64le(&mut f)? as usize;
        let n_kv      = read_u64le(&mut f)? as usize;

        // --- metadata key-value pairs ---
        let mut metadata = HashMap::new();
        for _ in 0..n_kv {
            let key = read_string(&mut f)?;
            let val = GgufMetaValue::read(&mut f)?;
            metadata.insert(key, val);
        }

        // --- tensor info ---
        let mut tensor_info = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name   = read_string(&mut f)?;
            let n_dims = read_u32le(&mut f)? as usize;
            let mut shape = vec![0usize; n_dims];
            for d in &mut shape {
                *d = read_u64le(&mut f)? as usize;
            }
            let type_id  = read_u32le(&mut f)?;
            let offset   = read_u64le(&mut f)?;
            tensor_info.push(GgufTensorInfo {
                name,
                shape,
                gguf_type: GgufType::from_u32(type_id),
                data_offset: offset,
            });
        }

        // Alignment for data section (default 32 if not set)
        let alignment = metadata.get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32) as u64;

        // Tensor data starts after the header, padded to alignment
        let current_pos = f.seek(SeekFrom::Current(0))?;
        let data_start = align_up(current_pos, alignment);

        Ok(GgufFile {
            metadata,
            tensor_info,
            data_start,
            file_path: path.to_string(),
        })
    }

    /// Read raw bytes for a tensor by its header index.
    pub fn read_tensor_bytes(&self, idx: usize) -> io::Result<Vec<u8>> {
        let info = &self.tensor_info[idx];
        let n_elem = info.n_elements();
        let n_bytes = info.gguf_type.byte_size(n_elem)
            .ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported,
                format!("unsupported gguf type {:?} for tensor {}", info.gguf_type, info.name)))?;

        let mut f = std::fs::File::open(&self.file_path)?;
        f.seek(SeekFrom::Start(self.data_start + info.data_offset))?;
        let mut buf = vec![0u8; n_bytes];
        f.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Look up a tensor by name. Returns its index in `tensor_info`.
    pub fn find_tensor(&self, name: &str) -> Option<usize> {
        self.tensor_info.iter().position(|t| t.name == name)
    }

    /// Decode a Q4_0 tensor into our internal Q4Mat format.
    ///
    /// ## GGUF Q4_0 block layout (18 bytes for 32 elements):
    ///   [scale: f16le, 2 bytes]
    ///   [qs: 16 bytes]
    ///
    /// Nibble packing in qs is SPLIT (not interleaved):
    ///   qs[k] low  nibble = element k        (k = 0..15)
    ///   qs[k] high nibble = element k + 16   (k = 0..15)
    ///
    /// Nibble values are unsigned 0..15; actual weight = (nibble - 8) * scale.
    ///
    /// ## Our Q4Mat layout (interleaved nibbles):
    ///   packed[k] low  nibble = element 2k
    ///   packed[k] high nibble = element 2k+1
    ///   values are 4-bit two's complement; actual weight = nibble_signed * scale
    ///   where nibble_signed = if nibble >= 8 { nibble - 16 } else { nibble }
    ///
    /// ## Conversion:
    ///   1. Reorder: split layout → interleaved layout
    ///   2. Remap nibble: v (0..15) → (v - 8) & 0x0F (our two's complement encoding)
    ///   3. Scale: use GGUF scale directly. GGUF dequant = (nibble-8)*scale.
    ///             After nibble remapping to two's complement, our dequant = signed*scale.
    ///             Both give the same result, so no scale adjustment needed.
    ///
    /// ## Shape:
    ///   GGUF stores weight matrices as [in_features, out_features] (transposed vs HuggingFace).
    ///   Our matmul uses `fused_linear` which does `input @ weight.T`, so weight must be
    ///   [out_features, in_features]. We transpose on load:
    ///   GGUF shape[0] = in_features = our cols, shape[1] = out_features = our rows.
    pub fn decode_q4_0_to_q4mat(&self, idx: usize) -> io::Result<Q4Mat> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q4_0,
            "decode_q4_0_to_q4mat called on non-Q4_0 tensor");

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem = info.n_elements();
        let n_blocks = (n_elem + 31) / 32;

        // GGUF shape is [in_features, out_features]; we store [out_features, in_features].
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),  // transpose
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unexpected shape rank {} for {}", info.shape.len(), info.name))),
        };

        let mut scales = Vec::with_capacity(n_blocks);
        let n_packed = (n_elem + 1) / 2;
        let mut packed = vec![0u8; n_packed];

        // Helper: write nibble `v` for absolute element index `elem` into our packed array.
        // Our layout: packed[elem/2], low nibble if even, high nibble if odd.
        let set_nibble = |packed: &mut Vec<u8>, elem: usize, v: u8| {
            let byte_idx = elem / 2;
            if elem % 2 == 0 {
                packed[byte_idx] = (packed[byte_idx] & 0xF0) | (v & 0x0F);
            } else {
                packed[byte_idx] = (packed[byte_idx] & 0x0F) | ((v & 0x0F) << 4);
            }
        };

        for b in 0..n_blocks {
            let block_off = b * 18;

            // f16 scale → f32.
            // GGUF dequant: (nibble - 8) * scale. After nibble remapping to two's complement,
            // our dequant does: signed_nibble * scale. Both are equivalent, so store as-is.
            let scale_bits = u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]);
            scales.push(f16_to_f32(scale_bits));

            let nibble_off = block_off + 2;
            let start_elem = b * 32;
            let end_elem   = (start_elem + 32).min(n_elem);

            // GGUF split layout:
            //   qs[k] low  nibble → element (start + k)       for k in 0..16
            //   qs[k] high nibble → element (start + k + 16)  for k in 0..16
            for k in 0..16 {
                let src = bytes[nibble_off + k];

                // Low nibble → element start + k
                let e0 = start_elem + k;
                if e0 < end_elem {
                    let v = src & 0x0F;
                    set_nibble(&mut packed, e0, v.wrapping_sub(8) & 0x0F);
                }

                // High nibble → element start + k + 16
                let e1 = start_elem + k + 16;
                if e1 < end_elem {
                    let v = (src >> 4) & 0x0F;
                    set_nibble(&mut packed, e1, v.wrapping_sub(8) & 0x0F);
                }
            }
        }

        Ok(Q4Mat { rows, cols, packed, scales })
    }

    /// Load a Q4_K tensor into a `Q4KMat` by copying the raw block bytes.
    ///
    /// The GGUF block layout (144 bytes/256 elements) is preserved verbatim;
    /// dequantization happens on-the-fly during matmul via
    /// `Q4KMat::dequantize_row_into`.
    ///
    /// Shape convention: GGUF stores [in_features, out_features]; we flip to
    /// [out_features, in_features] to match our row-major weight layout.
    pub fn decode_q4k_to_q4kmat(&self, idx: usize) -> io::Result<Q4KMat> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q4K,
            "decode_q4k_to_q4kmat called on non-Q4K tensor");

        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),  // transpose: GGUF [cols, rows] → ours [rows, cols]
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unexpected shape rank {} for {}", info.shape.len(), info.name))),
        };

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem   = rows * cols;
        let n_blocks = (n_elem + 255) / 256;
        assert_eq!(
            bytes.len(), n_blocks * 144,
            "Q4K tensor {} expected {} bytes, got {}", info.name, n_blocks * 144, bytes.len()
        );

        Ok(Q4KMat { rows, cols, blocks: bytes })
    }

    /// Decode an F32 tensor into a Vec<f32>.
    pub fn decode_f32(&self, idx: usize) -> io::Result<Vec<f32>> {
        let bytes = self.read_tensor_bytes(idx)?;
        let n = bytes.len() / 4;
        let mut out = vec![0.0f32; n];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
        Ok(out)
    }

    /// Decode a BF16 tensor into a Vec<u16>.
    pub fn decode_bf16(&self, idx: usize) -> io::Result<Vec<u16>> {
        let bytes = self.read_tensor_bytes(idx)?;
        let n = bytes.len() / 2;
        let mut out = vec![0u16; n];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
        }
        Ok(out)
    }

    /// Decode an F16 tensor into a Vec<f32> (convert on load).
    pub fn decode_f16_to_f32(&self, idx: usize) -> io::Result<Vec<f32>> {
        let bytes = self.read_tensor_bytes(idx)?;
        let n = bytes.len() / 2;
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let bits = u16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
            out.push(f16_to_f32(bits));
        }
        Ok(out)
    }

    /// Decode a Q4_0 tensor directly into a flat Vec<f32>.
    ///
    /// Elements are dequantized in GGUF memory order (row-major with
    /// GGUF's dimension convention, i.e. shape[0] is the fastest dimension).
    /// The output Vec has length == n_elements.
    ///
    /// Use this for tensors where you need plain f32 values rather than
    /// the packed Q4Mat format (e.g. embedding tables).
    pub fn decode_q4_0_to_f32(&self, idx: usize) -> io::Result<Vec<f32>> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q4_0,
            "decode_q4_0_to_f32 called on non-Q4_0 tensor");

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem = info.n_elements();
        let n_blocks = (n_elem + 31) / 32;
        let mut out = vec![0.0f32; n_elem];

        for b in 0..n_blocks {
            let block_off = b * 18;
            let scale_bits = u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]);
            let scale = f16_to_f32(scale_bits);

            let nibble_off = block_off + 2;
            let start_elem = b * 32;
            let end_elem = (start_elem + 32).min(n_elem);

            // GGUF split layout: qs[k] low=elem(start+k), high=elem(start+k+16)
            for k in 0..16 {
                let src = bytes[nibble_off + k];

                let e0 = start_elem + k;
                if e0 < end_elem {
                    let v = (src & 0x0F) as i8 - 8;
                    out[e0] = v as f32 * scale;
                }

                let e1 = start_elem + k + 16;
                if e1 < end_elem {
                    let v = ((src >> 4) & 0x0F) as i8 - 8;
                    out[e1] = v as f32 * scale;
                }
            }
        }

        Ok(out)
    }

    /// Decode a Q6_K tensor into a flat Vec<f32>.
    ///
    /// ## Q6_K block layout (210 bytes per 256 elements):
    ///   ql[128]    — lower 4 bits of each 6-bit quant (bytes 0–127)
    ///   qh[64]     — upper 2 bits of each 6-bit quant (bytes 128–191)
    ///   scales[16] — int8 scales, one per 16-element group (bytes 192–207)
    ///   d          — fp16 super-block scale (bytes 208–209)
    ///
    /// ## Memory layout (from llama.cpp dequantize_row_q6_K):
    /// The 256 elements are processed in 2 halves of 128 each.
    /// Within each half, elements are interleaved across 4 lanes of 32:
    ///   For l in 0..32:
    ///     q1 = (ql[l]    & 0xF) | (((qh[l] >> 0) & 3) << 4) - 32  → y[l]
    ///     q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32  → y[l+32]
    ///     q3 = (ql[l]    >> 4)  | (((qh[l] >> 4) & 3) << 4) - 32  → y[l+64]
    ///     q4 = (ql[l+32] >> 4)  | (((qh[l] >> 6) & 3) << 4) - 32  → y[l+96]
    /// Then advance ql by 64, qh by 32, y by 128 for second half.
    pub fn decode_q6k_to_f32(&self, idx: usize) -> io::Result<Vec<f32>> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q6K,
            "decode_q6k_to_f32 called on non-Q6K tensor");

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem = info.n_elements();
        let n_blocks = (n_elem + 255) / 256;
        let mut out = vec![0.0f32; n_elem];

        for b in 0..n_blocks {
            let block_off = b * 210;
            let ql_all = &bytes[block_off..block_off + 128];
            let qh_all = &bytes[block_off + 128..block_off + 192];
            let sc_all = &bytes[block_off + 192..block_off + 208]; // 16 int8 scales
            let d_bits = u16::from_le_bytes([bytes[block_off + 208], bytes[block_off + 209]]);
            let d = f16_to_f32(d_bits);

            let base = b * 256;
            let end  = (base + 256).min(n_elem);

            // Two halves of 128 elements each (j=0: first half, j=1: second half)
            for j in 0..2usize {
                let ql = &ql_all[j * 64..(j + 1) * 64]; // 64 bytes for this half
                let qh = &qh_all[j * 32..(j + 1) * 32]; // 32 bytes for this half
                let sc = &sc_all[j * 8..(j + 1) * 8];   // 8 scales for this half
                let y_base = base + j * 128;

                // For each l in 0..32, produce 4 output values at l, l+32, l+64, l+96
                for l in 0..32usize {
                    let is = l / 16; // 0 for l=0..15, 1 for l=16..31
                    let q1 = ((ql[l]      & 0x0F) | (((qh[l] >> 0) & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[l]      >>    4) | (((qh[l] >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[l + 32] >>    4) | (((qh[l] >> 6) & 3) << 4)) as i32 - 32;

                    // scale pairs: is=0 → sc[0],sc[2],sc[4],sc[6]; is=1 → sc[1],sc[3],sc[5],sc[7]
                    let s0 = d * sc[is]     as i8 as f32;
                    let s1 = d * sc[is + 2] as i8 as f32;
                    let s2 = d * sc[is + 4] as i8 as f32;
                    let s3 = d * sc[is + 6] as i8 as f32;

                    if y_base + l      < end { out[y_base + l]      = s0 * q1 as f32; }
                    if y_base + l + 32 < end { out[y_base + l + 32] = s1 * q2 as f32; }
                    if y_base + l + 64 < end { out[y_base + l + 64] = s2 * q3 as f32; }
                    if y_base + l + 96 < end { out[y_base + l + 96] = s3 * q4 as f32; }
                }
            }
        }

        Ok(out)
    }

    /// Decode a Q4_K tensor into a flat Vec<f32>.
    ///
    /// ## Q4_K block layout (144 bytes per 256 elements):
    ///   d          — fp16 super-block scale for scales (bytes 0–1)
    ///   dmin       — fp16 super-block scale for mins (bytes 2–3)
    ///   scales[12] — packed 6-bit scales/mins for 8 sub-groups (bytes 4–15)
    ///   qs[128]    — 4-bit quants, 2 per byte (bytes 16–143)
    ///
    /// The 256 elements are split into 4 chunks of 64. Each chunk uses:
    ///   - low nibbles of 32 bytes → 32 elements with scale/min pair `is`
    ///   - high nibbles of same 32 bytes → 32 elements with scale/min pair `is+1`
    ///
    /// get_scale_min_k4 extracts 6-bit scale/min for pair j from scales[12]:
    ///   if j < 4: scale = sc[j] & 0x3F;  min = sc[j+4] & 0x3F
    ///   else:     scale = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
    ///             min   = (sc[j+4] >> 4)   | ((sc[j+0] >> 6) << 4)
    pub fn decode_q4k_to_f32(&self, idx: usize) -> io::Result<Vec<f32>> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q4K,
            "decode_q4k_to_f32 called on non-Q4K tensor");

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem = info.n_elements();
        let n_blocks = (n_elem + 255) / 256;
        let mut out = vec![0.0f32; n_elem];

        // get_scale_min_k4: extract 6-bit scale and min for pair j (0..8)
        let get_scale_min = |sc: &[u8], j: usize| -> (f32, f32) {
            let (sc_val, min_val) = if j < 4 {
                (sc[j] & 0x3F, sc[j + 4] & 0x3F)
            } else {
                (
                    (sc[j + 4] & 0x0F) | ((sc[j - 4] >> 6) << 4),
                    (sc[j + 4] >> 4)   | ((sc[j + 0] >> 6) << 4),
                )
            };
            (sc_val as f32, min_val as f32)
        };

        for b in 0..n_blocks {
            let block_off = b * 144;
            let d_bits    = u16::from_le_bytes([bytes[block_off],     bytes[block_off + 1]]);
            let dmin_bits = u16::from_le_bytes([bytes[block_off + 2], bytes[block_off + 3]]);
            let d    = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);
            let sc   = &bytes[block_off + 4..block_off + 16];
            let qs   = &bytes[block_off + 16..block_off + 144];

            let base = b * 256;
            let end  = (base + 256).min(n_elem);

            // Process 4 chunks of 64 elements. Each chunk advances qs by 32 bytes
            // and uses 2 scale/min pairs.
            let mut q_off = 0usize; // offset into qs[]
            let mut is    = 0usize; // scale/min pair index
            let mut elem  = 0usize; // element within block

            for _chunk in 0..4 {
                let (d1, m1) = get_scale_min(sc, is);
                let (d2, m2) = get_scale_min(sc, is + 1);
                let scale1 = d * d1;
                let min1   = dmin * m1;
                let scale2 = d * d2;
                let min2   = dmin * m2;

                // First 32 elements: low nibbles of qs[q_off .. q_off+32]
                for l in 0..32usize {
                    if base + elem < end {
                        out[base + elem] = scale1 * (qs[q_off + l] & 0x0F) as f32 - min1;
                    }
                    elem += 1;
                }
                // Next 32 elements: high nibbles of qs[q_off .. q_off+32]
                for l in 0..32usize {
                    if base + elem < end {
                        out[base + elem] = scale2 * (qs[q_off + l] >> 4) as f32 - min2;
                    }
                    elem += 1;
                }

                q_off += 32;
                is    += 2;
            }
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Metadata value types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum GgufMetaValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(Vec<GgufMetaValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufMetaValue {
    fn read<R: Read>(f: &mut R) -> io::Result<Self> {
        let vtype = read_u32le(f)?;
        Self::read_typed(f, vtype)
    }

    fn read_typed<R: Read>(f: &mut R, vtype: u32) -> io::Result<Self> {
        Ok(match vtype {
            0  => Self::U8(read_u8(f)?),
            1  => Self::I8(read_u8(f)? as i8),
            2  => Self::U16(read_u16le(f)?),
            3  => Self::I16(read_u16le(f)? as i16),
            4  => Self::U32(read_u32le(f)?),
            5  => Self::I32(read_u32le(f)? as i32),
            6  => Self::F32(f32::from_bits(read_u32le(f)?)),
            7  => Self::Bool(read_u8(f)? != 0),
            8  => Self::Str(read_string(f)?),
            9  => {
                // array: elem_type (u32), count (u64), elements
                let elem_type = read_u32le(f)?;
                let count = read_u64le(f)? as usize;
                let mut arr = Vec::with_capacity(count.min(1024));
                for _ in 0..count {
                    arr.push(Self::read_typed(f, elem_type)?);
                }
                Self::Array(arr)
            }
            10 => Self::U64(read_u64le(f)?),
            11 => Self::I64(read_u64le(f)? as i64),
            12 => Self::F64(f64::from_bits(read_u64le(f)?)),
            _  => return Err(io::Error::new(io::ErrorKind::InvalidData,
                    format!("unknown metadata value type {vtype}"))),
        })
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(v)  => Some(*v as u64),
            Self::U16(v) => Some(*v as u64),
            Self::U32(v) => Some(*v as u64),
            Self::U64(v) => Some(*v),
            Self::I32(v) => Some(*v as u64),
            Self::I64(v) => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::F32(v) => Some(*v),
            Self::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Low-level I/O helpers
// ---------------------------------------------------------------------------

fn read_u8<R: Read>(f: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    f.read_exact(&mut b)?;
    Ok(b[0])
}

fn read_u16le<R: Read>(f: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    f.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

fn read_u32le<R: Read>(f: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64le<R: Read>(f: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    f.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

fn read_string<R: Read>(f: &mut R) -> io::Result<String> {
    let len = read_u64le(f)? as usize;
    let mut buf = vec![0u8; len];
    f.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn align_up(x: u64, align: u64) -> u64 {
    (x + align - 1) / align * align
}

/// Convert an IEEE 754 half-precision float (f16) bit pattern to f32.
pub fn f16_to_f32(bits: u16) -> f32 {
    // Extract components
    let sign     = ((bits >> 15) & 1) as u32;
    let exponent = ((bits >> 10) & 0x1F) as u32;
    let mantissa = (bits & 0x3FF) as u32;

    let f32_bits = if exponent == 0 {
        if mantissa == 0 {
            // ±zero
            sign << 31
        } else {
            // Denormal: shift mantissa to normalize
            let mut m = mantissa;
            let mut e = 0u32;
            while m & 0x400 == 0 { m <<= 1; e += 1; }
            m &= 0x3FF;
            let exp32 = 127 - 14 - e;
            (sign << 31) | (exp32 << 23) | (m << 13)
        }
    } else if exponent == 31 {
        // Inf or NaN
        (sign << 31) | (0xFF << 23) | (mantissa << 13)
    } else {
        // Normal number: rebias exponent (f16 bias=15, f32 bias=127)
        let exp32 = exponent + 127 - 15;
        (sign << 31) | (exp32 << 23) | (mantissa << 13)
    };

    f32::from_bits(f32_bits)
}
