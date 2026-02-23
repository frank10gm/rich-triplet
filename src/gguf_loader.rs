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
use crate::autograd2::Q4Mat;

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
    /// GGUF Q4_0 block: [f16 scale (2 bytes)] [16 bytes nibbles (32 × u4)]
    /// Nibble value range: 0..15 unsigned, actual = nibble - 8
    ///
    /// Our Q4Mat: same nibble packing but 4-bit two's complement (-7..7)
    /// with f32 scales where scale = absmax/7.
    ///
    /// Conversion:
    ///   GGUF_scale (f16) encodes absmax/8.
    ///   Our scale should be absmax/7.
    ///   So our_scale = gguf_scale * 8/7.
    ///   Nibbles: GGUF stores v ∈ 0..15; we store v - 8 re-encoded as 4-bit two's complement.
    ///   Equivalently: our_nibble = (v.wrapping_sub(8)) & 0x0F.
    pub fn decode_q4_0_to_q4mat(&self, idx: usize) -> io::Result<Q4Mat> {
        let info = &self.tensor_info[idx];
        assert_eq!(info.gguf_type, GgufType::Q4_0,
            "decode_q4_0_to_q4mat called on non-Q4_0 tensor");

        let bytes = self.read_tensor_bytes(idx)?;
        let n_elem = info.n_elements();
        let n_blocks = (n_elem + 31) / 32;

        // Parse shape: GGUF stores shape as [cols, rows] (Fortran order) for 2D tensors.
        // We want [rows, cols] (C order) as used in our Mat.
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),  // transpose: GGUF is [cols, rows]
            _ => return Err(io::Error::new(io::ErrorKind::InvalidData,
                format!("unexpected shape rank {} for {}", info.shape.len(), info.name))),
        };

        let mut scales = Vec::with_capacity(n_blocks);
        let n_packed = (n_elem + 1) / 2;
        let mut packed = vec![0u8; n_packed];

        for b in 0..n_blocks {
            let block_off = b * 18;
            // f16 scale
            let scale_bits = u16::from_le_bytes([bytes[block_off], bytes[block_off + 1]]);
            let gguf_scale = f16_to_f32(scale_bits);
            // Convert: GGUF scale = absmax/8, ours = absmax/7
            let our_scale = gguf_scale * (8.0 / 7.0);
            scales.push(our_scale);

            // 16 bytes of nibbles, 32 nibbles total
            // GGUF packing: byte k has nibbles for element 2k (low) and 2k+1 (high)
            // Our packing: element k is at packed[k/2], low nibble if k even, high if k odd
            // They are identical in packing order — just need to remap value (subtract 8).
            let nibble_off = block_off + 2;
            let start_elem = b * 32;
            let end_elem   = (start_elem + 32).min(n_elem);

            for byte_i in 0..16 {
                let src_byte = bytes[nibble_off + byte_i];
                let lo = src_byte & 0x0F;
                let hi = (src_byte >> 4) & 0x0F;

                let elem0 = start_elem + byte_i * 2;
                let elem1 = elem0 + 1;

                // Remap: GGUF unsigned 0..15 → two's complement (subtract 8)
                // (v - 8) & 0x0F gives the correct 4-bit two's complement nibble
                if elem0 < end_elem {
                    let our_lo = lo.wrapping_sub(8) & 0x0F;
                    packed[elem0 / 2] = if elem0 % 2 == 0 {
                        (packed[elem0 / 2] & 0xF0) | our_lo
                    } else {
                        (packed[elem0 / 2] & 0x0F) | (our_lo << 4)
                    };
                }
                if elem1 < end_elem {
                    let our_hi = hi.wrapping_sub(8) & 0x0F;
                    packed[elem1 / 2] = if elem1 % 2 == 0 {
                        (packed[elem1 / 2] & 0xF0) | our_hi
                    } else {
                        (packed[elem1 / 2] & 0x0F) | (our_hi << 4)
                    };
                }
            }
        }

        Ok(Q4Mat { rows, cols, packed, scales })
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
