#![allow(dead_code)]
// =============================================================================
// PNG writer -- 8-bit RGB
// =============================================================================
//
// The last step of a text-to-image pipeline, and the counterpart of `wav`:
// a decoder hands back f32 pixels in [-1, 1] and something has to make a file
// out of them.
//
// PNG is a signature, a header chunk, one or more data chunks and an end
// chunk, each chunk length-prefixed and CRC-32 suffixed:
//
//   \x89 P N G \r \n \x1a \n
//   <u32 len> "IHDR" <u32 width> <u32 height> <u8 depth=8> <u8 colour=2>
//                    <u8 compression=0> <u8 filter=0> <u8 interlace=0> <u32 crc>
//   <u32 len> "IDAT" <zlib stream> <u32 crc>
//   <u32 0>   "IEND" <u32 crc>
//
// ## Why there is no compressor here
//
// The IDAT payload is a zlib stream, which normally means a DEFLATE encoder --
// Huffman tables, a match finder, the whole apparatus. But DEFLATE has a
// *stored* block mode that emits literal bytes with a five-byte header, and a
// zlib stream made entirely of stored blocks is completely valid. Every
// decoder reads it.
//
// So this file needs a zlib header, stored blocks, an Adler-32 over the
// uncompressed data and a CRC-32 per chunk, and no compression algorithm at
// all. A 1024x1024 RGB image lands at about 3 MB rather than the 1.5 MB a real
// encoder would manage. That is the right trade for a project whose point is
// that nothing is imported -- and the alternative, linking the system zlib,
// would be the first third-party dependency in the codebase.
//
// Each scanline is prefixed with filter type 0 (None), which is what makes the
// stored-block trick work: any other filter would need the reconstruction to
// be inverted at read time, which is fine, but choosing filters well is most
// of what a PNG encoder does and none of it is free.

use crate::autograd2::Mat;

fn push_be32(out: &mut Vec<u8>, v: u32) {
    out.push((v >> 24) as u8);
    out.push((v >> 16) as u8);
    out.push((v >> 8) as u8);
    out.push(v as u8);
}

/// Append a length-prefixed, CRC-suffixed chunk.
///
/// The CRC covers the type and the payload but **not** the length, which is
/// the one detail of the format that a reader will reject silently-looking
/// files over.
fn push_chunk(out: &mut Vec<u8>, ty: &[u8; 4], payload: &[u8]) {
    push_be32(out, payload.len() as u32);
    let crc_start = out.len();
    out.extend_from_slice(ty);
    out.extend_from_slice(payload);
    let crc = crc32(&out[crc_start..], 0);
    push_be32(out, crc);
}

/// Wrap raw bytes in a zlib stream made of stored DEFLATE blocks.
///
/// Each block is `<u8 final> <u16 len> <u16 ~len> <len bytes>`, capped at
/// 65535 bytes, and the stream ends with an Adler-32 of everything stored.
fn zlib_stored(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65535 * 5 + 16);

    // CMF = deflate, 32 KiB window; FLG chosen so the pair is a multiple of 31.
    out.push(0x78);
    out.push(0x01);

    const MAX_BLOCK: usize = 65535;
    let mut offset = 0usize;
    loop {
        let len = MAX_BLOCK.min(raw.len() - offset);
        let is_final = offset + len >= raw.len();
        out.push(if is_final { 1 } else { 0 });
        out.push(len as u8);
        out.push((len >> 8) as u8);
        let nlen = !(len as u16);
        out.push(nlen as u8);
        out.push((nlen >> 8) as u8);
        out.extend_from_slice(&raw[offset..offset + len]);
        offset += len;
        if offset >= raw.len() {
            break;
        }
    }

    push_be32(&mut out, adler32(raw));
    out
}

// =============================================================================
// Checksums -- exposed for testing
// =============================================================================

const fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut n = 0u32;
    while n < 256 {
        let mut c = n;
        let mut k = 0;
        while k < 8 {
            c = if (c & 1) != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        t[n as usize] = c;
        n += 1;
    }
    t
}

static CRC32_TABLE: [u32; 256] = crc32_table();

/// CRC-32 (the zlib/PNG polynomial), continuing from `seed`.
pub fn crc32(bytes: &[u8], seed: u32) -> u32 {
    let mut c = seed ^ 0xFFFF_FFFF;
    for &b in bytes {
        c = CRC32_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Adler-32 over `bytes`.
pub fn adler32(bytes: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    // Reduce every 5552 bytes, the largest run that cannot overflow 32 bits.
    let mut i = 0usize;
    while i < bytes.len() {
        let chunk = 5552usize.min(bytes.len() - i);
        for k in 0..chunk {
            a += bytes[i + k] as u32;
            b += a;
        }
        a %= MOD;
        b %= MOD;
        i += chunk;
    }
    (b << 16) | a
}

// =============================================================================
// Encoding
// =============================================================================

/// Convert decoder output to 8-bit RGB.
///
/// `pixels` is [h*w, 3] in the spatial-major layout `conv2d` uses, holding
/// roughly [-1, 1]. The mapping is `x / 2 + 0.5`, then clamp, then scale by
/// 255 and round.
///
/// The order matters: clamping to [-1, 1] *before* the shift and clamping to
/// [0, 1] after are the same operation, but clamping to [0, 1] before the shift
/// crushes every negative value to mid-grey and flattens the shadows across the
/// whole image. It looks like a slightly hazy render rather than a bug.
pub fn f32_to_rgb8(pixels: &Mat) -> Vec<u8> {
    let mut out = Vec::with_capacity(pixels.rows * 3);
    for p in 0..pixels.rows {
        for c in 0..3 {
            // Shift into [0, 1] first, clamp second. Clamping the raw value to
            // [0, 1] instead would fold every negative pixel onto black rather
            // than onto mid-grey.
            let v = pixels.at(p, c) * 0.5 + 0.5;
            // NaN has to be caught by name: it compares false against both
            // bounds, so a clamp passes it straight through and the cast that
            // follows is meaningless. An infinity needs no special case --
            // clamping saturates it to the right end of the range.
            let clamped = if v.is_nan() { 0.0 } else { v.clamp(0.0, 1.0) };
            out.push((clamped * 255.0).round() as u8);
        }
    }
    out
}

/// Serialise 8-bit RGB samples as a PNG.
///
/// `rgb` is `h * w * 3` bytes, row-major.
pub fn encode_png(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut out: Vec<u8> = vec![0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];

    let mut ihdr = Vec::new();
    push_be32(&mut ihdr, width as u32);
    push_be32(&mut ihdr, height as u32);
    ihdr.push(8); // bit depth
    ihdr.push(2); // colour type 2 = truecolour RGB
    ihdr.push(0); // compression: deflate
    ihdr.push(0); // filter method 0
    ihdr.push(0); // no interlace
    push_chunk(&mut out, b"IHDR", &ihdr);

    // Filter byte 0 (None) in front of every scanline.
    let mut raw = Vec::with_capacity(height * (width * 3 + 1));
    for y in 0..height {
        raw.push(0);
        let base = y * width * 3;
        if base + width * 3 <= rgb.len() {
            raw.extend_from_slice(&rgb[base..base + width * 3]);
        } else {
            raw.resize(raw.len() + width * 3, 0);
        }
    }

    let stream = zlib_stored(&raw);
    push_chunk(&mut out, b"IDAT", &stream);
    push_chunk(&mut out, b"IEND", &[]);
    out
}

/// Write a decoder's [h*w, 3] output to a PNG file.
pub fn write_png(path: &str, pixels: &Mat, width: usize, height: usize) -> Result<(), String> {
    if pixels.cols != 3 {
        return Err(format!("write_png: expected 3 channels, got {}", pixels.cols));
    }
    if pixels.rows != width * height {
        return Err(format!(
            "write_png: rows {} != width*height {}",
            pixels.rows,
            width * height
        ));
    }

    let rgb = f32_to_rgb8(pixels);
    let bytes = encode_png(&rgb, width, height);

    use std::io::Write;
    let mut f = std::fs::File::create(path).map_err(|_| format!("write_png: cannot open {}", path))?;
    f.write_all(&bytes).map_err(|_| format!("write_png: short write to {}", path))?;
    Ok(())
}

// =============================================================================
// Image checks
// =============================================================================

/// Cheap statistics over a decoded image.
///
/// The image counterpart of `wave_stats`, and it exists for the same reason:
/// a wrong latent scale, a wrong GroupNorm epsilon and a wrong patch order all
/// produce something image-shaped, and looking at each one is slow. These
/// numbers separate "this is a picture" from "this is noise" without opening
/// the file.
#[derive(Clone, Debug)]
pub struct ImageStats {
    pub n: usize,
    pub mean: f32,
    pub rms: f32,
    pub min: f32,
    pub max: f32,
    /// Fraction of samples outside [-1, 1] before clamping.
    pub out_of_range_fraction: f32,
    /// Mean absolute difference between horizontally adjacent pixels.
    ///
    /// The useful one. A photograph is locally smooth and lands well under
    /// 0.1; uniform noise sits near 0.5. A VAE fed a latent that was not
    /// rescaled produces something in between, and this catches it.
    pub neighbour_delta: f32,
    /// True when every value is finite.
    pub finite: bool,
}

impl Default for ImageStats {
    fn default() -> Self {
        ImageStats {
            n: 0,
            mean: 0.0,
            rms: 0.0,
            min: 0.0,
            max: 0.0,
            out_of_range_fraction: 0.0,
            neighbour_delta: 0.0,
            finite: true,
        }
    }
}

/// `printf("%.*f")` for one value, spelling NaN the way the macOS libc does
/// (`nan` whatever its sign; Rust would print `NaN`).
fn c_fixed(v: f64, prec: usize) -> String {
    if v.is_nan() { "nan".to_string() } else { format!("{:.*}", prec, v) }
}

impl ImageStats {
    /// True when the image looks like a picture rather than noise or a flat
    /// field. Deliberately loose -- a smoke test, not a quality metric.
    pub fn looks_like_image(&self) -> bool {
        if !self.finite || self.n == 0 {
            return false;
        }
        // Noise sits near 0.5; a flat field sits near 0. A picture is in between.
        self.neighbour_delta > 0.0005
            && self.neighbour_delta < 0.35
            && self.rms > 0.01
            && self.out_of_range_fraction < 0.2
    }

    /// One-line summary for logs.
    pub fn describe(&self) -> String {
        format!(
            "n={} mean={} rms={} range=[{}, {}] oor={} neighbour={} {}",
            self.n,
            c_fixed(self.mean as f64, 4),
            c_fixed(self.rms as f64, 4),
            c_fixed(self.min as f64, 3),
            c_fixed(self.max as f64, 3),
            c_fixed(self.out_of_range_fraction as f64, 3),
            c_fixed(self.neighbour_delta as f64, 4),
            if self.looks_like_image() { "(image)" } else { "(not image-like)" }
        )
    }
}

pub fn image_stats(pixels: &Mat, width: usize, height: usize) -> ImageStats {
    let mut s = ImageStats { n: pixels.data.len(), ..Default::default() };
    if s.n == 0 {
        s.finite = true;
        return s;
    }

    let mut sum = 0.0f64;
    let mut sq = 0.0f64;
    let mut oor = 0usize;
    s.min = pixels.data[0];
    s.max = pixels.data[0];
    for &v in &pixels.data {
        if !v.is_finite() {
            s.finite = false;
            continue;
        }
        sum += v as f64;
        sq += v as f64 * v as f64;
        // std::min / std::max semantics: keep the current value unless the new
        // one compares strictly past it.
        if v < s.min {
            s.min = v;
        }
        if s.max < v {
            s.max = v;
        }
        if v < -1.0 || v > 1.0 {
            oor += 1;
        }
    }
    let n = s.n as f64;
    s.mean = (sum / n) as f32;
    s.rms = (sq / n).sqrt() as f32;
    s.out_of_range_fraction = (oor as f64 / n) as f32;

    // Horizontal neighbours only. The vertical direction says the same thing
    // and costs a second pass over a tensor that may be hundreds of megabytes.
    if width >= 2 && height >= 1 && pixels.rows == width * height {
        let mut delta = 0.0f64;
        let mut pairs = 0usize;
        for y in 0..height {
            for x in 0..width - 1 {
                let a = y * width + x;
                for c in 0..pixels.cols {
                    delta += (pixels.at(a, c) - pixels.at(a + 1, c)).abs() as f64;
                    pairs += 1;
                }
            }
        }
        if pairs > 0 {
            s.neighbour_delta = (delta / pairs as f64) as f32;
        }
    }
    s
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn be32(b: &[u8], at: usize) -> u32 {
        ((b[at] as u32) << 24) | ((b[at + 1] as u32) << 16) | ((b[at + 2] as u32) << 8) | b[at + 3] as u32
    }

    struct Chunk {
        ty: String,
        payload: Vec<u8>,
    }

    /// Walk the chunk list, checking every CRC as it goes.
    fn parse_png(bytes: &[u8]) -> Vec<Chunk> {
        const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
        assert!(bytes.len() > 8);
        assert_eq!(&bytes[..8], &SIG);

        let mut chunks = Vec::new();
        let mut at = 8usize;
        while at + 12 <= bytes.len() {
            let len = be32(bytes, at) as usize;
            assert!(at + 12 + len <= bytes.len());
            let ty = String::from_utf8(bytes[at + 4..at + 8].to_vec()).unwrap();
            let payload = bytes[at + 8..at + 8 + len].to_vec();
            let stated = be32(bytes, at + 8 + len);
            let actual = crc32(&bytes[at + 4..at + 8 + len], 0);
            assert_eq!(stated, actual);
            chunks.push(Chunk { ty, payload });
            at += 12 + len;
        }
        assert_eq!(at, bytes.len());
        chunks
    }

    /// Undo `zlib_stored`: the decoder side of the only DEFLATE mode this
    /// writes.
    fn inflate_stored(stream: &[u8]) -> Vec<u8> {
        assert!(stream.len() >= 6);
        assert_eq!(stream[0], 0x78);
        // The CMF/FLG pair must be a multiple of 31 or a real decoder rejects it.
        assert_eq!((stream[0] as u32 * 256 + stream[1] as u32) % 31, 0);

        let mut out = Vec::new();
        let mut at = 2usize;
        let mut is_final = false;
        while !is_final {
            assert!(at + 5 <= stream.len());
            let header = stream[at];
            assert_eq!(header & 0x06, 0); // stored block
            is_final = (header & 1) != 0;
            let len = stream[at + 1] as usize | ((stream[at + 2] as usize) << 8);
            let nlen = stream[at + 3] as usize | ((stream[at + 4] as usize) << 8);
            assert_eq!(len ^ 0xFFFF, nlen);
            at += 5;
            assert!(at + len <= stream.len());
            out.extend_from_slice(&stream[at..at + len]);
            at += len;
        }
        assert_eq!(at + 4, stream.len());
        let stated = be32(stream, at);
        assert_eq!(stated, adler32(&out));
        out
    }

    // -------------------------------------------------------------------------
    // Checksums
    // -------------------------------------------------------------------------

    #[test]
    fn crc32_matches_the_published_png_test_vectors() {
        // The IEND chunk's CRC is fixed and widely quoted: 0xAE426082 over "IEND".
        assert_eq!(crc32(b"IEND", 0), 0xAE42_6082);
        assert_eq!(crc32(b"123456789", 0), 0xCBF4_3926);
    }

    #[test]
    fn adler32_matches_its_definition() {
        assert_eq!(adler32(&[]), 1);
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn adler32_reduction_survives_more_than_one_5552_byte_run() {
        // The modular reduction happens per chunk; a run longer than one chunk
        // is the only thing that exercises the carry between them.
        let big: Vec<u8> = (0..20000usize).map(|i| (i * 7 + 3) as u8).collect();
        let mut a: u32 = 1;
        let mut b: u32 = 0;
        for &v in &big {
            a = (a + v as u32) % 65521;
            b = (b + a) % 65521;
        }
        assert_eq!(adler32(&big), (b << 16) | a);
    }

    // -------------------------------------------------------------------------
    // Pixel conversion
    // -------------------------------------------------------------------------

    #[test]
    fn f32_to_rgb8_maps_minus_one_to_one_onto_the_full_byte_range() {
        let p = Mat::new(vec![-1.0, 0.0, 1.0, -2.0, 2.0, 0.5, -0.5, 0.0, 0.0], 3, 3);
        let rgb = f32_to_rgb8(&p);
        assert_eq!(rgb.len(), 9);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 128);
        assert_eq!(rgb[2], 255);
        assert_eq!(rgb[3], 0);
        assert_eq!(rgb[4], 255);
        assert_eq!(rgb[5], 191);
        assert_eq!(rgb[6], 64);
    }

    #[test]
    fn f32_to_rgb8_shifts_before_it_clamps() {
        // A negative pixel must land in the lower half of the range, not on
        // black. Clamping to [0, 1] first would map every one of these to 0.
        let p = Mat::new(vec![-0.25, -0.5, -0.75], 1, 3);
        let rgb = f32_to_rgb8(&p);
        assert_eq!(rgb[0], 96);
        assert_eq!(rgb[1], 64);
        assert_eq!(rgb[2], 32);
    }

    #[test]
    fn f32_to_rgb8_saturates_infinities_and_blacks_out_nan() {
        let p = Mat::new(vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY], 1, 3);
        let rgb = f32_to_rgb8(&p);
        assert_eq!(rgb[0], 0);
        assert_eq!(rgb[1], 255);
        assert_eq!(rgb[2], 0);
    }

    // -------------------------------------------------------------------------
    // Container
    // -------------------------------------------------------------------------

    #[test]
    fn encode_png_produces_a_well_formed_file() {
        let (w, h) = (5usize, 3usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 13 + 7) as u8).collect();

        let png = encode_png(&rgb, w, h);
        let chunks = parse_png(&png);

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].ty, "IHDR");
        assert_eq!(chunks[1].ty, "IDAT");
        assert_eq!(chunks[2].ty, "IEND");
        assert!(chunks[2].payload.is_empty());

        let ihdr = &chunks[0].payload;
        assert_eq!(ihdr.len(), 13);
        assert_eq!(be32(ihdr, 0) as usize, w);
        assert_eq!(be32(ihdr, 4) as usize, h);
        assert_eq!(ihdr[8], 8); // bit depth
        assert_eq!(ihdr[9], 2); // truecolour
        assert_eq!(ihdr[10], 0); // deflate
        assert_eq!(ihdr[11], 0); // filter method
        assert_eq!(ihdr[12], 0); // no interlace
    }

    #[test]
    fn the_idat_stream_inflates_back_to_the_original_scanlines() {
        let (w, h) = (7usize, 4usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| (i * 31 + 11) as u8).collect();

        let png = encode_png(&rgb, w, h);
        let chunks = parse_png(&png);
        let raw = inflate_stored(&chunks[1].payload);

        assert_eq!(raw.len(), h * (w * 3 + 1));
        for y in 0..h {
            let at = y * (w * 3 + 1);
            assert_eq!(raw[at], 0); // filter type None
            for i in 0..w * 3 {
                assert_eq!(raw[at + 1 + i], rgb[y * w * 3 + i]);
            }
        }
    }

    #[test]
    fn an_image_larger_than_one_stored_block_still_round_trips() {
        // 65535 bytes is the stored-block cap; this crosses it several times,
        // which is the only way to exercise the non-final block header.
        let (w, h) = (300usize, 300usize);
        let rgb: Vec<u8> = (0..w * h * 3).map(|i| ((i * 97 + i / 251) & 0xFF) as u8).collect();
        assert!(h * (w * 3 + 1) > 4 * 65535);

        let png = encode_png(&rgb, w, h);
        let chunks = parse_png(&png);
        let raw = inflate_stored(&chunks[1].payload);

        assert_eq!(raw.len(), h * (w * 3 + 1));
        for y in 0..h {
            let at = y * (w * 3 + 1);
            assert_eq!(raw[at], 0);
            for i in 0..w * 3 {
                assert_eq!(raw[at + 1 + i], rgb[y * w * 3 + i]);
            }
        }
    }

    #[test]
    fn a_one_pixel_image_is_still_a_valid_png() {
        let rgb = [10u8, 20, 30];
        let png = encode_png(&rgb, 1, 1);
        let chunks = parse_png(&png);
        let raw = inflate_stored(&chunks[1].payload);
        assert_eq!(raw, vec![0u8, 10, 20, 30]);
    }

    // -------------------------------------------------------------------------
    // Image statistics
    // -------------------------------------------------------------------------

    #[test]
    fn image_stats_separates_a_picture_from_noise_and_from_a_flat_field() {
        let (w, h) = (32usize, 32usize);

        // A smooth gradient with some structure: locally smooth, globally varied.
        let mut picture = Mat::zeros(w * h, 3);
        for y in 0..h {
            for x in 0..w {
                let v = (x as f32 * 0.2).sin() * (y as f32 * 0.15).cos() * 0.7;
                for c in 0..3 {
                    *picture.at_mut(y * w + x, c) = v;
                }
            }
        }
        let ps = image_stats(&picture, w, h);
        assert!(ps.finite);
        assert!(ps.looks_like_image());

        // Uncorrelated noise: neighbours differ by about a third of the range.
        let mut noise = Mat::zeros(w * h, 3);
        let mut state: u32 = 12345;
        for v in noise.data.iter_mut() {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *v = (state >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
        }
        let ns = image_stats(&noise, w, h);
        assert!(ns.neighbour_delta > ps.neighbour_delta * 4.0);
        assert!(!ns.looks_like_image());

        // A flat field has no neighbour delta at all.
        let flat = Mat::zeros(w * h, 3);
        assert!(!image_stats(&flat, w, h).looks_like_image());
    }

    #[test]
    fn image_stats_reports_non_finite_values() {
        let mut m = Mat::zeros(4, 3);
        *m.at_mut(2, 1) = f32::NAN;
        let s = image_stats(&m, 2, 2);
        assert!(!s.finite);
        assert!(!s.looks_like_image());
    }

    #[test]
    fn image_stats_counts_out_of_range_samples() {
        let mut m = Mat::zeros(4, 3);
        *m.at_mut(0, 0) = 3.0;
        *m.at_mut(1, 1) = -5.0;
        let s = image_stats(&m, 2, 2);
        assert!(s.out_of_range_fraction > 0.16);
        assert!(s.out_of_range_fraction < 0.17);
        assert!(s.max > 2.9);
        assert!(s.min < -4.9);
    }
}
