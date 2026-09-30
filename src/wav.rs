// =============================================================================
// WAV writer -- 16-bit PCM
// =============================================================================
//
// The last step of a text-to-speech pipeline. A decoder hands back f32 samples
// in [-1, 1] and something has to make a file out of them.
//
// RIFF is three chunks and no compression:
//
//   "RIFF" <u32 total-8> "WAVE"
//   "fmt " <u32 16> <u16 format=1> <u16 channels> <u32 rate>
//          <u32 byte-rate> <u16 block-align> <u16 bits=16>
//   "data" <u32 bytes> <interleaved little-endian i16 samples>
//
// Every length is 32-bit, which caps a file at 4 GiB -- about 24 hours of mono
// 24 kHz audio, so not a limit speech synthesis reaches.

#![allow(dead_code)]

use std::io::{Read, Write};

fn push_u16(out: &mut Vec<u8>, v: u16) {
    out.push((v & 0xff) as u8);
    out.push(((v >> 8) & 0xff) as u8);
}

fn push_u32(out: &mut Vec<u8>, v: u32) {
    for i in 0..4 {
        out.push(((v >> (8 * i)) & 0xff) as u8);
    }
}

fn push_tag(out: &mut Vec<u8>, tag: &[u8; 4]) {
    out.extend_from_slice(tag);
}

/// Convert f32 samples to 16-bit PCM.
///
/// Scales by 32767 and clamps. A codec decoder ending in Tanh cannot exceed
/// [-1, 1], so the clamp is only insurance for callers that do not -- but
/// without it, out-of-range input wraps around into loud noise rather than
/// merely distorting, so it is worth the branch.
pub fn f32_to_pcm16(samples: &[f32]) -> Vec<i16> {
    samples
        .iter()
        .map(|&s| {
            // NaN carries no sign worth honouring, so it becomes silence.
            // Infinities do, so they clamp to the matching rail like any other
            // out-of-range sample.
            let v = if s.is_nan() { 0.0f32 } else { s };
            let scaled = v.clamp(-1.0, 1.0) * 32767.0f32;
            // `round` is half away from zero, like `std::lround`.
            scaled.round() as i16
        })
        .collect()
}

/// Serialise a WAV to bytes, for tests and for callers that do their own I/O.
pub fn encode_wav(samples: &[f32], sample_rate: usize, channels: usize) -> Vec<u8> {
    let pcm = f32_to_pcm16(samples);
    let data_bytes = (pcm.len() * 2) as u32;
    let bits: u16 = 16;
    let block_align = (channels * bits as usize / 8) as u16;

    let mut out = Vec::with_capacity(44 + pcm.len() * 2);

    push_tag(&mut out, b"RIFF");
    push_u32(&mut out, 36u32.wrapping_add(data_bytes)); // everything after this field
    push_tag(&mut out, b"WAVE");

    push_tag(&mut out, b"fmt ");
    push_u32(&mut out, 16); // PCM fmt chunk size
    push_u16(&mut out, 1); // format 1 == uncompressed PCM
    push_u16(&mut out, channels as u16);
    push_u32(&mut out, sample_rate as u32);
    push_u32(&mut out, (sample_rate * block_align as usize) as u32); // byte rate
    push_u16(&mut out, block_align);
    push_u16(&mut out, bits);

    push_tag(&mut out, b"data");
    push_u32(&mut out, data_bytes);
    for s in pcm {
        push_u16(&mut out, s as u16);
    }
    out
}

/// Write mono or interleaved multi-channel f32 samples as a 16-bit PCM WAV.
pub fn write_wav(
    path: &str,
    samples: &[f32],
    sample_rate: usize,
    channels: usize,
) -> Result<(), String> {
    if channels == 0 {
        return Err("wav: channel count must be at least 1".to_string());
    }
    if sample_rate == 0 {
        return Err("wav: sample rate must be at least 1".to_string());
    }
    if samples.len() % channels != 0 {
        return Err(format!(
            "wav: {} samples do not divide into {} channels",
            samples.len(),
            channels
        ));
    }

    let bytes = encode_wav(samples, sample_rate, channels);

    let mut f = std::fs::File::create(path)
        .map_err(|_| format!("wav: cannot open {} for writing", path))?;
    f.write_all(&bytes)
        .map_err(|_| format!("wav: short write to {}", path))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

fn read_u16(b: &[u8], off: usize) -> u16 {
    b[off] as u16 | ((b[off + 1] as u16) << 8)
}

fn read_u32(b: &[u8], off: usize) -> u32 {
    b[off] as u32 | ((b[off + 1] as u32) << 8) | ((b[off + 2] as u32) << 16) | ((b[off + 3] as u32) << 24)
}

fn tag_is(b: &[u8], off: usize, tag: &[u8; 4]) -> bool {
    &b[off..off + 4] == tag
}

/// A decoded WAV: samples in [-1, 1], interleaved if there is more than one
/// channel.
#[derive(Clone, Debug)]
pub struct WavFile {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub channels: usize,
}

impl Default for WavFile {
    fn default() -> Self {
        WavFile { samples: Vec::new(), sample_rate: 0, channels: 1 }
    }
}

impl WavFile {
    /// Average the channels down to one. Voice cloning wants mono, and every
    /// analysis path downstream assumes it.
    pub fn mono(&self) -> Vec<f32> {
        if self.channels <= 1 {
            return self.samples.clone();
        }
        let n = self.frames();
        let mut out = vec![0.0f32; n];
        for (i, o) in out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for c in 0..self.channels {
                sum += self.samples[i * self.channels + c];
            }
            *o = sum / self.channels as f32;
        }
        out
    }

    pub fn frames(&self) -> usize {
        if self.channels == 0 { 0 } else { self.samples.len() / self.channels }
    }
}

/// Parse a RIFF/WAVE file from memory.
///
/// Accepts 8-bit unsigned, 16/24/32-bit signed PCM (format 1) and 32/64-bit
/// float (format 3), plus WAVE_FORMAT_EXTENSIBLE (0xFFFE), whose real format
/// lives in the first two bytes of its extension. Chunks other than `fmt ` and
/// `data` are skipped, which is what makes files written by anything other
/// than this module readable -- `LIST`/`INFO` metadata is near universal.
pub fn decode_wav(bytes: &[u8]) -> Result<WavFile, String> {
    if bytes.len() < 12 {
        return Err("wav: file is too short to be RIFF".to_string());
    }
    if !tag_is(bytes, 0, b"RIFF") || !tag_is(bytes, 8, b"WAVE") {
        return Err("wav: not a RIFF/WAVE file".to_string());
    }

    let mut format: u16 = 0;
    let mut channels: u16 = 0;
    let mut sample_rate: u32 = 0;
    let mut bits: u16 = 0;
    let mut have_fmt = false;
    let mut data: &[u8] = &[];
    let mut have_data = false;

    // Walk the chunk list. Anything that is not fmt or data -- LIST, INFO,
    // fact, a trailing ID3 tag -- is skipped rather than rejected, since almost
    // no encoder writes the bare 44-byte header this module does.
    let mut off = 12usize;
    while off + 8 <= bytes.len() {
        let size = read_u32(bytes, off + 4);
        let body = off + 8;
        // A chunk claiming more than the file holds is truncation, not a
        // reason to lose what came before it.
        let avail = (size as usize).min(bytes.len() - body);

        if tag_is(bytes, off, b"fmt ") && avail >= 16 {
            format = read_u16(bytes, body);
            channels = read_u16(bytes, body + 2);
            sample_rate = read_u32(bytes, body + 4);
            bits = read_u16(bytes, body + 14);
            // WAVE_FORMAT_EXTENSIBLE hides the real format in the first two
            // bytes of its extension block; the rest of the GUID is fixed.
            if format == 0xFFFE && avail >= 26 {
                format = read_u16(bytes, body + 24);
            }
            have_fmt = true;
        } else if tag_is(bytes, off, b"data") {
            data = &bytes[body..body + avail];
            have_data = true;
        }

        // Chunks are word-aligned: an odd size is followed by a pad byte.
        off = body + avail + (avail % 2);
    }

    if !have_fmt {
        return Err("wav: no fmt chunk".to_string());
    }
    if !have_data {
        return Err("wav: no data chunk".to_string());
    }
    if channels == 0 {
        return Err("wav: zero channels".to_string());
    }
    if sample_rate == 0 {
        return Err("wav: zero sample rate".to_string());
    }

    let mut out = WavFile {
        samples: Vec::new(),
        sample_rate: sample_rate as usize,
        channels: channels as usize,
    };

    let scale_int = |v: i64, peak: i64| (v as f64 / peak as f64) as f32;

    if format == 1 && bits == 16 {
        out.samples.reserve(data.len() / 2);
        let mut i = 0;
        while i + 1 < data.len() {
            out.samples.push(scale_int(read_u16(data, i) as i16 as i64, 32768));
            i += 2;
        }
    } else if format == 1 && bits == 8 {
        // 8-bit PCM is the odd one out: unsigned, centred on 128.
        out.samples.reserve(data.len());
        for &v in data {
            out.samples.push((v as f32 - 128.0) / 128.0);
        }
    } else if format == 1 && bits == 24 {
        out.samples.reserve(data.len() / 3);
        let mut i = 0;
        while i + 2 < data.len() {
            let raw = data[i] as u32 | ((data[i + 1] as u32) << 8) | ((data[i + 2] as u32) << 16);
            // Sign-extend from 24 bits.
            let v = ((raw << 8) as i32) >> 8;
            out.samples.push(scale_int(v as i64, 8388608));
            i += 3;
        }
    } else if format == 1 && bits == 32 {
        out.samples.reserve(data.len() / 4);
        let mut i = 0;
        while i + 3 < data.len() {
            out.samples.push(scale_int(read_u32(data, i) as i32 as i64, 2147483648));
            i += 4;
        }
    } else if format == 3 && bits == 32 {
        out.samples.reserve(data.len() / 4);
        let mut i = 0;
        while i + 3 < data.len() {
            out.samples.push(f32::from_bits(read_u32(data, i)));
            i += 4;
        }
    } else if format == 3 && bits == 64 {
        out.samples.reserve(data.len() / 8);
        let mut i = 0;
        while i + 7 < data.len() {
            let mut raw = 0u64;
            for b in 0..8 {
                raw |= (data[i + b] as u64) << (8 * b);
            }
            out.samples.push(f64::from_bits(raw) as f32);
            i += 8;
        }
    } else {
        return Err(format!("wav: unsupported format {} at {} bits", format, bits));
    }

    // A truncated final frame would leave the channels out of phase for every
    // consumer downstream, so drop it rather than carry it.
    let whole = out.frames() * out.channels;
    out.samples.truncate(whole);
    if out.samples.is_empty() {
        return Err("wav: data chunk holds no whole frames".to_string());
    }
    Ok(out)
}

/// Read and parse a WAV file from disk.
pub fn read_wav(path: &str) -> Result<WavFile, String> {
    let mut f = std::fs::File::open(path)
        .map_err(|_| format!("wav: cannot open {} for reading", path))?;
    let mut bytes = Vec::new();
    f.read_to_end(&mut bytes)
        .map_err(|_| format!("wav: read error on {}", path))?;
    decode_wav(&bytes)
}

// ---------------------------------------------------------------------------
// Signal checks
// ---------------------------------------------------------------------------

/// Cheap statistics over a waveform.
///
/// Debugging a synthesis pipeline by ear is miserable, because a wrong RoPE
/// scale, a wrong codebook stride and a wrong convolution padding all sound
/// like the same noise. These numbers separate "this is speech" from "this is
/// broken" without listening: speech at 24 kHz sits near an RMS of 0.03-0.2
/// with a DC offset close to zero, while a decoder fed bad latents saturates
/// its output Tanh and lands nearer 0.5 with most samples at the rails.
#[derive(Clone, Debug, Default)]
pub struct WaveStats {
    pub n: usize,
    pub peak: f32,
    pub rms: f32,
    /// Mean sample value; a real recording is close to zero.
    pub dc: f32,
    /// Fraction of samples within 1e-3 of +-1, where Tanh has saturated.
    pub clipped_fraction: f32,
    /// Fraction of adjacent pairs that change sign -- a rough brightness
    /// proxy. Voiced speech runs well under 0.5; white noise sits near it.
    pub zero_crossing_rate: f32,
}

impl WaveStats {
    /// True when every sample is finite and inside [-1, 1].
    pub fn in_range(&self) -> bool {
        self.n > 0 && self.peak <= 1.0
    }

    /// True when the waveform looks like speech rather than noise or silence.
    ///
    /// Deliberately loose -- it is a smoke test meant to catch a pipeline that
    /// is producing garbage, not a quality metric.
    pub fn looks_like_speech(&self) -> bool {
        if !self.in_range() {
            return false;
        }
        // Quiet enough to be silence, or loud enough that the Tanh is pinned.
        if self.rms < 0.005 || self.rms > 0.45 {
            return false;
        }
        // A real waveform swings both ways around zero.
        if self.dc.abs() > 0.05 {
            return false;
        }
        // Saturation means the decoder was fed something it could not represent.
        if self.clipped_fraction > 0.02 {
            return false;
        }
        // Speech has voiced stretches; broadband noise crosses zero almost every
        // other sample.
        if self.zero_crossing_rate > 0.45 {
            return false;
        }
        true
    }

    /// One-line summary for logs.
    pub fn describe(&self) -> String {
        format!(
            "n={} peak={:.4} rms={:.4} dc={:.4} clipped={:.4} zcr={:.4}",
            self.n, self.peak, self.rms, self.dc, self.clipped_fraction, self.zero_crossing_rate
        )
    }
}

pub fn wave_stats(samples: &[f32]) -> WaveStats {
    let mut s = WaveStats { n: samples.len(), ..Default::default() };
    if samples.is_empty() {
        return s;
    }

    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut clipped = 0usize;
    let mut crossings = 0usize;
    for (i, &v) in samples.iter().enumerate() {
        if !v.is_finite() {
            s.peak = f32::INFINITY;
            return s;
        }
        sum += v as f64;
        sum_sq += v as f64 * v as f64;
        if s.peak < v.abs() {
            s.peak = v.abs();
        }
        if v.abs() >= 1.0f32 - 1e-3f32 {
            clipped += 1;
        }
        if i > 0 && ((samples[i - 1] < 0.0) != (v < 0.0)) {
            crossings += 1;
        }
    }

    let count = samples.len() as f64;
    s.dc = (sum / count) as f32;
    s.rms = (sum_sq / count).sqrt() as f32;
    s.clipped_fraction = (clipped as f64 / count) as f32;
    s.zero_crossing_rate = if samples.len() > 1 {
        (crossings as f64 / (count - 1.0)) as f32
    } else {
        0.0
    };
    s
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

    fn tag_at(b: &[u8], off: usize) -> &[u8] {
        &b[off..off + 4]
    }

    fn u32_at(b: &[u8], off: usize) -> u32 {
        read_u32(b, off)
    }

    fn u16_at(b: &[u8], off: usize) -> u16 {
        read_u16(b, off)
    }

    /// A 220 Hz sine at 24 kHz -- a stand-in for voiced speech.
    fn tone(n: usize, freq: f32, amplitude: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f32 / 24000.0;
                amplitude * (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    /// A per-test temp path, so parallel tests never share a file.
    fn temp_path(tag: &str) -> String {
        std::env::temp_dir().join(format!("rt_wav_{}.wav", tag)).to_string_lossy().into_owned()
    }

    // =========================================================================
    // PCM conversion
    // =========================================================================

    #[test]
    fn f32_to_pcm16_scales_by_32767() {
        let pcm = f32_to_pcm16(&[0.0, 1.0, -1.0, 0.5]);
        assert_eq!(pcm.len(), 4);
        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], 32767);
        assert_eq!(pcm[2], -32767);
        assert_eq!(pcm[3], 16384); // lround(0.5 * 32767)
    }

    #[test]
    fn f32_to_pcm16_clamps_instead_of_wrapping() {
        // Wrapping would turn a loud sample into the opposite rail, which is an
        // audible click rather than mild distortion.
        let pcm = f32_to_pcm16(&[2.0, -2.0, 1e9]);
        assert_eq!(pcm[0], 32767);
        assert_eq!(pcm[1], -32767);
        assert_eq!(pcm[2], 32767);
    }

    #[test]
    fn f32_to_pcm16_maps_non_finite_samples_to_silence() {
        let pcm = f32_to_pcm16(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY]);
        assert_eq!(pcm[0], 0);
        assert_eq!(pcm[1], 32767);
        assert_eq!(pcm[2], -32767);
    }

    // =========================================================================
    // RIFF structure
    // =========================================================================

    #[test]
    fn encode_wav_writes_a_valid_44_byte_header() {
        let samples = vec![0.0f32; 100];
        let b = encode_wav(&samples, 24000, 1);

        assert_eq!(b.len(), 44 + 200);
        assert_eq!(tag_at(&b, 0), b"RIFF");
        assert_eq!(u32_at(&b, 4), 36 + 200);
        assert_eq!(tag_at(&b, 8), b"WAVE");
        assert_eq!(tag_at(&b, 12), b"fmt ");
        assert_eq!(u32_at(&b, 16), 16);
        assert_eq!(u16_at(&b, 20), 1); // PCM
        assert_eq!(u16_at(&b, 22), 1); // mono
        assert_eq!(u32_at(&b, 24), 24000); // sample rate
        assert_eq!(u32_at(&b, 28), 48000); // byte rate = 24000 * 2
        assert_eq!(u16_at(&b, 32), 2); // block align
        assert_eq!(u16_at(&b, 34), 16); // bits
        assert_eq!(tag_at(&b, 36), b"data");
        assert_eq!(u32_at(&b, 40), 200);
    }

    #[test]
    fn encode_wav_sets_the_byte_rate_from_the_channel_count() {
        let samples = vec![0.0f32; 8];
        let b = encode_wav(&samples, 44100, 2);
        assert_eq!(u16_at(&b, 22), 2); // stereo
        assert_eq!(u16_at(&b, 32), 4); // block align = 2ch * 2 bytes
        assert_eq!(u32_at(&b, 28), 176400); // 44100 * 4
    }

    #[test]
    fn encode_wav_stores_samples_little_endian_after_the_header() {
        let b = encode_wav(&[1.0, -1.0], 24000, 1);
        assert_eq!(b[44], 0xff);
        assert_eq!(b[45], 0x7f); // 32767
        assert_eq!(b[46], 0x01);
        assert_eq!(b[47], 0x80); // -32767
    }

    #[test]
    fn write_wav_produces_a_file_that_reads_back_byte_for_byte() {
        let path = temp_path("roundtrip");
        let samples = tone(480, 220.0, 0.25);
        assert!(write_wav(&path, &samples, 24000, 1).is_ok());

        let read_back = std::fs::read(&path).expect("read back");
        let _ = std::fs::remove_file(&path);

        // Nothing should follow the data chunk.
        assert_eq!(read_back.len(), 44 + 960);
        assert_eq!(read_back, encode_wav(&samples, 24000, 1));
    }

    #[test]
    fn write_wav_rejects_impossible_parameters() {
        let path = temp_path("bad");
        let samples = vec![0.0f32; 3];
        assert!(write_wav(&path, &samples, 24000, 0).is_err());
        assert!(write_wav(&path, &samples, 0, 1).is_err());
        // 3 samples cannot be split into 2 channels.
        assert!(write_wav(&path, &samples, 24000, 2).is_err());
    }

    #[test]
    fn write_wav_reports_an_unwritable_path() {
        let samples = vec![0.0f32; 4];
        let r = write_wav("/nonexistent-dir-rt/out.wav", &samples, 24000, 1);
        assert!(r.is_err());
        assert!(r.unwrap_err().contains("cannot open"));
    }

    // =========================================================================
    // Signal checks
    // =========================================================================

    #[test]
    fn wave_stats_measures_peak_rms_and_dc() {
        // A full-scale sine has rms 1/sqrt(2) and no dc offset.
        let samples = tone(24000, 100.0, 1.0);
        let s = wave_stats(&samples);
        assert_eq!(s.n, 24000);
        assert!(s.peak > 0.99);
        assert!((s.rms - 0.7071).abs() < 0.01);
        assert!(s.dc.abs() < 0.01);
    }

    #[test]
    fn wave_stats_counts_zero_crossings() {
        // A 1 kHz tone at 24 kHz crosses zero twice per cycle: 2000 crossings a
        // second out of 24000 samples.
        let samples = tone(24000, 1000.0, 0.5);
        let s = wave_stats(&samples);
        assert!((s.zero_crossing_rate - 2000.0 / 24000.0).abs() < 0.01);
    }

    #[test]
    fn wave_stats_flags_a_non_finite_waveform() {
        let mut samples = vec![0.1f32; 100];
        samples[50] = f32::NAN;
        let s = wave_stats(&samples);
        assert!(!s.in_range());
    }

    #[test]
    fn looks_like_speech_accepts_a_speech_level_tone() {
        let samples = tone(24000, 220.0, 0.15);
        let s = wave_stats(&samples);
        assert!(s.looks_like_speech(), "{}", s.describe());
    }

    #[test]
    fn looks_like_speech_rejects_silence() {
        let samples = vec![0.0f32; 24000];
        assert!(!wave_stats(&samples).looks_like_speech());
    }

    #[test]
    fn looks_like_speech_rejects_a_saturated_waveform() {
        // What a decoder fed bad latents produces: the output Tanh pinned to its
        // rails. This is the failure mode the check exists to catch.
        let samples: Vec<f32> = (0..24000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let s = wave_stats(&samples);
        assert!(!s.looks_like_speech(), "{}", s.describe());
        assert!(s.clipped_fraction > 0.9);
    }

    #[test]
    fn looks_like_speech_rejects_a_dc_offset() {
        let mut samples = tone(24000, 220.0, 0.15);
        for v in samples.iter_mut() {
            *v += 0.3;
        }
        assert!(!wave_stats(&samples).looks_like_speech());
    }

    #[test]
    fn wave_stats_handles_an_empty_waveform() {
        let s = wave_stats(&[]);
        assert_eq!(s.n, 0);
        assert!(!s.in_range());
        assert!(!s.looks_like_speech());
    }

    // =========================================================================
    // Reading
    // =========================================================================

    #[test]
    fn decode_wav_round_trips_what_encode_wav_writes() {
        let samples = tone(2400, 220.0, 0.5);
        let bytes = encode_wav(&samples, 24000, 1);

        let got = decode_wav(&bytes).expect("decode");
        assert_eq!(got.sample_rate, 24000);
        assert_eq!(got.channels, 1);
        assert_eq!(got.samples.len(), samples.len());
        // 16-bit PCM quantizes to steps of 1/32767, so the round trip is lossy by
        // half a step and no more.
        for i in 0..samples.len() {
            assert!((got.samples[i] - samples[i]).abs() < 1.0 / 32000.0);
        }
    }

    #[test]
    fn decode_wav_keeps_stereo_interleaved_and_averages_it_on_request() {
        // Left is a constant +0.5, right a constant -0.25.
        let mut stereo = Vec::new();
        for _ in 0..100 {
            stereo.push(0.5f32);
            stereo.push(-0.25f32);
        }
        let got = decode_wav(&encode_wav(&stereo, 48000, 2)).expect("decode");
        assert_eq!(got.channels, 2);
        assert_eq!(got.frames(), 100);
        assert_eq!(got.samples.len(), 200);

        let mono = got.mono();
        assert_eq!(mono.len(), 100);
        for v in mono {
            assert!(approx(v, 0.125));
        }
    }

    #[test]
    fn decode_wav_skips_chunks_it_does_not_know() {
        // Real encoders write LIST/INFO metadata between fmt and data. A reader
        // that assumes the 44-byte header this module writes would reject them.
        let bytes = encode_wav(&tone(240, 220.0, 0.3), 24000, 1);
        let junk: [u8; 14] = [b'L', b'I', b'S', b'T', 6, 0, 0, 0, b'I', b'N', b'F', b'O', b'x', b'y'];

        let mut spliced: Vec<u8> = bytes[..36].to_vec();
        spliced.extend_from_slice(&junk);
        spliced.extend_from_slice(&bytes[36..]);
        // The RIFF size field covers everything after itself.
        let riff = (spliced.len() - 8) as u32;
        for i in 0..4 {
            spliced[4 + i] = ((riff >> (8 * i)) & 0xff) as u8;
        }

        let got = decode_wav(&spliced).expect("decode");
        assert_eq!(got.samples.len(), 240);
    }

    #[test]
    fn decode_wav_reads_8_24_and_32_bit_pcm_and_float() {
        // Build each format by hand around the same one-sample payload.
        let make = |format: u16, bits: u16, payload: &[u8]| {
            let mut b = Vec::new();
            push_tag(&mut b, b"RIFF");
            push_u32(&mut b, (36 + payload.len()) as u32);
            push_tag(&mut b, b"WAVE");
            push_tag(&mut b, b"fmt ");
            push_u32(&mut b, 16);
            push_u16(&mut b, format);
            push_u16(&mut b, 1);
            push_u32(&mut b, 24000);
            push_u32(&mut b, 24000);
            push_u16(&mut b, bits / 8);
            push_u16(&mut b, bits);
            push_tag(&mut b, b"data");
            push_u32(&mut b, payload.len() as u32);
            b.extend_from_slice(payload);
            b
        };

        // 8-bit is unsigned and centred on 128, unlike every other integer depth.
        let u8w = decode_wav(&make(1, 8, &[192])).expect("u8");
        assert!(approx(u8w.samples[0], 0.5));

        // 24-bit: 0x400000 is a quarter of full scale, and must sign-extend.
        let s24 = decode_wav(&make(1, 24, &[0x00, 0x00, 0x40])).expect("s24");
        assert!(approx(s24.samples[0], 0.5));
        let neg24 = decode_wav(&make(1, 24, &[0x00, 0x00, 0xC0])).expect("neg24");
        assert!(approx(neg24.samples[0], -0.5));

        let s32 = decode_wav(&make(1, 32, &[0x00, 0x00, 0x00, 0x40])).expect("s32");
        assert!(approx(s32.samples[0], 0.5));

        // Format 3 is IEEE float, already in [-1, 1].
        let f32w = decode_wav(&make(3, 32, &[0x00, 0x00, 0x00, 0xBF])).expect("f32");
        assert!(approx(f32w.samples[0], -0.5));
    }

    #[test]
    fn decode_wav_rejects_what_it_cannot_read() {
        assert!(decode_wav(&[]).is_err());
        assert!(decode_wav(&[0u8; 64]).is_err());

        // A valid header with a compressed format is a clear error rather than a
        // silent misread of the bytes.
        let mut bytes = encode_wav(&tone(240, 220.0, 0.3), 24000, 1);
        bytes[20] = 2; // WAVE_FORMAT_ADPCM
        assert!(decode_wav(&bytes).is_err());
    }

    #[test]
    fn read_wav_reports_a_missing_file() {
        assert!(read_wav("/nonexistent/path/to/nothing.wav").is_err());
    }
}
