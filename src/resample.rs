// =============================================================================
// Sample rate conversion -- windowed-sinc polyphase
// =============================================================================
//
// The OmniVoice codec runs its acoustic path at 24 kHz and its semantic path at
// 16 kHz, off the same waveform. Something has to convert between them, and
// which resampler it is matters: the semantic model's features feed a
// quantizer, so a resampler with a different filter than the reference's puts
// the latents somewhere the codebooks were never fit.
//
// This is a port of `torchaudio.functional.resample` at its defaults --
// `lowpass_filter_width=6`, `rolloff=0.99`, `sinc_interp_hann`.
//
// ## The idea
//
// Converting L to M is band-limited interpolation: upsample by M, low-pass,
// downsample by L. Doing that literally would build a signal M times too long.
// The polyphase form never does: it notes that only every L-th sample of the
// filtered signal survives, so it precomputes M filter *phases* -- one per
// output position within a cycle -- and each output sample is one dot product
// against the input at stride L.
//
//   kernel[p][k] = sinc(base * (k/L - p/M)) * hann(...) * base/L
//   out[i*M + p] = sum_k padded[i*L + k] * kernel[p][k]
//
// with `L = orig/gcd`, `M = new/gcd`, so 24 kHz to 16 kHz is 3 to 2: two
// phases, a 23-tap kernel, and one multiply-add pass over the input.
//
// ## The parts that are conventions, not mathematics
//
// The window is `cos(t*pi/width/2)^2` -- a Hann window in the sinc's own
// argument, not in sample index. `rolloff` pulls the cutoff to 0.99 of Nyquist
// so the transition band has somewhere to live. The kernel is clamped to
// +-`lowpass_filter_width` zero crossings before windowing, which is what makes
// it finite. All three are torchaudio's defaults and all three change the
// output if altered, so they are exposed but should be left alone.

#![allow(dead_code)]

/// torchaudio's default `lowpass_filter_width`.
pub const DEFAULT_LOWPASS_FILTER_WIDTH: usize = 6;
/// torchaudio's default `rolloff`.
pub const DEFAULT_ROLLOFF: f64 = 0.99;

/// The filter bank for one rate conversion: `phases` kernels of `taps` each.
///
/// Building it is the expensive part and depends only on the rates, so it is
/// separated from applying it.
#[derive(Clone, Debug)]
pub struct ResampleKernel {
    /// `new_freq / gcd` -- the number of output samples per cycle.
    pub phases: usize,
    /// `orig_freq / gcd` -- the input stride between cycles.
    pub stride: usize,
    /// Taps per phase: `2 * width + stride`.
    pub taps: usize,
    /// How far the kernel reaches back before its first tap.
    pub width: usize,
    /// `phases * taps`, phase-major.
    pub data: Vec<f32>,
}

impl Default for ResampleKernel {
    fn default() -> Self {
        ResampleKernel { phases: 1, stride: 1, taps: 0, width: 0, data: Vec::new() }
    }
}

impl ResampleKernel {
    pub fn phase(&self, p: usize) -> &[f32] {
        &self.data[p * self.taps..(p + 1) * self.taps]
    }
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// Build the polyphase filter bank for `orig_freq` to `new_freq`.
///
/// torchaudio's defaults are `lowpass_filter_width = 6` and `rolloff = 0.99`
/// (`DEFAULT_LOWPASS_FILTER_WIDTH`, `DEFAULT_ROLLOFF`).
pub fn make_resample_kernel(
    orig_freq: usize,
    new_freq: usize,
    lowpass_filter_width: usize,
    rolloff: f64,
) -> ResampleKernel {
    let mut k = ResampleKernel::default();
    if orig_freq == 0 || new_freq == 0 || lowpass_filter_width == 0 {
        return k;
    }

    // Reduce the ratio first: 24 kHz to 16 kHz is 3 to 2, not 24000 to 16000,
    // and the kernel is sized off the reduced numbers.
    let g = gcd(orig_freq, new_freq);
    let l = orig_freq / g;
    let m = new_freq / g;

    let width_f = lowpass_filter_width as f64;
    // The cutoff sits at the lower of the two Nyquists, pulled in by rolloff so
    // the filter's transition band has room below it.
    let base = l.min(m) as f64 * rolloff;
    let width = (width_f * l as f64 / base).ceil() as usize;

    k.phases = m;
    k.stride = l;
    k.width = width;
    k.taps = 2 * width + l;
    k.data = vec![0.0f32; k.phases * k.taps];

    let scale = base / l as f64;
    for p in 0..m {
        // Phase p samples the filter offset by p/M of an output period.
        let phase = -(p as f64) / m as f64;
        for j in 0..k.taps {
            let idx = (j as f64 - width as f64) / l as f64;
            // Clamping to +-width zero crossings is what truncates the sinc to
            // a finite kernel; the window then tapers what is left.
            let mut t = ((phase + idx) * base).clamp(-width_f, width_f);
            let w = (t * std::f64::consts::PI / width_f / 2.0).cos().powf(2.0);
            t *= std::f64::consts::PI;
            let sinc = if t == 0.0 { 1.0 } else { t.sin() / t };
            k.data[p * k.taps + j] = (sinc * w * scale) as f32;
        }
    }
    k
}

/// Resample with a kernel built ahead of time, for callers converting many
/// clips between the same pair of rates.
pub fn resample_with(input: &[f32], kernel: &ResampleKernel) -> Vec<f32> {
    if input.is_empty() || kernel.taps == 0 {
        return Vec::new();
    }

    let n = input.len();
    let l = kernel.stride;
    let m = kernel.phases;

    // Zero-pad by `width` in front and `width + stride` behind, so the first
    // and last output samples see a full kernel. That is what makes the output
    // length exactly floor(n / stride) + 1 cycles.
    let pad_front = kernel.width;
    let cycles = n / l + 1;

    let target = ((m as u64 * n as u64 + l as u64 - 1) / l as u64) as usize;
    let mut out = vec![0.0f32; cycles * m];

    for c in 0..cycles {
        let start = c * l;
        for p in 0..m {
            let tap = kernel.phase(p);
            let mut acc = 0.0f32;
            for (j, &coef) in tap.iter().enumerate() {
                // The pad is conceptual: samples outside the input are zero, so
                // they are skipped rather than materialised.
                let src = start + j;
                if src < pad_front {
                    continue;
                }
                let i = src - pad_front;
                if i >= n {
                    break;
                }
                acc += input[i] * coef;
            }
            out[c * m + p] = acc;
        }
    }

    out.truncate(target.min(out.len()));
    out
}

/// Resample `input` from `orig_freq` to `new_freq`.
///
/// Returns exactly `ceil(new_freq * n / orig_freq)` samples. Equal rates return
/// the input untouched, which is not merely an optimisation -- it is what
/// torchaudio does, and passing 1:1 audio through the filter would soften it
/// for no reason.
pub fn resample(input: &[f32], orig_freq: usize, new_freq: usize) -> Vec<f32> {
    if orig_freq == new_freq {
        return input.to_vec();
    }
    resample_with(
        input,
        &make_resample_kernel(orig_freq, new_freq, DEFAULT_LOWPASS_FILTER_WIDTH, DEFAULT_ROLLOFF),
    )
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

    fn default_kernel(from: usize, to: usize) -> ResampleKernel {
        make_resample_kernel(from, to, DEFAULT_LOWPASS_FILTER_WIDTH, DEFAULT_ROLLOFF)
    }

    /// A sine at `freq` sampled at `rate`.
    fn sine(n: usize, freq: f32, rate: f32, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq * (i as f32 / rate)).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        let mut sum = 0.0f64;
        for &v in x {
            sum += v as f64 * v as f64;
        }
        if x.is_empty() { 0.0 } else { (sum / x.len() as f64).sqrt() as f32 }
    }

    // =========================================================================
    // The kernel
    // =========================================================================

    #[test]
    fn the_24khz_to_16khz_kernel_is_two_phases_of_23_taps() {
        let k = default_kernel(24000, 16000);
        // 24000:16000 reduces to 3:2, so three input samples make two output ones.
        assert_eq!(k.stride, 3);
        assert_eq!(k.phases, 2);
        // width = ceil(6 * 3 / (2 * 0.99)) = 10, taps = 2 * width + stride.
        assert_eq!(k.width, 10);
        assert_eq!(k.taps, 23);
        assert_eq!(k.data.len(), 46);
    }

    #[test]
    fn the_kernel_matches_the_reference_filter() {
        let k = default_kernel(24000, 16000);
        let p0 = k.phase(0);
        let p1 = k.phase(1);

        // Phase 0 lands exactly on an input sample, so its centre tap is the bare
        // scale factor base/stride = 1.98/3 and it is symmetric about it.
        assert!(approx(p0[10], 0.66));
        assert!(approx(p0[9], 0.270691812));
        assert!(approx(p0[11], 0.270691812));
        assert!(approx(p0[12], -0.118959866));
        // The window is zero at both ends, which is what makes the truncation
        // inaudible.
        assert!(approx(p0[0], 0.0));
        assert!(approx(p0[22], 0.0));

        // Phase 1 sits half an output period along, so it has no centre tap.
        assert!(approx(p1[10], 0.00622774707));
        assert!(approx(p1[11], 0.543885589));
    }

    #[test]
    fn every_phase_has_unit_dc_gain() {
        // Each output sample is one phase's dot product, so each phase has to sum
        // to one on its own or a constant signal would come out scaled.
        for (from, to) in [(24000usize, 16000usize), (16000, 24000), (44100, 16000)] {
            let k = default_kernel(from, to);
            for p in 0..k.phases {
                let mut sum = 0.0f32;
                for &v in k.phase(p) {
                    sum += v;
                }
                assert!((sum - 1.0).abs() < 0.01);
            }
        }
    }

    // =========================================================================
    // Length
    // =========================================================================

    #[test]
    fn the_output_length_is_exactly_ceil_new_n_over_orig() {
        assert_eq!(resample(&sine(1000, 440.0, 24000.0, 0.5), 24000, 16000).len(), 667);
        assert_eq!(resample(&sine(1000, 440.0, 16000.0, 0.5), 16000, 24000).len(), 1500);
        assert_eq!(resample(&sine(999, 440.0, 48000.0, 0.5), 48000, 24000).len(), 500);
        // A ratio that does not reduce to anything small: 22050:16000 is 441:320.
        assert_eq!(resample(&sine(1000, 440.0, 22050.0, 0.5), 22050, 16000).len(), 726);
    }

    #[test]
    fn equal_rates_pass_the_samples_through_untouched() {
        let x = sine(64, 440.0, 24000.0, 0.5);
        let y = resample(&x, 24000, 24000);
        assert_eq!(y.len(), x.len());
        for i in 0..x.len() {
            // Bit-exact, not approximate: running 1:1 audio through the filter
            // would soften it for nothing.
            assert_eq!(y[i], x[i]);
        }
    }

    #[test]
    fn an_empty_input_resamples_to_nothing() {
        assert!(resample(&[], 24000, 16000).is_empty());
    }

    // =========================================================================
    // Signal
    // =========================================================================

    #[test]
    fn a_sine_survives_24khz_to_16khz() {
        // 440 Hz is far below either Nyquist, so the filter should leave it alone.
        let x = sine(4800, 440.0, 24000.0, 0.5);
        let y = resample(&x, 24000, 16000);
        assert_eq!(y.len(), 3200);

        let want = sine(3200, 440.0, 16000.0, 0.5);
        // Skip the kernel's reach at each end, where the zero padding shows.
        let mut i = 32;
        while i + 32 < y.len() {
            assert!((y[i] - want[i]).abs() < 2e-3);
            i += 1;
        }
        assert!((rms(&y) - rms(&x)).abs() < 1e-3);
    }

    #[test]
    fn a_constant_stays_constant() {
        let x = vec![0.25f32; 3000];
        let y = resample(&x, 24000, 16000);
        let mut i = 32;
        while i + 32 < y.len() {
            assert!((y[i] - 0.25).abs() < 1e-3);
            i += 1;
        }
    }

    #[test]
    fn upsampling_is_the_same_machinery() {
        let x = sine(3200, 440.0, 16000.0, 0.5);
        let y = resample(&x, 16000, 24000);
        assert_eq!(y.len(), 4800);

        let want = sine(4800, 440.0, 24000.0, 0.5);
        let mut i = 32;
        while i + 32 < y.len() {
            assert!((y[i] - want[i]).abs() < 2e-3);
            i += 1;
        }
    }

    #[test]
    fn content_above_the_new_nyquist_is_filtered_out() {
        // 10 kHz fits under 24 kHz's Nyquist but not under 16 kHz's, so the filter
        // has to remove it rather than let it fold back down as an audible alias.
        let x = sine(4800, 10000.0, 24000.0, 0.5);
        let y = resample(&x, 24000, 16000);
        let interior = &y[64..y.len() - 64];
        assert!(rms(interior) < 0.05);
    }

    #[test]
    fn down_then_up_returns_the_signal() {
        let x = sine(4800, 440.0, 24000.0, 0.5);
        let round = resample(&resample(&x, 24000, 16000), 16000, 24000);
        assert_eq!(round.len(), x.len());
        let mut i = 64;
        while i + 64 < round.len() {
            assert!((round[i] - x[i]).abs() < 5e-3);
            i += 1;
        }
    }

    #[test]
    fn a_prebuilt_kernel_gives_the_same_answer() {
        let x = sine(600, 440.0, 24000.0, 0.5);
        let k = default_kernel(24000, 16000);
        let a = resample(&x, 24000, 16000);
        let b = resample_with(&x, &k);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i], b[i]);
        }
    }
}
