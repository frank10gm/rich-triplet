// =============================================================================
// OmniVoice -- masked diffusion text to speech
// =============================================================================
//
// Ties `transformer6.rs` (the bidirectional Qwen3 backbone) to
// `omnivoice_codec.rs` (the acoustic decoder).
//
// ## The sequence
//
// A conditional pass is laid out as:
//
//   <|lang_start|>{lang}<|lang_end|><|instruct_start|>{instruct}<|instruct_end|>
//   <|text_start|>{text}<|text_end|>
//   [ target frames, all masked ]
//
// The unconditional pass for classifier-free guidance is the **target frames
// alone** -- no style, no text. Both run through the model each step, and their
// log-probabilities are combined.
//
// ## Voice cloning
//
// A reference clip does not add a mode. It adds a prefix:
//
//   <|denoise|><|lang_start|>...<|instruct_end|>
//   <|text_start|>{ref_text} {text}<|text_end|>
//   [ reference frames, decided ][ target frames, all masked ]
//
// so the model is continuing a recording it can see rather than imitating one
// it cannot, and every masked position attends to the reference through the
// same bidirectional attention it uses for everything else. The unconditional
// pass is unchanged -- still the masked frames alone -- which is what makes the
// guidance push *towards* the reference voice.
//
// ## The loop
//
// Every audio position starts masked. Each step runs both passes, scores every
// (codebook, position) pair by confidence, and unmasks the `k` best, where `k`
// comes from a timestep schedule. After `num_step` steps everything is decided.
//
//   log_probs = log_softmax(c + guidance * (c - u))     both already log_softmax
//   scores    = max(log_probs) - codebook_index * layer_penalty
//   unmask the top k of scores, ignoring positions already decided
//
// The layer penalty is what makes decoding coarse to fine: codebook 0 is
// unpenalised, so it is decided first and the later codebooks condition on it.
//
// ## Cost
//
// Nothing is cached, because unmasking a position changes every position that
// attends to it. So this is `num_step * 2` full-sequence forward passes -- 24
// at the defaults. That is more arithmetic than an autoregressive model of the
// same size would do, but it is all batched matmul rather than memory-bound
// GEMV, which is the shape BLAS is good at.
//
// ## Backends
//
// `omni_synthesize` runs its forward passes through `OmniForward`, so a GPU
// backend can stand in for `OmniLm`'s CPU pass without the loop noticing. The
// model is still needed for its config and its prompt layout.

#![allow(dead_code)]

use std::time::Instant;

use crate::duration::estimate_duration;
use crate::nn::InitRng;
use crate::omnivoice_codec::{OmniCodecConfig, OmniCodecDecoder};
use crate::resample::resample;
use crate::tokenizer::{HfBpeTokenizer, Tokenizer};
use crate::transformer6::{Config6, OmniForward, OmniLm, OmniToken};

// =============================================================================
// Generation config
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub struct OmniGenConfig {
    /// Unmasking steps.
    ///
    /// The reference ships 32. This is 12, which is three times faster and
    /// was judged indistinguishable by ear on Italian -- a listening call,
    /// since every number this project prints looks the same across the range.
    /// `--steps 32` restores the reference's setting.
    pub num_step: usize,
    /// Classifier-free guidance strength; 0 disables the unconditional pass
    /// entirely and halves the work.
    pub guidance_scale: f32,
    /// Gumbel noise added to the *position* scores, so the choice of which
    /// positions to unmask is stochastic even though the tokens are argmax.
    pub position_temperature: f32,
    /// Gumbel noise on the token choice itself. 0 means argmax.
    pub class_temperature: f32,
    /// Penalty per codebook index, pushing the coarse codebooks to be decided
    /// first.
    pub layer_penalty_factor: f32,
    /// Warps the timestep schedule; below 1 unmasks more early.
    pub t_shift: f32,
    pub seed: u64,

    // -- long text -----------------------------------------------------------
    //
    // The model was not trained to hold a voice across a monologue, and past
    // roughly half a minute a single pass stops sounding like speech at all --
    // the frames are there and the model runs out of things to put in them.
    // So long text is split, generated piece by piece, and joined.
    /// Split when the estimate exceeds this many seconds. Zero never splits.
    pub chunk_threshold_seconds: f32,
    /// Roughly how much audio one chunk should carry.
    pub chunk_seconds: f32,
    /// Silence between one chunk and the next. The pieces break at sentence
    /// ends, so this is the pause a reader would take there.
    pub chunk_gap_seconds: f32,
}

impl Default for OmniGenConfig {
    fn default() -> Self {
        OmniGenConfig {
            num_step: 12,
            guidance_scale: 2.0,
            position_temperature: 5.0,
            class_temperature: 0.0,
            layer_penalty_factor: 5.0,
            t_shift: 0.1,
            seed: 0,
            chunk_threshold_seconds: 30.0,
            chunk_seconds: 15.0,
            chunk_gap_seconds: 0.3,
        }
    }
}

impl OmniGenConfig {
    pub fn defaults() -> Self {
        OmniGenConfig::default()
    }
}

// =============================================================================
// Schedules
// =============================================================================

/// The warped timestep grid: `num_step + 1` values from 0 to 1.
///
/// `t_shift * t / (1 + (t_shift - 1) * t)` bends a linear grid so that with
/// `t_shift < 1` the early steps cover more of the interval, unmasking more
/// tokens up front.
pub fn omni_time_steps(num_step: usize, t_shift: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; num_step + 1];
    for (i, slot) in out.iter_mut().enumerate() {
        let t = i as f32 / num_step as f32;
        // Warp: below 1, t_shift front-loads the schedule.
        let denom = 1.0f32 + (t_shift - 1.0f32) * t;
        *slot = if denom == 0.0 { t } else { (t_shift * t) / denom };
    }
    out
}

/// How many (codebook, position) pairs to unmask at each step.
///
/// Sums to exactly `total` -- the last step takes whatever is left, so nothing
/// can be dropped by rounding.
pub fn omni_unmask_schedule(total: usize, num_step: usize, t_shift: f32) -> Vec<usize> {
    let steps = omni_time_steps(num_step, t_shift);
    let mut sched = Vec::with_capacity(num_step);

    let mut remaining = total;
    for i in 0..num_step {
        let n = if i + 1 == num_step {
            // The last step takes everything left, so rounding can never leave
            // a position masked.
            remaining
        } else {
            let share = steps[i + 1] - steps[i];
            let want = (total as f64 * share as f64).ceil() as usize;
            want.min(remaining)
        };
        sched.push(n);
        remaining -= n;
    }
    sched
}

// =============================================================================
// Requests
// =============================================================================

#[derive(Clone, Debug, Default)]
pub struct OmniRequest {
    pub text: String,
    /// Language hint, e.g. "Italian". Empty becomes "None".
    pub language: String,
    /// Free-text voice description, e.g. "a calm young woman". Empty becomes
    /// "None".
    pub instruct: String,
    /// Audio seconds to generate. Zero asks for the length heuristic.
    pub duration_seconds: f32,

    // -- voice cloning -------------------------------------------------------
    //
    // A reference clip is not a separate mode. Its codes are prepended to the
    // target frames as *already decided* positions and its transcript to the
    // text, so the model is asked to continue a recording it can see rather
    // than to imitate one it cannot. Everything else about the loop is the
    // same.
    /// What the reference clip says. Required alongside `ref_codes`: the model
    /// has to know which part of the text it has already heard.
    pub ref_text: String,
    /// The reference clip encoded, `[codebook][frame]`. Empty means no clone.
    pub ref_codes: Vec<Vec<u32>>,
    /// The reference clip's loudness before it was levelled for the encoder.
    /// Zero means unknown, which leaves the output gain alone.
    pub ref_rms: f32,

    pub generation: OmniGenConfig,
    pub debug: bool,
}

impl OmniRequest {
    /// Frames of reference audio, or zero.
    pub fn ref_frames(&self) -> usize {
        self.ref_codes.first().map_or(0, |s| s.len())
    }
}

#[derive(Clone, Debug)]
pub struct OmniResult {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub frames: usize,
    pub prompt_tokens: usize,
    pub forward_passes: usize,
    /// Pieces the text was split into; 1 when it was short enough to run whole.
    pub chunks: usize,
    pub generate_seconds: f64,
    pub decode_seconds: f64,
}

impl Default for OmniResult {
    fn default() -> Self {
        OmniResult {
            samples: Vec::new(),
            sample_rate: 24000,
            frames: 0,
            prompt_tokens: 0,
            forward_passes: 0,
            chunks: 1,
            generate_seconds: 0.0,
            decode_seconds: 0.0,
        }
    }
}

impl OmniResult {
    pub fn audio_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate as f64
        }
    }

    pub fn realtime_factor(&self) -> f64 {
        let audio = self.audio_seconds();
        if audio <= 0.0 { 0.0 } else { (self.generate_seconds + self.decode_seconds) / audio }
    }
}

// =============================================================================
// Prompt
// =============================================================================

/// Estimate how many frames a piece of text needs.
///
/// The reference's rule-based estimator, ported in `duration.rs`: a phonetic
/// weight per character, scaled against "Nice to meet you." at 25 frames.
/// `--duration` still overrides it, because getting the length wrong is
/// audible -- too few frames truncates mid-word, too many and the model runs
/// out of things to say and collapses into silence.
pub fn omni_estimate_frames(text: &str, codec: &OmniCodecConfig) -> usize {
    let rate = codec.sample_rate as f64 / codec.hop_length as f64;

    // The reference calibrates against "Nice to meet you." at 25 audio tokens,
    // and its codec runs at 25 Hz -- so the reference phrase is one second of
    // speech, and the threshold below which short text gets boosted is two.
    // Expressing them that way keeps the numbers identical here and correct if
    // the frame rate ever changes.
    let est = estimate_duration(text, "Nice to meet you.", rate, 2.0 * rate, 3.0);

    // Truncates rather than rounds, matching the reference's `max(1, int(est))`.
    (est as usize).max(1)
}

/// The same estimate, calibrated on a reference clip instead of the built-in
/// phrase.
///
/// Strictly better when there is one: it measures this speaker's rate rather
/// than an average one, so a slow voice gets the frames it needs. Falls back to
/// the built-in reference when `ref_text` is empty or `ref_frames` is zero.
pub fn omni_estimate_frames_from_reference(
    text: &str,
    ref_text: &str,
    ref_frames: usize,
    codec: &OmniCodecConfig,
) -> usize {
    if ref_text.is_empty() || ref_frames == 0 {
        return omni_estimate_frames(text, codec);
    }
    let rate = codec.sample_rate as f64 / codec.hop_length as f64;
    let est = estimate_duration(text, ref_text, ref_frames as f64, 2.0 * rate, 3.0);
    (est as usize).max(1)
}

// =============================================================================
// Reference clips
// =============================================================================

/// A reference clip, ready for the codec.
#[derive(Clone, Debug, Default)]
pub struct OmniReference {
    /// Mono, at the codec's sample rate, and a whole number of frames long.
    pub samples: Vec<f32>,
    /// The clip's loudness *before* levelling, which the synthesised audio is
    /// scaled back to at the end.
    pub rms: f32,
}

impl OmniReference {
    /// Seconds of audio, for the caller to warn about.
    pub fn seconds(&self, codec: &OmniCodecConfig) -> f64 {
        if codec.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / codec.sample_rate as f64
        }
    }
}

/// Prepare a reference recording: resample to the codec's rate, level it, and
/// trim it to a whole number of frames.
///
/// The levelling matters more than it looks. The codec was fit on speech at a
/// particular loudness, and a quiet recording encodes into a part of the
/// codebook space that carries a quiet voice rather than that voice quietly.
/// So a clip under 0.1 RMS is brought up to it, and the original loudness is
/// restored on the way out.
pub fn omni_prepare_reference(
    samples: &[f32],
    sample_rate: usize,
    codec: &OmniCodecConfig,
) -> Result<OmniReference, String> {
    if samples.is_empty() {
        return Err("omnivoice: the reference clip is empty".into());
    }
    if sample_rate == 0 {
        return Err("omnivoice: the reference clip has no sample rate".into());
    }

    let mut r = OmniReference { samples: resample(samples, sample_rate, codec.sample_rate), rms: 0.0 };

    // Whole frames only: the encoder would drop the remainder anyway, and
    // doing it here keeps the length the caller sees honest.
    let frames = r.samples.len() / codec.hop_length;
    if frames == 0 {
        return Err(format!(
            "omnivoice: the reference clip is shorter than one frame at {} Hz",
            codec.sample_rate
        ));
    }
    r.samples.truncate(frames * codec.hop_length);

    let mut sum_sq = 0.0f64;
    for &v in &r.samples {
        sum_sq += v as f64 * v as f64;
    }
    r.rms = (sum_sq / r.samples.len() as f64).sqrt() as f32;

    const TARGET_RMS: f32 = 0.1;
    if r.rms > 0.0 && r.rms < TARGET_RMS {
        let gain = TARGET_RMS / r.rms;
        for v in r.samples.iter_mut() {
            *v *= gain;
        }
    }
    Ok(r)
}

// =============================================================================
// Text
// =============================================================================

/// The CJK Unified Ideographs block, where a space between characters is
/// typesetting rather than a word boundary.
fn is_cjk(cp: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&cp)
}

/// The reference's text normalisation, applied to the prompt with or without a
/// reference clip.
///
/// Trims, joins the reference transcript in front, drops line breaks, folds
/// runs of spaces and tabs, swaps fullwidth parentheses for ASCII ones, and
/// removes whitespace next to a CJK character -- where a space is a typesetting
/// artefact rather than a word boundary and would be spoken as a pause.
pub fn omni_combine_text(text: &str, ref_text: &str) -> String {
    // `str::trim` strips the Unicode White_Space property, as the C++ helper does.
    let trimmed_ref = ref_text.trim();
    let trimmed = text.trim();
    let joined = if trimmed_ref.is_empty() {
        trimmed.to_string()
    } else {
        format!("{} {}", trimmed_ref, trimmed)
    };

    // Walk once, applying every rule: line breaks vanish, fullwidth
    // parentheses become ASCII, runs of spaces and tabs collapse to one.
    let mut out: Vec<char> = Vec::new();
    for cp in joined.chars() {
        if cp == '\r' || cp == '\n' {
            continue;
        }
        if cp == '\u{FF08}' {
            out.push('(');
            continue;
        }
        if cp == '\u{FF09}' {
            out.push(')');
            continue;
        }
        if cp == ' ' || cp == '\t' {
            if out.last() == Some(&' ') {
                continue;
            }
            out.push(' ');
            continue;
        }
        out.push(cp);
    }

    // Then drop the spaces that sit against a CJK character on either side.
    let mut result = String::new();
    for i in 0..out.len() {
        if out[i] == ' ' {
            let cjk_before = i > 0 && is_cjk(out[i - 1]);
            let cjk_after = i + 1 < out.len() && is_cjk(out[i + 1]);
            if cjk_before || cjk_after {
                continue;
            }
        }
        result.push(out[i]);
    }
    result
}

/// Append the text tokens of `s`, if any.
fn append_text(out: &mut Vec<OmniToken>, tok: &HfBpeTokenizer, s: &str) {
    if s.is_empty() {
        return;
    }
    for id in tok.encode(s) {
        out.push(OmniToken::text(id as usize));
    }
}

/// Build the conditional sequence: style markers, text, reference codes if any,
/// then masked frames.
pub fn omni_build_conditional(
    tok: &HfBpeTokenizer,
    cfg: &Config6,
    request: &OmniRequest,
    frames: usize,
) -> Result<Vec<OmniToken>, String> {
    if request.text.is_empty() {
        return Err("omnivoice: prompt text is empty".into());
    }
    if frames == 0 {
        return Err("omnivoice: asked for zero frames".into());
    }

    let ref_frames = request.ref_frames();
    for stream in &request.ref_codes {
        if stream.len() != ref_frames {
            return Err("omnivoice: the reference codebooks disagree on length".into());
        }
    }
    if !request.ref_codes.is_empty() && request.ref_codes.len() != cfg.num_audio_codebook {
        return Err(format!(
            "omnivoice: the reference has {} codebooks, expected {}",
            request.ref_codes.len(),
            cfg.num_audio_codebook
        ));
    }
    if ref_frames > 0 && request.ref_text.is_empty() {
        return Err("omnivoice: a reference clip needs its transcript, or the model cannot tell \
                    which part of the text it has already heard"
            .into());
    }

    let mut seq: Vec<OmniToken> = Vec::new();

    // A reference clip is a recording, so the model is told to clean it up
    // rather than reproduce its room.
    if ref_frames > 0 {
        seq.push(OmniToken::text(cfg.denoise));
    }

    // Style block. The markers are injected as raw ids -- encoding the literal
    // "<|lang_start|>" would split it into ordinary subword pieces.
    seq.push(OmniToken::text(cfg.lang_start));
    append_text(&mut seq, tok, if request.language.is_empty() { "None" } else { &request.language });
    seq.push(OmniToken::text(cfg.lang_end));

    seq.push(OmniToken::text(cfg.instruct_start));
    append_text(&mut seq, tok, if request.instruct.is_empty() { "None" } else { &request.instruct });
    seq.push(OmniToken::text(cfg.instruct_end));

    // Text block: the reference transcript and the target text as one run, so
    // the model reads them as continuous speech.
    seq.push(OmniToken::text(cfg.text_start));
    append_text(&mut seq, tok, &omni_combine_text(&request.text, &request.ref_text));
    seq.push(OmniToken::text(cfg.text_end));

    // Reference frames, already decided. These are ordinary audio positions
    // carrying real codes rather than the mask, which is the whole mechanism:
    // the target frames attend to them like any other position.
    for t in 0..ref_frames {
        let mut token = OmniToken { text_id: 0, audio: vec![0u32; cfg.num_audio_codebook] };
        for c in 0..cfg.num_audio_codebook {
            let code = request.ref_codes[c][t];
            if code as usize >= cfg.audio_mask_id {
                return Err(format!("omnivoice: reference code {} is out of range", code));
            }
            token.audio[c] = code;
        }
        seq.push(token);
    }

    // Target frames, all masked.
    for _ in 0..frames {
        seq.push(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
    }
    Ok(seq)
}

/// Build the unconditional sequence: the masked frames alone.
pub fn omni_build_unconditional(cfg: &Config6, frames: usize) -> Vec<OmniToken> {
    // No style, no text: the unconditional branch sees only the frames it is
    // trying to fill.
    let mut seq = Vec::with_capacity(frames);
    for _ in 0..frames {
        seq.push(OmniToken::masked(cfg.num_audio_codebook, cfg.audio_mask_id));
    }
    seq
}

// =============================================================================
// Long text
// =============================================================================

/// Punctuation a sentence may end on. Fullwidth forms included, since a CJK
/// clause ends on the wide comma rather than the ASCII one.
fn is_split_punctuation(cp: char) -> bool {
    matches!(
        cp,
        '.' | ',' | ';' | ':' | '!' | '?'
            | '\u{3002}' // 。
            | '\u{FF0C}' // ，
            | '\u{FF1B}' // ；
            | '\u{FF1A}' // ：
            | '\u{FF01}' // ！
            | '\u{FF1F}' // ？
    )
}

/// Quotes and brackets that close a sentence *after* its full stop, and so
/// belong to the piece that just ended rather than starting the next one.
fn is_closing_mark(cp: char) -> bool {
    matches!(
        cp,
        '"' | '\''
            | '\u{201C}' | '\u{201D}' // “ ”
            | '\u{2018}' | '\u{2019}' // ‘ ’
            | '\u{FF09}' // ）
            | ']' | '>'
            | '\u{300B}' // 》
            | '\u{300D}' // 」
            | '\u{3011}' // 】
    )
}

/// Words whose trailing full stop is not the end of a sentence.
fn is_abbreviation(word: &str) -> bool {
    const ABBREVIATIONS: &[&str] = &[
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "Rev.", "Fr.", "Hon.", "Pres.", "Gov.",
        "Capt.", "Gen.", "Sen.", "Rep.", "Col.", "Maj.", "Lt.", "Cmdr.", "Sgt.", "Cpl.", "Co.",
        "Corp.", "Inc.", "Ltd.", "Est.", "Dept.", "St.", "Ave.", "Blvd.", "Rd.", "Mt.", "Ft.",
        "No.", "Jan.", "Feb.", "Mar.", "Apr.", "Aug.", "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
        "i.e.", "e.g.", "vs.", "Vs.", "Etc.", "approx.", "fig.", "def.",
    ];
    ABBREVIATIONS.contains(&word)
}

/// The last whitespace-separated word of a run of code points.
fn last_word(cps: &[char]) -> String {
    let mut end = cps.len();
    while end > 0 && cps[end - 1].is_whitespace() {
        end -= 1;
    }
    let mut begin = end;
    while begin > 0 && !cps[begin - 1].is_whitespace() {
        begin -= 1;
    }
    cps[begin..end].iter().collect()
}

/// Split text into pieces of about `chunk_chars` characters, breaking at
/// punctuation.
///
/// Breaking mid-sentence would put a seam where the prosody is still rising,
/// so splits only happen after `.,;:!?` and their fullwidth counterparts. A
/// full stop that ends a known abbreviation -- "Dr.", "e.g.", "No." -- is not a
/// break, since splitting there would strand a title from its name.
///
/// Sentences are then merged greedily up to `chunk_chars`, so a chunk overruns
/// only when a single sentence already does. Pieces shorter than
/// `min_chunk_chars` are folded into a neighbour: a two-character chunk would
/// be given its own speaker and sound like one. The C++ default for
/// `min_chunk_chars` is `DEFAULT_MIN_CHUNK_CHARS`.
pub fn omni_chunk_text(text: &str, chunk_chars: usize, min_chunk_chars: usize) -> Vec<String> {
    let cps: Vec<char> = text.chars().collect();
    if cps.is_empty() {
        return Vec::new();
    }
    if chunk_chars == 0 {
        return vec![text.to_string()];
    }

    // 1. Break into sentences at punctuation, keeping the mark with what it
    //    ends.
    let mut sentences: Vec<Vec<char>> = Vec::new();
    let mut current: Vec<char> = Vec::new();
    for &cp in &cps {
        if current.is_empty() && !sentences.is_empty() && (is_split_punctuation(cp) || is_closing_mark(cp))
        {
            // A mark that opens a new sentence closes the previous one.
            sentences.last_mut().unwrap().push(cp);
            continue;
        }
        current.push(cp);
        if is_split_punctuation(cp) {
            if cp == '.' && is_abbreviation(&last_word(&current)) {
                continue;
            }
            sentences.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        sentences.push(current);
    }

    // 2. Merge greedily up to the target size.
    let mut merged: Vec<Vec<char>> = Vec::new();
    let mut chunk: Vec<char> = Vec::new();
    for sentence in sentences {
        if chunk.len() + sentence.len() <= chunk_chars {
            chunk.extend_from_slice(&sentence);
        } else {
            if !chunk.is_empty() {
                merged.push(std::mem::take(&mut chunk));
            }
            chunk = sentence;
        }
    }
    if !chunk.is_empty() {
        merged.push(chunk);
    }

    // 3. Fold away anything too short to be worth its own generation.
    let mut final_chunks: Vec<Vec<char>> = Vec::new();
    let first_is_short = !merged.is_empty() && merged[0].len() < min_chunk_chars;
    for (i, piece) in merged.into_iter().enumerate() {
        if i == 1 && first_is_short {
            // A short opener keeps the piece after it rather than the reverse.
            final_chunks.last_mut().unwrap().extend_from_slice(&piece);
        } else if piece.len() >= min_chunk_chars || final_chunks.is_empty() {
            final_chunks.push(piece);
        } else {
            final_chunks.last_mut().unwrap().extend_from_slice(&piece);
        }
    }

    let mut out = Vec::new();
    for c in &final_chunks {
        let s: String = c.iter().collect();
        let s = s.trim();
        if !s.is_empty() {
            out.push(s.to_string());
        }
    }
    out
}

/// `omni_chunk_text`'s C++ default for `min_chunk_chars`.
pub const DEFAULT_MIN_CHUNK_CHARS: usize = 3;

/// The span of `samples` that carries signal, widened by `margin` samples at
/// each end and clamped.
///
/// Loudness is RMS over 10 ms windows rather than a per-sample peak, which is
/// what the reference's silence detector uses too. The difference is not
/// theoretical: a chunk's quiet head crosses -50 dBFS on the odd sample while
/// averaging well below it, so a peak test keeps everything from the first
/// crossing and leaves half a second of near-silence sitting in a join. A
/// window still keeps a genuinely loud transient, and should -- that is
/// content, not noise.
///
/// Returns `(begin, end)`, and `begin == end` when nothing is above
/// `threshold_db`. The C++ defaults are `threshold_db = -50` and `margin = 0`.
pub fn omni_voiced_span(
    samples: &[f32],
    sample_rate: usize,
    threshold_db: f32,
    margin: usize,
) -> (usize, usize) {
    // -50 dBFS is the reference's silence threshold: about 1/316 of full scale,
    // below anything the codec produces for speech and above its noise floor.
    let threshold = 10.0f32.powf(threshold_db / 20.0f32);
    let win = (sample_rate / 100).max(1); // 10 ms
    let frames = samples.len() / win;
    if frames == 0 {
        return (0, samples.len());
    }

    let loud = |f: usize| -> bool {
        let mut sum = 0.0f64;
        for &v in &samples[f * win..(f + 1) * win] {
            sum += v as f64 * v as f64;
        }
        (sum / win as f64).sqrt() >= threshold as f64
    };

    let mut first = 0usize;
    while first < frames && !loud(first) {
        first += 1;
    }
    if first == frames {
        return (0, 0);
    }
    let mut last = frames;
    while last > first && !loud(last - 1) {
        last -= 1;
    }

    let begin_raw = first * win;
    let end_raw = samples.len().min(last * win);
    (
        if begin_raw > margin { begin_raw - margin } else { 0 },
        samples.len().min(end_raw + margin),
    )
}

/// Join chunk waveforms into one utterance.
///
/// Not an overlap-add: the pieces are separate utterances, not a continuous
/// signal cut in two, so crossfading them onto each other would sound like two
/// people talking over one another. Each is trimmed to what it actually says,
/// and a fixed silence is placed between them.
///
/// **Trimming is what makes the joins even.** A chunk's length comes from a
/// duration estimate, so it ends with however much silence the estimate
/// overshot by -- measured across four joins of one clip, the pause came out at
/// 113, 127, 191 and 649 ms for the same intended gap. Cutting each piece back
/// to its own speech makes every join the pause it was asked for.
///
/// The fade at each edge is a few milliseconds, only enough to stop a click.
/// The reference fades a tenth of a second, which is long enough to ramp the
/// last syllable of a chunk to nothing when the estimate was tight -- and it
/// usually is.
///
/// Both departures are confirmed by listening, not only by the measurements
/// above. That matters here for the same reason it matters for the RoPE
/// pairing: the numbers can show a pause is uneven or a fade lands on speech,
/// but not that the result sounds better. The C++ default gap is 0.3 s.
pub fn omni_cross_fade(chunks: &[Vec<f32>], sample_rate: usize, gap_seconds: f32) -> Vec<f32> {
    if chunks.is_empty() {
        return Vec::new();
    }
    if chunks.len() == 1 {
        // One piece is the whole utterance; nothing to join and nothing to
        // trim, so it passes through untouched.
        return chunks[0].clone();
    }

    // Long enough to stop a click at a discontinuity, short enough that it
    // cannot swallow a syllable. A tenth of a second -- what the reference
    // uses -- is not short enough.
    let fade = (0.008f32 * sample_rate as f32) as usize;
    // `std::max(0.0f, gap_seconds)`.
    let gap_clamped = if 0.0f32 < gap_seconds { gap_seconds } else { 0.0f32 };
    let gap = (gap_clamped * sample_rate as f32) as usize;

    let mut out: Vec<f32> = Vec::new();
    for chunk in chunks {
        // Trim to what this piece actually says, keeping the fade's worth of
        // room so the ramp runs over near-silence rather than over speech.
        let (begin, end) = omni_voiced_span(chunk, sample_rate, -50.0, fade);
        if begin == end {
            continue; // a piece with nothing in it contributes nothing
        }
        let len = end - begin;

        if !out.is_empty() {
            out.extend(std::iter::repeat_n(0.0f32, gap));
        }
        let at = out.len();
        out.extend_from_slice(&chunk[begin..end]);

        let n = fade.min(len / 2);
        for i in 0..n {
            let w = i as f32 / n as f32;
            out[at + i] *= w;
            out[at + len - 1 - i] *= w;
        }
    }
    out
}

// =============================================================================
// Sampling
// =============================================================================

/// Log-softmax a single codebook's slice, in place.
fn log_softmax(row: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in row.iter() {
        // `std::max(max_v, v)`.
        if max_v < v {
            max_v = v;
        }
    }
    let mut sum = 0.0f32;
    for &v in row.iter() {
        sum += (v - max_v).exp();
    }
    let log_sum = max_v + sum.ln();
    for v in row.iter_mut() {
        *v -= log_sum;
    }
}

/// -log(-log(U)) -- a standard Gumbel draw.
fn gumbel(rng: &mut InitRng) -> f32 {
    let u = rng.next_f32().clamp(1e-9f32, 1.0f32 - 1e-7f32);
    -(-(u.ln())).ln()
}

// -----------------------------------------------------------------------------
// std::partial_sort, as libc++ implements it
// -----------------------------------------------------------------------------
//
// The unmasking step picks the k best of every (codebook, position) score with
// `std::partial_sort`. Which of several *tied* scores make the cut at the k-th
// place is decided by the heap libc++ builds, not by the scores -- and ties are
// real: a confident prediction's log-probability rounds to exactly 0, and with
// `position_temperature = 0` there is no noise to split them. So this is the
// libc++ algorithm step for step (`__partial_sort_impl`, `__make_heap` with its
// arithmetic-type "assume both children" path, `__sift_down`, `__sift_up`,
// `__sort_heap` via Floyd's `__pop_heap`), over `usize` elements with an
// element comparator `comp(a, b)`.

fn heap_sift_down<F: Fn(usize, usize) -> bool>(
    v: &mut [usize],
    comp: &F,
    len: isize,
    start: isize,
    assume_both_children: bool,
) {
    let mut start = start;
    let mut child = start;
    if len < 2 || (len - 2) / 2 < child {
        return;
    }
    child = 2 * child + 1;
    if assume_both_children {
        child += comp(v[child as usize], v[child as usize + 1]) as isize;
    } else if child + 1 < len && comp(v[child as usize], v[child as usize + 1]) {
        child += 1;
    }
    // Already in heap order.
    if comp(v[child as usize], v[start as usize]) {
        return;
    }

    let top = v[start as usize];
    loop {
        v[start as usize] = v[child as usize];
        start = child;
        if (len - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        if assume_both_children {
            child += comp(v[child as usize], v[child as usize + 1]) as isize;
        } else if child + 1 < len && comp(v[child as usize], v[child as usize + 1]) {
            child += 1;
        }
        if comp(v[child as usize], top) {
            break;
        }
    }
    v[start as usize] = top;
}

/// `__sift_up(first, first + last, comp, len)`.
fn heap_sift_up<F: Fn(usize, usize) -> bool>(v: &mut [usize], last: isize, comp: &F, len: isize) {
    if len > 1 {
        let mut len = (len - 2) / 2;
        let mut ptr = len;
        let mut last = last - 1;
        if comp(v[ptr as usize], v[last as usize]) {
            let t = v[last as usize];
            loop {
                v[last as usize] = v[ptr as usize];
                last = ptr;
                if len == 0 {
                    break;
                }
                len = (len - 1) / 2;
                ptr = len;
                if !comp(v[ptr as usize], t) {
                    break;
                }
            }
            v[last as usize] = t;
        }
    }
}

fn heap_make<F: Fn(usize, usize) -> bool>(v: &mut [usize], comp: &F) {
    let n = v.len() as isize;
    // `usize` is arithmetic, so libc++ takes the "assume both children" path:
    // sift an odd-length prefix, then sift the last element up.
    let sift_down_n = if n & 1 == 1 { n } else { n - 1 };
    if n > 1 {
        let mut start = (sift_down_n - 2) / 2;
        while start >= 0 {
            heap_sift_down(v, comp, sift_down_n, start, true);
            start -= 1;
        }
        heap_sift_up(v, n, comp, n);
    }
}

fn heap_floyd_sift_down<F: Fn(usize, usize) -> bool>(v: &mut [usize], comp: &F, len: isize) -> isize {
    let mut hole = 0isize;
    let mut child_i = 0isize;
    let mut child = 0isize;
    loop {
        child_i += child + 1;
        child = 2 * child + 1;
        if child + 1 < len && comp(v[child_i as usize], v[child_i as usize + 1]) {
            child_i += 1;
            child += 1;
        }
        v[hole as usize] = v[child_i as usize];
        hole = child_i;
        if child > (len - 2) / 2 {
            return hole;
        }
    }
}

fn heap_pop<F: Fn(usize, usize) -> bool>(v: &mut [usize], last: isize, comp: &F, len: isize) {
    if len > 1 {
        let top = v[0];
        let hole = heap_floyd_sift_down(v, comp, len);
        let last = last - 1;
        if hole == last {
            v[hole as usize] = top;
        } else {
            v[hole as usize] = v[last as usize];
            let hole = hole + 1;
            v[last as usize] = top;
            heap_sift_up(v, hole, comp, hole);
        }
    }
}

fn heap_sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], comp: &F) {
    let mut last = v.len() as isize;
    let mut n = last;
    while n > 1 {
        heap_pop(v, last, comp, n);
        last -= 1;
        n -= 1;
    }
}

/// `std::partial_sort(v.begin(), v.begin() + middle, v.end(), comp)`.
fn libcxx_partial_sort<F: Fn(usize, usize) -> bool>(v: &mut [usize], middle: usize, comp: F) {
    if middle == 0 {
        return;
    }
    heap_make(&mut v[..middle], &comp);
    let len = middle as isize;
    for i in middle..v.len() {
        if comp(v[i], v[0]) {
            v.swap(i, 0);
            heap_sift_down(&mut v[..middle], &comp, len, 0, false);
        }
    }
    heap_sort(&mut v[..middle], &comp);
}

// =============================================================================
// Pipeline
// =============================================================================

/// One piece of text, start to finish: estimate its length, run the unmasking
/// loop, and hand back the codes. Chunked generation calls this once per piece
/// and every caller calls it at least once.
struct OmniChunkResult {
    codes: Vec<Vec<u32>>,
    prompt_tokens: usize,
    forward_passes: usize,
}

fn omni_generate_chunk(
    lm: &OmniLm,
    codec: &OmniCodecDecoder,
    tok: &HfBpeTokenizer,
    request: &OmniRequest,
    fwd: &dyn OmniForward,
) -> Result<OmniChunkResult, String> {
    let cfg = &lm.config;
    let codebooks = cfg.num_audio_codebook;
    let vocab = cfg.audio_vocab_size;
    let mask = cfg.audio_mask_id as u32;

    let frames = if request.duration_seconds > 0.0 {
        (request.duration_seconds as f64 * codec.config.sample_rate as f64
            / codec.config.hop_length as f64)
            .ceil() as usize
    } else {
        omni_estimate_frames_from_reference(
            &omni_combine_text(&request.text, ""),
            &request.ref_text,
            request.ref_frames(),
            &codec.config,
        )
    };
    if frames == 0 {
        return Err("omnivoice: computed zero frames".into());
    }

    let cond = omni_build_conditional(tok, cfg, request, frames)?;
    let uncond_base = omni_build_unconditional(cfg, frames);
    let use_guidance = request.generation.guidance_scale != 0.0;

    // Where the target frames start in the conditional sequence.
    let cond_offset = cond.len() - frames;

    // The working state: every (codebook, frame) pair, initially masked.
    let mut tokens = vec![mask; codebooks * frames];
    let at = |c: usize, t: usize| c * frames + t;

    let schedule = omni_unmask_schedule(codebooks * frames, request.generation.num_step, request.generation.t_shift);

    let mut rng = InitRng::new(request.generation.seed);
    let mut cond_seq = cond.clone();
    let mut uncond_seq = uncond_base;

    let mut log_probs = vec![0.0f32; codebooks * frames * vocab];
    let mut scores = vec![0.0f32; codebooks * frames];
    let mut predicted = vec![0u32; codebooks * frames];
    let mut order = vec![0usize; codebooks * frames];

    let mut passes = 0usize;

    for (step, &k) in schedule.iter().enumerate() {
        if k == 0 {
            continue;
        }

        let c_logits = fwd.forward(&cond_seq)?;
        passes += 1;
        let u_logits = if use_guidance {
            let u = fwd.forward(&uncond_seq)?;
            passes += 1;
            Some(u)
        } else {
            None
        };

        // Combine, per (codebook, frame).
        let mut u_row = vec![0.0f32; vocab];
        for t in 0..frames {
            for c in 0..codebooks {
                let base = (c * frames + t) * vocab;
                let row = &mut log_probs[base..base + vocab];
                for (v, slot) in row.iter_mut().enumerate() {
                    *slot = c_logits.at(cond_offset + t, c * vocab + v);
                }
                log_softmax(row);

                if let Some(u_logits) = &u_logits {
                    for (v, slot) in u_row.iter_mut().enumerate() {
                        *slot = u_logits.at(t, c * vocab + v);
                    }
                    log_softmax(&mut u_row);
                    // c + s * (c - u), renormalized.
                    for v in 0..vocab {
                        row[v] += request.generation.guidance_scale * (row[v] - u_row[v]);
                    }
                    log_softmax(row);
                }

                // The mask id is never a legitimate prediction.
                row[cfg.audio_mask_id] = f32::NEG_INFINITY;

                let mut best = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (v, &value) in row.iter().enumerate() {
                    if value > best_v {
                        best_v = value;
                        best = v;
                    }
                }
                if request.generation.class_temperature > 0.0 {
                    // Gumbel over the token choice itself.
                    best_v = f32::NEG_INFINITY;
                    for v in 0..vocab {
                        if v == cfg.audio_mask_id {
                            continue;
                        }
                        let g = row[v] + request.generation.class_temperature * gumbel(&mut rng);
                        if g > best_v {
                            best_v = g;
                            best = v;
                        }
                    }
                }

                predicted[at(c, t)] = best as u32;

                // Confidence, less a penalty that grows with the codebook
                // index so the coarse ones are decided first.
                let mut s = row[best] - c as f32 * request.generation.layer_penalty_factor;
                if request.generation.position_temperature > 0.0 {
                    s += request.generation.position_temperature * gumbel(&mut rng);
                }
                // Anything already decided is out of the running.
                if tokens[at(c, t)] != mask {
                    s = f32::NEG_INFINITY;
                }
                scores[at(c, t)] = s;
            }
        }

        // Unmask the k best.
        for (i, slot) in order.iter_mut().enumerate() {
            *slot = i;
        }
        let take = k.min(order.len());
        libcxx_partial_sort(&mut order, take, |a, b| scores[a] > scores[b]);
        for &idx in &order[..take] {
            if scores[idx] == f32::NEG_INFINITY {
                break; // nothing left that is still masked
            }
            tokens[idx] = predicted[idx];
        }

        // Feed the decisions back into both sequences.
        for t in 0..frames {
            for c in 0..codebooks {
                cond_seq[cond_offset + t].audio[c] = tokens[at(c, t)];
                uncond_seq[t].audio[c] = tokens[at(c, t)];
            }
        }

        if request.debug {
            let remaining = tokens.iter().filter(|&&v| v == mask).count();
            eprintln!(
                "[ omnivoice ] step {:2}/{} unmasked {}, {} still masked",
                step + 1,
                schedule.len(),
                take,
                remaining
            );
        }
    }

    // Anything still masked would be an out-of-range code downstream; the
    // schedule guarantees this cannot happen, so treat it as a bug rather than
    // clamping quietly.
    if tokens.contains(&mask) {
        return Err("omnivoice: the unmasking schedule left positions undecided".into());
    }

    let mut codes = vec![vec![0u32; frames]; codebooks];
    for (c, stream) in codes.iter_mut().enumerate() {
        for (t, slot) in stream.iter_mut().enumerate() {
            *slot = tokens[at(c, t)];
        }
    }

    Ok(OmniChunkResult { codes, prompt_tokens: cond.len(), forward_passes: passes })
}

/// Run the unmasking loop and decode the result.
///
/// `accel`, when given, runs the forward passes instead of `lm` -- the model is
/// still needed for its config and its prompt layout. Passing a backend whose
/// weights came from a different model is the caller's problem.
pub fn omni_synthesize(
    lm: &OmniLm,
    codec: &OmniCodecDecoder,
    tok: &HfBpeTokenizer,
    request: &OmniRequest,
    accel: Option<&dyn OmniForward>,
) -> Result<OmniResult, String> {
    // The model still owns the config and the prompt layout; only the forward
    // pass moves.
    let fwd: &dyn OmniForward = match accel {
        Some(a) => a,
        None => lm,
    };
    let codec_cfg = &codec.config;
    let rate = codec_cfg.sample_rate as f64 / codec_cfg.hop_length as f64;

    let start = Instant::now();

    // How long is this, and is it too long to say in one breath? An explicit
    // --duration is an instruction about the whole output, so it turns
    // splitting off rather than being divided among the pieces.
    let estimate = omni_estimate_frames_from_reference(
        &omni_combine_text(&request.text, ""),
        &request.ref_text,
        request.ref_frames(),
        codec_cfg,
    );
    let threshold = (request.generation.chunk_threshold_seconds as f64 * rate) as usize;
    let split = request.duration_seconds <= 0.0
        && threshold > 0
        && request.generation.chunk_seconds > 0.0
        && estimate > threshold;

    let mut pieces: Vec<String> = Vec::new();
    if split {
        // Characters per chunk, from this text's own measured density rather
        // than an average one: a line of digits is worth far more audio per
        // character than a line of Latin letters.
        let chars = request.text.chars().count();
        let per_chunk =
            request.generation.chunk_seconds as f64 * rate * chars as f64 / estimate as f64;
        pieces = omni_chunk_text(&request.text, (per_chunk as usize).max(1), DEFAULT_MIN_CHUNK_CHARS);
    }
    if pieces.len() < 2 {
        pieces = vec![request.text.clone()];
    }

    let mut waves: Vec<Vec<f32>> = Vec::with_capacity(pieces.len());
    let mut result = OmniResult {
        sample_rate: codec_cfg.sample_rate,
        chunks: pieces.len(),
        ..Default::default()
    };

    // What holds the voice across a split. With a reference clip every piece
    // uses it. Without one, the first piece invents a speaker and every piece
    // after it takes that piece as its reference -- so the seam is a breath
    // rather than a new person.
    let mut anchor_codes: Vec<Vec<u32>> = Vec::new();
    let mut anchor_text = String::new();

    let mut decode_seconds = 0.0f64;
    for (i, text) in pieces.iter().enumerate() {
        let mut piece = request.clone();
        piece.text = text.clone();
        if pieces.len() > 1 {
            // Each piece is measured on its own; the whole-text estimate was
            // only ever used to decide how to cut it.
            piece.duration_seconds = 0.0;
            if request.ref_codes.is_empty() && i > 0 {
                piece.ref_codes = anchor_codes.clone();
                piece.ref_text = anchor_text.clone();
            }
            if request.debug {
                eprintln!("[ omnivoice ] chunk {}/{}: \"{}\"", i + 1, pieces.len(), text);
            }
        }

        let chunk = omni_generate_chunk(lm, codec, tok, &piece, fwd)?;
        if i == 0 && request.ref_codes.is_empty() {
            anchor_codes = chunk.codes.clone();
            anchor_text = pieces[0].clone();
        }

        let decode_start = Instant::now();
        let mut samples = codec.decode(&chunk.codes)?;
        decode_seconds += decode_start.elapsed().as_secs_f64();

        // Put the reference's own loudness back. Without this a quiet recording
        // clones into a voice that is the right voice at the wrong level,
        // because the clip was brought up to 0.1 RMS before it was encoded.
        const TARGET_RMS: f32 = 0.1;
        if request.ref_rms > 0.0 && request.ref_rms < TARGET_RMS {
            let gain = request.ref_rms / TARGET_RMS;
            for v in samples.iter_mut() {
                *v *= gain;
            }
        }

        result.frames += chunk.codes.first().map_or(0, |s| s.len());
        result.prompt_tokens += chunk.prompt_tokens;
        result.forward_passes += chunk.forward_passes;
        waves.push(samples);
    }

    result.samples = omni_cross_fade(&waves, codec_cfg.sample_rate, request.generation.chunk_gap_seconds);
    result.decode_seconds = decode_seconds;
    result.generate_seconds = start.elapsed().as_secs_f64() - decode_seconds;
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_loader::GgufFile;
    use crate::omnivoice_codec::OmniCodecEncoder;
    use crate::transformer5::load_gguf_tokenizer;
    use crate::wav::wave_stats;

    const LM_PATH: &str = "models/omnivoice-base-Q8_0.gguf";
    const CODEC_PATH: &str = "models/omnivoice-tokenizer-Q8_0.gguf";

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-3
    }

    fn lm_present() -> bool {
        if std::path::Path::new(LM_PATH).exists() {
            return true;
        }
        eprintln!("skip: models/omnivoice-base-Q8_0.gguf not present");
        false
    }

    fn gguf_tokenizer() -> HfBpeTokenizer {
        let gguf = GgufFile::open(LM_PATH).expect("open LM GGUF");
        load_gguf_tokenizer(&gguf).unwrap_or_else(|e| panic!("{}", e))
    }

    /// A tokenizer with just enough vocabulary to encode the test prompts. The
    /// prompt layout is what is under test, not the merges.
    fn toy_tokenizer() -> HfBpeTokenizer {
        let json = concat!(
            r#"{"model":{"type":"BPE","vocab":{"a":0,"b":1,"c":2,"Ġ":3,"o":4,"e":5,"#,
            r#""i":6,"n":7,"N":8,"t":9,".":10,"s":11,"h":12,"l":13,"d":14},"merges":[]}}"#
        );
        HfBpeTokenizer::from_json_bytes(json.as_bytes()).expect("toy tokenizer")
    }

    /// `codebooks` streams of `frames` distinct codes.
    fn toy_codes(codebooks: usize, frames: usize) -> Vec<Vec<u32>> {
        let mut out = vec![vec![0u32; frames]; codebooks];
        for (c, stream) in out.iter_mut().enumerate() {
            for (t, slot) in stream.iter_mut().enumerate() {
                *slot = (c * 100 + t) as u32;
            }
        }
        out
    }

    // =========================================================================
    // Schedules
    // =========================================================================

    #[test]
    fn omni_time_steps_spans_0_to_1() {
        for shift in [0.1f32, 1.0, 3.0] {
            let ts = omni_time_steps(32, shift);
            assert_eq!(ts.len(), 33);
            assert!(approx(ts[0], 0.0));
            assert!(approx(*ts.last().unwrap(), 1.0));
            // Monotone, or the per-step share would go negative.
            for i in 1..ts.len() {
                assert!(ts[i] >= ts[i - 1]);
            }
        }
    }

    #[test]
    fn t_shift_below_1_front_loads_the_schedule() {
        // The warp is what decides how much is unmasked early. At shift 1 the grid
        // is linear; below 1 the early steps cover less of the interval, so fewer
        // tokens are committed before the model has context.
        let linear = omni_time_steps(10, 1.0);
        let shifted = omni_time_steps(10, 0.1);
        assert!(approx(linear[5], 0.5));
        assert!(shifted[5] < linear[5]);
    }

    #[test]
    fn omni_unmask_schedule_accounts_for_every_position() {
        // Nothing may be left masked by rounding: the codec would reject a mask id
        // as an out-of-range code.
        for total in [8usize, 800, 4001] {
            for steps in [1usize, 8, 32] {
                let s = omni_unmask_schedule(total, steps, 0.1);
                assert_eq!(s.len(), steps);
                assert_eq!(s.iter().sum::<usize>(), total);
            }
        }
    }

    #[test]
    fn omni_unmask_schedule_never_exceeds_what_is_left() {
        let s = omni_unmask_schedule(100, 32, 0.1);
        let mut seen = 0usize;
        for &n in &s {
            seen += n;
            assert!(seen <= 100);
        }
        assert_eq!(seen, 100);
    }

    #[test]
    fn a_single_step_unmasks_everything_at_once() {
        let s = omni_unmask_schedule(64, 1, 0.1);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], 64);
    }

    // =========================================================================
    // Prompt construction
    // =========================================================================

    #[test]
    fn omni_build_unconditional_is_target_frames_only() {
        // Classifier-free guidance conditions on nothing: no style block, no text.
        let cfg = Config6::omnivoice();
        let seq = omni_build_unconditional(&cfg, 10);
        assert_eq!(seq.len(), 10);
        for t in &seq {
            assert!(t.is_audio());
            assert_eq!(t.audio.len(), cfg.num_audio_codebook);
            assert_eq!(t.audio[0] as usize, cfg.audio_mask_id);
        }
    }

    #[test]
    fn omni_build_conditional_lays_out_style_text_then_frames() {
        if !lm_present() {
            return;
        }
        let tok = gguf_tokenizer();

        let cfg = Config6::omnivoice();
        let r = OmniRequest { text: "Ciao.".into(), language: "Italian".into(), ..Default::default() };

        let seq = omni_build_conditional(&tok, &cfg, &r, 10).expect("conditional");

        // The markers are injected as raw ids, never encoded as literal text.
        assert_eq!(seq[0].text_id, cfg.lang_start);
        assert!(!seq[0].is_audio());

        // The trailing block is the masked frames.
        assert!(seq.len() > 10);
        for t in &seq[seq.len() - 10..] {
            assert!(t.is_audio());
            assert_eq!(t.audio[0] as usize, cfg.audio_mask_id);
        }
        // Everything before them is text.
        for t in &seq[..seq.len() - 10] {
            assert!(!t.is_audio());
        }

        // The markers all appear, in order.
        let ids: Vec<usize> = seq.iter().filter(|t| !t.is_audio()).map(|t| t.text_id).collect();
        let pos = |id: usize| ids.iter().position(|&x| x == id).unwrap_or(ids.len());
        assert!(pos(cfg.lang_start) < pos(cfg.lang_end));
        assert!(pos(cfg.lang_end) < pos(cfg.instruct_start));
        assert!(pos(cfg.instruct_start) < pos(cfg.instruct_end));
        assert!(pos(cfg.instruct_end) < pos(cfg.text_start));
        assert!(pos(cfg.text_start) < pos(cfg.text_end));
    }

    #[test]
    fn omni_build_conditional_rejects_empty_input() {
        if !lm_present() {
            return;
        }
        let tok = gguf_tokenizer();

        let cfg = Config6::omnivoice();
        let mut r = OmniRequest::default();
        assert!(omni_build_conditional(&tok, &cfg, &r, 10).is_err());
        r.text = "Ciao.".into();
        assert!(omni_build_conditional(&tok, &cfg, &r, 0).is_err());
    }

    // =========================================================================
    // Text normalisation
    // =========================================================================

    #[test]
    fn omni_combine_text_trims_and_joins() {
        assert_eq!(omni_combine_text("  hello  ", ""), "hello");
        assert_eq!(omni_combine_text(" world ", "  hello "), "hello world");
        // An empty reference is not a leading space.
        assert_eq!(omni_combine_text("hello", "   "), "hello");
    }

    #[test]
    fn omni_combine_text_folds_whitespace() {
        // A line break inside a prompt would be spoken as nothing useful, and runs
        // of spaces as a pause that is not in the text.
        assert_eq!(omni_combine_text("a\nb", ""), "ab");
        assert_eq!(omni_combine_text("a\r\n\r\nb", ""), "ab");
        assert_eq!(omni_combine_text("a  \t  b", ""), "a b");
        assert_eq!(omni_combine_text("one   two    three", ""), "one two three");
    }

    #[test]
    fn omni_combine_text_normalises_fullwidth_parentheses() {
        assert_eq!(omni_combine_text("（x）", ""), "(x)");
    }

    #[test]
    fn omni_combine_text_removes_spaces_next_to_cjk() {
        // Between ideographs a space is typesetting, not a word boundary, and
        // reading it as one puts a pause where none belongs.
        assert_eq!(omni_combine_text("你好 世界", ""), "你好世界");
        assert_eq!(omni_combine_text("hello 你好", ""), "hello你好");
        assert_eq!(omni_combine_text("你好 world", ""), "你好world");
        // Latin text keeps its spaces.
        assert_eq!(omni_combine_text("hello world", ""), "hello world");
    }

    // =========================================================================
    // Reference clips
    // =========================================================================

    #[test]
    fn omni_prepare_reference_trims_to_whole_frames() {
        let c = OmniCodecConfig::defaults();
        // Three frames and a bit.
        let wav = vec![0.2f32; 3 * 960 + 137];
        let r = omni_prepare_reference(&wav, 24000, &c).expect("prepare");
        assert_eq!(r.samples.len(), 3 * 960);
        assert!(approx(r.seconds(&c) as f32, 0.12));
    }

    #[test]
    fn omni_prepare_reference_resamples_to_the_codec_rate() {
        let c = OmniCodecConfig::defaults();
        // One second at 48 kHz becomes one second at 24 kHz, which is 25 frames.
        let wav = vec![0.3f32; 48000];
        let r = omni_prepare_reference(&wav, 48000, &c).expect("prepare");
        assert_eq!(r.samples.len(), 25 * 960);
    }

    #[test]
    fn omni_prepare_reference_lifts_a_quiet_clip_to_the_codec_level() {
        let c = OmniCodecConfig::defaults();
        // A constant 0.02 has an RMS of 0.02, so it is scaled by five.
        let wav = vec![0.02f32; 5 * 960];
        let r = omni_prepare_reference(&wav, 24000, &c).expect("prepare");
        // The reported RMS is the original one, which is what the output is scaled
        // back to.
        assert!(approx(r.rms, 0.02));
        for &v in &r.samples {
            assert!(approx(v, 0.1));
        }
    }

    #[test]
    fn omni_prepare_reference_leaves_a_loud_clip_alone() {
        let c = OmniCodecConfig::defaults();
        let wav = vec![0.4f32; 5 * 960];
        let r = omni_prepare_reference(&wav, 24000, &c).expect("prepare");
        assert!(approx(r.rms, 0.4));
        for &v in &r.samples {
            assert!(approx(v, 0.4));
        }
    }

    #[test]
    fn omni_prepare_reference_rejects_what_it_cannot_use() {
        let c = OmniCodecConfig::defaults();
        assert!(omni_prepare_reference(&[], 24000, &c).is_err());
        assert!(omni_prepare_reference(&vec![0.1f32; 100], 24000, &c).is_err());
        assert!(omni_prepare_reference(&vec![0.1f32; 4800], 0, &c).is_err());
    }

    // =========================================================================
    // Length with a reference
    // =========================================================================

    #[test]
    fn a_reference_clip_calibrates_the_length_estimate() {
        let c = OmniCodecConfig::defaults();
        let text = "Domani andro al mercato con mia sorella.";
        let r = "Ciao, mi chiamo Giulia.";

        // A speaker who took 80 frames to say a 44-frame phrase is slow, and the
        // estimate has to follow them rather than the built-in average.
        let slow = omni_estimate_frames_from_reference(text, r, 80, &c);
        let fast = omni_estimate_frames_from_reference(text, r, 30, &c);
        assert!(slow > fast);

        // No usable reference falls back to the built-in phrase.
        assert_eq!(
            omni_estimate_frames_from_reference(text, "", 80, &c),
            omni_estimate_frames(text, &c)
        );
        assert_eq!(
            omni_estimate_frames_from_reference(text, r, 0, &c),
            omni_estimate_frames(text, &c)
        );
    }

    // =========================================================================
    // Cloning prompts
    // =========================================================================

    #[test]
    fn a_reference_clip_becomes_decided_frames_before_the_masked_ones() {
        let cfg = Config6::omnivoice();
        let tok = toy_tokenizer();

        let r = OmniRequest {
            text: "note".into(),
            ref_text: "abc".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook, 5),
            ..Default::default()
        };

        let seq = omni_build_conditional(&tok, &cfg, &r, 4).expect("conditional");

        // The denoise marker leads, and only when there is a recording to clean up.
        assert_eq!(seq[0].text_id, cfg.denoise);
        assert_eq!(seq[1].text_id, cfg.lang_start);

        // Nine audio positions: five carrying the reference, four masked.
        assert!(seq.len() > 9);
        let audio_start = seq.len() - 9;
        for t in &seq[..audio_start] {
            assert!(!t.is_audio());
        }
        for t in 0..5 {
            let token = &seq[audio_start + t];
            assert!(token.is_audio());
            for c in 0..cfg.num_audio_codebook {
                assert_eq!(token.audio[c] as usize, c * 100 + t);
                assert_ne!(token.audio[c] as usize, cfg.audio_mask_id);
            }
        }
        for t in 5..9 {
            let token = &seq[audio_start + t];
            assert!(token.is_audio());
            for &v in &token.audio {
                assert_eq!(v as usize, cfg.audio_mask_id);
            }
        }
    }

    #[test]
    fn without_a_reference_there_is_no_denoise_marker() {
        let cfg = Config6::omnivoice();
        let tok = toy_tokenizer();
        let r = OmniRequest { text: "note".into(), ..Default::default() };

        let seq = omni_build_conditional(&tok, &cfg, &r, 3).expect("conditional");
        assert_eq!(seq[0].text_id, cfg.lang_start);
        // Exactly the target frames, and every one of them masked.
        assert!(!seq[seq.len() - 4].is_audio());
    }

    #[test]
    fn the_target_frames_stay_at_the_end_whatever_precedes_them() {
        // `omni_synthesize` finds them by counting back from the end, so this is
        // the property that makes cloning need no change to the unmasking loop.
        let cfg = Config6::omnivoice();
        let tok = toy_tokenizer();

        let plain = OmniRequest { text: "note".into(), ..Default::default() };
        let cloned = OmniRequest {
            text: "note".into(),
            ref_text: "abc".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook, 6),
            ..Default::default()
        };

        let a = omni_build_conditional(&tok, &cfg, &plain, 4).expect("plain");
        let b = omni_build_conditional(&tok, &cfg, &cloned, 4).expect("cloned");
        for i in 1..=4 {
            assert!(a[a.len() - i].is_audio());
            assert!(b[b.len() - i].is_audio());
            assert_eq!(a[a.len() - i].audio[0] as usize, cfg.audio_mask_id);
            assert_eq!(b[b.len() - i].audio[0] as usize, cfg.audio_mask_id);
        }
    }

    #[test]
    fn a_reference_clip_is_rejected_without_its_transcript() {
        let cfg = Config6::omnivoice();
        let tok = toy_tokenizer();
        let r = OmniRequest {
            text: "note".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook, 3),
            ..Default::default()
        };
        assert!(omni_build_conditional(&tok, &cfg, &r, 4).is_err());
    }

    #[test]
    fn malformed_reference_codes_are_rejected() {
        let cfg = Config6::omnivoice();
        let tok = toy_tokenizer();

        let mut ragged = OmniRequest {
            text: "note".into(),
            ref_text: "abc".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook, 3),
            ..Default::default()
        };
        ragged.ref_codes[2].pop();
        assert!(omni_build_conditional(&tok, &cfg, &ragged, 4).is_err());

        let few = OmniRequest {
            text: "note".into(),
            ref_text: "abc".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook - 1, 3),
            ..Default::default()
        };
        assert!(omni_build_conditional(&tok, &cfg, &few, 4).is_err());

        let mut wild = OmniRequest {
            text: "note".into(),
            ref_text: "abc".into(),
            ref_codes: toy_codes(cfg.num_audio_codebook, 3),
            ..Default::default()
        };
        // The mask id is not a code, so it cannot appear in a reference.
        wild.ref_codes[0][1] = cfg.audio_mask_id as u32;
        assert!(omni_build_conditional(&tok, &cfg, &wild, 4).is_err());
    }

    // =========================================================================
    // End to end
    // =========================================================================

    #[test]
    fn omnivoice_clones_a_voice_end_to_end() {
        // Loads both halves of the model plus the codec's analysis path, so this
        // is tens of seconds rather than one.
        if !std::path::Path::new(LM_PATH).exists() || !std::path::Path::new(CODEC_PATH).exists() {
            eprintln!("skip: OmniVoice weights not present");
            return;
        }

        let codec_cfg = OmniCodecConfig::defaults();
        let tok = gguf_tokenizer();
        let lm = OmniLm::load(LM_PATH, Config6::omnivoice()).expect("load LM");
        let decoder = OmniCodecDecoder::load(CODEC_PATH, codec_cfg.clone()).expect("load decoder");
        let encoder = OmniCodecEncoder::load(CODEC_PATH, codec_cfg.clone()).expect("load encoder");

        // Build a reference clip the way a user would: real audio in, codes out.
        // Synthesising one first would double the runtime, so this decodes a fixed
        // set of codes instead -- the point is the plumbing, not the voice.
        const REF_FRAMES: usize = 25;
        let mut seed_codes = vec![vec![0u32; REF_FRAMES]; codec_cfg.n_codebooks];
        let mut state: u32 = 987654321;
        for stream in seed_codes.iter_mut() {
            for slot in stream.iter_mut() {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *slot = (state >> 16) % codec_cfg.codebook_size as u32;
            }
        }
        let ref_wav = decoder.decode(&seed_codes).expect("decode reference");

        let reference =
            omni_prepare_reference(&ref_wav, codec_cfg.sample_rate, &codec_cfg).expect("prepare");
        let ref_codes = encoder.encode(&reference.samples).expect("encode reference");
        assert_eq!(ref_codes[0].len(), REF_FRAMES);

        let mut request = OmniRequest {
            text: "Domani andro al mercato.".into(),
            language: "Italian".into(),
            ref_text: "Ciao, mi chiamo Giulia.".into(),
            ref_codes,
            ref_rms: reference.rms,
            duration_seconds: 1.0,
            ..Default::default()
        };
        request.generation.num_step = 4; // enough to exercise the loop, not to sound good
        request.generation.seed = 99;

        let out = omni_synthesize(&lm, &decoder, &tok, &request, None).expect("synthesize");
        assert_eq!(out.frames, 25);
        assert_eq!(out.samples.len(), 25 * codec_cfg.hop_length);
        assert_eq!(out.forward_passes, 8);

        // The reference's frames ride along in the conditional prompt, so it is
        // longer than the plain one by exactly their count.
        let mut plain = request.clone();
        plain.ref_codes.clear();
        plain.ref_text.clear();
        plain.ref_rms = 0.0;
        let bare = omni_synthesize(&lm, &decoder, &tok, &plain, None).expect("synthesize plain");
        assert!(out.prompt_tokens > bare.prompt_tokens);

        let stats = wave_stats(&out.samples);
        assert!(stats.in_range());
        for &v in &out.samples {
            assert!(v.is_finite());
        }

        // The reference changes what comes out. It cannot be checked that it
        // changes it in the *right* direction without listening, but a run that
        // ignored the codes entirely would land on the unconditioned waveform.
        assert_eq!(out.samples.len(), bare.samples.len());
        let mut num = 0.0f64;
        let mut da = 0.0f64;
        let mut db = 0.0f64;
        for i in 0..out.samples.len() {
            num += out.samples[i] as f64 * bare.samples[i] as f64;
            da += out.samples[i] as f64 * out.samples[i] as f64;
            db += bare.samples[i] as f64 * bare.samples[i] as f64;
        }
        assert!((num / (da * db).sqrt()).abs() < 0.5);
    }

    // =========================================================================
    // Chunking long text
    // =========================================================================

    #[test]
    fn omni_chunk_text_breaks_at_punctuation() {
        let got = omni_chunk_text("Uno due tre. Quattro cinque sei. Sette otto nove.", 20, 3);
        assert_eq!(got, vec!["Uno due tre.", "Quattro cinque sei.", "Sette otto nove."]);
    }

    #[test]
    fn omni_chunk_text_merges_sentences_up_to_the_target() {
        // Greedy: a chunk takes whole sentences until the next would overrun.
        let got = omni_chunk_text("Uno due tre. Quattro cinque sei. Sette otto nove.", 40, 3);
        assert_eq!(got, vec!["Uno due tre. Quattro cinque sei.", "Sette otto nove."]);
    }

    #[test]
    fn omni_chunk_text_keeps_a_whole_sentence_that_overruns() {
        // Breaking mid-sentence would put a seam where the prosody is still
        // rising, so an over-long sentence is left alone.
        let got = omni_chunk_text("Questa e una frase molto lunga che non finisce mai.", 10, 3);
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn omni_chunk_text_does_not_split_abbreviations() {
        // "Dr." is not the end of a sentence, and splitting there would strand a
        // title from its name.
        let got = omni_chunk_text("Dr. Rossi arrived. He waited.", 18, 3);
        assert_eq!(got, vec!["Dr. Rossi arrived.", "He waited."]);

        let eg = omni_chunk_text("Fruit, e.g. apples, is good.", 100, 3);
        assert_eq!(eg.len(), 1);
    }

    #[test]
    fn omni_chunk_text_keeps_closing_marks_with_what_they_close() {
        // The quote belongs to the sentence that just ended, not the next one.
        let got = omni_chunk_text("\"Uno due.\" Tre quattro.", 12, 3);
        assert_eq!(got, vec!["\"Uno due.\"", "Tre quattro."]);
    }

    #[test]
    fn omni_chunk_text_splits_on_fullwidth_punctuation() {
        // A CJK clause ends on the wide comma, not the ASCII one.
        let got = omni_chunk_text("你好世界。今天天气很好。", 6, 3);
        assert_eq!(got, vec!["你好世界。", "今天天气很好。"]);
    }

    #[test]
    fn omni_chunk_text_folds_away_pieces_too_short_to_stand_alone() {
        // A two-character chunk would be given its own speaker and sound like one.
        let got = omni_chunk_text("Uno due tre quattro. Si.", 20, 5);
        assert_eq!(got, vec!["Uno due tre quattro. Si."]);
    }

    #[test]
    fn omni_chunk_text_handles_the_degenerate_inputs() {
        assert!(omni_chunk_text("", 20, 3).is_empty());
        assert!(omni_chunk_text("   ", 20, 3).is_empty());
        // A zero target means "do not split".
        let whole = omni_chunk_text("Uno. Due. Tre.", 0, 3);
        assert_eq!(whole, vec!["Uno. Due. Tre."]);
    }

    #[test]
    fn omni_chunk_text_loses_no_text() {
        // Whatever the split, every character has to come out the other side.
        let text = "Oggi e una bella giornata. Domani andro al mercato, poi a Roma! Va bene?";
        for target in [5usize, 12, 30, 200] {
            let got = omni_chunk_text(text, target, 3);
            assert_eq!(got.join(" "), text);
        }
    }

    // =========================================================================
    // Joining the pieces
    // =========================================================================

    #[test]
    fn omni_voiced_span_finds_where_a_signal_actually_starts() {
        // 0.2 s of silence, 0.3 s of tone, 0.2 s of silence, at 3 kHz.
        const RATE: usize = 3000;
        let mut x = vec![0.0f32; RATE * 7 / 10];
        for i in 0..RATE * 3 / 10 {
            x[RATE * 2 / 10 + i] = 0.4;
        }
        let (begin, end) = omni_voiced_span(&x, RATE, -50.0, 0);
        // Resolved to the 10 ms window, so within one window of the true edges.
        assert!(begin <= RATE * 2 / 10);
        assert!(begin >= RATE * 2 / 10 - RATE / 100);
        assert!(end >= RATE * 5 / 10 - RATE / 100);
        assert!(end <= RATE * 5 / 10 + RATE / 100);
    }

    #[test]
    fn omni_voiced_span_ignores_a_noise_floor_that_grazes_the_threshold() {
        // The case that made this a window rather than a peak test. A chunk's
        // quiet head crosses -50 dBFS on the odd sample while averaging well below
        // it; a peak test keeps everything from the first crossing, and half a
        // second of near-silence lands in the middle of a join.
        const RATE: usize = 3000;
        let threshold = 10.0f32.powf(-50.0f32 / 20.0f32);
        let mut x = vec![0.0f32; RATE];
        for i in (0..RATE / 2).step_by(17) {
            x[i] = threshold * 3.0; // above the line on its own, not on average
        }
        for v in &mut x[RATE / 2..RATE * 3 / 4] {
            *v = 0.4;
        }
        let (begin, end) = omni_voiced_span(&x, RATE, -50.0, 0);
        assert!(begin >= RATE / 2 - RATE / 100);
        assert!(end <= RATE * 3 / 4 + RATE / 100);
    }

    #[test]
    fn omni_voiced_span_reports_nothing_for_silence() {
        let quiet = vec![0.0f32; 3000];
        let (begin, end) = omni_voiced_span(&quiet, 3000, -50.0, 0);
        assert_eq!(begin, end);
    }

    #[test]
    fn omni_voiced_span_widens_by_the_margin() {
        const RATE: usize = 3000;
        let mut x = vec![0.0f32; RATE];
        for v in &mut x[RATE / 2..RATE * 3 / 4] {
            *v = 0.4;
        }
        let (tight_begin, tight_end) = omni_voiced_span(&x, RATE, -50.0, 0);
        let (wide_begin, wide_end) = omni_voiced_span(&x, RATE, -50.0, 50);
        assert_eq!(wide_begin, tight_begin - 50);
        assert_eq!(wide_end, tight_end + 50);
        // And it cannot run off either end.
        let (all_begin, all_end) = omni_voiced_span(&x, RATE, -50.0, 100000);
        assert_eq!(all_begin, 0);
        assert_eq!(all_end, x.len());
    }

    #[test]
    fn omni_cross_fade_passes_a_single_chunk_through_untouched() {
        // One piece is the whole utterance: no join, and nothing trimmed. Short
        // text has to come out exactly as the codec produced it.
        let one = vec![vec![0.1f32, 0.2, 0.0, 0.0]];
        assert_eq!(omni_cross_fade(&one, 24000, 0.3), one[0]);
        assert!(omni_cross_fade(&[], 24000, 0.3).is_empty());
    }

    #[test]
    fn omni_cross_fade_puts_the_asked_for_pause_between_pieces() {
        // Both pieces are loud throughout, so nothing is trimmed and the join is
        // exactly the gap.
        const RATE: usize = 3000;
        let chunks = vec![vec![0.5f32; 1000], vec![0.5f32; 1000]];
        let got = omni_cross_fade(&chunks, RATE, 0.3);
        let gap = (0.3f32 * RATE as f32) as usize;
        assert_eq!(got.len(), 1000 + gap + 1000);
        for &v in &got[1000..1000 + gap] {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn omni_cross_fade_trims_each_piece_to_what_it_says() {
        // The point of the trim: a chunk's length comes from a duration estimate,
        // so it ends with however much silence the estimate overshot by. Without
        // trimming, the pause is the gap plus two accidents.
        const RATE: usize = 3000;
        let mut a = vec![0.0f32; 1500];
        let mut b = vec![0.0f32; 1500];
        for i in 0..500 {
            a[i] = 0.5; // speech, then 1000 samples of overshoot
            b[1000 + i] = 0.5; // 1000 samples of silence, then speech
        }
        let got = omni_cross_fade(&[a, b], RATE, 0.3);
        let gap = (0.3f32 * RATE as f32) as usize;
        let fade = (0.008f32 * RATE as f32) as usize;
        // Each piece keeps its 500 samples plus the fade's margin at the inner
        // edge; the dead 1000 samples on each side are gone.
        assert!(got.len() < 500 + gap + 500 + 4 * fade);
        assert!(got.len() > 500 + gap + 500);
    }

    #[test]
    fn omni_cross_fade_does_not_fade_away_a_whole_syllable() {
        // The reference ramps a tenth of a second to nothing at every edge, which
        // is long enough to swallow the last sound of a chunk when the estimate
        // was tight -- and it usually is. The fade here is a few milliseconds.
        const RATE: usize = 24000;
        let chunks = vec![vec![0.5f32; RATE], vec![0.5f32; RATE]];
        let got = omni_cross_fade(&chunks, RATE, 0.3);
        // 50 ms before the join the signal is still at full level.
        let join = RATE;
        assert!(approx(got[join - RATE / 20], 0.5));
        // And it does reach zero, so there is no click.
        assert!(got[join - 1] < 0.05);
    }

    #[test]
    fn omni_cross_fade_drops_a_piece_with_nothing_in_it() {
        const RATE: usize = 3000;
        let chunks = vec![vec![0.5f32; 500], vec![0.0f32; 500], vec![0.5f32; 500]];
        let got = omni_cross_fade(&chunks, RATE, 0.3);
        let gap = (0.3f32 * RATE as f32) as usize;
        // Two pieces and one gap, not three and two.
        assert_eq!(got.len(), 500 + gap + 500);
    }

    #[test]
    fn omni_cross_fade_handles_pieces_shorter_than_the_fade() {
        let chunks = vec![vec![0.5f32; 4], vec![0.5f32; 4]];
        let got = omni_cross_fade(&chunks, 3000, 0.3);
        for &v in &got {
            assert!(v.is_finite());
            assert!(v.abs() <= 1.0);
        }
    }

    #[test]
    fn omni_cross_fade_joins_three_pieces_with_two_gaps() {
        const RATE: usize = 3000;
        let chunks = vec![vec![0.5f32; 500], vec![0.5f32; 600], vec![0.5f32; 700]];
        let got = omni_cross_fade(&chunks, RATE, 0.3);
        let gap = (0.3f32 * RATE as f32) as usize;
        assert_eq!(got.len(), 500 + 600 + 700 + 2 * gap);
    }

    // =========================================================================
    // Frames (the omni_estimate_frames cases of tests/test_duration.cpp)
    // =========================================================================

    /// The phrase the reference calibrates against, at 25 frames -- one second.
    const DURATION_REF: &str = "Nice to meet you.";

    #[test]
    fn omni_estimate_frames_matches_the_reference_estimator() {
        let c = OmniCodecConfig::defaults();
        // 25 Hz, so these are also the reference implementation's token counts.
        assert_eq!(c.sample_rate / c.hop_length, 25);

        assert_eq!(omni_estimate_frames("Ciao, mi chiamo Giulia.", &c), 44);
        assert_eq!(
            omni_estimate_frames("Ciao, mi chiamo Giulia. Oggi e una bella giornata a Roma.", &c),
            84
        );
        assert_eq!(omni_estimate_frames("Hello world", &c), 35);
        assert_eq!(omni_estimate_frames("你好，世界！", &c), 38);
        assert_eq!(omni_estimate_frames(DURATION_REF, &c), 39);
    }

    #[test]
    fn omni_estimate_frames_never_asks_for_zero_frames() {
        let c = OmniCodecConfig::defaults();
        // Empty text weighs nothing, and zero frames would be a decode of nothing.
        assert_eq!(omni_estimate_frames("", &c), 1);
        assert_eq!(omni_estimate_frames("\u{0301}", &c), 1);
    }

    #[test]
    fn omni_estimate_frames_grows_with_the_text() {
        let c = OmniCodecConfig::defaults();
        let shorter = omni_estimate_frames("Ciao.", &c);
        let longer =
            omni_estimate_frames("Ciao, mi chiamo Giulia e oggi e una bella giornata a Roma.", &c);
        assert!(shorter > 0);
        assert!(longer > shorter);
        // Digits are spoken as words, so a year is worth more than its characters.
        assert!(omni_estimate_frames("2024", &c) > omni_estimate_frames("abcd", &c));
    }
}
