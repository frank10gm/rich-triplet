// =============================================================================
// Orpheus TTS -- text to speech over a Llama backbone and a SNAC codec
// =============================================================================
//
// Ties `transformer5.rs` (the Llama 3.2 backbone) to `snac.rs` (the codec
// decoder). The backbone generates audio tokens; this file works out which of
// its 156 940 vocabulary entries are audio codes, which SNAC codebook each one
// belongs to, and where a frame begins.
//
// ## The token protocol
//
// Orpheus extends Llama 3.2's 128 256-entry vocabulary with markers and audio
// codes. `<custom_token_0>` is id 128256, so `<custom_token_N>` is
// `128256 + N`, and the reference decoder computes a code as
// `N - 10 - (slot * 4096)`. Substituting gives:
//
//     code = id - 128266 - (slot * 4096)
//
// Audio ids therefore run from 128266 (`<custom_token_10>`) to 156937, seven
// codebook slots of 4096 each.
//
// A prompt is framed as:
//
//     [128000, 128259, 128000] + encode("{voice}: {text}")
//                              + [128009, 128260, 128261, 128257]
//
// and generation stops at 128258 (`<custom_token_2>`).
//
// Both 128000 (`<|begin_of_text|>`) tokens are there for a reason worth
// spelling out, because neither appears in the reference's source.
//
// The reference builds `[128259] + tokenizer(text) + [128009, ...]`, and its
// HuggingFace Llama tokenizer has `add_bos_token` set -- so `tokenizer(text)`
// already begins with 128000. That accounts for the inner one. It then
// *decodes the whole id sequence back to a string* and hands the string to
// vLLM, which tokenizes it again with the same setting and prepends a second
// 128000 at the very front. So the sequence the model is actually served, and
// therefore the one it behaves best on, carries both.
//
// Note also that 128009 (`<|eot_id|>`) appears *inside* the prompt as a
// separator. It is not a stop token here: treating it as one ends generation
// as soon as the model echoes the separator, which it readily does.
//
// ## Slot tracking and resynchronisation
//
// The seven codes of a frame are emitted in order, and slot `i` is offset by
// `i * 4096`. A token is only a valid code for the slot the stream currently
// expects if `code` lands inside `[0, 4096)`:
//
//   * a token really belonging to a later slot computes `code >= 4096`
//   * one belonging to an earlier slot computes `code < 0`
//
// So the range check *is* the slot check, and rejecting a token without
// advancing the slot counter lets the stream resynchronise after the model
// emits something unexpected.
//
// This differs from the reference in one place, deliberately. The reference
// accepts a code only when it is strictly positive, which throws away every
// legitimate code 0 -- and because a rejected token does not advance the slot,
// discarding a valid code 0 shifts every subsequent code by one slot and
// corrupts the rest of the utterance. Checking `0 <= code < 4096` keeps the
// resynchronisation behaviour and drops the off-by-one.
//
// ## Frame layout
//
// Seven codes cover four frames at the finest codebook rate, distributed
// 1:2:4 across the three time scales:
//
//     level 0 (stride 4):  t0
//     level 1 (stride 2):  t1, t4
//     level 2 (stride 1):  t2, t3, t5, t6
//
// which is 2048 samples, or 85.33 ms at 24 kHz. Realtime therefore needs about
// 82 tokens a second.

#![allow(dead_code)]

use std::time::Instant;

use crate::snac::{SnacDecoder, SnacNoise};
use crate::tokenizer::{HfBpeTokenizer, Tokenizer};
use crate::transformer3::SamplingParams;
use crate::transformer5::LlamaModel;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug)]
pub struct OrpheusConfig {
    /// Id of the code-0 token for slot 0, i.e. `<custom_token_10>`.
    pub audio_token_base: usize,
    /// Codes per frame group.
    pub codes_per_frame: usize,
    /// Entries in each SNAC codebook.
    pub codebook_size: usize,

    /// Prepended to the encoded prompt: `<custom_token_3>`.
    pub prompt_start: usize,
    /// Llama's `<|begin_of_text|>`. Emitted twice: once before the start
    /// marker and once after it. See the file header for why.
    pub bos_token: usize,
    /// Whether to emit the leading BOS that vLLM's re-tokenization adds.
    pub leading_bos: bool,
    /// Appended: `<|eot_id|>`, `<custom_token_4>`, `<custom_token_5>`,
    /// `<custom_token_1>`.
    pub prompt_end: Vec<usize>,
    /// Ends generation. `<custom_token_2>` is the end-of-audio marker.
    ///
    /// Deliberately does not include 128009: that token is a separator inside
    /// the prompt, and the model emits it freely without meaning to stop.
    pub stop_tokens: Vec<usize>,
}

impl Default for OrpheusConfig {
    fn default() -> Self {
        OrpheusConfig {
            audio_token_base: 128266,
            codes_per_frame: 7,
            codebook_size: 4096,
            prompt_start: 128259,
            bos_token: 128000,
            leading_bos: true,
            prompt_end: vec![128009, 128260, 128261, 128257],
            stop_tokens: vec![128258],
        }
    }
}

impl OrpheusConfig {
    pub fn defaults() -> Self {
        OrpheusConfig::default()
    }

    /// One past the last audio token id.
    pub fn audio_token_limit(&self) -> usize {
        self.audio_token_base + self.codes_per_frame * self.codebook_size
    }

    /// True when `id` falls in the audio range for any slot.
    pub fn is_audio_token(&self, id: usize) -> bool {
        id >= self.audio_token_base && id < self.audio_token_limit()
    }
}

/// A speaker one of the Orpheus fine-tunes was trained on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrpheusVoice {
    pub name: &'static str,
    /// ISO 639-1 code of the checkpoint that carries this voice.
    pub language: &'static str,
}

const fn voice(name: &'static str, language: &'static str) -> OrpheusVoice {
    OrpheusVoice { name, language }
}

static ORPHEUS_VOICES: [OrpheusVoice; 14] = [
    // canopylabs/orpheus-3b-0.1-ft
    voice("tara", "en"),
    voice("leah", "en"),
    voice("jess", "en"),
    voice("leo", "en"),
    voice("dan", "en"),
    voice("mia", "en"),
    voice("zac", "en"),
    voice("zoe", "en"),
    // canopylabs/3b-es_it-ft-research_release
    voice("javi", "es"),
    voice("sergio", "es"),
    voice("maria", "es"),
    voice("pietro", "it"),
    voice("giulia", "it"),
    voice("carlo", "it"),
];

/// Every trained voice across the published fine-tunes.
///
/// Which of these actually work depends on the weights loaded: the English
/// fine-tune knows only the `en` voices, the Spanish/Italian research release
/// only the `es` and `it` ones. A voice is nothing but a prompt prefix, so
/// naming one the checkpoint has never seen still synthesises -- it just will
/// not sound like a consistent speaker.
pub fn orpheus_voices() -> &'static [OrpheusVoice] {
    &ORPHEUS_VOICES
}

/// True when `voice` is a trained voice of some checkpoint.
pub fn orpheus_voice_known(voice: &str) -> bool {
    !orpheus_voice_language(voice).is_empty()
}

/// The language a voice belongs to, or empty when it is not a known voice.
pub fn orpheus_voice_language(voice: &str) -> &'static str {
    for v in orpheus_voices() {
        if v.name == voice {
            return v.language;
        }
    }
    ""
}

/// The trained voices for one language code.
pub fn orpheus_voices_for(language: &str) -> Vec<String> {
    let mut out = Vec::new();
    for v in orpheus_voices() {
        if v.language == language {
            out.push(v.name.to_string());
        }
    }
    out
}

/// `SamplingParams` with the defaults the C++ struct declares, which the
/// Rust struct has no `Default` for.
fn sampling_defaults() -> SamplingParams {
    SamplingParams {
        temperature: 1.0,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        seed: 0,
        eos_token_id: None,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        allowed_min: None,
        allowed_max: None,
        allowed_extra: Vec::new(),
    }
}

/// Sensible sampling defaults, matching the reference engine.
pub fn orpheus_default_sampling(seed: u64) -> SamplingParams {
    let mut p = sampling_defaults();
    p.temperature = 0.6;
    p.top_p = 0.8;
    p.top_k = 0;
    p.repetition_penalty = 1.3;
    p.seed = seed;
    p
}

// =============================================================================
// Code stream
// =============================================================================

/// Turns the sampled token stream into SNAC codes, tracking slot position and
/// resynchronising on unexpected tokens.
#[derive(Clone, Debug)]
pub struct OrpheusCodeStream {
    cfg: OrpheusConfig,
    /// Accepted codes in emission order.
    flat: Vec<u32>,
    rejected: usize,
}

impl OrpheusCodeStream {
    pub fn new(cfg: OrpheusConfig) -> Self {
        OrpheusCodeStream { cfg, flat: Vec::new(), rejected: 0 }
    }

    /// Offer a sampled id.
    ///
    /// Returns true when it was a valid code for the expected slot and was
    /// kept. A false return leaves the slot counter untouched, which is what
    /// lets the stream recover.
    pub fn push(&mut self, token_id: usize) -> bool {
        if token_id < self.cfg.audio_token_base {
            self.rejected += 1;
            return false;
        }

        let slot = self.flat.len() % self.cfg.codes_per_frame;
        let offset = self.cfg.audio_token_base + slot * self.cfg.codebook_size;
        if token_id < offset {
            // The token belongs to an earlier slot than the one expected.
            self.rejected += 1;
            return false;
        }

        let code = token_id - offset;
        if code >= self.cfg.codebook_size {
            // Belongs to a later slot. Not advancing the counter is what lets the
            // stream resynchronise.
            self.rejected += 1;
            return false;
        }

        self.flat.push(code as u32);
        true
    }

    /// Codes accepted so far.
    pub fn accepted(&self) -> usize {
        self.flat.len()
    }

    /// Codes rejected so far, for reporting -- a high count means the backbone
    /// is drifting out of the audio vocabulary.
    pub fn rejected(&self) -> usize {
        self.rejected
    }

    /// Complete 7-code groups.
    pub fn complete_groups(&self) -> usize {
        self.flat.len() / self.cfg.codes_per_frame
    }

    /// Frames at the finest codebook rate, i.e. 4 per complete group.
    pub fn frames(&self) -> usize {
        // Level 2 runs at the finest rate and takes four codes per group.
        self.complete_groups() * 4
    }

    /// The three SNAC code levels, truncated to complete groups.
    ///
    /// Empty when no group has completed. Level 0 gets one code per group,
    /// level 1 two, level 2 four.
    pub fn snac_codes(&self) -> Vec<Vec<u32>> {
        let groups = self.complete_groups();
        if groups == 0 {
            return Vec::new();
        }

        let mut level0: Vec<u32> = Vec::with_capacity(groups);
        let mut level1: Vec<u32> = Vec::with_capacity(groups * 2);
        let mut level2: Vec<u32> = Vec::with_capacity(groups * 4);

        for g in 0..groups {
            let base = g * self.cfg.codes_per_frame;
            // The 1:2:4 interleave. Slots 1 and 4 go to the middle scale; slots
            // 2, 3, 5 and 6 to the finest, in that order.
            level0.push(self.flat[base + 0]);
            level1.push(self.flat[base + 1]);
            level1.push(self.flat[base + 4]);
            level2.push(self.flat[base + 2]);
            level2.push(self.flat[base + 3]);
            level2.push(self.flat[base + 5]);
            level2.push(self.flat[base + 6]);
        }

        vec![level0, level1, level2]
    }
}

// =============================================================================
// Synthesis
// =============================================================================

#[derive(Clone, Debug)]
pub struct OrpheusRequest {
    pub text: String,
    pub voice: String,
    /// 1200 tokens is about 14.6 s of audio.
    pub max_new: usize,
    /// The reference uses temperature 0.6, top-p 0.8 and repetition penalty
    /// 1.3. Note that this project's penalty applies over a trailing 64-token
    /// window of generated ids, while the reference applies it over the whole
    /// context -- at 7 codes a frame, 64 tokens is about 9 frames.
    pub sampling: SamplingParams,
    pub noise: SnacNoise,
    /// Restrict sampling to the audio range after the prompt.
    ///
    /// The model is fine-tuned to emit only audio tokens here, so this changes
    /// nothing when it behaves. It stops a drifting model from emitting text
    /// that would otherwise be discarded as rejected codes.
    pub mask_to_audio: bool,
    pub debug: bool,
}

impl Default for OrpheusRequest {
    fn default() -> Self {
        OrpheusRequest {
            text: String::new(),
            voice: "tara".to_string(),
            max_new: 1200,
            sampling: sampling_defaults(),
            noise: SnacNoise::Seeded,
            mask_to_audio: true,
            debug: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrpheusResult {
    pub samples: Vec<f32>,
    pub sample_rate: usize,
    pub tokens_generated: usize,
    pub codes_accepted: usize,
    pub codes_rejected: usize,
    pub groups: usize,
    pub generate_seconds: f64,
    pub decode_seconds: f64,
}

impl Default for OrpheusResult {
    fn default() -> Self {
        OrpheusResult {
            samples: Vec::new(),
            sample_rate: 24000,
            tokens_generated: 0,
            codes_accepted: 0,
            codes_rejected: 0,
            groups: 0,
            generate_seconds: 0.0,
            decode_seconds: 0.0,
        }
    }
}

// =============================================================================
// Result helpers
// =============================================================================

impl OrpheusResult {
    /// Audio duration in seconds.
    pub fn audio_seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.samples.len() as f64 / self.sample_rate as f64
        }
    }

    /// Wall time over audio duration. Below 1.0 is faster than realtime.
    pub fn realtime_factor(&self) -> f64 {
        let audio = self.audio_seconds();
        if audio <= 0.0 { 0.0 } else { (self.generate_seconds + self.decode_seconds) / audio }
    }
}

// =============================================================================
// Prompt
// =============================================================================

/// Build the framed prompt token sequence for a voice and text.
///
/// The markers are injected as raw ids: encoding the literal text
/// "<custom_token_3>" would split it into ordinary subword pieces.
pub fn orpheus_build_prompt(
    tok: &HfBpeTokenizer,
    cfg: &OrpheusConfig,
    voice: &str,
    text: &str,
) -> Result<Vec<usize>, String> {
    if text.is_empty() {
        return Err("orpheus: prompt text is empty".to_string());
    }

    let body = format!("{}: {}", voice, text);
    let encoded: Vec<u32> = tok.encode(&body);
    if encoded.is_empty() {
        return Err("orpheus: prompt encoded to no tokens".to_string());
    }

    let mut ids: Vec<usize> = Vec::with_capacity(encoded.len() + 3 + cfg.prompt_end.len());
    if cfg.leading_bos {
        ids.push(cfg.bos_token);
    }
    ids.push(cfg.prompt_start);
    ids.push(cfg.bos_token);
    for &id in &encoded {
        ids.push(id as usize);
    }
    ids.extend_from_slice(&cfg.prompt_end);
    Ok(ids)
}

// =============================================================================
// Synthesis
// =============================================================================

/// Generate audio tokens and decode them to samples.
pub fn orpheus_synthesize(
    model: &LlamaModel,
    snac: &SnacDecoder,
    tok: &HfBpeTokenizer,
    request: &OrpheusRequest,
    cfg: &OrpheusConfig,
) -> Result<OrpheusResult, String> {
    let prompt = orpheus_build_prompt(tok, cfg, &request.voice, &request.text)?;

    if !orpheus_voice_known(&request.voice) {
        eprintln!(
            "[ orpheus ] Warning: '{}' is not a trained voice of any published \
             checkpoint; output will not sound like a consistent speaker",
            request.voice
        );
    }

    let mut params = request.sampling.clone();
    // Stops are handled entirely in the callback below, which can recognise
    // any number of them; the sampler's single eos slot would only cover one.
    params.eos_token_id = None;
    if request.mask_to_audio {
        params.allowed_min = Some(cfg.audio_token_base);
        params.allowed_max = Some(cfg.audio_token_limit());
        // The stop marker has to stay reachable or generation runs to max_new.
        params.allowed_extra = cfg.stop_tokens.clone();
    }

    if request.debug {
        let mut line = format!("[ orpheus ] prompt ({} tokens):", prompt.len());
        for &id in &prompt {
            line.push_str(&format!(" {}", id));
        }
        eprintln!("{}", line);
    }

    let mut stream = OrpheusCodeStream::new(cfg.clone());
    let mut hit_stop = false;

    let gen_start = Instant::now();
    let produced = model.generate(&prompt, request.max_new, &params, request.debug, |id| {
        if cfg.stop_tokens.contains(&id) {
            hit_stop = true;
            return false;
        }
        let want_slot = stream.accepted() % cfg.codes_per_frame;
        let kept = stream.push(id);
        if request.debug {
            // Which codebook slot the token's id actually
            // falls in, against the one the stream wanted.
            // A healthy stream walks 0,1,2,3,4,5,6 and
            // repeats; anything else is the frame structure
            // breaking down, which is invisible in the audio.
            let band: i64 = if cfg.is_audio_token(id) {
                ((id - cfg.audio_token_base) / cfg.codebook_size) as i64
            } else {
                -1
            };
            eprintln!(
                "[ orpheus ] id {:6} band {:2} want {} {}",
                id,
                band,
                want_slot,
                if kept { "keep" } else { "drop" }
            );
        }
        true
    });
    let gen_seconds = gen_start.elapsed().as_secs_f64();

    let mut result = OrpheusResult {
        sample_rate: snac.config.sampling_rate,
        tokens_generated: produced,
        codes_accepted: stream.accepted(),
        codes_rejected: stream.rejected(),
        groups: stream.complete_groups(),
        generate_seconds: gen_seconds,
        ..OrpheusResult::default()
    };

    if request.debug {
        eprintln!(
            "[ orpheus ] {} tokens, {} codes kept, {} rejected, {} groups, stop={}",
            produced,
            stream.accepted(),
            stream.rejected(),
            stream.complete_groups(),
            if hit_stop { "yes" } else { "no" }
        );
    }

    let codes = stream.snac_codes();
    if codes.is_empty() {
        return Err(format!(
            "orpheus: the model produced no complete 7-code group ({} tokens, {} codes kept, {} rejected)",
            produced,
            stream.accepted(),
            stream.rejected()
        ));
    }

    let dec_start = Instant::now();
    let samples = snac.decode(&codes, request.noise, request.sampling.seed)?;
    let dec_seconds = dec_start.elapsed().as_secs_f64();

    result.samples = samples;
    result.decode_seconds = dec_seconds;
    Ok(result)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd2::Mat;
    use crate::gguf_loader::{GgufFile, GgufMetaValue};
    use crate::snac::{SnacConfig, SnacQuantizer, SnacQuantizerLevel};
    use crate::tokenizer::PreTokenizer;
    use crate::transformer5::{Config5, LlamaKvCache, load_gguf_tokenizer};
    use crate::wav::wave_stats;

    /// An Orpheus checkpoint found on disk, with text it can actually speak.
    ///
    /// The published fine-tunes share an architecture and a vocabulary but not a
    /// language, and which one is present depends on what has been downloaded. So
    /// the integration tests discover the checkpoint rather than naming it, and
    /// take their prompt from whatever `general.languages` it declares -- an
    /// English sentence in an Italian voice would test very little.
    struct FoundModel {
        path: String,
        language: String,
        prompt: String,
        voice: String,
    }

    fn find_orpheus_model() -> Option<FoundModel> {
        if !std::path::Path::new("models").is_dir() {
            return None;
        }
        let entries = std::fs::read_dir("models").ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("gguf") {
                continue;
            }
            let path_str = path.to_string_lossy().to_string();
            let Ok(gguf) = GgufFile::open(&path_str) else {
                continue;
            };
            let arch = gguf.metadata.get("general.architecture");
            if arch.and_then(|a| a.as_str()) != Some("llama") {
                continue;
            }
            // The audio-token arithmetic depends on this exact vocabulary.
            let vocab = gguf.metadata.get("llama.vocab_size");
            if vocab.and_then(|v| v.as_u64()) != Some(156940) {
                continue;
            }

            let mut found = FoundModel {
                path: path_str,
                language: "en".to_string(),
                prompt: String::new(),
                voice: String::new(),
            };
            if let Some(GgufMetaValue::Array(array)) = gguf.metadata.get("general.languages") {
                for v in array {
                    if let Some(code) = v.as_str() {
                        // Prefer a language this build has voices for.
                        if !orpheus_voices_for(code).is_empty() {
                            found.language = code.to_string();
                            break;
                        }
                    }
                }
            }

            found.prompt = if found.language == "it" {
                "Ciao, oggi e una bella giornata.".to_string()
            } else if found.language == "es" {
                "Hola, hoy hace un dia estupendo.".to_string()
            } else {
                "Hello, my name is Tara.".to_string()
            };
            let voices = orpheus_voices_for(&found.language);
            found.voice = voices.first().cloned().unwrap_or_else(|| "tara".to_string());
            return Some(found);
        }
        None
    }

    /// The id of the code-`code` token for slot `slot`.
    fn audio_id(cfg: &OrpheusConfig, slot: usize, code: usize) -> usize {
        cfg.audio_token_base + slot * cfg.codebook_size + code
    }

    /// Feed one complete group whose codes are 0..6 in slot order.
    fn push_group(stream: &mut OrpheusCodeStream, cfg: &OrpheusConfig, first: usize) {
        for slot in 0..cfg.codes_per_frame {
            assert!(stream.push(audio_id(cfg, slot, first + slot)));
        }
    }

    // -------------------------------------------------------------------------
    // Token protocol
    // -------------------------------------------------------------------------

    #[test]
    fn audio_token_range_matches_the_custom_token_layout() {
        let cfg = OrpheusConfig::defaults();
        // <custom_token_0> is id 128256, and the reference offsets codes by 10,
        // so slot 0 code 0 is <custom_token_10> at 128266.
        assert_eq!(cfg.audio_token_base, 128266);
        // Seven codebooks of 4096: the last audio id is 156937.
        assert_eq!(cfg.audio_token_limit(), 156938);
        assert_eq!(cfg.audio_token_limit() - cfg.audio_token_base, 7 * 4096);

        assert!(!cfg.is_audio_token(128265));
        assert!(cfg.is_audio_token(128266));
        assert!(cfg.is_audio_token(156937));
        assert!(!cfg.is_audio_token(156938));
        // The markers are all below the audio range.
        assert!(!cfg.is_audio_token(cfg.prompt_start));
        for &id in &cfg.stop_tokens {
            assert!(!cfg.is_audio_token(id));
        }
    }

    #[test]
    fn the_audio_range_fits_inside_the_orpheus_vocabulary() {
        let cfg = OrpheusConfig::defaults();
        assert!(cfg.audio_token_limit() <= Config5::orpheus_3b().vocab_size);
    }

    #[test]
    fn voices_carry_the_language_of_their_checkpoint() {
        assert!(orpheus_voice_known("tara"));
        assert!(orpheus_voice_known("giulia"));
        assert!(!orpheus_voice_known("nobody"));

        assert_eq!(orpheus_voice_language("tara"), "en");
        assert_eq!(orpheus_voice_language("giulia"), "it");
        assert_eq!(orpheus_voice_language("javi"), "es");
        assert!(orpheus_voice_language("nobody").is_empty());

        // The English fine-tune has eight speakers; the Spanish/Italian research
        // release has three each.
        assert_eq!(orpheus_voices_for("en").len(), 8);
        assert_eq!(orpheus_voices_for("it"), vec!["pietro", "giulia", "carlo"]);
        assert_eq!(orpheus_voices_for("es"), vec!["javi", "sergio", "maria"]);
        assert!(orpheus_voices_for("xx").is_empty());
    }

    #[test]
    fn default_sampling_matches_the_reference_engine() {
        let p = orpheus_default_sampling(7);
        assert_eq!(p.temperature, 0.6f32);
        assert_eq!(p.top_p, 0.8f32);
        assert_eq!(p.repetition_penalty, 1.3f32);
        assert_eq!(p.seed, 7);
    }

    // -------------------------------------------------------------------------
    // Code stream
    // -------------------------------------------------------------------------

    #[test]
    fn code_stream_decodes_each_slot_at_its_own_offset() {
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());

        // Slot i's code c sits at base + i*4096 + c.
        for slot in 0..7 {
            assert!(stream.push(audio_id(&cfg, slot, 100 + slot)));
        }
        assert_eq!(stream.accepted(), 7);
        assert_eq!(stream.rejected(), 0);
        assert_eq!(stream.complete_groups(), 1);

        let codes = stream.snac_codes();
        assert_eq!(codes.len(), 3);
        // slot 0 -> level 0
        assert_eq!(codes[0], vec![100u32]);
        // slots 1 and 4 -> level 1
        assert_eq!(codes[1], vec![101u32, 104]);
        // slots 2, 3, 5, 6 -> level 2
        assert_eq!(codes[2], vec![102u32, 103, 105, 106]);
    }

    #[test]
    fn code_stream_keeps_code_0() {
        // The reference accepts a code only when strictly positive, discarding
        // every legitimate code 0. Since a rejected token does not advance the
        // slot counter, dropping one shifts every code after it by a slot and
        // corrupts the rest of the utterance.
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());
        for slot in 0..7 {
            assert!(stream.push(audio_id(&cfg, slot, 0)));
        }
        assert_eq!(stream.accepted(), 7);
        assert_eq!(stream.rejected(), 0);
        let codes = stream.snac_codes();
        assert_eq!(codes[0], vec![0u32]);
        assert_eq!(codes[1], vec![0u32, 0]);
        assert_eq!(codes[2], vec![0u32, 0, 0, 0]);
    }

    #[test]
    fn code_stream_ignores_non_audio_tokens() {
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg);
        assert!(!stream.push(0));
        assert!(!stream.push(1000));
        assert!(!stream.push(128265)); // one below the audio range
        assert!(!stream.push(200000)); // above the vocabulary
        assert_eq!(stream.accepted(), 0);
        assert_eq!(stream.rejected(), 4);
        assert!(stream.snac_codes().is_empty());
    }

    #[test]
    fn code_stream_rejects_without_advancing_the_slot() {
        // This is the resynchronisation property. A token for the wrong slot
        // computes a code outside [0, 4096), and refusing it while leaving the
        // counter alone means the next correct token still lands in the right slot.
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());

        // Expecting slot 0, but hand it a slot-3 token: code would be 3*4096+5.
        assert!(!stream.push(audio_id(&cfg, 3, 5)));
        assert_eq!(stream.accepted(), 0);
        assert_eq!(stream.rejected(), 1);

        // The stream is still waiting for slot 0, so a genuine slot-0 token lands.
        assert!(stream.push(audio_id(&cfg, 0, 42)));
        assert_eq!(stream.accepted(), 1);

        // Now expecting slot 1; a slot-0 token computes a negative code.
        assert!(!stream.push(audio_id(&cfg, 0, 1)));
        assert_eq!(stream.rejected(), 2);
        assert!(stream.push(audio_id(&cfg, 1, 7)));
        assert_eq!(stream.accepted(), 2);
    }

    #[test]
    fn code_stream_truncates_to_complete_groups() {
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());
        push_group(&mut stream, &cfg, 0);
        // Three extra codes: not a group, so they must not reach the codes.
        for slot in 0..3 {
            assert!(stream.push(audio_id(&cfg, slot, 200 + slot)));
        }
        assert_eq!(stream.accepted(), 10);
        assert_eq!(stream.complete_groups(), 1);

        let codes = stream.snac_codes();
        assert_eq!(codes[0].len(), 1);
        assert_eq!(codes[1].len(), 2);
        assert_eq!(codes[2].len(), 4);
    }

    #[test]
    fn code_stream_produces_snacs_1_2_4_code_ratio() {
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());
        const GROUPS: usize = 5;
        for g in 0..GROUPS {
            push_group(&mut stream, &cfg, g * 7);
        }
        assert_eq!(stream.complete_groups(), GROUPS);
        // Each group covers 4 frames at the finest rate.
        assert_eq!(stream.frames(), GROUPS * 4);

        let codes = stream.snac_codes();
        // The strides the SNAC quantizer will check: 4, 2, 1 against frames.
        assert_eq!(codes[0].len() * 4, stream.frames());
        assert_eq!(codes[1].len() * 2, stream.frames());
        assert_eq!(codes[2].len() * 1, stream.frames());
    }

    #[test]
    fn code_stream_output_is_accepted_by_the_snac_quantizer() {
        // The two halves of the pipeline agree on frame counts. This is the seam
        // where a stride or interleave mistake would show up, and it needs no
        // weights to check.
        let cfg = OrpheusConfig::defaults();
        let mut stream = OrpheusCodeStream::new(cfg.clone());
        for g in 0..3 {
            push_group(&mut stream, &cfg, g);
        }
        let codes = stream.snac_codes();

        let mut q = SnacQuantizer { levels: Vec::new(), latent_dim: 4 };
        for &stride in &SnacConfig::snac_24khz().vq_strides {
            q.levels.push(SnacQuantizerLevel {
                codebook: Mat::zeros(cfg.codebook_size, 2),
                out_proj_weight: Mat::zeros(4, 2),
                stride,
                ..SnacQuantizerLevel::default()
            });
        }

        let z = q.from_codes(&codes);
        assert!(z.is_ok());
        let z = z.unwrap();
        assert_eq!(z.rows, stream.frames());
        assert_eq!(z.rows, 12); // 3 groups x 4 frames
    }

    // -------------------------------------------------------------------------
    // Prompt framing
    // -------------------------------------------------------------------------

    #[test]
    fn orpheus_build_prompt_frames_the_encoded_text() {
        let Some(model) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let gguf = GgufFile::open(&model.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let tok = match load_gguf_tokenizer(&gguf) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        let cfg = OrpheusConfig::defaults();
        let ids = orpheus_build_prompt(&tok, &cfg, "tara", "Hello there.");
        assert!(ids.is_ok());
        let ids = ids.unwrap();

        // [BOS, start marker, BOS] + text + end markers. Both BOS tokens are
        // deliberate; see the orpheus.rs header.
        assert!(cfg.leading_bos);
        assert_eq!(ids[0], cfg.bos_token);
        assert_eq!(ids[1], cfg.prompt_start);
        assert_eq!(ids[2], cfg.bos_token);
        assert!(ids.len() > 3 + cfg.prompt_end.len());
        for i in 0..cfg.prompt_end.len() {
            assert_eq!(ids[ids.len() - cfg.prompt_end.len() + i], cfg.prompt_end[i]);
        }
        // The body carries no audio tokens.
        for &id in &ids {
            assert!(!cfg.is_audio_token(id));
        }
    }

    #[test]
    fn orpheus_build_prompt_can_drop_the_leading_bos() {
        let Some(model) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let gguf = GgufFile::open(&model.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let tok = load_gguf_tokenizer(&gguf);
        assert!(tok.is_ok());
        let tok = tok.unwrap();

        let mut cfg = OrpheusConfig::defaults();
        cfg.leading_bos = false;
        let ids = orpheus_build_prompt(&tok, &cfg, "tara", "Hello there.");
        assert!(ids.is_ok());
        let ids = ids.unwrap();
        assert_eq!(ids[0], cfg.prompt_start);
        assert_eq!(ids[1], cfg.bos_token);
    }

    #[test]
    fn eot_128009_is_a_prompt_separator_not_a_stop_token() {
        // It appears inside prompt_end, and the model emits it freely. Treating it
        // as a stop ends generation within a few tokens.
        let cfg = OrpheusConfig::defaults();
        assert!(cfg.prompt_end.contains(&128009));
        assert!(!cfg.stop_tokens.contains(&128009));
        assert_eq!(cfg.stop_tokens, vec![128258usize]);
    }

    #[test]
    fn orpheus_build_prompt_rejects_empty_text() {
        // Needs no tokenizer: the check comes first.
        let cfg = OrpheusConfig::defaults();
        let tok = HfBpeTokenizer::from_vocab_and_merges(
            vec!["a".to_string(), "b".to_string()],
            &[],
            true,
            PreTokenizer::Llama3,
        );
        assert!(tok.is_ok());
        let tok = tok.unwrap();
        let ids = orpheus_build_prompt(&tok, &cfg, "tara", "");
        assert!(ids.is_err());
    }

    // -------------------------------------------------------------------------
    // The embedded tokenizer
    // -------------------------------------------------------------------------

    #[test]
    fn load_gguf_tokenizer_reads_the_embedded_vocabulary() {
        let Some(model) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let gguf = GgufFile::open(&model.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let tok = match load_gguf_tokenizer(&gguf) {
            Ok(t) => t,
            Err(e) => panic!("{}", e),
        };

        assert_eq!(tok.vocab_size(), 156940);
        assert!(tok.byte_level());
        // tokenizer.ggml.pre is "llama-bpe", which groups digits.
        assert_eq!(tok.pre_tokenizer(), PreTokenizer::Llama3);

        // The audio tokens are where the code arithmetic assumes.
        assert_eq!(tok.token_text(128256), "<custom_token_0>");
        assert_eq!(tok.token_text(128266), "<custom_token_10>");
        assert_eq!(
            tok.token_text(OrpheusConfig::defaults().audio_token_base as u32),
            "<custom_token_10>"
        );

        // Round-trip ordinary text.
        let text = "tara: Hello there, friend.";
        let ids = tok.encode(text);
        assert!(!ids.is_empty());
        assert_eq!(tok.decode(&ids), text);
    }

    #[test]
    fn the_gguf_tokenizer_groups_digits_the_llama_3_way() {
        let Some(model) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let gguf = GgufFile::open(&model.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let tok = load_gguf_tokenizer(&gguf);
        assert!(tok.is_ok());
        let tok = tok.unwrap();

        // GPT-2 splits every digit; Llama 3's `\p{N}{1,3}` takes runs of three, so
        // "2024" is "202" + "4". Getting this wrong mispronounces every number.
        let llama_split = HfBpeTokenizer::pretokenize("2024", PreTokenizer::Llama3);
        assert_eq!(llama_split, vec!["202", "4"]);

        let gpt2_split = HfBpeTokenizer::pretokenize("2024", PreTokenizer::Gpt2);
        assert_eq!(gpt2_split, vec!["2", "0", "2", "4"]);

        // And the instance actually uses the Llama 3 clause, so the two encodings
        // differ.
        assert!(tok.encode("2024").len() < 4);
        // Round-tripping still recovers the text either way.
        assert_eq!(tok.decode(&tok.encode("in 2024 and 7")), "in 2024 and 7");
    }

    // -------------------------------------------------------------------------
    // RoPE scaling
    // -------------------------------------------------------------------------

    #[test]
    fn rope_freqs_divisors_stretch_the_low_frequency_bands() {
        let Some(found) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let mut model = LlamaModel::new_for_inference(Config5::orpheus_3b());
        // The unscaled frequencies, before the file is read.
        let plain = model.config.inv_freq();

        let gguf = GgufFile::open(&found.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let idx = gguf.find_tensor("rope_freqs.weight");
        assert!(idx.is_some());
        let divisors = gguf.decode_f32(idx.unwrap());
        assert!(divisors.is_ok());
        let divisors = divisors.unwrap();
        assert_eq!(divisors.len(), 64);

        // Divisors, not multipliers: they run from 1.0 up to 32.0.
        assert_eq!(divisors[0], 1.0f32);
        assert_eq!(*divisors.last().unwrap(), 32.0f32);

        model.config.rope_freq_divisors = divisors;
        let scaled = model.config.inv_freq();
        assert_eq!(scaled.len(), plain.len());

        // A divisor of 1 leaves the band alone; 32 divides it. Applying these as
        // multipliers would raise the last band by 32x instead -- a factor of 1024
        // the wrong way, on exactly the dimensions carrying long-range position.
        assert_eq!(scaled[0], plain[0]);
        assert!(scaled.last().unwrap() < plain.last().unwrap());
        assert!((scaled.last().unwrap() - plain.last().unwrap() / 32.0f32).abs() < 1e-12f32);
    }

    // -------------------------------------------------------------------------
    // End to end
    // -------------------------------------------------------------------------

    #[test]
    fn orpheus_synthesises_speech_shaped_audio() {
        // Tagged separately from [.integration] in the C++ suite: this one loads
        // 2.4 GB of weights and generates, so it is a minute of work rather than
        // a second.
        let found = find_orpheus_model();
        let Some(found) = found.filter(|_| std::path::Path::new("models/snac_24khz.bin").exists())
        else {
            eprintln!("skip: Orpheus or SNAC weights not present");
            return;
        };
        eprintln!("using {} ({}, voice {})", found.path, found.language, found.voice);

        let gguf = GgufFile::open(&found.path);
        assert!(gguf.is_ok());
        let gguf = gguf.unwrap();
        let tok = load_gguf_tokenizer(&gguf);
        assert!(tok.is_ok());
        let tok = tok.unwrap();

        let mut model = LlamaModel::new_for_inference(Config5::orpheus_3b());
        if let Err(e) = model.load_weights_from_gguf(&found.path) {
            panic!("{}", e);
        }
        model.quantize_lm_head();

        let snac = SnacDecoder::load("models/snac_24khz.bin", SnacConfig::snac_24khz());
        assert!(snac.is_ok());
        let snac = snac.unwrap();

        let request = OrpheusRequest {
            text: found.prompt.clone(),
            voice: found.voice.clone(),
            max_new: 210, // 30 groups, about 2.5 s
            sampling: orpheus_default_sampling(1234),
            ..OrpheusRequest::default()
        };

        let result =
            match orpheus_synthesize(&model, &snac, &tok, &request, &OrpheusConfig::defaults()) {
                Ok(r) => r,
                Err(e) => panic!("{}", e),
            };

        // The backbone should be emitting audio tokens almost exclusively.
        let info = format!(
            "tokens {} kept {} rejected {}",
            result.tokens_generated, result.codes_accepted, result.codes_rejected
        );
        assert!(result.codes_accepted > 0, "{}", info);
        assert!(result.groups > 0, "{}", info);
        assert!(result.codes_rejected * 10 < result.codes_accepted, "{}", info);

        // Length is fixed by the frame accounting, with no slack.
        assert_eq!(result.samples.len(), result.groups * 4 * 512);

        let stats = wave_stats(&result.samples);
        assert!(stats.in_range(), "{}", stats.describe());
        assert!(stats.looks_like_speech(), "{}", stats.describe());
    }

    // -------------------------------------------------------------------------
    // Decode-path self-consistency
    // -------------------------------------------------------------------------

    #[test]
    fn prefill_and_incremental_decode_agree() {
        // No oracle needed: running N tokens through prefill must give the same
        // final-position logits as running N-1 through prefill and the last one
        // through the incremental decode path. Any disagreement is a KV cache or
        // RoPE-offset bug, which otherwise shows up only as audio that starts
        // plausible and degrades.
        let Some(found) = find_orpheus_model() else {
            eprintln!("skip: no Orpheus GGUF in models/");
            return;
        };
        let mut model = LlamaModel::new_for_inference(Config5::orpheus_3b());
        if let Err(e) = model.load_weights_from_gguf(&found.path) {
            panic!("{}", e);
        }

        let ids: Vec<usize> = vec![128000, 128259, 128000, 83, 5169, 25, 22691, 11, 13];

        let mut full = LlamaKvCache::new(&model.config, 64);
        let all_at_once = model.forward_cached(&ids, &mut full);

        let mut split = LlamaKvCache::new(&model.config, 64);
        let head = &ids[..ids.len() - 1];
        // Only the cache it leaves behind matters here.
        let _ = model.forward_cached(head, &mut split);
        let incremental = model.forward_cached(&[*ids.last().unwrap()], &mut split);

        assert_eq!(all_at_once.cols, incremental.cols);

        // The two paths are not bit-identical and are not meant to be:
        // `gqa_attention_cached` runs per-head sgemm for prefill and sdot/saxpy for
        // a single decode query, so the reductions sum in different orders. Over 28
        // layers of 3072-wide f32 reductions that drifts by a few hundredths of a
        // logit. What has to hold is the ranking -- a wrong RoPE offset or a
        // misaligned cache moves logits by whole units and reorders the top of the
        // distribution.
        let top5 = |logits: &Mat| -> Vec<usize> {
            let mut idx: Vec<usize> = (0..logits.cols).collect();
            idx.sort_by(|&a, &b| logits.at(0, b).total_cmp(&logits.at(0, a)));
            idx.truncate(5);
            idx
        };

        let mut max_delta = 0.0f32;
        for c in 0..all_at_once.cols {
            max_delta = max_delta.max((all_at_once.at(0, c) - incremental.at(0, c)).abs());
        }

        let a = top5(&all_at_once);
        let b = top5(&incremental);
        let info = format!(
            "prefill top1 {} logit {}, decode top1 {} logit {}, max |delta| {}",
            a[0],
            all_at_once.at(0, a[0]),
            b[0],
            incremental.at(0, b[0]),
            max_delta
        );
        assert_eq!(a, b, "{}", info);
        // Loose enough for accumulation order, far tighter than any real bug.
        assert!(max_delta < 0.5f32, "{}", info);
    }
}
