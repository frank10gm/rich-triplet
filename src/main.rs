mod autograd;
mod autograd2;
mod clip_text;
mod conv1d;
mod conv2d;
mod dataset;
mod duration;
mod flux;
mod gguf_loader;
mod hubert;
#[cfg(feature = "metal")]
mod metal_ops;
#[cfg(feature = "metal")]
mod metal_decode;
#[cfg(feature = "metal")]
mod metal_decode_qwen35;
#[cfg(feature = "metal")]
mod metal_flux;
#[cfg(feature = "metal")]
mod metal_omnivoice;
mod ndarray;
mod nn;
mod nn2;
mod omnivoice;
mod omnivoice_codec;
mod orpheus;
mod png;
mod qlinear;
mod resample;
mod snac;
mod t5;
mod tensor;
mod tokenizer;
mod torch_pickle;
mod train;
mod train2;
mod transformer;
mod transformer2;
mod transformer3;
mod transformer4;
mod transformer5;
mod transformer6;
mod transformer_qwen35;
mod vae;
mod wav;

use autograd2::restore_checkpoint;
use dataset::TextDataset;
use nn::InitRng;
use tokenizer::{BpeTokenizer, CharTokenizer, Tokenizer};
use train::{TrainConfig, generate, train};
use train2::{TrainConfig2, train2};
use transformer::{Config, Gpt};
use transformer2::Gpt2;
use transformer3::{Config3, GptOssModel, SamplingParams};

// =============================================================================
// Bilingual training corpus — Italian and English
// =============================================================================

const CORPUS: &str = "
Il cielo sopra Milano era grigio come sempre. Giovanni guardava dalla finestra
del suo appartamento al quinto piano, pensando alla riunione del mattino.
La sua collega Chiara gli aveva detto che il progetto era in ritardo di due
settimane. Bisognava trovare una soluzione prima di venerdì.

The sky above London was the same color as a television tuned to a dead channel.
Thomas looked out from his office on the fifth floor, thinking about the morning
meeting. His colleague Sarah had told him the project was two weeks behind
schedule. They needed to find a solution before Friday.

La lingua è il vestito del pensiero. Ogni parola porta con sé il peso della
storia, la memoria di chi l'ha usata prima di noi. Imparare una lingua straniera
significa aprire una finestra su un altro mondo, un altro modo di vedere le cose.

Language is the dress of thought. Every word carries with it the weight of
history, the memory of those who used it before us. Learning a foreign language
means opening a window onto another world, another way of seeing things.

Il modello linguistico non capisce davvero le parole. Calcola le probabilità
delle sequenze di caratteri basandosi sui pattern nel testo di addestramento.
Eppure, da questi semplici calcoli, emerge qualcosa che assomiglia alla comprensione.

The language model does not truly understand words. It calculates probabilities
of character sequences based on patterns in the training text. Yet from these
simple calculations something emerges that resembles understanding.

Buongiorno, come stai? Sto bene, grazie. E tu? Anch'io sto bene.
Hello, how are you? I am well, thank you. And you? I am well too.

Milano, Roma, Firenze, Venezia, Napoli, Torino, Bologna, Palermo.
London, Paris, Berlin, Madrid, Rome, Amsterdam, Vienna, Prague.

uno due tre quattro cinque sei sette otto nove dieci
one two three four five six seven eight nine ten

il lo la i gli le un una dello della degli delle
the a an of in on at to for with from by

essere avere fare dire andare venire sapere potere volere
to be to have to do to say to go to come to know to can to want

bello brutto grande piccolo vecchio nuovo buono cattivo
beautiful ugly big small old new good bad

oggi ieri domani adesso sempre mai spesso raramente
today yesterday tomorrow now always never often rarely
";

// =============================================================================
// CLI argument parsing
// =============================================================================

struct CliArgs {
    /// --prompt TEXT     : text to complete (triggers generation mode)
    prompt: Option<String>,
    /// --system TEXT     : ChatML system prompt (Qwen 3.5 chat models)
    system: Option<String>,
    /// --weights DIR     : directory with .safetensors shards (GPT-OSS or Gemma 3)
    weights: Option<String>,
    /// --vocab PATH      : BPE vocab.json (required with --weights for GPT-OSS)
    vocab: Option<String>,
    /// --merges PATH     : BPE merges.txt (required with --weights for GPT-OSS)
    merges: Option<String>,
    /// --tokenizer-model PATH : SentencePiece .model file (required with --weights for Gemma 3)
    tokenizer_model: Option<String>,
    /// --tokenizer-dir DIR   : directory containing tokenizer.json (for GGUF, where tokenizer is separate)
    tokenizer_dir: Option<String>,
    /// --model NAME      : which architecture to use (gpt-oss | gemma3-1b | gemma3-4b)
    model: Option<String>,
    /// --max-new N       : tokens to generate (default 200)
    max_new: usize,
    /// --temp T          : sampling temperature (default 0.8)
    temperature: f32,
    /// --top-k K         : top-k cutoff (default 40, 0 = disabled)
    top_k: usize,
    /// --top-p P         : nucleus probability (default 0.95)
    top_p: f32,
    /// --rep-penalty R   : repetition penalty (default 1.1, 1.0 = disabled)
    rep_penalty: f32,
    /// --seed S          : RNG seed (default 42)
    seed: u64,
    /// --train-steps N   : steps for on-the-fly training (default 200)
    train_steps: usize,
    /// --checkpoint PATH : load a previously saved .ckpt before generating (skips training)
    checkpoint: Option<String>,
    /// --pretokenize SRC DST : tokenize SRC text file → DST .bin file, then exit
    pretokenize: Option<(String, String)>,
    /// --benchmark : run scalar-vs-tensor autograd benchmark
    benchmark: bool,
    /// --quantize : quantize weights to Q4 after loading (saves RAM, may improve output quality)
    quantize: bool,
    /// --debug : enable per-step diagnostic logging (h_rms, logit gaps, top-5 tokens)
    debug: bool,
    /// --draft-len N : max speculative draft tokens per step (default 4, 0 = disabled)
    draft_len: usize,
    /// --voice NAME : Orpheus speaker
    voice: Option<String>,
    /// --snac PATH : SNAC codec checkpoint (pytorch_model.bin)
    snac: Option<String>,
    /// --out PATH : where to write the synthesised WAV
    out: Option<String>,
    /// --no-audio-mask : let Orpheus sample outside the audio token range
    no_audio_mask: bool,
    /// --no-leading-bos : drop the BOS that vLLM's re-tokenization prepends
    no_leading_bos: bool,
    /// --language NAME : OmniVoice language hint
    language: Option<String>,
    /// --instruct TEXT : OmniVoice free-text voice description
    instruct: Option<String>,
    /// --ref-audio PATH : WAV of the voice to clone
    ref_audio: Option<String>,
    /// --ref-text TEXT : what that WAV says
    ref_text: Option<String>,
    /// --duration S : audio seconds to generate; 0 uses the length heuristic
    duration: f32,
    /// --steps N : OmniVoice unmasking steps, or FLUX denoising steps
    steps: usize,
    /// Whether --steps was given, so each model can keep its own default.
    steps_set: bool,
    /// --guidance G : classifier-free guidance scale; 0 disables it
    guidance: f32,
    /// --chunk-seconds S : audio per chunk when splitting long text; 0 never splits
    chunk_seconds: f32,
    /// --chunk-threshold S : split only when the estimate exceeds this
    chunk_threshold: f32,
    /// --chunk-gap S : silence between chunks
    chunk_gap: f32,
    /// --rope-interleaved : pair 2i with 2i+1 instead of i with i+head_dim/2
    rope_interleaved: bool,
    /// --t5 PATH : T5-XXL encoder GGUF, for text-to-image
    t5: Option<String>,
    /// --t5-tokenizer PATH : the encoder's SentencePiece `spiece.model`
    t5_tokenizer: Option<String>,
    /// --clip PATH : CLIP-L text encoder safetensors
    clip: Option<String>,
    /// --clip-tokenizer PATH : CLIP's `tokenizer.json`
    clip_tokenizer: Option<String>,
    /// --vae PATH : image autoencoder safetensors
    vae: Option<String>,
    /// --width / --height : image size in pixels
    width: usize,
    height: usize,
    /// --tile N : VAE tile size in latent pixels; 0 decodes the image whole
    vae_tile: usize,
    /// --cpu : run the diffusion transformer on the CPU even when Metal is on
    force_cpu: bool,
    /// --batch N : generate N images from consecutive seeds in one run
    batch: usize,
}

impl CliArgs {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut a = CliArgs {
            prompt: None,
            system: None,
            weights: None,
            vocab: None,
            merges: None,
            tokenizer_model: None,
            tokenizer_dir: None,
            model: None,
            max_new: 200,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
            rep_penalty: 1.1,
            seed: 42,
            train_steps: 200,
            checkpoint: None,
            pretokenize: None,
            benchmark: false,
            quantize: false,
            debug: false,
            draft_len: 0,
            voice: None,
            snac: None,
            out: None,
            no_audio_mask: false,
            no_leading_bos: false,
            language: None,
            instruct: None,
            ref_audio: None,
            ref_text: None,
            duration: 0.0,
            steps: 12,
            steps_set: false,
            guidance: 2.0,
            chunk_seconds: 15.0,
            chunk_threshold: 30.0,
            chunk_gap: 0.3,
            rope_interleaved: false,
            t5: None,
            t5_tokenizer: None,
            clip: None,
            clip_tokenizer: None,
            vae: None,
            width: 1024,
            height: 1024,
            vae_tile: 0,
            force_cpu: false,
            batch: 1,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--prompt" => {
                    i += 1;
                    if i < args.len() {
                        a.prompt = Some(args[i].clone());
                    }
                }
                "--system" => {
                    i += 1;
                    if i < args.len() {
                        a.system = Some(args[i].clone());
                    }
                }
                "--weights" => {
                    i += 1;
                    if i < args.len() {
                        a.weights = Some(args[i].clone());
                    }
                }
                "--vocab" => {
                    i += 1;
                    if i < args.len() {
                        a.vocab = Some(args[i].clone());
                    }
                }
                "--merges" => {
                    i += 1;
                    if i < args.len() {
                        a.merges = Some(args[i].clone());
                    }
                }
                "--tokenizer-model" => {
                    i += 1;
                    if i < args.len() {
                        a.tokenizer_model = Some(args[i].clone());
                    }
                }
                "--tokenizer-dir" => {
                    i += 1;
                    if i < args.len() {
                        a.tokenizer_dir = Some(args[i].clone());
                    }
                }
                "--model" => {
                    i += 1;
                    if i < args.len() {
                        a.model = Some(args[i].clone());
                    }
                }
                "--max-new" => {
                    i += 1;
                    if i < args.len() {
                        a.max_new = args[i].parse().unwrap_or(200);
                    }
                }
                "--temp" => {
                    i += 1;
                    if i < args.len() {
                        a.temperature = args[i].parse().unwrap_or(0.8);
                    }
                }
                "--top-k" => {
                    i += 1;
                    if i < args.len() {
                        a.top_k = args[i].parse().unwrap_or(40);
                    }
                }
                "--top-p" => {
                    i += 1;
                    if i < args.len() {
                        a.top_p = args[i].parse().unwrap_or(0.95);
                    }
                }
                "--rep-penalty" => {
                    i += 1;
                    if i < args.len() {
                        a.rep_penalty = args[i].parse().unwrap_or(1.1);
                    }
                }
                "--seed" => {
                    i += 1;
                    if i < args.len() {
                        a.seed = args[i].parse().unwrap_or(42);
                    }
                }
                "--train-steps" => {
                    i += 1;
                    if i < args.len() {
                        a.train_steps = args[i].parse().unwrap_or(200);
                    }
                }
                "--checkpoint" => {
                    i += 1;
                    if i < args.len() {
                        a.checkpoint = Some(args[i].clone());
                    }
                }
                "--pretokenize" => {
                    i += 1;
                    let src = if i < args.len() {
                        args[i].clone()
                    } else {
                        String::new()
                    };
                    i += 1;
                    let dst = if i < args.len() {
                        args[i].clone()
                    } else {
                        String::new()
                    };
                    a.pretokenize = Some((src, dst));
                }
                "--benchmark" => {
                    a.benchmark = true;
                }
                "--quantize" => {
                    a.quantize = true;
                }
                "--debug" => {
                    a.debug = true;
                }
                "--voice" => {
                    i += 1;
                    if i < args.len() {
                        a.voice = Some(args[i].clone());
                    }
                }
                "--snac" => {
                    i += 1;
                    if i < args.len() {
                        a.snac = Some(args[i].clone());
                    }
                }
                "--out" => {
                    i += 1;
                    if i < args.len() {
                        a.out = Some(args[i].clone());
                    }
                }
                "--no-audio-mask" => {
                    a.no_audio_mask = true;
                }
                "--no-leading-bos" => {
                    a.no_leading_bos = true;
                }
                "--language" => {
                    i += 1;
                    if i < args.len() {
                        a.language = Some(args[i].clone());
                    }
                }
                "--instruct" => {
                    i += 1;
                    if i < args.len() {
                        a.instruct = Some(args[i].clone());
                    }
                }
                "--chunk-seconds" => {
                    i += 1;
                    if i < args.len() {
                        a.chunk_seconds = args[i].parse().unwrap_or(15.0);
                    }
                }
                "--chunk-threshold" => {
                    i += 1;
                    if i < args.len() {
                        a.chunk_threshold = args[i].parse().unwrap_or(30.0);
                    }
                }
                "--chunk-gap" => {
                    i += 1;
                    if i < args.len() {
                        a.chunk_gap = args[i].parse().unwrap_or(0.3);
                    }
                }
                "--ref-audio" => {
                    i += 1;
                    if i < args.len() {
                        a.ref_audio = Some(args[i].clone());
                    }
                }
                "--ref-text" => {
                    i += 1;
                    if i < args.len() {
                        a.ref_text = Some(args[i].clone());
                    }
                }
                "--duration" => {
                    i += 1;
                    if i < args.len() {
                        a.duration = args[i].parse().unwrap_or(0.0);
                    }
                }
                "--steps" => {
                    i += 1;
                    if i < args.len() {
                        a.steps = args[i].parse().unwrap_or(12);
                        a.steps_set = true;
                    }
                }
                "--guidance" => {
                    i += 1;
                    if i < args.len() {
                        a.guidance = args[i].parse().unwrap_or(2.0);
                    }
                }
                "--t5" => {
                    i += 1;
                    if i < args.len() {
                        a.t5 = Some(args[i].clone());
                    }
                }
                "--t5-tokenizer" => {
                    i += 1;
                    if i < args.len() {
                        a.t5_tokenizer = Some(args[i].clone());
                    }
                }
                "--clip" => {
                    i += 1;
                    if i < args.len() {
                        a.clip = Some(args[i].clone());
                    }
                }
                "--clip-tokenizer" => {
                    i += 1;
                    if i < args.len() {
                        a.clip_tokenizer = Some(args[i].clone());
                    }
                }
                "--vae" => {
                    i += 1;
                    if i < args.len() {
                        a.vae = Some(args[i].clone());
                    }
                }
                "--width" => {
                    i += 1;
                    if i < args.len() {
                        a.width = args[i].parse().unwrap_or(a.width);
                    }
                }
                "--height" => {
                    i += 1;
                    if i < args.len() {
                        a.height = args[i].parse().unwrap_or(a.height);
                    }
                }
                "--tile" => {
                    i += 1;
                    if i < args.len() {
                        a.vae_tile = args[i].parse().unwrap_or(a.vae_tile);
                    }
                }
                "--batch" => {
                    // A missing or unparseable count means one image, and so
                    // does zero.
                    i += 1;
                    let n: usize = if i < args.len() { args[i].parse().unwrap_or(1) } else { 1 };
                    a.batch = n.max(1);
                }
                "--cpu" => {
                    a.force_cpu = true;
                }
                "--rope-interleaved" => {
                    a.rope_interleaved = true;
                }
                "--draft-len" => {
                    i += 1;
                    if i < args.len() {
                        a.draft_len = args[i].parse().unwrap_or(4);
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                other => {
                    eprintln!("Unknown argument: {other}");
                    print_help();
                    std::process::exit(1);
                }
            }
            i += 1;
        }
        a
    }
}

fn print_help() {
    println!("rich-triplet — LLM from scratch in Rust");
    println!();
    println!("USAGE:");
    println!("  rich-triplet --benchmark              Run scalar-vs-tensor autograd benchmark");
    println!("  rich-triplet --prompt TEXT            Train on corpus, then generate");
    println!("  rich-triplet --prompt TEXT \\");
    println!("               --weights DIR \\");
    println!("               --vocab vocab.json \\");
    println!("               --merges merges.txt      Load GPT-OSS weights, generate");
    println!("  rich-triplet --prompt TEXT \\");
    println!("               --weights DIR \\");
    println!("               --tokenizer-model tokenizer.model \\");
    println!("               --model gemma3-1b           Load Gemma 3 weights, generate");
    println!("  rich-triplet --pretokenize SRC DST    Tokenize SRC text file → DST .bin");
    println!();
    println!("OPTIONS:");
    println!("  --prompt TEXT            Prompt text to complete");
    println!("  --weights DIR            Directory with .safetensors shards");
    println!("  --vocab PATH             BPE vocab.json      (GPT-OSS)");
    println!("  --merges PATH            BPE merges.txt      (GPT-OSS)");
    println!("  --tokenizer-model PATH   SentencePiece .model (Gemma 3)");
    println!(
        "  --tokenizer-dir DIR      Dir with tokenizer.json (for GGUF, where tokenizer is separate)"
    );
    println!("  --model NAME             Architecture: gpt-oss | gemma3-1b | gemma3-4b | qwen35-0.8b |");
    println!("                           qwen35-4b | qwen35-9b | orpheus-3b | omnivoice");
    println!("  --max-new N              Tokens to generate          [default: 200]");
    println!("  --temp T                 Sampling temperature        [default: 0.8]");
    println!("  --top-k K                Top-K cutoff (0=disabled)   [default: 40]");
    println!("  --top-p P                Nucleus probability         [default: 0.95]");
    println!("  --rep-penalty R          Repetition penalty          [default: 1.1]");
    println!("  --seed S                 RNG seed                    [default: 42]");
    println!("  --train-steps N          Training steps (no-weights) [default: 200]");
    println!("  --checkpoint PATH        Load saved .ckpt instead of training");
    println!("  --benchmark              Run scalar-vs-tensor autograd benchmark");
    println!("  --debug                  Enable per-step diagnostic logging");
    println!("\nText to speech (--model orpheus-3b):");
    println!("  --voice NAME             en: tara leah jess leo dan mia zac zoe [default: tara]");
    println!("                           es: javi sergio maria   it: pietro giulia carlo");
    println!("                           (which work depends on the checkpoint loaded)");
    println!("  --snac PATH              SNAC 24 kHz checkpoint (pytorch_model.bin)");
    println!("  --out PATH               Output WAV                  [default: out.wav]");
    println!("  --no-audio-mask          Allow sampling outside the audio token range");
    println!("  --no-leading-bos         Drop the leading BOS token from the prompt");
    println!("\nFLUX text to image (--model flux-schnell, --model flux-dev):");
    println!("  --weights PATH           Transformer GGUF (Q4_K_M recommended)");
    println!("  --t5 PATH                T5-XXL encoder GGUF");
    println!("  --t5-tokenizer PATH      spiece.model (optional; the GGUF carries one)");
    println!("  --clip PATH              CLIP-L text encoder safetensors");
    println!("  --clip-tokenizer PATH    CLIP's tokenizer.json");
    println!("  --vae PATH               Autoencoder safetensors (ae.safetensors)");
    println!("  --width N --height N     Image size; multiples of 16 (default 1024)");
    println!("  --steps N                Denoising steps (schnell 4, dev 28)");
    println!("  --seed N                 Noise seed");
    println!("  --tile N                 VAE tile in latent pixels; 0 decodes whole");
    println!("  --batch N                Generate N images from consecutive seeds");
    println!("  --cpu                    Run the transformer on the CPU");
    println!("  --out PATH               Where to write the PNG (default out.png)");
    println!("\nOmniVoice (--model omnivoice):");
    println!("  --language NAME          Language hint, e.g. Italian    [default: None]");
    println!("  --instruct TEXT          Voice description              [default: None]");
    println!("  --chunk-seconds S        Audio per chunk for long text  [default: 15]");
    println!("  --chunk-threshold S      Split above this many seconds   [default: 30]");
    println!("  --chunk-gap S            Pause between chunks           [default: 0.3]");
    println!("  --ref-audio PATH         WAV of a voice to clone");
    println!("  --ref-text TEXT          What that WAV says (required with it)");
    println!("  --duration S             Audio seconds (0 = estimate)   [default: 0]");
    println!("  --steps N                Unmasking steps                [default: 12]");
    println!("  --guidance G             Guidance scale (0 = off)       [default: 2.0]");
    println!("  --rope-interleaved       Use interleaved RoPE pairing (debugging; the");
    println!("                           default half-split is the correct one)");
    println!("  --weights PATH           omnivoice-base GGUF");
    println!("  --snac PATH              omnivoice-tokenizer GGUF");
    println!();
    println!("  --pretokenize S D        Tokenize text file S, write binary D.bin");
    println!("                           Uses char tokenizer built from S.");
    println!("                           For BPE: also pass --vocab and --merges.");
}

/// Print `message` to stderr and exit with status 1.
fn die(message: &str) -> ! {
    eprintln!("{}", message);
    std::process::exit(1);
}

// =============================================================================
// Generation mode — GPT-OSS with loaded weights
// =============================================================================

fn run_gpt_oss(args: &CliArgs, prompt: &str) {
    use std::io::Write;

    let weights_dir = args.weights.as_deref().unwrap();
    let vocab_path = args
        .vocab
        .as_deref()
        .expect("--vocab required with --weights (path to vocab.json)");
    let merges_path = args
        .merges
        .as_deref()
        .expect("--merges required with --weights (path to merges.txt)");

    eprintln!("[ GPT-OSS ] Loading tokenizer...");
    let tok =
        BpeTokenizer::from_files(vocab_path, merges_path).expect("failed to load BPE tokenizer");

    eprintln!("[ GPT-OSS ] Building model (gpt-oss-20b config)...");
    let config = Config3::gpt_oss_20b();
    let mut rng = InitRng::new(0);
    let mut model = GptOssModel::new(config, &mut rng);

    eprintln!("[ GPT-OSS ] Loading weights from {}...", weights_dir);
    model
        .load_weights_from_dir(weights_dir)
        .expect("failed to load weights");

    let raw_ids = tok.encode(prompt);
    if raw_ids.is_empty() {
        eprintln!("Error: prompt encodes to zero tokens");
        std::process::exit(1);
    }
    let token_ids: Vec<usize> = raw_ids.iter().map(|&id| id as usize).collect();

    let params = SamplingParams {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
        ..SamplingParams::creative(args.seed)
    };

    // Print prompt first, then stream tokens
    print!("{}", prompt);
    std::io::stdout().flush().ok();

    model.generate_with_params_streaming(&token_ids, args.max_new, &params, |tok_id| {
        let text = tok.decode(&[tok_id as u32]);
        print!("{}", text);
        std::io::stdout().flush().ok();
    });
    println!();
}

// =============================================================================
// Generation mode — OmniVoice masked-diffusion text to speech
// =============================================================================

fn run_omnivoice(args: &CliArgs, prompt: &str) {
    use gguf_loader::GgufFile;
    use omnivoice::{OmniRequest, omni_prepare_reference, omni_synthesize};
    use omnivoice_codec::{OmniCodecConfig, OmniCodecDecoder, OmniCodecEncoder};
    use transformer5::{RopePairing, load_gguf_tokenizer};
    use transformer6::{Config6, OmniForward, OmniLm};
    use wav::{read_wav, wave_stats, write_wav};

    let lm_path = args.weights.as_deref().unwrap_or("models/omnivoice-base-Q8_0.gguf");
    let codec_path = args.snac.as_deref().unwrap_or("models/omnivoice-tokenizer-Q8_0.gguf");
    let out_path = args.out.as_deref().unwrap_or("out.wav");

    for p in [lm_path, codec_path] {
        if !std::path::Path::new(p).exists() {
            die(&format!(
                concat!(
                    "OmniVoice weights not found at {}\n",
                    "       Fetch both halves with:\n",
                    "         curl -L -o models/omnivoice-base-Q8_0.gguf \\\n",
                    "           https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/",
                    "omnivoice-base-Q8_0.gguf\n",
                    "         curl -L -o models/omnivoice-tokenizer-Q8_0.gguf \\\n",
                    "           https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/",
                    "omnivoice-tokenizer-Q8_0.gguf"
                ),
                p
            ));
        }
    }

    // The LM's GGUF carries its own byte-level vocabulary, so no separate
    // tokenizer file is needed.
    let gguf = GgufFile::open(lm_path)
        .unwrap_or_else(|e| die(&format!("failed to open GGUF: {}", e)));
    let tok = load_gguf_tokenizer(&gguf)
        .unwrap_or_else(|e| die(&format!("failed to build the tokenizer from GGUF: {}", e)));

    let mut cfg = Config6::omnivoice();
    cfg.rope_pairing = if args.rope_interleaved {
        RopePairing::Interleaved
    } else {
        RopePairing::HalfSplit
    };

    eprintln!("[ OmniVoice ] Loading the language model from {}...", lm_path);
    let mut lm = OmniLm::load(lm_path, cfg)
        .unwrap_or_else(|e| die(&format!("failed to load the language model: {}", e)));
    if args.quantize {
        let n = lm.quantize_projections_to_q4k();
        eprintln!(
            "[ OmniVoice ] Requantized {} projections -> {:.2} GB",
            n,
            lm.weight_bytes() as f64 / 1e9
        );
    }
    crate::transformer4::release_memory_to_os();
    crate::transformer4::print_rss("after weight load");

    eprintln!("[ OmniVoice ] Loading the codec from {}...", codec_path);
    let codec = OmniCodecDecoder::load(codec_path, OmniCodecConfig::defaults())
        .unwrap_or_else(|e| die(&format!("failed to load the codec: {}", e)));
    eprintln!("[ OmniVoice ] Codec decoder: {} parameters", codec.parameter_count());

    let mut request = OmniRequest {
        text: prompt.to_string(),
        language: args.language.clone().unwrap_or_default(),
        instruct: args.instruct.clone().unwrap_or_default(),
        duration_seconds: args.duration,
        ..OmniRequest::default()
    };

    // Voice cloning: read the reference, encode it, and hand the codes over as
    // decided positions. The analysis half of the codec is only loaded when
    // there is something to analyse -- it is bigger than the synthesis half.
    if let Some(ref ref_audio) = args.ref_audio {
        let ref_text = match args.ref_text.as_deref() {
            Some(text) if !text.is_empty() => text,
            _ => die(
                "--ref-audio needs --ref-text: the model has to know which part of the \
                 prompt it has already heard",
            ),
        };
        let wav = read_wav(ref_audio)
            .unwrap_or_else(|e| die(&format!("failed to read the reference audio: {}", e)));
        let reference = omni_prepare_reference(&wav.mono(), wav.sample_rate, &codec.config)
            .unwrap_or_else(|e| die(&format!("failed to prepare the reference audio: {}", e)));
        eprintln!(
            "[ OmniVoice ] Reference: {:.2} s, {} Hz, {} ch, rms {:.4}",
            reference.seconds(&codec.config),
            wav.sample_rate,
            wav.channels,
            reference.rms
        );
        if reference.seconds(&codec.config) > 20.0 {
            eprintln!(
                "[ OmniVoice ] Warning: reference clips over 20 s slow generation down and \
                 clone no better; 3-10 s is the useful range"
            );
        }

        eprintln!("[ OmniVoice ] Loading the codec encoder from {}...", codec_path);
        let encoder = OmniCodecEncoder::load(codec_path, OmniCodecConfig::defaults())
            .unwrap_or_else(|e| die(&format!("failed to load the codec encoder: {}", e)));
        let codes = encoder
            .encode(&reference.samples)
            .unwrap_or_else(|e| die(&format!("failed to encode the reference audio: {}", e)));
        eprintln!("[ OmniVoice ] Encoded the reference to {} frames", codes[0].len());
        request.ref_codes = codes;
        request.ref_text = ref_text.to_string();
        request.ref_rms = reference.rms;
    }
    request.debug = args.debug;
    request.generation.num_step = args.steps;
    request.generation.guidance_scale = args.guidance;
    request.generation.seed = args.seed;
    request.generation.chunk_seconds = args.chunk_seconds;
    request.generation.chunk_threshold_seconds = args.chunk_threshold;
    request.generation.chunk_gap_seconds = args.chunk_gap;

    let duration_field = if request.duration_seconds > 0.0 {
        format!("{:.1}s", request.duration_seconds)
    } else {
        "estimated".to_string()
    };
    eprintln!(
        "[ OmniVoice ] lang={} steps={} guidance={:.2} duration={} rope={} clone={}",
        if request.language.is_empty() { "None" } else { &request.language },
        request.generation.num_step,
        request.generation.guidance_scale,
        duration_field,
        if args.rope_interleaved { "interleaved" } else { "half-split" },
        if request.ref_codes.is_empty() { "off" } else { "on" }
    );
    eprintln!("[ OmniVoice ] Synthesising: \"{}\"", prompt);

    #[cfg(not(feature = "metal"))]
    let accel: Option<&dyn OmniForward> = None;
    // The whole forward pass in one command buffer. Unlike the decode engines,
    // this exists for the GEMMs rather than despite them: a diffusion step is a
    // full-sequence pass over a few hundred positions, which is the shape the
    // matrix units are for.
    #[cfg(feature = "metal")]
    let metal_ctx = {
        use metal_omnivoice::MetalOmniContext;
        eprintln!("[ Metal ] Uploading weights...");
        let t_upload = std::time::Instant::now();
        match MetalOmniContext::create(&mut lm, MetalOmniContext::MAX_TOKENS) {
            Some(ctx) => {
                eprintln!(
                    "[ Metal ] {:.2} GB in GPU buffers, uploaded in {:.1} s",
                    ctx.buffer_bytes() as f64 / 1e9,
                    t_upload.elapsed().as_secs_f64()
                );
                crate::transformer4::release_memory_to_os();
                crate::transformer4::print_rss("after the GPU upload");
                ctx
            }
            // create() frees the CPU weights as it goes, so a partial failure
            // leaves nothing to fall back to.
            None => die("the Metal engine failed to initialise"),
        }
    };
    #[cfg(feature = "metal")]
    let accel: Option<&dyn OmniForward> = Some(&metal_ctx);

    let result = omni_synthesize(&lm, &codec, &tok, &request, accel)
        .unwrap_or_else(|e| die(&format!("synthesis failed: {}", e)));

    let stats = wave_stats(&result.samples);
    if result.chunks > 1 {
        eprintln!(
            "[ OmniVoice ] Split into {} chunks; each after the first takes the first as \
             its reference",
            result.chunks
        );
    }
    eprintln!(
        "[ OmniVoice ] {} frames, {} prompt tokens, {} forward passes -> {} samples",
        result.frames,
        result.prompt_tokens,
        result.forward_passes,
        result.samples.len()
    );
    eprintln!(
        "[ OmniVoice ] {:.2} s audio in {:.2} s generate + {:.2} s decode (RTF {:.2})",
        result.audio_seconds(),
        result.generate_seconds,
        result.decode_seconds,
        result.realtime_factor()
    );
    eprintln!("[ OmniVoice ] waveform: {}", stats.describe());
    if !stats.looks_like_speech() {
        eprintln!("[ OmniVoice ] Warning: the waveform statistics do not look like speech");
    }

    if let Err(e) = write_wav(out_path, &result.samples, result.sample_rate, 1) {
        die(&format!("failed to write the WAV: {}", e));
    }
    eprintln!("[ OmniVoice ] Wrote {}", out_path);
}

// =============================================================================
// FLUX — text to image
// =============================================================================

/// Encode a prompt with CLIP-L, returning the pooled vector FLUX conditions on.
///
/// CLIP wants the sequence bracketed by BOS and EOT and padded to 77 with more
/// EOT. The padding matters: the pooled vector is read at the *first* EOT, so
/// padding with anything else would work equally well here -- but the model was
/// trained with EOT padding, and the per-token states the pooling reads through
/// depend on it.
fn encode_clip(weights: &str, tokenizer_path: &str, prompt: &str) -> Result<Vec<f32>, String> {
    use clip_text::{ClipTextConfig, ClipTextEncoder};
    use tokenizer::HfBpeTokenizer;

    let tok = HfBpeTokenizer::from_json_file(tokenizer_path)?;
    let cfg = ClipTextConfig::large();

    const BOS: u32 = 49406;
    let eot = cfg.eot_token_id;
    let max_positions = cfg.max_position_embeddings;
    let mut ids: Vec<u32> = vec![BOS];
    for id in tok.encode(prompt) {
        if ids.len() + 1 >= max_positions {
            break; // leave room for the EOT
        }
        ids.push(id);
    }
    ids.push(eot);
    ids.resize(max_positions, eot);

    let model = ClipTextEncoder::load(weights, cfg)?;
    let out = model.forward(&ids)?;
    Ok(out.pooled)
}

/// Encode a prompt with T5-XXL, returning the `[seq_len, 4096]` sequence.
///
/// schnell truncates or pads to 256. T5 appends `</s>` and has no BOS, and the
/// padding is id 0 -- the model was trained with an attention mask that hides
/// it, which this implementation does not have, so the padded positions do
/// contribute. They contribute what the reference implementation's unmasked
/// path would contribute, which is what matters for matching it.
///
/// `tokenizer_path` is optional: the distributed encoder GGUFs carry their own
/// unigram vocabulary, which is both one fewer download and one fewer way to
/// pair a checkpoint with the wrong tokenizer.
fn encode_t5(
    weights: &str,
    tokenizer_path: Option<&str>,
    prompt: &str,
    seq_len: usize,
) -> Result<autograd2::Mat, String> {
    use gguf_loader::GgufFile;
    use t5::{T5Config, T5Encoder, load_t5_gguf_tokenizer};
    use tokenizer::SentencePieceTokenizer;

    let gguf = GgufFile::open(weights).map_err(|e| e.to_string())?;
    let tok = match tokenizer_path {
        Some(path) => SentencePieceTokenizer::from_model_file(path),
        None => load_t5_gguf_tokenizer(&gguf),
    }
    .map_err(|e| format!("t5 tokenizer: {}", e))?;

    const EOS: u32 = 1;
    const PAD: u32 = 0;
    let mut ids = tok.encode(prompt);
    if ids.len() + 1 > seq_len {
        ids.truncate(seq_len - 1);
    }
    ids.push(EOS);
    ids.resize(seq_len, PAD);

    let mut model = T5Encoder::load_gguf(weights, T5Config::xxl())?;
    eprintln!(
        "[ FLUX ] T5-XXL: {:.2} B parameters",
        model.parameter_count() as f64 / 1e9
    );
    let seq = model.forward(&ids)?;
    model.free_weights();
    Ok(seq)
}

fn run_flux(args: &CliArgs, prompt: &str) {
    use flux::{FluxConfig, FluxModel, FluxSampleParams, flux_sample};
    use png::{image_stats, write_png};
    use vae::{VaeConfig, VaeDecoder};

    let Some(weights_path) = args.weights.as_deref() else {
        die("--model flux-schnell needs --weights pointing at the transformer GGUF");
    };
    let require = |v: &Option<String>, flag: &str| -> String {
        match v {
            Some(s) => s.clone(),
            None => die(&format!("--model flux-schnell needs {}", flag)),
        }
    };
    let t5_path = require(&args.t5, "--t5");
    let clip_path = require(&args.clip, "--clip");
    let clip_tok = require(&args.clip_tokenizer, "--clip-tokenizer");
    let vae_path = require(&args.vae, "--vae");
    let out_path = args.out.as_deref().unwrap_or("out.png");

    let is_dev = args.model.as_deref().is_some_and(|m| m.contains("dev"));
    let cfg = if is_dev { FluxConfig::dev() } else { FluxConfig::schnell() };
    // schnell is distilled to four steps; dev wants nearer thirty.
    let steps = if args.steps_set {
        args.steps
    } else if is_dev {
        28
    } else {
        4
    };

    const VAE_FACTOR: usize = 8;
    let align = VAE_FACTOR * cfg.patch_size;
    if args.width % align != 0 || args.height % align != 0 {
        die(&format!("--width and --height must be multiples of {}", align));
    }

    // Text first, and both encoders are released before the transformer is
    // loaded. T5-XXL is 2.8 GB at Q4_K and the transformer is 6.7 GB; holding
    // both at once is the difference between fitting in 18 GB and not.
    eprintln!("[ FLUX ] Encoding the prompt with CLIP-L...");
    let pooled = encode_clip(&clip_path, &clip_tok, prompt)
        .unwrap_or_else(|e| die(&format!("CLIP encoding failed: {}", e)));

    eprintln!("[ FLUX ] Encoding the prompt with T5-XXL...");
    let context = encode_t5(&t5_path, args.t5_tokenizer.as_deref(), prompt, 256)
        .unwrap_or_else(|e| die(&format!("T5 encoding failed: {}", e)));

    eprintln!("[ FLUX ] Loading the transformer from {}...", weights_path);
    let mut model = FluxModel::load_gguf(weights_path, cfg.clone())
        .unwrap_or_else(|e| die(&format!("failed to load the transformer: {}", e)));
    eprintln!(
        "[ FLUX ] {:.2} B parameters, {:.2} GB of weights",
        model.parameter_count() as f64 / 1e9,
        model.weight_bytes() as f64 / 1e9
    );

    let mut params = FluxSampleParams {
        width: args.width,
        height: args.height,
        steps,
        guidance: args.guidance,
        // schnell's scheduler does not shift; dev's does, as a function of the
        // sequence length. 1.0 is the identity either way for the four-step path.
        shift: if is_dev { 1.15 } else { 1.0 },
        ..FluxSampleParams::default()
    };

    let started = std::time::Instant::now();

    // Sample every seed before decoding any of them. The transformer is 7 GB
    // and the autoencoder wants the memory, so the two cannot be interleaved
    // without reloading one of them per image -- and a latent is 1 MB, so
    // holding the whole batch costs nothing.
    let mut latents: Vec<autograd2::Mat> = Vec::with_capacity(args.batch);
    {
        // The GPU engine takes the transformer's weights over as it uploads
        // them, and lives until the batch is sampled.
        #[cfg(feature = "metal")]
        let engine = if !args.force_cpu {
            let built = metal_flux::MetalFluxContext::create(
                &mut model,
                args.height / VAE_FACTOR,
                args.width / VAE_FACTOR,
                context.rows,
            )
            .unwrap_or_else(|e| die(&format!("failed to build the Metal engine: {}", e)));
            eprintln!(
                "[ FLUX ] Metal: {:.2} GB on the GPU",
                built.device_bytes() as f64 / 1e9
            );
            Some(built)
        } else {
            None
        };
        for i in 0..args.batch {
            params.seed = args.seed.wrapping_add(i as u64);
            let progress = if args.batch > 1 {
                format!(" [{}/{}]", i + 1, args.batch)
            } else {
                String::new()
            };
            eprintln!(
                "[ FLUX ] Sampling {}x{} in {} steps (seed {}){}...",
                args.width, args.height, steps, params.seed, progress
            );
            let one = std::time::Instant::now();
            #[cfg(feature = "metal")]
            let latent = match &engine {
                Some(engine) => {
                    metal_flux::flux_sample_metal(engine, &cfg, &context, &pooled, &params)
                }
                None => flux_sample(&model, &context, &pooled, &params),
            };
            #[cfg(not(feature = "metal"))]
            let latent = flux_sample(&model, &context, &pooled, &params);
            let latent = latent.unwrap_or_else(|e| die(&format!("sampling failed: {}", e)));
            eprintln!("[ FLUX ] Sampled in {:.1} s", one.elapsed().as_secs_f64());
            latents.push(latent);
        }
    }

    // The transformer is finished, and it is holding 7 GB the autoencoder would
    // rather have. Releasing it here is what lets the decode run whole-image.
    model.free_weights();

    eprintln!(
        "[ FLUX ] Decoding {} latent{}...",
        latents.len(),
        if latents.len() == 1 { "" } else { "s" }
    );
    let vae = VaeDecoder::load(&vae_path, VaeConfig::flux())
        .unwrap_or_else(|e| die(&format!("failed to load the autoencoder: {}", e)));

    let lat_h = args.height / VAE_FACTOR;
    let lat_w = args.width / VAE_FACTOR;
    // The last decoder level runs 128 channels at full resolution, which is
    // half a gigabyte per activation at 1024x1024, and a residual unit holds
    // three. Tiling caps that -- but it also decodes the overlaps twice, and
    // now that the mid-block attention is a pair of gemms rather than a triple
    // loop the whole-image path is the faster of the two at 1024x1024 (27 s
    // against 33 s). So the threshold sits above it: tile only when the latent
    // is larger than 128x128, where whole-image would want ~10 GB.
    let tile = if args.vae_tile != 0 {
        args.vae_tile
    } else if lat_h * lat_w > 128 * 128 {
        64
    } else {
        0
    };
    for (i, latent) in latents.iter().enumerate() {
        let pixels = if tile != 0 {
            vae.decode_tiled(latent, lat_h, lat_w, tile, tile / 4)
        } else {
            vae.decode(latent, lat_h, lat_w)
        }
        .unwrap_or_else(|e| die(&format!("decoding failed: {}", e)));

        let stats = image_stats(&pixels, args.width, args.height);
        eprintln!("[ FLUX ] {}", stats.describe());

        // A single image keeps the name it was given; a batch gets the seed
        // spliced in before the extension, so the files stay distinguishable
        // and say which seed produced them.
        let path = if latents.len() > 1 {
            let seed = format!("_s{}", args.seed.wrapping_add(i as u64));
            match out_path.rfind('.') {
                None => format!("{}{}", out_path, seed),
                Some(dot) => format!("{}{}{}", &out_path[..dot], seed, &out_path[dot..]),
            }
        } else {
            out_path.to_string()
        };
        if let Err(e) = write_png(&path, &pixels, args.width, args.height) {
            die(&format!("failed to write the image: {}", e));
        }
        eprintln!("[ FLUX ] Wrote {}", path);
    }
    eprintln!(
        "[ FLUX ] {} image{} in {:.1} s total",
        latents.len(),
        if latents.len() == 1 { "" } else { "s" },
        started.elapsed().as_secs_f64()
    );
}

// =============================================================================
// Generation mode — Orpheus text to speech
// =============================================================================

fn run_orpheus(args: &CliArgs, prompt: &str) {
    use gguf_loader::GgufFile;
    use orpheus::{OrpheusConfig, OrpheusRequest, orpheus_default_sampling, orpheus_synthesize};
    use snac::{SnacConfig, SnacDecoder};
    use transformer5::{Config5, LlamaModel, load_gguf_tokenizer};
    use wav::{wave_stats, write_wav};

    let weights_path = args.weights.as_deref().unwrap();
    let snac_path = args.snac.as_deref().unwrap_or("models/snac_24khz.bin");
    let out_path = args.out.as_deref().unwrap_or("out.wav");
    let voice = args.voice.as_deref().unwrap_or("tara");

    if !std::path::Path::new(snac_path).exists() {
        die(&format!(
            "SNAC codec weights not found at {}\n       Fetch them with:\n         curl -L -o {} \
             https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/pytorch_model.bin\n       \
             or point --snac at an existing copy.",
            snac_path, snac_path
        ));
    }

    // The GGUF carries its own vocabulary and merges, so no --tokenizer-dir is
    // needed -- which matters here, because the Orpheus repository is gated.
    eprintln!("[ Orpheus ] Reading {}...", weights_path);
    let gguf = GgufFile::open(weights_path)
        .unwrap_or_else(|e| die(&format!("failed to open GGUF: {}", e)));
    let tok = load_gguf_tokenizer(&gguf)
        .unwrap_or_else(|e| die(&format!("failed to build the tokenizer from GGUF: {}", e)));

    let mut model = LlamaModel::new_for_inference(Config5::orpheus_3b());
    if let Err(e) = model.load_weights_from_gguf(weights_path) {
        die(&format!("failed to load weights: {}", e));
    }

    // How the file was quantized decides what to do next.
    //
    // A Q4_K_M checkpoint keeps most projections native and lifts only
    // attn_v, ffn_down and output to Q6_K -- those are the quality-sensitive
    // ones, chosen deliberately. Flattening them would throw that away, so
    // only the lm_head is requantized: it is the largest single read per
    // token, and at 156 940 entries it costs 964 MB as BF16 against 271 MB as
    // Q4_K.
    //
    // A uniformly higher-precision file -- Q8_0, F16 -- has no Q4_K tensors at
    // all, so every projection widens to BF16 and the model lands near 6.6 GB
    // with roughly 3.5x the per-token memory traffic. There is no deliberate
    // choice to preserve there, so requantizing all of them is the right call.
    let projections = model.projection_count();
    let widened = model.bf16_projection_count();
    if widened * 2 > projections {
        eprintln!(
            "[ Orpheus ] {} of {} projections were widened to BF16 ({:.2} GB); \
             requantizing to Q4_K...",
            widened,
            projections,
            model.weight_bytes() as f64 / 1e9
        );
        let converted = model.quantize_projections_to_q4k();
        eprintln!(
            "[ Orpheus ] Requantized {} projections, now {:.2} GB",
            converted,
            model.weight_bytes() as f64 / 1e9
        );
    } else {
        eprintln!("[ Orpheus ] Quantizing lm_head to Q4_K...");
        model.quantize_lm_head();
    }
    crate::transformer4::release_memory_to_os();
    crate::transformer4::print_rss("after weight load");

    eprintln!("[ Orpheus ] Loading SNAC codec from {}...", snac_path);
    let snac = SnacDecoder::load(snac_path, SnacConfig::snac_24khz())
        .unwrap_or_else(|e| die(&format!("failed to load the SNAC codec: {}", e)));
    eprintln!("[ Orpheus ] SNAC decoder: {} parameters", snac.parameter_count());

    let mut request = OrpheusRequest {
        text: prompt.to_string(),
        voice: voice.to_string(),
        max_new: args.max_new,
        debug: args.debug,
        mask_to_audio: !args.no_audio_mask,
        sampling: orpheus_default_sampling(args.seed),
        ..OrpheusRequest::default()
    };
    // Explicit flags win over the reference defaults.
    request.sampling.temperature = args.temperature;
    request.sampling.top_p = args.top_p;
    request.sampling.top_k = args.top_k;
    request.sampling.repetition_penalty = args.rep_penalty;

    eprintln!(
        "[ Orpheus ] voice={} max_new={} temp={:.2} top_p={:.2} rep={:.2}",
        voice,
        request.max_new,
        request.sampling.temperature,
        request.sampling.top_p,
        request.sampling.repetition_penalty
    );
    eprintln!("[ Orpheus ] Synthesising: \"{}\"", prompt);

    let cfg = OrpheusConfig {
        leading_bos: !args.no_leading_bos,
        ..OrpheusConfig::defaults()
    };
    let result = orpheus_synthesize(&model, &snac, &tok, &request, &cfg)
        .unwrap_or_else(|e| die(&format!("synthesis failed: {}", e)));

    let stats = wave_stats(&result.samples);
    eprintln!(
        "[ Orpheus ] {} tokens -> {} codes ({} rejected) -> {} groups -> {} samples",
        result.tokens_generated,
        result.codes_accepted,
        result.codes_rejected,
        result.groups,
        result.samples.len()
    );
    eprintln!(
        "[ Orpheus ] {:.2} s audio in {:.2} s generate + {:.2} s decode (RTF {:.2})",
        result.audio_seconds(),
        result.generate_seconds,
        result.decode_seconds,
        result.realtime_factor()
    );
    eprintln!("[ Orpheus ] waveform: {}", stats.describe());
    if !stats.looks_like_speech() {
        // Not fatal -- a short or quiet clip fails this legitimately -- but a
        // pipeline fault shows up here first, and every fault in this pipeline
        // sounds the same.
        eprintln!("[ Orpheus ] Warning: the waveform statistics do not look like speech");
    }

    if let Err(e) = write_wav(out_path, &result.samples, result.sample_rate, 1) {
        die(&format!("failed to write the WAV: {}", e));
    }
    eprintln!("[ Orpheus ] Wrote {}", out_path);
}

// =============================================================================
// Generation mode — Gemma 3 with loaded weights
// =============================================================================

fn run_gemma3(args: &CliArgs, prompt: &str) {
    use std::io::Write;
    use tokenizer::HfBpeTokenizer;
    use transformer4::{Config4, Gemma3Model};

    let weights_path = args.weights.as_deref().unwrap();
    let model_name = args.model.as_deref().unwrap_or("gemma3-1b");

    // Determine if --weights points to a GGUF file or a directory of safetensors.
    // Accept both ".gguf" extension and extensionless files (e.g. Ollama blob paths).
    let is_gguf = weights_path.ends_with(".gguf")
        || (std::fs::metadata(weights_path).map(|m| m.is_file()).unwrap_or(false)
            && {
                // Confirm GGUF magic: first 4 bytes == b"GGUF"
                std::fs::File::open(weights_path).ok().and_then(|mut f| {
                    use std::io::Read;
                    let mut magic = [0u8; 4];
                    f.read_exact(&mut magic).ok().map(|_| magic == *b"GGUF")
                }).unwrap_or(false)
            });

    // The tokenizer lives next to the weights (directory for safetensors).
    // For GGUF files, the tokenizer is NOT included — pass --tokenizer-dir pointing
    // to the original safetensors directory (e.g. google/gemma-3-4b-it).
    let weights_dir = if is_gguf {
        // Parent directory of the .gguf file (used as fallback for cache only)
        std::path::Path::new(weights_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    } else {
        weights_path.trim_end_matches('/').to_string()
    };

    // Tokenizer directory: explicit --tokenizer-dir overrides, then weights_dir
    let tok_dir = args
        .tokenizer_dir
        .as_deref()
        .unwrap_or(&weights_dir)
        .trim_end_matches('/');

    // Try tokenizer.json (BPE) first, then fall back to tokenizer.model (SentencePiece)
    let tok_json_path = format!("{}/tokenizer.json", tok_dir);
    eprintln!("[ Gemma3 ] Loading tokenizer from {}...", tok_json_path);
    let tok = HfBpeTokenizer::from_json_file(&tok_json_path).unwrap_or_else(|e| {
        eprintln!(
            "Error: failed to load tokenizer from {}: {}",
            tok_json_path, e
        );
        if is_gguf && args.tokenizer_dir.is_none() {
            eprintln!("Hint: GGUF files do not include a tokenizer.");
            eprintln!("      Pass --tokenizer-dir pointing to your safetensors directory,");
            eprintln!("      e.g.: --tokenizer-dir /path/to/gemma-3-4b-it/");
        }
        std::process::exit(1);
    });
    eprintln!("[ Gemma3 ] Vocab size: {}", tok.vocab_size());

    let config = match model_name {
        "gemma3-4b" => Config4::gemma3_4b(),
        _ => Config4::gemma3_1b(), // default
    };
    eprintln!(
        "[ Gemma3 ] Building {} model ({} layers, hidden={})...",
        model_name, config.num_hidden_layers, config.hidden_size
    );

    let mut model = Gemma3Model::new_for_inference(config);

    if is_gguf {
        // --- GGUF path: load directly from .gguf file ---
        eprintln!("[ Gemma3 ] Loading weights from GGUF: {}...", weights_path);
        model
            .load_weights_from_gguf(weights_path)
            .expect("failed to load GGUF weights");
        crate::transformer4::release_memory_to_os();
    } else {
        // --- Safetensors path: try cache first, then load + save cache ---
        let cache_path = format!("{}/gemma3-{}.cache", weights_dir, model_name);
        let loaded_from_cache = model.load_cache(&cache_path).unwrap_or(false);

        if loaded_from_cache {
            eprintln!("[ Gemma3 ] Loaded weights from cache ({}).", cache_path);
        } else {
            eprintln!("[ Gemma3 ] Loading weights from {}...", weights_dir);
            model
                .load_weights_from_dir(&weights_dir)
                .expect("failed to load weights");
            eprintln!("[ Gemma3 ] Saving weight cache to {}...", cache_path);
            model.save_cache(&cache_path).expect("failed to save cache");
            eprintln!("[ Gemma3 ] Cache saved.");
        }
        crate::transformer4::release_memory_to_os();
    }
    crate::transformer4::print_rss("after weight load");

    if args.quantize {
        eprintln!("[ Gemma3 ] Quantizing weights to Q4...");
        model.quantize_all_weights();
        eprintln!("[ Gemma3 ] Quantization complete.");
    }

    // Gemma 3-IT requires the chat template:
    //   <bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n
    // Special token ids: bos=2, start_of_turn=105, end_of_turn=106, \n=107, user=2364
    // We inject these directly rather than via the tokenizer, which would
    // tokenize the literal text "<start_of_turn>" as subword pieces.
    let mut token_ids: Vec<usize> = vec![2, 105, 2364, 107]; // <bos><start_of_turn>user\n
    token_ids.extend(tok.encode(prompt).iter().map(|&id| id as usize));
    token_ids.extend_from_slice(&[106, 107, 105]); // <end_of_turn>\n<start_of_turn>
    // encode "model\n" — or just use the known ids: model=4368, \n=107
    token_ids.extend_from_slice(&[4368, 107]); // model\n
    if token_ids.is_empty() {
        eprintln!("Error: prompt encodes to zero tokens");
        std::process::exit(1);
    }

    // Diagnostic: show prompt token IDs and their text
    // eprintln!(
    //     "[ Gemma3-dbg ] Prompt token IDs ({} tokens):",
    //     token_ids.len()
    // );
    for (i, &tid) in token_ids.iter().enumerate() {
        let text = tok.decode(&[tid as u32]);
        eprintln!("  [{}] id={} text={:?}", i, tid, text);
    }

    print!("{}", prompt);
    std::io::stdout().flush().ok();

    model.generate_cached_streaming(
        &token_ids,
        args.max_new,
        args.temperature,
        args.top_k,
        args.top_p,
        args.rep_penalty,
        args.seed,
        args.debug,
        args.draft_len,
        |tok_id| {
            let text = tok.decode(&[tok_id as u32]);
            print!("{}", text);
            std::io::stdout().flush().ok();
        },
    );
    println!();
}

// =============================================================================
// Generation mode — Qwen 3.5 with loaded weights
// =============================================================================

fn run_qwen35(args: &CliArgs, prompt: &str) {
    use std::io::Write;
    use tokenizer::HfBpeTokenizer;
    use transformer_qwen35::{ConfigQwen35, Qwen35Model};

    let weights_path = args.weights.as_deref().unwrap();
    let model_name = args.model.as_deref().unwrap_or("qwen35-4b");

    let is_gguf = weights_path.ends_with(".gguf")
        || (std::fs::metadata(weights_path).map(|m| m.is_file()).unwrap_or(false)
            && {
                std::fs::File::open(weights_path).ok().and_then(|mut f| {
                    use std::io::Read;
                    let mut magic = [0u8; 4];
                    f.read_exact(&mut magic).ok().map(|_| magic == *b"GGUF")
                }).unwrap_or(false)
            });

    let weights_dir = if is_gguf {
        std::path::Path::new(weights_path)
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    } else {
        weights_path.trim_end_matches('/').to_string()
    };

    let tok_dir = args
        .tokenizer_dir
        .as_deref()
        .unwrap_or(&weights_dir)
        .trim_end_matches('/');

    let tok_json_path = format!("{}/tokenizer.json", tok_dir);
    eprintln!("[ Qwen3.5 ] Loading tokenizer from {}...", tok_json_path);
    let tok = HfBpeTokenizer::from_json_file(&tok_json_path).unwrap_or_else(|e| {
        eprintln!("Error: failed to load tokenizer from {}: {}", tok_json_path, e);
        if is_gguf && args.tokenizer_dir.is_none() {
            eprintln!("Hint: GGUF files do not include a tokenizer.");
            eprintln!("      Pass --tokenizer-dir pointing to your HF directory.");
        }
        std::process::exit(1);
    });
    eprintln!("[ Qwen3.5 ] Vocab size: {}", tok.vocab_size());

    let config = match model_name {
        "qwen35-9b" => ConfigQwen35::qwen35_9b(),
        "qwen35-0.8b" | "qwen35-0_8b" => ConfigQwen35::qwen35_0_8b(),
        _ => ConfigQwen35::qwen35_4b(),
    };
    eprintln!(
        "[ Qwen3.5 ] Building {} model ({} layers, hidden={})...",
        model_name, config.num_hidden_layers, config.hidden_size
    );

    let mut model = Qwen35Model::new_for_inference(config);

    if is_gguf {
        eprintln!("[ Qwen3.5 ] Loading weights from GGUF: {}...", weights_path);
        model.load_weights_from_gguf(weights_path).expect("failed to load GGUF weights");
        crate::transformer4::release_memory_to_os();
    } else {
        eprintln!("[ Qwen3.5 ] Loading weights from {}...", weights_path);
        model.load_weights_from_dir(weights_path).expect("failed to load safetensors weights");
        crate::transformer4::release_memory_to_os();
    }
    crate::transformer4::print_rss("after weight load");

    // Qwen3.5 chat template (ChatML):
    //   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
    // Special token IDs:
    //   <|im_start|> = 248045, <|im_end|> = 248046, <|endoftext|> = 248044
    let im_start: usize = 248045;
    let im_end: usize = 248046;
    let newline: usize = 198; // \n
    let mut token_ids: Vec<usize> = Vec::new();
    // An optional system turn precedes the user turn. Fine-tuned chat models
    // often bind their behaviour to the exact system prompt seen in training,
    // so leaving it out can change what the model does.
    if let Some(ref system) = args.system {
        token_ids.push(im_start);
        token_ids.extend(tok.encode("system").iter().map(|&id| id as usize));
        token_ids.push(newline);
        token_ids.extend(tok.encode(system).iter().map(|&id| id as usize));
        token_ids.push(im_end);
        token_ids.push(newline);
    }
    token_ids.push(im_start);
    token_ids.extend(tok.encode("user").iter().map(|&id| id as usize));
    token_ids.push(newline);
    token_ids.extend(tok.encode(prompt).iter().map(|&id| id as usize));
    token_ids.push(im_end);
    token_ids.push(newline);
    token_ids.push(im_start);
    token_ids.extend(tok.encode("assistant").iter().map(|&id| id as usize));
    token_ids.push(newline);

    if token_ids.is_empty() {
        eprintln!("Error: prompt encodes to zero tokens");
        std::process::exit(1);
    }

    for (i, &tid) in token_ids.iter().enumerate() {
        let text = tok.decode(&[tid as u32]);
        eprintln!("  [{}] id={} text={:?}", i, tid, text);
    }

    print!("{}", prompt);
    std::io::stdout().flush().ok();

    model.generate_cached_streaming(
        &token_ids,
        args.max_new,
        args.temperature,
        args.top_k,
        args.top_p,
        args.rep_penalty,
        args.seed,
        args.debug,
        |tok_id| {
            let text = tok.decode(&[tok_id as u32]);
            print!("{}", text);
            std::io::stdout().flush().ok();
        },
    );
    println!();
}

// =============================================================================
// Generation mode — trained Gpt2 on built-in corpus
// =============================================================================

fn run_gpt2_generate(args: &CliArgs, prompt: &str) {
    use std::io::Write;

    let tokenizer = CharTokenizer::from_text(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let context_length = 64;

    let (train_data, val_data) = TextDataset::train_val_split(CORPUS, &tokenizer, context_length);

    let model_config = Config {
        vocab_size,
        context_length,
        d_model: 64,
        n_layers: 4,
        n_heads: 4,
    };

    let mut rng = InitRng::new(42);
    let model = Gpt2::new(model_config, &mut rng);

    if let Some(ref ckpt_path) = args.checkpoint {
        eprintln!("[ Load ] Restoring from checkpoint: {}", ckpt_path);
        use nn2::Module2;
        restore_checkpoint(ckpt_path, &model.parameters()).expect("failed to restore checkpoint");
    } else {
        eprintln!(
            "[ Train ] vocab={} context={} steps={}",
            vocab_size, context_length, args.train_steps
        );
        let cfg = TrainConfig2 {
            max_steps: args.train_steps,
            eval_interval: args.train_steps / 5,
            learning_rate: 3e-3,
            grad_clip: 1.0,
            ..TrainConfig2::default()
        };
        train2(&model, &train_data, &val_data, &cfg);
    }

    // Stream tokens using the KV cache (O(T) per step instead of O(T²)).
    let token_ids: Vec<usize> = tokenizer
        .encode(prompt)
        .iter()
        .map(|&x| x as usize)
        .collect();
    let token_ids = if token_ids.is_empty() {
        vec![0usize]
    } else {
        token_ids
    };
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    model.generate_cached_streaming(
        &token_ids,
        args.max_new,
        args.temperature,
        args.top_k,
        |tok_id| {
            print!("{}", tokenizer.decode(&[tok_id as u32]));
            std::io::stdout().flush().ok();
        },
    );
    println!();
}

fn main() {
    let args = CliArgs::parse();

    // -------------------------------------------------------------------------
    // Pretokenize mode: tokenize a text file → binary .bin corpus
    // -------------------------------------------------------------------------
    if let Some((ref src, ref dst)) = args.pretokenize {
        use dataset::TokenizedDataset;
        eprintln!("[pretokenize] Reading: {}", src);
        eprintln!("[pretokenize] Output:  {}", dst);

        let n_tokens = if args.vocab.is_some() && args.merges.is_some() {
            // BPE tokenizer
            let tok = BpeTokenizer::from_files(
                args.vocab.as_deref().unwrap(),
                args.merges.as_deref().unwrap(),
            )
            .expect("failed to load BPE tokenizer");
            TokenizedDataset::write_bin_from_file(dst, src, &tok).expect("pretokenize failed")
        } else {
            // Char tokenizer — read full file to build vocab, then re-tokenize line-by-line
            let text = std::fs::read_to_string(src).expect("cannot read source file");
            let tok = CharTokenizer::from_text(&text);
            eprintln!("[pretokenize] Char vocab size: {}", tok.vocab_size());
            TokenizedDataset::write_bin(dst, &text, &tok).expect("pretokenize failed")
        };

        eprintln!("[pretokenize] Done: {} tokens → {}", n_tokens, dst);
        return;
    }

    // -------------------------------------------------------------------------
    // Generation mode
    // -------------------------------------------------------------------------
    if let Some(ref prompt) = args.prompt.clone() {
        let model_starts_with =
            |prefix: &str| args.model.as_deref().map_or(false, |m| m.starts_with(prefix));
        let is_flux = model_starts_with("flux");
        let is_omnivoice = model_starts_with("omnivoice");
        let is_orpheus = model_starts_with("orpheus");
        let is_qwen35 = model_starts_with("qwen35");
        // A .gguf file with no --model is assumed to be Gemma 3, which is the
        // only architecture this CLI ever loaded from GGUF first.
        let is_gemma3 = args.tokenizer_model.is_some()
            || model_starts_with("gemma3")
            || (!is_qwen35
                && !is_orpheus
                && !is_omnivoice
                && !is_flux
                && args
                    .weights
                    .as_deref()
                    .map_or(false, |w| w.ends_with(".gguf")));
        if is_flux {
            run_flux(&args, prompt);
        } else if is_omnivoice {
            run_omnivoice(&args, prompt);
        } else if is_orpheus {
            if args.weights.is_none() {
                die("--model orpheus-3b needs --weights pointing at the GGUF file");
            }
            run_orpheus(&args, prompt);
        } else if is_qwen35 {
            run_qwen35(&args, prompt);
        } else if is_gemma3 {
            run_gemma3(&args, prompt);
        } else if args.weights.is_some() {
            run_gpt_oss(&args, prompt);
        } else {
            run_gpt2_generate(&args, prompt);
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Benchmark mode
    // -------------------------------------------------------------------------
    if args.benchmark {
        run_benchmark(args.train_steps);
        return;
    }

    print_help();
}

// =============================================================================
// Benchmark mode — scalar autograd vs tensor autodiff
// =============================================================================

fn run_benchmark(train_steps: usize) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Rich Triplet — LLM from scratch in Rust              ║");
    println!("║        Scalar autograd  vs  Tensor autodiff benchmark       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let tokenizer = CharTokenizer::from_text(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let context_length = 16;

    let (train_data, val_data) = TextDataset::train_val_split(CORPUS, &tokenizer, context_length);

    println!("[ Setup ] Building tokenizer and dataset...");
    println!("         Vocabulary:     {} unique characters", vocab_size);
    println!("         Context window: {} tokens", context_length);
    println!(
        "         Random loss:    {:.4}  (= ln({}))",
        (vocab_size as f32).ln(),
        vocab_size
    );

    let model_config = Config {
        vocab_size,
        context_length,
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
    };

    let eval_interval = (train_steps / 5).max(1);

    // =========================================================================
    // Phase A — Scalar autograd (original engine)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!(" PHASE A: Scalar autograd  (one Value node per weight element)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut rng_a = InitRng::new(42);
    let scalar_model = Gpt::new(model_config.clone(), &mut rng_a);

    {
        use nn::Module;
        let n = scalar_model.parameters().len();
        println!(
            " Model:  {:.1}K scalar nodes  ({} param matrices × ~{} elements avg)",
            n as f32 / 1000.0,
            0,
            n
        );
    }

    let scalar_cfg = TrainConfig {
        max_steps: train_steps,
        eval_interval,
        learning_rate: 1e-3,
        grad_clip: 1.0,
    };

    let t_scalar_start = std::time::Instant::now();
    train(
        &scalar_model,
        &tokenizer,
        &train_data,
        &val_data,
        &scalar_cfg,
    );
    let t_scalar = t_scalar_start.elapsed();

    println!("\n[ Generation — scalar model ]");
    generate(&scalar_model, &tokenizer, "Il ", 80, 0.8, 5);
    generate(&scalar_model, &tokenizer, "The ", 80, 0.8, 5);

    // =========================================================================
    // Phase B — Tensor autodiff (new engine)
    // =========================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!(" PHASE B: Tensor autodiff  (one TensorNode per weight matrix)");
    println!("═══════════════════════════════════════════════════════════════");

    let mut rng_b = InitRng::new(42);
    let tensor_model = Gpt2::new(model_config.clone(), &mut rng_b);

    {
        use nn2::Module2;
        let params = tensor_model.parameters();
        let total_elems: usize = params.iter().map(|p| p.data().rows * p.data().cols).sum();
        println!(
            " Model:  {} tensor nodes  ({} elements total)",
            params.len(),
            total_elems
        );
    }

    let tensor_cfg = TrainConfig2 {
        max_steps: train_steps,
        eval_interval,
        learning_rate: 1e-3,
        grad_clip: 1.0,
        ..TrainConfig2::default()
    };

    let t_tensor_start = std::time::Instant::now();
    train2(&tensor_model, &train_data, &val_data, &tensor_cfg);
    let t_tensor = t_tensor_start.elapsed();

    println!("\n[ Generation — tensor model ]");
    for prompt_str in &["Il ", "The "] {
        use std::io::Write;
        let ids: Vec<usize> = tokenizer
            .encode(prompt_str)
            .iter()
            .map(|&x| x as usize)
            .collect();
        print!("{}", prompt_str);
        tensor_model.generate_cached_streaming(&ids, 80, 0.8, 5, |tok_id| {
            print!("{}", tokenizer.decode(&[tok_id as u32]));
            std::io::stdout().flush().ok();
        });
        println!();
    }

    // =========================================================================
    // Benchmark summary
    // =========================================================================
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                   Benchmark Summary                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║  Steps: {:4}                                                ║",
        train_steps
    );
    println!("║                                                              ║");
    println!(
        "║  Scalar autograd:   {:>8.2}s   ({:>5.0} ms/step)           ║",
        t_scalar.as_secs_f64(),
        t_scalar.as_millis() as f64 / train_steps as f64
    );
    println!(
        "║  Tensor autodiff:   {:>8.2}s   ({:>5.0} ms/step)           ║",
        t_tensor.as_secs_f64(),
        t_tensor.as_millis() as f64 / train_steps as f64
    );

    let speedup = t_scalar.as_secs_f64() / t_tensor.as_secs_f64();
    println!("║                                                              ║");
    println!(
        "║  Speedup:           {:>7.1}x                                ║",
        speedup
    );
    println!("║                                                              ║");
    println!("║  Why faster?                                                 ║");
    println!("║  • Scalar: ~400K nodes in graph → 400K backward visits      ║");
    println!("║  • Tensor:    ~60 nodes in graph →   60 backward visits     ║");
    println!("║  • Each tensor backward does a SIMD-able matmul instead     ║");
    println!("║    of millions of individual scalar multiply-accumulate ops  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nDone.");
}
