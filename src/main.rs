mod autograd;
mod autograd2;
mod dataset;
mod gguf_loader;
#[cfg(feature = "metal")]
mod metal_ops;
#[cfg(feature = "metal")]
mod metal_decode;
#[cfg(feature = "metal")]
mod metal_decode_qwen35;
mod ndarray;
mod nn;
mod nn2;
mod tensor;
mod tokenizer;
mod train;
mod train2;
mod transformer;
mod transformer2;
mod transformer3;
mod transformer4;
mod transformer_qwen35;

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
}

impl CliArgs {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut a = CliArgs {
            prompt: None,
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
    println!("  --model NAME             Architecture: gpt-oss | gemma3-1b | gemma3-4b | qwen35-4b | qwen35-9b");
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
    println!("  --pretokenize S D        Tokenize text file S, write binary D.bin");
    println!("                           Uses char tokenizer built from S.");
    println!("                           For BPE: also pass --vocab and --merges.");
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
    let mut token_ids: Vec<usize> = vec![im_start];
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
        let is_qwen35 = args
            .model
            .as_deref()
            .map_or(false, |m| m.starts_with("qwen35"));
        let is_gemma3 = args.tokenizer_model.is_some()
            || args
                .model
                .as_deref()
                .map_or(false, |m| m.starts_with("gemma3"))
            || (!is_qwen35 && args
                .weights
                .as_deref()
                .map_or(false, |w| w.ends_with(".gguf")));
        if is_qwen35 {
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
