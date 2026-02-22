mod tensor;
mod tokenizer;
mod dataset;
mod autograd;
mod nn;
mod transformer;
mod train;
mod autograd2;
mod nn2;
mod transformer2;
mod train2;
mod transformer3;
#[cfg(feature = "metal")]
mod metal_ops;

use tokenizer::{CharTokenizer, BpeTokenizer, Tokenizer};
use dataset::TextDataset;
use transformer::{Config, Gpt};
use train::{TrainConfig, train, generate};
use transformer2::Gpt2;
use train2::{TrainConfig2, train2};
use autograd2::restore_checkpoint;
use transformer3::{GptOssModel, Config3, SamplingParams};
use nn::InitRng;

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
    prompt:      Option<String>,
    /// --weights DIR     : directory with .safetensors shards for GPT-OSS
    weights:     Option<String>,
    /// --vocab PATH      : BPE vocab.json (required with --weights)
    vocab:       Option<String>,
    /// --merges PATH     : BPE merges.txt (required with --weights)
    merges:      Option<String>,
    /// --max-new N       : tokens to generate (default 200)
    max_new:     usize,
    /// --temp T          : sampling temperature (default 0.8)
    temperature: f32,
    /// --top-k K         : top-k cutoff (default 40, 0 = disabled)
    top_k:       usize,
    /// --top-p P         : nucleus probability (default 1.0 = disabled)
    top_p:       f32,
    /// --seed S          : RNG seed (default 42)
    seed:        u64,
    /// --train-steps N   : steps for on-the-fly training (default 200)
    train_steps: usize,
    /// --checkpoint PATH : load a previously saved .ckpt before generating (skips training)
    checkpoint: Option<String>,
}

impl CliArgs {
    fn parse() -> Self {
        let args: Vec<String> = std::env::args().skip(1).collect();
        let mut a = CliArgs {
            prompt: None, weights: None, vocab: None, merges: None,
            max_new: 200, temperature: 0.8, top_k: 40, top_p: 1.0,
            seed: 42, train_steps: 200, checkpoint: None,
        };
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--prompt"      => { i += 1; if i < args.len() { a.prompt      = Some(args[i].clone()); } }
                "--weights"     => { i += 1; if i < args.len() { a.weights     = Some(args[i].clone()); } }
                "--vocab"       => { i += 1; if i < args.len() { a.vocab       = Some(args[i].clone()); } }
                "--merges"      => { i += 1; if i < args.len() { a.merges      = Some(args[i].clone()); } }
                "--max-new"     => { i += 1; if i < args.len() { a.max_new     = args[i].parse().unwrap_or(200); } }
                "--temp"        => { i += 1; if i < args.len() { a.temperature = args[i].parse().unwrap_or(0.8); } }
                "--top-k"       => { i += 1; if i < args.len() { a.top_k       = args[i].parse().unwrap_or(40); } }
                "--top-p"       => { i += 1; if i < args.len() { a.top_p       = args[i].parse().unwrap_or(1.0); } }
                "--seed"        => { i += 1; if i < args.len() { a.seed        = args[i].parse().unwrap_or(42); } }
                "--train-steps"  => { i += 1; if i < args.len() { a.train_steps = args[i].parse().unwrap_or(200); } }
                "--checkpoint"   => { i += 1; if i < args.len() { a.checkpoint  = Some(args[i].clone()); } }
                "--help" | "-h"  => { print_help(); std::process::exit(0); }
                other => { eprintln!("Unknown argument: {other}"); print_help(); std::process::exit(1); }
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
    println!("  rich-triplet                          Run autograd benchmark");
    println!("  rich-triplet --prompt TEXT            Train on corpus, then generate");
    println!("  rich-triplet --prompt TEXT \\");
    println!("               --weights DIR \\");
    println!("               --vocab vocab.json \\");
    println!("               --merges merges.txt      Load GPT-OSS weights, generate");
    println!();
    println!("OPTIONS:");
    println!("  --prompt TEXT        Prompt text to complete");
    println!("  --weights DIR        Directory with .safetensors shards (GPT-OSS)");
    println!("  --vocab PATH         BPE vocab.json (required with --weights)");
    println!("  --merges PATH        BPE merges.txt (required with --weights)");
    println!("  --max-new N          Tokens to generate          [default: 200]");
    println!("  --temp T             Sampling temperature        [default: 0.8]");
    println!("  --top-k K            Top-K cutoff (0=disabled)   [default: 40]");
    println!("  --top-p P            Nucleus probability         [default: 1.0]");
    println!("  --seed S             RNG seed                    [default: 42]");
    println!("  --train-steps N      Training steps (no-weights) [default: 200]");
    println!("  --checkpoint PATH    Load saved .ckpt instead of training");
}

// =============================================================================
// Generation mode — GPT-OSS with loaded weights
// =============================================================================

fn run_gpt_oss(args: &CliArgs, prompt: &str) {
    use std::io::Write;

    let weights_dir = args.weights.as_deref().unwrap();
    let vocab_path  = args.vocab.as_deref()
        .expect("--vocab required with --weights (path to vocab.json)");
    let merges_path = args.merges.as_deref()
        .expect("--merges required with --weights (path to merges.txt)");

    eprintln!("[ GPT-OSS ] Loading tokenizer...");
    let tok = BpeTokenizer::from_files(vocab_path, merges_path)
        .expect("failed to load BPE tokenizer");

    eprintln!("[ GPT-OSS ] Building model (gpt-oss-20b config)...");
    let config = Config3::gpt_oss_20b();
    let mut rng = InitRng::new(0);
    let mut model = GptOssModel::new(config, &mut rng);

    eprintln!("[ GPT-OSS ] Loading weights from {}...", weights_dir);
    model.load_weights_from_dir(weights_dir)
        .expect("failed to load weights");

    let raw_ids = tok.encode(prompt);
    if raw_ids.is_empty() {
        eprintln!("Error: prompt encodes to zero tokens");
        std::process::exit(1);
    }
    let token_ids: Vec<usize> = raw_ids.iter().map(|&id| id as usize).collect();

    let params = SamplingParams {
        temperature: args.temperature,
        top_k:       args.top_k,
        top_p:       args.top_p,
        seed:        args.seed,
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
// Generation mode — trained Gpt2 on built-in corpus
// =============================================================================

fn run_gpt2_generate(args: &CliArgs, prompt: &str) {
    use std::io::Write;

    let tokenizer = CharTokenizer::from_text(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let context_length = 64;

    let (train_data, val_data) =
        TextDataset::train_val_split(CORPUS, &tokenizer, context_length);

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
        restore_checkpoint(ckpt_path, &model.parameters())
            .expect("failed to restore checkpoint");
    } else {
        eprintln!("[ Train ] vocab={} context={} steps={}", vocab_size, context_length, args.train_steps);
        let cfg = TrainConfig2 {
            max_steps:     args.train_steps,
            eval_interval: args.train_steps / 5,
            learning_rate: 3e-3,
            grad_clip:     1.0,
            ..TrainConfig2::default()
        };
        train2(&model, &tokenizer, &train_data, &val_data, &cfg);
    }

    // Stream tokens using the KV cache (O(T) per step instead of O(T²)).
    let token_ids: Vec<usize> = tokenizer.encode(prompt).iter().map(|&x| x as usize).collect();
    let token_ids = if token_ids.is_empty() { vec![0usize] } else { token_ids };
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    model.generate_cached_streaming(&token_ids, args.max_new, args.temperature, args.top_k, |tok_id| {
        print!("{}", tokenizer.decode(&[tok_id as u32]));
        std::io::stdout().flush().ok();
    });
    println!();
}

fn main() {
    let args = CliArgs::parse();

    // -------------------------------------------------------------------------
    // Generation mode
    // -------------------------------------------------------------------------
    if let Some(ref prompt) = args.prompt.clone() {
        if args.weights.is_some() {
            run_gpt_oss(&args, prompt);
        } else {
            run_gpt2_generate(&args, prompt);
        }
        return;
    }

    // -------------------------------------------------------------------------
    // Benchmark mode (default)
    // -------------------------------------------------------------------------
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
        (vocab_size as f32).ln(), vocab_size
    );

    let model_config = Config {
        vocab_size,
        context_length,
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
    };

    let train_steps = args.train_steps;
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
        println!(" Model:  {:.1}K scalar nodes  ({} param matrices × ~{} elements avg)",
            n as f32 / 1000.0, 0, n);
    }

    let scalar_cfg = TrainConfig {
        max_steps: train_steps,
        eval_interval,
        learning_rate: 1e-3,
        grad_clip: 1.0,
    };

    let t_scalar_start = std::time::Instant::now();
    train(&scalar_model, &tokenizer, &train_data, &val_data, &scalar_cfg);
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
        let total_elems: usize = params.iter()
            .map(|p| p.data().rows * p.data().cols)
            .sum();
        println!(" Model:  {} tensor nodes  ({} elements total)",
            params.len(), total_elems);
    }

    let tensor_cfg = TrainConfig2 {
        max_steps: train_steps,
        eval_interval,
        learning_rate: 1e-3,
        grad_clip: 1.0,
        ..TrainConfig2::default()
    };

    let t_tensor_start = std::time::Instant::now();
    train2(&tensor_model, &tokenizer, &train_data, &val_data, &tensor_cfg);
    let t_tensor = t_tensor_start.elapsed();

    println!("\n[ Generation — tensor model ]");
    for prompt_str in &["Il ", "The "] {
        use std::io::Write;
        let ids: Vec<usize> = tokenizer.encode(prompt_str).iter().map(|&x| x as usize).collect();
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
    println!("║  Steps: {:4}                                                ║", train_steps);
    println!("║                                                              ║");
    println!("║  Scalar autograd:   {:>8.2}s   ({:>5.0} ms/step)           ║",
        t_scalar.as_secs_f64(),
        t_scalar.as_millis() as f64 / train_steps as f64);
    println!("║  Tensor autodiff:   {:>8.2}s   ({:>5.0} ms/step)           ║",
        t_tensor.as_secs_f64(),
        t_tensor.as_millis() as f64 / train_steps as f64);

    let speedup = t_scalar.as_secs_f64() / t_tensor.as_secs_f64();
    println!("║                                                              ║");
    println!("║  Speedup:           {:>7.1}x                                ║", speedup);
    println!("║                                                              ║");
    println!("║  Why faster?                                                 ║");
    println!("║  • Scalar: ~400K nodes in graph → 400K backward visits      ║");
    println!("║  • Tensor:    ~60 nodes in graph →   60 backward visits     ║");
    println!("║  • Each tensor backward does a SIMD-able matmul instead     ║");
    println!("║    of millions of individual scalar multiply-accumulate ops  ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    println!("\nDone.");
}
