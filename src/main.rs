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

use tokenizer::CharTokenizer;
use tokenizer::Tokenizer;
use dataset::TextDataset;
use transformer::{Config, Gpt};
use train::{TrainConfig, train, generate};
use transformer2::Gpt2;
use train2::{TrainConfig2, train2, generate2};
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

fn main() {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║        Rich Triplet — LLM from scratch in Rust              ║");
    println!("║        Scalar autograd  vs  Tensor autodiff benchmark       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // -------------------------------------------------------------------------
    // Shared setup: tokenizer + dataset
    // -------------------------------------------------------------------------
    println!("[ Setup ] Building tokenizer and dataset...");
    let tokenizer = CharTokenizer::from_text(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let context_length = 16;

    let (train_data, val_data) = TextDataset::train_val_split(CORPUS, &tokenizer, context_length);

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

    let train_steps = 200;
    let eval_interval = 40;

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
    println!("  Italian:  ");
    generate(&scalar_model, &tokenizer, "Il ", 80, 0.8, 5);
    println!("  English:  ");
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
    };

    let t_tensor_start = std::time::Instant::now();
    train2(&tensor_model, &tokenizer, &train_data, &val_data, &tensor_cfg);
    let t_tensor = t_tensor_start.elapsed();

    println!("\n[ Generation — tensor model ]");
    println!("  Italian:  ");
    generate2(&tensor_model, &tokenizer, "Il ", 80, 0.8, 5);
    println!("  English:  ");
    generate2(&tensor_model, &tokenizer, "The ", 80, 0.8, 5);

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
