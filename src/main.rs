mod tensor;
mod tokenizer;
mod dataset;
mod autograd;
mod nn;
mod transformer;
mod train;

use tokenizer::CharTokenizer;
use tokenizer::Tokenizer;
use dataset::TextDataset;
use transformer::{Config, Gpt};
use train::{TrainConfig, train, generate};
use nn::{InitRng, Module};

// =============================================================================
// Bilingual training corpus — Italian and English
// =============================================================================
//
// A small but real bilingual text. The model will learn:
//   - Italian character patterns and common words
//   - English character patterns and common words
//   - That both languages co-exist (bilingual capability)
//
// For a real model you'd use gigabytes of text. For this educational model,
// a few thousand characters is enough to see learning happen and generate
// recognizable text patterns.
//
// The model has NO idea these are two languages. It just sees token sequences
// and learns to predict the next token. The bilingual "understanding" emerges
// purely from the statistical patterns in the data.

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
    println!("╔══════════════════════════════════════════════════╗");
    println!("║     Rich Triplet — LLM from scratch in Rust      ║");
    println!("╚══════════════════════════════════════════════════╝");
    println!();

    // -------------------------------------------------------------------------
    // Step 1: Build the tokenizer from the corpus
    // -------------------------------------------------------------------------
    println!("[ 1/4 ] Building tokenizer...");
    let tokenizer = CharTokenizer::from_text(CORPUS);
    println!(
        "        Vocabulary: {} unique characters",
        tokenizer.vocab_size()
    );

    // -------------------------------------------------------------------------
    // Step 2: Build the dataset
    // -------------------------------------------------------------------------
    println!("[ 2/4 ] Building dataset...");

    // NOTE: Our scalar autograd engine is intentionally simple for learning —
    // real frameworks use tensor-level autodiff and GPU parallelism.
    // We use a short context and small model so training completes on CPU.
    let context_length = 16;
    let (train_data, val_data) = TextDataset::train_val_split(
        CORPUS,
        &tokenizer,
        context_length,
    );
    println!(
        "        Context window: {} tokens",
        context_length
    );

    // -------------------------------------------------------------------------
    // Step 3: Build the model
    // -------------------------------------------------------------------------
    println!("[ 3/4 ] Building model...");

    let config = Config {
        vocab_size: tokenizer.vocab_size(),
        context_length,
        d_model: 32,
        n_layers: 2,
        n_heads: 2,
    };

    let mut rng = InitRng::new(42);
    let model = Gpt::new(config.clone(), &mut rng);

    let n_params = model.parameters().len();
    println!("        Parameters:    {}", n_params);
    println!("        d_model:       {}", config.d_model);
    println!("        n_layers:      {}", config.n_layers);
    println!("        n_heads:       {}", config.n_heads);
    println!(
        "        Random baseline loss: {:.4}  (= log({}))",
        (tokenizer.vocab_size() as f32).ln(),
        tokenizer.vocab_size()
    );

    // -------------------------------------------------------------------------
    // Step 4: Train
    // -------------------------------------------------------------------------
    println!("[ 4/4 ] Training...");

    let train_cfg = TrainConfig {
        max_steps: 200,
        eval_interval: 40,
        learning_rate: 1e-3,
        grad_clip: 1.0,
    };

    train(&model, &tokenizer, &train_data, &val_data, &train_cfg);

    // -------------------------------------------------------------------------
    // Step 5: Generate text
    // -------------------------------------------------------------------------
    println!("\n[ Generation ] Italian prompt:");
    generate(&model, &tokenizer, "Il ", 80, 0.8, 5);

    println!("\n[ Generation ] English prompt:");
    generate(&model, &tokenizer, "The ", 80, 0.8, 5);

    println!("\n[ Generation ] Bilingual prompt:");
    generate(&model, &tokenizer, "La ", 80, 0.8, 5);

    println!("\nDone.");
}
