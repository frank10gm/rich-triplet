/// # Dataset — feeding text to the model
///
/// A language model is trained on a deceptively simple task:
///
///   **Given these N tokens, predict the next one.**
///
/// That's it. No labels, no human annotations, no reward signal.
/// Just next-token prediction, repeated billions of times.
/// This is called *self-supervised learning* — the text itself provides
/// the supervision signal.
///
/// ## How training examples are created
///
/// We take a long sequence of token ids and slide a window of size
/// `context_length` (also called `block_size`) over it:
///
///   tokens = [5, 23, 7, 14, 88, 3, 42, 19, ...]
///
///   context_length = 4
///
///   Example 1:  input = [5, 23, 7, 14]   target = [23, 7, 14, 88]
///   Example 2:  input = [23, 7, 14, 88]  target = [7, 14, 88, 3]
///   Example 3:  input = [7, 14, 88, 3]   target = [14, 88, 3, 42]
///   ...
///
/// Notice: target[i] = input[i+1]. For every position in the input,
/// the target is the *next* token. The model learns to predict it.
///
/// This means each training window of length T contains T separate
/// prediction tasks packed together — very data-efficient!
///
/// ## Batching
///
/// We don't train on one example at a time — that's slow and the gradient
/// signal is noisy. Instead we collect B examples into a *batch*:
///
///   inputs shape:  [batch_size, context_length]
///   targets shape: [batch_size, context_length]
///
/// All B examples are processed in parallel (in the real implementation,
/// on a GPU). The gradients are averaged across the batch before the
/// weight update. Larger batches → smoother gradients → more stable training.
///
/// Typical values in real models:
///   GPT-2 small:  context_length=1024, batch_size=64
///   GPT-3:        context_length=2048, batch_size=3.2M tokens
///   Our model:    context_length=64,   batch_size=4  (limited by our CPU)

use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

// =============================================================================
// TextDataset
// =============================================================================

pub struct TextDataset {
    /// The entire corpus encoded as a flat sequence of token ids.
    /// All the text is concatenated into one long sequence.
    tokens: Vec<u32>,

    /// How many tokens the model sees at once (its "memory").
    /// Also called block_size or sequence_length.
    pub context_length: usize,
}

impl TextDataset {
    /// Build a dataset from raw text using any tokenizer.
    pub fn from_text(text: &str, tokenizer: &dyn Tokenizer, context_length: usize) -> Self {
        let tokens = tokenizer.encode(text);
        println!(
            "Dataset: {} chars → {} tokens (vocab size {})",
            text.len(),
            tokens.len(),
            tokenizer.vocab_size()
        );
        TextDataset { tokens, context_length }
    }

    /// Total number of (input, target) pairs available.
    ///
    /// We need at least context_length + 1 tokens to form one pair
    /// (context_length for input + 1 for the last target token).
    pub fn len(&self) -> usize {
        if self.tokens.len() <= self.context_length {
            0
        } else {
            self.tokens.len() - self.context_length
        }
    }

    /// Get a single (input, target) pair at position `idx`.
    ///
    /// Returns two 1-D tensors of length `context_length`:
    ///   input:  tokens[idx .. idx + context_length]
    ///   target: tokens[idx+1 .. idx + context_length + 1]
    pub fn get_pair(&self, idx: usize) -> (Tensor, Tensor) {
        assert!(idx + self.context_length < self.tokens.len(), "index out of bounds");

        let input: Vec<f32> = self.tokens[idx..idx + self.context_length]
            .iter()
            .map(|&t| t as f32)
            .collect();

        let target: Vec<f32> = self.tokens[idx + 1..idx + self.context_length + 1]
            .iter()
            .map(|&t| t as f32)
            .collect();

        (
            Tensor::new(input, vec![self.context_length]),
            Tensor::new(target, vec![self.context_length]),
        )
    }

    /// Sample a random batch of (input, target) pairs.
    ///
    /// Returns two 2-D tensors of shape [batch_size, context_length].
    /// `rng_seed` is used for reproducible sampling.
    pub fn random_batch(&self, batch_size: usize, rng_seed: u64) -> (Tensor, Tensor) {
        let n = self.len();
        assert!(n >= batch_size, "dataset too small for batch size {}", batch_size);

        // Simple LCG random number generator (no dependency needed)
        let mut rng = Lcg::new(rng_seed);

        let mut input_data = Vec::with_capacity(batch_size * self.context_length);
        let mut target_data = Vec::with_capacity(batch_size * self.context_length);

        for _ in 0..batch_size {
            let idx = (rng.next() as usize) % n;
            let (inp, tgt) = self.get_pair(idx);
            input_data.extend_from_slice(&inp.data);
            target_data.extend_from_slice(&tgt.data);
        }

        let shape = vec![batch_size, self.context_length];
        (
            Tensor::new(input_data, shape.clone()),
            Tensor::new(target_data, shape),
        )
    }

    /// Split the dataset into train and validation portions.
    ///
    /// Convention: first 90% = training, last 10% = validation.
    /// We never train on validation data — it's used to measure
    /// how well the model generalizes to text it hasn't seen.
    pub fn train_val_split(text: &str, tokenizer: &dyn Tokenizer, context_length: usize)
        -> (TextDataset, TextDataset)
    {
        let all_tokens = tokenizer.encode(text);
        let split = (all_tokens.len() as f64 * 0.9) as usize;

        let train_tokens = all_tokens[..split].to_vec();
        let val_tokens = all_tokens[split..].to_vec();

        println!(
            "Train/val split: {} / {} tokens",
            train_tokens.len(),
            val_tokens.len()
        );

        (
            TextDataset { tokens: train_tokens, context_length },
            TextDataset { tokens: val_tokens, context_length },
        )
    }
}

// =============================================================================
// Minimal LCG random number generator
// =============================================================================
//
// We implement our own tiny RNG to avoid adding a dependency.
// LCG (Linear Congruential Generator):
//   state = (a * state + c) mod m
//
// Parameters from Knuth / "Numerical Recipes":
//   a = 6364136223846793005
//   c = 1442695040888963407
//   m = 2^64  (implicit from u64 overflow)

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed.wrapping_add(1) }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::CharTokenizer;

    fn make_dataset(text: &str, ctx: usize) -> TextDataset {
        let tok = CharTokenizer::from_text(text);
        TextDataset::from_text(text, &tok, ctx)
    }

    #[test]
    fn test_dataset_len() {
        // "abcdefghij" = 10 chars, context_length = 3
        // Valid start positions: 0..6  (need idx + 3 < 10, so idx <= 6)
        // len = 10 - 3 = 7
        let ds = make_dataset("abcdefghij", 3);
        assert_eq!(ds.len(), 7);
    }

    #[test]
    fn test_get_pair_offset_by_one() {
        // The target must be input shifted by one position
        let text = "abcdefghij";
        let tok = CharTokenizer::from_text(text);
        let ds = TextDataset::from_text(text, &tok, 4);

        let (inp, tgt) = ds.get_pair(0);
        // input[0] and target[0] are different (target is one step ahead)
        // target[i] == input[i+1] for all i in 0..context_length-1
        for i in 0..3 {
            assert_eq!(
                tgt.data[i], inp.data[i + 1],
                "target[{}] should equal input[{}]", i, i + 1
            );
        }
    }

    #[test]
    fn test_batch_shape() {
        let ds = make_dataset("abcdefghijklmnopqrstuvwxyz", 4);
        let (inp, tgt) = ds.random_batch(3, 42);
        assert_eq!(inp.shape, vec![3, 4]);
        assert_eq!(tgt.shape, vec![3, 4]);
    }

    #[test]
    fn test_batch_reproducible() {
        // Same seed → same batch
        let ds = make_dataset("abcdefghijklmnopqrstuvwxyz", 4);
        let (a1, _) = ds.random_batch(3, 99);
        let (a2, _) = ds.random_batch(3, 99);
        assert_eq!(a1.data, a2.data);
    }

    #[test]
    fn test_batch_different_seeds() {
        // Different seeds → (very likely) different batches
        let ds = make_dataset("abcdefghijklmnopqrstuvwxyz", 4);
        let (a, _) = ds.random_batch(3, 1);
        let (b, _) = ds.random_batch(3, 2);
        assert_ne!(a.data, b.data);
    }

    #[test]
    fn test_train_val_split_proportions() {
        let text = "abcdefghij".repeat(100); // 1000 chars
        let tok = CharTokenizer::from_text(&text);
        let (train, val) = TextDataset::train_val_split(&text, &tok, 4);
        // ~90% train, ~10% val
        let total = train.tokens.len() + val.tokens.len();
        let train_ratio = train.tokens.len() as f64 / total as f64;
        assert!((train_ratio - 0.9).abs() < 0.01);
    }
}
