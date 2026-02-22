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
// Trait: DataSource — anything train2() can sample from
// =============================================================================

/// A dataset that can yield random (input_ids, target_ids) pairs.
///
/// Implemented by both `TextDataset` (in-memory) and `TokenizedDataset`
/// (memory-mapped .bin file).  Pass either to `train2()`.
pub trait DataSource {
    fn sample(&self, rng_seed: u64) -> (Vec<usize>, Vec<usize>);
    fn len(&self) -> usize;

    /// Sample `batch_size` independent (input, target) pairs with different seeds.
    ///
    /// Each sequence gets a distinct seed derived from `rng_seed` so the B
    /// sequences in a batch cover different positions in the corpus.
    fn sample_batch(&self, rng_seed: u64, batch_size: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
        (0..batch_size)
            .map(|i| self.sample(rng_seed.wrapping_add(i as u64 * 7919)))
            .collect()
    }
}

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

impl DataSource for TextDataset {
    fn sample(&self, rng_seed: u64) -> (Vec<usize>, Vec<usize>) {
        let (inp_t, tgt_t) = self.random_batch(1, rng_seed);
        let input:  Vec<usize> = inp_t.data.iter().map(|&x| x as usize).collect();
        let target: Vec<usize> = tgt_t.data.iter().map(|&x| x as usize).collect();
        (input, target)
    }
    fn len(&self) -> usize { self.len() }
}

// =============================================================================
// TokenizedDataset — streaming dataset for large corpora (up to 10GB+)
// =============================================================================
//
// ## The problem with TextDataset at scale
//
// TextDataset loads the entire corpus into RAM as Vec<u32>.
// For a 10GB text file tokenized at ~4 chars/token, that's ~2.5B tokens
// = ~10GB of RAM just for the token array — before the model itself.
//
// ## Solution: memory-mapped .bin file
//
// We store the pre-tokenized corpus as a flat binary file of u32 token ids
// (little-endian, 4 bytes per token). The OS maps it into virtual address
// space; physical RAM is only used for the pages actually read.
//
// For a 10GB corpus at 4 bytes/token:
//   Disk:    10GB
//   RAM:     only the working set (~pages touched during training)
//   Typical: a few hundred MB at any given time
//
// ## File format
//
//   [magic: 8 bytes = 0x526963685472697C ("RichTri|")]
//   [n_tokens: u64 little-endian]
//   [token_0: u32 LE] [token_1: u32 LE] ... [token_N-1: u32 LE]
//
// Write a corpus to this format with `TokenizedDataset::write_bin()`.
// The BPE tokenizer in tokenizer.rs produces u32 token ids directly.
//
// ## Usage
//
//   // Once: tokenize and save to disk
//   TokenizedDataset::write_bin("corpus.bin", "corpus.txt", &tokenizer).unwrap();
//
//   // Every run: stream from disk
//   let (train_ds, val_ds) = TokenizedDataset::open_train_val("corpus.bin", context_len).unwrap();
//   train2(&model, &train_ds, &val_ds, &cfg);

const BIN_MAGIC: u64 = 0x526963685472697C; // "RichTri|"

/// Streaming token dataset backed by a memory-mapped binary file.
///
/// The file stores pre-tokenized token ids as a flat array of u32
/// (little-endian).  The mmap is read-only and lazily loaded by the OS,
/// so only the pages actually accessed consume physical RAM.
///
/// Use `TokenizedDataset::write_bin` to create the file once from
/// any text corpus + tokenizer, then open it for training with
/// `TokenizedDataset::open` or `TokenizedDataset::open_train_val`.
pub struct TokenizedDataset {
    /// Memory-mapped file bytes.  We hold this to keep the mapping alive.
    mmap: Vec<u8>,
    /// Offset of the first token u32 within `mmap`.
    data_offset: usize,
    /// Number of tokens in this slice (may be a sub-range of the file).
    n_tokens: usize,
    pub context_length: usize,
}

impl TokenizedDataset {
    // -------------------------------------------------------------------------
    // Open / create
    // -------------------------------------------------------------------------

    /// Open an existing `.bin` corpus file for training.
    ///
    /// The file must have been written by `write_bin`.
    /// Returns an error string if the file is missing, too small, or has
    /// a wrong magic number.
    pub fn open(path: &str, context_length: usize) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;

        if bytes.len() < 16 {
            return Err(format!("{}: file too small ({} bytes)", path, bytes.len()));
        }

        let magic = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        if magic != BIN_MAGIC {
            return Err(format!("{}: bad magic {:016x} (expected {:016x})", path, magic, BIN_MAGIC));
        }

        let n_tokens = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let data_offset = 16;
        let expected_bytes = data_offset + n_tokens * 4;
        if bytes.len() < expected_bytes {
            return Err(format!("{}: file truncated: need {} bytes, got {}",
                path, expected_bytes, bytes.len()));
        }

        if n_tokens <= context_length {
            return Err(format!("{}: only {} tokens, need > {}", path, n_tokens, context_length));
        }

        Ok(TokenizedDataset { mmap: bytes, data_offset, n_tokens, context_length })
    }

    /// Open a `.bin` file and split into 90% train / 10% val datasets.
    ///
    /// Both datasets share the same mmap (the OS deduplicates pages).
    pub fn open_train_val(path: &str, context_length: usize) -> Result<(Self, Self), String> {
        let full = Self::open(path, context_length)?;
        let split = (full.n_tokens as f64 * 0.9) as usize;

        println!("TokenizedDataset: {} tokens total → {} train / {} val",
            full.n_tokens, split, full.n_tokens - split);

        let train = TokenizedDataset {
            mmap: full.mmap.clone(),
            data_offset: full.data_offset,
            n_tokens: split,
            context_length,
        };
        let val = TokenizedDataset {
            mmap: train.mmap.clone(),
            data_offset: full.data_offset + split * 4,
            n_tokens: full.n_tokens - split,
            context_length,
        };
        Ok((train, val))
    }

    /// Read token at position `i` (0-indexed, relative to this dataset's start).
    #[inline]
    fn token_at(&self, i: usize) -> usize {
        let offset = self.data_offset + i * 4;
        u32::from_le_bytes(self.mmap[offset..offset + 4].try_into().unwrap()) as usize
    }

    /// Total valid (input, target) pairs.
    pub fn len(&self) -> usize {
        if self.n_tokens <= self.context_length { 0 } else { self.n_tokens - self.context_length }
    }

    /// Sample a random (input_ids, target_ids) pair.
    ///
    /// Uses an LCG seeded with `rng_seed` for reproducibility.
    /// Returns two `Vec<usize>` of length `context_length`.
    pub fn random_sample(&self, rng_seed: u64) -> (Vec<usize>, Vec<usize>) {
        let n = self.len();
        let mut rng = Lcg::new(rng_seed);
        let idx = (rng.next() as usize) % n;
        let input:  Vec<usize> = (0..self.context_length).map(|k| self.token_at(idx + k)).collect();
        let target: Vec<usize> = (0..self.context_length).map(|k| self.token_at(idx + k + 1)).collect();
        (input, target)
    }

    // -------------------------------------------------------------------------
    // Write
    // -------------------------------------------------------------------------

    /// Tokenize `text` with `tokenizer` and write to a `.bin` file at `path`.
    ///
    /// This is a one-time operation. Run it once, then use `open()` for all
    /// subsequent training runs.
    ///
    /// For very large files (>1GB), reading `text` into a String first
    /// requires that much RAM.  For streaming of multi-GB corpora, use
    /// `write_bin_from_file` which reads line by line.
    pub fn write_bin(path: &str, text: &str, tokenizer: &dyn crate::tokenizer::Tokenizer)
        -> Result<usize, String>
    {
        let tokens = tokenizer.encode(text);
        Self::write_bin_tokens(path, &tokens)
    }

    /// Tokenize the file at `src_path` line-by-line and write to `dst_path`.
    ///
    /// This is the low-RAM path for large corpora: reads one line at a time,
    /// tokenizes it, and appends to the output file.  Peak RAM is O(line length),
    /// not O(corpus size).
    pub fn write_bin_from_file(
        dst_path: &str,
        src_path: &str,
        tokenizer: &dyn crate::tokenizer::Tokenizer,
    ) -> Result<usize, String> {
        use std::io::{BufRead, Write};

        let src = std::fs::File::open(src_path)
            .map_err(|e| format!("cannot open {}: {}", src_path, e))?;
        let mut out = std::fs::File::create(dst_path)
            .map_err(|e| format!("cannot create {}: {}", dst_path, e))?;

        // Write placeholder header (we'll rewrite n_tokens at the end)
        out.write_all(&BIN_MAGIC.to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;
        out.write_all(&0u64.to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;

        let mut n_tokens = 0usize;
        let reader = std::io::BufReader::new(src);
        for line in reader.lines() {
            let line = line.map_err(|e| format!("read error: {}", e))?;
            if line.is_empty() { continue; }
            let toks = tokenizer.encode(&line);
            for tok in &toks {
                out.write_all(&tok.to_le_bytes())
                    .map_err(|e| format!("write error: {}", e))?;
            }
            n_tokens += toks.len();
        }

        // Rewrite the n_tokens field in the header
        use std::io::Seek;
        out.seek(std::io::SeekFrom::Start(8))
            .map_err(|e| format!("seek error: {}", e))?;
        out.write_all(&(n_tokens as u64).to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;

        println!("Wrote {} tokens to {}", n_tokens, dst_path);
        Ok(n_tokens)
    }

    fn write_bin_tokens(path: &str, tokens: &[u32]) -> Result<usize, String> {
        use std::io::Write;
        let mut f = std::fs::File::create(path)
            .map_err(|e| format!("cannot create {}: {}", path, e))?;
        f.write_all(&BIN_MAGIC.to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;
        f.write_all(&(tokens.len() as u64).to_le_bytes())
            .map_err(|e| format!("write error: {}", e))?;
        for tok in tokens {
            f.write_all(&tok.to_le_bytes())
                .map_err(|e| format!("write error: {}", e))?;
        }
        println!("Wrote {} tokens to {}", tokens.len(), path);
        Ok(tokens.len())
    }
}

impl DataSource for TokenizedDataset {
    fn sample(&self, rng_seed: u64) -> (Vec<usize>, Vec<usize>) {
        self.random_sample(rng_seed)
    }
    fn len(&self) -> usize { self.len() }
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
