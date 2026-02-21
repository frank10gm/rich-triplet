/// # Tokenizer — turning text into numbers (and back)
///
/// A language model never sees text directly. It sees a sequence of integers
/// called *tokens*. The tokenizer's job is to define the mapping:
///
///   "ciao" → [99, 12, 4, 77]    (encode)
///   [99, 12, 4, 77] → "ciao"    (decode)
///
/// ## Why not just use character codes (ASCII/Unicode)?
///
/// You could — and we'll start with exactly that approach for simplicity.
/// But the vocabulary would be tiny (only ~100 characters) and the model
/// would have to learn to spell every word from scratch. It would need very
/// long sequences to represent even short sentences.
///
/// ## What real LLMs use: Byte-Pair Encoding (BPE)
///
/// BPE starts with individual characters, then iteratively merges the most
/// frequent *pair* of tokens into a new single token. After enough merges:
///
///   "italiano" might become a single token
///   "the" is almost certainly a single token
///   "antidisestablishmentarianism" gets split into ~5 tokens
///
/// GPT-2 has ~50,000 tokens. GPT-4 uses ~100,000.
/// A larger vocabulary means shorter sequences (faster) but a bigger
/// embedding table (more memory).
///
/// ## Our plan
///
/// 1. `CharTokenizer`  — simplest possible, maps each unique character to an id
/// 2. `BpeTokenizer`   — the real thing, learned from training data
///
/// We start with `CharTokenizer` so you can see the structure clearly,
/// then build BPE on top of the same interface.

use std::collections::HashMap;

// =============================================================================
// Shared trait — both tokenizers implement the same interface
// =============================================================================

pub trait Tokenizer {
    /// Convert a string of text into a sequence of token ids.
    fn encode(&self, text: &str) -> Vec<u32>;

    /// Convert a sequence of token ids back into text.
    fn decode(&self, ids: &[u32]) -> String;

    /// Total number of distinct tokens in the vocabulary.
    fn vocab_size(&self) -> usize;

    /// The id used to represent "unknown" / out-of-vocabulary items.
    fn unk_id(&self) -> u32;
}

// =============================================================================
// 1. Character-level tokenizer
// =============================================================================
//
// Algorithm:
//   1. Scan all text and collect every unique character
//   2. Sort them for determinism
//   3. Assign each character an integer id, starting from 0
//
// Example on "ciao mondo":
//   unique chars sorted: [' ', 'a', 'c', 'd', 'i', 'm', 'n', 'o']
//   vocabulary:  ' '→0, 'a'→1, 'c'→2, 'd'→3, 'i'→4, 'm'→5, 'n'→6, 'o'→7
//
//   encode("ciao") → [2, 4, 1, 7]
//   decode([2,4,1,7]) → "ciao"

pub struct CharTokenizer {
    /// char → token id
    char_to_id: HashMap<char, u32>,

    /// token id → char  (inverse mapping, for decoding)
    id_to_char: Vec<char>,
}

impl CharTokenizer {
    /// Build the vocabulary by scanning the provided training text.
    pub fn from_text(text: &str) -> Self {
        // Collect unique characters
        let mut chars: Vec<char> = text.chars().collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        // Sort for determinism across runs
        chars.sort();

        // Build the two-way mapping
        let mut char_to_id = HashMap::new();
        let mut id_to_char = Vec::new();

        for (id, ch) in chars.iter().enumerate() {
            char_to_id.insert(*ch, id as u32);
            id_to_char.push(*ch);
        }

        CharTokenizer { char_to_id, id_to_char }
    }

    /// Show the full vocabulary (useful for debugging small datasets).
    pub fn print_vocab(&self) {
        println!("Vocabulary ({} tokens):", self.id_to_char.len());
        for (id, ch) in self.id_to_char.iter().enumerate() {
            let display = match ch {
                '\n' => "\\n".to_string(),
                '\t' => "\\t".to_string(),
                ' '  => "SPACE".to_string(),
                c    => c.to_string(),
            };
            println!("  {:4} → {:?}", id, display);
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&self.unk_id()))
            .collect()
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|&id| {
                self.id_to_char
                    .get(id as usize)
                    .copied()
                    .unwrap_or('\u{FFFD}') // Unicode replacement character for unknown
            })
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.id_to_char.len()
    }

    fn unk_id(&self) -> u32 {
        // We use the last id as the unknown token
        // In a real tokenizer you'd reserve id 0 for this
        self.id_to_char.len().saturating_sub(1) as u32
    }
}

// =============================================================================
// 2. Byte-Pair Encoding (BPE) tokenizer
// =============================================================================
//
// BPE was introduced in 2016 (Sennrich et al.) and is used by GPT-2, GPT-3,
// GPT-4, LLaMA, Mistral, and almost every modern LLM.
//
// ## The training algorithm (offline, run once on corpus):
//
//   Start: vocabulary = all individual bytes (256 tokens, ids 0-255)
//
//   Repeat N times (N = number of merges you want):
//     1. Count every adjacent pair of tokens in the entire corpus
//     2. Find the most frequent pair, e.g. ('t', 'h')
//     3. Add a new token "th" to the vocabulary
//     4. Replace every occurrence of ('t', 'h') in the corpus with "th"
//
//   After N merges the vocabulary has 256 + N tokens.
//   GPT-2 used 50,000 - 256 = ~49,744 merges.
//
// ## The encoding algorithm (online, used at inference/training time):
//
//   Given a new string:
//   1. Start by splitting into individual bytes
//   2. Repeatedly apply the learned merges in the order they were learned
//      (earlier merges take priority)
//   3. Stop when no more merges can be applied
//
// ## Example trace with 2 merges:
//
//   corpus: "aaabdaaabac"
//   initial tokens: ['a','a','a','b','d','a','a','a','b','a','c']
//
//   iter 1: most frequent pair = ('a','a') → 6 times
//           new token: "aa" (id 256 if byte-level)
//           corpus becomes: ['aa','a','b','d','aa','a','b','a','c']
//
//   iter 2: most frequent pair = ('aa','a') → 2 times
//           new token: "aaa"
//           corpus becomes: ['aaa','b','d','aaa','b','a','c']
//
//   Now encoding "aaab" → apply merge 1: "aa"+"b" → still ["aa","a","b"]
//                        → apply merge 2: "aa"+"a" → ["aaa","b"]
//                        → result: [id_of_aaa, id_of_b]

pub struct BpeTokenizer {
    /// The complete vocabulary: id → string representation of that token
    vocab: Vec<String>,

    /// string → id (inverse of vocab, for fast lookup)
    token_to_id: HashMap<String, u32>,

    /// The merge rules, in the order they were learned.
    /// Each rule is (left_token, right_token) → merged_token_string.
    /// To encode, we apply these rules in order.
    merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    /// Learn BPE merges from a text corpus.
    ///
    /// `num_merges` controls vocabulary size: final vocab = 256 + num_merges
    pub fn train(text: &str, num_merges: usize) -> Self {
        // ---- Step 1: initialize vocabulary with all 256 bytes ----
        // We work at the byte level (not char level) so we can handle any UTF-8.
        // Each byte 0-255 gets its own token.
        let mut vocab: Vec<String> = (0u8..=255)
            .map(|b| {
                // Represent each byte as a string.
                // Printable ASCII stays as-is; others get a hex escape.
                if b.is_ascii_graphic() || b == b' ' {
                    (b as char).to_string()
                } else {
                    format!("<0x{:02X}>", b)
                }
            })
            .collect();

        let mut token_to_id: HashMap<String, u32> = vocab
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, s)| (s, i as u32))
            .collect();

        // ---- Step 2: tokenize the entire corpus into byte-level tokens ----
        // We represent the corpus as a sequence of token ids for efficiency.
        let mut corpus: Vec<u32> = text
            .bytes()
            .map(|b| b as u32)
            .collect();

        let mut merges: Vec<(String, String)> = Vec::new();

        println!("BPE training: {} bytes, {} merges requested", corpus.len(), num_merges);

        // ---- Step 3: iteratively merge the most frequent pair ----
        for merge_idx in 0..num_merges {
            // Count all adjacent pairs in the corpus
            let pair_counts = count_pairs(&corpus);

            if pair_counts.is_empty() {
                println!("No more pairs to merge at step {}", merge_idx);
                break;
            }

            // Find the most frequent pair (tie-break by pair value for determinism)
            let best_pair = pair_counts
                .iter()
                .max_by_key(|&(pair, count)| (*count, *pair))
                .map(|(pair, _)| *pair)
                .unwrap();

            let (left_id, right_id) = best_pair;
            let left_str = vocab[left_id as usize].clone();
            let right_str = vocab[right_id as usize].clone();
            let merged_str = format!("{}{}", left_str, right_str);

            // Assign a new id to the merged token
            let new_id = vocab.len() as u32;
            vocab.push(merged_str.clone());
            token_to_id.insert(merged_str.clone(), new_id);
            merges.push((left_str.clone(), right_str.clone()));

            if merge_idx < 10 || merge_idx % 100 == 0 {
                println!(
                    "  merge {:4}: {:?} + {:?} → {:?} (id {}), count={}",
                    merge_idx,
                    left_str,
                    right_str,
                    merged_str,
                    new_id,
                    pair_counts[&best_pair]
                );
            }

            // Apply the merge: replace all (left_id, right_id) pairs in corpus
            corpus = apply_merge(&corpus, left_id, right_id, new_id);
        }

        println!("BPE done. Vocabulary size: {}", vocab.len());

        BpeTokenizer { vocab, token_to_id, merges }
    }

    /// Encode using the learned merges.
    ///
    /// Note: this is a simplified "re-apply all merges in order" approach.
    /// Production tokenizers (like tiktoken) use more efficient data structures,
    /// but the result is identical.
    fn encode_str(&self, text: &str) -> Vec<u32> {
        // Start: every byte is its own token
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        // Apply each merge rule in the order it was learned
        for (left_str, right_str) in &self.merges {
            // Find the ids for the two parts
            let left_id = match self.token_to_id.get(left_str) {
                Some(&id) => id,
                None => continue,
            };
            let right_id = match self.token_to_id.get(right_str) {
                Some(&id) => id,
                None => continue,
            };
            let merged_str = format!("{}{}", left_str, right_str);
            let merged_id = match self.token_to_id.get(&merged_str) {
                Some(&id) => id,
                None => continue,
            };

            tokens = apply_merge(&tokens, left_id, right_id, merged_id);
        }

        tokens
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_str(text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        // Concatenate the string representation of each token,
        // then interpret the resulting bytes as UTF-8.
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get(id as usize) {
                if token_str.starts_with("<0x") && token_str.ends_with('>') {
                    // Parse hex-escaped byte: "<0xXX>"
                    if let Ok(b) = u8::from_str_radix(&token_str[3..5], 16) {
                        bytes.push(b);
                    }
                } else {
                    bytes.extend_from_slice(token_str.as_bytes());
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn unk_id(&self) -> u32 {
        // BPE on bytes has no true UNK — every byte sequence is encodable
        0
    }
}

// =============================================================================
// BPE helper functions
// =============================================================================

/// Count every adjacent (left, right) pair of token ids in a sequence.
fn count_pairs(tokens: &[u32]) -> HashMap<(u32, u32), usize> {
    let mut counts = HashMap::new();
    for window in tokens.windows(2) {
        let pair = (window[0], window[1]);
        *counts.entry(pair).or_insert(0) += 1;
    }
    counts
}

/// Replace all occurrences of (left_id, right_id) in tokens with merged_id.
///
/// We scan left-to-right. When we find the pair, we emit merged_id and skip
/// both original tokens. Otherwise we emit the current token unchanged.
fn apply_merge(tokens: &[u32], left_id: u32, right_id: u32, merged_id: u32) -> Vec<u32> {
    let mut result = Vec::with_capacity(tokens.len());
    let mut i = 0;
    while i < tokens.len() {
        // Check if this position starts the target pair
        if i + 1 < tokens.len() && tokens[i] == left_id && tokens[i + 1] == right_id {
            result.push(merged_id);
            i += 2; // skip both tokens of the pair
        } else {
            result.push(tokens[i]);
            i += 1;
        }
    }
    result
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // --- CharTokenizer tests ---

    #[test]
    fn test_char_encode_decode_roundtrip() {
        let text = "ciao mondo, hello world!";
        let tok = CharTokenizer::from_text(text);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_char_vocab_size() {
        // "abc" has 3 unique chars
        let tok = CharTokenizer::from_text("abc");
        assert_eq!(tok.vocab_size(), 3);

        // "aabbcc" still has 3 unique chars
        let tok2 = CharTokenizer::from_text("aabbcc");
        assert_eq!(tok2.vocab_size(), 3);
    }

    #[test]
    fn test_char_bilingual() {
        let text = "buongiorno, good morning!";
        let tok = CharTokenizer::from_text(text);
        let ids = tok.encode("buon");
        let back = tok.decode(&ids);
        assert_eq!(back, "buon");
    }

    #[test]
    fn test_char_deterministic() {
        // Two tokenizers built from the same text must produce identical encodings
        let text = "un piccolo modello linguistico";
        let tok1 = CharTokenizer::from_text(text);
        let tok2 = CharTokenizer::from_text(text);
        assert_eq!(tok1.encode("piccolo"), tok2.encode("piccolo"));
    }

    // --- BPE helper tests ---

    #[test]
    fn test_count_pairs() {
        // [1, 2, 1, 2, 1] → pair (1,2) appears twice, (2,1) appears twice
        let tokens = vec![1u32, 2, 1, 2, 1];
        let counts = count_pairs(&tokens);
        assert_eq!(counts[&(1, 2)], 2);
        assert_eq!(counts[&(2, 1)], 2);
    }

    #[test]
    fn test_apply_merge() {
        // Replace pair (1,2) with 99 in [1,2,3,1,2]
        // → [99, 3, 99]
        let tokens = vec![1u32, 2, 3, 1, 2];
        let result = apply_merge(&tokens, 1, 2, 99);
        assert_eq!(result, vec![99, 3, 99]);
    }

    #[test]
    fn test_apply_merge_no_overlap() {
        // Merges must NOT be overlapping: [1,1,1] with pair (1,1)→99
        // should give [99, 1], not [99, <overlap>]
        let tokens = vec![1u32, 1, 1];
        let result = apply_merge(&tokens, 1, 1, 99);
        assert_eq!(result, vec![99, 1]);
    }

    // --- BPE tokenizer tests ---

    #[test]
    fn test_bpe_encode_decode_roundtrip() {
        let corpus = "buongiorno buongiorno buongiorno hello hello world world world";
        let tok = BpeTokenizer::train(corpus, 20);
        let ids = tok.encode("buongiorno");
        let back = tok.decode(&ids);
        assert_eq!(back, "buongiorno");
    }

    #[test]
    fn test_bpe_compression() {
        // After training on a text with many "buongiorno",
        // encoding it should produce fewer tokens than bytes.
        let word = "buongiorno ";
        let corpus = word.repeat(50);
        let tok = BpeTokenizer::train(&corpus, 30);

        let byte_count = word.len();
        let token_count = tok.encode(word).len();

        // The tokenizer should compress: fewer tokens than bytes
        assert!(
            token_count < byte_count,
            "expected compression: {} tokens should be < {} bytes",
            token_count,
            byte_count
        );
    }

    #[test]
    fn test_bpe_vocab_size_grows_with_merges() {
        let corpus = "abcabc abcabc abcabc";
        let tok0 = BpeTokenizer::train(corpus, 0);
        let tok5 = BpeTokenizer::train(corpus, 5);
        assert!(tok5.vocab_size() > tok0.vocab_size());
    }
}
