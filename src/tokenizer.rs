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

    /// Fast merge lookup: (left_id, right_id) → merged_id.
    /// Built from `merges` at construction time.
    /// Encoding uses this instead of iterating all rules — O(T) per merge step.
    merge_map: HashMap<(u32, u32), u32>,
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

        let merge_map = build_merge_map(&merges, &token_to_id);

        BpeTokenizer { vocab, token_to_id, merges, merge_map }
    }

    // -------------------------------------------------------------------------
    // Load a pre-trained BPE tokenizer from vocab.json + merges.txt
    // -------------------------------------------------------------------------
    //
    // GPT-2, GPT-OSS, LLaMA, and Mistral all ship tokenizers in this format:
    //
    //   vocab.json   — maps token string → integer id (JSON object)
    //   merges.txt   — one merge rule per line: "left right"
    //                  (first line is usually a comment "#version: 0.2")
    //
    // These files can be downloaded from HuggingFace model repos, e.g.:
    //
    //   https://huggingface.co/openai/gpt-oss-20b/resolve/main/vocab.json
    //   https://huggingface.co/openai/gpt-oss-20b/resolve/main/merges.txt
    //
    // ## Encoding conventions
    //
    // Tokens in vocab.json use the GPT-2 byte encoding:
    //   ' '  → 'Ġ'  (U+0120)   — space prefix
    //   '\n' → 'Ċ'  (U+010A)   — newline
    //   raw bytes > 127 → 'Ā'-'ŀ' range (U+0100–U+016F)
    //
    // This loader handles the full mapping bidirectionally.
    //
    // ## tiktoken format
    //
    // tiktoken (used by GPT-4 and GPT-OSS) uses a different on-disk format:
    //   - A single `.tiktoken` file with lines: base64(token_bytes) rank
    //   - The merge rules are implicit in the rank ordering
    //
    // `from_tiktoken_file()` handles this format.

    /// Load a BPE tokenizer from a `vocab.json` + `merges.txt` pair.
    ///
    /// These files are the standard HuggingFace tokenizer format used by
    /// GPT-2, GPT-OSS, LLaMA, and most open-weight models.
    ///
    /// ## Example
    ///
    /// ```no_run
    /// let tok = BpeTokenizer::from_files("./gpt-oss-20b/vocab.json",
    ///                                    "./gpt-oss-20b/merges.txt")
    ///               .expect("failed to load tokenizer");
    /// let ids = tok.encode("Hello, world!");
    /// println!("{}", tok.decode(&ids));
    /// ```
    pub fn from_files(vocab_path: &str, merges_path: &str) -> Result<Self, String> {
        let vocab_str = std::fs::read_to_string(vocab_path)
            .map_err(|e| format!("cannot read {}: {}", vocab_path, e))?;
        let merges_str = std::fs::read_to_string(merges_path)
            .map_err(|e| format!("cannot read {}: {}", merges_path, e))?;
        Self::from_vocab_and_merges_str(&vocab_str, &merges_str)
    }

    /// Load from in-memory strings (useful for testing without disk I/O).
    pub fn from_vocab_and_merges_str(vocab_json: &str, merges_txt: &str) -> Result<Self, String> {
        // --- Parse vocab.json ---
        // Format: {"token_string": id, ...}
        // We need to build: id → decoded_bytes  (then to String)
        let mut id_to_token: HashMap<u32, String> = HashMap::new();
        let max_id = parse_vocab_json(vocab_json, &mut id_to_token)
            .map_err(|e| format!("vocab.json parse error: {}", e))?;

        let vocab_size = max_id as usize + 1;
        let mut vocab: Vec<String> = vec![String::new(); vocab_size];
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_size);

        for (id, tok) in &id_to_token {
            // Decode GPT-2 byte encoding to real bytes, then to UTF-8-ish string
            let decoded = gpt2_decode_token(tok);
            vocab[*id as usize] = decoded.clone();
            token_to_id.insert(decoded, *id);
        }

        // Fill any gaps (shouldn't happen with well-formed vocab.json)
        for (i, s) in vocab.iter_mut().enumerate() {
            if s.is_empty() {
                *s = format!("<unk_{}>", i);
            }
        }

        // --- Parse merges.txt ---
        // Each non-comment line: "left right"
        let mut merges: Vec<(String, String)> = Vec::new();
        for line in merges_txt.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            let mut parts = line.splitn(2, ' ');
            let left_enc  = parts.next().unwrap_or("").to_string();
            let right_enc = parts.next().unwrap_or("").to_string();
            if left_enc.is_empty() || right_enc.is_empty() { continue; }
            // Decode the GPT-2 encoding to get the actual token strings
            let left  = gpt2_decode_token(&left_enc);
            let right = gpt2_decode_token(&right_enc);
            merges.push((left, right));
        }

        let merge_map = build_merge_map(&merges, &token_to_id);

        Ok(BpeTokenizer { vocab, token_to_id, merges, merge_map })
    }

    /// Load from a tiktoken `.tiktoken` file (GPT-4 / GPT-OSS format).
    ///
    /// ## tiktoken format
    ///
    /// Each line of the file is:
    ///   `<base64_encoded_token_bytes> <rank>`
    ///
    /// where `rank` is the merge priority (lower = earlier merge).
    /// The first 256 entries (ranks 0-255) are the individual bytes.
    /// Entries with rank ≥ 256 are merged tokens, in merge order.
    ///
    /// There is no separate merges file — the merge order is encoded in the ranks.
    ///
    /// ## Usage
    ///
    /// ```no_run
    /// // Download from: https://huggingface.co/openai/gpt-oss-20b/resolve/main/cl100k_base.tiktoken
    /// let tok = BpeTokenizer::from_tiktoken_file("./cl100k_base.tiktoken")
    ///               .expect("failed to load");
    /// ```
    pub fn from_tiktoken_file(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        Self::from_tiktoken_str(&content)
    }

    /// Load tiktoken from an in-memory string.
    pub fn from_tiktoken_str(content: &str) -> Result<Self, String> {
        // Collect (rank, token_bytes) pairs
        let mut entries: Vec<(u32, Vec<u8>)> = Vec::new();
        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() { continue; }
            let mut parts = line.splitn(2, ' ');
            let b64  = parts.next().unwrap_or("");
            let rank_str = parts.next().unwrap_or("");
            let rank: u32 = rank_str.trim().parse()
                .map_err(|_| format!("bad rank: {:?}", rank_str))?;
            let token_bytes = base64_decode(b64)
                .map_err(|e| format!("base64 error on {:?}: {}", b64, e))?;
            entries.push((rank, token_bytes));
        }

        // Sort by rank to get canonical order
        entries.sort_by_key(|(rank, _)| *rank);

        // Build vocab: id = rank (tiktoken rank IS the token id)
        let vocab_size = entries.iter().map(|(r, _)| *r as usize + 1).max().unwrap_or(0);
        let mut vocab: Vec<String> = vec![String::new(); vocab_size];
        let mut token_to_id: HashMap<String, u32> = HashMap::with_capacity(vocab_size);

        for (rank, bytes) in &entries {
            // Store raw bytes as a Rust string (may not be valid UTF-8 for non-text tokens;
            // we use lossy conversion and the decoder reconstructs bytes from the vocab).
            let s = bytes_to_token_str(bytes);
            vocab[*rank as usize] = s.clone();
            token_to_id.insert(s, *rank);
        }

        // Derive merges: any token whose bytes can be split into two existing tokens
        // in a way that minimizes sum-of-ranks is the canonical merge.
        // In tiktoken, tokens with rank ≥ 256 were created by merging two lower-rank tokens.
        // We find the split that minimizes left_rank + right_rank for each such token.
        let mut merges: Vec<(String, String)> = Vec::new();
        // Build a map from bytes → rank for fast lookup
        let bytes_to_rank: HashMap<Vec<u8>, u32> = entries.iter()
            .map(|(r, b)| (b.clone(), *r))
            .collect();

        for (rank, bytes) in &entries {
            if *rank < 256 || bytes.len() < 2 { continue; }
            // Find the split that gives the lowest max-rank (earliest merge)
            let mut best: Option<(usize, u32)> = None; // (split_pos, max_rank_of_parts)
            for split in 1..bytes.len() {
                let left  = &bytes[..split];
                let right = &bytes[split..];
                if let (Some(&lr), Some(&rr)) = (bytes_to_rank.get(left), bytes_to_rank.get(right)) {
                    let score = lr.max(rr);
                    if best.is_none() || score < best.unwrap().1 {
                        best = Some((split, score));
                    }
                }
            }
            if let Some((split, _)) = best {
                let left_str  = bytes_to_token_str(&bytes[..split]);
                let right_str = bytes_to_token_str(&bytes[split..]);
                merges.push((left_str, right_str));
            }
        }

        let merge_map = build_merge_map(&merges, &token_to_id);

        Ok(BpeTokenizer { vocab, token_to_id, merges, merge_map })
    }

    // -------------------------------------------------------------------------
    // Fast encoding using the merge map
    // -------------------------------------------------------------------------
    //
    // The naive approach (re-apply all merges in order) is O(V * T) where V is
    // vocabulary size and T is token count. For tiktoken with 100k+ tokens this
    // is unacceptably slow.
    //
    // The fast approach (used by tiktoken):
    //   1. Start with byte-level tokens
    //   2. In a single pass, find the minimum-priority merge that can be applied
    //   3. Apply it, then re-check only the neighbors of the merged pair
    //   4. Repeat until no more merges are possible
    //
    // This runs in O(T log V) using a priority queue, or O(T * V) in the worst
    // case. Our implementation uses the simpler O(T * V) approach with the
    // merge_map for O(1) lookups per pair.

    /// Encode using the fast merge map.
    fn encode_str(&self, text: &str) -> Vec<u32> {
        // Start: every byte is its own token (id = byte value)
        let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();

        // Repeatedly find and apply the highest-priority (lowest merge index) merge.
        // We iterate until no merge is applicable.
        loop {
            let mut best_pos: Option<usize>  = None;
            let mut best_pri: Option<usize>  = None; // lower = higher priority

            for i in 0..tokens.len().saturating_sub(1) {
                let pair = (tokens[i], tokens[i + 1]);
                if let Some(&merged_id) = self.merge_map.get(&pair) {
                    // Find the priority of this merge in the merge list
                    let pri = self.merges.iter().position(|(l, r)| {
                        self.token_to_id.get(l).copied() == Some(pair.0)
                            && self.token_to_id.get(r).copied() == Some(pair.1)
                            && self.token_to_id.get(&format!("{}{}", l, r)).copied() == Some(merged_id)
                    });
                    if let Some(p) = pri {
                        if best_pri.is_none() || p < best_pri.unwrap() {
                            best_pri = Some(p);
                            best_pos = Some(i);
                        }
                    }
                }
            }

            match best_pos {
                None => break,
                Some(pos) => {
                    let pair = (tokens[pos], tokens[pos + 1]);
                    let merged_id = *self.merge_map.get(&pair).unwrap();
                    tokens.remove(pos + 1);
                    tokens[pos] = merged_id;
                }
            }
        }

        tokens
    }
}

impl Tokenizer for BpeTokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        self.encode_str(text)
    }

    fn decode(&self, ids: &[u32]) -> String {
        // Concatenate the byte representation of each token, then convert to UTF-8.
        //
        // Tokens stored by train() use either plain ASCII or "<0xXX>" escape.
        // Tokens stored by from_files()/from_tiktoken_*() use bytes_to_token_str()
        // which stores each byte as a char in the range 0x00-0xFF (Latin-1).
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(token_str) = self.vocab.get(id as usize) {
                if token_str.starts_with("<0x") && token_str.ends_with('>') {
                    // train() format: hex-escaped non-printable byte
                    if let Ok(b) = u8::from_str_radix(&token_str[3..5], 16) {
                        bytes.push(b);
                    }
                } else {
                    // from_files() and from_tiktoken_*() format:
                    // Each char's code point IS the byte value (Latin-1 encoding).
                    // chars() correctly iterates Unicode scalar values.
                    for c in token_str.chars() {
                        let cp = c as u32;
                        if cp <= 0xFF {
                            bytes.push(cp as u8);
                        }
                        // chars above 0xFF can't happen with bytes_to_token_str()
                    }
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

// ---- Fast merge map construction ----

/// Build a (left_id, right_id) → merged_id lookup map from the merge list.
/// Pairs that appear multiple times (shouldn't happen in valid BPE) keep the last entry.
fn build_merge_map(merges: &[(String, String)], token_to_id: &HashMap<String, u32>) -> HashMap<(u32, u32), u32> {
    let mut map = HashMap::new();
    for (left, right) in merges {
        let merged = format!("{}{}", left, right);
        if let (Some(&l), Some(&r), Some(&m)) = (
            token_to_id.get(left),
            token_to_id.get(right),
            token_to_id.get(&merged),
        ) {
            map.insert((l, r), m);
        }
    }
    map
}

// ---- GPT-2 byte encoding / decoding ----
//
// GPT-2 uses a bijective mapping from the 256 byte values to printable Unicode
// characters so that every token string can be displayed as valid UTF-8.
// The mapping was chosen so that printable ASCII (33-126) and space (32) map
// to themselves, and the remaining 33+128 = 161 byte values map to U+0100..U+016F.
//
// Reference:
//   https://github.com/openai/gpt-2/blob/master/src/encoder.py  (bytes_to_unicode)

/// The GPT-2 byte→unicode mapping: returns an array where byte b maps to char MAP[b].
///
/// Matches the Python reference implementation exactly:
///   bs  = range(33,127) + range(161,256)   — printable Latin-1
///   cs  = bs.copy()                         — these bytes map to themselves
///   For every byte NOT in bs (0-32, 127-160):
///     bs.append(b), cs.append(chr(256 + n)), n += 1
///
/// Result: bytes 0-32 and 127-160 map to U+0100..U+0142.
/// In particular: space (32) → U+0120 = Ġ
pub fn gpt2_bytes_to_unicode() -> [char; 256] {
    let mut result = ['\0'; 256];
    // Initial set: printable ASCII (! to ~) and ¡ to ÿ
    let mut bs: Vec<u8> = (b'!'..=b'~').collect();   // 33..=126
    bs.extend(b'\xA1'..=b'\xFF');                     // 161..=255
    // cs is the same bytes cast to chars (these map to themselves)
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    // The remaining 33 + 34 = 67 bytes map to U+0100..U+0142
    let mut n = 0u32;
    for b in 0u8..=255u8 {
        if !bs.contains(&b) {
            cs.push(char::from_u32(0x100 + n).unwrap());
            bs.push(b);
            n += 1;
        }
    }
    for (&b, &c) in bs.iter().zip(&cs) {
        result[b as usize] = c;
    }
    result
}

/// Decode a GPT-2 encoded token string back to raw bytes.
/// E.g. "Ġhello" → b" hello" (leading Ġ = space)
fn gpt2_decode_token_to_bytes(s: &str) -> Vec<u8> {
    let map = gpt2_bytes_to_unicode();
    // Build the inverse: char → byte
    let mut inv = HashMap::new();
    for (b, &c) in map.iter().enumerate() {
        inv.insert(c, b as u8);
    }
    s.chars().filter_map(|c| inv.get(&c).copied()).collect()
}

/// Decode a GPT-2 encoded token to a Rust String (the raw bytes as a string).
/// We use a lossless byte-string representation so decode() can reconstruct bytes.
fn gpt2_decode_token(s: &str) -> String {
    let bytes = gpt2_decode_token_to_bytes(s);
    bytes_to_token_str(&bytes)
}

/// Represent raw bytes as a String that can be stored in the vocab.
/// We use ISO-8859-1 style: bytes 0-127 as ASCII, bytes 128-255 as U+0080–U+00FF.
fn bytes_to_token_str(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

// ---- vocab.json parser ----
//
// Parses: {"token_string": int_id, ...}
// We do a minimal hand-written JSON string scan because we want zero dependencies.

fn parse_vocab_json(json: &str, out: &mut HashMap<u32, String>) -> Result<u32, String> {
    // Work on character indices (not byte indices) to handle multi-byte UTF-8 chars like Ġ.
    let chars: Vec<char> = json.chars().collect();
    let n = chars.len();
    let mut i = 0;
    let mut max_id = 0u32;

    // Skip outer '{'
    while i < n && chars[i] != '{' { i += 1; }
    i += 1;

    loop {
        // Skip whitespace, commas
        while i < n && (chars[i] == ' ' || chars[i] == ',' || chars[i] == '\n' ||
                        chars[i] == '\r' || chars[i] == '\t') { i += 1; }
        if i >= n || chars[i] == '}' { break; }

        // Read key string
        if chars[i] != '"' { return Err(format!("expected '\"' at char pos {}", i)); }
        i += 1; // skip opening quote
        let mut key = String::new();
        // Scan for closing quote (handle \" escapes)
        while i < n {
            if chars[i] == '\\' && i + 1 < n {
                match chars[i + 1] {
                    '"'  => { key.push('"');  i += 2; }
                    '\\' => { key.push('\\'); i += 2; }
                    'n'  => { key.push('\n'); i += 2; }
                    other => { key.push('\\'); key.push(other); i += 2; }
                }
                continue;
            }
            if chars[i] == '"' { break; }
            key.push(chars[i]);
            i += 1;
        }
        i += 1; // skip closing quote

        // Skip colon and whitespace
        while i < n && (chars[i] == ':' || chars[i] == ' ') { i += 1; }

        // Read integer value
        let val_start = i;
        while i < n && chars[i].is_ascii_digit() { i += 1; }
        let id_str: String = chars[val_start..i].iter().collect();
        let id: u32 = id_str.parse()
            .map_err(|_| format!("bad id {:?} at char pos {}", id_str, val_start))?;

        // Store the GPT-2 encoded key (we'll decode it in from_vocab_and_merges_str)
        out.insert(id, key);
        if id > max_id { max_id = id; }
    }

    Ok(max_id)
}

// ---- Base64 decoder (for tiktoken format) ----
//
// Standard base64, no padding required to be correct but tolerated.

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut lookup = [255u8; 256];
    for (i, &c) in ALPHABET.iter().enumerate() { lookup[c as usize] = i as u8; }

    let s = s.trim_end_matches('=');
    let mut out = Vec::with_capacity(s.len() * 3 / 4 + 1);
    let bytes = s.as_bytes();
    let mut buf = 0u32;
    let mut bits = 0u32;

    for &b in bytes {
        let val = lookup[b as usize];
        if val == 255 { return Err(format!("invalid base64 char: {}", b as char)); }
        buf = (buf << 6) | val as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
            buf &= (1 << bits) - 1;
        }
    }

    Ok(out)
}

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

    // --- BPE from_vocab_and_merges_str tests ---

    /// Build a minimal vocab.json + merges.txt for ASCII-only text.
    ///
    /// We use the GPT-2 encoding where space → Ġ, printable ASCII stays as-is.
    /// We only include a safe subset (letters, digits, space, newline) to avoid
    /// JSON-special characters (", \) in token strings.
    fn make_tiny_vocab_merges() -> (String, String) {
        let map = gpt2_bytes_to_unicode();

        // Include only "safe" bytes that don't need escaping in the JSON key.
        // We include: a-z (97-122), A-Z (65-90), 0-9 (48-57), space (32), newline (10).
        let safe_bytes: Vec<u8> = {
            let mut v: Vec<u8> = (b'a'..=b'z').collect();
            v.extend(b'A'..=b'Z');
            v.extend(b'0'..=b'9');
            v.push(b' ');
            v.push(b'\n');
            v
        };

        let mut vocab_entries: Vec<String> = Vec::new();
        for &b in &safe_bytes {
            let ch = map[b as usize];
            // JSON-escape any char that looks like a quote or backslash
            let ch_str = if ch == '"' { "\\\"".to_string() } else if ch == '\\' { "\\\\".to_string() } else { ch.to_string() };
            vocab_entries.push(format!("\"{}\":{}", ch_str, b));
        }

        // Add merged token "ab" = id 256, "ab " = id 257 (space = Ġ)
        let a_char = map[b'a' as usize];
        let b_char = map[b'b' as usize];
        let sp_char = map[b' ' as usize];
        vocab_entries.push(format!("\"{}{}\":256", a_char, b_char));
        vocab_entries.push(format!("\"{}{}{}\":257", a_char, b_char, sp_char));

        let vocab_json = format!("{{{}}}", vocab_entries.join(","));
        let merges_txt = format!("#version: 0.2\n{} {}\n{} {}\n",
            a_char, b_char,     // merge 1: a + b → ab
            format!("{}{}", a_char, b_char), sp_char);  // merge 2: ab + Ġ → abĠ
        (vocab_json, merges_txt)
    }

    #[test]
    fn test_from_vocab_merges_encode_decode() {
        let (vocab_json, merges_txt) = make_tiny_vocab_merges();
        let tok = BpeTokenizer::from_vocab_and_merges_str(&vocab_json, &merges_txt)
            .expect("from_vocab_and_merges_str failed");

        // "ab " should encode to a single token (merge applied)
        let ids = tok.encode("ab ");
        let back = tok.decode(&ids);
        assert_eq!(back, "ab ", "round-trip failed: got {:?}", back);
    }

    #[test]
    fn test_from_vocab_merges_compression() {
        let (vocab_json, merges_txt) = make_tiny_vocab_merges();
        let tok = BpeTokenizer::from_vocab_and_merges_str(&vocab_json, &merges_txt)
            .expect("from_vocab_and_merges_str failed");

        // "ab ab ab" has 7 bytes but should compress with merges
        let ids = tok.encode("ab ab ab");
        // Each "ab " should become 1 token (id 257), leaving final "ab" as 1 token
        // → 3 + 1 = 4 tokens, much fewer than 8 bytes
        assert!(ids.len() < 8, "expected compression: {} tokens for 8 bytes", ids.len());
    }

    #[test]
    fn test_from_vocab_merges_vocab_size() {
        let (vocab_json, merges_txt) = make_tiny_vocab_merges();
        let tok = BpeTokenizer::from_vocab_and_merges_str(&vocab_json, &merges_txt)
            .expect("from_vocab_and_merges_str failed");
        // 256 byte tokens + 2 merged tokens = 258
        assert_eq!(tok.vocab_size(), 258);
    }

    // --- base64 decoder tests ---

    #[test]
    fn test_base64_decode_hello() {
        // "hello" in base64 = "aGVsbG8="
        let result = base64_decode("aGVsbG8=").unwrap();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_base64_decode_no_padding() {
        // Without padding
        let result = base64_decode("aGVsbG8").unwrap();
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_base64_decode_empty() {
        let result = base64_decode("").unwrap();
        assert!(result.is_empty());
    }

    // --- tiktoken format tests ---

    #[test]
    fn test_from_tiktoken_str_roundtrip() {
        // Build a minimal tiktoken file: bytes 0-255 (ranks 0-255) + 1 merge
        let mut lines: Vec<String> = Vec::new();
        // Single-byte tokens
        for b in 0u8..=255 {
            let b64 = base64_encode(&[b]);
            lines.push(format!("{} {}", b64, b));
        }
        // Merged token "ab" = bytes [97, 98], rank 256
        let b64_ab = base64_encode(b"ab");
        lines.push(format!("{} 256", b64_ab));
        let content = lines.join("\n");

        let tok = BpeTokenizer::from_tiktoken_str(&content).expect("tiktoken load failed");

        // "ab" should encode to 1 token
        let ids = tok.encode("ab");
        assert_eq!(ids.len(), 1, "expected 'ab' to merge to 1 token, got {:?}", ids);

        // Decode should give back "ab"
        let back = tok.decode(&ids);
        assert_eq!(back, "ab");
    }

    #[test]
    fn test_from_tiktoken_str_vocab_size() {
        let mut lines: Vec<String> = Vec::new();
        for b in 0u8..=255 {
            lines.push(format!("{} {}", base64_encode(&[b]), b));
        }
        let content = lines.join("\n");
        let tok = BpeTokenizer::from_tiktoken_str(&content).expect("tiktoken load failed");
        assert_eq!(tok.vocab_size(), 256);
    }

    // --- gpt2 byte encoding tests ---

    #[test]
    fn test_gpt2_decode_space() {
        // Ġ (U+0120) should decode to space (byte 0x20)
        let decoded = gpt2_decode_token_to_bytes("Ġ");
        assert_eq!(decoded, b" ", "Ġ should decode to space");
    }

    #[test]
    fn test_gpt2_decode_ascii_token() {
        // "hello" stays as "hello" (all printable ASCII)
        let decoded = gpt2_decode_token_to_bytes("hello");
        assert_eq!(decoded, b"hello");
    }

    #[test]
    fn test_gpt2_roundtrip() {
        let map = gpt2_bytes_to_unicode();
        // Every byte should round-trip through the mapping
        for b in 0u8..=255 {
            let encoded = map[b as usize].to_string();
            let decoded = gpt2_decode_token_to_bytes(&encoded);
            assert_eq!(decoded, vec![b], "byte {} round-trip failed", b);
        }
    }
}

// =============================================================================
// SentencePiece Unigram tokenizer
// =============================================================================
//
// ## Background: why unigram, not BPE?
//
// GPT-2 / GPT-4 use BPE (byte-pair encoding): a deterministic, greedy merge
// algorithm.  LLaMA, Gemma, Mistral, and most open-weight models use a
// SentencePiece *unigram language model* tokenizer instead.
//
// The unigram model assigns a log-probability to every possible segmentation
// of the input string, then uses the Viterbi algorithm to find the single best
// (= maximum-probability) segmentation.  This gives more linguistically
// motivated splits in many non-English languages.
//
// ## File format (.model)
//
// A SentencePiece .model file is a serialised protobuf (`ModelProto`).  To
// avoid pulling in a protobuf crate we support two loading paths:
//
//   1. `from_vocab_text(text)` — loads a plain-text format:
//         <piece>\t<score>
//      one entry per line (score is log-prob, e.g. -5.3).
//      Useful for testing and for models that export their vocab in text form.
//
//   2. `from_model_bytes(bytes)` — parses the raw protobuf binary used by
//      HuggingFace SentencePiece models.  We implement just enough of the
//      protobuf wire format to read `ModelProto.pieces[].{piece, score}`.
//      No external dependency needed.
//
// ## Encoding algorithm — Viterbi forward pass
//
//   For input string s of length N (in bytes):
//   best[i] = best log-prob to reach position i
//   back[i] = (start, piece_id) that achieved best[i]
//
//   For each end position i (1..=N):
//     For each piece p in the vocabulary:
//       if s[i-len(p)..i] == p.text:
//         score = best[i - len(p)] + p.log_prob
//         if score > best[i]: update best[i], back[i]
//
// ## Decode
//
//   Concatenate pieces, replacing the leading ▁ (U+2581, LOWER ONE EIGHTH BLOCK)
//   with a space.  The first ▁ at position 0 is dropped (SentencePiece adds it
//   to mark a word boundary but the input did not start with a space).

/// A single vocabulary entry in a SentencePiece unigram model.
#[derive(Clone, Debug)]
pub struct SentencePiece {
    /// The piece text (may contain ▁ = U+2581 for word-initial position)
    pub text: String,
    /// Log-probability assigned by the unigram language model
    pub log_prob: f32,
}

/// A SentencePiece unigram tokenizer.
///
/// Implements the same `Tokenizer` trait as `CharTokenizer` and
/// `BpeTokenizer` so it can be used as a drop-in replacement.
// =============================================================================
// Trie for fast SentencePiece lookup
// =============================================================================
//
// The naive Viterbi scans all V pieces for every end position → O(N · V).
// For LLaMA-3 (V = 128,256) and a 1024-token context this is ~130M comparisons.
//
// A trie reduces this to O(N · L) where L = max piece length (typically ≤ 32).
//
// ## Structure
//
// Each node stores a `HashMap<u8, usize>` pointing to child node indices.
// When a terminal piece ends at a node, the node stores `(piece_id, log_prob)`.
//
// ## Lookup
//
// To find all pieces that start at position `start` in `s`:
//   Walk the trie byte-by-byte starting at node 0 (root).
//   At each node, if the node has a terminal entry, yield it as a candidate.
//   If the next byte has no child in the trie, stop.
//
// This replaces the inner V-loop with a trie walk of at most L steps.

/// A single node in the piece trie.
struct TrieNode {
    /// child_byte → child_node_index in the trie Vec
    children: HashMap<u8, usize>,
    /// If a piece ends here: (piece_id, log_prob)
    terminal: Option<(u32, f32)>,
}

impl TrieNode {
    fn new() -> Self { TrieNode { children: HashMap::new(), terminal: None } }
}

pub struct SentencePieceTokenizer {
    /// Vocabulary table; index = token id
    pub pieces: Vec<SentencePiece>,
    /// `piece_text → token_id` for O(1) lookup
    pub piece_to_id: HashMap<String, u32>,
    /// Trie over piece bytes for O(N·L) Viterbi instead of O(N·V)
    trie: Vec<TrieNode>,
    /// Special token id for unknown bytes (id 0 by convention)
    unk_id: u32,
}

impl SentencePieceTokenizer {
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------

    /// Load from a plain-text vocab file: one `<piece>\t<score>` per line.
    ///
    /// This matches the output of `spm_export_vocab --output_format=tsv`.
    ///
    /// ```text
    /// <unk>   0
    /// <s>     0
    /// </s>    0
    /// ▁the    -2.4
    /// ▁of     -2.9
    /// ```
    pub fn from_vocab_text(text: &str) -> Result<Self, String> {
        let mut pieces = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() { continue; }
            // Split on last tab — piece text may itself contain tabs in theory
            let tab = line.rfind('\t').ok_or_else(|| {
                format!("line {}: expected tab-separated <piece>\\t<score>, got {:?}", lineno + 1, line)
            })?;
            let piece = line[..tab].to_string();
            let score: f32 = line[tab+1..].trim().parse().map_err(|e| {
                format!("line {}: cannot parse score {:?}: {}", lineno + 1, &line[tab+1..], e)
            })?;
            pieces.push(SentencePiece { text: piece, log_prob: score });
        }
        if pieces.is_empty() {
            return Err("empty vocabulary".into());
        }
        Ok(Self::from_pieces(pieces))
    }

    /// Load from the raw bytes of a SentencePiece `.model` protobuf file.
    ///
    /// We parse just the fields we need from `ModelProto`:
    ///   field 1 = trainer_spec (skip)
    ///   field 2 = normalizer_spec (skip)
    ///   field 3 = pieces[] → each has field 1=piece (string), field 2=score (float)
    ///
    /// No external crate required — we implement the wire-format subset here.
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self, String> {
        let pieces = proto_read_pieces(bytes)?;
        if pieces.is_empty() {
            return Err("no pieces found in protobuf".into());
        }
        Ok(Self::from_pieces(pieces))
    }

    /// Load from a `.model` file on disk.
    pub fn from_model_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("cannot read {}: {}", path, e))?;
        Self::from_model_bytes(&bytes)
    }

    fn from_pieces(pieces: Vec<SentencePiece>) -> Self {
        let mut piece_to_id = HashMap::new();
        for (i, p) in pieces.iter().enumerate() {
            piece_to_id.insert(p.text.clone(), i as u32);
        }
        let unk_id = piece_to_id.get("<unk>").copied().unwrap_or(0);

        // Build trie
        let trie = Self::build_trie(&pieces);

        SentencePieceTokenizer { pieces, piece_to_id, trie, unk_id }
    }

    /// Build a byte trie over all piece texts.
    ///
    /// Node 0 is the root. Each piece is inserted byte-by-byte;
    /// at the final byte of each piece we store `(piece_id, log_prob)`.
    fn build_trie(pieces: &[SentencePiece]) -> Vec<TrieNode> {
        let mut trie = vec![TrieNode::new()]; // node 0 = root
        for (id, piece) in pieces.iter().enumerate() {
            let bytes = piece.text.as_bytes();
            let mut node_idx = 0usize;
            for &b in bytes {
                let next = if let Some(&c) = trie[node_idx].children.get(&b) {
                    c
                } else {
                    let new_idx = trie.len();
                    trie[node_idx].children.insert(b, new_idx);
                    trie.push(TrieNode::new());
                    new_idx
                };
                node_idx = next;
            }
            // Only keep the entry with the highest log_prob if two pieces are identical
            if trie[node_idx].terminal.as_ref().map_or(true, |&(_, lp)| piece.log_prob > lp) {
                trie[node_idx].terminal = Some((id as u32, piece.log_prob));
            }
        }
        trie
    }

    /// Iterate over all pieces that start at byte position `start` in `s`.
    ///
    /// Uses the trie to walk at most `max_piece_len` steps instead of
    /// scanning all V pieces.
    ///
    /// Yields `(end_pos, piece_id, log_prob)` for each match found.
    fn trie_matches<'a>(&'a self, s: &'a [u8], start: usize) -> impl Iterator<Item=(usize, u32, f32)> + 'a {
        let mut node_idx = 0usize;
        let mut pos = start;
        let mut results = Vec::new();
        while pos < s.len() {
            let b = s[pos];
            match self.trie[node_idx].children.get(&b) {
                None => break,
                Some(&next) => {
                    node_idx = next;
                    pos += 1;
                    if let Some(&(id, lp)) = self.trie[node_idx].terminal.as_ref() {
                        results.push((pos, id, lp));
                    }
                }
            }
        }
        results.into_iter()
    }

    // -------------------------------------------------------------------------
    // Encoding (Viterbi forward pass)
    // -------------------------------------------------------------------------

    fn encode_bytes(&self, s: &[u8]) -> Vec<u32> {
        let n = s.len();
        if n == 0 { return Vec::new(); }

        const NEG_INF: f32 = f32::NEG_INFINITY;

        // best[i] = best log-prob for the prefix s[0..i]
        let mut best: Vec<f32> = vec![NEG_INF; n + 1];
        // back[i] = (start_pos, piece_id) that achieved best[i]
        let mut back: Vec<(usize, u32)> = vec![(0, self.unk_id); n + 1];
        best[0] = 0.0;

        // Trie-based Viterbi: for each start position, walk the trie to find
        // all pieces that begin there.  O(N · L) instead of O(N · V).
        for start in 0..n {
            if best[start] == NEG_INF { continue; }
            for (end, id, log_prob) in self.trie_matches(s, start) {
                let score = best[start] + log_prob;
                if score > best[end] {
                    best[end] = score;
                    back[end] = (start, id);
                }
            }
        }

        // Fallback pass: fill any positions still at NEG_INF with single-byte UNK
        for i in 1..=n {
            if best[i] == NEG_INF {
                best[i] = best[i - 1] + (-100.0);
                back[i] = (i - 1, self.unk_id);
            }
        }

        // Traceback
        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (start, id) = back[pos];
            ids.push(id);
            pos = start;
        }
        ids.reverse();
        ids
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn unk_id(&self) -> u32 { self.unk_id }

    fn encode(&self, text: &str) -> Vec<u32> {
        // Prepend ▁ (U+2581) to mark the start of the text, matching the
        // convention used during SentencePiece training.
        let mut s = String::with_capacity(text.len() + 3); // U+2581 is 3 UTF-8 bytes
        s.push('\u{2581}');
        s.push_str(text);
        // Replace spaces with ▁ (SentencePiece maps space → ▁ during normalisation)
        let normalised = s.replace(' ', "\u{2581}");
        self.encode_bytes(normalised.as_bytes())
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            let piece = if (id as usize) < self.pieces.len() {
                &self.pieces[id as usize].text
            } else {
                "?"
            };
            // ▁ (U+2581) → space, but drop the very first ▁ we added in encode
            out.push_str(piece);
        }
        // Replace ▁ with space, strip leading space introduced by normalisation
        let decoded = out.replace('\u{2581}', " ");
        decoded.trim_start_matches(' ').to_string()
    }

    fn vocab_size(&self) -> usize { self.pieces.len() }
}

// =============================================================================
// Minimal protobuf parser — just enough for SentencePiece ModelProto
// =============================================================================
//
// Protobuf wire format:
//   Each field is a (tag, wire_type) varint followed by the value.
//   tag = field_number << 3 | wire_type
//   wire_type 0 = varint, 1 = 64-bit, 2 = length-delimited, 5 = 32-bit
//
// ModelProto structure (only fields we care about):
//   field 3: repeated SentencePieceProto pieces = {
//     field 1: bytes piece
//     field 2: float score
//     field 3: SentencePieceType type (enum, varint — we skip)
//   }

fn proto_read_pieces(bytes: &[u8]) -> Result<Vec<SentencePiece>, String> {
    let mut cursor = 0usize;
    let mut pieces = Vec::new();

    while cursor < bytes.len() {
        let (tag_wire, adv) = proto_varint(bytes, cursor)?;
        cursor += adv;
        let field_number = tag_wire >> 3;
        let wire_type    = tag_wire & 0x7;

        match wire_type {
            0 => { // varint — skip
                let (_, adv) = proto_varint(bytes, cursor)?;
                cursor += adv;
            }
            1 => { // 64-bit — skip
                cursor += 8;
            }
            2 => { // length-delimited
                let (len, adv) = proto_varint(bytes, cursor)?;
                cursor += adv;
                let end = cursor + len as usize;
                if field_number == 1 {
                    // SentencePiece ModelProto: field 1 = repeated SentencePieceProto pieces
                    let piece = proto_read_one_piece(&bytes[cursor..end])?;
                    pieces.push(piece);
                }
                cursor = end;
            }
            5 => { // 32-bit — skip
                cursor += 4;
            }
            _ => return Err(format!("unknown wire type {} at offset {}", wire_type, cursor)),
        }
    }
    Ok(pieces)
}

fn proto_read_one_piece(bytes: &[u8]) -> Result<SentencePiece, String> {
    let mut cursor = 0usize;
    let mut text_opt: Option<String> = None;
    let mut score_opt: Option<f32> = None;

    while cursor < bytes.len() {
        let (tag_wire, adv) = proto_varint(bytes, cursor)?;
        cursor += adv;
        let field_number = tag_wire >> 3;
        let wire_type    = tag_wire & 0x7;

        match wire_type {
            0 => { let (_, adv) = proto_varint(bytes, cursor)?; cursor += adv; }
            1 => { cursor += 8; }
            2 => {
                let (len, adv) = proto_varint(bytes, cursor)?;
                cursor += adv;
                let end = cursor + len as usize;
                if field_number == 1 {
                    text_opt = Some(String::from_utf8_lossy(&bytes[cursor..end]).into_owned());
                }
                cursor = end;
            }
            5 => {
                // 32-bit little-endian float
                if cursor + 4 > bytes.len() {
                    return Err("truncated 32-bit field".into());
                }
                if field_number == 2 {
                    let raw = [bytes[cursor], bytes[cursor+1], bytes[cursor+2], bytes[cursor+3]];
                    score_opt = Some(f32::from_le_bytes(raw));
                }
                cursor += 4;
            }
            _ => return Err(format!("unknown wire type {} in SentencePieceProto", wire_type)),
        }
    }

    Ok(SentencePiece {
        text:     text_opt.unwrap_or_default(),
        log_prob: score_opt.unwrap_or(0.0),
    })
}

/// Read a protobuf varint from `bytes` starting at `offset`.
/// Returns `(value, bytes_consumed)`.
fn proto_varint(bytes: &[u8], offset: usize) -> Result<(u64, usize), String> {
    let mut result = 0u64;
    let mut shift  = 0u32;
    let mut i      = offset;
    loop {
        if i >= bytes.len() {
            return Err(format!("truncated varint at offset {}", i));
        }
        let b = bytes[i] as u64;
        result |= (b & 0x7f) << shift;
        i += 1;
        if b & 0x80 == 0 { break; }
        shift += 7;
        if shift >= 64 {
            return Err("varint too long".into());
        }
    }
    Ok((result, i - offset))
}

// =============================================================================
// SentencePiece tests
// =============================================================================

#[cfg(test)]
mod sentencepiece_tests {
    use super::*;

    /// Build a tiny vocabulary suitable for unit-testing.
    /// Uses simple ASCII pieces with uniform log-probs.
    fn tiny_vocab() -> SentencePieceTokenizer {
        // The ▁ prefix is U+2581 (3 UTF-8 bytes: 0xE2 0x96 0x81)
        let sp = '\u{2581}';
        let text = format!(
            "<unk>\t0\n\
             <s>\t0\n\
             </s>\t0\n\
             {sp}hello\t-1.0\n\
             {sp}world\t-2.0\n\
             {sp}hi\t-3.0\n\
             {sp}h\t-4.0\n\
             e\t-4.0\n\
             l\t-4.0\n\
             o\t-4.0\n\
             w\t-4.0\n\
             r\t-4.0\n\
             d\t-4.0\n\
             i\t-4.0\n"
        );
        SentencePieceTokenizer::from_vocab_text(&text).expect("tiny_vocab failed")
    }

    #[test]
    fn test_vocab_size() {
        let tok = tiny_vocab();
        assert_eq!(tok.vocab_size(), 14);
    }

    #[test]
    fn test_encode_hello_single_token() {
        let tok = tiny_vocab();
        // "hello" should encode as [▁hello] — one token
        let ids = tok.encode("hello");
        assert_eq!(ids.len(), 1,
            "expected 1 token for 'hello', got {:?}", ids);
    }

    #[test]
    fn test_decode_hello() {
        let tok = tiny_vocab();
        let ids = tok.encode("hello");
        let back = tok.decode(&ids);
        assert_eq!(back, "hello", "round-trip failed: {:?}", back);
    }

    #[test]
    fn test_encode_two_words() {
        // "hello world" → [▁hello, ▁world] — 2 tokens
        let tok = tiny_vocab();
        let ids = tok.encode("hello world");
        assert_eq!(ids.len(), 2,
            "expected 2 tokens for 'hello world', got {} tokens: {:?}", ids.len(), ids);
    }

    #[test]
    fn test_decode_two_words() {
        let tok = tiny_vocab();
        let ids = tok.encode("hello world");
        let back = tok.decode(&ids);
        assert_eq!(back, "hello world");
    }

    #[test]
    fn test_roundtrip_hi() {
        let tok = tiny_vocab();
        let ids = tok.encode("hi");
        let back = tok.decode(&ids);
        assert_eq!(back, "hi");
    }

    #[test]
    fn test_from_vocab_text_error_on_missing_tab() {
        let result = SentencePieceTokenizer::from_vocab_text("nospace\n");
        assert!(result.is_err(), "expected error for line without tab");
    }

    #[test]
    fn test_from_vocab_text_error_on_empty() {
        let result = SentencePieceTokenizer::from_vocab_text("");
        assert!(result.is_err(), "expected error for empty vocab");
    }

    #[test]
    fn test_piece_to_id_lookup() {
        let tok = tiny_vocab();
        let sp = '\u{2581}';
        let key = format!("{sp}hello");
        let id = tok.piece_to_id.get(&key).copied().unwrap();
        assert_eq!(id, 3, "▁hello should be id 3");
    }

    // --- Protobuf round-trip test ---
    // Build a minimal ModelProto binary by hand and verify we parse it correctly.

    fn encode_varint(v: u64) -> Vec<u8> {
        let mut out = Vec::new();
        let mut v = v;
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 { out.push(b); break; }
            out.push(b | 0x80);
        }
        out
    }

    fn encode_len_delimited(field: u64, data: &[u8]) -> Vec<u8> {
        let mut out = encode_varint((field << 3) | 2); // wire type 2
        out.extend(encode_varint(data.len() as u64));
        out.extend_from_slice(data);
        out
    }

    fn encode_f32_field(field: u64, v: f32) -> Vec<u8> {
        let mut out = encode_varint((field << 3) | 5); // wire type 5
        out.extend_from_slice(&v.to_le_bytes());
        out
    }

    fn make_model_proto(vocab: &[(&str, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (piece, score) in vocab {
            // Build SentencePieceProto {field 1: piece, field 2: score}
            let mut sp_proto = Vec::new();
            sp_proto.extend(encode_len_delimited(1, piece.as_bytes()));
            sp_proto.extend(encode_f32_field(2, *score));
            // Wrap as field 1 (pieces) of ModelProto — standard SentencePiece format
            out.extend(encode_len_delimited(1, &sp_proto));
        }
        out
    }

    #[test]
    fn test_proto_parse_roundtrip() {
        let sp = '\u{2581}';
        let piece_hello = format!("{sp}hello");
        let piece_world = format!("{sp}world");
        let vocab: Vec<(&str, f32)> = vec![
            ("<unk>", 0.0),
            ("<s>",   0.0),
            ("</s>",  0.0),
            (&piece_hello, -1.0),
            (&piece_world, -2.0),
        ];
        let bytes = make_model_proto(&vocab);
        let tok = SentencePieceTokenizer::from_model_bytes(&bytes)
            .expect("from_model_bytes failed");
        assert_eq!(tok.vocab_size(), 5);
        let sp_str = format!("{sp}hello");
        let id = tok.piece_to_id.get(&sp_str).copied().unwrap();
        assert_eq!(id, 3);
        assert!((tok.pieces[id as usize].log_prob - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_proto_encode_decode() {
        let sp = '\u{2581}';
        let sp_hello = format!("{sp}hello");
        let sp_world = format!("{sp}world");
        let sp_h = format!("{sp}h");
        let vocab: Vec<(&str, f32)> = vec![
            ("<unk>", 0.0),
            ("<s>",   0.0),
            ("</s>",  0.0),
            (&sp_hello, -1.0),
            (&sp_world, -2.0),
            (&sp_h, -4.0),
            ("e",  -4.0),
            ("l",  -4.0),
            ("o",  -4.0),
        ];
        let bytes = make_model_proto(&vocab);
        let tok = SentencePieceTokenizer::from_model_bytes(&bytes).unwrap();
        let ids = tok.encode("hello world");
        let back = tok.decode(&ids);
        assert_eq!(back, "hello world",
            "proto round-trip failed: {:?}", back);
    }
}

// ---- Base64 encoder (for tests only) ----
#[cfg(test)]
fn base64_encode(data: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[((n >> 18) & 63) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 63) as usize] as char);
        if chunk.len() > 1 { out.push(ALPHABET[((n >> 6) & 63) as usize] as char); } else { out.push('='); }
        if chunk.len() > 2 { out.push(ALPHABET[(n & 63) as usize] as char); } else { out.push('='); }
    }
    out
}

// =============================================================================
// HfBpeTokenizer — loads a HuggingFace tokenizer.json BPE tokenizer
// =============================================================================
//
// Supports the Gemma 3 tokenizer format:
//   - Normalizer:    replace ' ' with '▁' (U+2581)
//   - Pre-tokenizer: split on ' ', prepending '▁' to each word
//   - Model:         BPE with vocab (token→id) and ordered merge rules
//
// The tokenizer.json file is the canonical source; tokenizer.model is legacy.

/// A BPE tokenizer loaded from a HuggingFace `tokenizer.json` file.
pub struct HfBpeTokenizer {
    /// vocab[id] = token string
    id_to_token: Vec<String>,
    /// token string → id  (for encoding)
    token_to_id: HashMap<String, u32>,
    /// merge_rank[pair] = priority (lower = applied first)
    merge_rank: HashMap<(String, String), usize>,
    unk_id: u32,
    /// If true, use GPT-2 byte-level encoding (Qwen, GPT-2 style)
    /// If false, use SentencePiece ▁ encoding (Gemma style)
    byte_level: bool,
}

impl HfBpeTokenizer {
    /// Load from the raw JSON bytes of a `tokenizer.json` file.
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, String> {
        let s = std::str::from_utf8(bytes).map_err(|e| format!("utf8 error: {}", e))?;
        Self::from_json_str(s)
    }

    /// Load from the path to a `tokenizer.json` file.
    pub fn from_json_file(path: &str) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {}", path, e))?;
        Self::from_json_bytes(&bytes)
    }

    fn from_json_str(s: &str) -> Result<Self, String> {
        // Extract the "model" object
        let model_start = s.find("\"model\"").ok_or("missing \"model\" key")?;
        // Find the vocab object and merges array inside model
        let vocab_map = Self::parse_vocab(s)?;
        let merges = Self::parse_merges(s)?;

        // Build id_to_token (sorted by id)
        let vocab_size = vocab_map.len();
        let mut id_to_token = vec![String::new(); vocab_size];
        for (tok, &id) in &vocab_map {
            if (id as usize) < vocab_size {
                id_to_token[id as usize] = tok.clone();
            }
        }

        // Build merge_rank
        let mut merge_rank = HashMap::new();
        for (rank, pair) in merges.iter().enumerate() {
            merge_rank.insert(pair.clone(), rank);
        }

        let unk_id = vocab_map.get("<unk>").copied().unwrap_or(0);

        // Detect ByteLevel pre-tokenizer from the JSON
        let byte_level = s.contains("\"ByteLevel\"");

        Ok(HfBpeTokenizer { id_to_token, token_to_id: vocab_map, merge_rank, unk_id, byte_level })
    }

    /// Parse the "model"."vocab" object: returns token→id map.
    fn parse_vocab(s: &str) -> Result<HashMap<String, u32>, String> {
        // Find "vocab": { ... }
        let vocab_key = "\"vocab\"";
        let vocab_start = s.find(vocab_key).ok_or("missing vocab key")?;
        let brace = s[vocab_start..].find('{').ok_or("vocab not an object")? + vocab_start;
        let content = Self::extract_brace_block(s, brace)?;

        let mut map = HashMap::new();
        // Parse "token": id pairs
        let mut pos = 0;
        let bytes = content.as_bytes();
        while pos < bytes.len() {
            // Skip whitespace and commas
            while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n'
                || bytes[pos] == b'\r' || bytes[pos] == b'\t' || bytes[pos] == b',') {
                pos += 1;
            }
            if pos >= bytes.len() || bytes[pos] == b'}' { break; }
            if bytes[pos] != b'"' { pos += 1; continue; }
            // Read key string
            let (key, adv) = Self::parse_json_string(&content[pos..])?;
            pos += adv;
            // Skip whitespace and colon
            while pos < bytes.len() && bytes[pos] != b':' { pos += 1; }
            pos += 1; // skip colon
            while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n'
                || bytes[pos] == b'\r' || bytes[pos] == b'\t') {
                pos += 1;
            }
            // Read number
            let num_start = pos;
            while pos < bytes.len() && (bytes[pos].is_ascii_digit()) { pos += 1; }
            if pos > num_start {
                let id: u32 = content[num_start..pos].parse().unwrap_or(0);
                map.insert(key, id);
            }
        }
        Ok(map)
    }

    /// Parse the "model"."merges" array: returns list of (left, right) pairs.
    fn parse_merges(s: &str) -> Result<Vec<(String, String)>, String> {
        // Find "merges": [ ... ]
        let merges_key = "\"merges\"";
        let merges_start = s.find(merges_key).ok_or("missing merges key")?;
        let bracket = s[merges_start..].find('[').ok_or("merges not an array")? + merges_start;
        let content = Self::extract_bracket_block(s, bracket)?;

        let mut merges = Vec::new();
        let bytes = content.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n'
                || bytes[pos] == b'\r' || bytes[pos] == b'\t' || bytes[pos] == b',') {
                pos += 1;
            }
            if pos >= bytes.len() || bytes[pos] == b']' { break; }
            if bytes[pos] == b'[' {
                // Array format: ["left", "right"]
                pos += 1;
                while pos < bytes.len() && bytes[pos] != b'"' { pos += 1; }
                let (left, adv) = Self::parse_json_string(&content[pos..])?;
                pos += adv;
                while pos < bytes.len() && bytes[pos] != b'"' { pos += 1; }
                let (right, adv) = Self::parse_json_string(&content[pos..])?;
                pos += adv;
                while pos < bytes.len() && bytes[pos] != b']' { pos += 1; }
                pos += 1;
                merges.push((left, right));
            } else if bytes[pos] == b'"' {
                // String format: "left right"
                let (pair_str, adv) = Self::parse_json_string(&content[pos..])?;
                pos += adv;
                // Split on first space
                if let Some(sp) = pair_str.find(' ') {
                    merges.push((pair_str[..sp].to_string(), pair_str[sp+1..].to_string()));
                }
            } else {
                pos += 1;
            }
        }
        Ok(merges)
    }

    /// Extract the content of a `{...}` block starting at `start` (index of `{`).
    fn extract_brace_block(s: &str, start: usize) -> Result<String, String> {
        let bytes = s.as_bytes();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        for (i, &b) in bytes[start..].iter().enumerate() {
            if escape { escape = false; continue; }
            if b == b'\\' && in_string { escape = true; continue; }
            if b == b'"' { in_string = !in_string; continue; }
            if in_string { continue; }
            if b == b'{' { depth += 1; }
            if b == b'}' {
                depth -= 1;
                if depth == 0 {
                    return Ok(s[start + 1..start + i].to_string());
                }
            }
        }
        Err("unclosed brace block".into())
    }

    /// Extract the content of a `[...]` block starting at `start` (index of `[`).
    fn extract_bracket_block(s: &str, start: usize) -> Result<String, String> {
        let bytes = s.as_bytes();
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escape = false;
        for (i, &b) in bytes[start..].iter().enumerate() {
            if escape { escape = false; continue; }
            if b == b'\\' && in_string { escape = true; continue; }
            if b == b'"' { in_string = !in_string; continue; }
            if in_string { continue; }
            if b == b'[' { depth += 1; }
            if b == b']' {
                depth -= 1;
                if depth == 0 {
                    return Ok(s[start + 1..start + i].to_string());
                }
            }
        }
        Err("unclosed bracket block".into())
    }

    /// Parse a JSON string starting at `s[0] == '"'`. Returns (value, bytes_consumed).
    fn parse_json_string(s: &str) -> Result<(String, usize), String> {
        let bytes = s.as_bytes();
        if bytes.is_empty() || bytes[0] != b'"' {
            return Err(format!("expected '\"', got {:?}", &s[..s.len().min(4)]));
        }
        let mut out = String::new();
        let mut i = 1;
        while i < bytes.len() {
            if bytes[i] == b'\\' {
                i += 1;
                if i >= bytes.len() { break; }
                match bytes[i] {
                    b'"'  => out.push('"'),
                    b'\\' => out.push('\\'),
                    b'/'  => out.push('/'),
                    b'n'  => out.push('\n'),
                    b'r'  => out.push('\r'),
                    b't'  => out.push('\t'),
                    b'u'  => {
                        // \uXXXX
                        if i + 4 < bytes.len() {
                            let hex = &s[i+1..i+5];
                            if let Ok(cp) = u32::from_str_radix(hex, 16) {
                                if let Some(c) = char::from_u32(cp) { out.push(c); }
                            }
                            i += 4;
                        }
                    }
                    b => out.push(b as char),
                }
            } else if bytes[i] == b'"' {
                i += 1;
                break;
            } else {
                // Multi-byte UTF-8 passthrough
                let ch_len = {
                    let b0 = bytes[i];
                    if b0 < 0x80 { 1 }
                    else if b0 < 0xE0 { 2 }
                    else if b0 < 0xF0 { 3 }
                    else { 4 }
                };
                let end = (i + ch_len).min(bytes.len());
                if let Ok(c) = std::str::from_utf8(&bytes[i..end]) {
                    out.push_str(c);
                }
                i += ch_len;
                continue;
            }
            i += 1;
        }
        Ok((out, i))
    }

    /// GPT-2 byte-level encode: pre-tokenize → byte-to-unicode → BPE per word
    fn encode_byte_level(&self, text: &str) -> Vec<u32> {
        let b2u = gpt2_bytes_to_unicode();
        let words = Self::gpt2_pretokenize(text);
        let mut ids = Vec::new();
        for word in &words {
            // Convert each byte to GPT-2 unicode char
            let unicode_word: String = word.bytes().map(|b| b2u[b as usize]).collect();
            ids.extend(self.bpe_encode_word(&unicode_word));
        }
        ids
    }

    /// GPT-2 byte-level decode: token strings → bytes → UTF-8
    fn decode_byte_level(&self, ids: &[u32]) -> String {
        let b2u = gpt2_bytes_to_unicode();
        // Build inverse: char → byte
        let mut u2b: HashMap<char, u8> = HashMap::new();
        for (b, &c) in b2u.iter().enumerate() {
            u2b.insert(c, b as u8);
        }
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            if let Some(tok) = self.id_to_token.get(id as usize) {
                for c in tok.chars() {
                    if let Some(&b) = u2b.get(&c) {
                        bytes.push(b);
                    }
                    // Special tokens (like <|im_start|>) have chars not in the map — skip
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Simple GPT-2-style pre-tokenizer (splits text into words for BPE).
    /// Handles: letter sequences with optional leading space/punct, digits,
    /// punctuation, whitespace. No regex crate needed.
    fn gpt2_pretokenize(text: &str) -> Vec<&str> {
        let mut words: Vec<&str> = Vec::new();
        let chars: Vec<(usize, char)> = text.char_indices().collect();
        let mut i = 0;

        while i < chars.len() {
            let (byte_start, ch) = chars[i];

            // Case 1: Contractions ('s, 't, 're, 've, 'm, 'll, 'd)
            if ch == '\'' && i + 1 < chars.len() {
                let next_lower = chars[i + 1].1.to_ascii_lowercase();
                // Single-char contractions
                if matches!(next_lower, 's' | 't' | 'd' | 'm') {
                    let byte_end = if i + 2 < chars.len() { chars[i + 2].0 } else { text.len() };
                    words.push(&text[byte_start..byte_end]);
                    i += 2;
                    continue;
                }
                // Two-char contractions
                if i + 2 < chars.len() {
                    let c2 = chars[i + 2].1.to_ascii_lowercase();
                    let pair = (next_lower, c2);
                    if matches!(pair, ('r', 'e') | ('v', 'e') | ('l', 'l')) {
                        let byte_end = if i + 3 < chars.len() { chars[i + 3].0 } else { text.len() };
                        words.push(&text[byte_start..byte_end]);
                        i += 3;
                        continue;
                    }
                }
            }

            // Case 2: Optional non-letter-digit char + letter sequence
            if ch.is_alphabetic() || ch == '_' {
                // Just letters
                let mut j = i + 1;
                while j < chars.len() && (chars[j].1.is_alphabetic() || chars[j].1 == '_') { j += 1; }
                let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i = j;
                continue;
            }
            // Non-letter-digit followed by letters: group them (e.g., " is" → one word)
            if !ch.is_alphanumeric() && ch != '\r' && ch != '\n'
                && i + 1 < chars.len() && chars[i + 1].1.is_alphabetic()
            {
                let mut j = i + 1;
                while j < chars.len() && (chars[j].1.is_alphabetic() || chars[j].1 == '_') { j += 1; }
                let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i = j;
                continue;
            }

            // Case 3: Single digit
            if ch.is_ascii_digit() {
                let byte_end = if i + 1 < chars.len() { chars[i + 1].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i += 1;
                continue;
            }

            // Case 4: Newlines
            if ch == '\r' || ch == '\n' {
                let mut j = i + 1;
                while j < chars.len() && (chars[j].1 == '\r' || chars[j].1 == '\n') { j += 1; }
                let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i = j;
                continue;
            }

            // Case 5: Whitespace run (not newlines)
            if ch.is_whitespace() {
                let mut j = i + 1;
                while j < chars.len() && chars[j].1.is_whitespace()
                    && chars[j].1 != '\r' && chars[j].1 != '\n' { j += 1; }
                let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i = j;
                continue;
            }

            // Case 6: Optional leading space + punctuation/symbols
            if ch == ' ' && i + 1 < chars.len() && !chars[i + 1].1.is_alphanumeric()
                && !chars[i + 1].1.is_whitespace()
            {
                let mut j = i + 1;
                while j < chars.len() && !chars[j].1.is_alphanumeric()
                    && !chars[j].1.is_whitespace() { j += 1; }
                let byte_end = if j < chars.len() { chars[j].0 } else { text.len() };
                words.push(&text[byte_start..byte_end]);
                i = j;
                continue;
            }

            // Case 7: Single character (punctuation, symbol, etc.)
            let byte_end = if i + 1 < chars.len() { chars[i + 1].0 } else { text.len() };
            words.push(&text[byte_start..byte_end]);
            i += 1;
        }

        words
    }

    /// Encode a single pre-tokenized word (already has ▁ prefix) using BPE.
    fn bpe_encode_word(&self, word: &str) -> Vec<u32> {
        if word.is_empty() { return Vec::new(); }

        // Initialize: each Unicode char is a symbol
        let chars: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if chars.is_empty() { return Vec::new(); }

        // Use index-based representation for fast merging
        // symbols[i] = Some(token_str), None means merged away
        let mut symbols: Vec<Option<String>> = chars.into_iter().map(Some).collect();

        // Iteratively apply the highest-priority merge
        loop {
            let mut best_rank = usize::MAX;
            let mut best_i = usize::MAX;
            let mut best_j = usize::MAX;

            // Find all adjacent (non-None) pairs
            let indices: Vec<usize> = symbols.iter().enumerate()
                .filter_map(|(i, s)| if s.is_some() { Some(i) } else { None })
                .collect();

            for w in indices.windows(2) {
                let (i, j) = (w[0], w[1]);
                let left  = symbols[i].as_ref().unwrap();
                let right = symbols[j].as_ref().unwrap();
                if let Some(&rank) = self.merge_rank.get(&(left.clone(), right.clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_i = i;
                        best_j = j;
                    }
                }
            }

            if best_rank == usize::MAX { break; } // no more merges possible

            // Apply the best merge: concatenate symbols[best_i] and symbols[best_j]
            let left  = symbols[best_i].take().unwrap();
            let right = symbols[best_j].take().unwrap();
            symbols[best_i] = Some(left + &right);
            // symbols[best_j] remains None
        }

        // Collect the remaining (non-None) symbols and look up their IDs
        symbols.into_iter().flatten().map(|tok| {
            self.token_to_id.get(&tok).copied().unwrap_or(self.unk_id)
        }).collect()
    }
}

impl Tokenizer for HfBpeTokenizer {
    fn unk_id(&self) -> u32 { self.unk_id }

    fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() { return Vec::new(); }

        if self.byte_level {
            return self.encode_byte_level(text);
        }

        // SentencePiece style: replace spaces with ▁
        let normalized = text.replace(' ', "\u{2581}");

        // Split into words at ▁ boundaries (keep ▁ as prefix of each word)
        let mut words: Vec<String> = Vec::new();
        let mut current = String::new();
        for ch in normalized.chars() {
            if ch == '\u{2581}' && !current.is_empty() {
                words.push(current.clone());
                current = String::from('\u{2581}');
            } else {
                current.push(ch);
            }
        }
        if !current.is_empty() { words.push(current); }

        let mut ids = Vec::new();
        for word in &words {
            ids.extend(self.bpe_encode_word(word));
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        if self.byte_level {
            return self.decode_byte_level(ids);
        }
        let mut out = String::new();
        for &id in ids {
            let tok = if (id as usize) < self.id_to_token.len() {
                &self.id_to_token[id as usize]
            } else {
                continue;
            };
            out.push_str(tok);
        }
        out.replace('\u{2581}', " ")
    }

    fn vocab_size(&self) -> usize { self.id_to_token.len() }
}
