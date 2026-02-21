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
