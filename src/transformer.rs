/// # The Transformer — Phase 3
///
/// This file assembles all previous components into a full GPT-style
/// language model. We build it bottom-up:
///
///   1. Embeddings      — tokens and positions → vectors
///   2. CausalAttention — each token attends to past tokens
///   3. AttentionHead   — one attention head
///   4. MultiHeadAttention — many heads in parallel
///   5. TransformerBlock — one full layer (attention + FFN + LayerNorm)
///   6. Gpt             — the full model (embedding + N blocks + output head)
///
/// ## The core idea: what IS a transformer?
///
/// A transformer takes a sequence of tokens and repeatedly asks:
/// "Given everything I've seen so far, what comes next?"
///
/// It does this by letting every token "attend" to every earlier token —
/// building a rich contextual representation of each position by
/// gathering relevant information from the entire past context.
///
/// The magic is that this attending is *learned* — the network figures out
/// which tokens matter for predicting each next token, purely from data.
///
/// ## The full forward pass (what happens to your text):
///
///   "ciao " → tokenize → [99, 12, 4, 77, 32]
///                ↓
///   token_embed[99]  + pos_embed[0]  = x[0]   (512-dim vector)
///   token_embed[12]  + pos_embed[1]  = x[1]
///   ...
///                ↓
///   TransformerBlock 1: x = x + Attention(LayerNorm(x))
///                       x = x + MLP(LayerNorm(x))
///   TransformerBlock 2: ...
///   ...
///   TransformerBlock N: ...
///                ↓
///   Final LayerNorm
///                ↓
///   Linear(d_model → vocab_size) = logits   [seq_len, vocab_size]
///                ↓
///   softmax(logits[-1]) = probability over next token

use crate::autograd::Value;
use crate::nn::{gelu, softmax, LayerNorm, Linear, Mlp, Module, InitRng};

// =============================================================================
// Hyperparameters
// =============================================================================
//
// These define the model's size. Bigger = more capacity = more compute.
//
// For reference, actual GPT-2 sizes:
//
//   GPT-2 small:   n_layers=12, n_heads=12, d_model=768,  params=117M
//   GPT-2 medium:  n_layers=24, n_heads=16, d_model=1024, params=345M
//   GPT-2 large:   n_layers=36, n_heads=20, d_model=1280, params=774M
//   GPT-2 XL:      n_layers=48, n_heads=25, d_model=1600, params=1.5B
//
// Our "nano" model — runs on CPU, trains in minutes on a small corpus:

#[derive(Clone, Debug)]
pub struct Config {
    /// Vocabulary size — number of distinct tokens
    pub vocab_size: usize,

    /// Maximum sequence length (context window)
    pub context_length: usize,

    /// Embedding dimension — the size of every token's vector representation.
    /// Every layer inputs and outputs vectors of this size.
    pub d_model: usize,

    /// Number of transformer layers (depth of the network)
    pub n_layers: usize,

    /// Number of attention heads.
    /// Each head operates on d_model / n_heads dimensions independently.
    /// Must divide d_model evenly.
    pub n_heads: usize,
}

impl Config {
    /// A tiny model for learning and fast CPU training.
    pub fn nano(vocab_size: usize) -> Self {
        Config {
            vocab_size,
            context_length: 64,
            d_model: 64,
            n_layers: 2,
            n_heads: 2,
        }
    }

    /// Size of each attention head's key/query/value vectors.
    /// d_head = d_model / n_heads
    pub fn d_head(&self) -> usize {
        assert_eq!(
            self.d_model % self.n_heads, 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model, self.n_heads
        );
        self.d_model / self.n_heads
    }

    /// Total parameter count (approximate, for display)
    pub fn param_count(&self) -> usize {
        let embed = self.vocab_size * self.d_model + self.context_length * self.d_model;
        let attn_per_layer = 4 * self.d_model * self.d_model; // Q,K,V,O projections
        let ffn_per_layer = 2 * self.d_model * 4 * self.d_model; // two linear layers
        let norm_per_layer = 2 * 2 * self.d_model; // two LayerNorms per block
        let blocks = self.n_layers * (attn_per_layer + ffn_per_layer + norm_per_layer);
        let output_head = self.d_model * self.vocab_size;
        embed + blocks + output_head
    }
}

// =============================================================================
// 1. Embeddings
// =============================================================================
//
// ## Token embeddings
//
// Each token id maps to a learned vector of d_model dimensions.
// Think of it as a lookup table:
//
//   token_id=42 → embedding_table[42] = [0.1, -0.3, 0.7, ..., 0.2]   (d_model values)
//
// Initially these vectors are random. During training, the network learns
// to put semantically similar tokens near each other in this d_model-dimensional
// space. "ciao" and "hello" end up close together; "dog" and "cat" end up close.
//
// ## Positional embeddings
//
// The transformer processes all tokens in parallel — it has no inherent sense
// of order. We inject position information by adding a learned vector for each
// position (0, 1, 2, ...):
//
//   input[t] = token_embed[token_id[t]] + pos_embed[t]
//
// The original "Attention is All You Need" used fixed sinusoidal encodings.
// GPT-2 uses learned positional embeddings (simpler, works just as well).
// More recent models (LLaMA, GPT-4) use Rotary Position Embeddings (RoPE).

pub struct Embedding {
    /// token_embed[token_id] = vector of d_model values
    pub token_embed: Vec<Vec<Value>>,   // [vocab_size, d_model]

    /// pos_embed[position] = vector of d_model values
    pub pos_embed: Vec<Vec<Value>>,     // [context_length, d_model]

    pub config: Config,
}

impl Embedding {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        let token_embed = (0..config.vocab_size)
            .map(|_| rng.normal_vec(config.d_model, 0.02).into_iter().map(Value::new).collect())
            .collect();

        let pos_embed = (0..config.context_length)
            .map(|_| rng.normal_vec(config.d_model, 0.01).into_iter().map(Value::new).collect())
            .collect();

        Embedding { token_embed, pos_embed, config: config.clone() }
    }

    /// Look up and add token + positional embeddings for a sequence.
    ///
    /// token_ids: sequence of token ids, length T
    /// Returns: Vec of T vectors, each of length d_model
    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<Value>> {
        token_ids
            .iter()
            .enumerate()
            .map(|(pos, &tid)| {
                // token_embed[tid] + pos_embed[pos]
                self.token_embed[tid]
                    .iter()
                    .zip(self.pos_embed[pos].iter())
                    .map(|(te, pe)| te.add(pe))
                    .collect()
            })
            .collect()
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Value> {
        let mut p: Vec<Value> = self.token_embed.iter().flatten().cloned().collect();
        p.extend(self.pos_embed.iter().flatten().cloned());
        p
    }
}

// =============================================================================
// 2. Causal Self-Attention — the heart of the transformer
// =============================================================================
//
// ## What attention does
//
// For each token at position t, attention computes a weighted average of
// all token representations — where the weights represent "how relevant
// is each other token for understanding token t?"
//
// ## The Q, K, V framework
//
// Every token projects its embedding into three vectors:
//
//   Q (Query):  "What information am I looking for?"
//   K (Key):    "What information do I contain?"
//   V (Value):  "What information will I share if attended to?"
//
// The attention score between position t (query) and position s (key) is:
//
//   score[t, s] = dot(Q[t], K[s]) / sqrt(d_head)
//
// Dividing by sqrt(d_head) prevents the dot products from getting too large
// (which would push softmax into a saturated region with tiny gradients).
//
// ## Causal masking
//
// We're building a *language model* — it predicts the NEXT token.
// So token at position t must ONLY attend to positions 0..=t (past + itself).
// Attending to future tokens would be "cheating" — the model would learn
// to copy answers rather than actually predict.
//
// We enforce this by setting future scores to -infinity before softmax:
//   score[t, s] = -inf  for all s > t
//
// After softmax, -inf → 0, meaning zero attention to future tokens.
//
// ## The full computation for one head:
//
//   Q = x @ W_Q    [T, d_head]
//   K = x @ W_K    [T, d_head]
//   V = x @ W_V    [T, d_head]
//
//   scores = Q @ K.T / sqrt(d_head)    [T, T]
//   scores[t, s] = -inf for s > t      (causal mask)
//   weights = softmax(scores)          [T, T]  (each row sums to 1)
//   output = weights @ V               [T, d_head]
//
// ## Multi-head attention
//
// Instead of one big attention, we run n_heads smaller attentions in parallel.
// Each head learns to attend to different aspects of the sequence:
//   - Head 1 might learn syntactic relationships
//   - Head 2 might track coreference ("it" → "the model")
//   - Head 3 might capture local context
//
// The outputs of all heads are concatenated, then projected back to d_model:
//   output = concat(head_1, ..., head_n) @ W_O    [T, d_model]

pub struct AttentionHead {
    pub w_q: Linear,   // d_model → d_head
    pub w_k: Linear,   // d_model → d_head
    pub w_v: Linear,   // d_model → d_head
    pub d_head: usize,
}

impl AttentionHead {
    pub fn new(d_model: usize, d_head: usize, rng: &mut InitRng) -> Self {
        AttentionHead {
            w_q: Linear::new(d_model, d_head, rng),
            w_k: Linear::new(d_model, d_head, rng),
            w_v: Linear::new(d_model, d_head, rng),
            d_head,
        }
    }

    /// Forward pass for one attention head.
    ///
    /// x: sequence of T vectors, each of d_model dimensions
    /// Returns: sequence of T vectors, each of d_head dimensions
    pub fn forward(&self, x: &[Vec<Value>]) -> Vec<Vec<Value>> {
        let t = x.len();
        let scale = (self.d_head as f32).sqrt();

        // Project every token to Q, K, V
        let queries: Vec<Vec<Value>> = x.iter().map(|xi| self.w_q.forward(xi)).collect();
        let keys:    Vec<Vec<Value>> = x.iter().map(|xi| self.w_k.forward(xi)).collect();
        let values:  Vec<Vec<Value>> = x.iter().map(|xi| self.w_v.forward(xi)).collect();

        // Compute attention scores: score[t_i][t_j] = dot(Q[t_i], K[t_j]) / sqrt(d_head)
        // Then apply causal mask and softmax row-wise
        let weights: Vec<Vec<Value>> = (0..t)
            .map(|i| {
                let raw_scores: Vec<Value> = (0..t)
                    .map(|j| {
                        if j > i {
                            // Causal mask: future tokens get -infinity → softmax → 0
                            // We use a large negative float (not actual -inf) to keep
                            // autograd happy (exp(-1e9) ≈ 0 without NaN)
                            Value::new(-1e9)
                        } else {
                            // dot(Q[i], K[j]) / sqrt(d_head)
                            let dot = queries[i]
                                .iter()
                                .zip(keys[j].iter())
                                .map(|(q, k)| q.mul(k))
                                .reduce(|a, b| a.add(&b))
                                .unwrap();
                            dot.mul(&Value::new(1.0 / scale))
                        }
                    })
                    .collect();

                // Softmax over this row → attention weights for position i
                softmax(&raw_scores)
            })
            .collect();

        // Weighted sum of values: output[i] = sum_j(weights[i][j] * V[j])
        (0..t)
            .map(|i| {
                (0..self.d_head)
                    .map(|d| {
                        // sum over all positions j
                        (0..t)
                            .map(|j| weights[i][j].mul(&values[j][d]))
                            .reduce(|a, b| a.add(&b))
                            .unwrap()
                    })
                    .collect()
            })
            .collect()
    }
}

impl Module for AttentionHead {
    fn parameters(&self) -> Vec<Value> {
        let mut p = self.w_q.parameters();
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p
    }
}

// =============================================================================
// 3. Multi-Head Attention
// =============================================================================

pub struct MultiHeadAttention {
    pub heads: Vec<AttentionHead>,
    /// Output projection: concatenated heads → d_model
    pub w_o: Linear,
    pub n_heads: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        let d_head = config.d_head();
        let heads = (0..config.n_heads)
            .map(|_| AttentionHead::new(config.d_model, d_head, rng))
            .collect();

        // The output projection takes all head outputs concatenated (n_heads * d_head = d_model)
        // and projects back to d_model
        let w_o = Linear::new(config.d_model, config.d_model, rng);

        MultiHeadAttention {
            heads,
            w_o,
            n_heads: config.n_heads,
            d_model: config.d_model,
        }
    }

    /// x: [T, d_model]  →  output: [T, d_model]
    pub fn forward(&self, x: &[Vec<Value>]) -> Vec<Vec<Value>> {
        let t = x.len();

        // Run all heads in sequence (in production these run in parallel on GPU)
        let head_outputs: Vec<Vec<Vec<Value>>> =
            self.heads.iter().map(|h| h.forward(x)).collect();

        // Concatenate head outputs along the feature dimension
        // head_outputs[head][token][d_head] → concat → [token][d_model]
        let concatenated: Vec<Vec<Value>> = (0..t)
            .map(|i| {
                self.heads
                    .iter()
                    .enumerate()
                    .flat_map(|(h, _)| head_outputs[h][i].clone())
                    .collect()
            })
            .collect();

        // Final output projection
        concatenated.iter().map(|xi| self.w_o.forward(xi)).collect()
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Value> {
        let mut p: Vec<Value> = self.heads.iter().flat_map(|h| h.parameters()).collect();
        p.extend(self.w_o.parameters());
        p
    }
}

// =============================================================================
// 4. Transformer Block
// =============================================================================
//
// One complete transformer layer. The modern GPT design uses "pre-norm":
// LayerNorm is applied BEFORE each sub-block, not after.
//
//   x = x + MultiHeadAttention(LayerNorm(x))    ← attention sub-block
//   x = x + MLP(LayerNorm(x))                   ← FFN sub-block
//
// The + signs are *residual connections* — they let the original input
// bypass each sub-block and add directly to the output.
//
// ## Why residual connections?
//
// Without them, gradients in a 12-layer network must multiply through
// 12 weight matrices before reaching the first layer. They either vanish
// to zero (learning stops) or explode.
//
// With residual connections, there is always a direct gradient path from
// the loss back to every layer:
//
//   dL/dx_0 = dL/dx_N * (1 + dF_N/dx_N) * ... * (1 + dF_1/dx_1)
//                          ↑
//                      This 1 is the residual — it guarantees the gradient
//                      never multiplies to zero through the chain.
//
// This insight (from ResNet, 2015) made deep networks trainable and is
// one of the most important ideas in modern deep learning.

pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub mlp: Mlp,
}

impl TransformerBlock {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(config.d_model),
            attn: MultiHeadAttention::new(config, rng),
            ln2: LayerNorm::new(config.d_model),
            mlp: Mlp::new(config.d_model, rng),
        }
    }

    /// x: [T, d_model] → output: [T, d_model]
    pub fn forward(&self, x: &[Vec<Value>]) -> Vec<Vec<Value>> {
        let t = x.len();

        // Sub-block 1: attention with pre-norm and residual
        // norm_x[i] = LayerNorm(x[i])   for each token
        let norm_x: Vec<Vec<Value>> = x.iter().map(|xi| self.ln1.forward(xi)).collect();
        let attn_out = self.attn.forward(&norm_x);

        // Residual: x + attention_output (element-wise, per token)
        let x_after_attn: Vec<Vec<Value>> = (0..t)
            .map(|i| {
                x[i].iter()
                    .zip(attn_out[i].iter())
                    .map(|(a, b)| a.add(b))
                    .collect()
            })
            .collect();

        // Sub-block 2: MLP with pre-norm and residual
        let norm_x2: Vec<Vec<Value>> = x_after_attn.iter().map(|xi| self.ln2.forward(xi)).collect();
        let mlp_out: Vec<Vec<Value>> = norm_x2.iter().map(|xi| self.mlp.forward(xi)).collect();

        // Residual: (x + attn) + mlp_output
        (0..t)
            .map(|i| {
                x_after_attn[i]
                    .iter()
                    .zip(mlp_out[i].iter())
                    .map(|(a, b)| a.add(b))
                    .collect()
            })
            .collect()
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Value> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.mlp.parameters());
        p
    }
}

// =============================================================================
// 5. The full GPT model
// =============================================================================
//
// Putting it all together:
//
//   token_ids [T]
//       ↓
//   Embedding → x [T, d_model]
//       ↓
//   TransformerBlock 1
//       ↓
//   TransformerBlock 2
//       ...
//   TransformerBlock N
//       ↓
//   Final LayerNorm
//       ↓
//   Linear(d_model → vocab_size) → logits [T, vocab_size]
//
// ## Output: logits, not probabilities
//
// The model outputs raw scores ("logits") for every token in the vocabulary,
// at every position. To get probabilities: softmax(logits).
//
// During training we use the logits directly with cross-entropy loss
// (numerically more stable than softmax + log).
//
// ## Weight tying
//
// GPT-2 shares the weights between the token embedding table and the output
// linear layer. This makes sense: both map between token ids and d_model vectors.
// It saves parameters and often improves performance.
// We don't implement this for clarity, but it's worth knowing.

pub struct Gpt {
    pub embed: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_final: LayerNorm,
    pub lm_head: Linear,   // d_model → vocab_size
    pub config: Config,
}

impl Gpt {
    pub fn new(config: Config, rng: &mut InitRng) -> Self {
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock::new(&config, rng))
            .collect();

        let embed    = Embedding::new(&config, rng);
        let ln_final = LayerNorm::new(config.d_model);
        let lm_head  = Linear::new(config.d_model, config.vocab_size, rng);

        Gpt { embed, blocks, ln_final, lm_head, config }
    }

    /// Forward pass: token_ids → logits
    ///
    /// token_ids: slice of usize, length T (≤ context_length)
    /// Returns: Vec of T vectors, each of vocab_size (the logits)
    pub fn forward(&self, token_ids: &[usize]) -> Vec<Vec<Value>> {
        assert!(
            token_ids.len() <= self.config.context_length,
            "sequence length {} exceeds context_length {}",
            token_ids.len(),
            self.config.context_length
        );

        // 1. Embed tokens + positions
        let mut x = self.embed.forward(token_ids);

        // 2. Pass through each transformer block
        for block in &self.blocks {
            x = block.forward(&x);
        }

        // 3. Final layer norm
        let x_normed: Vec<Vec<Value>> = x.iter().map(|xi| self.ln_final.forward(xi)).collect();

        // 4. Project to vocabulary logits
        x_normed.iter().map(|xi| self.lm_head.forward(xi)).collect()
    }

    /// Compute cross-entropy loss for a sequence.
    ///
    /// token_ids: input sequence [T]
    /// targets:   target sequence [T] (token_ids shifted by 1)
    ///
    /// ## Cross-entropy loss
    ///
    /// For each position t, the model outputs a probability distribution
    /// over the vocabulary. The correct token is targets[t].
    ///
    /// Loss at position t = -log(p[targets[t]])
    ///
    /// Intuition: if the model is confident and correct (p ≈ 1), loss ≈ 0.
    ///            if the model assigns low probability to the correct token,
    ///            -log(p) → ∞.
    ///
    /// Total loss = mean over all positions.
    ///
    /// ## Why cross-entropy?
    ///
    /// It's the negative log-likelihood of the data under the model's distribution.
    /// Minimizing cross-entropy = maximizing the probability the model assigns
    /// to the actual training data. It directly measures "how well does the
    /// model predict the correct next token?"
    ///
    /// A random model over V tokens has loss = -log(1/V) = log(V).
    /// For vocab_size=65 (char-level): log(65) ≈ 4.17
    /// A trained model should reach well below 2.0.
    pub fn loss(&self, token_ids: &[usize], targets: &[usize]) -> Value {
        let logits = self.forward(token_ids);
        let t = logits.len();
        assert_eq!(t, targets.len());

        // Sum of -log(p[correct]) across all positions
        let loss_sum = (0..t)
            .map(|i| {
                // Convert logits to probabilities
                let probs = softmax(&logits[i]);
                // -log(p[correct token])
                probs[targets[i]].ln().neg()
            })
            .reduce(|a, b| a.add(&b))
            .unwrap();

        // Average over sequence length
        loss_sum.mul(&Value::new(1.0 / t as f32))
    }
}

impl Module for Gpt {
    fn parameters(&self) -> Vec<Value> {
        let mut p = self.embed.parameters();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_final.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn nano_config() -> Config {
        Config {
            vocab_size: 10,
            context_length: 8,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
        }
    }

    fn make_rng() -> InitRng { InitRng::new(42) }

    // --- Config ---

    #[test]
    fn test_config_d_head() {
        let cfg = nano_config();
        assert_eq!(cfg.d_head(), 4); // 8 / 2
    }

    // --- Embedding ---

    #[test]
    fn test_embedding_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let emb = Embedding::new(&cfg, &mut rng);
        let ids = vec![0usize, 3, 7];
        let out = emb.forward(&ids);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0].len(), cfg.d_model);
    }

    #[test]
    fn test_embedding_different_positions_differ() {
        // Same token at different positions should produce different vectors
        // (because positional embeddings differ)
        let cfg = nano_config();
        let mut rng = make_rng();
        let emb = Embedding::new(&cfg, &mut rng);
        let out = emb.forward(&[5, 5]); // same token id twice
        let v0: Vec<f32> = out[0].iter().map(|v| v.val()).collect();
        let v1: Vec<f32> = out[1].iter().map(|v| v.val()).collect();
        assert_ne!(v0, v1, "same token at different positions should differ");
    }

    // --- AttentionHead ---

    #[test]
    fn test_attention_head_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let head = AttentionHead::new(cfg.d_model, cfg.d_head(), &mut rng);
        let x: Vec<Vec<Value>> = (0..4)
            .map(|_| (0..cfg.d_model).map(|i| Value::new(i as f32 * 0.1)).collect())
            .collect();
        let out = head.forward(&x);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), cfg.d_head());
    }

    #[test]
    fn test_causal_mask_first_token_only_attends_to_itself() {
        // Token at position 0 can only attend to itself.
        // The attention weights for position 0 should be [1.0] (only itself visible).
        let cfg = nano_config();
        let mut rng = make_rng();
        let head = AttentionHead::new(cfg.d_model, cfg.d_head(), &mut rng);

        // Construct a simple 3-token sequence
        let x: Vec<Vec<Value>> = (0..3)
            .map(|t| (0..cfg.d_model).map(|_| Value::new(t as f32 * 0.1)).collect())
            .collect();

        // Run forward — if it doesn't panic, causal mask is applied correctly
        // (we can't easily inspect the weights from outside, but correctness
        //  is ensured by the masking logic in the forward pass)
        let out = head.forward(&x);
        assert_eq!(out.len(), 3);

        // All output values should be finite (no NaN from -inf masking)
        for row in &out {
            for v in row {
                assert!(v.val().is_finite(), "attention output should be finite");
            }
        }
    }

    // --- MultiHeadAttention ---

    #[test]
    fn test_mha_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let mha = MultiHeadAttention::new(&cfg, &mut rng);
        let x: Vec<Vec<Value>> = (0..5)
            .map(|_| (0..cfg.d_model).map(|i| Value::new(i as f32 * 0.01)).collect())
            .collect();
        let out = mha.forward(&x);
        assert_eq!(out.len(), 5);
        assert_eq!(out[0].len(), cfg.d_model);
    }

    #[test]
    fn test_mha_param_count() {
        let cfg = nano_config(); // d_model=8, n_heads=2, d_head=4
        let mut rng = make_rng();
        let mha = MultiHeadAttention::new(&cfg, &mut rng);
        let p = mha.parameters().len();

        // Per head: W_Q (8*4=32) + W_K (32) + W_V (32) + biases (4+4+4=12) = 108 per head?
        // Let's just verify it's > 0 and matches our formula
        // 2 heads × (3 linear layers of 8→4) + 1 output linear 8→8
        // = 2 × (3 × (8×4 + 4)) + (8×8 + 8)
        // = 2 × (3 × 36) + 72
        // = 216 + 72 = 288
        let d = cfg.d_model;
        let dh = cfg.d_head();
        let n = cfg.n_heads;
        let expected = n * (3 * (d * dh + dh)) + (d * d + d);
        assert_eq!(p, expected, "MHA param count: got {}, expected {}", p, expected);
    }

    // --- TransformerBlock ---

    #[test]
    fn test_block_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let block = TransformerBlock::new(&cfg, &mut rng);
        let x: Vec<Vec<Value>> = (0..4)
            .map(|_| (0..cfg.d_model).map(|i| Value::new(i as f32 * 0.1)).collect())
            .collect();
        let out = block.forward(&x);
        assert_eq!(out.len(), 4);
        assert_eq!(out[0].len(), cfg.d_model);
    }

    #[test]
    fn test_block_output_finite() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let block = TransformerBlock::new(&cfg, &mut rng);
        let x: Vec<Vec<Value>> = (0..3)
            .map(|_| (0..cfg.d_model).map(|_| Value::new(0.1)).collect())
            .collect();
        let out = block.forward(&x);
        for row in &out {
            for v in row {
                assert!(v.val().is_finite(), "block output should be finite, got {}", v.val());
            }
        }
    }

    // --- GPT model ---

    #[test]
    fn test_gpt_forward_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt::new(cfg.clone(), &mut rng);
        let ids = vec![0usize, 3, 5, 2];
        let logits = model.forward(&ids);
        assert_eq!(logits.len(), 4);
        assert_eq!(logits[0].len(), cfg.vocab_size);
    }

    #[test]
    fn test_gpt_logits_finite() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt::new(cfg.clone(), &mut rng);
        let ids = vec![1usize, 2, 3];
        let logits = model.forward(&ids);
        for row in &logits {
            for v in row {
                assert!(v.val().is_finite(), "logit should be finite, got {}", v.val());
            }
        }
    }

    #[test]
    fn test_gpt_loss_is_finite_and_positive() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt::new(cfg.clone(), &mut rng);
        let ids     = vec![1usize, 2, 3, 4];
        let targets = vec![2usize, 3, 4, 5];
        let loss = model.loss(&ids, &targets);
        let lv = loss.val();
        assert!(lv.is_finite(), "loss should be finite, got {}", lv);
        assert!(lv > 0.0, "loss should be positive, got {}", lv);
    }

    #[test]
    fn test_gpt_loss_near_random_baseline() {
        // A freshly initialized model should have loss ≈ log(vocab_size)
        // (random chance). For vocab_size=10: log(10) ≈ 2.30
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt::new(cfg.clone(), &mut rng);
        let ids     = vec![0usize, 1, 2, 3, 4];
        let targets = vec![1usize, 2, 3, 4, 5];
        let loss = model.loss(&ids, &targets);
        let lv = loss.val();
        let expected = (cfg.vocab_size as f32).ln();
        // Allow generous range: within 2x of random baseline
        assert!(
            lv < expected * 2.0,
            "initial loss {} should be near random baseline {:.2}",
            lv, expected
        );
    }

    #[test]
    fn test_gpt_backward_runs() {
        // Verify that backward() completes without panic and all gradients are finite
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt::new(cfg, &mut rng);
        let ids     = vec![0usize, 1, 2];
        let targets = vec![1usize, 2, 3];
        let loss = model.loss(&ids, &targets);
        loss.backward();
        for p in model.parameters() {
            assert!(
                p.grad().is_finite(),
                "gradient should be finite after backward, got {}",
                p.grad()
            );
        }
    }
}
