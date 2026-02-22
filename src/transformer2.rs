/// # Tensor-Level Transformer (transformer2)
///
/// Same GPT architecture as transformer.rs, but every operation is a
/// TensorNode matrix op instead of a scalar Value loop.
///
/// ## Key differences from scalar version
///
/// | Component          | Scalar                              | Tensor                          |
/// |--------------------|-------------------------------------|---------------------------------|
/// | Embedding lookup   | Vec<Vec<Value>> with element loops  | index into TensorNode rows      |
/// | Attention Q,K,V    | 3 × Linear loops over T tokens      | 3 × Linear2 matmuls             |
/// | Attention scores   | T² scalar dot products              | 1 causal_attention node         |
/// | MHA concat + proj  | Vec flatten + Linear loop           | hstack rows + Linear2 matmul    |
/// | TransformerBlock   | ~50K nodes for d_model=32           | ~20 nodes                       |
/// | Full GPT (2 blocks)| ~400K nodes                         | ~60 nodes                       |
///
/// ## Embedding as a matrix
///
/// token_embed: [vocab_size, d_model]   — each row is one token's vector
/// pos_embed:   [context_length, d_model]
///
/// For a sequence of T token ids, embedding forward = row-index then add:
///   x[t] = token_embed[token_ids[t]] + pos_embed[t]
///
/// We store the full tables as TensorNode leaves, gather rows into a [T, d_model]
/// matrix, and register a backward that scatters gradients back to the rows.

use crate::autograd2::{TensorNode, Mat};
use crate::nn::InitRng;
use crate::nn2::{Linear2, LayerNorm2, Mlp2, Module2, Trainable};
use std::cell::RefCell;

// =============================================================================
// Config (shared with scalar version — we re-use the same struct)
// =============================================================================

pub use crate::transformer::Config;

// =============================================================================
// Embedding2
// =============================================================================

pub struct Embedding2 {
    /// Full token embedding table [vocab_size, d_model]
    pub token_embed: TensorNode,
    /// Full positional embedding table [context_length, d_model]
    pub pos_embed: TensorNode,
    pub vocab_size: usize,
    pub context_length: usize,
    pub d_model: usize,
}

impl Embedding2 {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        let te_data = Mat::new(
            rng.normal_vec(config.vocab_size * config.d_model, 0.02),
            config.vocab_size, config.d_model,
        );
        let pe_data = Mat::new(
            rng.normal_vec(config.context_length * config.d_model, 0.01),
            config.context_length, config.d_model,
        );
        Embedding2 {
            token_embed: TensorNode::leaf(te_data),
            pos_embed:   TensorNode::leaf(pe_data),
            vocab_size:  config.vocab_size,
            context_length: config.context_length,
            d_model:     config.d_model,
        }
    }

    /// token_ids: &[usize] of length T  →  output: TensorNode [T, d_model]
    ///
    /// Forward: gather token rows + positional rows and add them.
    ///
    /// Backward: gradient for token_embed[id] += dOut[t]
    ///           gradient for pos_embed[t]    += dOut[t]
    ///
    /// This is implemented as a custom fused node that stores the id sequence.
    pub fn forward(&self, token_ids: &[usize]) -> TensorNode {
        let t = token_ids.len();
        let d = self.d_model;

        let te = self.token_embed.data().clone(); // [V, d]
        let pe = self.pos_embed.data().clone();   // [C, d]

        // Gather: out[row] = te[token_ids[row]] + pe[row]
        let out_data = Mat::from_fn(t, d, |row, col| {
            te.at(token_ids[row], col) + pe.at(row, col)
        });
        let out = TensorNode::leaf(out_data);

        let te_node = self.token_embed.clone();
        let pe_node = self.pos_embed.clone();
        let out_c   = out.clone();
        let ids     = token_ids.to_vec();

        out.set_backward(
            Box::new(move || {
                let dout = out_c.grad().clone(); // [T, d]

                // Scatter dout rows back to token_embed rows
                let mut dte = te_node.grad().clone();
                for (row, &tid) in ids.iter().enumerate() {
                    for col in 0..d {
                        *dte.at_mut(tid, col) += dout.at(row, col);
                    }
                }
                te_node.set_grad(dte);

                // Scatter dout rows back to pos_embed rows
                let mut dpe = pe_node.grad().clone();
                for row in 0..ids.len() {
                    for col in 0..d {
                        *dpe.at_mut(row, col) += dout.at(row, col);
                    }
                }
                pe_node.set_grad(dpe);
            }),
            vec![self.token_embed.clone(), self.pos_embed.clone()],
        );
        out
    }
}

impl Module2 for Embedding2 {
    fn parameters(&self) -> Vec<TensorNode> {
        vec![self.token_embed.clone(), self.pos_embed.clone()]
    }
}

// =============================================================================
// AttentionHead2
// =============================================================================
//
// One attention head. x [T, d_model] → output [T, d_head]
//
//   Q = x @ W_Q.T + b_Q   [T, d_head]
//   K = x @ W_K.T + b_K   [T, d_head]
//   V = x @ W_V.T + b_V   [T, d_head]
//   output = causal_attention(Q, K, V, d_head)   [T, d_head]

pub struct AttentionHead2 {
    pub w_q: Linear2,
    pub w_k: Linear2,
    pub w_v: Linear2,
    pub d_head: usize,
}

impl AttentionHead2 {
    pub fn new(d_model: usize, d_head: usize, rng: &mut InitRng) -> Self {
        AttentionHead2 {
            w_q: Linear2::new(d_model, d_head, rng),
            w_k: Linear2::new(d_model, d_head, rng),
            w_v: Linear2::new(d_model, d_head, rng),
            d_head,
        }
    }

    /// x: [T, d_model]  →  output: [T, d_head]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let q = self.w_q.forward(x);
        let k = self.w_k.forward(x);
        let v = self.w_v.forward(x);
        TensorNode::flash_attention(&q, &k, &v, self.d_head)
    }
}

impl Module2 for AttentionHead2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.w_q.parameters();
        p.extend(self.w_k.parameters());
        p.extend(self.w_v.parameters());
        p
    }
}

// =============================================================================
// MultiHeadAttention2
// =============================================================================
//
// Run n_heads in sequence, concatenate outputs, project with W_O.
//
// In production all heads run in parallel, but on single-threaded CPU
// sequential is fine — the matmul sizes dominate, not the loop overhead.

pub struct MultiHeadAttention2 {
    pub heads: Vec<AttentionHead2>,
    pub w_o: Linear2,       // [n_heads*d_head, d_model] = [d_model, d_model]
    pub d_model: usize,
}

impl MultiHeadAttention2 {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        let d_head = config.d_head();
        let heads = (0..config.n_heads)
            .map(|_| AttentionHead2::new(config.d_model, d_head, rng))
            .collect();
        MultiHeadAttention2 {
            heads,
            w_o: Linear2::new(config.d_model, config.d_model, rng),
            d_model: config.d_model,
        }
    }

    /// x: [T, d_model]  →  output: [T, d_model]
    ///
    /// Uses `batched_gqa_attention` so all heads run as a single batched matmul
    /// instead of a sequential per-head loop.
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let t  = x.data().rows;
        let d  = self.d_model;
        let dh = d / self.heads.len();
        let n  = self.heads.len();

        // Project each head's Q, K, V: each [T, d_head]
        // Then concatenate into [T, n_heads*d_head] for the batched call.
        let qs: Vec<TensorNode> = self.heads.iter().map(|h| h.w_q.forward(x)).collect();
        let ks: Vec<TensorNode> = self.heads.iter().map(|h| h.w_k.forward(x)).collect();
        let vs: Vec<TensorNode> = self.heads.iter().map(|h| h.w_v.forward(x)).collect();

        // Fused concat Q: [T, n*d_head]
        let q_concat = Self::concat_heads(&qs, t, n, dh);
        let k_concat = Self::concat_heads(&ks, t, n, dh);
        let v_concat = Self::concat_heads(&vs, t, n, dh);

        // Single batched GQA call (n_q == n_kv == n_heads, group_size=1)
        let attn_out = TensorNode::batched_gqa_attention(
            &q_concat, &k_concat, &v_concat,
            n, n, dh,
        );

        // Final projection [T, d_model] → [T, d_model]
        self.w_o.forward(&attn_out)
    }

    /// Concatenate a list of [T, d_head] TensorNodes into [T, n*d_head].
    ///
    /// Registers a backward that scatters the gradient back to each head's node.
    fn concat_heads(heads: &[TensorNode], t: usize, n: usize, dh: usize) -> TensorNode {
        let concat_data = Mat::from_fn(t, n * dh, |row, col| {
            heads[col / dh].data().at(row, col % dh)
        });
        let concat = TensorNode::leaf(concat_data);
        let heads_c: Vec<TensorNode> = heads.to_vec();
        let concat_c = concat.clone();

        concat.set_backward(
            Box::new(move || {
                let dconcat = concat_c.grad().clone();
                for (h, head) in heads_c.iter().enumerate() {
                    let mut dh_grad = head.grad().clone();
                    for row in 0..t {
                        for c in 0..dh {
                            *dh_grad.at_mut(row, c) += dconcat.at(row, h * dh + c);
                        }
                    }
                    head.set_grad(dh_grad);
                }
            }),
            heads.to_vec(),
        );
        concat
    }
}

impl Module2 for MultiHeadAttention2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p: Vec<TensorNode> = self.heads.iter().flat_map(|h| h.parameters()).collect();
        p.extend(self.w_o.parameters());
        p
    }
}

// =============================================================================
// TransformerBlock2
// =============================================================================
//
// Pre-norm residual block:
//   x = x + Attn(LN1(x))
//   x = x + MLP(LN2(x))

pub struct TransformerBlock2 {
    pub ln1: LayerNorm2,
    pub attn: MultiHeadAttention2,
    pub ln2: LayerNorm2,
    pub mlp: Mlp2,
}

impl TransformerBlock2 {
    pub fn new(config: &Config, rng: &mut InitRng) -> Self {
        TransformerBlock2 {
            ln1: LayerNorm2::new(config.d_model),
            attn: MultiHeadAttention2::new(config, rng),
            ln2: LayerNorm2::new(config.d_model),
            mlp: Mlp2::new(config.d_model, rng),
        }
    }

    /// x: [T, d_model]  →  output: [T, d_model]
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        // Attention sub-block: x = x + Attn(LN1(x))
        let attn_out = self.attn.forward(&self.ln1.forward(x));
        let x2 = residual_add(x, &attn_out);

        // MLP sub-block: x = x + MLP(LN2(x))
        let mlp_out = self.mlp.forward(&self.ln2.forward(&x2));
        residual_add(&x2, &mlp_out)
    }
}

impl Module2 for TransformerBlock2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.ln1.parameters();
        p.extend(self.attn.parameters());
        p.extend(self.ln2.parameters());
        p.extend(self.mlp.parameters());
        p
    }
}

// =============================================================================
// Helpers
// =============================================================================

/// Element-wise add of two [T, d] TensorNodes — the residual connection.
///
/// Uses TensorNode::add which already has a correct backward (dA = dC, dB = dC).
fn residual_add(a: &TensorNode, b: &TensorNode) -> TensorNode {
    a.add(b)
}

// =============================================================================
// Gpt2 — the full model
// =============================================================================

pub struct Gpt2 {
    pub embed: Embedding2,
    pub blocks: Vec<TransformerBlock2>,
    pub ln_final: LayerNorm2,
    pub lm_head: Linear2,   // d_model → vocab_size
    pub config: Config,
}

impl Gpt2 {
    pub fn new(config: Config, rng: &mut InitRng) -> Self {
        let blocks = (0..config.n_layers)
            .map(|_| TransformerBlock2::new(&config, rng))
            .collect();
        let embed    = Embedding2::new(&config, rng);
        let ln_final = LayerNorm2::new(config.d_model);
        let lm_head  = Linear2::new(config.d_model, config.vocab_size, rng);
        Gpt2 { embed, blocks, ln_final, lm_head, config }
    }

    /// Forward: token_ids → logits [T, vocab_size]
    pub fn forward(&self, token_ids: &[usize]) -> TensorNode {
        assert!(
            token_ids.len() <= self.config.context_length,
            "sequence length {} exceeds context_length {}",
            token_ids.len(), self.config.context_length
        );

        let mut x = self.embed.forward(token_ids);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let x_normed = self.ln_final.forward(&x);
        self.lm_head.forward(&x_normed)
    }

    /// Cross-entropy loss for next-token prediction.
    ///
    /// token_ids: input sequence [T]
    /// targets:   target sequence [T]  (token_ids shifted by 1)
    ///
    /// Returns a scalar TensorNode (shape [1, 1]).
    ///
    /// ## Numerically stable softmax + cross-entropy
    ///
    /// We compute: loss = mean_t(-log(softmax(logits[t])[targets[t]]))
    ///
    /// Combined formula for row r, correct class c:
    ///   loss_r = -logits[r,c] + log(sum_j exp(logits[r,j]))
    ///   (= -logits[r,c] + log_sum_exp(logits[r]))
    ///
    /// Gradient w.r.t. logits[r,j]:
    ///   d loss / d logits[r,j] = (softmax(logits[r])[j] - one_hot(j==c)) / T
    ///
    /// We implement this as a fused node to avoid building a [T, V] softmax
    /// node in the graph (we only need the loss scalar).
    pub fn loss(&self, token_ids: &[usize], targets: &[usize]) -> TensorNode {
        let logits_node = self.forward(token_ids);
        let logits = logits_node.data().clone(); // [T, V]
        let t = logits.rows;
        let v = logits.cols;
        assert_eq!(t, targets.len());

        // Compute softmax per row and cross-entropy
        let mut probs = Mat::zeros(t, v);
        let mut loss_val = 0.0f32;
        for r in 0..t {
            let row_max = (0..v).map(|c| logits.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0f32;
            for c in 0..v {
                let e = (logits.at(r, c) - row_max).exp();
                *probs.at_mut(r, c) = e;
                sum_exp += e;
            }
            for c in 0..v { *probs.at_mut(r, c) /= sum_exp; }
            loss_val += -probs.at(r, targets[r]).ln();
        }
        loss_val /= t as f32;

        let loss = TensorNode::leaf(Mat::new(vec![loss_val], 1, 1));

        let logits_c  = logits_node.clone();
        let probs_stored = probs;
        let targets_v    = targets.to_vec();

        loss.set_backward(
            Box::new(move || {
                // d loss / d logits[r, j] = (p[r,j] - one_hot(j == targets[r])) / T
                let mut dlogits = probs_stored.clone();
                for r in 0..t {
                    *dlogits.at_mut(r, targets_v[r]) -= 1.0;
                }
                let dlogits = dlogits.scale(1.0 / t as f32);
                let new_g = logits_c.grad().clone().add(&dlogits);
                logits_c.set_grad(new_g);
            }),
            vec![logits_node],
        );
        loss
    }
}

impl Module2 for Gpt2 {
    fn parameters(&self) -> Vec<TensorNode> {
        let mut p = self.embed.parameters();
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.ln_final.parameters());
        p.extend(self.lm_head.parameters());
        p
    }
}

impl Trainable for Gpt2 {
    fn forward_tokens(&self, token_ids: &[usize]) -> TensorNode { self.forward(token_ids) }
    fn loss_tokens(&self, token_ids: &[usize], targets: &[usize]) -> TensorNode { self.loss(token_ids, targets) }
}

impl Gpt2 {
    /// Tie the language model head weights to the token embedding weights.
    ///
    /// ## Why weight tying?
    ///
    /// The embedding table maps token id → d_model vector.
    /// The lm_head maps d_model vector → logit per token.
    ///
    /// These two operations are inverses of each other: both learn a
    /// per-token representation in the same d_model space.  Sharing them:
    ///   - Reduces total parameters by vocab_size * d_model (≈25M for GPT-2)
    ///   - Improves generalisation (the embedding and output projection stay consistent)
    ///   - Is the standard in GPT-2, LLaMA, Mistral, and most open-weight models
    ///
    /// After calling this, `lm_head.weight` and `embed.token_embed` point to
    /// the same `TensorNode`.  Both the forward pass and the backward pass
    /// will accumulate gradients into the same underlying storage.
    pub fn tie_weights(&mut self) {
        // lm_head.weight has shape [vocab_size, d_model]  (same as token_embed)
        assert_eq!(
            (self.lm_head.weight.data().rows, self.lm_head.weight.data().cols),
            (self.embed.token_embed.data().rows, self.embed.token_embed.data().cols),
            "tie_weights: lm_head.weight and embed.token_embed have different shapes"
        );
        self.lm_head.weight = self.embed.token_embed.clone();
    }
}

// =============================================================================
// KV Cache for Gpt2
// =============================================================================
//
// Autoregressive generation without a KV cache is O(T²): every new token
// requires re-computing attention over all T previous tokens from scratch.
//
// With a KV cache:
//   • Prefill stage: run the full prompt [T_p tokens] through all layers once,
//     storing K and V for each layer.
//   • Decode stage: each new token only runs through attention once (1×d),
//     reading the cached K/V from all previous positions.  → O(T) per step.
//
// ## Structure
//
// `Gpt2KvCache` holds one `HeadKvCache` per attention head per block.
// Each `HeadKvCache` stores a growing matrix:
//   k_cache: [seq_len, d_head]
//   v_cache: [seq_len, d_head]
//
// ## Implementation note
//
// The standard multi-head attention in `transformer2.rs` separates each head
// into its own `AttentionHead2` that independently projects and attends.
// The cached version simply lets each head's K/V matrix grow incrementally.

/// KV cache for a single attention head.
pub struct HeadKvCache {
    pub k: Mat,    // [seq_len, d_head]  — capped at context_length rows
    pub v: Mat,    // [seq_len, d_head]
    d_head: usize,
    max_len: usize,
}

impl HeadKvCache {
    fn new(d_head: usize, max_len: usize) -> Self {
        HeadKvCache {
            k:      Mat::zeros(0, d_head),
            v:      Mat::zeros(0, d_head),
            d_head,
            max_len,
        }
    }

    /// Append one new row to the cache, evicting the oldest if at capacity.
    fn append_k(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.d_head);
        let start = if self.k.rows >= self.max_len {
            // Drop the oldest row by skipping the first d_head elements
            self.d_head
        } else {
            0
        };
        let mut new_data = Vec::with_capacity(self.max_len * self.d_head);
        new_data.extend_from_slice(&self.k.data[start..]);
        new_data.extend_from_slice(row);
        let new_rows = new_data.len() / self.d_head;
        self.k = Mat::new(new_data, new_rows, self.d_head);
    }

    fn append_v(&mut self, row: &[f32]) {
        assert_eq!(row.len(), self.d_head);
        let start = if self.v.rows >= self.max_len {
            self.d_head
        } else {
            0
        };
        let mut new_data = Vec::with_capacity(self.max_len * self.d_head);
        new_data.extend_from_slice(&self.v.data[start..]);
        new_data.extend_from_slice(row);
        let new_rows = new_data.len() / self.d_head;
        self.v = Mat::new(new_data, new_rows, self.d_head);
    }
}

/// Per-block KV cache: one `HeadKvCache` per attention head.
pub struct BlockKvCache {
    pub heads: Vec<HeadKvCache>,
}

impl BlockKvCache {
    fn new(n_heads: usize, d_head: usize, max_len: usize) -> Self {
        BlockKvCache { heads: (0..n_heads).map(|_| HeadKvCache::new(d_head, max_len)).collect() }
    }
}

/// Full model KV cache: one `BlockKvCache` per transformer block.
pub struct Gpt2KvCache {
    pub blocks: Vec<RefCell<BlockKvCache>>,
}

impl Gpt2KvCache {
    pub fn new(config: &Config) -> Self {
        let d_head  = config.d_head();
        let max_len = config.context_length;
        let blocks = (0..config.n_layers)
            .map(|_| RefCell::new(BlockKvCache::new(config.n_heads, d_head, max_len)))
            .collect();
        Gpt2KvCache { blocks }
    }
}

// =============================================================================
// Cached attention forward
// =============================================================================

/// Compute attention for a single new token using the KV cache.
///
/// x_new: [1, d_model] — the new token's representation
/// cache:  per-head K/V cache for this block
///
/// Returns: [1, d_model] attended output
fn mha_forward_cached(
    attn: &MultiHeadAttention2,
    x_new: &TensorNode,
    cache: &mut BlockKvCache,
) -> TensorNode {
    let d = attn.d_model;
    let n_heads = attn.heads.len();
    let dh = d / n_heads;
    let t_new = x_new.data().rows; // always 1 during decode, T during prefill

    // Compute Q, K, V projections for the new token(s)
    let mut head_outs: Vec<TensorNode> = Vec::with_capacity(n_heads);

    for (h, head) in attn.heads.iter().enumerate() {
        let q_new = head.w_q.forward(x_new); // [t_new, d_head]
        let k_new = head.w_k.forward(x_new); // [t_new, d_head]
        let v_new = head.w_v.forward(x_new); // [t_new, d_head]

        // Append each new token's K, V to the cache
        let kd = k_new.data();
        let vd = v_new.data();
        for r in 0..t_new {
            let k_row: Vec<f32> = (0..dh).map(|c| kd.at(r, c)).collect();
            let v_row: Vec<f32> = (0..dh).map(|c| vd.at(r, c)).collect();
            cache.heads[h].append_k(&k_row);
            cache.heads[h].append_v(&v_row);
        }

        // Attend Q_new over full cached K, V (no causal mask needed —
        // the cache only contains past tokens, so all are valid context)
        let t_total = cache.heads[h].k.rows;
        let k_full = TensorNode::leaf(cache.heads[h].k.clone());
        let v_full = TensorNode::leaf(cache.heads[h].v.clone());

        // Scaled dot-product: Q [t_new, dh] × K.T [dh, t_total] → [t_new, t_total]
        let scale = (dh as f32).sqrt();
        let scores_data = Mat::from_fn(t_new, t_total, |tq, tk| {
            let dot: f32 = (0..dh).map(|d| q_new.data().at(tq, d) * k_full.data().at(tk, d)).sum();
            dot / scale
        });
        // Softmax
        let attn_weights_data = {
            let mut w = Mat::zeros(t_new, t_total);
            for r in 0..t_new {
                let mx = (0..t_total).map(|c| scores_data.at(r, c)).fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = (0..t_total).map(|c| (scores_data.at(r, c) - mx).exp()).collect();
                let s: f32 = exps.iter().sum();
                for c in 0..t_total { *w.at_mut(r, c) = exps[c] / s; }
            }
            w
        };
        // Weighted sum of V: [t_new, t_total] × [t_total, dh] → [t_new, dh]
        let out_data = Mat::from_fn(t_new, dh, |tq, d| {
            (0..t_total).map(|tk| attn_weights_data.at(tq, tk) * v_full.data().at(tk, d)).sum()
        });
        head_outs.push(TensorNode::leaf(out_data));
    }

    // Concatenate heads → [t_new, d_model]
    let concat_data = Mat::from_fn(t_new, d, |row, col| {
        head_outs[col / dh].data().at(row, col % dh)
    });
    let concat = TensorNode::leaf(concat_data);
    attn.w_o.forward(&concat)
}

/// Cached forward for a single `TransformerBlock2`.
fn block_forward_cached(
    block: &TransformerBlock2,
    x: &TensorNode,
    cache: &mut BlockKvCache,
) -> TensorNode {
    let attn_out = mha_forward_cached(&block.attn, &block.ln1.forward(x), cache);
    let x2 = x.add(&attn_out);
    let mlp_out = block.mlp.forward(&block.ln2.forward(&x2));
    x2.add(&mlp_out)
}

impl Gpt2 {
    /// Autoregressive generation with KV cache — O(T) per step.
    ///
    /// Equivalent to generating from `Gpt2` but avoids re-computing the full
    /// context for every token.  Provides the same interface as
    /// `GptOssModel::generate_cached`.
    ///
    /// ## Parameters
    ///   token_ids:    prompt token ids
    ///   max_new:      number of new tokens to generate
    ///   temperature:  ≤ 0 = greedy, else temperature-scaled sampling
    ///
    /// ## Returns
    ///   Vec of new token ids (not including the prompt).
    pub fn generate_cached(&self, token_ids: &[usize], max_new: usize, temperature: f32) -> Vec<usize> {
        let cache = Gpt2KvCache::new(&self.config);

        // --- Prefill: run the full prompt, populate cache ---
        let t_prompt = token_ids.len();
        let mut x = self.embed.forward(token_ids);
        for (bi, block) in self.blocks.iter().enumerate() {
            x = block_forward_cached(block, &x, &mut cache.blocks[bi].borrow_mut());
        }
        let first_logits = self.lm_head.forward(&self.ln_final.forward(&x));

        let mut generated = Vec::with_capacity(max_new);
        let first_tok = gpt2_sample_token(&first_logits.data(), t_prompt - 1, temperature);
        generated.push(first_tok);

        // --- Decode: one token at a time ---
        let mut prev_tok = first_tok;
        for _ in 1..max_new {
            let x = self.embed.forward(&[prev_tok]);
            let mut x2 = x;
            for (bi, block) in self.blocks.iter().enumerate() {
                x2 = block_forward_cached(block, &x2, &mut cache.blocks[bi].borrow_mut());
            }
            let logits = self.lm_head.forward(&self.ln_final.forward(&x2));
            prev_tok = gpt2_sample_token(&logits.data(), 0, temperature);
            generated.push(prev_tok);
        }
        generated
    }

    /// Autoregressive generation with KV cache and streaming callback — O(T) per step.
    ///
    /// Like `generate_cached` but supports top-k sampling and calls `callback`
    /// for each generated token, allowing streaming output.
    ///
    /// - `temperature <= 0` → greedy argmax
    /// - `top_k == 0`       → full-vocabulary sampling
    pub fn generate_cached_streaming<F>(
        &self,
        token_ids: &[usize],
        max_new: usize,
        temperature: f32,
        top_k: usize,
        mut callback: F,
    ) where F: FnMut(usize) {
        let cache = Gpt2KvCache::new(&self.config);

        // Prefill: run full prompt through the cache
        let t_prompt = token_ids.len();
        let mut x = self.embed.forward(token_ids);
        for (bi, block) in self.blocks.iter().enumerate() {
            x = block_forward_cached(block, &x, &mut cache.blocks[bi].borrow_mut());
        }
        let first_logits = self.lm_head.forward(&self.ln_final.forward(&x));
        let mut prev_tok = gpt2_sample_token_topk(&first_logits.data(), t_prompt - 1, temperature, top_k);
        callback(prev_tok);

        // Decode: one token at a time using the cache
        for _ in 1..max_new {
            let x = self.embed.forward(&[prev_tok]);
            let mut x2 = x;
            for (bi, block) in self.blocks.iter().enumerate() {
                x2 = block_forward_cached(block, &x2, &mut cache.blocks[bi].borrow_mut());
            }
            let logits = self.lm_head.forward(&self.ln_final.forward(&x2));
            prev_tok = gpt2_sample_token_topk(&logits.data(), 0, temperature, top_k);
            callback(prev_tok);
        }
    }
}  // end impl Gpt2

/// Temperature + top-k sampler for Gpt2 generation.
fn gpt2_sample_token_topk(logits: &Mat, pos: usize, temperature: f32, top_k: usize) -> usize {
    let v = logits.cols;
    if temperature <= 0.0 {
        return (0..v).max_by(|&a, &b|
            logits.at(pos, a).partial_cmp(&logits.at(pos, b)).unwrap()
        ).unwrap();
    }
    let row_max = (0..v).map(|c| logits.at(pos, c)).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = (0..v)
        .map(|c| ((logits.at(pos, c) - row_max) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    for p in &mut probs { *p /= sum; }

    // Top-k filter
    let k = if top_k == 0 { v } else { top_k.min(v) };
    let mut sorted = probs.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k - 1];
    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p >= threshold { p } else { 0.0 }).collect();
    let fsum: f32 = filtered.iter().sum();
    if fsum > 0.0 { for p in &mut filtered { *p /= fsum; } }

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(54321);
    let seed = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let rand_val = {
        let s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 33) as f32 / (u32::MAX as f32)
    };
    let mut cumsum = 0.0f32;
    for (i, &p) in filtered.iter().enumerate() {
        cumsum += p;
        if rand_val <= cumsum { return i; }
    }
    filtered.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap_or(0)
}

/// Simple temperature sampler for Gpt2 generation (greedy or argmax).
fn gpt2_sample_token(logits: &Mat, pos: usize, temperature: f32) -> usize {
    let v = logits.cols;
    if temperature <= 0.0 {
        return (0..v).max_by(|&a, &b|
            logits.at(pos, a).partial_cmp(&logits.at(pos, b)).unwrap()
        ).unwrap();
    }
    let row_max = (0..v).map(|c| logits.at(pos, c)).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = (0..v).map(|c| ((logits.at(pos, c) - row_max) / temperature).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    probs.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd2::Mat;

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

    // --- Embedding2 ---

    #[test]
    fn test_embedding2_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let emb = Embedding2::new(&cfg, &mut rng);
        let out = emb.forward(&[0, 3, 7]);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (3, cfg.d_model));
    }

    #[test]
    fn test_embedding2_different_positions() {
        // Same token at positions 0 and 1 should differ (positional encoding)
        let cfg = nano_config();
        let mut rng = make_rng();
        let emb = Embedding2::new(&cfg, &mut rng);
        let out = emb.forward(&[5, 5]);
        let d = out.data();
        let row0: Vec<f32> = (0..cfg.d_model).map(|c| d.at(0, c)).collect();
        let row1: Vec<f32> = (0..cfg.d_model).map(|c| d.at(1, c)).collect();
        assert_ne!(row0, row1);
    }

    #[test]
    fn test_embedding2_backward_finite() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let emb = Embedding2::new(&cfg, &mut rng);
        let out = emb.forward(&[1, 2, 3]);
        out.seed_grad_ones();
        out.call_backward_fn();
        for p in emb.parameters() {
            assert!(p.grad().data.iter().all(|x| x.is_finite()));
        }
    }

    // --- AttentionHead2 ---

    #[test]
    fn test_attn_head2_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let head = AttentionHead2::new(cfg.d_model, cfg.d_head(), &mut rng);
        let x = TensorNode::leaf(Mat::zeros(4, cfg.d_model));
        let out = head.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, cfg.d_head()));
    }

    // --- MultiHeadAttention2 ---

    #[test]
    fn test_mha2_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let mha = MultiHeadAttention2::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::zeros(5, cfg.d_model));
        let out = mha.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (5, cfg.d_model));
    }

    // --- TransformerBlock2 ---

    #[test]
    fn test_block2_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let block = TransformerBlock2::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(4, cfg.d_model, |r, c| (r * cfg.d_model + c) as f32 * 0.1));
        let out = block.forward(&x);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, cfg.d_model));
    }

    #[test]
    fn test_block2_output_finite() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let block = TransformerBlock2::new(&cfg, &mut rng);
        let x = TensorNode::leaf(Mat::from_fn(3, cfg.d_model, |_, _| 0.1));
        let out = block.forward(&x);
        assert!(out.data().data.iter().all(|x| x.is_finite()));
    }

    // --- Gpt2 ---

    #[test]
    fn test_gpt2_forward_output_shape() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let out = model.forward(&[0, 3, 5, 2]);
        let d = out.data();
        assert_eq!((d.rows, d.cols), (4, cfg.vocab_size));
    }

    #[test]
    fn test_gpt2_logits_finite() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let out = model.forward(&[1, 2, 3]);
        assert!(out.data().data.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gpt2_loss_finite_positive() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let loss = model.loss(&[1, 2, 3, 4], &[2, 3, 4, 5]);
        let lv = loss.data().at(0, 0);
        assert!(lv.is_finite(), "loss should be finite, got {}", lv);
        assert!(lv > 0.0, "loss should be positive, got {}", lv);
    }

    #[test]
    fn test_gpt2_loss_near_random_baseline() {
        // Random model should produce loss ≈ log(vocab_size)
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let loss = model.loss(&[0, 1, 2, 3, 4], &[1, 2, 3, 4, 5]);
        let lv = loss.data().at(0, 0);
        let expected = (cfg.vocab_size as f32).ln();
        assert!(lv < expected * 2.0,
            "initial loss {} should be near random baseline {:.2}", lv, expected);
    }

    #[test]
    fn test_gpt2_backward_runs() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg, &mut rng);
        let loss = model.loss(&[0, 1, 2], &[1, 2, 3]);
        loss.backward();
        for p in model.parameters() {
            assert!(
                p.grad().data.iter().all(|x| x.is_finite()),
                "gradient should be finite after backward"
            );
        }
    }

    // --- KV cache ---

    #[test]
    fn test_generate_cached_output_length() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let toks = model.generate_cached(&[0, 1, 2], 5, 0.0);
        assert_eq!(toks.len(), 5, "expected 5 new tokens");
    }

    #[test]
    fn test_generate_cached_tokens_in_vocab() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let toks = model.generate_cached(&[1, 2, 3], 4, 0.0);
        for &t in &toks {
            assert!(t < cfg.vocab_size, "token {} ≥ vocab_size {}", t, cfg.vocab_size);
        }
    }

    #[test]
    fn test_generate_cached_first_token_matches_forward() {
        // The first generated token (greedy) must equal predict_next on the prompt.
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let prompt = vec![0usize, 2, 4];

        // predict_next via full forward
        let logits = model.forward(&prompt);
        let ld = logits.data();
        let t = ld.rows;
        let v = ld.cols;
        let uncached = (0..v).max_by(|&a, &b| ld.at(t-1, a).partial_cmp(&ld.at(t-1, b)).unwrap()).unwrap();

        let cached = model.generate_cached(&prompt, 1, 0.0)[0];
        assert_eq!(cached, uncached,
            "cached first token {} != uncached {}", cached, uncached);
    }

    #[test]
    fn test_generate_cached_deterministic() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let r1 = model.generate_cached(&[0, 1], 6, 0.0);
        let r2 = model.generate_cached(&[0, 1], 6, 0.0);
        assert_eq!(r1, r2, "greedy generation must be deterministic");
    }

    #[test]
    fn test_kv_cache_respects_context_window() {
        // Generate more tokens than context_length — cache must not grow beyond it.
        let cfg = nano_config(); // context_length = 8
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let cache = Gpt2KvCache::new(&cfg);

        // Simulate appending context_length + 4 tokens to one head cache
        let d_head = cfg.d_head();
        let row = vec![0.1f32; d_head];
        let head_cache = &mut cache.blocks[0].borrow_mut().heads[0];
        for _ in 0..(cfg.context_length + 4) {
            head_cache.append_k(&row);
            head_cache.append_v(&row);
        }
        assert_eq!(head_cache.k.rows, cfg.context_length,
            "KV cache k rows {} should be capped at context_length {}",
            head_cache.k.rows, cfg.context_length);
        assert_eq!(head_cache.v.rows, cfg.context_length,
            "KV cache v rows {} should be capped at context_length {}",
            head_cache.v.rows, cfg.context_length);
    }

    #[test]
    fn test_generate_cached_streaming_beyond_context() {
        // Generating more tokens than context_length must complete without panic.
        let cfg = nano_config(); // context_length = 8
        let mut rng = make_rng();
        let model = Gpt2::new(cfg.clone(), &mut rng);
        let max_new = cfg.context_length + 4;
        let mut count = 0usize;
        model.generate_cached_streaming(&[0, 1, 2], max_new, 0.0, 0, |tok| {
            assert!(tok < cfg.vocab_size);
            count += 1;
        });
        assert_eq!(count, max_new);
    }

    // --- Weight tying ---

    #[test]
    fn test_tie_weights_shares_tensor() {
        // After tie_weights(), mutating lm_head.weight data should affect embed.token_embed
        let cfg = nano_config();
        let mut rng = make_rng();
        let mut model = Gpt2::new(cfg.clone(), &mut rng);
        model.tie_weights();

        // Write a distinctive value into lm_head.weight[0,0]
        let mut wd = model.lm_head.weight.data().clone();
        *wd.at_mut(0, 0) = 99.0;
        model.lm_head.weight.set_data(wd);

        // embed.token_embed must see the same change (same Rc)
        assert_eq!(model.embed.token_embed.data().at(0, 0), 99.0,
            "tie_weights: embed.token_embed and lm_head.weight must share storage");
    }

    #[test]
    fn test_tie_weights_forward_still_runs() {
        let cfg = nano_config();
        let mut rng = make_rng();
        let mut model = Gpt2::new(cfg.clone(), &mut rng);
        model.tie_weights();
        let logits = model.forward(&[0, 1, 2, 3]);
        let d = logits.data();
        assert_eq!((d.rows, d.cols), (4, cfg.vocab_size));
        assert!(d.data.iter().all(|v| v.is_finite()), "tied-weight forward should be finite");
    }

    #[test]
    fn test_tie_weights_reduces_parameter_count() {
        // Count duplicate TensorNode pointers in parameters() BEFORE and AFTER tying.
        // Before: all parameter TensorNodes should be distinct.
        // After tying: lm_head.weight == embed.token_embed → one duplicate entry in the list.
        let cfg = nano_config();
        let mut rng = make_rng();

        let count_duplicates = |m: &Gpt2| {
            let ps = m.parameters();
            let total = ps.len();
            let mut ptrs: Vec<*const f32> = ps.iter()
                .map(|p| p.data().data.as_ptr())
                .collect();
            ptrs.sort();
            ptrs.dedup();
            total - ptrs.len() // number of duplicates
        };

        let m = Gpt2::new(cfg.clone(), &mut rng);
        let dups_before = count_duplicates(&m);

        let mut m2 = Gpt2::new(cfg.clone(), &mut rng);
        m2.tie_weights();
        let dups_after = count_duplicates(&m2);

        assert!(dups_after > dups_before,
            "tie_weights should introduce at least 1 duplicate: before={} after={}", dups_before, dups_after);
    }

    #[test]
    fn test_tie_weights_backward_accumulates_into_embed() {
        // After tying, the grad of lm_head.weight and embed.token_embed should be the same object
        let cfg = nano_config();
        let mut rng = make_rng();
        let mut model = Gpt2::new(cfg.clone(), &mut rng);
        model.tie_weights();

        let loss = model.loss(&[0, 1, 2], &[1, 2, 3]);
        loss.backward();

        // The gradient on lm_head.weight and embed.token_embed must be identical
        let g_lm   = model.lm_head.weight.grad().clone();
        let g_emb  = model.embed.token_embed.grad().clone();
        assert_eq!(g_lm.rows, g_emb.rows);
        assert_eq!(g_lm.cols, g_emb.cols);
        // Both point to same storage, so values must be equal
        for (a, b) in g_lm.data.iter().zip(g_emb.data.iter()) {
            assert_eq!(*a, *b, "tied weight grads must be identical");
        }
    }
}
