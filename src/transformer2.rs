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
use crate::nn2::{Linear2, LayerNorm2, Mlp2, Module2};

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
        TensorNode::causal_attention(&q, &k, &v, self.d_head)
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
    pub fn forward(&self, x: &TensorNode) -> TensorNode {
        let t   = x.data().rows;
        let d   = self.d_model;
        let dh  = d / self.heads.len();

        // Collect each head's output: [T, d_head]
        let head_outs: Vec<TensorNode> = self.heads.iter().map(|h| h.forward(x)).collect();

        // Concatenate along feature dim: → [T, d_model]
        // We do this as a custom fused node so the backward scatters back correctly.
        let concat_data = Mat::from_fn(t, d, |row, col| {
            let h = col / dh;
            let c = col % dh;
            head_outs[h].data().at(row, c)
        });
        let concat = TensorNode::leaf(concat_data);
        let head_outs_c: Vec<TensorNode> = head_outs.clone();
        let concat_c = concat.clone();

        concat.set_backward(
            Box::new(move || {
                let dconcat = concat_c.grad().clone(); // [T, d_model]
                for (h, head_out) in head_outs_c.iter().enumerate() {
                    // Scatter columns [h*dh .. (h+1)*dh] back to head_out.grad
                    let mut dh_grad = head_out.grad().clone();
                    for row in 0..t {
                        for c in 0..dh {
                            *dh_grad.at_mut(row, c) += dconcat.at(row, h * dh + c);
                        }
                    }
                    head_out.set_grad(dh_grad);
                }
            }),
            head_outs,
        );

        // Final projection [T, d_model] → [T, d_model]
        self.w_o.forward(&concat)
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
}
