/// # Tensor-Level Training (train2)
///
/// Same AdamW optimizer and training loop as train.rs, but operating on
/// TensorNode parameters instead of scalar Values.
///
/// ## What changes
///
/// train.rs:
///   params: Vec<Value>   — one scalar per weight element
///   optimizer: Vec<f32> m/v — one slot per scalar
///
/// train2.rs:
///   params: Vec<TensorNode>  — one matrix per weight tensor
///   optimizer: for each TensorNode, m/v are Mat of same shape
///
/// The AdamW math is identical — we just apply it element-wise over each
/// parameter matrix instead of over a flat vector of scalars.
///
/// ## Performance difference
///
/// With scalar autograd, each of the ~400K graph nodes must be visited during
/// backward, and the gradient for each scalar updated one-by-one.
///
/// With tensor autograd, backward visits only ~60 nodes total, and each node's
/// backward does a small number of matrix operations (matmul, scale, etc.)
/// that the compiler can vectorize with SIMD.
///
/// On a typical CPU (no GPU), expect 50-200x speedup for our nano model.

use crate::autograd2::{TensorNode, Mat};
use crate::nn2::Module2;

// =============================================================================
// AdamW2 — same algorithm as AdamW, but over TensorNode parameters
// =============================================================================

pub struct AdamW2 {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub step: u32,

    /// First moment: same shape as each parameter
    m: Vec<Mat>,

    /// Second moment: same shape as each parameter
    v: Vec<Mat>,
}

impl AdamW2 {
    /// Create an optimizer initialized from the model's current parameter list.
    ///
    /// We capture the shapes of all parameters now so we can allocate m/v buffers.
    pub fn new(params: &[TensorNode], lr: f32) -> Self {
        let m: Vec<Mat> = params.iter()
            .map(|p| Mat::zeros(p.data().rows, p.data().cols))
            .collect();
        let v: Vec<Mat> = params.iter()
            .map(|p| Mat::zeros(p.data().rows, p.data().cols))
            .collect();
        AdamW2 {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.1, step: 0,
            m, v,
        }
    }

    /// Perform one optimizer step.
    ///
    /// Call AFTER loss.backward() and BEFORE zero_grad().
    pub fn step(&mut self, params: &[TensorNode]) {
        assert_eq!(params.len(), self.m.len(),
            "parameter count changed: expected {}, got {}",
            self.m.len(), params.len());

        self.step += 1;
        let t = self.step as f32;
        let bc1 = 1.0 - self.beta1.powf(t);
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in params.iter().enumerate() {
            let w = param.data().clone();  // current weights [r, c]
            let g = param.grad().clone();  // gradient [r, c]

            // Skip if gradient is all zeros (e.g. unused embedding rows)
            if g.data.iter().all(|&x| x == 0.0) { continue; }

            // Weight decay: w *= (1 - lr * wd)
            let wd_factor = 1.0 - self.lr * self.weight_decay;
            let w_decayed = w.scale(wd_factor);

            // Update first moment: m = β₁·m + (1-β₁)·g  (element-wise)
            let m_new = Mat::from_fn(w.rows, w.cols, |r, c| {
                self.beta1 * self.m[i].at(r, c) + (1.0 - self.beta1) * g.at(r, c)
            });

            // Update second moment: v = β₂·v + (1-β₂)·g²
            let v_new = Mat::from_fn(w.rows, w.cols, |r, c| {
                let gv = g.at(r, c);
                self.beta2 * self.v[i].at(r, c) + (1.0 - self.beta2) * gv * gv
            });

            // Adam update: w -= lr · m̂ / (√v̂ + ε)
            let w_new = Mat::from_fn(w.rows, w.cols, |r, c| {
                let m_hat = m_new.at(r, c) / bc1;
                let v_hat = v_new.at(r, c) / bc2;
                w_decayed.at(r, c) - self.lr * m_hat / (v_hat.sqrt() + self.eps)
            });

            // Write updated values back
            self.m[i] = m_new;
            self.v[i] = v_new;
            param.set_data(w_new);
        }
    }
}

// =============================================================================
// Training configuration
// =============================================================================

pub struct TrainConfig2 {
    pub max_steps: usize,
    pub eval_interval: usize,
    pub learning_rate: f32,
    pub grad_clip: f32,
}

impl Default for TrainConfig2 {
    fn default() -> Self {
        TrainConfig2 {
            max_steps: 500,
            eval_interval: 50,
            learning_rate: 3e-4,
            grad_clip: 1.0,
        }
    }
}

// =============================================================================
// Training loop
// =============================================================================

use crate::transformer2::Gpt2;
use crate::tokenizer::CharTokenizer;
use crate::dataset::TextDataset;

/// Run the full training loop on a Gpt2 model.
///
/// Returns the final training loss.
pub fn train2(
    model: &Gpt2,
    _tokenizer: &CharTokenizer,
    train_data: &TextDataset,
    val_data: &TextDataset,
    cfg: &TrainConfig2,
) -> f32 {
    let params = model.parameters();
    let n_params: usize = params.iter().map(|p| p.data().rows * p.data().cols).collect::<Vec<_>>().iter().sum();
    let mut optimizer = AdamW2::new(&params, cfg.learning_rate);

    println!(
        "\n[tensor] Training: {} parameters (in {} tensors), {} steps",
        n_params, params.len(), cfg.max_steps
    );
    println!("{:-<65}", "");

    let mut last_loss = f32::INFINITY;

    for step in 0..cfg.max_steps {
        // ---- 1. Sample one training example ----
        let (inp_t, tgt_t) = train_data.random_batch(1, step as u64 + 1);
        let token_ids: Vec<usize>  = inp_t.data.iter().map(|&x| x as usize).collect();
        let target_ids: Vec<usize> = tgt_t.data.iter().map(|&x| x as usize).collect();

        // ---- 2. Zero gradients ----
        for p in model.parameters() { p.zero_grad(); }

        // ---- 3. Forward + loss ----
        let loss = model.loss(&token_ids, &target_ids);
        last_loss = loss.data().at(0, 0);

        // ---- 4. Backward ----
        loss.backward();

        // ---- 5. Gradient clipping ----
        let params_now = model.parameters();
        let grad_norm: f32 = params_now.iter()
            .map(|p| p.grad().data.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        if grad_norm > cfg.grad_clip {
            let scale = cfg.grad_clip / grad_norm;
            for p in &params_now {
                let g_scaled = p.grad().scale(scale);
                p.set_grad(g_scaled);
            }
        }

        // ---- 6. Optimizer step ----
        optimizer.step(&params_now);

        // ---- 7. Logging ----
        if step % cfg.eval_interval == 0 || step == cfg.max_steps - 1 {
            let val_loss = estimate_loss2(model, val_data, 5);
            println!(
                "step {:4}/{} | train_loss: {:.4} | val_loss: {:.4} | grad_norm: {:.4}",
                step, cfg.max_steps, last_loss, val_loss, grad_norm
            );
        }
    }

    println!("{:-<65}", "");
    last_loss
}

fn estimate_loss2(model: &Gpt2, data: &TextDataset, n_samples: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n_samples {
        let (inp_t, tgt_t) = data.random_batch(1, i as u64 + 9999);
        let token_ids:  Vec<usize> = inp_t.data.iter().map(|&x| x as usize).collect();
        let target_ids: Vec<usize> = tgt_t.data.iter().map(|&x| x as usize).collect();
        for p in model.parameters() { p.zero_grad(); }
        let loss = model.loss(&token_ids, &target_ids);
        total += loss.data().at(0, 0);
    }
    total / n_samples as f32
}

// =============================================================================
// Text generation
// =============================================================================

pub fn generate2(
    model: &Gpt2,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> String {
    use crate::tokenizer::Tokenizer;

    let ctx_len = model.config.context_length;
    let vocab_size = model.config.vocab_size;

    let mut token_ids: Vec<usize> = tokenizer.encode(prompt)
        .iter()
        .map(|&x| x as usize)
        .collect();

    print!("{}", prompt);

    for _ in 0..max_new_tokens {
        let context: Vec<usize> = if token_ids.len() > ctx_len {
            token_ids[token_ids.len() - ctx_len..].to_vec()
        } else {
            token_ids.clone()
        };

        for p in model.parameters() { p.zero_grad(); }
        let logits_node = model.forward(&context);
        let logits = logits_node.data(); // [T, V]
        let t = logits.rows;

        // Get last row and apply temperature
        let mut probs = vec![0.0f32; vocab_size];
        let row_max = (0..vocab_size).map(|c| logits.at(t - 1, c)).fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for c in 0..vocab_size {
            let e = ((logits.at(t - 1, c) - row_max) / temperature).exp();
            probs[c] = e;
            sum_exp += e;
        }
        for p in &mut probs { *p /= sum_exp; }

        let next_token = sample_top_k(&probs, top_k);
        let ch = tokenizer.decode(&[next_token as u32]);
        print!("{}", ch);
        token_ids.push(next_token);
    }

    println!();

    tokenizer.decode(&token_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())
}

/// Sample from the top-k entries of a probability distribution.
fn sample_top_k(probs: &[f32], k: usize) -> usize {
    let k = k.min(probs.len());
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let threshold = sorted[k - 1];

    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p >= threshold { p } else { 0.0 }).collect();
    let sum: f32 = filtered.iter().sum();
    for p in &mut filtered { *p /= sum; }

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(54321);
    let seed = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let rand_val = {
        let s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 32) as f32 / u32::MAX as f32
    };

    let mut cumulative = 0.0f32;
    for (i, &p) in filtered.iter().enumerate() {
        cumulative += p;
        if rand_val <= cumulative { return i; }
    }
    filtered.len() - 1
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer2::{Config, Gpt2};
    use crate::nn::InitRng;

    fn make_tiny_model(vocab_size: usize) -> Gpt2 {
        let cfg = Config {
            vocab_size,
            context_length: 8,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
        };
        let mut rng = InitRng::new(0);
        Gpt2::new(cfg, &mut rng)
    }

    // --- AdamW2 ---

    #[test]
    fn test_adamw2_step_increments() {
        let model = make_tiny_model(10);
        let params = model.parameters();
        let mut opt = AdamW2::new(&params, 1e-3);
        assert_eq!(opt.step, 0);
        // Manually seed a non-zero gradient
        for p in &params {
            let (r, c) = { let d = p.data(); (d.rows, d.cols) };
            p.set_grad(Mat::ones(r, c));
        }
        opt.step(&params);
        assert_eq!(opt.step, 1);
    }

    #[test]
    fn test_adamw2_decreases_simple_loss() {
        // One parameter [1,1] minimizing (w - target)^2.
        // We manually set gradient each step.
        let target = 2.0f32;
        let w = TensorNode::leaf(Mat::new(vec![0.0], 1, 1));
        let params = vec![w.clone()];
        let mut opt = AdamW2::new(&params, 0.3);

        for _ in 0..200 {
            let val = w.data().at(0, 0);
            // gradient of (w-target)^2 = 2*(w-target)
            w.set_grad(Mat::new(vec![2.0 * (val - target)], 1, 1));
            opt.step(&params);
        }

        let final_val = w.data().at(0, 0);
        assert!(
            (final_val - target).abs() < 0.2,
            "AdamW2 should minimize loss: w = {:.4}, target = {:.4}",
            final_val, target
        );
    }

    // --- Gradient clipping ---

    #[test]
    fn test_grad_clip2_scales_down() {
        let w = TensorNode::leaf(Mat::ones(2, 2)); // 4 elements
        // Set gradient = [10, 10, 10, 10], norm = sqrt(4*100) = 20
        w.set_grad(Mat::new(vec![10.0; 4], 2, 2));

        let clip_norm = 1.0f32;
        let grad_norm: f32 = { w.grad().data.iter().map(|x| x * x).sum::<f32>().sqrt() };

        assert!(grad_norm > clip_norm);
        if grad_norm > clip_norm {
            let scale = clip_norm / grad_norm;
            let clipped = w.grad().scale(scale); // Ref dropped after this line
            w.set_grad(clipped);
        }

        let new_norm: f32 = w.grad().data.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((new_norm - 1.0).abs() < 1e-5,
            "clipped norm should be 1.0, got {}", new_norm);
    }

    // --- sample_top_k ---

    #[test]
    fn test_sample_top_k_valid_index() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        for _ in 0..20 {
            let idx = sample_top_k(&probs, 2);
            assert!(idx < 4);
        }
    }

    #[test]
    fn test_sample_top_k_1_returns_argmax() {
        let probs = vec![0.1, 0.05, 0.8, 0.05];
        for _ in 0..10 {
            let idx = sample_top_k(&probs, 1);
            assert_eq!(idx, 2);
        }
    }

    // --- Full training smoke test ---

    #[test]
    fn test_loss_decreases_after_training2() {
        use crate::tokenizer::{CharTokenizer, Tokenizer};
        use crate::dataset::TextDataset;

        let corpus = "abcabcabc".repeat(20);
        let corpus = corpus.as_str();
        let tokenizer = CharTokenizer::from_text(corpus);
        let model = make_tiny_model(tokenizer.vocab_size());

        let (train_ds, val_ds) = TextDataset::train_val_split(
            corpus, &tokenizer, model.config.context_length
        );

        let initial_loss = {
            let ids: Vec<usize> = tokenizer.encode("abcabc").iter().map(|&x| x as usize).collect();
            let tgt: Vec<usize> = tokenizer.encode("bcabca").iter().map(|&x| x as usize).collect();
            for p in model.parameters() { p.zero_grad(); }
            let loss = model.loss(&ids, &tgt);
            loss.data().at(0, 0)
        };

        let cfg = TrainConfig2 {
            max_steps: 30,
            eval_interval: 30,
            learning_rate: 1e-2,
            grad_clip: 1.0,
        };

        train2(&model, &tokenizer, &train_ds, &val_ds, &cfg);

        let final_loss = {
            let ids: Vec<usize> = tokenizer.encode("abcabc").iter().map(|&x| x as usize).collect();
            let tgt: Vec<usize> = tokenizer.encode("bcabca").iter().map(|&x| x as usize).collect();
            for p in model.parameters() { p.zero_grad(); }
            let loss = model.loss(&ids, &tgt);
            loss.data().at(0, 0)
        };

        assert!(
            final_loss < initial_loss,
            "loss should decrease after training: {:.4} → {:.4}",
            initial_loss, final_loss
        );
    }
}
