/// # Training — Phase 4
///
/// We have a model. We have gradients. Now we need to use those gradients
/// to actually improve the model. That's the optimizer's job.
///
/// ## Gradient descent in one line
///
///   weight -= learning_rate * gradient
///
/// If the gradient says "increasing this weight increases loss", we decrease
/// it. By a small amount (learning_rate), so we don't overshoot.
///
/// ## Why not plain SGD?
///
/// Plain SGD (Stochastic Gradient Descent) works, but:
///   - Needs careful tuning of learning rate (too high → diverge, too low → slow)
///   - Different parameters may need different effective learning rates
///   - Sparse gradients (rare tokens rarely updated) update too slowly
///
/// Modern LLMs use Adam or AdamW, which solve all three problems.
///
/// ## AdamW — Adaptive Moment Estimation with Weight Decay
///
/// Adam (Kingma & Ba, 2014) maintains two running statistics per parameter:
///
///   m = first moment  (exponential moving average of gradients)
///       ≈ "which direction have gradients been pointing recently?"
///
///   v = second moment (exponential moving average of gradient²)
///       ≈ "how large/noisy have gradients been recently?"
///
/// The update rule:
///
///   m = β₁ · m + (1 - β₁) · grad          (decay old, add new gradient)
///   v = β₂ · v + (1 - β₂) · grad²         (decay old, add new gradient²)
///
///   m̂ = m / (1 - β₁ᵗ)                     (bias correction, important early on)
///   v̂ = v / (1 - β₂ᵗ)
///
///   weight -= lr · m̂ / (√v̂ + ε)
///
/// Intuition:
///   - m̂ is a smoothed gradient direction (reduces noise from mini-batches)
///   - √v̂ in the denominator normalizes by historical gradient magnitude
///     → parameters with large gradients get smaller effective lr
///     → parameters with small/sparse gradients get larger effective lr
///   - ε prevents division by zero
///
/// ## The "W" in AdamW: weight decay
///
/// Weight decay adds L2 regularization: it gently pushes all weights toward 0.
/// This prevents overfitting — the model can't memorize training data by
/// making any single weight arbitrarily large.
///
/// Regular Adam absorbs weight decay into the gradient, which interacts
/// badly with the adaptive scaling. AdamW applies it separately:
///
///   weight -= lr · weight · weight_decay    (applied BEFORE the Adam step)
///
/// This is the correct way (Loshchilov & Hutter, 2019) and what GPT-2 uses.
///
/// ## Hyperparameters
///
/// Standard values (used by GPT-2, LLaMA, etc.):
///   β₁ = 0.9      (gradient momentum)
///   β₂ = 0.999    (gradient² momentum)
///   ε  = 1e-8     (numerical stability)
///   weight_decay = 0.1
///   learning_rate = 3e-4  (for small models; larger models use smaller lr)

use crate::autograd::Value;
use crate::nn::Module;

// =============================================================================
// AdamW Optimizer
// =============================================================================

pub struct AdamW {
    /// Learning rate α
    pub lr: f32,

    /// β₁: momentum for first moment (gradient direction)
    pub beta1: f32,

    /// β₂: momentum for second moment (gradient magnitude)
    pub beta2: f32,

    /// ε: prevents division by zero
    pub eps: f32,

    /// Weight decay coefficient (L2 regularization strength)
    pub weight_decay: f32,

    /// Current step number (for bias correction). Starts at 1.
    pub step: u32,

    /// First moment estimates — one per parameter, same order as parameters()
    m: Vec<f32>,

    /// Second moment estimates — one per parameter
    v: Vec<f32>,

    /// Number of parameters this optimizer was built for
    n_params: usize,
}

impl AdamW {
    /// Create an AdamW optimizer with standard hyperparameters.
    pub fn new(n_params: usize, lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.1,
            step: 0,
            m: vec![0.0; n_params],
            v: vec![0.0; n_params],
            n_params,
        }
    }

    /// Perform one optimizer step.
    ///
    /// Call this AFTER loss.backward() and BEFORE zero_grad().
    ///
    /// params: the model's parameters in the same order as when AdamW was created
    pub fn step(&mut self, params: &[Value]) {
        assert_eq!(
            params.len(), self.n_params,
            "parameter count changed: expected {}, got {}",
            self.n_params, params.len()
        );

        self.step += 1;
        let t = self.step as f32;

        // Bias correction factors
        // Early in training (t=1), m and v are biased toward 0 because they
        // started at 0. We divide by (1 - β^t) to correct for this.
        let bc1 = 1.0 - self.beta1.powf(t);  // approaches 1 as t grows
        let bc2 = 1.0 - self.beta2.powf(t);

        for (i, param) in params.iter().enumerate() {
            let g = param.grad();

            // Skip parameters with zero gradient (e.g. unused embeddings)
            if g == 0.0 { continue; }

            // Weight decay: gently shrink the weight toward 0
            // Applied to the raw weight value, not through the gradient
            {
                let current = param.val();
                let decayed = current * (1.0 - self.lr * self.weight_decay);
                // We directly mutate the Value's internal data
                // (accessing through the Rc<RefCell<>> we own)
                param.set_val(decayed);
            }

            // Update first moment: m = β₁·m + (1-β₁)·g
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;

            // Update second moment: v = β₂·v + (1-β₂)·g²
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;

            // Bias-corrected moments
            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // Adam update: w -= lr · m̂ / (√v̂ + ε)
            let delta = self.lr * m_hat / (v_hat.sqrt() + self.eps);
            param.set_val(param.val() - delta);
        }
    }
}

// =============================================================================
// Training loop
// =============================================================================
//
// The training loop is deceptively simple:
//
//   for each step:
//     1. Sample a random batch from the training data
//     2. Forward pass → compute loss
//     3. Backward pass → compute gradients
//     4. Optimizer step → update weights
//     5. Zero gradients → prepare for next step
//
// That's it. Repeat millions of times. This is how GPT-3 was trained.
//
// ## What "learning" actually looks like
//
// At step 0: loss ≈ log(vocab_size)  — random guessing
// After a few hundred steps: loss drops noticeably
// After many steps: loss plateaus at some irreducible minimum
//   (the model can't perfectly predict the future — language is stochastic)
//
// ## Why use batches?
//
// A batch of B examples gives B gradient estimates, averaged together.
// This reduces noise and allows more stable updates.
// Also: modern hardware (GPUs) are massively parallel — processing B examples
// takes almost the same time as processing 1.
//
// ## Gradient clipping
//
// Sometimes gradients spike to very large values (exploding gradients).
// We cap the total gradient norm at a threshold (1.0 is standard for LLMs):
//
//   if ‖g‖ > clip_norm:
//       g = g · (clip_norm / ‖g‖)
//
// This prevents a single bad batch from destroying the model's weights.

use crate::transformer::Gpt;
use crate::tokenizer::CharTokenizer;
use crate::dataset::TextDataset;

pub struct TrainConfig {
    pub max_steps: usize,
    pub eval_interval: usize,
    pub learning_rate: f32,
    pub grad_clip: f32,
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            max_steps: 500,
            eval_interval: 50,
            learning_rate: 3e-4,
            grad_clip: 1.0,
        }
    }
}

/// Run the full training loop.
///
/// Returns the final training loss.
pub fn train(
    model: &Gpt,
    tokenizer: &CharTokenizer,
    train_data: &TextDataset,
    val_data: &TextDataset,
    cfg: &TrainConfig,
) -> f32 {
    let params = model.parameters();
    let n_params = params.len();
    let mut optimizer = AdamW::new(n_params, cfg.learning_rate);

    println!(
        "\nTraining: {} parameters, {} steps",
        n_params, cfg.max_steps
    );
    println!("{:-<55}", "");

    let mut last_loss = f32::INFINITY;

    for step in 0..cfg.max_steps {
        // ---- 1. Sample one training example (batch_size=1 for our nano model) ----
        // We use step as seed so each step sees a different example,
        // but runs are reproducible.
        let (inp_t, tgt_t) = train_data.random_batch(1, step as u64 + 1);

        // Convert f32 tensor row back to usize token ids
        let token_ids: Vec<usize> = inp_t.data.iter().map(|&x| x as usize).collect();
        let target_ids: Vec<usize> = tgt_t.data.iter().map(|&x| x as usize).collect();

        // ---- 2. Zero gradients from previous step ----
        model.zero_grad();

        // ---- 3. Forward pass + loss ----
        let loss = model.loss(&token_ids, &target_ids);
        last_loss = loss.val();

        // ---- 4. Backward pass ----
        loss.backward();

        // ---- 5. Gradient clipping ----
        // Compute total gradient norm: ‖g‖ = sqrt(sum of all g²)
        let params_now = model.parameters();
        let grad_norm: f32 = params_now
            .iter()
            .map(|p| p.grad() * p.grad())
            .sum::<f32>()
            .sqrt();

        if grad_norm > cfg.grad_clip {
            let scale = cfg.grad_clip / grad_norm;
            for p in &params_now {
                // Scale gradient down in place
                let g = p.grad();
                p.set_grad(g * scale);
            }
        }

        // ---- 6. Optimizer step ----
        optimizer.step(&params_now);

        // ---- 7. Logging ----
        if step % cfg.eval_interval == 0 || step == cfg.max_steps - 1 {
            // Estimate validation loss on a few examples
            let val_loss = estimate_loss(model, val_data, 5);
            println!(
                "step {:4}/{} | train_loss: {:.4} | val_loss: {:.4} | grad_norm: {:.4}",
                step, cfg.max_steps, last_loss, val_loss, grad_norm
            );
        }
    }

    println!("{:-<55}", "");
    last_loss
}

/// Estimate loss on a dataset by averaging over `n_samples` random examples.
/// We run forward-only (no backward) to save memory.
fn estimate_loss(model: &Gpt, data: &TextDataset, n_samples: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n_samples {
        let (inp_t, tgt_t) = data.random_batch(1, i as u64 + 9999);
        let token_ids: Vec<usize>  = inp_t.data.iter().map(|&x| x as usize).collect();
        let target_ids: Vec<usize> = tgt_t.data.iter().map(|&x| x as usize).collect();
        model.zero_grad();
        let loss = model.loss(&token_ids, &target_ids);
        total += loss.val();
    }
    total / n_samples as f32
}

// =============================================================================
// Text generation — Phase 5
// =============================================================================
//
// After training, we want to generate text. This is the inference (decoding) loop.
//
// ## Autoregressive generation
//
// The model predicts one token at a time. Each new token is fed back
// as input for the next prediction:
//
//   prompt: "ciao "
//   step 1: model sees ["c","i","a","o"," "] → predicts "m" → append
//   step 2: model sees ["c","i","a","o"," ","m"] → predicts "o" → append
//   ...
//
// ## Sampling strategies
//
// Given the probability distribution over the next token, how do we pick one?
//
//   1. Greedy (argmax): always pick the most probable token
//      → deterministic, often repetitive ("the the the the...")
//
//   2. Temperature sampling: divide logits by T before softmax
//      T < 1.0: sharper distribution (more confident, less creative)
//      T > 1.0: flatter distribution (more random, more creative)
//      T = 1.0: sample from the model's learned distribution
//
//   3. Top-k sampling: only sample from the k most probable tokens
//      Prevents the model from randomly picking very unlikely tokens
//
//   4. Top-p (nucleus) sampling: sample from the smallest set of tokens
//      whose cumulative probability exceeds p (e.g. 0.9)
//      Adapts to the sharpness of the distribution automatically
//
// We implement temperature + top-k, which is what GPT-2 used.

pub fn generate(
    model: &Gpt,
    tokenizer: &CharTokenizer,
    prompt: &str,
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
) -> String {
    use crate::tokenizer::Tokenizer;
    use crate::nn::softmax;

    let ctx_len = model.config.context_length;

    // Encode the prompt into token ids
    let mut token_ids: Vec<usize> = tokenizer.encode(prompt)
        .iter()
        .map(|&x| x as usize)
        .collect();

    print!("{}", prompt);

    for _ in 0..max_new_tokens {
        // Truncate to context window if needed (sliding window)
        let context: Vec<usize> = if token_ids.len() > ctx_len {
            token_ids[token_ids.len() - ctx_len..].to_vec()
        } else {
            token_ids.clone()
        };

        // Forward pass — we only need the logits for the LAST position
        model.zero_grad();
        let logits = model.forward(&context);
        let last_logits = &logits[logits.len() - 1];

        // Apply temperature: divide logits by T
        // Low T → peaky distribution (confident)
        // High T → flat distribution (random)
        let scaled: Vec<Value> = last_logits
            .iter()
            .map(|v| Value::new(v.val() / temperature))
            .collect();

        // Convert to probabilities
        let probs_vals: Vec<f32> = softmax(&scaled).iter().map(|v| v.val()).collect();

        // Top-k filtering: zero out all but the top-k probabilities
        let next_token = sample_top_k(&probs_vals, top_k);

        // Decode and print the new token
        let ch = tokenizer.decode(&[next_token as u32]);
        print!("{}", ch);

        // Append to context for next step
        token_ids.push(next_token);
    }

    println!(); // newline after generation

    // Return the full generated text (prompt + new tokens)
    tokenizer.decode(&token_ids.iter().map(|&x| x as u32).collect::<Vec<_>>())
}

/// Sample from a probability distribution, restricted to the top-k entries.
///
/// Algorithm:
///   1. Find the k largest probabilities
///   2. Zero out all others
///   3. Renormalize to sum to 1
///   4. Sample using a simple LCG RNG
fn sample_top_k(probs: &[f32], k: usize) -> usize {
    let k = k.min(probs.len());

    // Find the k-th largest probability (threshold)
    let mut sorted = probs.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap()); // descending
    let threshold = sorted[k - 1];

    // Build filtered distribution
    let mut filtered: Vec<f32> = probs.iter().map(|&p| if p >= threshold { p } else { 0.0 }).collect();

    // Renormalize
    let sum: f32 = filtered.iter().sum();
    for p in &mut filtered {
        *p /= sum;
    }

    // Sample using a simple approach: pick randomly weighted by probability
    // We use a simple LCG seeded by current time (via a static counter)
    static SAMPLE_COUNTER: std::sync::atomic::AtomicU64 =
        std::sync::atomic::AtomicU64::new(12345);
    let seed = SAMPLE_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let rand_val = lcg_float(seed);

    let mut cumulative = 0.0f32;
    for (i, &p) in filtered.iter().enumerate() {
        cumulative += p;
        if rand_val <= cumulative {
            return i;
        }
    }
    filtered.len() - 1 // fallback
}

fn lcg_float(seed: u64) -> f32 {
    let s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (s >> 32) as f32 / u32::MAX as f32
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformer::{Config, Gpt};
    use crate::tokenizer::CharTokenizer;
    use crate::dataset::TextDataset;
    use crate::nn::Module;

    fn make_tiny_model(vocab_size: usize) -> Gpt {
        let cfg = Config {
            vocab_size,
            context_length: 8,
            d_model: 8,
            n_layers: 1,
            n_heads: 2,
        };
        let mut rng = crate::nn::InitRng::new(0);
        Gpt::new(cfg, &mut rng)
    }

    // --- AdamW ---

    #[test]
    fn test_adamw_decreases_loss_on_simple_function() {
        // Minimize f(x) = x² using AdamW
        // Gradient: df/dx = 2x
        // Starting at x=3.0, should converge toward x=0
        use crate::autograd::Value;

        let x = Value::new(3.0);
        let mut opt = AdamW::new(1, 0.1);

        for _ in 0..100 {
            x.zero_grad();
            let loss = x.mul(&x); // f(x) = x²
            loss.backward();
            opt.step(&[x.clone()]);
        }

        assert!(
            x.val().abs() < 0.1,
            "AdamW should minimize x²: x = {:.4} after 100 steps",
            x.val()
        );
    }

    #[test]
    fn test_adamw_step_count_increments() {
        let mut opt = AdamW::new(1, 0.01);
        let x = Value::new(1.0);
        x.set_grad(0.5); // manually set gradient
        assert_eq!(opt.step, 0);
        opt.step(&[x]);
        assert_eq!(opt.step, 1);
    }

    // --- Gradient clipping ---

    #[test]
    fn test_grad_clip_scales_down_large_gradients() {
        use crate::autograd::Value;

        // Build a graph that produces a large gradient, then clip it
        // x = 1000.0, loss = x (so grad = 1000.0)
        let x = Value::new(1000.0);
        let loss = x.clone(); // dL/dx = 1.0... use mul to get large grad
        // Actually: loss = x * 1000 → grad of x = 1000
        let big = Value::new(1000.0);
        let scaled_loss = x.mul(&big);
        scaled_loss.backward(); // x.grad = 1000.0

        let clip_norm = 1.0f32;
        let grad_norm = x.grad().abs();

        assert!(grad_norm > clip_norm, "gradient should be large before clipping");

        if grad_norm > clip_norm {
            let scale = clip_norm / grad_norm;
            x.set_grad(x.grad() * scale);
        }

        // After clipping: x.grad = 1000 * (1.0 / 1000) = 1.0 (norm = clip_norm)
        assert!(
            (x.grad() - 1.0).abs() < 1e-5,
            "clipped gradient norm should equal clip_norm=1.0, got {}",
            x.grad()
        );
    }

    // --- sample_top_k ---

    #[test]
    fn test_sample_top_k_returns_valid_index() {
        let probs = vec![0.1, 0.5, 0.3, 0.1];
        for _ in 0..20 {
            let idx = sample_top_k(&probs, 2);
            assert!(idx < 4, "sampled index {} out of range", idx);
        }
    }

    #[test]
    fn test_sample_top_k_1_returns_argmax() {
        // top-k=1 should always return the argmax
        let probs = vec![0.1, 0.05, 0.8, 0.05];
        for _ in 0..10 {
            let idx = sample_top_k(&probs, 1);
            assert_eq!(idx, 2, "top-1 should always return argmax (index 2)");
        }
    }

    // --- Full training smoke test ---

    #[test]
    fn test_loss_decreases_after_training() {
        // A tiny model trained on a tiny corpus should decrease loss.
        // This is a smoke test — it just checks the training loop runs
        // and the loss goes in the right direction.
        // Repeat enough times so dataset has many windows of length context_length=8
        let corpus = "abcabcabc".repeat(20);
        let corpus = corpus.as_str();
        let tokenizer = CharTokenizer::from_text(corpus);
        let model = make_tiny_model(tokenizer.vocab_size());

        use crate::tokenizer::Tokenizer;
        let (train_ds, val_ds) = TextDataset::train_val_split(
            corpus, &tokenizer, model.config.context_length
        );

        let initial_loss = {
            let ids: Vec<usize> = tokenizer.encode("abcabc").iter().map(|&x| x as usize).collect();
            let tgt: Vec<usize> = tokenizer.encode("bcabca").iter().map(|&x| x as usize).collect();
            model.zero_grad();
            model.loss(&ids, &tgt).val()
        };

        let cfg = TrainConfig {
            max_steps: 30,
            eval_interval: 30,
            learning_rate: 1e-2,
            grad_clip: 1.0,
        };

        train(&model, &tokenizer, &train_ds, &val_ds, &cfg);

        let final_loss = {
            let ids: Vec<usize> = tokenizer.encode("abcabc").iter().map(|&x| x as usize).collect();
            let tgt: Vec<usize> = tokenizer.encode("bcabca").iter().map(|&x| x as usize).collect();
            model.zero_grad();
            model.loss(&ids, &tgt).val()
        };

        assert!(
            final_loss < initial_loss,
            "loss should decrease after training: {:.4} → {:.4}",
            initial_loss, final_loss
        );
    }
}
