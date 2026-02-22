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

use crate::autograd2::{TensorNode, Mat, save_checkpoint};
use crate::nn2::{Module2, Trainable};

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

/// Hyperparameters for `AdamW2`.
///
/// All fields have production-proven defaults.  Override only what you need:
///
/// ```rust
/// let hps = AdamW2Params { lr: 3e-4, weight_decay: 0.01, ..AdamW2Params::default() };
/// let mut opt = AdamW2::with_params(&model_params, hps);
/// ```
#[derive(Clone, Debug)]
pub struct AdamW2Params {
    /// Learning rate (step size). Common values: 1e-4 – 1e-3 for fine-tuning,
    /// 3e-4 for pretraining small models.
    pub lr: f32,
    /// First-moment decay (momentum). Standard: 0.9.
    pub beta1: f32,
    /// Second-moment decay. Standard: 0.999 for Adam, 0.95 for Muon-like.
    pub beta2: f32,
    /// Numerical stability constant. Standard: 1e-8.
    pub eps: f32,
    /// L2 regularisation coefficient. 0.1 for GPT-scale models, 0.01 for fine-tunes.
    pub weight_decay: f32,
}

impl Default for AdamW2Params {
    fn default() -> Self {
        AdamW2Params { lr: 3e-4, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.1 }
    }
}

impl AdamW2 {
    /// Create an optimizer with default hyperparameters.
    pub fn new(params: &[TensorNode], lr: f32) -> Self {
        Self::with_params(params, AdamW2Params { lr, ..AdamW2Params::default() })
    }

    /// Create an optimizer with fully customised hyperparameters.
    pub fn with_params(params: &[TensorNode], hp: AdamW2Params) -> Self {
        let m: Vec<Mat> = params.iter()
            .map(|p| Mat::zeros(p.data().rows, p.data().cols))
            .collect();
        let v: Vec<Mat> = params.iter()
            .map(|p| Mat::zeros(p.data().rows, p.data().cols))
            .collect();
        AdamW2 {
            lr: hp.lr, beta1: hp.beta1, beta2: hp.beta2,
            eps: hp.eps, weight_decay: hp.weight_decay, step: 0,
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
// Learning-rate scheduler — cosine decay with linear warmup
// =============================================================================
//
// ## Why a scheduler?
//
// A fixed learning rate either:
//   a) starts too high → unstable early training (loss explodes)
//   b) stays too high → fails to converge to the optimum late in training
//
// The cosine-warmup schedule is the de-facto standard for transformer training:
//
//   1. Linear warmup 0 → lr_max over `warmup_steps` steps.
//      Prevents large gradient steps from corrupting random initialisation.
//
//   2. Cosine decay lr_max → lr_min over the remaining steps.
//      Smooth decay prevents over-shooting the minimum.
//
// ## Usage
//
//   let sched = LrScheduler { lr_max: 3e-4, lr_min: 3e-5,
//                             warmup_steps: 100, total_steps: 1000 };
//   let lr = sched.get(step);
//   optimizer.set_lr(lr);

/// Cosine annealing with linear warmup.
#[derive(Clone, Debug)]
pub struct LrScheduler {
    /// Peak learning rate (reached after warmup).
    pub lr_max: f32,
    /// Minimum learning rate at end of cosine decay. Usually lr_max / 10.
    pub lr_min: f32,
    /// Number of warmup steps (linear 0 → lr_max).
    pub warmup_steps: usize,
    /// Total number of training steps.
    pub total_steps: usize,
}

impl LrScheduler {
    /// Return the scheduled learning rate at the given step (0-indexed).
    pub fn get(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.lr_max * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            // Cosine decay from lr_max → lr_min
            let progress = (step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.lr_min + (self.lr_max - self.lr_min) * cosine
        }
    }
}

impl AdamW2 {
    /// Update the learning rate in-place (used with `LrScheduler`).
    pub fn set_lr(&mut self, lr: f32) { self.lr = lr; }
}

// =============================================================================
// Training configuration
// =============================================================================

pub struct TrainConfig2 {
    pub max_steps: usize,
    pub eval_interval: usize,
    pub learning_rate: f32,
    pub grad_clip: f32,
    /// How many micro-steps to accumulate gradients over before an optimizer
    /// step.  Effective batch size = batch_size * accumulate_steps.
    /// Set to 1 to disable (default).
    pub accumulate_steps: usize,
    /// Number of independent sequences per training step.
    ///
    /// Each step samples `batch_size` sequences, runs forward+backward on each
    /// (via `loss_batch_tokens`), and averages the gradients before the optimizer
    /// step.  Larger batches → smoother gradients → more stable training.
    ///
    /// Typical values: 1 (default, same as before), 4–8 for personal models.
    /// Memory cost is linear in batch_size (no GPU parallelism — sequences are
    /// processed one at a time, gradients accumulated).
    pub batch_size: usize,
    /// Label-smoothing ε.  0.0 = standard cross-entropy (default).
    /// Typical: 0.1.  Replaces the one-hot target with:
    ///   y_smooth[v] = (1 - ε) * one_hot[v] + ε / vocab_size
    pub label_smoothing: f32,
    /// If Some(path), save a binary checkpoint after training completes.
    /// The file is written by `save_checkpoint` and can be reloaded with
    /// `load_checkpoint`.  None = don't save (default).
    pub checkpoint_path: Option<String>,
    /// Stop training early if val loss does not improve for this many eval
    /// intervals.  0 = disabled (default).  Typical value: 5.
    pub early_stopping_patience: usize,
}

impl Default for TrainConfig2 {
    fn default() -> Self {
        TrainConfig2 {
            max_steps: 500,
            eval_interval: 50,
            learning_rate: 3e-4,
            grad_clip: 1.0,
            accumulate_steps: 1,
            batch_size: 1,
            label_smoothing: 0.0,
            checkpoint_path: None,
            early_stopping_patience: 0,
        }
    }
}

// =============================================================================
// Training loop
// =============================================================================

use crate::dataset::DataSource;

/// Compute token-level accuracy: fraction of positions where argmax(logits) == target.
///
/// `logits` is a `[T, V]` matrix, `targets` is a slice of T token ids.
pub fn token_accuracy(logits: &Mat, targets: &[usize]) -> f32 {
    let t = logits.rows;
    assert_eq!(t, targets.len(), "token_accuracy: logits rows ({}) != targets len ({})", t, targets.len());
    let v = logits.cols;
    let correct: usize = (0..t)
        .filter(|&row| {
            let pred = (0..v)
                .max_by(|&a, &b| logits.at(row, a).partial_cmp(&logits.at(row, b)).unwrap())
                .unwrap_or(0);
            pred == targets[row]
        })
        .count();
    correct as f32 / t as f32
}

/// Run the full training loop on any model that implements `Trainable`.
///
/// Works with `Gpt2` (transformer2) and `GptOssModel` (transformer3).
/// Returns the final training loss.
pub fn train2<T: Trainable, D: DataSource>(
    model: &T,
    train_data: &D,
    val_data: &D,
    cfg: &TrainConfig2,
) -> f32 {
    let params = model.parameters();
    let n_params: usize = params.iter().map(|p| p.data().rows * p.data().cols).collect::<Vec<_>>().iter().sum();
    let mut optimizer = AdamW2::new(&params, cfg.learning_rate);

    // Cosine scheduler: warmup for 5% of steps, decay to lr/10
    let sched = LrScheduler {
        lr_max: cfg.learning_rate,
        lr_min: cfg.learning_rate * 0.1,
        warmup_steps: (cfg.max_steps / 20).max(1),
        total_steps: cfg.max_steps,
    };

    let batch_size = cfg.batch_size.max(1);
    println!(
        "\n[tensor] Training: {} parameters (in {} tensors), {} steps (batch={}, accumulate={})",
        n_params, params.len(), cfg.max_steps, batch_size, cfg.accumulate_steps
    );
    println!("{:-<65}", "");

    let mut last_loss = f32::INFINITY;
    let mut last_acc = 0.0f32;
    let accum = cfg.accumulate_steps.max(1);
    let mut best_val_loss = f32::INFINITY;
    let mut patience_counter = 0usize;

    for step in 0..cfg.max_steps {
        // ---- 1. Update learning rate ----
        optimizer.set_lr(sched.get(step));

        // ---- 2. Zero gradients at the start of each accumulation window ----
        if step % accum == 0 {
            for p in model.parameters() { p.zero_grad(); }
        }

        // ---- 3. Sample a batch of training examples ----
        let batch_raw = train_data.sample_batch(step as u64 + 1, batch_size);
        // Keep (token_ids, target_ids) for the first sequence for metrics
        let (first_tokens, first_targets) = &batch_raw[0];

        // ---- 4. Forward + loss (batch) ----
        // loss_batch_tokens runs forward+backward for each sequence internally,
        // scaling each upstream gradient by 1/B.  Returns mean loss as a leaf.
        let batch_refs: Vec<(&[usize], &[usize])> = batch_raw.iter()
            .map(|(inp, tgt)| (inp.as_slice(), tgt.as_slice()))
            .collect();
        let loss = model.loss_batch_tokens(&batch_refs);

        // ---- 5. NaN/Inf guard — skip step if loss is not finite ----
        let loss_val = loss.data().at(0, 0);
        if !loss_val.is_finite() {
            eprintln!("[warn] step {}: non-finite loss ({:.4}), skipping", step, loss_val);
            continue;
        }

        // Compute metrics on the first sequence of the batch
        {
            let logits_node = model.forward_tokens(first_tokens);
            let logits = logits_node.data();
            last_loss = cross_entropy_smoothed(&logits, first_targets, cfg.label_smoothing);
            last_acc  = token_accuracy(&logits, first_targets);
        }

        // ---- 6. (Backward already done inside loss_batch_tokens) ----

        // Only update weights at the end of each accumulation window
        let is_update_step = (step + 1) % accum == 0 || step == cfg.max_steps - 1;
        if !is_update_step { continue; }

        // ---- 7. NaN/Inf guard on gradients ----
        let params_now = model.parameters();
        let grad_norm: f32 = params_now.iter()
            .map(|p| p.grad().data.iter().map(|x| x * x).sum::<f32>())
            .sum::<f32>()
            .sqrt();

        if !grad_norm.is_finite() {
            eprintln!("[warn] step {}: non-finite grad_norm ({:.4}), skipping optimizer step", step, grad_norm);
            for p in model.parameters() { p.zero_grad(); }
            continue;
        }

        // ---- 8. Gradient clipping ----
        if grad_norm > cfg.grad_clip {
            let scale = cfg.grad_clip / grad_norm;
            for p in &params_now {
                let g_scaled = p.grad().scale(scale);
                p.set_grad(g_scaled);
            }
        }

        // ---- 9. Optimizer step ----
        optimizer.step(&params_now);

        // ---- 10. Logging ----
        let display_step = step / accum;
        let display_total = cfg.max_steps / accum;
        if display_step % (cfg.eval_interval / accum).max(1) == 0 || step == cfg.max_steps - 1 {
            let val_loss = estimate_loss2(model, val_data, 5);
            println!(
                "step {:4}/{} | lr: {:.2e} | train_loss: {:.4} | val_loss: {:.4} | acc: {:.3} | grad_norm: {:.4}",
                display_step, display_total,
                optimizer.lr, last_loss, val_loss, last_acc, grad_norm
            );

            // Early stopping
            if cfg.early_stopping_patience > 0 {
                if val_loss < best_val_loss {
                    best_val_loss = val_loss;
                    patience_counter = 0;
                } else {
                    patience_counter += 1;
                    if patience_counter >= cfg.early_stopping_patience {
                        println!("[early stop] val loss has not improved for {} evals, stopping.", patience_counter);
                        break;
                    }
                }
            }
        }
    }

    println!("{:-<65}", "");

    // ---- 11. Checkpoint save ----
    if let Some(ref path) = cfg.checkpoint_path {
        let params_final = model.parameters();
        let named: Vec<(String, &TensorNode)> = params_final.iter()
            .enumerate()
            .map(|(i, p)| (format!("param_{}", i), p))
            .collect();
        let named_refs: Vec<(&str, &TensorNode)> = named.iter()
            .map(|(n, p)| (n.as_str(), *p))
            .collect();
        match save_checkpoint(path, &named_refs) {
            Ok(()) => println!("[checkpoint] Saved {} tensors to {}", named_refs.len(), path),
            Err(e) => eprintln!("[checkpoint] Failed to save {}: {}", path, e),
        }
    }

    last_loss
}

/// Cross-entropy with optional label smoothing.
///
/// Standard cross-entropy (smoothing=0): L = -log(softmax(logits)[target])
///
/// Label-smoothed (smoothing=ε):
///   L = -sum_v y_smooth[v] * log_softmax[v]
///   where y_smooth[v] = (1-ε) * one_hot(v == target) + ε / V
///
/// Returns the mean loss across all T positions.
fn cross_entropy_smoothed(logits: &Mat, targets: &[usize], smoothing: f32) -> f32 {
    let t = logits.rows;
    let v = logits.cols;
    let mut total = 0.0f32;
    for row in 0..t {
        // Numerically stable log-softmax
        let max_l = (0..v).map(|c| logits.at(row, c)).fold(f32::NEG_INFINITY, f32::max);
        let sum_exp: f32 = (0..v).map(|c| (logits.at(row, c) - max_l).exp()).sum();
        let log_sum = sum_exp.ln();

        if smoothing == 0.0 {
            let tgt = targets[row];
            let log_prob = logits.at(row, tgt) - max_l - log_sum;
            total -= log_prob;
        } else {
            // Smooth: each vocab entry contributes ε/V, target also contributes (1-ε)
            let base: f32 = (0..v).map(|c| {
                let log_p = logits.at(row, c) - max_l - log_sum;
                (smoothing / v as f32) * log_p
            }).sum();
            let tgt = targets[row];
            let log_p_tgt = logits.at(row, tgt) - max_l - log_sum;
            total -= base + (1.0 - smoothing) * log_p_tgt;
        }
    }
    total / t as f32
}

fn estimate_loss2<T: Trainable, D: DataSource>(model: &T, data: &D, n_samples: usize) -> f32 {
    let mut total = 0.0f32;
    for i in 0..n_samples {
        let (token_ids, target_ids) = data.sample(i as u64 + 9999);
        for p in model.parameters() { p.zero_grad(); }
        let loss = model.loss_tokens(&token_ids, &target_ids);
        total += loss.data().at(0, 0);
        for p in model.parameters() { p.zero_grad(); }
    }
    total / n_samples as f32
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

    // --- LrScheduler ---

    #[test]
    fn test_lr_scheduler_warmup_starts_near_zero() {
        // With 10 warmup steps, step 0 returns lr_max * 1/10
        let sched = LrScheduler { lr_max: 1e-3, lr_min: 1e-4, warmup_steps: 10, total_steps: 100 };
        let lr0 = sched.get(0);
        // lr0 = lr_max / warmup_steps = 1e-3/10 = 1e-4; must be strictly less than lr_max
        assert!(lr0 < sched.lr_max, "warmup step 0 should be < lr_max, got {}", lr0);
        assert!(lr0 > 0.0, "warmup step 0 should be > 0, got {}", lr0);
    }

    #[test]
    fn test_lr_scheduler_warmup_reaches_max() {
        let sched = LrScheduler { lr_max: 1e-3, lr_min: 1e-4, warmup_steps: 10, total_steps: 100 };
        let lr_peak = sched.get(9); // last warmup step (index 9 = step 10/10 of warmup)
        assert!((lr_peak - 1e-3).abs() < 1e-6, "warmup should reach lr_max, got {}", lr_peak);
    }

    #[test]
    fn test_lr_scheduler_decay_is_monotone() {
        let sched = LrScheduler { lr_max: 1e-3, lr_min: 1e-4, warmup_steps: 5, total_steps: 50 };
        let mut prev = f32::INFINITY;
        for step in 5..50 {
            let lr = sched.get(step);
            assert!(lr <= prev + 1e-9, "cosine decay should be monotone at step {}", step);
            prev = lr;
        }
    }

    #[test]
    fn test_lr_scheduler_ends_at_lr_min() {
        // At the very last step (total_steps - 1), progress = (T-1-warmup)/(T-warmup).
        // For total_steps=50, warmup=5: progress = 44/45 < 1.0, so cos is not exactly -1.
        // We verify the last step is close to lr_min and strictly below lr_max.
        let sched = LrScheduler { lr_max: 1e-3, lr_min: 1e-4, warmup_steps: 5, total_steps: 50 };
        let lr_end = sched.get(49);
        assert!(lr_end < sched.lr_max,
            "final lr should be < lr_max, got {}", lr_end);
        // Should be close to lr_min (within 5% of the lr_max - lr_min range)
        let tolerance = (sched.lr_max - sched.lr_min) * 0.05;
        assert!((lr_end - sched.lr_min).abs() < tolerance,
            "final lr {} should be close to lr_min {} (tol {})", lr_end, sched.lr_min, tolerance);
    }

    // --- token_accuracy ---

    #[test]
    fn test_token_accuracy_perfect() {
        // logits: identity matrix → argmax row i = i
        let logits = Mat::from_fn(4, 4, |r, c| if r == c { 10.0 } else { 0.0 });
        let targets = vec![0usize, 1, 2, 3];
        let acc = token_accuracy(&logits, &targets);
        assert!((acc - 1.0).abs() < 1e-6, "perfect accuracy should be 1.0, got {}", acc);
    }

    #[test]
    fn test_token_accuracy_zero() {
        // logits: argmax row i = 0 always, but targets = 1,2,3,4
        let logits = Mat::from_fn(4, 5, |_, c| if c == 0 { 10.0 } else { 0.0 });
        let targets = vec![1usize, 2, 3, 4];
        let acc = token_accuracy(&logits, &targets);
        assert!((acc - 0.0).abs() < 1e-6, "zero accuracy expected, got {}", acc);
    }

    #[test]
    fn test_token_accuracy_half() {
        // Rows 0,1 correct; rows 2,3 wrong
        let logits = Mat::new(vec![
            10.0, 0.0,  // row 0: argmax = 0
             0.0, 10.0, // row 1: argmax = 1
            10.0, 0.0,  // row 2: argmax = 0
             0.0, 10.0, // row 3: argmax = 1
        ], 4, 2);
        let targets = vec![0usize, 1, 1, 0]; // rows 2 and 3 wrong
        let acc = token_accuracy(&logits, &targets);
        assert!((acc - 0.5).abs() < 1e-6, "expected 0.5, got {}", acc);
    }

    // --- cross_entropy_smoothed ---

    #[test]
    fn test_cross_entropy_no_smoothing_is_positive() {
        let logits = Mat::from_fn(3, 5, |r, c| (r * 5 + c) as f32 * 0.1);
        let targets = vec![1usize, 2, 3];
        let loss = cross_entropy_smoothed(&logits, &targets, 0.0);
        assert!(loss.is_finite() && loss > 0.0, "CE loss should be finite positive, got {}", loss);
    }

    #[test]
    fn test_cross_entropy_smoothing_higher_than_no_smoothing() {
        // Label smoothing increases entropy → higher loss on non-optimal logits
        let logits = Mat::from_fn(2, 4, |r, c| if r == c { 10.0 } else { -1.0 });
        let targets = vec![0usize, 1];
        let loss_no_smooth = cross_entropy_smoothed(&logits, &targets, 0.0);
        let loss_smooth    = cross_entropy_smoothed(&logits, &targets, 0.1);
        assert!(loss_smooth > loss_no_smooth,
            "smoothed loss ({}) should be > unsmoothed ({})", loss_smooth, loss_no_smooth);
    }

    #[test]
    fn test_adamw2_set_lr() {
        let model = make_tiny_model(10);
        let params = model.parameters();
        let mut opt = AdamW2::new(&params, 1e-3);
        opt.set_lr(5e-4);
        assert!((opt.lr - 5e-4).abs() < 1e-10, "set_lr should update lr field");
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
            accumulate_steps: 1,
            batch_size: 1,
            label_smoothing: 0.0,
            checkpoint_path: None,
            early_stopping_patience: 0,
        };

        train2(&model, &train_ds, &val_ds, &cfg);

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

    #[test]
    fn test_training_with_label_smoothing() {
        use crate::tokenizer::{CharTokenizer, Tokenizer};
        use crate::dataset::TextDataset;

        let corpus = "abcabcabc".repeat(10);
        let corpus = corpus.as_str();
        let tokenizer = CharTokenizer::from_text(corpus);
        let model = make_tiny_model(tokenizer.vocab_size());
        let (train_ds, val_ds) = TextDataset::train_val_split(
            corpus, &tokenizer, model.config.context_length
        );
        let cfg = TrainConfig2 {
            max_steps: 10,
            eval_interval: 10,
            learning_rate: 1e-2,
            grad_clip: 1.0,
            accumulate_steps: 1,
            batch_size: 1,
            label_smoothing: 0.1,
            checkpoint_path: None,
            early_stopping_patience: 0,
        };
        let loss = train2(&model, &train_ds, &val_ds, &cfg);
        assert!(loss.is_finite() && loss > 0.0,
            "training with label smoothing should produce finite loss, got {}", loss);
    }

    #[test]
    fn test_training_with_grad_accumulation() {
        use crate::tokenizer::{CharTokenizer, Tokenizer};
        use crate::dataset::TextDataset;

        let corpus = "abcabcabc".repeat(10);
        let corpus = corpus.as_str();
        let tokenizer = CharTokenizer::from_text(corpus);
        let model = make_tiny_model(tokenizer.vocab_size());
        let (train_ds, val_ds) = TextDataset::train_val_split(
            corpus, &tokenizer, model.config.context_length
        );
        let cfg = TrainConfig2 {
            max_steps: 10,
            eval_interval: 10,
            learning_rate: 1e-2,
            grad_clip: 1.0,
            accumulate_steps: 2,
            batch_size: 1,
            label_smoothing: 0.0,
            checkpoint_path: None,
            early_stopping_patience: 0,
        };
        let loss = train2(&model, &train_ds, &val_ds, &cfg);
        assert!(loss.is_finite(), "training with grad accumulation should be finite");
    }

    // --- Batch training ---

    #[test]
    fn test_sample_batch_returns_b_sequences() {
        use crate::tokenizer::CharTokenizer;
        use crate::dataset::{TextDataset, DataSource};

        let text = "abcdefghijklmnopqrstuvwxyz".repeat(5);
        let tok = CharTokenizer::from_text(&text);
        let ds = TextDataset::from_text(&text, &tok, 4);
        let batch = ds.sample_batch(42, 3);
        assert_eq!(batch.len(), 3);
        for (inp, tgt) in &batch {
            assert_eq!(inp.len(), 4);
            assert_eq!(tgt.len(), 4);
        }
    }

    #[test]
    fn test_sample_batch_sequences_differ() {
        use crate::tokenizer::CharTokenizer;
        use crate::dataset::{TextDataset, DataSource};

        let text = "abcdefghijklmnopqrstuvwxyz".repeat(5);
        let tok = CharTokenizer::from_text(&text);
        let ds = TextDataset::from_text(&text, &tok, 4);
        let batch = ds.sample_batch(42, 3);
        // The three sequences should not all be identical
        let all_same = batch.windows(2).all(|w| w[0].0 == w[1].0);
        assert!(!all_same, "batch sequences should differ");
    }

    #[test]
    fn test_loss_batch_tokens_finite() {
        use crate::tokenizer::{CharTokenizer, Tokenizer};
        use crate::dataset::{TextDataset, DataSource};

        let corpus = "abcabcabc".repeat(20);
        let tok = CharTokenizer::from_text(&corpus);
        let model = make_tiny_model(tok.vocab_size());
        let ds = TextDataset::from_text(&corpus, &tok, model.config.context_length);

        let batch_raw = ds.sample_batch(7, 3);
        let batch_refs: Vec<(&[usize], &[usize])> = batch_raw.iter()
            .map(|(i, t)| (i.as_slice(), t.as_slice()))
            .collect();

        for p in model.parameters() { p.zero_grad(); }
        let loss = model.loss_batch_tokens(&batch_refs);
        assert!(loss.data().at(0, 0).is_finite(), "batch loss should be finite");
        assert!(loss.data().at(0, 0) > 0.0, "batch loss should be positive");
    }

    #[test]
    fn test_training_with_batch_size_4() {
        use crate::tokenizer::{CharTokenizer, Tokenizer};
        use crate::dataset::TextDataset;

        let corpus = "abcabcabc".repeat(20);
        let corpus = corpus.as_str();
        let tokenizer = CharTokenizer::from_text(corpus);
        let model = make_tiny_model(tokenizer.vocab_size());
        let (train_ds, val_ds) = TextDataset::train_val_split(
            corpus, &tokenizer, model.config.context_length
        );
        let cfg = TrainConfig2 {
            max_steps: 20,
            eval_interval: 20,
            learning_rate: 1e-2,
            grad_clip: 1.0,
            accumulate_steps: 1,
            batch_size: 4,
            label_smoothing: 0.0,
            checkpoint_path: None,
            early_stopping_patience: 0,
        };
        let loss = train2(&model, &train_ds, &val_ds, &cfg);
        assert!(loss.is_finite() && loss > 0.0,
            "batch training should produce finite loss, got {}", loss);
    }
}
