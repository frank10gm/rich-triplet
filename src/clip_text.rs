#![allow(dead_code)]
// =============================================================================
// CLIP-L text encoder
// =============================================================================
//
// The smaller of FLUX's two text encoders, and the one whose output is a single
// vector rather than a sequence. T5 carries the prompt's content; this carries
// something closer to its overall register, and it enters the transformer
// through the same modulation path as the timestep.
//
// Twelve layers, width 768, twelve heads, learned positional embeddings, 77
// tokens. Ordinary except in three places:
//
// ## The mask is causal
//
// A text *encoder* that reads left to right looks like a mistake and is not.
// CLIP trains its text tower autoregressively-masked and pools from the last
// token, so bidirectional attention here changes every embedding it produces.
//
// ## The activation is `quick_gelu`
//
//   x * sigmoid(1.702 * x)
//
// A pre-tanh approximation of GELU that OpenAI trained with and that survives
// in the checkpoint. Substituting exact GELU or the tanh approximation shifts
// the pooled vector by a few percent -- enough to change which image a prompt
// produces, not enough to look broken.
//
// ## Pooling reads the end-of-text position
//
// The pooled vector is the final-layer-norm hidden state at the position of the
// EOT token, found by taking the **argmax over the token ids**. That works
// because EOT (49407) is the highest id in the vocabulary, and it is how
// diffusers does it. Taking the last position instead picks up padding, since
// the sequence is padded to 77 with EOT itself -- so the two agree only when
// the prompt is exactly 77 tokens long.
//
// FLUX uses the pooled output only. The per-token sequence is computed anyway
// because it is what the pooling reads.

use std::collections::HashMap;

use crate::autograd2::Mat;
use crate::conv2d::quick_gelu_inplace;
use crate::qlinear::QLinear;
use crate::transformer3::{parse_safetensors_header, SafeTensor};
// The widening reader lives with the VAE loader; see its note on why the
// `transformer3` one is not used for the f32 conversion.
use crate::vae::read_tensor_f32;

// =============================================================================
// Config
// =============================================================================

#[derive(Clone, Debug, PartialEq)]
pub struct ClipTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub max_position_embeddings: usize,
    /// The highest id in the vocabulary, and the one pooling searches for.
    pub eot_token_id: u32,
    pub layer_norm_eps: f32,
}

impl Default for ClipTextConfig {
    fn default() -> Self {
        ClipTextConfig {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            n_layers: 12,
            n_heads: 12,
            max_position_embeddings: 77,
            eot_token_id: 49407,
            layer_norm_eps: 1e-5,
        }
    }
}

impl ClipTextConfig {
    /// openai/clip-vit-large-patch14, which is what FLUX and SD3 both use.
    pub fn large() -> Self {
        ClipTextConfig {
            vocab_size: 49408,
            hidden_size: 768,
            intermediate_size: 3072,
            n_layers: 12,
            n_heads: 12,
            max_position_embeddings: 77,
            eot_token_id: 49407,
            layer_norm_eps: 1e-5,
        }
    }
}

// =============================================================================
// Norm
// =============================================================================

/// LayerNorm with affine parameters, accumulating in f64.
pub fn clip_layer_norm(x: &Mat, weight: &[f32], bias: &[f32], eps: f32) -> Mat {
    debug_assert!(weight.len() == x.cols, "clip_layer_norm: weight length != hidden_size");
    let cols = x.cols;
    let mut out = Mat::zeros(x.rows, cols);
    let n = cols as f64;
    for r in 0..x.rows {
        let src = &x.data[r * cols..(r + 1) * cols];
        let mut sum = 0.0f64;
        for c in 0..cols {
            sum += src[c] as f64;
        }
        let mean = sum / n;
        let mut var = 0.0f64;
        for c in 0..cols {
            let d = src[c] as f64 - mean;
            var += d * d;
        }
        let inv = (1.0 / (var / n + eps as f64).sqrt()) as f32;
        let mean_f = mean as f32;
        let dst = &mut out.data[r * cols..(r + 1) * cols];
        for c in 0..cols {
            dst[c] = (src[c] - mean_f) * inv * weight[c] + if bias.is_empty() { 0.0 } else { bias[c] };
        }
    }
    out
}

// =============================================================================
// Layers
// =============================================================================

#[derive(Clone, Default)]
pub struct ClipAttention {
    pub q: QLinear,
    pub k: QLinear,
    pub v: QLinear,
    pub out: QLinear,
    pub n_heads: usize,
    pub head_dim: usize,
}

impl ClipAttention {
    /// Causal, and scaled by `1/sqrt(head_dim)` -- unlike T5 next door, which
    /// is neither.
    pub fn forward(&self, x: &Mat) -> Mat {
        let t = x.rows;
        let head_dim = self.head_dim;
        let inner = self.n_heads * head_dim;

        let q = self.q.forward(x);
        let k = self.k.forward(x);
        let v = self.v.forward(x);
        debug_assert!(q.cols == inner, "clip attention: projection width != n_heads * head_dim");

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        let mut context = Mat::zeros(t, inner);
        let mut scores = vec![0.0f32; t];
        for h in 0..self.n_heads {
            let off = h * head_dim;
            for i in 0..t {
                let qi = &q.data[i * q.cols + off..i * q.cols + off + head_dim];
                let mut max_score = f32::NEG_INFINITY;
                // Causal: position `i` reads keys 0..i. CLIP's text tower is
                // masked even though nothing downstream generates text.
                for j in 0..=i {
                    let kj = &k.data[j * k.cols + off..j * k.cols + off + head_dim];
                    let mut acc = 0.0f32;
                    for c in 0..head_dim {
                        acc += qi[c] * kj[c];
                    }
                    scores[j] = acc * scale;
                    if max_score < scores[j] {
                        max_score = scores[j];
                    }
                }
                let mut denom = 0.0f32;
                for j in 0..=i {
                    scores[j] = (scores[j] - max_score).exp();
                    denom += scores[j];
                }
                let inv = 1.0f32 / denom;
                let dst = &mut context.data[i * inner + off..i * inner + off + head_dim];
                for j in 0..=i {
                    let weight = scores[j] * inv;
                    let vj = &v.data[j * v.cols + off..j * v.cols + off + head_dim];
                    for c in 0..head_dim {
                        dst[c] += weight * vj[c];
                    }
                }
            }
        }
        self.out.forward(&context)
    }
}

#[derive(Clone, Default)]
pub struct ClipMlp {
    pub fc1: QLinear,
    pub fc2: QLinear,
}

impl ClipMlp {
    pub fn forward(&self, x: &Mat) -> Mat {
        let mut h = self.fc1.forward(x);
        quick_gelu_inplace(&mut h);
        self.fc2.forward(&h)
    }
}

#[derive(Clone, Default)]
pub struct ClipLayer {
    pub norm1_weight: Vec<f32>,
    pub norm1_bias: Vec<f32>,
    pub attn: ClipAttention,
    pub norm2_weight: Vec<f32>,
    pub norm2_bias: Vec<f32>,
    pub mlp: ClipMlp,
}

impl ClipLayer {
    pub fn forward(&self, x: &Mat, eps: f32) -> Mat {
        let mut h = self.attn.forward(&clip_layer_norm(x, &self.norm1_weight, &self.norm1_bias, eps));
        for (d, s) in h.data.iter_mut().zip(&x.data) {
            *d += *s;
        }
        let mut m = self.mlp.forward(&clip_layer_norm(&h, &self.norm2_weight, &self.norm2_bias, eps));
        for (d, s) in m.data.iter_mut().zip(&h.data) {
            *d += *s;
        }
        m
    }
}

// =============================================================================
// Encoder
// =============================================================================

#[derive(Clone)]
pub struct ClipTextOutput {
    /// [T, hidden_size] after the final layer norm.
    pub sequence: Mat,
    /// The row of `sequence` at the EOT position. This is what FLUX consumes.
    pub pooled: Vec<f32>,
    /// Where the EOT token was found.
    pub eot_index: usize,
}

#[derive(Clone)]
pub struct ClipTextEncoder {
    pub cfg: ClipTextConfig,

    pub token_embedding: Mat,    // [vocab_size, hidden]
    pub position_embedding: Mat, // [max_positions, hidden]
    pub layers: Vec<ClipLayer>,
    pub final_norm_weight: Vec<f32>,
    pub final_norm_bias: Vec<f32>,
}

impl Default for ClipTextEncoder {
    fn default() -> Self {
        ClipTextEncoder {
            cfg: ClipTextConfig::default(),
            token_embedding: Mat::zeros(0, 0),
            position_embedding: Mat::zeros(0, 0),
            layers: Vec::new(),
            final_norm_weight: Vec::new(),
            final_norm_bias: Vec::new(),
        }
    }
}

// -----------------------------------------------------------------------------
// Loading
// -----------------------------------------------------------------------------

type TensorMap = HashMap<String, SafeTensor>;

/// Checkpoints ship under two prefixes: HuggingFace's `text_model.` and the
/// flat ComfyUI export that drops it. Try both before failing.
fn find_tensor<'a>(map: &'a mut TensorMap, suffix: &str) -> Option<&'a mut SafeTensor> {
    let prefixed = format!("text_model.{}", suffix);
    if map.contains_key(&prefixed) {
        return map.get_mut(&prefixed);
    }
    map.get_mut(suffix)
}

fn take_mat(map: &mut TensorMap, suffix: &str, rows: usize, cols: usize) -> Result<Mat, String> {
    let t = find_tensor(map, suffix).ok_or_else(|| format!("clip: missing tensor {}", suffix))?;
    if t.data.len() != rows * cols {
        return Err(format!("clip: {} has {} elements, expected {}", suffix, t.data.len(), rows * cols));
    }
    Ok(Mat::new(std::mem::take(&mut t.data), rows, cols))
}

fn take_vec(map: &mut TensorMap, suffix: &str, len: usize) -> Result<Vec<f32>, String> {
    let t = find_tensor(map, suffix).ok_or_else(|| format!("clip: missing tensor {}", suffix))?;
    if t.data.len() != len {
        return Err(format!("clip: {} has {} elements, expected {}", suffix, t.data.len(), len));
    }
    Ok(std::mem::take(&mut t.data))
}

fn take_linear(map: &mut TensorMap, prefix: &str, out_features: usize, in_features: usize) -> Result<QLinear, String> {
    let w = take_mat(map, &format!("{}.weight", prefix), out_features, in_features)?;
    let b = take_vec(map, &format!("{}.bias", prefix), out_features)?;
    Ok(QLinear::from_f32(w, b))
}

impl ClipTextEncoder {
    /// Load from a `clip_l.safetensors` in either the HuggingFace
    /// `text_model.*` layout or the flat ComfyUI one.
    pub fn load(path: &str, cfg: ClipTextConfig) -> Result<ClipTextEncoder, String> {
        let mut file = std::fs::File::open(path).map_err(|_| format!("safetensors: cannot open {}", path))?;
        let (data_offset, entries) = parse_safetensors_header(&mut file)?;

        let mut map = TensorMap::new();
        for e in &entries {
            // The vision tower shares a file in some exports and is three times
            // the size of what is wanted here.
            if e.name.contains("vision_model") {
                continue;
            }
            let t = read_tensor_f32(&mut file, data_offset, e)?;
            map.insert(e.name.clone(), t);
        }
        if map.is_empty() {
            return Err(format!("clip: no tensors in {}", path));
        }

        let mut m = ClipTextEncoder { cfg: cfg.clone(), ..Default::default() };

        m.token_embedding = take_mat(&mut map, "embeddings.token_embedding.weight", cfg.vocab_size, cfg.hidden_size)?;
        m.position_embedding = take_mat(
            &mut map,
            "embeddings.position_embedding.weight",
            cfg.max_position_embeddings,
            cfg.hidden_size,
        )?;

        let head_dim = cfg.hidden_size / cfg.n_heads;
        m.layers.reserve(cfg.n_layers);
        for i in 0..cfg.n_layers {
            let p = format!("encoder.layers.{}.", i);
            let mut layer = ClipLayer::default();

            layer.norm1_weight = take_vec(&mut map, &format!("{p}layer_norm1.weight"), cfg.hidden_size)?;
            layer.norm1_bias = take_vec(&mut map, &format!("{p}layer_norm1.bias"), cfg.hidden_size)?;

            layer.attn.q = take_linear(&mut map, &format!("{p}self_attn.q_proj"), cfg.hidden_size, cfg.hidden_size)?;
            layer.attn.k = take_linear(&mut map, &format!("{p}self_attn.k_proj"), cfg.hidden_size, cfg.hidden_size)?;
            layer.attn.v = take_linear(&mut map, &format!("{p}self_attn.v_proj"), cfg.hidden_size, cfg.hidden_size)?;
            layer.attn.out = take_linear(&mut map, &format!("{p}self_attn.out_proj"), cfg.hidden_size, cfg.hidden_size)?;
            layer.attn.n_heads = cfg.n_heads;
            layer.attn.head_dim = head_dim;

            layer.norm2_weight = take_vec(&mut map, &format!("{p}layer_norm2.weight"), cfg.hidden_size)?;
            layer.norm2_bias = take_vec(&mut map, &format!("{p}layer_norm2.bias"), cfg.hidden_size)?;

            layer.mlp.fc1 = take_linear(&mut map, &format!("{p}mlp.fc1"), cfg.intermediate_size, cfg.hidden_size)?;
            layer.mlp.fc2 = take_linear(&mut map, &format!("{p}mlp.fc2"), cfg.hidden_size, cfg.intermediate_size)?;

            m.layers.push(layer);
        }

        m.final_norm_weight = take_vec(&mut map, "final_layer_norm.weight", cfg.hidden_size)?;
        m.final_norm_bias = take_vec(&mut map, "final_layer_norm.bias", cfg.hidden_size)?;

        Ok(m)
    }

    // -------------------------------------------------------------------------
    // Forward
    // -------------------------------------------------------------------------

    /// Encode token ids.
    ///
    /// The sequence is used as given: CLIP's positional embedding table has
    /// exactly `max_position_embeddings` rows, so anything longer is an error
    /// rather than something to wrap around.
    pub fn forward(&self, tokens: &[u32]) -> Result<ClipTextOutput, String> {
        let cfg = &self.cfg;
        if tokens.is_empty() {
            return Err("clip forward: empty token sequence".to_string());
        }
        if tokens.len() > cfg.max_position_embeddings {
            return Err(format!(
                "clip forward: {} tokens exceeds the positional table's {} rows",
                tokens.len(),
                cfg.max_position_embeddings
            ));
        }
        if self.token_embedding.rows != cfg.vocab_size {
            return Err("clip forward: embedding table not loaded".to_string());
        }

        let t = tokens.len();
        let hs = cfg.hidden_size;
        let mut x = Mat::zeros(t, hs);
        for (i, &tokid) in tokens.iter().enumerate() {
            if tokid as usize >= cfg.vocab_size {
                return Err(format!("clip forward: token id {} out of range", tokid));
            }
            let tok = &self.token_embedding.data[tokid as usize * hs..(tokid as usize + 1) * hs];
            let pos = &self.position_embedding.data[i * hs..(i + 1) * hs];
            let dst = &mut x.data[i * hs..(i + 1) * hs];
            for c in 0..hs {
                dst[c] = tok[c] + pos[c];
            }
        }

        for layer in &self.layers {
            x = layer.forward(&x, cfg.layer_norm_eps);
        }

        let sequence = clip_layer_norm(&x, &self.final_norm_weight, &self.final_norm_bias, cfg.layer_norm_eps);

        // The EOT position, by argmax over the ids. Padding is EOT too, so the
        // first match is the real one and `argmax` finds it -- taking the last
        // position instead would read the tail of the padding.
        let mut eot = 0usize;
        let mut best = 0u32;
        for (i, &tokid) in tokens.iter().enumerate() {
            if tokid > best {
                best = tokid;
                eot = i;
            }
        }
        let pooled = sequence.data[eot * hs..(eot + 1) * hs].to_vec();
        Ok(ClipTextOutput { sequence, pooled, eot_index: eot })
    }

    pub fn parameter_count(&self) -> usize {
        let mut n = self.token_embedding.numel()
            + self.position_embedding.numel()
            + self.final_norm_weight.len()
            + self.final_norm_bias.len();
        for l in &self.layers {
            n += l.norm1_weight.len() + l.norm1_bias.len() + l.norm2_weight.len() + l.norm2_bias.len();
            for p in [&l.attn.q, &l.attn.k, &l.attn.v, &l.attn.out, &l.mlp.fc1, &l.mlp.fc2] {
                n += p.out_features * p.in_features + p.bias.len();
            }
        }
        n
    }

    pub fn free_weights(&mut self) {
        self.token_embedding = Mat::zeros(0, 0);
        self.position_embedding = Mat::zeros(0, 0);
        self.layers.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    fn spread(rows: usize, cols: usize, scale: f32, salt: usize) -> Mat {
        Mat::from_fn(rows, cols, |r, c| {
            let i = ((r * 131 + c * 37 + salt * 7919) % 251) as f32;
            ((i * 0.41).sin() * 0.7 + (i * 0.13).cos() * 0.3) * scale
        })
    }

    fn spread_vec(n: usize, scale: f32, salt: usize) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let k = ((i * 53 + salt * 6151) % 241) as f32;
                ((k * 0.29).sin() * 0.6 + (k * 0.17).cos() * 0.4) * scale
            })
            .collect()
    }

    fn linear(out: usize, inp: usize, salt: usize, with_bias: bool) -> QLinear {
        QLinear::from_f32(
            spread(out, inp, 0.2, salt),
            if with_bias { spread_vec(out, 0.05, salt + 1) } else { Vec::new() },
        )
    }

    fn clip_tiny() -> ClipTextConfig {
        ClipTextConfig {
            vocab_size: 40,
            hidden_size: 16,
            intermediate_size: 32,
            n_layers: 2,
            n_heads: 4,
            max_position_embeddings: 12,
            eot_token_id: 39,
            ..Default::default()
        }
    }

    fn make_clip(cfg: &ClipTextConfig) -> ClipTextEncoder {
        let mut m = ClipTextEncoder {
            cfg: cfg.clone(),
            token_embedding: spread(cfg.vocab_size, cfg.hidden_size, 1.0, 3),
            position_embedding: spread(cfg.max_position_embeddings, cfg.hidden_size, 0.3, 4),
            ..Default::default()
        };
        let head_dim = cfg.hidden_size / cfg.n_heads;
        for i in 0..cfg.n_layers {
            let mut l = ClipLayer::default();
            l.norm1_weight = vec![1.0; cfg.hidden_size];
            l.norm1_bias = vec![0.0; cfg.hidden_size];
            l.norm2_weight = vec![1.0; cfg.hidden_size];
            l.norm2_bias = vec![0.0; cfg.hidden_size];
            l.attn.q = linear(cfg.hidden_size, cfg.hidden_size, 20 + i * 8, true);
            l.attn.k = linear(cfg.hidden_size, cfg.hidden_size, 21 + i * 8, true);
            l.attn.v = linear(cfg.hidden_size, cfg.hidden_size, 22 + i * 8, true);
            l.attn.out = linear(cfg.hidden_size, cfg.hidden_size, 23 + i * 8, true);
            l.attn.n_heads = cfg.n_heads;
            l.attn.head_dim = head_dim;
            l.mlp.fc1 = linear(cfg.intermediate_size, cfg.hidden_size, 24 + i * 8, true);
            l.mlp.fc2 = linear(cfg.hidden_size, cfg.intermediate_size, 25 + i * 8, true);
            m.layers.push(l);
        }
        m.final_norm_weight = vec![1.0; cfg.hidden_size];
        m.final_norm_bias = vec![0.0; cfg.hidden_size];
        m
    }

    #[test]
    fn the_clip_l_config_matches_the_checkpoint_flux_ships_with() {
        let c = ClipTextConfig::large();
        assert_eq!(c.hidden_size, 768);
        assert_eq!(c.intermediate_size, 3072);
        assert_eq!(c.n_layers, 12);
        assert_eq!(c.n_heads, 12);
        assert_eq!(c.max_position_embeddings, 77);
        assert_eq!(c.vocab_size, 49408);
        assert_eq!(c.eot_token_id, 49407);
        // EOT is the highest id, which is what makes argmax pooling work.
        assert_eq!(c.eot_token_id as usize, c.vocab_size - 1);
    }

    #[test]
    fn the_clip_encoder_is_causal() {
        // Changing the last token must leave the first token's row untouched.
        let m = make_clip(&clip_tiny());
        let mut tokens = vec![2u32, 5, 9, 14];
        let base = m.forward(&tokens).unwrap();
        *tokens.last_mut().unwrap() = 21;
        let poked = m.forward(&tokens).unwrap();
        for c in 0..base.sequence.cols {
            assert!(approx(base.sequence.at(0, c), poked.sequence.at(0, c), 1e-5));
        }
    }

    #[test]
    fn pooling_reads_the_argmax_token_not_the_last_position() {
        let cfg = clip_tiny();
        let m = make_clip(&cfg);
        // EOT (39) in the middle, padding after it.
        let tokens = [2u32, 5, 39, 9, 9];
        let out = m.forward(&tokens).unwrap();
        assert_eq!(out.eot_index, 2);
        for c in 0..cfg.hidden_size {
            assert!(approx(out.pooled[c], out.sequence.at(2, c), 1e-7));
        }
        // The last position holds something else entirely.
        let mut delta = 0.0f32;
        for c in 0..cfg.hidden_size {
            delta = delta.max((out.pooled[c] - out.sequence.at(4, c)).abs());
        }
        assert!(delta > 1e-4);
    }

    #[test]
    fn the_clip_encoder_uses_its_positional_table() {
        // The same token at two positions must embed differently.
        let m = make_clip(&clip_tiny());
        let a = m.forward(&[7, 7, 7]).unwrap();
        let mut delta = 0.0f32;
        for c in 0..a.sequence.cols {
            delta = delta.max((a.sequence.at(0, c) - a.sequence.at(1, c)).abs());
        }
        assert!(delta > 1e-4);
    }

    #[test]
    fn the_clip_encoder_refuses_a_sequence_longer_than_its_positional_table() {
        let m = make_clip(&clip_tiny());
        let long_seq = vec![5u32; 13];
        assert!(m.forward(&long_seq).is_err());
        assert!(m.forward(&[]).is_err());
        assert!(m.forward(&[0, 999]).is_err());
    }

    #[test]
    fn clip_layer_norm_subtracts_the_mean() {
        // Unlike T5's RMSNorm next door, a constant row normalizes to zero.
        let x = Mat::new(vec![3.0; 8], 1, 8);
        let w = vec![1.0f32; 8];
        let b = vec![0.0f32; 8];
        let out = clip_layer_norm(&x, &w, &b, 1e-5);
        for c in 0..8 {
            assert!(approx(out.at(0, c), 0.0, 1e-4));
        }
    }
}
