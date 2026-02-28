#!/usr/bin/env python3
"""
Numpy reference for Gemma3-4b decode step.

Runs prefill with the 16-token chat-template prompt for "What is 2+2?",
then runs one decode step for token 236778 ("2") — the first generated token.

Prints per-layer hidden-state stats so we can compare with the Rust model.
"""

import numpy as np
import json
import struct
import os
import sys

# ── Config ─────────────────────────────────────────────────────────────────

WEIGHTS_DIR = "models/gemma-3-4b-it"
INDEX_FILE  = f"{WEIGHTS_DIR}/model.safetensors.index.json"

# Gemma3-4b config
VOCAB       = 262208
HIDDEN      = 2560
N_LAYERS    = 34
N_Q_HEADS   = 8
N_KV_HEADS  = 4
HEAD_DIM    = 256
INTERMEDIATE= 10240
WINDOW      = 1024
THETA_LOCAL = 10_000.0
THETA_GLOBAL= 8_000_000.0
SCALE       = 1.0 / (256.0 ** 0.5)   # attn_scale = 1/sqrt(query_pre_attn_scalar)
EMBED_SCALE = HIDDEN ** 0.5           # embedding scaling
EPS         = 1e-6

# Prompt: chat template + "What is 2+2?"
# [bos, start_of_turn, user, \n, What, _is, _, 2, +, 2, ?, end_of_turn, \n, start_of_turn, model, \n]
PROMPT_TOKENS = [2, 105, 2364, 107, 3689, 563, 236743, 236778, 236862, 236778, 236881, 106, 107, 105, 4368, 107]
DECODE_TOKEN  = 236778   # "2" — first generated token; we generate step 2

# ── Safetensors loading ─────────────────────────────────────────────────────

def bf16_to_f32_arr(raw_bytes):
    """Convert raw BF16 bytes to float32 numpy array."""
    u16 = np.frombuffer(raw_bytes, dtype='<u2')
    return (u16.astype(np.uint32) << 16).view(np.float32)

_st_cache = {}   # path → (header_len, header_dict)

def _open_st(path):
    if path not in _st_cache:
        with open(path, 'rb') as f:
            hlen = struct.unpack('<Q', f.read(8))[0]
            hdr  = json.loads(f.read(hlen))
        _st_cache[path] = (hlen, hdr)
    return _st_cache[path]

def load_tensor(name):
    """Load one tensor by name using the index.json shard map."""
    with open(INDEX_FILE) as f:
        index = json.load(f)
    shard_file = index['weight_map'].get(name)
    if shard_file is None:
        raise KeyError(f"Tensor {name!r} not found in index")
    path = f"{WEIGHTS_DIR}/{shard_file}"
    hlen, hdr = _open_st(path)
    info = hdr[name]
    offs = info['data_offsets']
    dtype = info['dtype']
    shape = info['shape']
    with open(path, 'rb') as f:
        f.seek(8 + hlen + offs[0])
        raw = f.read(offs[1] - offs[0])
    if dtype == 'BF16':
        arr = bf16_to_f32_arr(raw).reshape(shape)
    elif dtype == 'F32':
        arr = np.frombuffer(raw, dtype='<f4').reshape(shape).copy()
    else:
        raise ValueError(f"Unsupported dtype {dtype} for {name}")
    return arr.astype(np.float32)

# ── Prefix for HF tensor names ───────────────────────────────────────────────

PFX = "language_model.model"

def layer_pfx(i):
    return f"{PFX}.layers.{i}"

# ── Math helpers ─────────────────────────────────────────────────────────────

def rms_norm_gemma3(x, gamma, eps=EPS):
    """x: [T, D], gamma: [D]  →  out: [T, D] using (1+gamma) variant."""
    sq_mean = (x ** 2).mean(axis=-1, keepdims=True)
    x_hat   = x / np.sqrt(sq_mean + eps)
    return x_hat * (1.0 + gamma)

def gelu_tanh(x):
    """PyTorch-style approximate GELU."""
    c = np.sqrt(2.0 / np.pi)
    return x * 0.5 * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))

def rope_rotate(x, n_heads, n_new, head_dim, theta, offset):
    """Apply RoPE at absolute positions offset..offset+n_new.
    x: [n_new, n_heads * head_dim]
    Returns same shape.
    """
    out = x.copy()
    pairs = head_dim // 2
    for row in range(n_new):
        pos = offset + row
        for i in range(pairs):
            angle = pos / (theta ** (2.0 * i / head_dim))
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for h in range(n_heads):
                c0 = h * head_dim + 2 * i
                c1 = h * head_dim + 2 * i + 1
                x0 = x[row, c0]
                x1 = x[row, c1]
                out[row, c0] = x0 * cos_a - x1 * sin_a
                out[row, c1] = x1 * cos_a + x0 * sin_a
    return out

def per_head_norm(x, gamma, n_heads, head_dim, eps=EPS):
    """Per-head RMSNorm (Gemma3 variant).
    x: [T, n_heads * head_dim], gamma: [head_dim]
    """
    T = x.shape[0]
    out = np.zeros_like(x)
    for h in range(n_heads):
        sl = slice(h * head_dim, (h + 1) * head_dim)
        head = x[:, sl]
        sq_mean = (head ** 2).mean(axis=-1, keepdims=True)
        normed  = head / np.sqrt(sq_mean + eps)
        out[:, sl] = normed * (1.0 + gamma)
    return out

def gqa_attention(q, k_cache, v_cache, k_start, k_end, n_q, n_kv, d, scale, causal=False, t_q_offset=0):
    """GQA attention.
    q:       [t_q, n_q * d]
    k_cache: [max_seq, n_kv * d]
    v_cache: [max_seq, n_kv * d]
    Returns: [t_q, n_q * d]
    """
    t_q  = q.shape[0]
    t_kv = k_end - k_start
    group = n_q // n_kv
    kv_stride = n_kv * d
    out  = np.zeros((t_q, n_q * d), dtype=np.float32)

    for qh in range(n_q):
        kvh    = qh // group
        kv_off = kvh * d
        q_h    = q[:, qh * d : (qh + 1) * d]               # [t_q, d]
        k_h    = k_cache[k_start:k_end, kv_off:kv_off + d] # [t_kv, d]
        v_h    = v_cache[k_start:k_end, kv_off:kv_off + d] # [t_kv, d]

        scores = (q_h @ k_h.T) * scale   # [t_q, t_kv]

        if causal:
            # Mask: query r (absolute pos = t_q_offset + r) can attend to
            # keys at absolute positions k_start..k_end-1 ≤ t_q_offset+r
            for r in range(t_q):
                abs_r = t_q_offset + r
                max_kv = abs_r - k_start   # largest k index allowed
                if max_kv < 0:
                    scores[r, :] = -1e30
                else:
                    max_kv = min(max_kv, t_kv - 1)
                    scores[r, max_kv + 1:] = -1e30

        # softmax per row
        s_max  = scores.max(axis=-1, keepdims=True)
        exp_s  = np.exp(scores - s_max)
        exp_s  = exp_s / exp_s.sum(axis=-1, keepdims=True)

        out[:, qh * d:(qh + 1) * d] = exp_s @ v_h

    return out

# ── Layer forward ─────────────────────────────────────────────────────────────

class LayerWeights:
    pass

def load_layer(i):
    lw = LayerWeights()
    pfx = layer_pfx(i)
    lw.input_norm_gamma        = load_tensor(f"{pfx}.input_layernorm.weight")
    lw.post_attn_norm_gamma    = load_tensor(f"{pfx}.post_attention_layernorm.weight")
    lw.pre_ffn_norm_gamma      = load_tensor(f"{pfx}.pre_feedforward_layernorm.weight")
    lw.post_ffn_norm_gamma     = load_tensor(f"{pfx}.post_feedforward_layernorm.weight")
    lw.q_norm_gamma            = load_tensor(f"{pfx}.self_attn.q_norm.weight")
    lw.k_norm_gamma            = load_tensor(f"{pfx}.self_attn.k_norm.weight")
    lw.q_proj                  = load_tensor(f"{pfx}.self_attn.q_proj.weight")   # [nq*d, hidden]
    lw.k_proj                  = load_tensor(f"{pfx}.self_attn.k_proj.weight")   # [nkv*d, hidden]
    lw.v_proj                  = load_tensor(f"{pfx}.self_attn.v_proj.weight")   # [nkv*d, hidden]
    lw.o_proj                  = load_tensor(f"{pfx}.self_attn.o_proj.weight")   # [hidden, nq*d]
    lw.gate_proj               = load_tensor(f"{pfx}.mlp.gate_proj.weight")      # [inter, hidden]
    lw.up_proj                 = load_tensor(f"{pfx}.mlp.up_proj.weight")        # [inter, hidden]
    lw.down_proj               = load_tensor(f"{pfx}.mlp.down_proj.weight")      # [hidden, inter]
    is_global                  = (i % 6 == 5)
    lw.rope_theta              = THETA_GLOBAL if is_global else THETA_LOCAL
    lw.sliding_window          = None if is_global else WINDOW
    return lw

def layer_forward(x, lw, k_cache, v_cache, seq_offset, causal=False):
    """One Gemma3 layer forward.
    x: [t_q, hidden]
    k_cache, v_cache: pre-allocated [max_seq, n_kv*d], modified in-place
    seq_offset: current cache seq_len before this step
    Returns (new_x, new_seq_len)
    """
    n_new = x.shape[0]
    t_q   = n_new

    # 1. Input norm
    normed = rms_norm_gemma3(x, lw.input_norm_gamma)

    # 2. Projections: [t_q, nq*d], [t_q, nkv*d]
    q = normed @ lw.q_proj.T
    k = normed @ lw.k_proj.T
    v = normed @ lw.v_proj.T

    # 3. Per-head RMSNorm on Q/K
    q = per_head_norm(q, lw.q_norm_gamma, N_Q_HEADS, HEAD_DIM)
    k = per_head_norm(k, lw.k_norm_gamma, N_KV_HEADS, HEAD_DIM)

    # 4. RoPE
    q = rope_rotate(q, N_Q_HEADS,  n_new, HEAD_DIM, lw.rope_theta, seq_offset)
    k = rope_rotate(k, N_KV_HEADS, n_new, HEAD_DIM, lw.rope_theta, seq_offset)

    # 5. Append to cache
    k_cache[seq_offset:seq_offset + n_new] = k
    v_cache[seq_offset:seq_offset + n_new] = v
    new_seq_len = seq_offset + n_new

    # 6. Context window
    k_end   = new_seq_len
    k_start = (k_end - lw.sliding_window) if lw.sliding_window else 0
    k_start = max(k_start, 0)

    # 7. GQA attention
    attn_out = gqa_attention(q, k_cache, v_cache, k_start, k_end,
                              N_Q_HEADS, N_KV_HEADS, HEAD_DIM, SCALE,
                              causal=causal, t_q_offset=seq_offset)

    # 8. Output projection
    attn_out = attn_out @ lw.o_proj.T

    # 9. Post-attn norm + residual
    attn_out = rms_norm_gemma3(attn_out, lw.post_attn_norm_gamma)
    x2 = x + attn_out

    # 10. Pre-FFN norm
    normed2 = rms_norm_gemma3(x2, lw.pre_ffn_norm_gamma)

    # 11. MLP (gelu_tanh gate)
    gate    = gelu_tanh(normed2 @ lw.gate_proj.T)
    up      = normed2 @ lw.up_proj.T
    hidden  = gate * up
    mlp_out = hidden @ lw.down_proj.T

    # 12. Post-FFN norm + residual
    mlp_out = rms_norm_gemma3(mlp_out, lw.post_ffn_norm_gamma)
    x3 = x2 + mlp_out

    return x3, new_seq_len

# ── Main ──────────────────────────────────────────────────────────────────────

print("Loading embed_tokens...", flush=True)
embed = load_tensor(f"{PFX}.embed_tokens.weight")   # [vocab, hidden]
print(f"  embed shape: {embed.shape}", flush=True)

print("Loading final norm...", flush=True)
final_norm_gamma = load_tensor(f"{PFX}.norm.weight")

print(f"Running prefill ({len(PROMPT_TOKENS)} tokens)...", flush=True)

# Initial embeddings (scaled)
embed_scale = float(HIDDEN ** 0.5)
x = embed[PROMPT_TOKENS] * embed_scale   # [t_prompt, hidden]

# KV caches for all layers: [max_seq, n_kv * head_dim]
max_seq = 2048
kv_shape = (max_seq, N_KV_HEADS * HEAD_DIM)
k_caches = [np.zeros(kv_shape, dtype=np.float32) for _ in range(N_LAYERS)]
v_caches = [np.zeros(kv_shape, dtype=np.float32) for _ in range(N_LAYERS)]
seq_lens  = [0] * N_LAYERS

t_prompt = len(PROMPT_TOKENS)

# Prefill: run all tokens through all layers
print("Loading and running layers for prefill...", flush=True)
layers = []
for i in range(N_LAYERS):
    sys.stdout.write(f"\r  Layer {i:02d}/{N_LAYERS}..."); sys.stdout.flush()
    lw = load_layer(i)
    layers.append(lw)
    x, seq_lens[i] = layer_forward(x, lw, k_caches[i], v_caches[i],
                                    seq_offset=0, causal=True)
print(f"\nPrefill done. seq_lens[0]={seq_lens[0]}", flush=True)

# After prefill, sample first token (just pick argmax from last row)
normed_last = rms_norm_gemma3(x[[-1], :], final_norm_gamma)   # [1, hidden]
logits_pre  = normed_last @ embed.T                              # [1, vocab]
prefill_top = np.argsort(logits_pre[0])[::-1][:5]
print(f"Prefill top-5 tokens: {[(int(t), round(float(logits_pre[0, t]), 3)) for t in prefill_top]}", flush=True)
print(f"(Expected first token: 236778='2')", flush=True)

# ── Decode step 1 ─────────────────────────────────────────────────────────────
print(f"\nRunning decode step 1 with token {DECODE_TOKEN}...", flush=True)

x_dec = embed[[DECODE_TOKEN]] * embed_scale   # [1, hidden]
vals = x_dec[0, :8]
print(f"[ dec1 embed tok={DECODE_TOKEN} ] first8={[round(v, 5) for v in vals.tolist()]}", flush=True)

for i in range(N_LAYERS):
    x_dec, seq_lens[i] = layer_forward(
        x_dec, layers[i], k_caches[i], v_caches[i],
        seq_offset=seq_lens[i],   # = t_prompt (after prefill)
        causal=False,             # decode: no causal mask needed (single query)
    )
    vals  = x_dec[0, :8]
    mean  = x_dec.mean()
    std   = x_dec.std()
    print(f"[ dec1 L{i:02d} ] first8={[round(v, 5) for v in vals.tolist()]}  mean={mean:.4f}  std={std:.4f}", flush=True)

# Final prediction
normed_dec = rms_norm_gemma3(x_dec, final_norm_gamma)
logits_dec = normed_dec @ embed.T   # [1, vocab]
top_idx    = np.argsort(logits_dec[0])[::-1][:5]
print(f"\nDecode step 1 top-5 tokens: {[(int(t), round(float(logits_dec[0, t]), 3)) for t in top_idx]}", flush=True)
print("(Expected: token 900 = ' +' should be in top-1)", flush=True)
