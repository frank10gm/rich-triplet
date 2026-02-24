#!/usr/bin/env python3
"""Minimal GGUF norm weight dumper.

Reads the tensor index from the GGUF file and prints the first 10 values
of every norm-related tensor so we can verify what the GGUF actually stores.
"""

import os
import struct
import sys

GGUF_PATH = "models/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf"

# ── low-level helpers ──────────────────────────────────────────────────────


def _ru8(f):
    return struct.unpack("<B", f.read(1))[0]


def _ru16(f):
    return struct.unpack("<H", f.read(2))[0]


def _ru32(f):
    return struct.unpack("<I", f.read(4))[0]


def _ru64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _ri8(f):
    return struct.unpack("<b", f.read(1))[0]


def _ri16(f):
    return struct.unpack("<h", f.read(2))[0]


def _ri32(f):
    return struct.unpack("<i", f.read(4))[0]


def _ri64(f):
    return struct.unpack("<q", f.read(8))[0]


def _rf32(f):
    return struct.unpack("<f", f.read(4))[0]


def _rf64(f):
    return struct.unpack("<d", f.read(8))[0]


def _rstr(f):
    n = _ru64(f)
    return f.read(n).decode("utf-8", errors="replace")


def f16_to_f32(bits):
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    if e == 0:
        v = (m / 1024.0) * (2.0**-14)
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1.0 + m / 1024.0) * (2.0 ** (e - 15))
    return -v if s else v


# ── skip a single metadata value without decoding it ──────────────────────
# We need this to be robust; the tricky types are STRING (8) and ARRAY (10).

# GGUF v3 value types (from spec):
#   0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
#   6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
_SCALAR_SIZES = {
    0: 1,  # UINT8
    1: 1,  # INT8
    2: 2,  # UINT16
    3: 2,  # INT16
    4: 4,  # UINT32
    5: 4,  # INT32
    6: 4,  # FLOAT32
    7: 1,  # BOOL
    10: 8,  # UINT64
    11: 8,  # INT64
    12: 8,  # FLOAT64
}


def _skip_value(f, vtype):
    """Skip one metadata value of the given type."""
    if vtype in _SCALAR_SIZES:
        f.seek(_SCALAR_SIZES[vtype], 1)
    elif vtype == 8:  # STRING
        n = _ru64(f)
        f.seek(n, 1)
    elif vtype == 9:  # ARRAY
        elem_type = _ru32(f)
        count = _ru64(f)
        if elem_type in _SCALAR_SIZES:
            f.seek(_SCALAR_SIZES[elem_type] * count, 1)
        else:
            # strings or nested arrays — iterate
            for _ in range(count):
                _skip_value(f, elem_type)
    else:
        raise ValueError(f"Unknown metadata value type: {vtype}")


# ── GGUF index reader ──────────────────────────────────────────────────────

DTYPE_NAMES = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    30: "BF16",
}


def read_gguf_index(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != b"GGUF":
            raise ValueError(f"Not a GGUF file (magic={magic!r})")
        version = _ru32(f)
        n_tensors = _ru64(f)
        n_meta = _ru64(f)

        print(f"GGUF  version={version}  tensors={n_tensors}  meta_entries={n_meta}")

        for i in range(n_meta):
            key = _rstr(f)
            vtype = _ru32(f)
            _skip_value(f, vtype)

        tensors = []
        for _ in range(n_tensors):
            name = _rstr(f)
            ndims = _ru32(f)
            dims = [_ru64(f) for _ in range(ndims)]
            dtype = _ru32(f)
            off = _ru64(f)
            tensors.append((name, dims, dtype, off))

        pos = f.tell()
        data_start = (pos + 31) & ~31  # 32-byte aligned

    return tensors, data_start


# ── tensor value readers ───────────────────────────────────────────────────


def read_f32_vals(path, data_start, offset, n):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [_rf32(f) for _ in range(n)]


def read_f16_vals(path, data_start, offset, n):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [f16_to_f32(_ru16(f)) for _ in range(n)]


# ── safetensors helpers ────────────────────────────────────────────────────

import json


def bf16_to_f32(bits):
    return struct.unpack("f", struct.pack("I", bits << 16))[0]


def read_st_tensor(path, name, n=10):
    try:
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            if name not in header:
                return None
            info = header[name]
            offs = info["data_offsets"]
            dtype = info["dtype"]
            f.seek(8 + hlen + offs[0])
            raw = f.read(offs[1] - offs[0])
        if dtype == "F32":
            vals = list(struct.unpack(f"<{len(raw) // 4}f", raw))
        elif dtype == "BF16":
            import array

            vals = [bf16_to_f32(x) for x in array.array("H", raw)]
        elif dtype == "F16":
            import array

            vals = [f16_to_f32(x) for x in array.array("H", raw)]
        else:
            return None
        return dtype, vals[:n]
    except Exception:
        return None


# ── main ───────────────────────────────────────────────────────────────────

SF_PATHS = [
    "models/gemma-3-4b-it/model-00001-of-00002.safetensors",
    "models/gemma-3-4b-it/model-00002-of-00002.safetensors",
]

# Which GGUF tensors are "norm" tensors we want to inspect
NORM_KEYWORDS = [
    "norm.weight",  # matches attn_norm, ffn_norm, output_norm, etc.
    "q_norm",
    "k_norm",
]

# GGUF name → safetensors name (for layers 0..2, plus output_norm)
SF_MAP = {
    "blk.0.attn_norm.weight": "language_model.model.layers.0.input_layernorm.weight",
    "blk.0.post_attn_norm.weight": "language_model.model.layers.0.post_attention_layernorm.weight",
    "blk.0.ffn_norm.weight": "language_model.model.layers.0.pre_feedforward_layernorm.weight",
    "blk.0.ffn_post_norm.weight": "language_model.model.layers.0.post_feedforward_layernorm.weight",
    "blk.0.attn_q_norm.weight": "language_model.model.layers.0.self_attn.q_norm.weight",
    "blk.0.attn_k_norm.weight": "language_model.model.layers.0.self_attn.k_norm.weight",
    "blk.1.attn_norm.weight": "language_model.model.layers.1.input_layernorm.weight",
    "blk.1.post_attn_norm.weight": "language_model.model.layers.1.post_attention_layernorm.weight",
    "blk.1.ffn_norm.weight": "language_model.model.layers.1.pre_feedforward_layernorm.weight",
    "blk.1.ffn_post_norm.weight": "language_model.model.layers.1.post_feedforward_layernorm.weight",
    "output_norm.weight": "language_model.model.norm.weight",
}

N = 10

print("=" * 72)
print(f"Reading {GGUF_PATH}")
print("=" * 72)

tensors, data_start = read_gguf_index(GGUF_PATH)
gguf_map = {name: (dims, dtype, off) for name, dims, dtype, off in tensors}
print(f"data_start = {data_start}\n")

# Print all norm tensors found in the GGUF
print("All norm-like tensors in GGUF:")
for name, dims, dtype, off in tensors:
    is_norm = any(kw in name for kw in NORM_KEYWORDS)
    if is_norm:
        dname = DTYPE_NAMES.get(dtype, f"type{dtype}")
        print(f"  {name}  dims={dims}  dtype={dname}  offset={off}")
print()

# Detailed comparison for known pairs
print("=" * 72)
print("Detailed value comparison (GGUF raw vs safetensors):")
print("=" * 72)

for gguf_name, sf_name in SF_MAP.items():
    if gguf_name not in gguf_map:
        print(f"\n[MISSING] {gguf_name}")
        continue

    dims, dtype, off = gguf_map[gguf_name]
    dname = DTYPE_NAMES.get(dtype, f"type{dtype}")
    n_elem = 1
    for d in dims:
        n_elem *= d
    k = min(N, n_elem)

    if dtype == 0:  # F32
        gguf_vals = read_f32_vals(GGUF_PATH, data_start, off, k)
    elif dtype == 1:  # F16
        gguf_vals = read_f16_vals(GGUF_PATH, data_start, off, k)
    else:
        print(f"\n{gguf_name}: dtype={dname} not decoded")
        continue

    # Look for matching safetensors value
    sf_result = None
    for sf_path in SF_PATHS:
        if os.path.exists(sf_path):
            r = read_st_tensor(sf_path, sf_name, k)
            if r is not None:
                sf_result = r
                break

    short_gguf = gguf_name.split("blk.")[-1] if "blk." in gguf_name else gguf_name
    print(f"\n--- {short_gguf}  [{dname}]")
    gv = [round(v, 6) for v in gguf_vals]
    print(f"  GGUF raw       : {gv}")
    print(f"  GGUF (raw-1)   : {[round(v - 1, 6) for v in gguf_vals]}")
    print(f"  GGUF (1+raw)   : {[round(1 + v, 6) for v in gguf_vals]}")

    if sf_result:
        sf_dtype, sf_vals = sf_result
        sv = [round(v, 6) for v in sf_vals]
        print(f"  SafeTensors    : {sv}  [{sf_dtype}]")
        diff_raw = [abs(g - s) for g, s in zip(gguf_vals, sf_vals)]
        diff_sub1 = [abs((g - 1) - s) for g, s in zip(gguf_vals, sf_vals)]
        diff_add1 = [abs((g + 1) - s) for g, s in zip(gguf_vals, sf_vals)]
        md_raw = sum(diff_raw) / len(diff_raw)
        md_sub1 = sum(diff_sub1) / len(diff_sub1)
        md_add1 = sum(diff_add1) / len(diff_add1)
        print(f"  mean|gguf-sf|       = {md_raw:.6f}")
        print(f"  mean|(gguf-1)-sf|   = {md_sub1:.6f}")
        print(f"  mean|(gguf+1)-sf|   = {md_add1:.6f}")
        best = min(
            ("gguf==sf", md_raw),
            ("gguf-1==sf", md_sub1),
            ("gguf+1==sf", md_add1),
            key=lambda x: x[1],
        )
        print(f"  *** BEST FIT: {best[0]}  (mean_err={best[1]:.6f})")
        if best[0] == "gguf==sf":
            print("      => GGUF stores weight directly (NO -1 fix needed)")
        elif best[0] == "gguf-1==sf":
            print("      => GGUF stores (1+weight), so -1 fix IS correct")
        else:
            print("      => GGUF stores (weight-1), so +1 fix needed (!)")
    else:
        print(f"  SafeTensors: NOT FOUND ({sf_name})")
