#!/usr/bin/env python3
"""Fast GGUF norm weight reader.

Skips large metadata arrays quickly (e.g. the 262 k-entry tokenizer vocab)
so the script completes in seconds instead of minutes.
"""

import struct
import sys

# ── low-level helpers ──────────────────────────────────────────────────────


def _ru8(f):
    return struct.unpack("<B", f.read(1))[0]


def _ru16(f):
    return struct.unpack("<H", f.read(2))[0]


def _ru32(f):
    return struct.unpack("<I", f.read(4))[0]


def _ru64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _rf32(f):
    return struct.unpack("<f", f.read(4))[0]


def _rf64(f):
    return struct.unpack("<d", f.read(8))[0]


def _ri8(f):
    return struct.unpack("<b", f.read(1))[0]


def _ri16(f):
    return struct.unpack("<h", f.read(2))[0]


def _ri32(f):
    return struct.unpack("<i", f.read(4))[0]


def _ri64(f):
    return struct.unpack("<q", f.read(8))[0]


def _rstr(f):
    n = _ru64(f)
    return f.read(n).decode("utf-8", errors="replace")


# Byte sizes for each meta value type (used for fast skipping)
_FIXED_SIZES = {
    0: 1,  # uint8
    1: 1,  # int8
    2: 2,  # uint16
    3: 2,  # int16
    4: 4,  # uint32
    5: 4,  # int32
    6: 4,  # float32
    7: 8,  # uint64
    9: 1,  # bool
    11: 8,  # int64
    12: 8,  # float64
}


def _skip_val(f, vtype):
    """Skip a metadata value without decoding it (fast path for arrays)."""
    if vtype == 8:  # string
        n = _ru64(f)
        f.seek(n, 1)
    elif vtype == 10:  # array
        elem_type = _ru32(f)
        count = _ru64(f)
        if elem_type == 8:  # array of strings — variable length, must iterate
            for _ in range(count):
                n = _ru64(f)
                f.seek(n, 1)
        elif elem_type == 10:  # array of arrays — recurse
            for _ in range(count):
                _skip_val(f, 10)
        else:
            sz = _FIXED_SIZES.get(elem_type)
            if sz is None:
                raise ValueError(f"Unknown element type {elem_type} in array")
            f.seek(sz * count, 1)
    else:
        sz = _FIXED_SIZES.get(vtype)
        if sz is None:
            raise ValueError(f"Unknown value type {vtype}")
        f.seek(sz, 1)


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


# ── GGUF index reader ──────────────────────────────────────────────────────


def read_gguf_index(path):
    """Return (tensors list, data_start) where tensors = [(name, dims, dtype, offset)]."""
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"GGUF", f"Not a GGUF file: {magic!r}"
        version = _ru32(f)
        tensor_count = _ru64(f)
        meta_count = _ru64(f)

        print(f"GGUF version={version}, tensors={tensor_count}, meta={meta_count}")

        for i in range(meta_count):
            key = _rstr(f)
            vtype = _ru32(f)
            _skip_val(f, vtype)

        tensors = []
        for _ in range(tensor_count):
            name = _rstr(f)
            ndims = _ru32(f)
            dims = [_ru64(f) for _ in range(ndims)]
            dtype = _ru32(f)
            off = _ru64(f)
            tensors.append((name, dims, dtype, off))

        pos = f.tell()
        data_start = (pos + 31) & ~31  # align to 32 bytes

    return tensors, data_start


def read_f16_vals(path, data_start, offset, n):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [f16_to_f32(_ru16(f)) for _ in range(n)]


def read_f32_vals(path, data_start, offset, n):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [_rf32(f) for _ in range(n)]


# ── safetensors reader ─────────────────────────────────────────────────────

import json


def bf16_to_f32(bits):
    b = struct.pack("I", bits << 16)
    return struct.unpack("f", b)[0]


def read_st_tensor(path, name, n=10):
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
    count = len(raw) // (4 if dtype == "F32" else 2)
    if dtype == "F32":
        vals = list(struct.unpack(f"<{count}f", raw))
    elif dtype == "BF16":
        import array

        vals = [bf16_to_f32(x) for x in array.array("H", raw)]
    else:
        return None
    return info["shape"], dtype, vals[:n]


# ── main ───────────────────────────────────────────────────────────────────

GGUF_PATH = "models/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf"
SF_PATHS = [
    "models/gemma-3-4b-it/model-00001-of-00002.safetensors",
    "models/gemma-3-4b-it/model-00002-of-00002.safetensors",
]

# Map: gguf_name → safetensors_name
NORM_MAP = [
    ("blk.0.attn_norm.weight", "language_model.model.layers.0.input_layernorm.weight"),
    (
        "blk.0.post_attn_norm.weight",
        "language_model.model.layers.0.post_attention_layernorm.weight",
    ),
    (
        "blk.0.ffn_norm.weight",
        "language_model.model.layers.0.pre_feedforward_layernorm.weight",
    ),
    (
        "blk.0.ffn_post_norm.weight",
        "language_model.model.layers.0.post_feedforward_layernorm.weight",
    ),
    (
        "blk.0.attn_q_norm.weight",
        "language_model.model.layers.0.self_attn.q_norm.weight",
    ),
    (
        "blk.0.attn_k_norm.weight",
        "language_model.model.layers.0.self_attn.k_norm.weight",
    ),
    ("output_norm.weight", "language_model.model.norm.weight"),
]

N = 8

print("Building GGUF tensor index (fast skip of metadata)...")
tensors, data_start = read_gguf_index(GGUF_PATH)
gguf_map = {name: (dims, dtype, off) for name, dims, dtype, off in tensors}
print(f"data_start offset = {data_start}\n")

DTYPE_NAMES = {0: "F32", 1: "F16", 30: "BF16"}

for gguf_name, sf_name in NORM_MAP:
    print("=" * 72)
    # ── GGUF side ──
    if gguf_name not in gguf_map:
        print(f"[GGUF] {gguf_name}: NOT FOUND")
        gguf_vals = None
    else:
        dims, dtype, off = gguf_map[gguf_name]
        n_elem = 1
        for d in dims:
            n_elem *= d
        k = min(N, n_elem)
        dname = DTYPE_NAMES.get(dtype, f"type{dtype}")
        if dtype == 1:  # F16
            gguf_vals = read_f16_vals(GGUF_PATH, data_start, off, k)
        elif dtype == 0:  # F32
            gguf_vals = read_f32_vals(GGUF_PATH, data_start, off, k)
        else:
            gguf_vals = None
        print(f"GGUF  {gguf_name}  [{dname}]")
        if gguf_vals:
            print(f"  raw      : {[round(v, 5) for v in gguf_vals]}")
            print(f"  raw - 1  : {[round(v - 1.0, 5) for v in gguf_vals]}")
        else:
            print(f"  (type {dtype} not decoded)")

    # ── safetensors side ──
    sf_vals = None
    for sf_path in SF_PATHS:
        r = read_st_tensor(sf_path, sf_name, N)
        if r is not None:
            _, _, sf_vals = r
            break

    print(f"SF    {sf_name.split('.')[-2] + '.' + sf_name.split('.')[-1]}")
    if sf_vals:
        print(f"  values   : {[round(v, 5) for v in sf_vals]}")
    else:
        print("  NOT FOUND in either safetensors shard")

    # ── verdict ──
    if gguf_vals and sf_vals:
        diff_direct = [abs(g - s) for g, s in zip(gguf_vals, sf_vals)]
        diff_minus1 = [abs((g - 1.0) - s) for g, s in zip(gguf_vals, sf_vals)]
        mean_d = sum(diff_direct) / len(diff_direct)
        mean_m1 = sum(diff_minus1) / len(diff_minus1)
        print(f"  mean |gguf - sf|       = {mean_d:.6f}")
        print(f"  mean |(gguf-1) - sf|   = {mean_m1:.6f}")
        if mean_d < mean_m1:
            print("  *** VERDICT: GGUF stores the SAME values as safetensors.")
            print("               The -1 adjustment is INCORRECT.")
        else:
            print("  *** VERDICT: GGUF stores (1 + safetensors value).")
            print("               The -1 adjustment is CORRECT.")
    print()
