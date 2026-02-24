import struct


def f16_to_f32(bits):
    s = (bits >> 15) & 1
    e = (bits >> 10) & 0x1F
    m = bits & 0x3FF
    if e == 0:
        v = (m / (1 << 10)) * (2 ** (-14))
    elif e == 31:
        v = float("inf") if m == 0 else float("nan")
    else:
        v = (1 + m / (1 << 10)) * (2 ** (e - 15))
    return -v if s else v


def bf16_to_f32(bits):
    bits32 = bits << 16
    return struct.unpack("f", struct.pack("I", bits32))[0]


# ─── GGUF reader ────────────────────────────────────────────────────────────


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


def _rval(f, vtype):
    if vtype == 0:
        return _ru8(f)
    if vtype == 1:
        return _ri8(f)
    if vtype == 2:
        return _ru16(f)
    if vtype == 3:
        return _ri16(f)
    if vtype == 4:
        return _ru32(f)
    if vtype == 5:
        return _ri32(f)
    if vtype == 6:
        return _rf32(f)
    if vtype == 7:
        return _ru64(f)
    if vtype == 8:
        return _rstr(f)
    if vtype == 9:
        return bool(_ru8(f))
    if vtype == 10:
        et = _ru32(f)
        cnt = _ru64(f)
        return [_rval(f, et) for _ in range(cnt)]
    if vtype == 11:
        return _ri64(f)
    if vtype == 12:
        return _rf64(f)
    raise ValueError(f"unknown meta type {vtype}")


GGML_TYPE = {0: "f32", 1: "f16", 2: "q4_0", 30: "bf16"}


def read_gguf_tensors(path):
    with open(path, "rb") as f:
        f.read(4)  # magic
        _ru32(f)  # version
        tc = _ru64(f)  # tensor count
        mc = _ru64(f)  # meta count
        for _ in range(mc):
            _rstr(f)
            vt = _ru32(f)
            _rval(f, vt)
        tensors = []
        for _ in range(tc):
            name = _rstr(f)
            ndims = _ru32(f)
            dims = [_ru64(f) for _ in range(ndims)]
            dtype = _ru32(f)
            off = _ru64(f)
            tensors.append((name, dims, dtype, off))
        pos = f.tell()
        data_start = (pos + 31) & ~31
    return tensors, data_start


def read_gguf_f16_tensor(path, data_start, offset, n_vals=10):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [f16_to_f32(_ru16(f)) for _ in range(n_vals)]


def read_gguf_f32_tensor(path, data_start, offset, n_vals=10):
    with open(path, "rb") as f:
        f.seek(data_start + offset)
        return [_rf32(f) for _ in range(n_vals)]


# ─── safetensors reader ──────────────────────────────────────────────────────


def read_safetensors_tensor(path, tensor_name, n_vals=10):
    import json

    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        if tensor_name not in header:
            return None
        info = header[tensor_name]
        shape = info["shape"]
        dtype = info["dtype"]
        offsets = info["data_offsets"]
        f.seek(8 + header_len + offsets[0])
        nbytes = offsets[1] - offsets[0]
        data = f.read(nbytes)
    if dtype == "F32":
        vals = list(struct.unpack(f"<{len(data) // 4}f", data))
    elif dtype == "BF16":
        import array as arr

        shorts = list(arr.array("H", data))
        vals = [bf16_to_f32(s) for s in shorts]
    else:
        return None
    return shape, dtype, vals[:n_vals]


# ─── main ────────────────────────────────────────────────────────────────────

GGUF_PATH = "models/gemma-3-4b-it-qat-q4_0-gguf/gemma-3-4b-it-q4_0.gguf"
SF1_PATH = "models/gemma-3-4b-it/model-00001-of-00002.safetensors"
SF2_PATH = "models/gemma-3-4b-it/model-00002-of-00002.safetensors"

NORM_PAIRS = [
    # (gguf_name,                   safetensors_suffix)
    ("blk.0.attn_norm.weight", "input_layernorm.weight"),
    ("blk.0.post_attn_norm.weight", "post_attention_layernorm.weight"),
    ("blk.0.ffn_norm.weight", "pre_feedforward_layernorm.weight"),
    ("blk.0.ffn_post_norm.weight", "post_feedforward_layernorm.weight"),
    ("blk.0.attn_q_norm.weight", "self_attn.q_norm.weight"),
    ("blk.0.attn_k_norm.weight", "self_attn.k_norm.weight"),
    ("output_norm.weight", "model.norm.weight"),
]

N = 10

print("Reading GGUF tensor index …")
tensors, data_start = read_gguf_tensors(GGUF_PATH)
gguf_map = {name: (dims, dtype, off) for name, dims, dtype, off in tensors}

print(f"data_start = {data_start}\n")

for gguf_name, sf_suffix in NORM_PAIRS:
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
        if dtype == 1:  # F16
            gguf_vals = read_gguf_f16_tensor(GGUF_PATH, data_start, off, k)
            dtype_str = "F16"
        elif dtype == 0:  # F32
            gguf_vals = read_gguf_f32_tensor(GGUF_PATH, data_start, off, k)
            dtype_str = "F32"
        else:
            gguf_vals = None
            dtype_str = f"type{dtype}"

    # ── safetensors side ──
    sf_full = f"language_model.model.layers.0.{sf_suffix}"
    sf_norm = f"language_model.model.{sf_suffix}"  # for output_norm
    sf_vals = None
    for sf_path in [SF1_PATH, SF2_PATH]:
        for sf_name in [sf_full, sf_norm]:
            r = read_safetensors_tensor(sf_path, sf_name, N)
            if r is not None:
                _, _, sf_vals = r
                break
        if sf_vals is not None:
            break

    # ── print comparison ──
    print(f"{'=' * 70}")
    print(f"GGUF:   {gguf_name}  (dtype={dtype_str if gguf_vals else '?'})")
    if gguf_vals:
        print(f"  raw:      {[round(v, 5) for v in gguf_vals]}")
        print(f"  raw-1:    {[round(v - 1.0, 5) for v in gguf_vals]}")
    else:
        print("  NOT FOUND")

    print(f"SF:     {sf_suffix}")
    if sf_vals:
        print(f"  values:   {[round(v, 5) for v in sf_vals]}")
    else:
        print("  NOT FOUND")

    if gguf_vals and sf_vals:
        diff_raw = [round(g - s, 5) for g, s in zip(gguf_vals, sf_vals)]
        diff_minus1 = [round((g - 1) - s, 5) for g, s in zip(gguf_vals, sf_vals)]
        print(f"  gguf - sf:     {diff_raw}")
        print(f"  (gguf-1) - sf: {diff_minus1}")
        avg_diff = sum(abs(d) for d in diff_raw) / len(diff_raw)
        avg_diff1 = sum(abs(d) for d in diff_minus1) / len(diff_minus1)
        print(f"  mean |gguf-sf|={avg_diff:.5f}   mean |(gguf-1)-sf|={avg_diff1:.5f}")
        if avg_diff1 < avg_diff:
            print("  => GGUF stores (1+w), subtract-1 fix is CORRECT")
        else:
            print("  => GGUF stores w directly, subtract-1 fix is WRONG")
    print()
