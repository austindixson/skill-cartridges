"""Bit-level helpers for proofs: raw parameter bytes, independent safetensors reader, hashes."""
from __future__ import annotations

import hashlib, json, os, struct

import mlx.core as mx
import numpy as np
import zstandard
from mlx.utils import tree_flatten

_NP = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32, "I64": np.int64, "U8": np.uint8}


def snapshot(model) -> dict[str, np.ndarray]:
    """name -> raw-bits numpy copy (uint16 for bf16) of every parameter."""
    out = {}
    for n, w in tree_flatten(model.parameters()):
        mx.eval(w)
        if w.dtype in (mx.bfloat16, mx.float16):
            out[n] = np.array(w.view(mx.uint16))
        elif w.dtype == mx.float32:
            out[n] = np.array(w.view(mx.uint32))
        else:
            raise AssertionError(f"{n}: {w.dtype}")
    return out


def digest(snap: dict[str, np.ndarray]) -> str:
    h = hashlib.sha256()
    for n in sorted(snap):
        h.update(np.ascontiguousarray(snap[n]).tobytes())
    return h.hexdigest()


def read_safetensors_raw(path: str, names=None) -> dict[str, np.ndarray]:
    """Parse the safetensors header by hand; return raw-bit arrays (bf16 as uint16)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for k, v in hdr.items():
            if k == "__metadata__" or (names is not None and k not in names):
                continue
            a, b = v["data_offsets"]
            f.seek(base + a)
            out[k] = np.frombuffer(f.read(b - a), dtype=_NP[v["dtype"]]).reshape(v["shape"])
    return out


def independent_keep(mask_dir: str, name: str) -> np.ndarray:
    """Re-implementation of upstream decode straight from raw bytes (no deltune.mask, no safe_open)."""
    t = read_safetensors_raw(os.path.join(mask_dir, "model.safetensors"), {f"{name}.shape", f"{name}.packed"})
    shape = tuple(int(x) for x in t[f"{name}.shape"])
    packed = np.frombuffer(zstandard.ZstdDecompressor().decompress(t[f"{name}.packed"].tobytes()), dtype=np.uint8)
    return np.unpackbits(packed)[: int(np.prod(shape))].astype(bool).reshape(shape)
