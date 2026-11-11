"""Decode Deletion Tuning masks (numpy only). Mirrors upstream src/apply_mask.py.

Mask dir: config.json + model.safetensors with `<name>.shape` (int64) and `<name>.packed`
(zstd(np.packbits(keep_bool.ravel()))). True = keep, False = delete.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import zstandard
from safetensors import safe_open

FORMAT = "packbits_zstd"


@dataclass(frozen=True)
class DeletedIndices:
    """Compact form for one parameter: sorted flat (row-major) indices of deleted entries."""
    name: str
    shape: tuple[int, ...]
    indices: np.ndarray  # sorted, unique, uint32 if numel < 2**32 else int64

    @property
    def numel(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n_deleted(self) -> int:
        return int(self.indices.size)


def read_config(mask_dir: str) -> dict:
    with open(os.path.join(mask_dir, "config.json")) as f:
        cfg = json.load(f)
    assert cfg["format"] == FORMAT, f"unsupported mask format {cfg['format']!r}"
    return cfg


def _names(f) -> list[str]:
    keys = list(f.keys())
    names = [k[: -len(".shape")] for k in keys if k.endswith(".shape")]
    for n in names:
        assert f"{n}.packed" in keys, f"{n}: .shape without .packed"
    assert len(keys) == 2 * len(names), "unexpected extra tensors in mask file"
    return names


def iter_keep_masks(mask_dir: str) -> Iterator[tuple[str, np.ndarray]]:
    """Lazily yield (name, keep_bool array) one parameter at a time (upstream-identical decode)."""
    read_config(mask_dir)
    dctx = zstandard.ZstdDecompressor()
    with safe_open(os.path.join(mask_dir, "model.safetensors"), framework="np") as f:
        for name in _names(f):
            shape = tuple(int(x) for x in f.get_tensor(f"{name}.shape").tolist())
            numel = int(np.prod(shape))
            packed = np.frombuffer(dctx.decompress(f.get_tensor(f"{name}.packed").tobytes()), dtype=np.uint8)
            yield name, np.unpackbits(packed)[:numel].astype(bool).reshape(shape)


def iter_deleted(mask_dir: str) -> Iterator[DeletedIndices]:
    """Lazily yield compact deleted-index records; at most one layer's bool array alive at a time."""
    for name, keep in iter_keep_masks(mask_dir):
        flat = keep.reshape(-1)
        idx = np.flatnonzero(~flat)  # ascending
        idx = idx.astype(np.uint32 if flat.size < 2**32 else np.int64)
        yield DeletedIndices(name, tuple(keep.shape), idx)


def load_mask(mask_dir: str) -> dict[str, np.ndarray]:
    """name -> boolean keep mask (True keeps). Materializes everything; asserts config totals."""
    masks = dict(iter_keep_masks(mask_dir))
    cfg = read_config(mask_dir)
    total = sum(m.size for m in masks.values())
    zero = sum(int((~m).sum()) for m in masks.values())
    assert total == cfg["total_params"], (total, cfg["total_params"])
    assert zero == cfg["zero_params"], (zero, cfg["zero_params"])
    return masks


def load_deleted(mask_dir: str) -> dict[str, DeletedIndices]:
    """Compact form for the whole mask (indices only); asserts config totals."""
    out = {d.name: d for d in iter_deleted(mask_dir)}
    cfg = read_config(mask_dir)
    assert sum(d.numel for d in out.values()) == cfg["total_params"]
    assert sum(d.n_deleted for d in out.values()) == cfg["zero_params"]
    return out
