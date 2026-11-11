"""Load-time Deletion Tuning mask toggle for mlx-lm models.

Base weights on disk are never touched. on() zeroes the deleted entries in memory after stashing only
those entries' original values (raw 16-bit patterns); off() writes them back bit-exactly.

Name mapping rule: mask keys are HF names (`model.layers.N.mlp.gate_proj.weight`); mlx-lm's Qwen2/Llama
trees flatten (mlx.utils.tree_flatten(model.parameters())) to the identical dotted names, so the map is the
identity. It is verified (exactly one parameter per key, identical shape), not assumed.
"""
from __future__ import annotations

import time
import weakref

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from .mask import DeletedIndices, iter_deleted, load_deleted

_UINT_VIEW = {mx.bfloat16: mx.uint16, mx.float16: mx.uint16, mx.float32: mx.uint32}


_ACTIVE: dict[int, weakref.ref] = {}  # id(model) -> weakref to the model that has a mask toggle ON (any toggle class)


def _claim(model) -> None:
    """Mark `model` as having an active mask toggle; raise if one already is (toggles do not compose)."""
    ref = _ACTIVE.get(id(model))
    if ref is not None and ref() is model:
        raise RuntimeError("another mask toggle is active on this model; turn it off first")
    key = id(model)
    _ACTIVE[key] = weakref.ref(model, lambda r, key=key: _ACTIVE.get(key) is r and _ACTIVE.pop(key, None))


def _release(model) -> None:
    _ACTIVE.pop(id(model), None)


def _get_param(model, name: str) -> mx.array:
    """Return the live parameter array object by dotted name (walks modules/lists)."""
    obj = model
    for part in name.split("."):
        obj = obj[int(part)] if part.isdigit() else (obj[part] if isinstance(obj, dict) else getattr(obj, part))
    assert isinstance(obj, mx.array), name
    return obj


class MaskToggle:
    def __init__(self, model, mask_dir: str):
        self.model = model
        self.mask_dir = mask_dir
        self.timings_ms: dict[str, list[float]] = {"decode": [], "on": [], "off": []}
        t = time.perf_counter()
        self.deleted: dict[str, DeletedIndices] = load_deleted(mask_dir)
        self.timings_ms["decode"].append((time.perf_counter() - t) * 1e3)
        flat = dict(tree_flatten(model.parameters()))
        self.name_map: dict[str, str] = {}
        for name, d in self.deleted.items():
            assert name in flat, f"mask key {name!r} has no parameter in the mlx model"
            assert tuple(flat[name].shape) == d.shape, f"{name}: mask {d.shape} vs param {tuple(flat[name].shape)}"
            assert flat[name].dtype in _UINT_VIEW, f"{name}: unsupported dtype {flat[name].dtype} (quantized?)"
            self.name_map[name] = name
        assert len(set(self.name_map.values())) == len(self.name_map)
        self._idx = {n: mx.array(d.indices.astype(np.uint32 if d.numel < 2**32 else np.int64)) for n, d in self.deleted.items()}
        self._stash: dict[str, mx.array] = {}  # raw bits of deleted entries only
        self.is_on = False

    @property
    def n_deleted(self) -> int:
        return sum(d.n_deleted for d in self.deleted.values())

    def stash_nbytes(self) -> int:
        return sum(v.nbytes for v in self._stash.values())

    def _write(self, name: str, raw_values: mx.array | None) -> None:
        """Scatter raw bits (or zeros if None) into the parameter in place, keeping dtype."""
        w = _get_param(self.model, self.name_map[name])
        raw = w.view(_UINT_VIEW[w.dtype]).reshape(-1)
        raw[self._idx[name]] = 0 if raw_values is None else raw_values
        new = raw.reshape(w.shape).view(w.dtype)
        mx.eval(new)
        # Rebind in the module tree (in-place __setitem__ on a view does not reach the original array).
        parts = self.name_map[name].split(".")
        self.model.update(_nest(parts, new))

    def on(self) -> None:
        if self.is_on:
            return
        _claim(self.model)
        t = time.perf_counter()
        try:
            for name in self.deleted:
                w = _get_param(self.model, self.name_map[name])
                raw = w.view(_UINT_VIEW[w.dtype]).reshape(-1)
                vals = raw[self._idx[name]]
                mx.eval(vals)
                self._stash[name] = vals
                self._write(name, None)
        except BaseException:
            try:  # roll back: write every stashed layer's originals back, then forget them
                for name in list(self._stash):
                    self._write(name, self._stash[name])
                    del self._stash[name]
                _release(self.model)
            except BaseException:  # rollback itself failed: stay ON with the stash intact so off() can retry
                self.is_on = True
            raise
        self.is_on = True
        self.timings_ms["on"].append((time.perf_counter() - t) * 1e3)

    def off(self) -> None:
        if not self.is_on:
            return
        t = time.perf_counter()
        for name in self.deleted:
            if name in self._stash:  # layers restored by a failed earlier off() are already gone from the stash
                self._write(name, self._stash[name])
                del self._stash[name]
        self.is_on = False
        _release(self.model)
        self.timings_ms["off"].append((time.perf_counter() - t) * 1e3)


def _nest(parts: list[str], leaf):
    out = leaf
    for p in reversed(parts):
        out = {p: out}
    return _fix_lists(out)


def _fix_lists(tree):
    """Convert {'layers': {'3': x}} dict-with-digit-keys into mlx's list form with empty dicts elsewhere."""
    if isinstance(tree, dict):
        if tree and all(k.isdigit() for k in tree):
            n = max(int(k) for k in tree) + 1
            return [_fix_lists(tree[str(i)]) if str(i) in tree else {} for i in range(n)]
        return {k: _fix_lists(v) for k, v in tree.items()}
    return tree


def apply_mask(model, mask, on: bool = True) -> MaskToggle:
    """Functional wrapper. `mask` is a mask dir or an existing MaskToggle for this model."""
    toggle = mask if isinstance(mask, MaskToggle) else MaskToggle(model, mask)
    assert toggle.model is model
    toggle.on() if on else toggle.off()
    return toggle
