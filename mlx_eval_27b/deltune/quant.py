"""MLX 4-bit quantization of a Deletion Tuning mask, all in memory (nothing quantized is ever written to disk).

Settings: mlx-lm's defaults for `convert -q` (mlx_lm.utils.quantize_model, mode "affine"): group_size 64, bits 4.
Every nn.Linear and nn.Embedding (embed_tokens, tied lm_head) is quantized; norms/biases stay bf16.

Geometry: for a gate_proj weight W[out, in] each row is cut into in/group_size groups of consecutive input-dim
weights; a group has its own bf16 scale and bias, w ~ scale*q + bias with q in 0..15. Packed layout: a group of 64
4-bit codes is exactly 8 uint32 words (nibble i of a word = bits 4i..4i+3), so a layer's `weight` is (out, in/8)
and reshapes to (out*n_groups, 8); scales/biases are (out, n_groups). Groups are independent, so touching only the
groups that contain a masked entry leaves every other group bit-identical.

Variants (names used everywhere):
  q4-base  bf16 base -> quantize
  (a)      mask ON in bf16 -> quantize                         ("mask_then_quant")
  (b)      quantize base -> per masked layer dequantize, zero masked entries, re-quantize affected groups
           ("quant_then_mask")
  (c)      our own variant: like (b) but each affected group gets a scale/bias that puts 0 on the grid
           ("exact_zero")
`QuantMaskToggle` switches (a)/(b)/(c) ON/OFF on a q4-base model by swapping only the affected groups' packed
words / scales / biases (OFF restores a stash of the originals bit-exactly).
KL helpers are reused from deltune/evals/kl.py.
"""
from __future__ import annotations

import time

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.utils import quantize_model

from .mask import DeletedIndices, load_deleted
from .toggle import MaskToggle, _claim, _release

GROUP_SIZE, BITS, MODE = 64, 4, "affine"
WORDS_PER_GROUP = GROUP_SIZE * BITS // 32  # 8
MODES = ("mask_then_quant", "quant_then_mask", "exact_zero")  # (a), (b), (c)


# ---------------------------------------------------------------- builders (oracle routes, whole model)
def quantize_inplace(model) -> dict:
    model, cfg = quantize_model(model, {}, GROUP_SIZE, BITS, MODE)
    mx.eval(model.parameters())
    return cfg["quantization"]


def build_variant(name: str, model_dir: str, mask_dir: str):
    """Return (model, tokenizer) for bf16-base | bf16-ON | q4-base | q4-a | q4-b | q4-c, built in memory."""
    model, tok = load(model_dir)
    if name == "bf16-base":
        return model, tok
    if name == "bf16-ON":
        MaskToggle(model, mask_dir).on()
        return model, tok
    if name == "q4-base":
        quantize_inplace(model)
    elif name == "q4-a":  # oracle route: whole-model quantize of masked bf16 weights
        MaskToggle(model, mask_dir).on()
        quantize_inplace(model)
    elif name in ("q4-b", "q4-c"):
        quantize_inplace(model)
        QuantMaskToggle(model, mask_dir, "quant_then_mask" if name == "q4-b" else "exact_zero").on()
    else:
        raise ValueError(name)
    return model, tok


# ---------------------------------------------------------------- layer helpers
def layer_module(model, name: str):
    """Module owning parameter `name` (e.g. model.layers.3.mlp.gate_proj.weight -> the gate_proj module)."""
    obj = model
    for part in name.split(".")[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def affected_group_ids(idx: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Sorted unique flat group ids (row * n_groups + col // GROUP_SIZE) holding >= 1 deleted entry."""
    _, in_f = shape
    assert in_f % GROUP_SIZE == 0
    ng = in_f // GROUP_SIZE
    i = idx.astype(np.int64)
    return np.unique((i // in_f) * ng + (i % in_f) // GROUP_SIZE)


def quantize_groups(wg: mx.array):
    """(n, 64) bf16 -> packed (n, 8) uint32, scales (n,), biases (n,)."""
    q, s, b = mx.quantize(wg, GROUP_SIZE, BITS, MODE)
    return q, s[:, 0], b[:, 0]


def unpack_codes(q: np.ndarray) -> np.ndarray:
    """(..., 8) uint32 -> (..., 64) uint8 codes."""
    sh = 4 * np.arange(8, dtype=np.uint64)
    return ((q.astype(np.uint64)[..., None] >> sh) & 15).astype(np.uint8).reshape(*q.shape[:-1], 64)


def pack_codes(c: np.ndarray) -> np.ndarray:
    sh = 4 * np.arange(8, dtype=np.uint64)
    w = (c.reshape(*c.shape[:-1], 8, 8).astype(np.uint64) << sh).sum(-1)
    return w.astype(np.uint32)


def _bf16_round_np(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


_FEASIBLE: dict[int, np.ndarray] = {}


def _feasible_scales(k: int) -> np.ndarray:
    """Sorted positive bf16 scales s with k*s exactly representable in bf16 (so bias = -k*s is exact and
    code k dequantizes to exactly 0 whatever the kernel's accumulation order)."""
    if k not in _FEASIBLE:
        u = np.arange(0x3000, 0x4400, dtype=np.uint32) << 16  # ~2^-9 .. 2^9, all bf16 normals
        s = u.view(np.float32)
        _FEASIBLE[k] = s if k == 0 else s[_bf16_round_np(s * k) == s * k]
    return _FEASIBLE[k]


def exact_zero_requant(wg: np.ndarray, chunk: int = 20_000):
    """(n, 64) float groups that contain 0 -> (packed uint32 (n,8), scales (n,), biases (n,), k (n,)).
    Per group pick the integer k in 0..15 (the code that decodes to 0) and the smallest bf16 scale covering the
    range with k*s exact; bias = -k*s; the k with least squared error wins."""
    n = wg.shape[0]
    codes = np.empty((n, 64), np.uint8)
    S = np.empty(n, np.float32)
    K = np.empty(n, np.int8)
    for a in range(0, n, chunk):
        w = wg[a:a + chunk].astype(np.float32)
        lo, hi = w.min(1), w.max(1)
        best = None
        for k in range(16):
            need = np.zeros_like(lo)
            if k < 15:
                need = np.maximum(need, hi / (15 - k))
            if k > 0:
                need = np.maximum(need, -lo / k)
            need = np.maximum(need, 2.0 ** -20)
            f = _feasible_scales(k)
            s = f[np.minimum(np.searchsorted(f, need), len(f) - 1)]
            q = np.clip(np.rint(w / s[:, None] + k), 0, 15)
            err = (((q - k) * s[:, None] - w) ** 2).sum(1)
            if best is None:
                best = [err, s, q.astype(np.uint8), np.full(len(lo), k, np.int8)]
            else:
                m = err < best[0]
                best[0] = np.where(m, err, best[0])
                best[1] = np.where(m, s, best[1])
                best[2] = np.where(m[:, None], q.astype(np.uint8), best[2])
                best[3] = np.where(m, k, best[3])
        codes[a:a + chunk], S[a:a + chunk], K[a:a + chunk] = best[2], best[1], best[3]
    return pack_codes(codes), S, -K.astype(np.float32) * S, K


def zero_in_groups(wg: mx.array, gid: np.ndarray, idx: np.ndarray, in_f: int) -> mx.array:
    """Set the masked entries (flat indices idx of the full layer) to +0.0 inside gathered groups wg (n,64) bf16."""
    ng = in_f // GROUP_SIZE
    i = idx.astype(np.int64)
    g = (i // in_f) * ng + (i % in_f) // GROUP_SIZE
    pos = np.searchsorted(gid, g)
    assert np.array_equal(gid[pos], g)
    flat = np.array(wg.view(mx.uint16)).copy()
    flat[pos, i % GROUP_SIZE] = 0  # in_f % 64 == 0, so column % 64 == flat idx % 64
    return mx.array(flat).view(mx.bfloat16)


def gather_groups(layer, gid_mx: mx.array):
    """Packed words (n,8), scales (n,), biases (n,) of selected groups of a quantized layer."""
    n_total = layer.scales.size
    return (layer.weight.reshape(n_total, WORDS_PER_GROUP)[gid_mx], layer.scales.reshape(-1)[gid_mx],
            layer.biases.reshape(-1)[gid_mx])


def dequant_groups(q, s, b) -> mx.array:
    return mx.dequantize(q, s[:, None], b[:, None], GROUP_SIZE, BITS, MODE)


def scatter_groups(layer, gid_mx: mx.array, q, s, b) -> None:
    out_f = layer.weight.shape[0]
    n_total = layer.scales.size
    wq = layer.weight.reshape(n_total, WORDS_PER_GROUP)
    wq[gid_mx] = q
    sc = layer.scales.reshape(-1)
    sc[gid_mx] = s
    bi = layer.biases.reshape(-1)
    bi[gid_mx] = b
    layer.weight = wq.reshape(out_f, -1)
    layer.scales = sc.reshape(out_f, -1)
    layer.biases = bi.reshape(out_f, -1)


# ---------------------------------------------------------------- quantized toggle
class QuantMaskToggle:
    """ON/OFF switch for a mask on a q4-base model (in-memory quantized). Construction precomputes, per masked
    gate_proj, the affected groups' ON values (packed words, scales, biases) for the chosen `mode` and stashes the
    originals. on()/off() only scatter those arrays; everything else is untouched.

    mode "quant_then_mask" (b) / "exact_zero" (c) need only the quantized model. mode "mask_then_quant" (a)
    additionally needs the bf16 base model (`bf16_model`, read at build time only, never modified): (a) is
    quantize(bf16 with zeros) per group, which differs from q4-base only in the affected groups.
    """

    def __init__(self, model, mask_dir: str, mode: str, bf16_model=None):
        assert mode in MODES, mode
        self.model, self.mode = model, mode
        self.timings_ms: dict[str, list[float]] = {"build": [], "on": [], "off": []}
        t = time.perf_counter()
        self.deleted: dict[str, DeletedIndices] = load_deleted(mask_dir)
        self.layers: dict[str, dict] = {}
        for name, d in self.deleted.items():
            lay = layer_module(model, name)
            assert hasattr(lay, "scales") and lay.weight.dtype == mx.uint32, f"{name}: not a quantized layer"
            assert (lay.group_size, lay.bits, lay.mode) == (GROUP_SIZE, BITS, MODE)
            out_f, in_f = d.shape
            gid = affected_group_ids(d.indices, d.shape)
            gmx = mx.array(gid.astype(np.uint32))
            old = gather_groups(lay, gmx)
            if mode == "mask_then_quant":
                assert bf16_model is not None
                wsrc = layer_module(bf16_model, name).weight
                assert wsrc.shape == d.shape and wsrc.dtype == mx.bfloat16
                wg = wsrc.reshape(out_f * (in_f // GROUP_SIZE), GROUP_SIZE)[gmx]
                new = quantize_groups(zero_in_groups(wg, gid, d.indices, in_f))
            else:
                wg0 = zero_in_groups(dequant_groups(*old), gid, d.indices, in_f)
                if mode == "quant_then_mask":
                    new = quantize_groups(wg0)
                else:
                    p, s, b, _ = exact_zero_requant(np.array(wg0.astype(mx.float32)))
                    new = (mx.array(p), mx.array(s).astype(mx.bfloat16), mx.array(b).astype(mx.bfloat16))
            mx.eval(*new, *old)
            self.layers[name] = {"gid": gmx, "n_groups_total": out_f * (in_f // GROUP_SIZE), "new": new, "old": old}
        self.timings_ms["build"].append((time.perf_counter() - t) * 1e3)
        self.is_on = False

    @property
    def n_affected_groups(self) -> int:
        return sum(int(v["gid"].size) for v in self.layers.values())

    @property
    def n_groups_total(self) -> int:
        return sum(v["n_groups_total"] for v in self.layers.values())

    def stash_nbytes(self) -> int:
        return sum(a.nbytes for v in self.layers.values() for a in v["old"])

    def on_values_nbytes(self) -> int:
        return sum(a.nbytes for v in self.layers.values() for a in v["new"])

    def _set(self, which: str) -> None:
        for name, v in self.layers.items():
            scatter_groups(layer_module(self.model, name), v["gid"], *v[which])
        mx.eval(self.model.parameters())

    def on(self) -> None:
        if self.is_on:
            return
        _claim(self.model)
        t = time.perf_counter()
        try:
            self._set("new")
        except BaseException:
            try:  # roll back (scatter of the stashed originals is idempotent per layer)
                self._set("old")
                _release(self.model)
            except BaseException:  # rollback failed: stay ON so off() can retry
                self.is_on = True
            raise
        self.is_on = True
        self.timings_ms["on"].append((time.perf_counter() - t) * 1e3)

    def off(self) -> None:
        if not self.is_on:
            return
        t = time.perf_counter()
        self._set("old")  # stash is never consumed, so a failed off() can simply be retried
        self.is_on = False
        _release(self.model)
        self.timings_ms["off"].append((time.perf_counter() - t) * 1e3)


def raw(x: mx.array) -> np.ndarray:
    """numpy copy of raw bits (bf16/f16 as uint16, f32 as uint32) so comparisons are bit-exact."""
    if x.dtype in (mx.bfloat16, mx.float16):
        x = x.view(mx.uint16)
    elif x.dtype == mx.float32:
        x = x.view(mx.uint32)
    return np.array(x)


def quant_state(model) -> dict[str, np.ndarray]:
    """Raw-bit snapshot of every parameter (floats viewed as integers) for bit-exact comparison."""
    from mlx.utils import tree_flatten
    return {k: raw(v) for k, v in tree_flatten(model.parameters())}


def states_equal(a: dict, b: dict) -> tuple[bool, list[str]]:
    bad = [k for k in a.keys() | b.keys() if k not in a or k not in b or a[k].shape != b[k].shape
           or not np.array_equal(a[k], b[k])]
    return not bad, sorted(bad)


def pct(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max()),
            "mean": float(x.mean())}
