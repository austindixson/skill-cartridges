"""Apply a deletion mask (skill cartridge) to MLX affine-quantized gate_proj layers, live and reversibly.

Packing layout (proved in test_quant_cartridge.py; MLX docs: "packed from the lower to upper bits"):
  weight uint32 (out, in*bits/32); word j of a row holds codes in[j*8 .. j*8+7] (bits=4), element k at bits 4k..4k+3.
  scales/biases (out, in/group_size), group g = input columns [g*gs, (g+1)*gs) of the row.
  dequant w = s*q + b (computed by mx.dequantize; scales/biases dtype, bf16 for this model).

Two forms:
  apply_nearest_zero : keep scales/biases, set deleted codes to the code whose *actual MLX-dequantized* value is closest to 0.
  apply_requant      : dequantize, zero deleted weights, mx.quantize again (weight, scales, biases all replaced).
"""
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------- bit helpers
def unpack_codes(w, bits=4):
    """uint32 (O, I*bits/32) -> uint32 codes (O, I)."""
    per = 32 // bits
    sh = mx.arange(per, dtype=mx.uint32) * bits
    c = (w[..., None] >> sh) & ((1 << bits) - 1)
    return c.reshape(w.shape[0], -1)


def pack_codes(c, bits=4):
    """uint32 codes (O, I) -> uint32 (O, I*bits/32). Inverse of unpack_codes."""
    per = 32 // bits
    sh = mx.arange(per, dtype=mx.uint32) * bits
    c = c.astype(mx.uint32).reshape(c.shape[0], -1, per) << sh
    return c.sum(axis=-1).astype(mx.uint32)  # fields are disjoint, so sum == OR


def code_value_table(scales, biases, group_size, bits=4, mode="affine"):
    """(O, G, 2^bits): the value MLX's own dequantize gives for every code in every group (exact, incl. rounding)."""
    n = 1 << bits
    O, G = scales.shape
    codes = mx.concatenate([mx.arange(n, dtype=mx.uint32), mx.zeros(group_size - n, dtype=mx.uint32)])
    row = pack_codes(codes[None, :], bits)  # (1, group_size*bits/32)
    wq = mx.broadcast_to(row, (O * G, row.shape[1]))
    tab = mx.dequantize(wq, scales.reshape(O * G, 1), biases.reshape(O * G, 1),
                        group_size=group_size, bits=bits, mode=mode)
    return tab[:, :n].reshape(O, G, n)


def nearest_zero_codes(scales, biases, group_size, bits=4, mode="affine"):
    """(O, G) uint32: code with smallest |dequantized value| per group; exact ties -> smaller code; s==0 -> code 0."""
    tab = mx.abs(code_value_table(scales, biases, group_size, bits, mode).astype(mx.float32))
    n = tab.shape[-1]
    m = tab.min(axis=-1, keepdims=True)
    cand = mx.where(tab == m, mx.arange(n, dtype=mx.uint32), mx.array(n, dtype=mx.uint32))
    return cand.min(axis=-1)


def _expand_groups(x, group_size):
    """(O, G) -> (O, G*group_size) by repeat along columns."""
    O, G = x.shape
    return mx.broadcast_to(x[:, :, None], (O, G, group_size)).reshape(O, G * group_size)


def _sum(x):
    return float(mx.sum(x.astype(mx.float32)).item())


# ---------------------------------------------------------------- core (raw arrays)
def nearest_zero_arrays(w, s, b, keep, group_size=64, bits=4, mode="affine", with_stats=True):
    """Returns (w_new, stats). keep: bool mx array (O, I), True = keep. Timing covers only building w_new."""
    assert mode == "affine" and bits == 4
    dele = mx.logical_not(keep)
    t0 = time.perf_counter()
    q = unpack_codes(w, bits)
    qstar = _expand_groups(nearest_zero_codes(s, b, group_size, bits, mode), group_size)
    w_new = pack_codes(mx.where(dele, qstar, q), bits)
    mx.eval(w_new)
    secs = time.perf_counter() - t0
    st = {"seconds_compute": secs}
    if with_stats:
        q_new = unpack_codes(w_new, bits)
        before = mx.abs(mx.dequantize(w, s, b, group_size=group_size, bits=bits, mode=mode).astype(mx.float32))
        after = mx.abs(mx.dequantize(w_new, s, b, group_size=group_size, bits=bits, mode=mode).astype(mx.float32))
        step = mx.abs(_expand_groups(s.astype(mx.float32), group_size))
        sd = dele & (step > 0)
        n = int(_sum(dele))
        st.update(
            deleted=n,
            codes_changed=int(_sum(dele & (q_new != q))),
            nondeleted_codes_changed=int(_sum(mx.logical_not(dele) & (q_new != q))),
            deleted_exact_zero_after=int(_sum(dele & (after == 0))),
            sum_abs_before=_sum(mx.where(dele, before, 0)),
            sum_abs_after=_sum(mx.where(dele, after, 0)),
            max_abs_before=float(mx.max(mx.where(dele, before, 0)).item()),
            max_abs_after=float(mx.max(mx.where(dele, after, 0)).item()),
            n_step_nonzero=int(_sum(sd)),
            sum_after_over_step=_sum(mx.where(sd, after / mx.where(step > 0, step, 1), 0)),
            max_after_over_step=float(mx.max(mx.where(sd, after / mx.where(step > 0, step, 1), 0)).item()),
        )
    return w_new, st


def requant_arrays(w, s, b, keep, group_size=64, bits=4, mode="affine", with_stats=True):
    """Returns (w_new, s_new, b_new, stats)."""
    assert mode == "affine" and bits == 4
    dele = mx.logical_not(keep)
    t0 = time.perf_counter()
    d = mx.dequantize(w, s, b, group_size=group_size, bits=bits, mode=mode)
    d = mx.where(dele, mx.zeros_like(d), d)
    w2, s2, b2 = mx.quantize(d, group_size, bits, mode=mode)
    mx.eval(w2, s2, b2)
    secs = time.perf_counter() - t0
    st = {"seconds_compute": secs}
    if with_stats:
        old = mx.dequantize(w, s, b, group_size=group_size, bits=bits, mode=mode).astype(mx.float32)
        new = mx.dequantize(w2, s2, b2, group_size=group_size, bits=bits, mode=mode).astype(mx.float32)
        nd = mx.logical_not(dele)
        chg = nd & (new != old)
        diff = mx.where(chg, mx.abs(new - old), 0)
        O, I = old.shape
        G = I // group_size
        gdel = dele.reshape(O, G, group_size).any(axis=-1)
        gchg = (new != old).reshape(O, G, group_size).any(axis=-1)
        gpar = (mx.not_equal(s2, s) | mx.not_equal(b2, b)) | gchg
        st.update(
            deleted=int(_sum(dele)),
            deleted_exact_zero_after=int(_sum(dele & (new == 0))),
            sum_abs_after=_sum(mx.where(dele, mx.abs(new), 0)),
            max_abs_after=float(mx.max(mx.where(dele, mx.abs(new), 0)).item()),
            nondeleted=int(_sum(nd)),
            nondeleted_changed=int(_sum(chg)),
            sum_abs_change=_sum(diff),
            max_abs_change=float(mx.max(diff).item()),
            groups=O * G,
            groups_with_deletion=int(_sum(gdel)),
            groups_with_changed_nondeleted=int(_sum(chg.reshape(O, G, group_size).any(axis=-1))),
            groups_any_param_or_value_changed=int(_sum(gpar)),
        )
    return w2, s2, b2, st


def _finalize(st):
    """Derive means from sums (in place)."""
    n = st.get("deleted", 0)
    if "sum_abs_before" in st and n:
        st["mean_abs_before"] = st["sum_abs_before"] / n
    if "sum_abs_after" in st and n:
        st["mean_abs_after"] = st["sum_abs_after"] / n
    if st.get("n_step_nonzero"):
        st["mean_after_over_step"] = st["sum_after_over_step"] / st["n_step_nonzero"]
    if st.get("nondeleted_changed"):
        st["mean_abs_change_of_changed"] = st["sum_abs_change"] / st["nondeleted_changed"]
    if st.get("nondeleted"):
        st["mean_abs_change_all_nondeleted"] = st.get("sum_abs_change", 0) / st["nondeleted"]
    if st.get("groups"):
        st["frac_groups_with_deletion"] = st["groups_with_deletion"] / st["groups"]
        st["frac_groups_any_changed"] = st["groups_any_param_or_value_changed"] / st["groups"]
    return st


_SUM_KEYS = ["deleted", "codes_changed", "nondeleted_codes_changed", "deleted_exact_zero_after", "sum_abs_before",
             "sum_abs_after", "n_step_nonzero", "sum_after_over_step", "nondeleted", "nondeleted_changed",
             "sum_abs_change", "groups", "groups_with_deletion", "groups_with_changed_nondeleted",
             "groups_any_param_or_value_changed", "seconds_compute"]
_MAX_KEYS = ["max_abs_before", "max_abs_after", "max_after_over_step", "max_abs_change"]


def aggregate(per_layer):
    tot = {}
    for st in per_layer.values():
        for k in _SUM_KEYS:
            if k in st:
                tot[k] = tot.get(k, 0) + st[k]
        for k in _MAX_KEYS:
            if k in st:
                tot[k] = max(tot.get(k, 0), st[k])
    return _finalize(tot)


# ---------------------------------------------------------------- model-level API
def _modules(model):
    import cartridge  # provided by the harness worker (JOB/results/cartridge.py)
    mods = cartridge.gate_proj_modules(model)
    for L, m in mods.items():
        assert isinstance(m, nn.QuantizedLinear), (L, type(m))
        assert m.bits == 4 and m.mode == "affine", (L, m.bits, m.mode)
        assert m.get("biases") is not None, L
    return mods


def _keep_mx(keep, L, shape):
    k = keep[L]
    assert tuple(k.shape) == tuple(shape), (L, k.shape, shape)
    return mx.array(k)


def _shape(m):
    return (m.weight.shape[0], m.weight.shape[1] * 32 // m.bits)


def _originals(mods):
    return {L: (m.weight, m.scales, m.biases) for L, m in mods.items()}


def precompute_nearest_zero(model, keep, with_stats=True):
    mods = _modules(model)
    new, per = {}, {}
    t0 = time.perf_counter()
    secs_core = 0.0
    for L in sorted(mods):
        m = mods[L]
        new[L], per[L] = nearest_zero_arrays(m.weight, m.scales, m.biases, _keep_mx(keep, L, _shape(m)),
                                             m.group_size, m.bits, m.mode, with_stats)
        secs_core += per[L]["seconds_compute"]
    stats = {"per_layer": {L: _finalize(s) for L, s in per.items()}}
    stats["total"] = aggregate(per)
    stats["seconds_compute"] = secs_core  # array-building only (stats work excluded)
    stats["seconds_wall_incl_stats_and_mask_upload"] = time.perf_counter() - t0
    return {"form": "nearest_zero", "new": new, "stats": stats}


def swap_in(model, prepared):
    mods = _modules(model)
    saved = {"form": prepared["form"], "orig": _originals(mods), "stats": dict(prepared["stats"])}
    t0 = time.perf_counter()
    for L, m in mods.items():
        n = prepared["new"][L]
        m.weight = n[0] if isinstance(n, tuple) else n
        if isinstance(n, tuple):
            m.scales, m.biases = n[1], n[2]
    mx.eval([a for m in mods.values() for a in (m.weight, m.scales, m.biases)])
    saved["stats"]["seconds_swap"] = time.perf_counter() - t0
    return saved


def apply_nearest_zero(model, keep):
    prepared = precompute_nearest_zero(model, keep)
    saved = swap_in(model, prepared)
    return saved


def precompute_requant(model, keep, with_stats=True):
    mods = _modules(model)
    new, per = {}, {}
    secs_core = 0.0
    for L in sorted(mods):
        m = mods[L]
        w2, s2, b2, per[L] = requant_arrays(m.weight, m.scales, m.biases, _keep_mx(keep, L, _shape(m)),
                                            m.group_size, m.bits, m.mode, with_stats)
        new[L] = (w2, s2, b2)
        secs_core += per[L]["seconds_compute"]
    stats = {"per_layer": {L: _finalize(s) for L, s in per.items()}, "total": aggregate(per),
             "seconds_compute": secs_core}
    return {"form": "requant", "new": new, "stats": stats}


def apply_requant(model, keep):
    return swap_in(model, precompute_requant(model, keep))


def eject(model, saved):
    for L, m in _modules(model).items():
        m.weight, m.scales, m.biases = saved["orig"][L]
    mx.eval([a for m in _modules(model).values() for a in (m.weight, m.scales, m.biases)])


def verify_ejected(model, saved, expect_layers=32):
    mods = _modules(model)
    assert len(mods) == len(saved["orig"]) == expect_layers, (len(mods), len(saved["orig"]))
    for L, m in mods.items():
        w, s, b = saved["orig"][L]
        for cur, ref in ((m.weight, w), (m.scales, s), (m.biases, b)):
            if cur.dtype != ref.dtype or cur.shape != ref.shape or not bool(mx.array_equal(cur, ref).item()):
                return False
    return True
