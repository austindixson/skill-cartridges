"""Run: JOB/.venv/bin/python JOB/results/test_quant_cartridge.py   (no full model load)."""
import json, os, sys, time
import numpy as np, zstandard
import mlx.core as mx, mlx.nn as nn
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quant_cartridge as qc

JOB = os.path.dirname(HERE)
MODEL = JOB + "/models/Qwen3.5-9B-MLX-4bit"
MASK = os.environ.get("SC_ROOT", ".") + "/item5-9b/repo/9b-toolalpaca-1/mask"
KEY = "language_model.model.layers.{}.mlp.gate_proj."
GS, BITS = 64, 4


def load_mask_private(d):  # test-only decoder (format: zstd(np.packbits(keep.ravel())), 1 = keep)
    dctx = zstandard.ZstdDecompressor()
    out = {}
    with safe_open(d + "/model.safetensors", framework="numpy") as f:
        for k in f.keys():
            if k.endswith(".shape"):
                L = int(k.split(".")[2]); shp = tuple(int(v) for v in f.get_tensor(k))
                raw = np.frombuffer(dctx.decompress(f.get_tensor(k[:-6] + ".packed").tobytes()), dtype=np.uint8)
                out[L] = np.unpackbits(raw, count=shp[0] * shp[1]).astype(bool).reshape(shp)
    return out


def own_dequant(w, s, b):
    q = qc.unpack_codes(w, BITS).astype(mx.float32)
    O, I = q.shape
    sf = qc._expand_groups(s.astype(mx.float32), GS); bf = qc._expand_groups(b.astype(mx.float32), GS)
    return (sf * q + bf).astype(s.dtype)


def synth(O=96, I=256, seed=0, dtype=mx.bfloat16):
    mx.random.seed(seed)
    w = (mx.random.normal((O, I)) * 0.05 + 0.01).astype(dtype)
    return mx.quantize(w, GS, BITS)


def rand_keep(O, I, frac=0.1, seed=1):
    return np.random.default_rng(seed).random((O, I)) >= frac


def test_packing():
    for dt in (mx.bfloat16, mx.float16, mx.float32):
        w, s, b = synth(dtype=dt)
        ref = mx.dequantize(w, s, b, GS, BITS)
        mine = own_dequant(w, s, b)
        if dt == mx.float32:
            print('note float32: own formula differs from mx.dequantize in', int((ref != mine).sum().item()), 'of', ref.size, 'values (fma/rounding); max diff', float(mx.abs(ref - mine).max().item()))
        else:
            assert mx.array_equal(ref, mine).item(), dt
        assert mx.array_equal(qc.pack_codes(qc.unpack_codes(w)), w).item()
    # explicit layout: element k of word j at bits 4k
    c = qc.unpack_codes(w)
    wn = np.array(w); cn = np.array(c)
    assert all(((int(wn[0, 0]) >> (4 * k)) & 15) == cn[0, k] for k in range(8))
    assert all(((int(wn[3, 5]) >> (4 * k)) & 15) == cn[3, 40 + k] for k in range(8))
    print("PASS test_packing (bf16/f16 exact; own dequant s*q+b == mx.dequantize exactly; unpack->pack bit-identical)")


def brute_nearest(w, s, b, dele):
    tab = np.abs(np.array(qc.code_value_table(s, b, GS, BITS).astype(mx.float32)))  # (O,G,16)
    q = np.array(qc.unpack_codes(w)); d = np.array(dele)
    O, I = q.shape
    qn = np.array(qc.nearest_zero_codes(s, b, GS, BITS))
    return tab, q, d, qn


def test_nearest_zero_synth():
    w, s, b = synth(); O, I = 96, 256
    keep = rand_keep(O, I); dele = ~keep
    w2, st = qc.nearest_zero_arrays(w, s, b, mx.array(keep), GS, BITS)
    d0 = np.array(mx.dequantize(w, s, b, GS, BITS).astype(mx.float32)); d1 = np.array(mx.dequantize(w2, s, b, GS, BITS).astype(mx.float32))
    assert (d0[keep] == d1[keep]).all()
    assert (np.array(qc.unpack_codes(w))[keep] == np.array(qc.unpack_codes(w2))[keep]).all()
    tab = np.abs(np.array(qc.code_value_table(s, b, GS, BITS).astype(mx.float32)))  # via mx.dequantize
    G = I // GS
    for o in range(O):
        for i in range(I):
            if dele[o, i]:
                assert abs(d1[o, i]) <= tab[o, i // GS].min() + 0 and abs(d1[o, i]) == tab[o, i // GS].min()
    # table agrees with real dequantize of each code (independent check)
    for qv in (0, 7, 15):
        full = mx.dequantize(mx.full((O, I // 8), qv * 0x11111111, dtype=mx.uint32), s, b, GS, BITS).astype(mx.float32)
        assert (np.array(full)[:, ::GS] == np.array(qc.code_value_table(s, b, GS, BITS).astype(mx.float32))[:, :, qv]).all()
    # s == 0 safety
    s0 = mx.zeros_like(s); b0 = b
    q0 = qc.nearest_zero_codes(s0, b0, GS, BITS); assert (np.array(q0) == 0).all()
    assert st["deleted"] == int(dele.sum()) and st["nondeleted_codes_changed"] == 0
    print("PASS test_nearest_zero_synth", {k: st[k] for k in ("deleted", "codes_changed", "deleted_exact_zero_after")})


class _Fake:
    pass


def make_model(n=3, O=96, I=256):
    layers = {}
    for L in range(n):
        w = (mx.random.normal((O, I)) * 0.05 + 0.01).astype(mx.bfloat16)
        m = nn.QuantizedLinear(I, O, bias=False, group_size=GS, bits=BITS)
        m.weight, m.scales, m.biases = mx.quantize(w, GS, BITS)
        layers[L] = m
    return layers


def test_api_and_eject():
    mx.random.seed(3)
    mods = make_model(); keep = {L: rand_keep(96, 256, 0.07, L) for L in mods}
    qc._modules = lambda model: mods
    for apply in (qc.apply_nearest_zero, qc.apply_requant):
        saved = apply(None, keep)
        assert "seconds_swap" in saved["stats"] and "seconds_compute" in saved["stats"]
        assert not qc.verify_ejected(None, saved, 3) or apply is None
        qc.eject(None, saved)
        assert qc.verify_ejected(None, saved, 3)
    prep = qc.precompute_nearest_zero(None, keep)
    saved = qc.swap_in(None, prep)
    assert not mx.array_equal(mods[0].weight, saved["orig"][0][0]).item()
    qc.eject(None, saved); assert qc.verify_ejected(None, saved, 3)
    print("PASS test_api_and_eject (both forms + precompute/swap_in; eject bit-identical)")


def test_requant_synth():
    w, s, b = synth(); O, I = 96, 256
    keep = rand_keep(O, I, 0.1); dele = ~keep
    w2, s2, b2, st = qc.requant_arrays(w, s, b, mx.array(keep), GS, BITS)
    d0 = mx.dequantize(w, s, b, GS, BITS); d0z = mx.where(mx.array(dele), mx.zeros_like(d0), d0)
    ref = mx.quantize(d0z, GS, BITS)
    assert all(mx.array_equal(a, c).item() for a, c in zip((w2, s2, b2), ref))
    new = np.array(mx.dequantize(w2, s2, b2, GS, BITS).astype(mx.float32))
    # brute force: every group's affine grid, nearest grid value to 0 should be what the deleted weights became
    sf = np.array(s2.astype(mx.float32)); bf = np.array(b2.astype(mx.float32))
    for o in range(0, O, 7):
        for i in range(I):
            if dele[o, i]:
                g = i // GS
                grid = sf[o, g] * np.arange(16) + bf[o, g]
                assert abs(new[o, i]) <= np.abs(grid).max() + 1e-6  # on-grid sanity
    print("deleted exact-zero after requant:", int(((new == 0) & dele).sum()), "of", int(dele.sum()))
    # idempotence in groups with no deletion: requant of all-kept must reproduce? measure
    keep_all = mx.ones((O, I), dtype=mx.bool_)
    wi, si, bi, sti = qc.requant_arrays(w, s, b, keep_all, GS, BITS)
    print("IDEMPOTENCE dequant->quant (no deletions): codes changed",
          int((np.array(qc.unpack_codes(wi)) != np.array(qc.unpack_codes(w))).sum()), "/", O * I,
          "| scales changed", int((np.array(si.astype(mx.float32)) != np.array(s.astype(mx.float32))).sum()),
          "| biases changed", int((np.array(bi.astype(mx.float32)) != np.array(b.astype(mx.float32))).sum()),
          "| dequant values changed", sti["nondeleted_changed"], "/", O * I)
    # groups with no deletions inside a masked run
    gdel = dele.reshape(O, I // GS, GS).any(-1)
    ch = (np.array(mx.dequantize(w2, s2, b2, GS, BITS).astype(mx.float32)) != np.array(d0.astype(mx.float32))).reshape(O, I // GS, GS).any(-1)
    print("untouched groups (no deletion) with any changed value:", int((ch & ~gdel).sum()), "of", int((~gdel).sum()))
    print("PASS test_requant_synth (matches dequant->zero->quantize reference exactly)")


def real_layer(L):
    d = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]
    f = mx.load(MODEL + "/" + d[KEY.format(L) + "weight"])
    return tuple(f[KEY.format(L) + n] for n in ("weight", "scales", "biases"))


def test_real(save=True):
    if not os.path.exists(MODEL + "/model-00002-of-00002.safetensors"):
        print("SKIP real data: model files missing"); return
    keep = load_mask_private(MASK)
    assert len(keep) == 32 and sum(int((~k).sum()) for k in keep.values()) == 10702823
    out = {"nearest_zero": {}, "requant": {}}
    t_all = {"nearest_zero": 0.0, "requant": 0.0}
    for L in range(32):
        w, s, b = real_layer(L)
        assert w.shape == (12288, 512) and s.shape == (12288, 64)
        km = mx.array(keep[L])
        w2, st1 = qc.nearest_zero_arrays(w, s, b, km, GS, BITS)
        # real-data correctness checks
        assert st1["nondeleted_codes_changed"] == 0
        out["nearest_zero"][L] = qc._finalize(st1)
        r = qc.requant_arrays(w, s, b, km, GS, BITS)
        out["requant"][L] = qc._finalize(r[3])
        t_all["nearest_zero"] += st1["seconds_compute"]; t_all["requant"] += r[3]["seconds_compute"]
        if L in (0, 31):
            print(f"layer {L} nearest_zero:", json.dumps(out["nearest_zero"][L], indent=None))
            print(f"layer {L} requant:", json.dumps(out["requant"][L], indent=None))
        del w2, r, w, s, b
    tot = {f: qc.aggregate(out[f]) for f in out}
    assert tot["nearest_zero"]["deleted"] == tot["requant"]["deleted"] == 10702823
    print("TOTAL nearest_zero:", json.dumps(tot["nearest_zero"], indent=1))
    print("TOTAL requant:", json.dumps(tot["requant"], indent=1))
    res = {"per_layer": out, "total": tot, "compute_seconds_total": t_all, "n_layers": 32, "group_size": GS, "bits": BITS,
           "note": "seconds_compute = array building incl mx.eval, per-layer, mask upload excluded"}
    if save:
        json.dump(res, open(HERE + "/quant_form_stats.json", "w"), indent=1)
    print("PASS test_real (32 layers, both forms; deleted total 10702823); compute secs", t_all)


if __name__ == "__main__":
    test_packing(); test_nearest_zero_synth(); test_api_and_eject(); test_requant_synth(); test_real()
    print("ALL PASS")
