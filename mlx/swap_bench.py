"""Wall time to APPLY / EJECT a deletion-mask cartridge on an already-loaded MLX model (live toggle).
  swap_bench.py --model DIR --mask DIR --reps 10 --out F.json [--dense]
Timed regions all end with mx.eval inside the region; clock = time.perf_counter. Stats are never computed here."""
import argparse, json, statistics as st, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx_lm
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartridge, quant_cartridge as qc, run_eval  # noqa: E402

PROMPT = "List three practical uses of a hash map."
ATTRS = ("weight", "scales", "biases")


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1e3, r


def summ(x):
    return {"samples_ms": x, "min": min(x), "median": st.median(x), "max": max(x)}


def host_bytes(mods, dense):
    """Independent host copy of raw bytes (bf16-safe) so verification does not rely on object identity."""
    return {L: [np.array(mx.view(getattr(m, a), mx.uint8)) for a in (("weight",) if dense else ATTRS)]
            for L, m in mods.items()}


def same_host(mods, ref, dense):
    return all(np.array_equal(np.array(mx.view(getattr(m, a), mx.uint8)), r)
               for L, m in mods.items() for a, r in zip(("weight",) if dense else ATTRS, ref[L]))


def new_arrays(form, mods, keep):
    """COLD core, no stats: host mask -> GPU, bit work, eval. Returns {L: {attr: array}}."""
    out = {}
    for L, m in mods.items():
        k = mx.array(keep[L])  # host->GPU upload (inside timed region)
        if form == "dense":
            out[L] = {"weight": mx.where(k, m.weight, mx.zeros((), dtype=m.weight.dtype))}
            mx.eval(out[L]["weight"])
        elif form == "nearest_zero":
            w, _ = qc.nearest_zero_arrays(m.weight, m.scales, m.biases, k, m.group_size, m.bits, m.mode, with_stats=False)
            out[L] = {"weight": w}
        else:
            w, s, b, _ = qc.requant_arrays(m.weight, m.scales, m.biases, k, m.group_size, m.bits, m.mode, with_stats=False)
            out[L] = {"weight": w, "scales": s, "biases": b}
    return out


def assign(mods, new):
    """Swap prepared arrays into modules and force evaluation."""
    for L, d in new.items():
        for a, v in d.items():
            setattr(mods[L], a, v)
    mx.eval([v for d in new.values() for v in d.values()])


def logits(model, ids):
    lg = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
    mx.eval(lg)
    return lg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--mask", required=True)
    ap.add_argument("--reps", type=int, default=10); ap.add_argument("--out", required=True)
    ap.add_argument("--dense", action="store_true", help="model is bf16/unquantized")
    a = ap.parse_args()
    dense = a.dense
    res = {"model": a.model, "mask": a.mask, "dense": dense, "mlx": mx.__version__ if hasattr(mx, "__version__") else None,
           "mlx_lm": mlx_lm.__version__, "timestamp": datetime.now().astimezone().isoformat(), "forms": {}}
    try:
        import mlx.core as _m; from importlib.metadata import version; res["mlx"] = version("mlx")
    except Exception:
        pass

    model, tok = load(a.model)
    mx.eval(model.parameters())
    mods = cartridge.gate_proj_modules(model)
    assert len(mods) == 32, len(mods)
    ids = run_eval.build_prompts(tok, [{"prompt": PROMPT}])[1][0]
    base_lg = logits(model, ids)
    base_host = host_bytes(mods, dense)

    ms, keep = zip(*[timed(lambda: cartridge.load_mask(a.mask)) for _ in range(max(1, a.reps))])
    keep = keep[0]
    res["mask_decode_ms"] = summ(list(ms)); res["deleted_count"] = cartridge.deleted_count(keep)

    if dense:
        orig = {L: m.weight for L, m in mods.items()}
        eject_fn = lambda: cartridge.eject_dense(model, orig)
        verify = lambda: same_host(mods, base_host, True)
    else:
        saved = {"orig": qc._originals(mods)}
        eject_fn = lambda: qc.eject(model, saved)
        verify = lambda: qc.verify_ejected(model, saved) and same_host(mods, base_host, False)

    for form in (["dense"] if dense else ["nearest_zero", "requant"]):
        r = {"cold_apply_ms": [], "eject_after_cold_ms": [], "warm_apply_ms": [], "eject_after_warm_ms": []}
        passes = 0
        for _ in range(a.reps):  # cold: host keep -> live, evaluated
            ms_a, _ = timed(lambda: assign(mods, new_arrays(form, mods, keep)))
            ms_e, _ = timed(eject_fn)
            r["cold_apply_ms"].append(ms_a); r["eject_after_cold_ms"].append(ms_e); passes += bool(verify())
        prepared = new_arrays(form, mods, keep)  # cached, evaluated "server-side" cartridge
        for _ in range(a.reps):  # warm: only the swap (+eval)
            ms_a, _ = timed(lambda: assign(mods, prepared))
            ms_e, _ = timed(eject_fn)
            r["warm_apply_ms"].append(ms_a); r["eject_after_warm_ms"].append(ms_e); passes += bool(verify())
        # functional check
        assign(mods, prepared); lg_app = logits(model, ids); eject_fn(); lg_ej = logits(model, ids)
        assign(mods, prepared); lg_re = logits(model, ids); eject_fn()
        f = {"logits_base_equal_after_eject": bool(mx.array_equal(base_lg, lg_ej).item()),
             "max_abs_logit_diff_applied_vs_base": float(mx.max(mx.abs(lg_app - base_lg)).item()),
             "reapply_equal": bool(mx.array_equal(lg_app, lg_re).item()), "ejected_after_func": bool(verify())}
        res["forms"][form] = {**{k: summ(v) for k, v in r.items()}, "verify_ejected_pass": passes,
                              "verify_ejected_expected": 2 * a.reps, "functional": f}
        del prepared
    res["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    Path(a.out).write_text(json.dumps(res, indent=1))

    d = res["mask_decode_ms"]
    print(f"deleted={res['deleted_count']}  mask_decode median {d['median']:.1f} ms  peak {res['peak_memory_gb']:.2f} GB")
    print(f"{'form':14}{'metric':22}{'min':>9}{'median':>9}{'max':>9}  (ms)")
    for form, v in res["forms"].items():
        for k in ("cold_apply_ms", "warm_apply_ms", "eject_after_cold_ms", "eject_after_warm_ms"):
            print(f"{form:14}{k:22}{v[k]['min']:9.2f}{v[k]['median']:9.2f}{v[k]['max']:9.2f}")
        print(f"{form:14}verify {v['verify_ejected_pass']}/{v['verify_ejected_expected']}  func {v['functional']}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
