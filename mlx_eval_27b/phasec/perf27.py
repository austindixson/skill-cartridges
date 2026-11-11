"""Stage E: single-stream perf, base vs ON, arms interleaved; 1 warmup + 3 measured runs per arm."""
from __future__ import annotations

import statistics, time

import mlx.core as mx

from deltune.evals import kl as K
from mlx_lm.generate import generate_step

from .common import mp_line, rss_bytes


def one_run(model, ids, n_dec: int) -> dict:
    t0 = time.perf_counter(); tf = None; n = 0
    for tok, _ in generate_step(mx.array(ids), model, max_tokens=n_dec):
        if tf is None:
            tf = time.perf_counter()
        n += 1
    t1 = time.perf_counter()
    return {"prompt_tps": len(ids) / (tf - t0), "decode_tps": (n - 1) / (t1 - tf), "n_tokens": n, "ttft_s": tf - t0}


def run_stage(model, tok, toggle, prompt_tokens: int = 512, decode_tokens: int = 128, repeats: int = 3) -> dict:
    ids = [int(x) for x in K.load_wikitext_tokens(tok, prompt_tokens)]
    assert len(ids) == prompt_tokens
    out = {"prompt_tokens": prompt_tokens, "decode_tokens": decode_tokens, "repeats": repeats, "memory_pressure_before": mp_line(),
           "rss_before": rss_bytes(), "decode_note": "generate_step, no EOS stop (speed only); run 0 per arm is the warmup and is excluded from medians"}
    runs = {"base": [], "on": []}; peaks = {"base": [], "on": []}
    for rep in range(1 + repeats):
        for arm in ("base", "on"):
            toggle.on() if arm == "on" else toggle.off()
            mx.reset_peak_memory()
            r = one_run(model, ids, decode_tokens)
            r["peak_memory_bytes"] = mx.get_peak_memory(); r["warmup"] = rep == 0
            runs[arm].append(r)
    toggle.off()
    for arm in runs:
        m = [r for r in runs[arm] if not r["warmup"]]
        out[arm] = {"prompt_tps_median": statistics.median(r["prompt_tps"] for r in m), "decode_tps_median": statistics.median(r["decode_tps"] for r in m),
                    "peak_memory_bytes_max": max(r["peak_memory_bytes"] for r in m), "runs": runs[arm]}
    out["rss_after"] = rss_bytes(); out["memory_pressure_after"] = mp_line()
    out["toggle_timings_ms"] = toggle.timings_ms
    return out
