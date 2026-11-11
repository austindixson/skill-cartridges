"""Stage D: wikitext-2 PPL base vs ON and exact forward KL(base||ON), protocol of scripts/eval_ppl_kl.py, resumable per window."""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deltune.evals import kl as K
from deltune.stats import paired_bootstrap

from .common import RESULTS, log

OUT = RESULTS / "kl"
FIELDS = ("kl_on", "kl_off", "lp_base", "lp_on", "lp_off", "agree_on", "agree_off")


def token_stats_chunked(lp_base: mx.array, lp_other: mx.array, targets: np.ndarray, chunk: int = 256) -> dict:
    """deltune.evals.kl.token_stats over position chunks (bounds the exp()/product temporaries of a 248k vocab)."""
    parts = [K.token_stats(lp_base[a:a + chunk], lp_other[a:a + chunk], targets[a:a + chunk]) for a in range(0, lp_base.shape[0], chunk)]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def run_window(model, toggle, w: np.ndarray, chunk: int = 256) -> dict:
    tgt = w[1:]
    toggle.off()
    lpb = K.logp_window(model, w); mx.eval(lpb)
    toggle.on()
    s_on = token_stats_chunked(lpb, K.logp_window(model, w), tgt, chunk)
    toggle.off()
    s_off = token_stats_chunked(lpb, K.logp_window(model, w), tgt, chunk)
    del lpb
    return {"kl_on": s_on["kl"], "kl_off": s_off["kl"], "lp_base": s_on["lp_base"], "lp_on": s_on["lp_other"], "lp_off": s_off["lp_other"],
            "agree_on": s_on["agree"], "agree_off": s_off["agree"]}


def run_stage(model, tok, toggle, guard=None, limit=None, out_dir: Path = OUT, n_tokens: int = K.N_TOKENS) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ids = K.load_wikitext_tokens(tok, n_tokens)
    W = K.windows(ids)
    n = limit or len(W)
    log("kl: tokens", len(ids), "windows", W.shape)
    t0 = time.time()
    for wi in range(n):
        f = out_dir / f"win_{wi:03d}.npz"
        if f.exists():
            continue
        if guard:
            guard.check(f"kl/{wi}")
        r = run_window(model, toggle, W[wi])
        np.savez(f.with_suffix(".tmp.npz"), **r); f.with_suffix(".tmp.npz").rename(f)
        mx.clear_cache()
        log(f"[kl] window {wi + 1}/{n} {time.time() - t0:.0f}s kl_on mean {r['kl_on'].mean():.5f} kl_off max {np.abs(r['kl_off']).max()}")
    toggle.off()
    return assemble(out_dir, W[:n], len(ids))


def assemble(out_dir: Path, W: np.ndarray, n_tokens_used: int) -> dict:
    """Pure CPU: per-window npz files -> the ppl_kl_qwen2.5-0.5b.json schema."""
    acc = {k: [] for k in FIELDS}
    for wi in range(len(W)):
        z = np.load(out_dir / f"win_{wi:03d}.npz")
        for k in FIELDS:
            acc[k].append(z[k])
    nb = np.array([-x.mean() for x in acc["lp_base"]]); no = np.array([-x.mean() for x in acc["lp_on"]])
    A = {k: np.concatenate(v) for k, v in acc.items()}
    kl = A["kl_on"]
    q = lambda x: {"mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max()), "min": float(x.min())}
    return {
        "protocol": {"dataset": "Salesforce/wikitext wikitext-2-raw-v1 test", "join": "\\n\\n", "tokens_used": n_tokens_used, "window": K.WINDOW,
                     "stride": K.WINDOW, "n_windows": int(len(W)), "scored_tokens_per_window": K.WINDOW - 1, "scored_tokens": int(len(kl)),
                     "tokens_dropped_tail": int(n_tokens_used - W.size), "dtype": "bfloat16 logits, fp32 log-softmax",
                     "note": "tokenized with the Qwen3.8 tokenizer (vocab ~248k): same text, same 50,000-token/1024-window scheme, but token "
                             "boundaries (and therefore the exact text span covered) differ from the Qwen2.5-0.5B run; PPL is not comparable across tokenizers"},
        "ppl_base": float(np.exp(nb.mean())), "ppl_on": float(np.exp(no.mean())),
        "ppl_diff_on_minus_base": paired_bootstrap(nb, no, stat=lambda a, b: np.exp(b) - np.exp(a)),
        "log_ppl_diff_on_minus_base": paired_bootstrap(nb, no),
        "forward_kl_exact_per_token": {**q(kl), "bootstrap_over_windows_mean": paired_bootstrap(
            np.zeros(len(W)), np.array([x.mean() for x in np.split(kl, len(W))]))},
        "forward_kl_sampled_token_on_text_tokens": q(A["lp_base"] - A["lp_on"]),
        "top1_agreement_base_vs_on": float(A["agree_on"].mean()),
        "gates": {"kl_base_vs_off_max_abs": float(np.abs(A["kl_off"]).max()), "kl_base_vs_off_all_exactly_zero": bool((A["kl_off"] == 0).all()),
                  "logprob_base_vs_off_exactly_equal": bool((A["lp_base"] == A["lp_off"]).all()),
                  "top1_agreement_base_vs_off": float(A["agree_off"].mean()), "kl_on_min": float(kl.min()), "kl_on_n_negative": int((kl < 0).sum())},
    }
