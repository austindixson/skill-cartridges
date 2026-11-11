"""Paired statistics: bootstrap CI for a difference of means, McNemar exact test, mean-over-tasks bootstrap."""
from __future__ import annotations

from math import comb

import numpy as np

N_BOOT = 10_000
SEED = 0


def _boot_idx(rng, n: int, n_boot: int, chunk_budget: int = 20_000_000):
    """Yield (start, idx) blocks of resample index matrices, bounded in memory."""
    step = max(1, chunk_budget // max(n, 1))
    for s in range(0, n_boot, step):
        yield rng.integers(0, n, size=(min(step, n_boot - s), n))


def paired_bootstrap(a, b, n_boot: int = N_BOOT, seed: int = SEED, stat=None) -> dict:
    """Paired bootstrap over items. Default statistic: mean(b) - mean(a). `stat(mean_a, mean_b)` may be
    given for a transformed statistic (e.g. perplexity difference). Returns point estimate and 95% percentile CI."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 0
    stat = stat or (lambda ma, mb: mb - ma)
    rng = np.random.default_rng(seed)
    out = []
    for idx in _boot_idx(rng, len(a), n_boot):
        out.append(stat(a[idx].mean(axis=1), b[idx].mean(axis=1)))
    out = np.concatenate(out)
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"n": int(len(a)), "mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "diff": float(stat(a.mean(), b.mean())), "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_bootstrap_tasks(pairs, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Unweighted mean over tasks of (mean(b) - mean(a)); items are resampled within each task (paired), then averaged."""
    rng = np.random.default_rng(seed)
    tot = np.zeros(n_boot)
    point = 0.0
    for a, b in pairs:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        d = b - a
        point += d.mean()
        got = np.concatenate([d[idx].mean(axis=1) for idx in _boot_idx(rng, len(d), n_boot)])
        tot += got
    k = len(pairs)
    tot /= k
    lo, hi = np.percentile(tot, [2.5, 97.5])
    return {"n_tasks": k, "diff": float(point / k), "ci_lo": float(lo), "ci_hi": float(hi)}


def mcnemar_exact(a, b) -> dict:
    """a, b paired 0/1 correctness (a = base, b = ON). b_cnt = a right & b wrong; c_cnt = a wrong & b right.
    Two-sided exact binomial p (p0 = 0.5) on the discordant pairs."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    nb = int((a & ~b).sum())
    nc = int((~a & b).sum())
    n = nb + nc
    if n == 0:
        return {"b": nb, "c": nc, "p": 1.0}
    k = min(nb, nc)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return {"b": nb, "c": nc, "p": float(p)}
