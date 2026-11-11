"""Per-token log-probs / exact forward KL on fixed windows of one text, MLX, bf16 logits with fp32 log-softmax."""
from __future__ import annotations

import mlx.core as mx
import numpy as np

WINDOW = 1024
N_TOKENS = 50_000


def load_wikitext_tokens(tok, n_tokens: int = N_TOKENS) -> np.ndarray:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok.encode("\n\n".join(ds["text"]), add_special_tokens=False)
    return np.array(ids[:n_tokens], dtype=np.int32)


def windows(ids: np.ndarray, window: int = WINDOW) -> np.ndarray:
    n = len(ids) // window
    return ids[: n * window].reshape(n, window)


def logp_window(model, w: np.ndarray) -> mx.array:
    """(window-1, vocab) fp32 log-probs predicting tokens 1..window-1 of one window (no cross-window context)."""
    logits = model(mx.array(w[None, :-1]))[0].astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def token_stats(lp_base: mx.array, lp_other: mx.array, targets: np.ndarray) -> dict:
    """Per-position: exact forward KL(base||other), sampled-token log-ratio, NLLs, top-1 agreement."""
    kl = mx.sum(mx.exp(lp_base) * (lp_base - lp_other), axis=-1)
    t = mx.array(targets.astype(np.int32))[:, None]
    lb = mx.take_along_axis(lp_base, t, axis=-1)[:, 0]
    lo = mx.take_along_axis(lp_other, t, axis=-1)[:, 0]
    agree = mx.argmax(lp_base, axis=-1) == mx.argmax(lp_other, axis=-1)
    mx.eval(kl, lb, lo, agree)
    return {"kl": np.array(kl, dtype=np.float64), "lp_base": np.array(lb, dtype=np.float64),
            "lp_other": np.array(lo, dtype=np.float64), "agree": np.array(agree)}
