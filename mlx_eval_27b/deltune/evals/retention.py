"""Zero-shot multiple-choice by log-likelihood in MLX, reproducing lm-evaluation-harness (task YAMLs from
EleutherAI/lm-evaluation-harness main: hellaswag, arc_challenge, mmlu (default), winogrande).

Request = (context, continuation); score = sum of continuation-token log-probs given context (bf16 logits, fp32
log-softmax). Tokenisation follows lm-eval's HF `_encode_pair` (trailing context whitespace moves to the
continuation; continuation tokens = tokens(ctx+cont)[len(tokens(ctx)):]); no BOS, no special tokens, no chat template.
"""
from __future__ import annotations

import re

import mlx.core as mx
import numpy as np

TASKS = ("hellaswag", "mmlu", "arc_challenge", "winogrande")
CHANCE = {"hellaswag": 0.25, "mmlu": 0.25, "arc_challenge": 0.25, "winogrande": 0.5}
# primary metric per task: lm-eval `acc_norm` for hellaswag (as the paper's table), `acc` otherwise.
PRIMARY = {"hellaswag": "acc_norm", "mmlu": "acc", "arc_challenge": "acc", "winogrande": "acc"}
SEED = 1234
CACHE_LIMIT = 2 * 2**30
N_DEFAULT = {"hellaswag": 1000, "mmlu": 1000, "arc_challenge": None, "winogrande": None}


# ---------- lm-eval preprocessing (verbatim logic) ----------
def hellaswag_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")  # brackets are WikiHow artifacts
    text = re.sub("\\[.*?\\]", "", text)
    return text.replace("  ", " ")


def hellaswag_doc(doc) -> dict:
    ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
    return {"context": hellaswag_preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [hellaswag_preprocess(e) for e in doc["endings"]], "gold": int(doc["label"])}


def arc_doc(doc) -> dict:
    return {"context": f"Question: {doc['question']}\nAnswer:", "choices": list(doc["choices"]["text"]),
            "gold": doc["choices"]["label"].index(doc["answerKey"])}


def mmlu_doc(doc) -> dict:
    c = doc["choices"]
    desc = f"The following are multiple choice questions (with answers) about {doc['subject'].replace('_', ' ')}.\n\n"
    q = f"{doc['question'].strip()}\nA. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:"
    return {"context": desc + q, "choices": ["A", "B", "C", "D"], "gold": int(doc["answer"])}


def winogrande_requests(doc) -> dict:
    """lm-eval winogrande: context = sentence up to '_' with each option substituted, continuation (shared) = text after '_'."""
    idx = doc["sentence"].index("_")
    cont = doc["sentence"][idx + 1:].strip()
    return {"contexts": [doc["sentence"][:idx] + o for o in (doc["option1"], doc["option2"])], "cont": cont,
            "gold": {"1": 0, "2": 1}[doc["answer"]]}


def make_item(task: str, doc, ident) -> dict:
    """-> {id, gold, requests: [(ctx, cont)], char_lens}. cont carries lm-eval's target_delimiter " "."""
    if task == "winogrande":
        w = winogrande_requests(doc)
        reqs = [(c, " " + w["cont"]) for c in w["contexts"]]
        return {"id": ident, "gold": w["gold"], "requests": reqs, "char_lens": [len(w["cont"])] * 2}
    d = {"hellaswag": hellaswag_doc, "arc_challenge": arc_doc, "mmlu": mmlu_doc}[task](doc)
    return {"id": ident, "gold": d["gold"], "requests": [(d["context"], " " + c) for c in d["choices"]],
            "char_lens": [len(c) for c in d["choices"]]}  # lm-eval completion_len = len(choice string)


def pick_indices(n_total: int, n: int | None, seed: int = SEED) -> list[int]:
    if n is None or n >= n_total:
        return list(range(n_total))
    return sorted(np.random.default_rng(seed).choice(n_total, size=n, replace=False).tolist())


def load_items(task: str, n: int | None, seed: int = SEED) -> list[dict]:
    from datasets import load_dataset
    if task == "hellaswag":
        ds, ids = load_dataset("Rowan/hellaswag", split="validation"), None
    elif task == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test")
    elif task == "arc_challenge":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    elif task == "winogrande":
        ds = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
    else:
        raise ValueError(task)
    out = []
    for i in pick_indices(len(ds), n, seed):
        doc = ds[i]
        ident = doc.get("id") or doc.get("qID") or f"{task}-{i}"
        out.append(make_item(task, doc, f"{ident}|idx{i}"))
    return out


# ---------- scoring ----------
def encode_pair(tok, context: str, continuation: str) -> tuple[list[int], list[int]]:
    n_sp = len(context) - len(context.rstrip())
    if n_sp:
        continuation = context[-n_sp:] + continuation
        context = context[:-n_sp]
    whole = tok.encode(context + continuation, add_special_tokens=False)
    ctx = tok.encode(context, add_special_tokens=False)
    return ctx, whole[len(ctx):]


def _head(model, h):
    return model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)


def score_pairs(model, tok, pairs: list[tuple[str, str]], max_tokens: int = 8192, max_seqs: int = 32) -> np.ndarray:
    """Sum log-prob of each continuation given its context. Batches by length (right-padded; causal attention makes
    padding inert), and projects only continuation positions through the LM head. max_seqs=1 gives unbatched scoring."""
    mx.set_cache_limit(CACHE_LIMIT)  # variable shapes otherwise grow the MLX buffer cache without bound (41 GB seen)
    enc = [encode_pair(tok, c, k) for c, k in pairs]
    seqs = [c + k for c, k in enc]
    assert all(len(k) > 0 for _, k in enc)
    order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
    scores = np.zeros(len(seqs), dtype=np.float64)
    i = 0
    while i < len(order):
        L = len(seqs[order[i]]) - 1
        bs = max(1, min(max_seqs, max_tokens // max(L, 1)))
        batch = order[i:i + bs]
        i += bs
        inp = np.zeros((len(batch), L), dtype=np.int32)
        rows, pos, tgt, owner = [], [], [], []
        for r, j in enumerate(batch):
            s = seqs[j]
            inp[r, :len(s) - 1] = s[:-1]
            nc = len(enc[j][1])
            for p in range(len(s) - 1 - nc, len(s) - 1):
                rows.append(r); pos.append(p); tgt.append(s[p + 1]); owner.append(j)
        h = model.model(mx.array(inp))
        sel = h[mx.array(rows), mx.array(pos)]
        logits = _head(model, sel)  # bf16
        lp = logits.astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        tok_lp = mx.take_along_axis(lp, mx.array(tgt)[:, None], axis=-1)[:, 0]
        mx.eval(tok_lp)
        np.add.at(scores, np.array(owner), np.array(tok_lp, dtype=np.float64))
    return scores


def score_items(model, tok, items: list[dict], **kw) -> list[list[float]]:
    pairs = [r for it in items for r in it["requests"]]
    flat = score_pairs(model, tok, pairs, **kw)
    out, k = [], 0
    for it in items:
        n = len(it["requests"])
        out.append(flat[k:k + n].tolist())
        k += n
    return out


def predictions(items: list[dict], scores: list[list[float]]) -> dict:
    pred = [int(np.argmax(s)) for s in scores]
    pred_norm = [int(np.argmax(np.array(s) / np.array(it["char_lens"], dtype=float))) for it, s in zip(items, scores)]
    return {"acc": pred, "acc_norm": pred_norm}
