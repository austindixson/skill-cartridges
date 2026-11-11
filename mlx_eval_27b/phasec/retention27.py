"""Stage C: forgetting probe (hellaswag / mmlu / arc_challenge / winogrande), same items + scoring as the 0.5B run
(deltune.evals.retention), adapted to the qwen3_5 module tree, with a pure-cache dedup of identical input sequences,
a timing probe + budgeted plan, and chunked, paired, resumable scoring."""
from __future__ import annotations

import json, math, time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deltune.evals import retention as R
from deltune.stats import mcnemar_exact, paired_bootstrap, paired_bootstrap_tasks

from .common import RESULTS, ROOT, append_jsonl, log, read_jsonl

OUT = RESULTS / "retention"
REF05 = ROOT / "results/retention_qwen2.5-0.5b"


# ------------------------------------------------------------------ items
def load_task(task: str, ref_dir: Path = REF05) -> list[dict]:
    """Items exactly as the 0.5B run built them; ids asserted equal, in order, to the 0.5B per-item file."""
    items = R.load_items(task, R.N_DEFAULT[task])
    ref_ids = [r["id"] for r in read_jsonl(ref_dir / f"{task}.jsonl")]
    assert [it["id"] for it in items] == ref_ids, f"{task}: item ids differ from the 0.5B run"
    return items


def select(items: list, k: int, mode: str = "prefix") -> list[int]:
    """Positions (in the 0.5B order) of the k items scored. prefix = first k; strided = evenly spaced (representative of
    datasets whose order is not random, e.g. MMLU 'all' is sorted by subject)."""
    n = len(items)
    assert 0 < k <= n
    return list(range(k)) if mode == "prefix" else [int(i * n / k) for i in range(k)]


# ------------------------------------------------------------------ scorer
def _parts(model):
    lm = getattr(model, "language_model", model)
    head = lm.model.embed_tokens.as_linear if lm.args.tie_word_embeddings else lm.lm_head
    return lm.model, head


def score_pairs_dedup(model, tok, pairs: list[tuple[str, str]], max_tokens: int = 4096, max_seqs: int = 16, dedup: bool = True) -> np.ndarray:
    """Sum log-prob of each continuation given its context; same math as deltune.evals.retention.score_pairs (bf16 logits,
    fp32 log-softmax, right-padded length-sorted batches, LM head only at continuation positions) but each distinct input
    token sequence is run through the network once (dedup=True): e.g. the 4 MMLU choices share one forward pass.
    The cache changes no request's inputs; scores differ from the non-dedup path only by batch-shape kernel noise."""
    mx.set_cache_limit(R.CACHE_LIMIT)
    body, head = _parts(model)
    enc = [R.encode_pair(tok, c, k) for c, k in pairs]
    assert all(len(k) > 0 and len(c) > 0 for c, k in enc)
    seqs = [c + k for c, k in enc]
    uid_of, uniq = {}, []  # unique input sequence = seq[:-1]
    req_uid = []
    for i, s in enumerate(seqs):
        key = tuple(s[:-1]) if dedup else (i,)
        if key not in uid_of:
            uid_of[key] = len(uniq); uniq.append(s[:-1])
        req_uid.append(uid_of[key])
    triples = {u: [] for u in range(len(uniq))}  # uid -> [(pos, target, request)]
    for i, s in enumerate(seqs):
        nc = len(enc[i][1])
        for p in range(len(s) - 1 - nc, len(s) - 1):
            triples[req_uid[i]].append((p, s[p + 1], i))
    order = sorted(range(len(uniq)), key=lambda u: -len(uniq[u]))
    scores = np.zeros(len(seqs), dtype=np.float64)
    i = 0
    while i < len(order):
        L = len(uniq[order[i]])
        bs = max(1, min(max_seqs, max_tokens // max(L, 1)))
        batch = order[i:i + bs]; i += bs
        inp = np.zeros((len(batch), L), dtype=np.int32)
        sel_of, rows, pos, selrow, tgt, owner = {}, [], [], [], [], []
        for r, u in enumerate(batch):
            inp[r, :len(uniq[u])] = uniq[u]
            for p, t, req in triples[u]:
                if (r, p) not in sel_of:
                    sel_of[(r, p)] = len(rows); rows.append(r); pos.append(p)
                selrow.append(sel_of[(r, p)]); tgt.append(t); owner.append(req)
        h = body(mx.array(inp))
        lp = head(h[mx.array(rows), mx.array(pos)]).astype(mx.float32)
        lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
        tok_lp = lp[mx.array(selrow), mx.array(tgt)]
        mx.eval(tok_lp)
        np.add.at(scores, np.array(owner), np.array(tok_lp, dtype=np.float64))
    return scores


def score_items_dedup(model, tok, items: list[dict], **kw) -> list[list[float]]:
    flat = score_pairs_dedup(model, tok, [r for it in items for r in it["requests"]], **kw)
    out, k = [], 0
    for it in items:
        n = len(it["requests"]); out.append(flat[k:k + n].tolist()); k += n
    return out


# ------------------------------------------------------------------ probe + plan
def probe(model, tok, items_by_task: dict, n_probe: int = 8, **kw) -> dict:
    """Base-arm seconds/item for the first n_probe items of every task (scores discarded)."""
    rates = {}
    for t, items in items_by_task.items():
        sub = items[:n_probe]
        t0 = time.perf_counter(); score_items_dedup(model, tok, sub, **kw); dt = time.perf_counter() - t0
        rates[t] = {"s_per_item": dt / len(sub), "n_probe": len(sub), "seconds": dt}
        log(f"[retention probe] {t}: {dt / len(sub):.2f} s/item")
        mx.clear_cache()
    return rates


def make_plan(n_by_task: dict, s_per_item: dict, budget_s: float, frac: float | None = None, mode: str = "prefix") -> dict:
    """Largest single fraction f (4 decimals, <= 1) with sum_t floor(f n_t) s_t * 2 arms <= budget; `frac` overrides."""
    full = 2 * sum(n_by_task[t] * s_per_item[t] for t in n_by_task)
    f = frac if frac is not None else min(1.0, math.floor(budget_s / full * 1e4) / 1e4)
    k = {t: max(1, math.floor(f * n)) for t, n in n_by_task.items()}
    est = 2 * sum(k[t] * s_per_item[t] for t in k)
    return {"fraction": f, "fraction_source": "override" if frac is not None else "budget", "budget_s": budget_s, "subset_mode": mode,
            "n_full": n_by_task, "n_scored": k, "s_per_item_base_probe": s_per_item, "estimated_seconds_both_arms": est,
            "estimated_seconds_full_set_both_arms": full,
            "note": ("scored items are the first floor(f*n) of each task's seed-1234 subset (sorted dataset order), a prefix, same for both arms"
                     if mode == "prefix" else "scored items are evenly spaced over each task's seed-1234 subset, same for both arms")}


# ------------------------------------------------------------------ chunked paired scoring
def row_of(it: dict, pos: int, sb: list, so: list) -> dict:
    pb, po = R.predictions([it], [sb]), R.predictions([it], [so])
    return {"id": it["id"], "pos": pos, "gold": it["gold"], "char_lens": it["char_lens"], "scores_base": sb, "scores_on": so,
            "pred_base": pb["acc"][0], "pred_on": po["acc"][0], "pred_norm_base": pb["acc_norm"][0], "pred_norm_on": po["acc_norm"][0]}


def run_task(model, tok, toggle, task: str, items: list, positions: list[int], out_dir: Path = OUT, chunk: int = 50,
             guard=None, **kw) -> int:
    path = out_dir / f"{task}.jsonl"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in positions if items[p]["id"] not in done]
    for a in range(0, len(todo), chunk):
        ps = todo[a:a + chunk]; sub = [items[p] for p in ps]
        if guard:
            guard.check(f"retention/{task}/{a}")
        t0 = time.perf_counter()
        toggle.off(); sb = score_items_dedup(model, tok, sub, **kw); t1 = time.perf_counter()
        toggle.on(); so = score_items_dedup(model, tok, sub, **kw); t2 = time.perf_counter()
        toggle.off()
        for p, it, b, o in zip(ps, sub, sb, so):
            append_jsonl(path, row_of(it, p, b, o))
        append_jsonl(out_dir / "chunks.jsonl", {"task": task, "n": len(sub), "s_base": t1 - t0, "s_on": t2 - t1, "done_after": len(done) + a + len(sub)})
        mx.clear_cache()
        log(f"[retention/{task}] {len(done) + a + len(sub)}/{len(positions)} base {t1 - t0:.0f}s on {t2 - t1:.0f}s")
    return len(todo)


def run_stage(model, tok, toggle, guard=None, budget_s: float = 7200, frac: float | None = None, mode: str = "prefix", n_probe: int = 8,
              limit_tasks=R.TASKS, out_dir: Path = OUT, chunk: int = 50, **kw) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    items = {t: load_task(t) for t in limit_tasks}
    plan_p = out_dir / "plan.json"
    if plan_p.exists():
        plan = json.loads(plan_p.read_text()); log("retention: reusing plan", plan["n_scored"])
    else:
        toggle.off()
        rates = probe(model, tok, items, n_probe, **kw)
        plan = make_plan({t: len(v) for t, v in items.items()}, {t: r["s_per_item"] for t, r in rates.items()}, budget_s, frac, mode)
        plan["probe"] = rates
        plan_p.write_text(json.dumps(plan, indent=1)); log("retention plan:", plan["fraction"], plan["n_scored"], f"est {plan['estimated_seconds_both_arms']:.0f}s")
    for t in limit_tasks:
        run_task(model, tok, toggle, t, items[t], select(items[t], plan["n_scored"][t], plan["subset_mode"]), out_dir, chunk, guard, **kw)
    return analyze(out_dir, plan)


# ------------------------------------------------------------------ CPU analysis
def analyze(out_dir: Path = OUT, plan: dict | None = None) -> dict:
    plan = plan or json.loads((out_dir / "plan.json").read_text())
    summ = {"seed": R.SEED, "plan": plan, "tasks": {}, "scoring": "deltune.evals.retention loglik (dedup of identical inputs), bf16 logits / fp32 log-softmax",
            "subset": plan["note"]}
    pairs = []
    for t in R.TASKS:
        rows = read_jsonl(out_dir / f"{t}.jsonl")
        if not rows:
            continue
        gold = np.array([r["gold"] for r in rows])
        row = {"n": len(rows), "n_full_task": plan["n_full"][t], "primary_metric": R.PRIMARY[t], "chance": R.CHANCE[t], "is_prefix": plan["subset_mode"] == "prefix"}
        for metric, kb, ko in (("acc", "pred_base", "pred_on"), ("acc_norm", "pred_norm_base", "pred_norm_on")):
            if t == "winogrande" and metric == "acc_norm":
                continue
            a = np.array([r[kb] for r in rows]) == gold; b = np.array([r[ko] for r in rows]) == gold
            a, b = a.astype(float), b.astype(float)
            row[metric] = {"base": float(a.mean()), "on": float(b.mean()), **paired_bootstrap(a, b), **mcnemar_exact(a, b)}
            if metric == R.PRIMARY[t]:
                pairs.append((a, b))
        summ["tasks"][t] = row
    if pairs:
        summ["mean_over_tasks_primary"] = {**paired_bootstrap_tasks(pairs), "base_mean": float(np.mean([a.mean() for a, _ in pairs])),
                                           "on_mean": float(np.mean([b.mean() for _, b in pairs])), "n_per_task": {t: v["n"] for t, v in summ["tasks"].items()}}
    return summ
