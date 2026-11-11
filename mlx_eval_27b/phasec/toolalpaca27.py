"""Stage B: ToolAlpaca (97 items) base vs ON, protocol of train/train_toolalpaca.py (thinking=False, greedy, 1024 new
tokens, stop on tokenizer.eos_token_id only, decode skip_special_tokens=True, one user turn). Single-stream decoding."""
from __future__ import annotations

import json, time
from pathlib import Path

import mlx.core as mx
import numpy as np

from deltune.evals import toolalpaca as T
from deltune.gen import generate_single
from deltune.stats import mcnemar_exact, paired_bootstrap

from .common import REF_DIR, RESULTS, append_jsonl, log, read_jsonl

MAX_NEW = 1024
BF16 = {"base": (42, 97), "mask": (65, 97), "gained": 29, "lost": 6}
OUT = RESULTS / "toolalpaca"


def chat_prompt(tokenizer, user_text: str, thinking: bool = False) -> str:
    """Verbatim copy of train/train_toolalpaca.py chat_prompt (tests assert equality with that file's function)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def load_items(tokenizer) -> list[dict]:
    """97 test items in test order; gold of item i asserted equal to the bf16 job's reference file i."""
    from datasets import load_from_disk
    data = load_from_disk(str(T.DATA_DIR)).to_list()
    assert len(data) == 97
    for arm in ("base", "mask"):
        ref = json.loads((REF_DIR / f"eval_{arm}_responses.json").read_text())
        assert len(ref) == 97 and all(r["golden_answer"] == d["golden_answer"] for r, d in zip(ref, data)), f"gold order mismatch vs {arm}"
    for i, ex in enumerate(data):
        ex["index"] = i
        ex["prompt_text"] = chat_prompt(tokenizer, ex["prompt"], False)
        ex["prompt_ids"] = tokenizer(ex["prompt_text"], add_special_tokens=False)["input_ids"]
    return data


def done_indices(path) -> set[int]:
    return {r["index"] for r in read_jsonl(path)}


def run_arm(model, tok, items, arm: str, path, guard=None, limit=None, max_new: int = MAX_NEW, every: int = 5) -> int:
    """Generate for every item not yet in `path`; append one JSONL row per finished item. Returns #new rows."""
    todo = [it for it in items[:limit] if it["index"] not in done_indices(path)]
    t0, n = time.time(), 0
    for it in todo:
        if guard:
            guard.check(f"toolalpaca/{arm}/{it['index']}")
        t = time.perf_counter()
        r = generate_single(model, it["prompt_ids"], [tok.eos_token_id], max_new_tokens=max_new, max_model_len=262144, index=it["index"])
        text = tok.decode(r.token_ids, skip_special_tokens=True)
        score = T.evaluate_correctness([text], [it["golden_answer"]])[0]
        append_jsonl(path, {"index": it["index"], "prompt_tokens": len(it["prompt_ids"]), "output": text, "n_generated": r.n_generated,
                            "finish_reason": r.finish_reason, "score": score, "seconds": time.perf_counter() - t})
        mx.clear_cache(); n += 1
        if n % every == 0 or n == len(todo):
            log(f"[toolalpaca/{arm}] {n}/{len(todo)} new, item {it['index']} score {score} {r.n_generated} tok {time.perf_counter()-t:.0f}s, elapsed {time.time()-t0:.0f}s")
    return n


def run_stage(model, tok, toggle, guard=None, limit=None, out_dir: Path = OUT) -> dict:
    items = load_items(tok)
    n_lim = limit or len(items)
    for arm in ("base", "on"):
        p = out_dir / f"{arm}.jsonl"
        if len(done_indices(p) & set(range(n_lim))) == n_lim:
            continue
        toggle.on() if arm == "on" else toggle.off()
        run_arm(model, tok, items, arm, p, guard, limit)
    toggle.off()
    return analyze(out_dir)


# ------------------------------------------------------------------ CPU-only analysis
def analyze(out_dir: Path = OUT, ref_dir: Path = REF_DIR, items_gold: list | None = None) -> dict:
    """Pure CPU. Rows for items finished in BOTH arms are used (paired)."""
    rows = {a: {r["index"]: r for r in read_jsonl(out_dir / f"{a}.jsonl")} for a in ("base", "on")}
    ref = {a: json.loads((ref_dir / f"eval_{a}_responses.json").read_text()) for a in ("base", "mask")}
    idx = sorted(set(rows["base"]) & set(rows["on"]))
    assert idx, "no item finished in both arms"
    if items_gold is not None:
        assert all(ref["base"][i]["golden_answer"] == items_gold[i] for i in idx)
    sb = np.array([rows["base"][i]["score"] for i in idx]); so = np.array([rows["on"][i]["score"] for i in idx])
    out = {"n_items": len(idx), "complete": len(idx) == 97, "base_correct": int(sb.sum()), "on_correct": int(so.sum()),
           "base_em_pct": 100 * float(sb.mean()), "on_em_pct": 100 * float(so.mean()),
           "delta_pct_points": paired_bootstrap(100 * sb, 100 * so, n_boot=10_000, seed=0), "mcnemar": mcnemar_exact(sb, so),
           "bf16_job": {"base": "42/97", "mask": "65/97", "gained": BF16["gained"], "lost": BF16["lost"],
                        "note": "bf16 job: paired bootstrap CI +13.4..+35.1 pts, exact McNemar p = 0.00012"},
           "finish_reason": {a: {k: sum(rows[a][i]["finish_reason"] == k for i in idx) for k in ("stop", "length")} for a in rows},
           "mean_generated_tokens": {a: float(np.mean([rows[a][i]["n_generated"] for i in idx])) for a in rows},
           "seconds_total": {a: float(sum(rows[a][i]["seconds"] for i in idx)) for a in rows}}
    agree = {}
    for arm, rkey, refkey in (("base", "base", "base"), ("on", "on", "mask")):
        rc = np.array([ref[refkey][i]["correct"] for i in idx], dtype=bool); mc = np.array([rows[rkey][i]["score"] for i in idx], dtype=bool)
        same = [i for i in idx if rows[rkey][i]["output"] == ref[refkey][i]["response"]]
        agree[arm] = {"vs": f"eval_{refkey}_responses.json", "table_4bit_rows_bf16_cols": {
            "4bit_right_bf16_right": int((mc & rc).sum()), "4bit_right_bf16_wrong": int((mc & ~rc).sum()),
            "4bit_wrong_bf16_right": int((~mc & rc).sum()), "4bit_wrong_bf16_wrong": int((~mc & ~rc).sum())},
            "identical_response_strings": len(same), "flipped_to_right_vs_bf16_idx": [i for i, a, b in zip(idx, mc, rc) if a and not b],
            "flipped_to_wrong_vs_bf16_idx": [i for i, a, b in zip(idx, mc, rc) if b and not a]}
    out["agreement_with_bf16"] = agree
    out["flips_base_to_on"] = {"gained_idx": [i for i, a, b in zip(idx, sb, so) if b and not a], "lost_idx": [i for i, a, b in zip(idx, sb, so) if a and not b]}
    return out
