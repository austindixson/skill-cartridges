"""Paired base-vs-mask stats for one run dir: per eval set n, base, mask, gained, lost, exact two-sided McNemar p."""
import json, sys
from math import comb
from pathlib import Path

def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)

def sets(obj):
    return obj if isinstance(obj, dict) else {"main": obj}  # ToolAlpaca/Science runs store a plain list

d = Path(sys.argv[1])
base = sets(json.loads((d / "eval_base_responses.json").read_text()))
mask = sets(json.loads((d / "eval_mask_responses.json").read_text()))
summ = json.loads((d / "summary.json").read_text())
out = {"run": d.name, "deleted": summ.get("deleted"), "n_masked_weights": summ.get("n_masked_weights"),
       "sparsity": summ.get("sparsity"), "mask_mb": summ.get("mask_mb"), "steps": summ.get("steps_done"), "sets": {}}
for name in base:
    B = [bool(r["correct"]) for r in base[name]]; M = [bool(r["correct"]) for r in mask[name]]
    g = sum((not b) and m for b, m in zip(B, M)); l = sum(b and (not m) for b, m in zip(B, M))
    rec = {"n": len(B), "base": sum(B), "mask": sum(M), "gained": g, "lost": l, "mcnemar_p": mcnemar_exact(g, l)}
    if "hits" in base[name][0]:
        k = len(base[name][0]["gold"])
        rec["base_key_acc"] = sum(r["hits"] for r in base[name]) / (k * len(B))
        rec["mask_key_acc"] = sum(r["hits"] for r in mask[name]) / (k * len(M))
    out["sets"][name] = rec
print(json.dumps(out, indent=1))
