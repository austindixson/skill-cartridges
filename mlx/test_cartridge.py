"""No-model tests. Run: JOB/.venv/bin/python JOB/results/test_cartridge.py"""
import os  # SC_ROOT: folder holding the mask and result directories (see README)
import sys
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cartridge, stats

B = os.environ.get("SC_ROOT", ".") + ""
M9 = f"{B}/item5-9b/repo/9b-toolalpaca-1/mask"
M7T = f"{B}/item1-overlap/masks/sdft/qwen2.5-7b-instruct-toolalpaca"
M7S = f"{B}/item1-overlap/masks/sdft/qwen2.5-7b-instruct-science"


def test_masks():
    for d, n in ((M9, 10_702_823), (M7T, 14_892_491), (M7S, 33_607_213)):
        k = cartridge.load_mask(d)
        assert len(k) == (32 if d == M9 else 28) and cartridge.deleted_count(k) == n, (d, cartridge.deleted_count(k))
        print(f"PASS decode {Path(d).name}: {len(k)} layers, deleted {n}")
    a, b = cartridge.load_mask(M7T), cartridge.load_mask(M7S)
    u, i = cartridge.deleted_count(cartridge.combine([a, b], "union")), cartridge.deleted_count(cartridge.combine([a, b], "intersection"))
    assert u == 47_359_157 and i == 1_140_547, (u, i)
    print(f"PASS union {u} intersection {i}")


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [type("L", (), {})() for _ in range(2)]
        for l in self.layers:
            l.mlp = type("M", (), {})()
            l.mlp.gate_proj = nn.Linear(16, 8, bias=False)
            l.mlp.gate_proj.weight = (mx.random.normal((8, 16)) + 1).astype(mx.bfloat16)
        self.model = self


def test_apply_eject():
    m = Tiny()
    orig = {i: np.array(l.mlp.gate_proj.weight.view(mx.uint16)) for i, l in enumerate(m.layers)}
    rng = np.random.default_rng(0)
    keep = {i: rng.random((8, 16)) > 0.1 for i in range(2)}
    saved = cartridge.apply_dense(m, keep)
    for i, l in enumerate(m.layers):
        w = l.mlp.gate_proj.weight
        assert w.dtype == mx.bfloat16 and (np.array(w.astype(mx.float32))[~keep[i]] == 0).all()
        assert (np.array(w.view(mx.uint16))[keep[i]] == orig[i][keep[i]]).all()
    cartridge.eject_dense(m, saved)
    for i, l in enumerate(m.layers):
        assert (np.array(l.mlp.gate_proj.weight.view(mx.uint16)) == orig[i]).all()
    print("PASS synthetic apply/eject bit-exact")


def test_stats():
    r = stats.summarize(97, 42, 69, 27, 0)
    assert (r["gained"], r["lost"], r["net"]) == (27, 0, 27)
    assert abs(r["mcnemar_p"] - 2 * 0.5 ** 27) < 1e-18 and abs(r["mcnemar_p"] - 0.5 ** 26) < 1e-18
    assert round(r["ci_pp"][0], 1) == 19.6 and round(r["ci_pp"][1], 1) == 37.1, r["ci_pp"]
    print(f"PASS stats (27,0,97): p={r['mcnemar_p']:.3e}=2^-26, CI {r['ci_pp'][0]:.1f}..{r['ci_pp'][1]:.1f}")


if __name__ == "__main__":
    test_masks(); test_apply_eject(); test_stats(); print("ALL PASS")
