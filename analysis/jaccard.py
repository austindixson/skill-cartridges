"""Jaccard of deleted-weight sets between two packbits_zstd masks; verifies counts against config zero_params."""
import json, sys, numpy as np, zstandard
from safetensors import safe_open
def load(d):
    f = safe_open(f"{d}/model.safetensors", "np"); out = {}
    for k in f.keys():
        if k.endswith(".packed"):
            base = k[:-7]; shape = tuple(int(x) for x in f.get_tensor(base + ".shape"))
            bits = np.unpackbits(np.frombuffer(zstandard.ZstdDecompressor().decompress(f.get_tensor(k).tobytes()), np.uint8))
            out[base] = bits[: int(np.prod(shape))] == 0  # True = deleted (zero) weight
    cfg = json.load(open(f"{d}/config.json")); n = sum(int(v.sum()) for v in out.values())
    assert n == cfg["zero_params"], (n, cfg["zero_params"]); return out, cfg["total_params"]
a, N = load(sys.argv[1]); b, _ = load(sys.argv[2])
na = sum(int(v.sum()) for v in a.values()); nb = sum(int(v.sum()) for v in b.values())
inter = sum(int((a[k] & b[k]).sum()) for k in a)
union = na + nb - inter; exp = na * nb / N
print(json.dumps({"deleted_a": na, "deleted_b": nb, "intersection": inter, "jaccard": inter / union,
                  "expected_intersection_if_random": exp, "overlap_vs_chance": inter / exp}, indent=1))
