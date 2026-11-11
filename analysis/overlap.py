"""Pairwise overlap of deletion-tuning masks. Usage: python -I overlap.py [masks_dir] [results_dir]
Format (upstream sparse_linear.compress_mask): per layer, np.packbits(keep.ravel()) -> zstd; bit 1 = KEEP.
We unpack each layer (count=numel, so pad bits are dropped), invert to deleted=1, repack per layer
(pad bits = 0). All counts are exact integer popcounts; layers are never concatenated across byte boundaries."""
import csv, json, os, re, sys
from itertools import combinations
import numpy as np, zstandard
from safetensors import safe_open

def popcount(x):
    return int(np.bitwise_count(x).sum(dtype=np.uint64))

def layer_idx(k):
    return int(re.match(r"model\.layers\.(\d+)\.mlp\.gate_proj\.weight\.shape$", k).group(1))

def load_mask(d):
    cfg = json.load(open(d + "/config.json"))
    assert cfg["format"] == "packbits_zstd"
    dctx = zstandard.ZstdDecompressor()
    layers = []
    with safe_open(d + "/model.safetensors", framework="numpy") as f:
        keys = sorted((k for k in f.keys() if k.endswith(".shape")), key=layer_idx)
        assert len(keys) + sum(k.endswith(".packed") for k in f.keys()) == len(list(f.keys()))
        for k in keys:
            name = k[:-6]
            shape = tuple(int(v) for v in f.get_tensor(k))
            n = shape[0] * shape[1]
            raw = np.frombuffer(dctx.decompress(f.get_tensor(name + ".packed").tobytes()), dtype=np.uint8)
            assert raw.size == (n + 7) // 8, (name, raw.size, n)
            keep = np.unpackbits(raw, count=n)
            dele = np.packbits(1 - keep)
            layers.append(dict(shape=shape, n=n, packed=dele, deleted=int(n - keep.sum(dtype=np.int64))))
    N = sum(l["n"] for l in layers); Z = sum(l["deleted"] for l in layers)
    ok = dict(total_ok=N == cfg["total_params"], zero_ok=Z == cfg["zero_params"])
    assert all(ok.values()), (d, N, cfg["total_params"], Z, cfg["zero_params"])
    return dict(layers=layers, N=N, Z=Z, cfg=cfg, ok=ok, shapes=[l["shape"] for l in layers])

def pair_stats(A, B):
    inter_l = [popcount(np.bitwise_and(a["packed"], b["packed"])) for a, b in zip(A["layers"], B["layers"])]
    N, a, b, i = A["N"], A["Z"], B["Z"], sum(inter_l)
    u = a + b - i; p, q = a / N, b / N
    J = i / u if u else float("nan"); J0 = p * q / (p + q - p * q)
    E = a * b / N
    El = sum(x["deleted"] * y["deleted"] / x["n"] for x, y in zip(A["layers"], B["layers"]))
    J0l = El / (a + b - El)
    return dict(N=N, A=a, B=b, inter=i, union=u, p=p, q=q, J=J, J0=J0, J_over_J0=J / J0, J_minus_J0=J - J0,
                E_inter=E, inter_over_E=i / E, E_layer=El, J0_layer=J0l, J_over_J0_layer=J / J0l,
                union_frac=u / N, p_plus_q=p + q, overlap_coef=i / min(a, b)), inter_l

def group_of(path):
    m = re.search(r"llama-3\.2-(1b|3b)-instruct", path)
    if m: return "Llama-3.2-%s-Instruct" % m.group(1).upper()
    if "qwen2.5-7b-instruct" in path: return "Qwen2.5-7B-Instruct"
    return "Qwen2.5-7B(control)"

def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def per_layer(A, B, inter_l):
    rows = []
    for l, (a, b, i) in enumerate(zip(A["layers"], B["layers"], inter_l)):
        p, q = a["deleted"] / a["n"], b["deleted"] / b["n"]; J = i / (a["deleted"] + b["deleted"] - i); J0 = p * q / (p + q - p * q)
        rows.append(dict(layer=l, N_l=a["n"], A_l=a["deleted"], B_l=b["deleted"], inter_l=i, J_l=J, J0_l=J0, ratio=J / J0))
    return rows

def rowcol(A, B):
    out = {}
    for axis, nm in ((1, "row"), (0, "col")):
        ca = cb = ci = tot = 0
        for a, b in zip(A["layers"], B["layers"]):
            sh = a["shape"]
            ma = np.unpackbits(a["packed"], count=a["n"]).reshape(sh).any(axis=axis)
            mb = np.unpackbits(b["packed"], count=b["n"]).reshape(sh).any(axis=axis)
            ca += int(ma.sum()); cb += int(mb.sum()); ci += int((ma & mb).sum()); tot += ma.size
        out[nm] = dict(units=tot, A_hit=ca, B_hit=cb, inter=ci, A_frac=ca / tot, B_frac=cb / tot, J=ci / (ca + cb - ci))
    return out

def main(mdir, rdir):
    paths = sorted(os.path.relpath(r, mdir) for r, _, fs in os.walk(mdir) if "model.safetensors" in fs)
    masks, inv = {}, []
    for p in paths:
        m = masks[p] = load_mask(mdir + "/" + p)
        inv.append(dict(path=p, group=group_of(p), n_layers=len(m["layers"]), N=m["N"], deleted=m["Z"], density=m["Z"] / m["N"],
                        config_zero_params=m["cfg"]["zero_params"], config_total_params=m["cfg"]["total_params"],
                        file_bytes=os.path.getsize(mdir + "/" + p + "/model.safetensors"), **m["ok"]))
        print("OK", p, m["N"], m["Z"], flush=True)
    write_csv(rdir + "/mask_inventory.csv", inv)
    groups = {}
    for p in paths: groups.setdefault(group_of(p), []).append(p)
    for g, ps in groups.items():
        if g.startswith("Qwen2.5-7B("): continue
        rows = []
        for x, y in combinations(ps, 2):
            if masks[x]["shapes"] != masks[y]["shapes"]:
                print("SKIP", x, y, "shapes differ"); continue
            s, il = pair_stats(masks[x], masks[y])
            rows.append(dict(group=g, mask_A=x, mask_B=y, **s))
            if g == "Qwen2.5-7B-Instruct":
                write_csv(rdir + "/qwen7b_toolalpaca_vs_science_per_layer.csv", per_layer(masks[x], masks[y], il))
                print("ROWCOL", x, y, json.dumps(rowcol(masks[x], masks[y])))
        if rows:
            write_csv(rdir + "/pairs_%s.csv" % g, rows); print(g, len(ps), "masks", len(rows), "pairs")
    ctl = groups["Qwen2.5-7B(control)"][0]; rows = []
    for y in groups["Qwen2.5-7B-Instruct"]:
        if masks[ctl]["shapes"] != masks[y]["shapes"]:
            print("SKIP control", ctl, y); continue
        s, _ = pair_stats(masks[ctl], masks[y]); rows.append(dict(group="CONTROL cross-base", mask_A=ctl, mask_B=y, **s))
        print("CONTROL", ctl, y, json.dumps(rowcol(masks[ctl], masks[y])))
    if rows: write_csv(rdir + "/pairs_control_cross_base_qwen7b.csv", rows)

if __name__ == "__main__":
    base = os.environ.get("SC_ROOT", ".") + "/item1-overlap"
    main(sys.argv[1] if len(sys.argv) > 1 else base + "/masks", sys.argv[2] if len(sys.argv) > 2 else base + "/results")
