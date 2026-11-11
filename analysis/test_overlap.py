import os  # SC_ROOT: folder holding the mask and result directories (see README)
import random, sys
sys.path.insert(0, os.environ.get("SC_ROOT", ".") + "/item1-overlap/results")
import numpy as np, zstandard, overlap
random.seed(1); rng = np.random.default_rng(1)
def mk(keeps):  # emulate upstream compress_mask output, then decode as overlap does
    layers = []
    for keep in keeps:
        n = keep.size; raw = np.frombuffer(zstandard.ZstdCompressor(level=3).compress(np.packbits(keep).tobytes()), np.uint8)
        dec = np.frombuffer(zstandard.ZstdDecompressor().decompress(raw.tobytes()), np.uint8)
        k = np.unpackbits(dec, count=n); layers.append(dict(shape=(n // 7, 7), n=n, packed=np.packbits(1 - k), deleted=int(n - k.sum())))
    return dict(layers=layers, N=sum(l["n"] for l in layers), Z=sum(l["deleted"] for l in layers))
sizes = [7 * 13, 7 * 5, 7 * 3]   # 91,35,21 bits: none multiple of 8
for trial in range(20):
    dels = [[set(random.sample(range(n), random.randint(0, n))) for n in sizes] for _ in range(2)]
    ms = []
    for d in dels:
        keeps = []
        for n, s in zip(sizes, d):
            k = np.ones(n, np.uint8); k[list(s)] = 0; keeps.append(k)
        ms.append(mk(keeps))
    s, il = overlap.pair_stats(*ms)
    # global index sets via offsets
    off = np.cumsum([0] + sizes)
    SA = {off[i] + x for i, st in enumerate(dels[0]) for x in st}; SB = {off[i] + x for i, st in enumerate(dels[1]) for x in st}
    assert s["A"] == len(SA) and s["B"] == len(SB) and s["inter"] == len(SA & SB) and s["union"] == len(SA | SB), (s, len(SA), len(SB))
    assert s["N"] == sum(sizes) and abs(s["J"] - len(SA & SB) / len(SA | SB)) < 1e-15 if SA | SB else True
    El = sum(len(a) * len(b) / n for a, b, n in zip(dels[0], dels[1], sizes)); assert abs(s["E_layer"] - El) < 1e-9
    assert sum(il) == len(SA & SB)
# padding bits: all-kept layer must give zero deleted despite pad zeros
m = mk([np.ones(13, np.uint8)]); assert m["Z"] == 0 and overlap.popcount(m["layers"][0]["packed"]) == 0
m = mk([np.zeros(13, np.uint8)]); assert m["Z"] == 13 and overlap.popcount(m["layers"][0]["packed"]) == 13
print("ALL TESTS PASS")
