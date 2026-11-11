"""Deletion-mask decoder + live bf16 apply/eject for mlx models.
Mask format (packbits_zstd): per layer `model.layers.L.mlp.gate_proj.weight.{shape,packed}`;
packed = zstd(np.packbits(keep.ravel())), bit 1 = keep, 0 = deleted. Logic follows
item1-overlap/results/overlap.py."""
import json, re
import numpy as np
import zstandard
from safetensors import safe_open
import mlx.core as mx


def _layer_idx(k):
    return int(re.match(r"model\.layers\.(\d+)\.(?:mlp\.gate_proj|feed_forward\.w1)\.weight\.shape$", k).group(1))


def load_mask(mask_dir):
    """layer index -> boolean KEEP array (out, in). Asserts counts against config.json."""
    cfg = json.load(open(f"{mask_dir}/config.json"))
    assert cfg["format"] == "packbits_zstd", cfg.get("format")
    dctx = zstandard.ZstdDecompressor()
    out, total, zero = {}, 0, 0
    with safe_open(f"{mask_dir}/model.safetensors", framework="numpy") as f:
        keys = [k for k in f.keys() if k.endswith(".shape")]
        assert len(keys) * 2 == len(list(f.keys()))
        for k in sorted(keys, key=_layer_idx):
            shape = tuple(int(v) for v in f.get_tensor(k))
            n = shape[0] * shape[1]
            raw = np.frombuffer(dctx.decompress(f.get_tensor(k[:-6] + ".packed").tobytes()), dtype=np.uint8)
            assert raw.size == (n + 7) // 8
            keep = np.unpackbits(raw, count=n).astype(bool).reshape(shape)
            out[_layer_idx(k)] = keep
            total += n
            zero += n - int(keep.sum(dtype=np.int64))
    assert total == cfg["total_params"], (total, cfg["total_params"])
    assert zero == cfg["zero_params"], (zero, cfg["zero_params"])
    return out


def combine(masks, how):
    """union = delete what either deletes (AND of keeps); intersection = delete only what all delete (OR of keeps)."""
    assert how in ("union", "intersection") and masks
    op = np.logical_and if how == "union" else np.logical_or
    res = {}
    for l in masks[0]:
        k = masks[0][l].copy()
        for m in masks[1:]:
            assert m[l].shape == k.shape
            k = op(k, m[l])
        res[l] = k
    return res


def deleted_count(keep):
    return sum(int(k.size - k.sum(dtype=np.int64)) for k in keep.values())


def gate_proj_modules(model, keep=None):
    """Text decoder per-layer mlp.gate_proj modules. `model.model.layers` is the decoder for both plain
    mlx_lm models and the VLM-converted Qwen3.5 (Model.model -> language_model.model)."""
    mods = {i: layer.mlp.gate_proj for i, layer in enumerate(model.model.layers)}
    if keep is not None:
        assert sorted(mods) == sorted(keep), (len(mods), len(keep))
        for l, m in mods.items():
            assert logical_shape(m) == keep[l].shape, (l, logical_shape(m), keep[l].shape)
    return mods


def logical_shape(m):
    """(out, in) of a Linear / QuantizedLinear."""
    out, w_in = m.weight.shape
    if hasattr(m, "scales"):  # packed: in = w_in * 32 / bits
        w_in = w_in * 32 // m.bits
    return (out, w_in)


def apply_dense(model, keep):
    """Non-quantized gate_proj: weight <- where(keep, w, 0) in w's dtype. Returns {layer: original weight}."""
    mods = gate_proj_modules(model, keep)
    saved, zeros = {}, {}
    for l, m in mods.items():
        assert not hasattr(m, "scales"), "apply_dense needs non-quantized gate_proj"
        w = m.weight
        assert w.dtype in (mx.bfloat16, mx.float16, mx.float32)
        saved[l] = w
        m.weight = mx.where(mx.array(keep[l]), w, mx.zeros((), dtype=w.dtype))
        mx.eval(m.weight)
    for l, m in mods.items():
        ndel = int(keep[l].size - keep[l].sum(dtype=np.int64))
        z = int(mx.sum(m.weight == 0).item())
        zeros[l] = z
        assert z >= ndel, (l, z, ndel)
    return saved


def eject_dense(model, saved):
    mods = gate_proj_modules(model)
    for l, w in saved.items():
        mods[l].weight = w
    mx.eval([mods[l].weight for l in saved])
    for l, w in saved.items():
        assert mods[l].weight is w and mods[l].weight.dtype == w.dtype
