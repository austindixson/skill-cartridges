"""Stage A (toggle) for the 27B 4-bit model: name mapping, raw-bit digests, disk-vs-memory check, quant->mask toggle stats.

Name mapping: mask keys are text-only HF names `model.layers.N.mlp.gate_proj.weight`; in the mlx qwen3_5 `Model` the
module lives at `language_model.model.layers.N.mlp.gate_proj` (params .weight/.scales/.biases, checkpoint names
`language_model.model.layers.N.mlp.gate_proj.*`). `Model.language_model` is a `TextModel` that has `.model` itself, so
deltune's QuantMaskToggle (which resolves `layer_module(root, mask_key)`) is simply rooted at `model.language_model`:
the mask key then resolves unchanged. Everything is verified, not assumed (verify_mapping).
"""
from __future__ import annotations

import glob, hashlib, json, statistics, struct, time

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from deltune.mask import load_deleted
from deltune.quant import (BITS, GROUP_SIZE, MODE, QuantMaskToggle, dequant_groups, gather_groups, layer_module, pct, raw)

from .common import PREFIX

_NP = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32, "U32": np.uint32, "U8": np.uint8, "I64": np.int64, "F64": np.uint64}


# ------------------------------------------------------------------ disk (independent safetensors header parser)
def disk_index(model_dir) -> dict:
    """tensor name -> (shard path, header entry, data base offset)."""
    idx = {}
    for p in sorted(glob.glob(f"{model_dir}/*.safetensors")):
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k, v in hdr.items():
            if k != "__metadata__":
                assert k not in idx, f"{k} in two shards"
                idx[k] = (p, v, 8 + n)
    return idx


def disk_bytes(entry) -> bytes:
    p, v, base = entry
    a, b = v["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + a)
        return f.read(b - a)


def disk_raw(entry) -> np.ndarray:
    return np.frombuffer(disk_bytes(entry), dtype=_NP[entry[1]["dtype"]]).reshape(entry[1]["shape"])


# ------------------------------------------------------------------ digests
def raw_np(w: mx.array) -> np.ndarray:
    """Raw bits as numpy (bf16/f16 as uint16, f32 as uint32, uint32 as is)."""
    mx.eval(w)
    return raw(w)


def sha(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).data).hexdigest()


def param_hashes(model) -> dict[str, str]:
    """name -> sha256 of raw bytes, every parameter (packed uint32 weights, scales, biases, norms, ...), streamed."""
    return {n: sha(raw_np(w)) for n, w in sorted(tree_flatten(model.parameters()), key=lambda x: x[0])}


def combine(hs: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{n}:{h}\n" for n, h in sorted(hs.items())).encode()).hexdigest()


def disk_vs_memory(model, disk: dict, base_h: dict[str, str]) -> dict:
    """Compare every loaded tensor's bytes with the raw bytes in the shards. Names that do not map are reported."""
    flat = dict(tree_flatten(model.parameters()))
    not_on_disk = sorted(set(flat) - set(disk))
    mism, ok = [], 0
    for n in flat:
        if n in disk:
            same = hashlib.sha256(disk_bytes(disk[n])).hexdigest() == base_h[n]
            ok += same
            if not same:
                mism.append({"name": n, "disk_shape": disk[n][1]["shape"], "mem_shape": list(flat[n].shape)})
    return {"tensors_in_memory": len(flat), "equal_to_disk_bytes": ok, "params_not_on_disk": not_on_disk,
            "disk_bytes_mismatch": mism, "disk_tensors_not_loaded_sample": sorted(set(disk) - set(flat))[:5],
            "n_disk_tensors_not_loaded": len(set(disk) - set(flat)), "n_shards": len({e[0] for e in disk.values()})}


# ------------------------------------------------------------------ mapping + toggle
def verify_mapping(model, deleted: dict, disk: dict | None = None) -> dict:
    """Each mask key must map to exactly one quantized module whose logical (out, in) shape equals the mask shape."""
    flat = dict(tree_flatten(model.parameters()))
    out = {}
    for key, d in deleted.items():
        base = PREFIX + key[: -len(".weight")]
        assert key.endswith(".weight")
        names = [base + s for s in (".weight", ".scales", ".biases")]
        assert all(n in flat for n in names), f"{key}: missing params {[n for n in names if n not in flat]}"
        assert sum(n.startswith(base + ".") for n in flat) == 3, f"{key}: unexpected extra params under {base}"
        w = flat[names[0]]
        assert w.dtype == mx.uint32 and (w.shape[0], w.shape[1] * 32 // BITS) == d.shape, (key, w.shape, d.shape)
        lay = layer_module(model.language_model, key)
        assert (lay.group_size, lay.bits, lay.mode) == (GROUP_SIZE, BITS, MODE), (key, lay.group_size, lay.bits, lay.mode)
        if disk is not None:
            assert all(n in disk for n in names), f"{key}: not in checkpoint headers"
        out[key] = base
    assert len(set(out.values())) == len(out)
    return out


def make_toggle(model, mask_dir) -> QuantMaskToggle:
    verify_mapping(model, load_deleted(str(mask_dir)))
    return QuantMaskToggle(model.language_model, str(mask_dir), "quant_then_mask")


def layer_stats(tg: QuantMaskToggle, name: str) -> tuple[dict, np.ndarray]:
    """Masked-entry statistics of one gate_proj with the toggle OFF (ON values come from the toggle's stored arrays,
    identical to what on() scatters; stage A re-reads the live layer after on() and asserts equality).
    Returns (row, |dequantized ON value| / std(dequantized base layer) at every masked entry)."""
    d, v = tg.deleted[name], tg.layers[name]
    out_f, in_f = d.shape
    ng = in_f // GROUP_SIZE
    gid = np.array(v["gid"]).astype(np.int64)
    idx = d.indices.astype(np.int64)
    pos = np.searchsorted(gid, (idx // in_f) * ng + (idx % in_f) // GROUP_SIZE)
    col = idx % GROUP_SIZE
    q_old, s_old, b_old = v["old"]
    q_new, s_new, b_new = v["new"]
    dq = lambda q, s, b: np.array(dequant_groups(q, s, b).astype(mx.float32))
    dq_old, dq_new = dq(q_old, s_old, b_old), dq(q_new, s_new, b_new)
    changed = (raw(q_old) != raw(q_new)).any(1) | (raw(s_old) != raw(s_new)) | (raw(b_old) != raw(b_new))
    lay = layer_module(tg.model, name)
    std = float(mx.std(mx.dequantize(lay.weight, lay.scales, lay.biases, GROUP_SIZE, BITS, MODE).astype(mx.float32)))
    on_vals = dq_new[pos, col]
    rel = (np.abs(on_vals) / std).astype(np.float32)
    row = {"deleted": int(idx.size), "affected_groups": int(gid.size), "total_groups": int(v["n_groups_total"]),
           "changed_groups": int(changed.sum()), "already_zero_in_base": int((dq_old[pos, col] == 0).sum()),
           "exact_zero_after_on": int((on_vals == 0).sum()), "layer_dequant_std": std,
           "rel_abs_median": float(np.median(rel)), "rel_abs_p95": float(np.percentile(rel, 95)), "rel_abs_max": float(rel.max())}
    return row, rel


def verify_on_state(tg: QuantMaskToggle, base_h: dict[str, str]) -> dict:
    """With the toggle ON: masked layers hold the ON values in affected groups, and with the stashed originals scattered
    back every masked-layer tensor hashes to base; every other tensor hashes to base directly."""
    model = tg.model
    bad_on, bad_other, n_other = [], [], 0
    masked = {}
    for name, v in tg.layers.items():
        lay = layer_module(model, name)
        gmx = v["gid"]
        got = gather_groups(lay, gmx)
        if not all(np.array_equal(raw(a), raw(b)) for a, b in zip(got, v["new"])):
            bad_on.append(name)
        gid = np.array(gmx).astype(np.int64)
        base_name = PREFIX + name[: -len(".weight")]
        for suffix, arr, old in (("weight", lay.weight, v["old"][0]), ("scales", lay.scales, v["old"][1]), ("biases", lay.biases, v["old"][2])):
            a = raw_np(arr).copy()
            a2 = a.reshape(v["n_groups_total"], -1)
            a2[gid] = raw(old).reshape(len(gid), -1)
            masked[f"{base_name}.{suffix}"] = sha(a)
    for n, w in sorted(((PREFIX + k, v) for k, v in tree_flatten(model.parameters())), key=lambda x: x[0]):  # toggle is rooted at language_model
        h = masked.get(n)
        if h is None:
            h, n_other = sha(raw_np(w)), n_other + 1
            if h != base_h[n]:
                bad_other.append(n)
        elif h != base_h[n]:
            bad_other.append(n)
    assert n_other + len(masked) == len(base_h), "on-state check did not cover every base tensor"
    return {"layers_with_on_values_in_place": len(tg.layers) - len(bad_on), "layers_wrong_on_values": bad_on,
            "masked_layer_tensors_checked": len(masked), "other_tensors_checked": n_other,
            "mismatching_tensors": bad_other}


def logits_of(model, ids: list[int]) -> mx.array:
    l = model(mx.array(ids)[None]).astype(mx.float32)
    mx.eval(l)
    return l


def run_toggle(model, mask_dir, model_dir, prompt_ids: list[int], cycles: int = 3, log=print, tg=None) -> dict:
    """Whole of stage A. Asserts the OFF/ON invariants; returns the result dict (no side effects besides toggling)."""
    out = {}
    disk = disk_index(model_dir)
    deleted = load_deleted(str(mask_dir))
    out["name_mapping"] = {"rule": "mask key K -> param 'language_model.'+K[:-len('.weight')]+{.weight,.scales,.biases}; toggle rooted "
                                   "at model.language_model (mask key resolves unchanged)", "layers": len(deleted),
                           "verified": bool(verify_mapping(model, deleted, disk))}
    t = time.perf_counter(); base_h = param_hashes(model); out["digest_pass_s"] = time.perf_counter() - t
    out["base_digest"] = combine(base_h); out["n_tensors"] = len(base_h)
    out["digest_method"] = "sha256 over 'name:sha256(raw bytes)\\n' lines, sorted names; bf16 as uint16 bits, uint32 packed words as is"
    t = time.perf_counter(); out["disk_vs_memory"] = disk_vs_memory(model, disk, base_h); out["disk_vs_memory"]["s"] = time.perf_counter() - t
    log("base digest", out["base_digest"][:16], "disk_vs_memory ok", out["disk_vs_memory"]["equal_to_disk_bytes"], "/", out["disk_vs_memory"]["tensors_in_memory"])
    l_base = logits_of(model, prompt_ids)

    tg = tg or make_toggle(model, mask_dir)
    out["build_ms"] = tg.timings_ms["build"][0]
    out["stash_bytes"] = tg.stash_nbytes(); out["on_values_bytes"] = tg.on_values_nbytes()
    rows, rels = {}, []
    for name in tg.layers:
        rows[name], rel = layer_stats(tg, name); rels.append(rel)
    allrel = np.concatenate(rels); del rels
    tot = lambda k: int(sum(r[k] for r in rows.values()))
    out["groups"] = {"affected": tg.n_affected_groups, "changed": tot("changed_groups"), "total_in_masked_layers": tg.n_groups_total}
    out["masked_entries"] = {"deleted_total": tot("deleted"), "already_zero_in_4bit_base": tot("already_zero_in_base"),
                             "exact_zero_after_on": tot("exact_zero_after_on"),
                             "fraction_exact_zero_after_on": tot("exact_zero_after_on") / tot("deleted"),
                             "rel_abs_dequant_after_on_over_layer_std": {"median": float(np.median(allrel)), "p95": float(np.percentile(allrel, 95)),
                                                                         "max": float(allrel.max()), "mean": float(allrel.mean())},
                             "per_layer": rows}
    del allrel
    log("stats: deleted", tot("deleted"), "exact zero after ON", tot("exact_zero_after_on"))

    on_t, off_t = [], []
    for i in range(cycles):
        tg.on(); on_t.append(tg.timings_ms["on"][-1])
        if i == 0:
            out["on_state"] = verify_on_state(tg, base_h)
            assert not out["on_state"]["mismatching_tensors"] and not out["on_state"]["layers_wrong_on_values"], out["on_state"]
            l_on = logits_of(model, prompt_ids)
        tg.off(); off_t.append(tg.timings_ms["off"][-1])
    tg.on(); tg.off()
    off_h = param_hashes(model)
    assert off_h == base_h, [n for n in base_h if off_h.get(n) != base_h[n]][:5]
    l_off = logits_of(model, prompt_ids)
    d_off, d_on = float(mx.abs(l_base - l_off).max()), float(mx.abs(l_base - l_on).max())
    assert d_off == 0.0 and d_on > 0.0, (d_off, d_on)
    out["off_exact"] = {"per_tensor_hashes_equal_base": True, "digest_after_off": combine(off_h), "digest_base": out["base_digest"],
                        "logits_prompt_tokens": len(prompt_ids), "max_abs_diff_base_vs_off": d_off, "max_abs_diff_base_vs_on": d_on,
                        "cycles_before_check": cycles + 1}
    out["timings_ms"] = {"on_median": statistics.median(on_t), "off_median": statistics.median(off_t), "on_raw": on_t, "off_raw": off_t,
                         "note": "on/off include mx.eval of all language-model params; build_ms includes mask decode + per-layer requantization"}
    return out
