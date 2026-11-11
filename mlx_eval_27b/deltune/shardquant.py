"""Layer-by-layer mask + MLX 4-bit quantization, never materializing the whole bf16 model.

Weights are bound lazily from the safetensors shards (mlx_lm.utils.load_model(lazy=True)). For each decoder layer:
materialize it (bf16, one layer), scatter zeros into mlp.gate_proj.weight at the masked indices (raw 16-bit bits, as in
deltune.toggle), quantize with mlx-lm's defaults (affine, group_size 64, bits 4; every Linear/Embedding whose last dim % 64 == 0),
evaluate, move on. Embeddings and lm_head are quantized the same way. Peak extra memory is about one bf16 layer.
Nothing is written to disk. Mask-ON vs base are two separate builds (call this twice, one at a time).
"""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.utils import load_model, load_tokenizer

from .mask import iter_deleted

GS, BITS = 64, 4


def _zero_entries(w: mx.array, idx: np.ndarray) -> mx.array:
    raw = w.view(mx.uint16).reshape(-1)
    raw[mx.array(idx)] = 0
    out = raw.reshape(w.shape).view(w.dtype)
    mx.eval(out)
    return out


def build_quantized(model_dir: str, mask_dir: str | None = None, group_size: int = GS, bits: int = BITS):
    """Return (model, tokenizer, stats). mask_dir=None builds the plain 4-bit base."""
    t0 = time.perf_counter()
    model, config = load_model(Path(model_dir), lazy=True)
    tok = load_tokenizer(Path(model_dir))
    deleted = {d.name: d for d in iter_deleted(mask_dir)} if mask_dir else {}
    layers = model.model.layers
    stats = {"layers": len(layers), "group_size": group_size, "bits": bits, "mask": mask_dir, "per_layer_ms": [],
             "deleted_total": 0, "deleted_exactly_zero_after_quant": 0}
    for i, layer in enumerate(layers):
        t = time.perf_counter()
        name = f"model.layers.{i}.mlp.gate_proj.weight"
        if name in deleted:
            d = deleted[name]
            w = layer.mlp.gate_proj.weight
            assert w.dtype == mx.bfloat16 and tuple(w.shape) == d.shape, (name, w.dtype, w.shape)
            layer.mlp.gate_proj.weight = _zero_entries(w, d.indices.astype(np.int64))
        nn.quantize(layer, group_size=group_size, bits=bits)
        mx.eval(layer.parameters())
        if name in deleted:
            q = layer.mlp.gate_proj
            deq = mx.dequantize(q.weight, q.scales, q.biases, group_size=group_size, bits=bits).reshape(-1)
            z = int((deq[mx.array(deleted[name].indices.astype(np.int64))] == 0).sum())
            stats["deleted_total"] += deleted[name].n_deleted
            stats["deleted_exactly_zero_after_quant"] += z
        stats["per_layer_ms"].append((time.perf_counter() - t) * 1e3)
        mx.clear_cache()
    nn.quantize(model, group_size=group_size, bits=bits, class_predicate=lambda p, m: hasattr(m, "to_quantized")
                and not p.startswith("model.layers") and m.weight.shape[-1] % group_size == 0)  # embed_tokens, lm_head
    mx.eval(model.parameters())
    assert not mask_dir or stats["deleted_total"] == sum(d.n_deleted for d in deleted.values())
    stats["build_seconds"] = time.perf_counter() - t0
    stats["active_memory_gb"] = mx.get_active_memory() / 1e9
    stats["peak_memory_gb"] = mx.get_peak_memory() / 1e9
    return model, tok, stats
