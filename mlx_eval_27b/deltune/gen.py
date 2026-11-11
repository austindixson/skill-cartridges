"""Deterministic greedy generation on MLX for lists of token-id prompts.

Batched path: mlx-lm's own continuous-batching `BatchGenerator` (right-padded prefill with per-sequence cache
offsets, KV cache filtered as sequences finish). Reference path: mlx-lm's single-sequence `generate_step`.

Stop rule (mirrors vLLM SamplingParams(temperature=0, max_tokens=N) for this model): generation ends when the model
emits an id in `eos_ids` (Qwen2.5-0.5B base: only 151643 <|endoftext|>, from generation_config.json == tokenizer
eos), or after `max_new_tokens`, or when prompt+generated reaches `max_model_len` (vLLM's context limit).
The stop token counts as a generated token (as in vLLM's token_ids) but is not part of `token_ids`/text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import mlx.core as mx
from mlx_lm.generate import BatchGenerator, generate_step


def greedy_first_index(logprobs: mx.array) -> mx.array:
    """Argmax that breaks exact ties toward the lowest token id, independent of batch shape.

    bf16 logits tie exactly all the time (ulp is 0.125 at logit ~20); mx.argmax's tie winner depends on the
    reduction kernel chosen for the array shape, so single-stream and batched runs would pick different ids.
    """
    top = logprobs.max(axis=-1, keepdims=True)
    ids = mx.arange(logprobs.shape[-1], dtype=mx.int32)
    return mx.where(logprobs == top, ids, logprobs.shape[-1]).min(axis=-1)


@dataclass
class GenResult:
    index: int
    token_ids: list[int]  # generated ids, stop token excluded
    n_generated: int  # generated tokens including a stop token, if any (as vLLM counts)
    finish_reason: str  # "stop" (EOS) | "length" (max_new_tokens or max_model_len)


def token_budget(prompt_len: int, max_new_tokens: int, max_model_len: int) -> int:
    budget = min(max_new_tokens, max_model_len - prompt_len)
    assert budget > 0, f"prompt of {prompt_len} tokens leaves no room under max_model_len={max_model_len}"
    return budget


def generate_batch(
    model,
    prompts: Sequence[Sequence[int]],
    eos_ids: Sequence[int],
    *,
    max_new_tokens: int = 4096,
    max_model_len: int = 4608,
    batch_size: int = 32,
    prefill_batch_size: int = 8,
    on_result: Callable[[GenResult], None] | None = None,
) -> list[GenResult]:
    """Greedy-decode all prompts; results come back in prompt order (on_result fires as each one finishes)."""
    budgets = [token_budget(len(p), max_new_tokens, max_model_len) for p in prompts]
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in eos_ids],
        sampler=greedy_first_index,
        completion_batch_size=batch_size,
        prefill_batch_size=prefill_batch_size,
    )
    uids = gen.insert([list(p) for p in prompts], budgets)
    index_of = {uid: i for i, uid in enumerate(uids)}
    toks: dict[int, list[int]] = {uid: [] for uid in uids}
    results: dict[int, GenResult] = {}
    try:
        while responses := gen.next_generated():
            for r in responses:
                n = len(toks[r.uid]) + 1
                if r.finish_reason != "stop":
                    toks[r.uid].append(r.token)
                if r.finish_reason is not None:
                    res = GenResult(index_of[r.uid], toks.pop(r.uid), n, r.finish_reason)
                    results[res.index] = res
                    if on_result:
                        on_result(res)
    finally:
        gen.close()
    assert len(results) == len(prompts), "some prompts never finished"
    return [results[i] for i in range(len(prompts))]


def generate_single(
    model,
    prompt: Sequence[int],
    eos_ids: Sequence[int],
    *,
    max_new_tokens: int = 4096,
    max_model_len: int = 4608,
    index: int = 0,
) -> GenResult:
    """Single-sequence greedy reference (mlx-lm generate_step, argmax, no batching or padding)."""
    budget = token_budget(len(prompt), max_new_tokens, max_model_len)
    out: list[int] = []
    finish = "length"
    n = 0
    for tok, _ in generate_step(mx.array(list(prompt)), model, max_tokens=budget, sampler=greedy_first_index):
        n += 1
        if tok in eos_ids:
            finish = "stop"
            break
        out.append(tok)
    return GenResult(index, out, n, finish)
