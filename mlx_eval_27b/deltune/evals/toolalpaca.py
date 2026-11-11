"""ToolAlpaca exact-match eval (SDFT authors' protocol) for MLX.

Scoring functions are loaded at runtime from the SDFT authors' pinned `eval_tooluse.py`
(idanshen/Self-Distillation @ d7757321), not redistributed here. Prompt: upstream `load_test_data`: one user turn = dataset `prompt`, tokenizer chat template with
add_generation_prompt=True (Qwen2.5 template auto-inserts its default system prompt). Greedy, max_tokens 1024,
stop on tokenizer.eos_token_id (<|im_end|>) -- plus <|endoftext|> which vLLM also takes from generation_config.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_EVAL = Path(os.environ.get("SDFT_DIR", ROOT / "vendor/Self-Distillation")) / "eval_tooluse.py"
DATA_DIR = Path(os.environ.get("SDFT_DIR", ROOT / "vendor/Self-Distillation")) / "data/tooluse_data/eval_data"
MAX_NEW_TOKENS = 1024
PAPER = {"base": (40, 97), "on": (67, 97)}


# The three scoring functions (extract_actions, extract_action_inputs, evaluate_correctness) are NOT redistributed
# here: they belong to the SDFT authors' repository (idanshen/Self-Distillation), which carries no license file.
# They are loaded at runtime from a local checkout of that repo at the pinned commit (see README, "Evaluation data").
def _load_upstream_scorer() -> dict:
    import ast
    tree = ast.parse(UPSTREAM_EVAL.read_text())
    wanted = {"extract_actions", "extract_action_inputs", "evaluate_correctness"}
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in keep} == wanted, f"scoring functions missing in {UPSTREAM_EVAL}"
    ns: dict = {"re": re, "json": json, "Counter": Counter}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(UPSTREAM_EVAL), "exec"), ns)  # noqa: S102
    return ns


def extract_actions(text):
    return _load_upstream_scorer()["extract_actions"](text)


def extract_action_inputs(text):
    return _load_upstream_scorer()["extract_action_inputs"](text)


def evaluate_correctness(responses, golden_answers):
    return _load_upstream_scorer()["evaluate_correctness"](responses, golden_answers)


def load_items(tokenizer) -> list[dict]:
    """Upstream load_test_data, same order (97 items); adds `prompt_text` (chat-templated string)."""
    from datasets import load_from_disk
    data = load_from_disk(str(DATA_DIR)).to_list()
    for ex in data:
        ex["prompt_text"] = tokenizer.apply_chat_template(
            [{"role": "user", "content": ex["prompt"]}], tokenize=False, add_generation_prompt=True)
    return data
