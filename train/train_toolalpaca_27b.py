# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch==2.15.0",
#   "transformers==5.22.2",
#   "accelerate==1.15.0",
#   "datasets==5.2.0",
#   "numpy==2.5.4",
#   "zstandard==0.25.0",
#   "safetensors==0.8.0",
#   "huggingface_hub==2.4.0",
#   "flash-linear-attention==0.6.0 ; sys_platform == 'linux'",
# ]
# ///
"""Deletion Tuning (arXiv 2611.10866) SFT mask training on ToolAlpaca tool use, for big Qwen3.5-family models.

What it does, in order:
  1. fetch upstream `deleting-weights` (pinned commit) and the SDFT authors' repo (pinned commit, ToolAlpaca data and
     their scoring code) as GitHub tarballs, unless local copies are given;
  2. load the text part of the model in bf16 (AutoModelForCausalLM) and split its decoder layers over the devices;
  3. replace every text-layer `mlp.gate_proj` nn.Linear by upstream's SparseLinear, whose mask logits live on the
     weight's own device (upstream `convert_to_sparse_model` puts them all on `model.device`), Wanda-initialise the
     logits with upstream `wanda_calibration`, free the Wanda scores;
  4. run 3 probe optimizer steps, measure s/step, and fit the number of steps to `--wallclock_budget_s`
     (exit code 3 immediately if fewer than `--min_steps` fit);
  5. evaluate the base model (greedy, ToolAlpaca test set, exact match with the SDFT authors' `eval_tooluse.py`
     scoring functions, loaded from the pinned file), train, checkpoint hard masks, evaluate the masked model;
  6. write `mask/` (upstream `save_mask` format), `metrics.jsonl`, `summary.json`; optionally upload them to a private
     HF storage bucket (OFF unless `--out_bucket` or DELTUNE_OUT_BUCKET is set).

Recipe defaults are upstream `scripts/train_sft.sh` (400 steps, lr 5e-4, batch 4, sharpness 100->1000, init sparsity
1/2048, prior strength 3, weight_reg 1e-5). Deviations from upstream train_sft.py, each on purpose:
  * the mask is built once per optimizer step (upstream `build_step_weights`) instead of in every forward and
    checkpoint recomputation; the sparse-weight regulariser, which that path does not support, is added to the logit
    gradients analytically (identical gradient, checked in tests); `--mask_path upstream` runs upstream's own path;
  * the loss runs lm_head only on answer-token positions (same mean cross-entropy as `model(labels=...)`, avoids a
    [B, S, 248k] fp32 logits tensor);
  * AdamW is fused (CUDA) or non-foreach (otherwise) to avoid foreach temporaries on 45 GB of optimizer state;
  * Wanda calibration uses `--wanda_samples` training prompts, not the whole training set;
  * layer-split over devices with plain forward pre-hooks (no accelerate dispatch), so cpu+mps splits also work;
  * no accelerate Accelerator; bf16 autocast only on CUDA.
"""

import argparse
import ast
import gc
import json
import math
import os
import random
import resource
import signal
import sys
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils._pytree import tree_map

UPSTREAM_REPO = "agentic-learning-ai-lab/deleting-weights"
UPSTREAM_SHA = "94f0340ebb574f95bb41d844165253a6785e215e"
SDFT_REPO = "idanshen/Self-Distillation"
SDFT_SHA = "d77573212fa0a3ae2eeb64b9b44db1c251f75e3e"
EXIT_BUDGET = 3
EXIT_UPLOAD = 4
T0 = time.monotonic()
STOP = {"flag": False}


def log(msg: str) -> None:
    """Print a timestamped line (seconds since script start) and flush."""
    print(f"[{time.monotonic() - T0:8.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------------------------------------------
# upstream fetching
# ----------------------------------------------------------------------------------------------------------------
def fetch_pinned(repo: str, sha: str, cache_dir: Path) -> Path:
    """Download the GitHub tarball of `repo` at `sha` (no git needed) and return the extracted directory."""
    dest = cache_dir / f"{repo.split('/')[1]}-{sha}"
    if dest.is_dir():
        return dest
    cache_dir.mkdir(parents=True, exist_ok=True)
    tar_path = cache_dir / f"{repo.split('/')[1]}-{sha}.tar.gz"
    url = f"https://codeload.github.com/{repo}/tar.gz/{sha}"
    log(f"downloading {url}")
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(cache_dir, filter="data")
    assert dest.is_dir(), f"tarball did not extract to {dest}"
    return dest


def check_git_head(path: Path, sha: str) -> None:
    """If `path` is a git checkout (or its parent repo is), assert its HEAD is the pinned commit."""
    import subprocess

    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    if out.returncode == 0:
        assert out.stdout.strip() == sha, f"{path} is at {out.stdout.strip()}, expected pinned {sha}"


def load_scorer(sdft_dir: Path) -> dict:
    """Load the SDFT authors' scoring functions from their pinned `eval_tooluse.py` without importing vllm.

    Only the three scoring FunctionDefs and the stdlib imports they need are compiled; everything else in that file
    (vllm, argparse, main) is dropped.
    """
    src_path = sdft_dir / "eval_tooluse.py"
    tree = ast.parse(src_path.read_text())
    wanted = {"extract_actions", "extract_action_inputs", "evaluate_correctness"}
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in keep} == wanted, f"scoring functions missing in {src_path}"
    imports = [n for n in tree.body if isinstance(n, ast.Import) and {a.name for a in n.names} <= {"re", "json"}]
    imports += [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "collections"]
    module = ast.Module(body=imports + keep, type_ignores=[])
    ns: dict = {}
    exec(compile(module, str(src_path), "exec"), ns)  # noqa: S102 (pinned, reviewed upstream source)
    return ns


# ----------------------------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------------------------
def chat_prompt(tokenizer, user_text: str, thinking: bool) -> str:
    """The SDFT eval prompt: one user turn, generation prompt; Qwen3.5 templates open `<think>` unless disabled."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, enable_thinking=thinking
    )


def build_data(args, tokenizer, sdft_dir: Path, sft_data):
    """Return (train_examples, test_examples). Train = prompt + '\\n'.join(golden_response) + eos, loss on answer only."""
    from datasets import load_from_disk

    root = sdft_dir / "data" / "tooluse_data"
    train_raw = load_from_disk(str(root / "train_data")).to_list()
    test_raw = load_from_disk(str(root / "eval_data")).to_list()
    log(f"ToolAlpaca: {len(train_raw)} train, {len(test_raw)} test (full sets)")
    rng = random.Random(args.seed)
    rng.shuffle(train_raw)
    train, dropped = [], 0
    for ex in train_raw:
        prompt_ids = tokenizer(chat_prompt(tokenizer, ex["prompt"], args.thinking), add_special_tokens=False,
                               return_tensors="pt")["input_ids"][0]
        answer = "\n".join(ex["golden_response"]) + tokenizer.eos_token  # eos = <|im_end|>
        answer_ids = tokenizer(answer, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(prompt_ids) + len(answer_ids) > args.max_seq_len:
            dropped += 1
            continue
        train.append(sft_data.encode(prompt_ids=prompt_ids, answer_ids=answer_ids))
        if args.train_limit and len(train) >= args.train_limit:
            break
    test = test_raw[: args.eval_limit] if args.eval_limit else test_raw
    log(f"train examples kept {len(train)} (dropped {dropped} longer than {args.max_seq_len} tokens); eval n={len(test)}")
    assert len(train) >= args.batch_size, "not enough training examples"
    return train, test


@torch.no_grad()
def evaluate_em(model, tokenizer, test, scorer: dict, args, d0: torch.device, tag: str) -> dict:
    """Greedy HF generate on the test set, exact match by the SDFT scoring functions. Returns metrics + responses."""
    model.eval()
    prompts = [chat_prompt(tokenizer, ex["prompt"], args.thinking) for ex in test]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    responses = [None] * len(prompts)
    t0 = time.monotonic()
    for b in range(0, len(order), args.eval_batch_size):
        idx = order[b : b + args.eval_batch_size]
        enc = tokenizer([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(d0)
        out = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False, temperature=None, top_p=None, top_k=None,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
        for j, i in enumerate(idx):
            responses[i] = tokenizer.decode(out[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        log(f"[{tag}] eval {min(b + len(idx), len(order))}/{len(order)} elapsed {time.monotonic() - t0:.0f}s")
    scores = scorer["evaluate_correctness"](responses, [ex["golden_answer"] for ex in test])
    return {"em": sum(scores) / len(scores), "correct": int(sum(scores)), "n": len(scores), "seconds": time.monotonic() - t0,
            "responses": [{"response": r, "golden_answer": ex["golden_answer"], "correct": bool(s)}
                          for r, ex, s in zip(responses, test, scores)]}


# ----------------------------------------------------------------------------------------------------------------
# device split
# ----------------------------------------------------------------------------------------------------------------
def resolve_devices(spec: str) -> list[torch.device]:
    """`auto` = all CUDA devices, else MPS, else CPU; otherwise a comma list such as `cuda:0,cuda:1` or `mps,cpu`."""
    if spec == "auto":
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.empty(0, device="mps" if torch.backends.mps.is_available() else "cpu").device]
    return [torch.empty(0, device=s).device for s in spec.split(",")]  # normalises "mps" -> mps:0


def layer_device_index(n_layers: int, n_devices: int) -> list[int]:
    """Contiguous equal split of the layers: 64 layers on 2 devices -> 0 x32 then 1 x32."""
    return [min(i * n_devices // n_layers, n_devices - 1) for i in range(n_layers)]


def _mover(device: torch.device):
    """Forward pre-hook that moves every tensor in the call's args and kwargs to `device`."""
    def move(_module, args, kwargs):
        f = lambda x: x.to(device) if isinstance(x, torch.Tensor) else x  # noqa: E731
        return tree_map(f, args), tree_map(f, kwargs)
    return move


def place_model(model: nn.Module, devices: list[torch.device]) -> list[int]:
    """Put embeddings on devices[0], decoder layers split evenly, norm + lm_head on devices[-1]; return layer->device."""
    inner = model.model
    layers = inner.layers
    if len(devices) > 1:
        assert model.lm_head.weight is not inner.embed_tokens.weight, "tied embeddings cannot be split over devices"
    d_of = layer_device_index(len(layers), len(devices))
    inner.embed_tokens.to(devices[0])
    inner.rotary_emb.to(devices[0])
    for i, layer in enumerate(layers):
        layer.to(devices[d_of[i]])
        if len(devices) > 1:
            layer.register_forward_pre_hook(_mover(devices[d_of[i]]), with_kwargs=True)
    inner.norm.to(devices[-1])
    model.lm_head.to(devices[-1])
    if len(devices) > 1:
        inner.norm.register_forward_pre_hook(_mover(devices[-1]), with_kwargs=True)
        model.lm_head.register_forward_pre_hook(_mover(devices[-1]), with_kwargs=True)
        # generate() concatenates sampled tokens onto input_ids that live on devices[0]
        model.lm_head.register_forward_hook(lambda _m, _a, out: out.to(devices[0]))
    placed = {p.device for p in model.parameters()} | {b.device for b in model.buffers()}
    assert placed <= set(devices), f"parameters left on {placed - set(devices)}"
    return d_of


def convert_text_gate_proj(model: nn.Module, SparseLinear, sharpness: list[float], total_steps: int) -> int:
    """Replace each text-layer `mlp.gate_proj` Linear by a SparseLinear with logits on the weight's own device.

    Upstream `convert_to_sparse_model` would also work for the module swap but gives every layer's logits
    `model.device`, i.e. all on the first device; with a layer split that is wrong, so this builds the SparseLinear
    (upstream class) directly with `target_device=weight.device`. The frozen weight Parameter is reused, not copied.
    """
    n = 0
    for name, mlp in ((f"model.layers.{i}.mlp", l.mlp) for i, l in enumerate(model.model.layers)):
        child = mlp.gate_proj
        assert isinstance(child, nn.Linear) and child.bias is None, f"{name}.gate_proj is {type(child)}"
        sl = SparseLinear(child.in_features, child.out_features, sharpness=sharpness, total_steps=total_steps,
                          device=torch.device("meta"), dtype=child.weight.dtype, target_device=child.weight.device)
        sl.weight = child.weight  # frozen bf16 weight stays where it is
        mlp.gate_proj = sl
        n += 1
    return n


# ----------------------------------------------------------------------------------------------------------------
# loss / step
# ----------------------------------------------------------------------------------------------------------------
def answer_loss(model: nn.Module, batch: dict, d0: torch.device) -> torch.Tensor:
    """Mean next-token cross-entropy over answer tokens (labels != -100), lm_head applied to those positions only."""
    ids = batch["input_ids"].to(d0)
    am = batch["attention_mask"].to(d0)
    labels = batch["label_ids"]
    seq = ids.shape[1]
    offsets = (seq - am.sum(dim=1)).unsqueeze(1)  # left padding: positions count from the first real token
    pos = (torch.arange(seq, device=d0).unsqueeze(0) - offsets).clamp(min=0)
    hidden = model.model(input_ids=ids, attention_mask=am, position_ids=pos, use_cache=False).last_hidden_state
    tgt = labels[:, 1:]
    sel = (tgt != -100).to(hidden.device)
    logits = model.lm_head(hidden[:, :-1][sel]).float()
    return F.cross_entropy(logits, tgt.to(logits.device)[sel.to(logits.device)])


def sparse_layers(model: nn.Module, SparseLinear) -> list:
    """All SparseLinear modules in model order."""
    return [m for m in model.modules() if isinstance(m, SparseLinear)]


@torch.no_grad()
def add_regularizer_grad(layers: list, lam: float) -> None:
    """Add d/d logit of (-lam * sum((W * m)^2)), m = sigmoid(logit * s), to each logit's grad.

    This is the gradient of upstream's `loss - weight_reg * (1 - progress) * sum_sparse_weight_norms` for the
    step-weight path, which cannot carry that term through autograd. d/dg [-lam W^2 m^2] = -2 lam W^2 m * s m (1 - m).
    """
    for m in layers:
        dm = m.diff_mask
        s = m._step_sharpness
        soft = torch.sigmoid(dm.gate * s)
        w2 = m.weight.reshape(-1).float().square()
        dm.gate.grad.add_(w2.mul_(soft).mul_(soft).mul_(1.0 - soft).mul_(-2.0 * lam * s))


def train_step(model, batch, layers, sl_mod, args, d0, step: int, total: int, autocast) -> tuple[float, float]:
    """One optimizer-step worth of forward + backward; returns (ce_loss, regularizer value or 0). Grads left on logits."""
    lam = args.weight_reg * (1.0 - step / total)
    if args.mask_path == "step":
        sl_mod.build_step_weights(model)
        with autocast():
            ce = answer_loss(model, batch, d0)
        ce.backward()
        if lam > 0:
            add_regularizer_grad(layers, lam)
        sl_mod.drop_step_weights(model)
        return ce.item(), 0.0
    with autocast():
        ce = answer_loss(model, batch, d0)
        reg = sum(m.sparse_weight_norm.float().to(ce.device) for m in layers if hasattr(m, "sparse_weight_norm"))
        loss = ce - lam * reg if lam > 0 else ce
    loss.backward()
    return ce.item(), float(reg.detach()) if lam > 0 else 0.0


# ----------------------------------------------------------------------------------------------------------------
# budget
# ----------------------------------------------------------------------------------------------------------------
def fit_steps(requested: int, done: int, s_step: float, time_left_s: float, reserve_s: float, save_cost_s: float,
              save_steps: int) -> int:
    """Largest total step count <= requested whose remaining steps fit in `time_left_s - reserve_s`.

    Each remaining step is charged 5% over the measured s/step plus its share of periodic checkpoint cost.
    """
    if math.isinf(time_left_s):
        return requested
    per_step = s_step * 1.05 + save_cost_s / max(save_steps, 1)
    avail = time_left_s - reserve_s
    more = int(avail // per_step) if avail > 0 else 0
    return min(requested, done + more)


def set_total_steps(layers: list, total: int) -> None:
    """Change the sharpness schedule length of every mask (progress = current_step / total_steps)."""
    for m in layers:
        m.diff_mask.total_steps = total


def peak_mem(devices: list[torch.device]) -> dict:
    """Peak allocated GB per CUDA device since the last reset, else process peak RSS."""
    out = {}
    for d in devices:
        if d.type == "cuda":
            out[str(d)] = torch.cuda.max_memory_allocated(d) / 1e9
    out["process_peak_rss_gb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    if any(d.type == "mps" for d in devices):
        out["mps_driver_allocated_gb"] = torch.mps.driver_allocated_memory() / 1e9
    return out


def reset_peak(devices: list[torch.device]) -> None:
    """Reset CUDA peak-memory counters."""
    for d in devices:
        if d.type == "cuda":
            torch.cuda.reset_peak_memory_stats(d)


def sync(devices: list[torch.device]) -> None:
    """Block until queued work on all devices finished (for honest timing)."""
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)
        elif d.type == "mps":
            torch.mps.synchronize()


# ----------------------------------------------------------------------------------------------------------------
# artifacts
# ----------------------------------------------------------------------------------------------------------------
def upload_outputs(out_dir: Path, bucket: str, run_name: str) -> None:
    """Sync the output folder to `hf://buckets/<bucket>/<run_name>` using the token in HF_TOKEN (Jobs secret)."""
    from huggingface_hub import HfApi

    assert os.environ.get("HF_TOKEN"), "--out_bucket set but HF_TOKEN is not in the environment (pass --secrets HF_TOKEN)"
    dest = f"hf://buckets/{bucket}/{run_name}"
    HfApi().sync_bucket(source=str(out_dir), dest=dest)
    log(f"uploaded {out_dir} -> {dest}")


def write_summary(out_dir: Path, summary: dict) -> None:
    """Write summary.json and print it."""
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY_JSON " + json.dumps(summary), flush=True)


# ----------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    """Command line; defaults are upstream scripts/train_sft.sh plus the job's budget controls."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF model id or local directory (text part is loaded)")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--run_name", default=None, help="sub-folder name inside the bucket (default: time stamp)")
    p.add_argument("--cache_dir", default=os.path.expanduser("~/.cache/deltune"))
    p.add_argument("--upstream_dir", default=None, help="local deleting-weights checkout at the pinned commit")
    p.add_argument("--sdft_dir", default=None, help="local Self-Distillation checkout at the pinned commit")
    p.add_argument("--devices", default="auto", help="auto | cpu | mps | cuda:0,cuda:1 | mps,cpu ...")
    p.add_argument("--seed", type=int, default=0)
    # recipe (upstream scripts/train_sft.sh)
    p.add_argument("--train_steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--init_sparsity", type=float, default=0.00048828125)
    p.add_argument("--prior_strength", type=float, default=3.0)
    p.add_argument("--sharpness_start", type=float, default=100.0)
    p.add_argument("--sharpness_end", type=float, default=1000.0)
    p.add_argument("--weight_reg", type=float, default=1e-5)
    p.add_argument("--mask_path", choices=["step", "upstream"], default="step")
    # data / eval
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--wanda_samples", type=int, default=256)
    p.add_argument("--wanda_batch_size", type=int, default=4)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--eval_limit", type=int, default=0)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=1024, help="eval_tooluse.py default")
    p.add_argument("--untie_embeddings", action="store_true",
                   help="give a tied lm_head its own frozen copy so the layer split works (Qwen3.5-0.8B is tied; 27B is not)")
    p.add_argument("--thinking", action="store_true", help="leave Qwen3.5 thinking ON (default: OFF, empty <think>)")
    # checkpoints / budget
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--wallclock_budget_s", type=float, default=0.0, help="0 = unlimited; measured from script start")
    p.add_argument("--min_steps", type=int, default=100, help="exit 3 if fewer steps than this fit the budget")
    p.add_argument("--probe_steps", type=int, default=3)
    p.add_argument("--eval_reserve_s", type=float, default=900.0, help="reserve for both evals at the first budget check")
    p.add_argument("--save_reserve_s", type=float, default=180.0, help="reserve for final save + upload")
    p.add_argument("--save_cost_s", type=float, default=30.0, help="assumed cost of one periodic checkpoint")
    p.add_argument("--dump_ref_logits", action="store_true", help="small models: save masked-model logits for apply_mask check")
    p.add_argument("--out_bucket", default=os.environ.get("DELTUNE_OUT_BUCKET"), help="private HF bucket 'user/name' (OFF if unset)")
    return p.parse_args()


def main() -> int:
    """Run the whole job; returns the process exit code."""
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or time.strftime("run-%Y%m%d-%H%M%S")
    cache = Path(args.cache_dir)
    log(f"args: {json.dumps(vars(args))}")

    # -- upstream code -------------------------------------------------------------------------------------------
    if args.upstream_dir:
        up = Path(args.upstream_dir)
        check_git_head(up, UPSTREAM_SHA)
    else:
        up = fetch_pinned(UPSTREAM_REPO, UPSTREAM_SHA, cache)
    sdft = Path(args.sdft_dir) if args.sdft_dir else fetch_pinned(SDFT_REPO, SDFT_SHA, cache)
    if args.sdft_dir:
        check_git_head(sdft, SDFT_SHA)
    sys.path.insert(0, str(up / "src"))
    import sft_data  # upstream: encode, collate_train
    import wanda_calibration
    from learnable_sparsity import sparse_linear as sl_mod
    from learnable_sparsity.sparse_linear import SparseLinear, extract_all_masks, save_mask, set_weight_norm, step_all_sparse_modules
    scorer = load_scorer(sdft)

    # -- model ---------------------------------------------------------------------------------------------------
    import importlib.util
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    devices = resolve_devices(args.devices)
    d0 = devices[0]
    fla_on = importlib.util.find_spec("fla") is not None
    log(f"torch {torch.__version__} transformers {transformers.__version__} devices={devices} fla_installed={fla_on}")
    phase: dict[str, float] = {}
    t = time.monotonic()
    model_path = args.model
    if not os.path.isdir(model_path):
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"])
    phase["download_s"] = time.monotonic() - t
    t = time.monotonic()
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model, info = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, output_loading_info=True)
    assert not info["missing_keys"], f"missing keys when loading text model: {info['missing_keys'][:5]}"
    unexpected_bad = [k for k in info["unexpected_keys"] if not (k.startswith("mtp.") or "visual" in k)]
    assert not unexpected_bad, f"unexpected non-vision/non-mtp keys: {unexpected_bad[:5]}"
    log(f"loaded {type(model).__name__}; unexpected keys (vision/mtp): {len(info['unexpected_keys'])}")
    for p_ in model.parameters():
        p_.requires_grad_(False)
    if args.untie_embeddings and model.lm_head.weight is model.model.embed_tokens.weight:
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.detach().clone(), requires_grad=False)
        model.config.tie_word_embeddings = False
        log("untied lm_head from embed_tokens (frozen copy)")
    n_layers = len(model.model.layers)
    d_of = place_model(model, devices)
    model.config.use_cache = True
    sync(devices)
    phase["load_s"] = time.monotonic() - t
    log(f"{n_layers} layers placed: " + ", ".join(f"{d}:{d_of.count(i)}" for i, d in enumerate(devices)))

    train, test = build_data(args, tokenizer, sdft, sft_data)
    collate = lambda b: sft_data.collate_train(b, pad_token_id=tokenizer.pad_token_id)  # noqa: E731

    # -- convert + wanda -----------------------------------------------------------------------------------------
    reset_peak(devices)
    t = time.monotonic()
    n_conv = convert_text_gate_proj(model, SparseLinear, [args.sharpness_start, args.sharpness_end], args.train_steps)
    layers = sparse_layers(model, SparseLinear)
    assert len(layers) == n_conv == n_layers
    n_weights = sum(m.weight.numel() for m in layers)
    for m in layers:
        assert m.diff_mask.gate.device == m.weight.device
    set_weight_norm(model, args.mask_path == "upstream" and args.weight_reg > 0)
    log(f"converted {n_conv} gate_proj layers, {n_weights:,} logits (fp32 {n_weights * 4 / 1e9:.1f} GB)")
    calib = train[: args.wanda_samples]
    loader = torch.utils.data.DataLoader(calib, batch_size=args.wanda_batch_size, shuffle=False, collate_fn=collate)
    scores = wanda_calibration.compute_wanda_scores(model=model, dataloader=loader, device=d0, num_samples=len(calib),
                                                    target_modules=["gate_proj"])
    assert len(scores) == n_conv, f"wanda scored {len(scores)} layers, expected {n_conv}"
    wanda_calibration.init_masks_from_wanda(model=model, wanda_scores=scores, init_sparsity=args.init_sparsity,
                                            prior_strength=args.prior_strength)
    del scores, loader
    gc.collect()
    for d in devices:
        if d.type == "cuda":
            torch.cuda.empty_cache()
    sync(devices)
    phase["wanda_s"] = time.monotonic() - t
    mem = {"convert_wanda": peak_mem(devices)}
    init_deleted = sum(int((m.diff_mask.gate <= 0).sum()) for m in layers)
    log(f"wanda done in {phase['wanda_s']:.0f}s; initial hard-mask deleted {init_deleted:,} ({init_deleted / n_weights:.4%})")

    # -- optimizer -----------------------------------------------------------------------------------------------
    params = [m.diff_mask.gate for m in layers]
    all_cuda = all(p_.is_cuda for p_ in params)
    optim = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                              foreach=False, fused=True if all_cuda else None)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if all_cuda else (lambda: torch.autocast("cpu", enabled=False))

    signal.signal(signal.SIGTERM, lambda *_: (STOP.__setitem__("flag", True), log("SIGTERM received: will save and stop")))
    summary: dict = {"args": vars(args), "devices": [str(d) for d in devices], "n_train": len(train), "n_eval": len(test),
                     "n_masked_weights": n_weights, "fla_installed": fla_on, "torch": torch.__version__,
                     "transformers": transformers.__version__, "model": args.model, "exit_reason": None}

    def left() -> float:
        return args.wallclock_budget_s - (time.monotonic() - T0) if args.wallclock_budget_s > 0 else float("inf")

    metrics_f = open(out_dir / "metrics.jsonl", "a")

    def emit(rec: dict) -> None:
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()

    def early_exit(reason: str, code: int) -> int:
        summary.update(exit_reason=reason, phase_wall_s=phase, wallclock_s=time.monotonic() - T0, peak_mem_gb=mem)
        write_summary(out_dir, summary)
        log(f"EXIT {code}: {reason}")
        return code

    # -- training loop (probe, budget fit, base eval, rest) ------------------------------------------------------
    perm_rng = random.Random(args.seed + 1)
    order: list[int] = []

    def next_batch() -> dict:
        nonlocal order
        if len(order) < args.batch_size:
            order = list(range(len(train)))
            perm_rng.shuffle(order)
        idx, order = order[: args.batch_size], order[args.batch_size :]
        return collate([train[i] for i in idx])

    total = args.train_steps
    step_times: list[float] = []
    step = 0
    base_eval = None
    t_base_eval = 0.0
    stage = 0
    stop_reason = "completed"
    reset_peak(devices)
    t_train0 = time.monotonic()
    model.train()
    while step < total:
        if STOP["flag"]:
            stop_reason = "sigterm"
            break
        if stage == 2:
            reserve = 1.3 * t_base_eval + args.save_reserve_s
            if left() < reserve + step_times[-1]:
                stop_reason = "budget_guard"
                break
        ts = time.monotonic()
        batch = next_batch()
        ce, reg = train_step(model, batch, layers, sl_mod, args, d0, step, total, autocast)
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step_all_sparse_modules(model)
        step += 1
        sync(devices)
        step_times.append(time.monotonic() - ts)
        if step % args.log_every == 0 or step <= args.probe_steps:
            deleted = sum(int((m.diff_mask.gate <= 0).sum()) for m in layers)
            sharp = layers[0].diff_mask.sharpness[0] + (layers[0].diff_mask.sharpness[1] - layers[0].diff_mask.sharpness[0]) * layers[0].diff_mask.progress
            rec = {"step": step, "ce_loss": ce, "grad_norm": float(gnorm), "s_step": step_times[-1], "sharpness": sharp,
                   "deleted_frac": deleted / n_weights, "elapsed_s": time.monotonic() - T0, "total_steps": total}
            emit(rec)
            ema = sum(step_times[-10:]) / len(step_times[-10:])
            log(f"step {step}/{total} ce={ce:.4f} |g|={float(gnorm):.3g} {step_times[-1]:.2f}s/step deleted={deleted / n_weights:.4%} "
                f"ETA {(total - step) * ema / 60:.1f} min (budget left {left() / 60:.1f} min)")
        if step == args.probe_steps and stage == 0:
            s_step = sum(step_times[1:]) / max(len(step_times) - 1, 1) if len(step_times) > 1 else step_times[0]
            new_total = fit_steps(total, step, s_step, left(), args.eval_reserve_s + args.save_reserve_s, args.save_cost_s, args.save_steps)
            log(f"budget check 1: s/step={s_step:.2f}, left {left():.0f}s -> fits {new_total} of {total} steps")
            if new_total < args.min_steps:
                msg = f"only {new_total} steps fit the budget (< --min_steps {args.min_steps}); stopping before further spend"
                return early_exit("budget: " + msg, EXIT_BUDGET)
            total = new_total
            set_total_steps(layers, total)
            mem["probe"] = peak_mem(devices)
            stage = 1
            # base eval: SparseLinear.sparse=False applies the unmasked bf16 weight, i.e. exactly the base model
            for m in layers:
                m.sparse = False
            model.eval()
            t = time.monotonic()
            base_eval = evaluate_em(model, tokenizer, test, scorer, args, d0, "base")
            t_base_eval = time.monotonic() - t
            phase["base_eval_s"] = t_base_eval
            for m in layers:
                m.sparse = True
            log(f"BASE exact match {base_eval['correct']}/{base_eval['n']} = {base_eval['em']:.4f} in {t_base_eval:.0f}s")
            new_total = fit_steps(total, step, s_step, left(), 1.3 * t_base_eval + args.save_reserve_s, args.save_cost_s, args.save_steps)
            log(f"budget check 2 (measured eval {t_base_eval:.0f}s): fits {new_total} of {total} steps")
            if new_total < args.min_steps:
                summary.update(base_em=base_eval["em"], base_correct=base_eval["correct"])
                return early_exit(f"budget: only {new_total} steps fit after base eval (< --min_steps {args.min_steps})", EXIT_BUDGET)
            total = new_total
            set_total_steps(layers, total)
            stage = 2
            model.train()
            reset_peak(devices)
            continue
        if step % args.save_steps == 0 and step < total:
            ts = time.monotonic()
            save_mask(extract_all_masks(model), str(out_dir / f"mask_step_{step}"))
            model.train()
            log(f"checkpoint mask_step_{step} saved in {time.monotonic() - ts:.1f}s")
            if args.out_bucket:
                try:
                    upload_outputs(out_dir, args.out_bucket, run_name)
                except Exception as e:  # noqa: BLE001 - a failed periodic upload must not kill training; final upload is strict
                    log(f"WARNING periodic upload failed: {e!r}")
    sync(devices)
    phase["train_s"] = time.monotonic() - t_train0 - t_base_eval
    mem["train"] = peak_mem(devices)
    steady = step_times[1:] if len(step_times) > 1 else step_times
    s_per_step = sum(steady) / len(steady)
    if base_eval is None:  # SIGTERM or tiny run ended before the probe point
        stop_reason = stop_reason if stop_reason != "completed" else "ended_before_probe"

    # -- final mask ----------------------------------------------------------------------------------------------
    t = time.monotonic()
    mask_dir = out_dir / "mask"
    masks = extract_all_masks(model)
    save_mask(masks, str(mask_dir))
    stats = json.loads((mask_dir / "config.json").read_text())
    del masks
    phase["save_s"] = time.monotonic() - t
    log(f"final mask saved to {mask_dir}: sparsity {stats['sparsity_ratio']:.4%} ({stats['zero_params']:,} deleted), "
        f"{stats['compressed_size_mb']:.2f} MB compressed")
    summary.update(steps_done=step, steps_requested=args.train_steps, steps_planned_after_budget=total, stop_reason=stop_reason,
                   s_per_step=s_per_step, sparsity=stats["sparsity_ratio"], deleted=stats["zero_params"],
                   mask_mb=stats["compressed_size_mb"], init_deleted_frac=init_deleted / n_weights)
    if base_eval is not None:
        summary.update(base_em=base_eval["em"], base_correct=base_eval["correct"])
        (out_dir / "eval_base_responses.json").write_text(json.dumps(base_eval["responses"], indent=1))

    # -- mask eval ----------------------------------------------------------------------------------------------
    need = 1.3 * t_base_eval
    if STOP["flag"] or base_eval is None or left() < need:
        summary["mask_eval_skipped"] = f"stop={STOP['flag']}, left={left():.0f}s, need~{need:.0f}s"
        log("mask eval SKIPPED: " + summary["mask_eval_skipped"])
    else:
        model.eval()
        if args.mask_path == "step":
            sl_mod.build_step_weights(model)  # eval mode: constant hard-masked weights built once
        t = time.monotonic()
        mask_eval = evaluate_em(model, tokenizer, test, scorer, args, d0, "mask")
        phase["mask_eval_s"] = time.monotonic() - t
        if args.dump_ref_logits:
            enc = tokenizer([chat_prompt(tokenizer, test[0]["prompt"], args.thinking)], return_tensors="pt", add_special_tokens=False).to(d0)
            with torch.no_grad():
                lg = model(**enc).logits[0, -4:].float().cpu()
            torch.save({"prompt_ids": enc["input_ids"].cpu(), "logits": lg}, out_dir / "ref_logits.pt")
        if args.mask_path == "step":
            sl_mod.drop_step_weights(model)
        summary.update(mask_em=mask_eval["em"], mask_correct=mask_eval["correct"])
        (out_dir / "eval_mask_responses.json").write_text(json.dumps(mask_eval["responses"], indent=1))
        log(f"MASK exact match {mask_eval['correct']}/{mask_eval['n']} = {mask_eval['em']:.4f} in {phase['mask_eval_s']:.0f}s")
        mem["mask_eval"] = peak_mem(devices)

    summary.update(exit_reason="ok", phase_wall_s=phase, wallclock_s=time.monotonic() - T0, peak_mem_gb=mem)
    write_summary(out_dir, summary)
    metrics_f.close()
    if args.out_bucket:
        try:
            upload_outputs(out_dir, args.out_bucket, run_name)
        except Exception as e:  # noqa: BLE001
            log(f"FATAL final upload failed: {e!r}")
            return EXIT_UPLOAD
    return 0


if __name__ == "__main__":
    sys.exit(main())
