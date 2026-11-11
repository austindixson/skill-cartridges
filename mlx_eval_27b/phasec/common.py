"""Shared pieces for the Phase C 27B eval: paths, pinned hashes, loading (weights from the 4-bit dir, tokenizer/template
from the Qwen tokenizer dir, in memory), result metadata, memory guard, JSONL helpers. Read-only wrt ../deltune."""
from __future__ import annotations

import hashlib, importlib.metadata as md, json, os, resource, subprocess, sys, time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import mlx.core as mx

HERE = Path(__file__).resolve().parents[1]  # phaseC-27b
ROOT = HERE.parent  # deltune
MODEL_DIR = ROOT / "models/Qwen3.8-27B-4bit"
TOK_DIR = ROOT / "models/Qwen3.8-27B-tokenizer"
MASK_DIR = ROOT / "masks/27b-toolalpaca-1/mask"
REF_DIR = ROOT / "masks/27b-toolalpaca-1"
RESULTS = HERE / "results"
MACHINE = "Apple M1 Max, 64 GB"
CACHE_LIMIT = 2 * 1024**3
MASK_SHA = {"model.safetensors": "b898709ad8466008950e001d13058d71797935059650431954d81f5d84950858",
            "config.json": "7c8eaa8ff6971e0485a1cabcadaaa4d0209b3dd90dc95b26027f53fb0b6a0b89"}
TOK_SHA_PREFIX = "0997f410"
QUANT = {"group_size": 64, "bits": 4, "mode": "affine"}
PREFIX = "language_model."  # mask key `model.layers.N...` -> mlx param `language_model.model.layers.N...`
FREE_MIN_PCT, SWAP_MAX_GROWTH_MB, PAUSE_S = 25, 5 * 1024, 30

mx.set_cache_limit(CACHE_LIMIT)  # every MLX process that imports phasec


def pt() -> str:
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d %I:%M:%S %p PT")


def log(*a) -> None:
    print(f"[{pt()}]", *a, flush=True)


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while c := f.read(1 << 22):
            h.update(c)
    return h.hexdigest()


def hf_revision(d: Path) -> str:
    t = sorted((d / ".cache/huggingface/trees").glob("*.json"))
    return t[0].stem if t else "unknown"


def meta(extra: dict | None = None) -> dict:
    tok_sha = sha256_file(TOK_DIR / "tokenizer.json")
    m = {"machine": MACHINE, "time": pt(),
         "tokenizer": {"source": "Qwen/Qwen3.8-27B (models/Qwen3.8-27B-tokenizer, not the mlx-community tokenizer.json: "
                                 "its pre_tokenizer is known-broken)", "revision": hf_revision(TOK_DIR),
                       "tokenizer_json_sha256": tok_sha},
         "model": {"dir": "models/Qwen3.8-27B-4bit", "repo": "mlx-community/Qwen3.8-27B-4bit", "revision": hf_revision(MODEL_DIR),
                   "quantization": QUANT},
         "mask": {"dir": "masks/27b-toolalpaca-1/mask", **{f"sha256_{k}": v for k, v in MASK_SHA.items()}},
         "versions": {p: md.version(p) for p in ["mlx", "mlx-lm", "transformers", "numpy", "safetensors"]},
         "mlx_cache_limit_bytes": CACHE_LIMIT}
    m.update(extra or {})
    return m


def write_result(name: str, body: dict, extra_meta: dict | None = None, root: Path = RESULTS) -> Path:
    """<root>/<name>.json with machine/tokenizer/model/version metadata."""
    p = Path(root) / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({**meta(extra_meta), **body}, indent=1, default=float))
    os.replace(tmp, p)
    return p


@contextmanager
def phase(name: str):
    """Resets the peak counter; yields a dict that receives peak_memory_bytes; clears the cache afterwards."""
    mx.clear_cache(); mx.reset_peak_memory()
    out = {"phase": name}
    t = time.perf_counter()
    try:
        yield out
    finally:
        out["peak_memory_bytes"] = mx.get_peak_memory()
        out["phase_seconds"] = time.perf_counter() - t
        mx.clear_cache()


# ------------------------------------------------------------------ jsonl
def read_jsonl(p) -> list[dict]:
    p = Path(p)
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(p, row: dict) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=float) + "\n")
        f.flush(); os.fsync(f.fileno())


# ------------------------------------------------------------------ memory guard
def free_pct() -> int:
    out = subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout
    line = [l for l in out.splitlines() if "free percentage" in l][-1]
    return int(line.split(":")[1].strip().rstrip("%"))


def swap_used_mb() -> float:
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(s.split("used =")[1].split("M")[0])


def mp_line() -> str:
    return subprocess.run(["memory_pressure"], capture_output=True, text=True).stdout.strip().splitlines()[-1]


def rss_bytes() -> dict:
    cur = int(subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True).stdout.strip()) * 1024
    return {"rss_current": cur, "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}  # bytes on macOS


class Guard:
    """check(): pause (30 s re-check) while free% < 25 or swap grew > 5 GB since process start; log every pause."""

    def __init__(self, free_fn=free_pct, swap_fn=swap_used_mb, sleep=time.sleep, log_path=RESULTS / "pauses.jsonl"):
        self.free_fn, self.swap_fn, self.sleep, self.log_path = free_fn, swap_fn, sleep, Path(log_path)
        self.swap0 = swap_fn()
        self.n_pauses = 0

    def ok(self) -> tuple[bool, int, float]:
        f, s = self.free_fn(), self.swap_fn()
        return (f >= FREE_MIN_PCT and s - self.swap0 <= SWAP_MAX_GROWTH_MB), f, s

    def check(self, where: str = "") -> None:
        good, f, s = self.ok()
        if good:
            return
        t0 = time.time()
        row = {"time": pt(), "where": where, "free_pct": f, "swap_mb": s, "swap_growth_mb": s - self.swap0, "action": "pause"}
        append_jsonl(self.log_path, row); log("PAUSE", row)
        while not good:
            self.sleep(PAUSE_S)
            good, f, s = self.ok()
        self.n_pauses += 1
        row = {"time": pt(), "where": where, "free_pct": f, "swap_mb": s, "action": "resume", "paused_s": time.time() - t0}
        append_jsonl(self.log_path, row); log("RESUME", row)


# ------------------------------------------------------------------ loading
def assert_mask_files() -> None:
    for fn, want in MASK_SHA.items():
        got = sha256_file(MASK_DIR / fn)
        assert got == want, f"mask {fn}: sha256 {got} != pinned {want}"


def load_tokenizer(tok_dir=TOK_DIR):
    from transformers import AutoTokenizer
    got = sha256_file(Path(tok_dir) / "tokenizer.json")
    assert got.startswith(TOK_SHA_PREFIX), f"tokenizer.json sha256 {got} does not start with {TOK_SHA_PREFIX}"
    tok = AutoTokenizer.from_pretrained(str(tok_dir))
    assert tok.eos_token == "<|im_end|>" and tok.eos_token_id == 248046, (tok.eos_token, tok.eos_token_id)
    return tok


def load_27b(model_dir=MODEL_DIR, tok_dir=TOK_DIR):
    """-> (model, tokenizer, config). Weights from the 4-bit dir via mlx-lm's load_model; tokenizer + chat template from
    the Qwen tokenizer dir (in memory; nothing is copied into either dir)."""
    from mlx_lm.utils import load_model
    assert_mask_files()
    tok = load_tokenizer(tok_dir)
    model, config = load_model(Path(model_dir))
    mx.eval(model.parameters())
    assert {k: config["quantization"].get(k) for k in QUANT} == QUANT, config["quantization"]
    extra = {k: v for k, v in config["quantization"].items() if k not in QUANT}
    assert not extra, f"per-layer quantization overrides present: {list(extra)[:5]}"
    assert model.language_model.args.num_hidden_layers == 64 and not model.language_model.args.tie_word_embeddings
    return model, tok, config
