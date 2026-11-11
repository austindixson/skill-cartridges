"""Score base vs masked mlx models on ToolAlpaca / Science with a live in-memory mask toggle.
Protocol replicates train/train_science.py (chat template enable_thinking=False, add_special_tokens=False,
greedy, SDFT scorer from the pinned repo). One model load, many conditions (apply -> eval -> eject).
Single condition:  --model D --task T [--masks D ...] [--compose union|intersection] --out F.json
Several:           --model D --task T --conditions '[{"tag":"base"},{"tag":"x","masks":[..],"compose":"union"}]' --outdir DIR"""
import argparse, ast, gc, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx_lm
from mlx_lm import load, stream_generate
from mlx_lm.generate import BatchGenerator, BatchStats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cartridge  # noqa: E402

SDFT = HERE.parent / "vendor" / "Self-Distillation"
PIN = "d77573212fa0a3ae2eeb64b9b44db1c251f75e3e"
TASKS = {"toolalpaca": dict(data="tooluse_data", n=97, gold="golden_answer", max_new=1024, script="eval_tooluse.py",
                            fns={"extract_actions", "extract_action_inputs", "evaluate_correctness"}),
         "science": dict(data="science_data", n=507, gold="answer", max_new=2048, script="eval_science.py",
                         fns={"extract_xml_answer", "evaluate_correctness"})}


def log(*a):
    print(*a, flush=True)


def load_scorer(task):
    """Same AST extraction as train/train_science.py: only the scoring FunctionDefs, no vllm import."""
    head = subprocess.run(["git", "-C", str(SDFT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == PIN, f"{SDFT} at {head}, expected {PIN}"
    t = TASKS[task]
    src = SDFT / t["script"]
    tree = ast.parse(src.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in t["fns"]]
    assert {n.name for n in keep} == t["fns"]
    imports = [n for n in tree.body if isinstance(n, ast.Import) and {a.name for a in n.names} <= {"re", "json"}]
    imports += [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "collections"]
    ns = {}
    exec(compile(ast.Module(body=imports + keep, type_ignores=[]), str(src), "exec"), ns)
    return ns["evaluate_correctness"]


def load_data(task, limit):
    from datasets import load_from_disk
    t = TASKS[task]
    ds = load_from_disk(str(SDFT / "data" / t["data"] / "eval_data")).to_list()
    assert len(ds) == t["n"], f"{task}: n={len(ds)}, expected {t['n']}"
    return ds[:limit] if limit else ds


def build_prompts(tokenizer, ds):
    texts = []
    for ex in ds:
        msgs = ex["prompt"] if isinstance(ex["prompt"], list) else [{"role": "user", "content": ex["prompt"]}]
        texts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
    ids = [tokenizer.encode(t, add_special_tokens=False) for t in texts]
    return texts, ids


def decode(tokenizer, toks):
    return tokenizer._tokenizer.decode(toks, skip_special_tokens=True)


def generate_batch(model, tokenizer, ids, order, max_new, bs, chunk=0):
    """Continuous batching, greedy, prompts inserted shortest-first. Returns token lists in dataset order.
    chunk>0: a fresh BatchGenerator per `chunk` prompts. A long-lived BatchGenerator on Qwen3.5 leaks live Metal
    buffers per finish/insert event (freed only when the generator is dropped) and hits the 499000 resource limit."""
    out = [None] * len(ids)
    st, fin, t0 = BatchStats(), 0, time.monotonic()
    steps, ntok, last_beat = 0, 0, t0
    for c0 in range(0, len(order), chunk or len(order)):
        sub = order[c0:c0 + (chunk or len(order))]
        gen = BatchGenerator(model, stop_tokens=[[t] for t in tokenizer.eos_token_ids], max_tokens=max_new,
                             completion_batch_size=bs, prefill_batch_size=min(bs, 8))
        uids = gen.insert([ids[i] for i in sub], [max_new] * len(sub))
        toks = {u: [] for u in uids}
        with gen.stats() as cst:
            while rs := gen.next_generated():
                steps += 1
                ntok += len(rs)
                if time.monotonic() - last_beat >= 30:
                    last_beat = time.monotonic()
                    log(f"  [alive {last_beat - t0:.0f}s] steps {steps} tokens {ntok} active {len(rs)} done {fin}/{len(order)} "
                        f"mem {mx.get_active_memory() / 1e9:.1f}GB")
                for r in rs:
                    if r.finish_reason is not None:
                        fin += 1
                        if fin % 8 == 0 or fin == len(order):
                            log(f"  gen {fin}/{len(order)} elapsed {time.monotonic() - t0:.0f}s")
                    if r.finish_reason != "stop":
                        toks[r.uid].append(r.token)
        st += cst
        for u, i in zip(uids, sub):
            out[i] = toks[u]
        gen.close()
        del gen, rs
        gc.collect()
        mx.clear_cache()
    return out, st


def generate_seq(model, tokenizer, ids, k, max_new):
    """Batch-1 generation of the first k prompts; returns token lists and decode tok/s (generation_tps)."""
    outs, gtok, gtime = [], 0, 0.0
    for i in range(k):
        toks, last = [], None
        for r in stream_generate(model, tokenizer, ids[i], max_tokens=max_new):
            last = r
            toks.append(r.token)
        if last.finish_reason == "stop" and toks and toks[-1] in tokenizer.eos_token_ids:
            toks = toks[:-1]
        outs.append(toks)
        gtok += last.generation_tokens
        gtime += last.generation_tokens / last.generation_tps if last.generation_tps else 0.0
        log(f"  seq {i + 1}/{k} gen_tps {last.generation_tps:.1f}")
    return outs, (gtok / gtime if gtime else None)


# ---- exact before/after weight check -------------------------------------------------------------------------
def _raw(a):
    return a if a.dtype == mx.uint32 else a.view(mx.uint16)


def snapshot(model):
    """Independent host copies of all gate_proj arrays (weight/scales/biases)."""
    snap = {}
    for l, m in cartridge.gate_proj_modules(model).items():
        for n in ("weight", "scales", "biases"):
            if hasattr(m, n):
                snap[(l, n)] = np.array(_raw(getattr(m, n)))
    return snap


def assert_identical(model, snap):
    mods = cartridge.gate_proj_modules(model)
    for (l, n), ref in snap.items():
        assert mx.array_equal(mx.array(ref), _raw(getattr(mods[l], n))).item(), f"layer {l} {n} differs from original"


def make_keep(masks, compose):
    ms = [cartridge.load_mask(d) for d in masks]
    if len(ms) > 1:
        assert compose in ("union", "intersection"), "multiple masks need --compose"
        return cartridge.combine(ms, compose)
    return ms[0]


def apply_mask(model, keep, form):
    if form == "none":
        return cartridge.apply_dense(model, keep), cartridge.eject_dense
    import quant_cartridge as qc
    return {"nearest_zero": qc.apply_nearest_zero, "requant": qc.apply_requant}[form](model, keep), qc.eject


def run_condition(model, tokenizer, args, cond, ds, ids, order, score, snap):
    tag, masks = cond["tag"], cond.get("masks") or []
    compose, form = cond.get("compose", args.compose), cond.get("quant_form", args.quant_form)
    t = TASKS[args.task]
    max_new = args.max_new_tokens or t["max_new"]
    log(f"== condition {tag}: masks={masks} compose={compose} quant_form={form}")
    deleted, apply_s, saved, ejecter = 0, 0.0, None, None
    if masks:
        keep = make_keep(masks, compose)
        deleted = cartridge.deleted_count(keep)
        t0 = time.monotonic()
        saved, ejecter = apply_mask(model, keep, form)
        mx.eval(model.parameters())
        apply_s = time.monotonic() - t0
        log(f"  applied {deleted} deletions in {apply_s:.2f}s")
    mx.reset_peak_memory()
    t0 = time.monotonic()
    toks, st = generate_batch(model, tokenizer, ids, order, max_new, args.batch_size, args.chunk)
    wall = time.monotonic() - t0
    responses = [decode(tokenizer, x) for x in toks]
    golds = [ex[t["gold"]] for ex in ds]
    scores = [bool(s) for s in score(responses, golds)]
    res = dict(model=args.model, task=args.task, tag=tag, masks=masks, compose=compose if len(masks) > 1 else None,
               quant_form=form, deleted_applied=deleted, n=len(ds), correct=sum(scores), batch_size=args.batch_size, chunk=args.chunk,
               max_new_tokens=max_new,
               items=[dict(idx=i, response=r, gold=g, correct=s) for i, (r, g, s) in enumerate(zip(responses, golds, scores))],
               wall_seconds=wall, generated_tokens=sum(len(x) for x in toks),
               gen_tok_per_s=st.generation_tps, gen_tok_per_s_wall=sum(len(x) for x in toks) / wall,
               prompt_tok_per_s=st.prompt_tps, peak_memory_gb=mx.get_peak_memory() / 1e9, apply_seconds=apply_s)
    if args.decode_bench:
        k = min(args.decode_bench, len(ds))
        stoks, tps = generate_seq(model, tokenizer, ids, k, max_new)
        res["decode_bench"] = dict(k=k, batch1_decode_tok_per_s=tps,
                                   seq_identical_to_batch=sum(a == b for a, b in zip(stoks, toks[:k])))
        log(f"  decode bench: {tps} tok/s; identical to batch {res['decode_bench']['seq_identical_to_batch']}/{k}")
    eject_s = 0.0
    if masks:
        t0 = time.monotonic()
        ejecter(model, saved)
        mx.eval(model.parameters())
        eject_s = time.monotonic() - t0
        assert_identical(model, snap)
        log(f"  ejected in {eject_s:.2f}s; weights bit-identical to originals")
    res.update(eject_seconds=eject_s, mlx=mx.__version__ if hasattr(mx, "__version__") else None,
               mlx_lm=mlx_lm.__version__, timestamp=datetime.now().astimezone().isoformat())
    log(f"  [{tag}] {res['correct']}/{res['n']} correct, {res['gen_tok_per_s']:.1f} gen tok/s, wall {wall:.0f}s")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--masks", nargs="*", default=[])
    ap.add_argument("--compose", choices=["union", "intersection"])
    ap.add_argument("--quant-form", default="none", choices=["none", "nearest_zero", "requant"])
    ap.add_argument("--conditions", help="JSON list of {tag, masks?, compose?, quant_form?}")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=0, help="fresh BatchGenerator per N prompts (0 = one for all)")
    ap.add_argument("--max-new-tokens", type=int, default=0)
    ap.add_argument("--decode-bench", type=int, default=0)
    ap.add_argument("--out"); ap.add_argument("--outdir"); ap.add_argument("--tag", default="run")
    args = ap.parse_args()
    conds = json.loads(args.conditions) if args.conditions else [dict(tag=args.tag, masks=args.masks)]
    outdir = Path(args.outdir or (os.path.dirname(args.out) if args.out else "."))
    outdir.mkdir(parents=True, exist_ok=True)

    score, ds = load_scorer(args.task), load_data(args.task, args.limit)
    log(f"loading {args.model}")
    t0 = time.monotonic()
    model, tokenizer = load(args.model)
    mx.eval(model.parameters())
    log(f"loaded in {time.monotonic() - t0:.0f}s; eos={tokenizer.eos_token_ids}")
    texts, ids = build_prompts(tokenizer, ds)
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    snap = snapshot(model)
    for cond in conds:
        path = Path(args.out) if (args.out and not args.conditions) else outdir / f"{args.task}__{cond['tag']}.json"
        if args.conditions and path.exists():
            try:
                done = json.loads(path.read_text()).get("n") == len(ds)
            except Exception:
                done = False
            if done:
                log(f"== condition {cond['tag']}: SKIPPED, complete result exists at {path}")
                continue
        res = run_condition(model, tokenizer, args, cond, ds, ids, order, score, snap)
        path.write_text(json.dumps(res, indent=1))
        log(f"  wrote {path}")


if __name__ == "__main__":
    main()
