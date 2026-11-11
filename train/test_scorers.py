import sys, importlib.util, random, time
sys.argv=['x']
spec=importlib.util.spec_from_file_location('t','train_tasks.py'); t=importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
tr, te, he = t.load_code_data(0)
print("mbpp train", len(tr), "test", len(te), "he", len(he))
t0=time.time()
resp=["Here you go:\n```python\n"+t._norm_code(ex["code"])+"```" for ex in te]
r=t.score_items(te, resp); print("MBPP gold self-score", sum(x["correct"] for x in r), "/", len(r), f"{time.time()-t0:.0f}s")
bad=[x["id"] for x in r if not x["correct"]]; print("gold-fail ids", bad[:20])
resp=["```python\n"+ex["prompt"]+ex["canonical_solution"]+"```" for ex in he]
r=t.score_items(he, resp); print("HE full-func gold", sum(x["correct"] for x in r), "/", len(r))
resp=["```python\n"+ex["canonical_solution"]+"```" for ex in he]
r=t.score_items(he, resp); print("HE body-only gold", sum(x["correct"] for x in r), "/", len(r))
r=t.score_items(he, ["```python\ndef f(): pass\n```"]*len(he)); print("HE wrong", sum(x["correct"] for x in r))
r=t.score_items(te, ["no code"]*len(te)); print("MBPP wrong", sum(x["correct"] for x in r))
class A: pass
a=A(); a.seed=0; a.mem_keys=16; a.mem_updates=[4,20]; a.mem_query=4; a.mem_train_n=50; a.mem_test_n=200; a.mem_long_keys=32; a.mem_long_updates=[20,40]; a.mem_long_n=100
mtr, mte, mlong = t.load_memory_data(a)
r=t.score_items(mte, [ex["answer_text"] for ex in mte]); print("mem gold", sum(x["correct"] for x in r), len(r))
r=t.score_items(mte, ["\n".join(f"{k}: 1" for k in ex["keys"]) for ex in mte]); print("mem wrong", sum(x["correct"] for x in r))
# leakage check: test prompts not in train
a.mem_train_n=4000; mtr,_,_=t.load_memory_data(a)
print("mem train/test prompt overlap", len({e['prompt'] for e in mtr} & {e['prompt'] for e in mte}))
print("mbpp train/test task_id overlap", len({e['task_id'] for e in tr} & {e['task_id'] for e in te}))
print(mte[0]["prompt"][:300]); print("...", mte[0]["prompt"][-300:]); print(mte[0]["answer_text"])
import statistics; print("lines test", statistics.mean(e['n_lines'] for e in mte), max(e['n_lines'] for e in mte), "long", statistics.mean(e['n_lines'] for e in mlong), max(e['n_lines'] for e in mlong))
