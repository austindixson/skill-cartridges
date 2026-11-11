import os  # SC_ROOT: folder holding the mask and result directories (see README)
import csv, statistics as st
R=os.environ.get("SC_ROOT", ".") + "/item1-overlap/results/"
def rd(n): return list(csv.DictReader(open(R+n)))
def kind(p):
    return "gsm8k" if "/gsm8k" in p or "gsm8k" in p.split("/")[1] else ("mmlu" if "mmlu-pro" in p else "bbeh")
out=[]
for g in ("Llama-3.2-3B-Instruct","Llama-3.2-1B-Instruct"):
    rows=rd(f"pairs_{g}.csv"); cls={}
    for r in rows:
        a,b=sorted((kind(r["mask_A"]),kind(r["mask_B"])))
        k="gsm8k-vs-others" if "gsm8k" in (a,b) else f"{a}x{b}"
        cls.setdefault(k,[]).append(r)
    out.append(f"\n### {g}: {len(rows)} pairs\n\n| class | n | metric | min | median | max |\n|---|---|---|---|---|---|")
    for k,rs in sorted(cls.items()):
        for m in ("J","J0","J_over_J0","J_over_J0_layer","union_frac"):
            v=[float(r[m]) for r in rs]; out.append(f"| {k} | {len(rs)} | {m} | {min(v):.6g} | {st.median(v):.6g} | {max(v):.6g} |")
rows=rd("pairs_Llama-3.2-3B-Instruct.csv"); mm=[r for r in rows if kind(r["mask_A"])==kind(r["mask_B"])=="mmlu"]
names=sorted({r["mask_A"] for r in mm}|{r["mask_B"] for r in mm}); sh=lambda p:p.split("/")[-1]
M={(r["mask_A"],r["mask_B"]):float(r["J_over_J0"]) for r in mm}
out.append("\n### Llama-3.2-3B MMLU-Pro 14x14 J/J0 (symmetric; diagonal blank)\n\n| |"+"|".join(sh(n) for n in names)+"|\n|"+"---|"*(len(names)+1))
for a in names: out.append("|"+sh(a)+"|"+"|".join("" if a==b else f"{M.get((a,b),M.get((b,a))):.3f}" for b in names)+"|")
s=sorted(mm,key=lambda r:float(r["J_over_J0"]))
f=lambda r:f"{sh(r['mask_A'])} / {sh(r['mask_B'])}: J/J0={float(r['J_over_J0']):.4f}"
out.append("\nBottom 5: "+"; ".join(f(r) for r in s[:5])+"\n\nTop 5: "+"; ".join(f(r) for r in s[-5:][::-1]))
open(R+"summary_tables.md","w").write("\n".join(out)); print("\n".join(out))
