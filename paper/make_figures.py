# All numbers below are measured values (2026-11-10 and 2026-11-11). No synthetic data.
import json, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({'font.size':9,'figure.dpi':200,'savefig.bbox':'tight'})
B='#9aa5b1'; C='#1f6feb'; R='#d1495b'
# Fig 1: ToolAlpaca (97 items) base vs cartridge
labels=['LFM3-1.7B\n(bf16)','Qwen2.5-7B\n(published)','Qwen3.5-9B\n(bf16, 3 seeds)','Qwen3.5-9B\n(4-bit, 3 seeds)','Qwen3.8-27B\n(bf16)','Qwen3.8-27B\n(4-bit)']
base=[11,41,42,38,42,41]; mask=[59,68,69.33,64.33,65,67]
fig,ax=plt.subplots(figsize=(6.8,2.8)); x=np.arange(len(labels)); w=0.38
ax.bar(x-w/2,base,w,color=B,label='base'); ax.bar(x+w/2,mask,w,color=C,label='with cartridge')
ax.errorbar([2+w/2],[69.33],yerr=[1.53],fmt='none',ecolor='k',capsize=3,lw=1)
ax.errorbar([3+w/2],[64.33],yerr=[2.89],fmt='none',ecolor='k',capsize=3,lw=1)
ax.scatter([2+w/2]*3,[69,71,68],s=8,color='k',zorder=3)
ax.scatter([3+w/2]*3,[66,66,61],s=8,color='k',zorder=3)
for i,(b,m) in enumerate(zip(base,mask)):
    ax.text(i-w/2,b+1,f'{b}',ha='center',fontsize=7)
    if i==2:
        ax.text(i+w/2,73.5,'69.3±1.5',ha='center',fontsize=7)
    elif i==3:
        ax.text(i+w/2,70.0,'64.3±2.9',ha='center',fontsize=7)
    else:
        ax.text(i+w/2,m+1.8,f'{m:g}',ha='center',fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=6.5); ax.set_ylabel('ToolAlpaca correct (of 97)'); ax.set_ylim(0,88)
ax.axhline(65,ls=':',c=R,lw=1); ax.text(5.45,80,'dotted line: 27B + mask (bf16) = 65',fontsize=6.5,color=R,ha='right')
ax.legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,0.9)); fig.savefig('figures/fig1_toolalpaca.pdf'); fig.savefig('figures/fig1_toolalpaca.png')
# Fig 2: overlap observed vs chance
ov=json.load(open('data/overlap_9b_toolalpaca.json'))
names=['9B TA s1∩s2','9B TA s1∩s3','9B TA s2∩s3','9B TA s1∩s2∩s3','9B Mem s1∩s2','7B TA∩Sci\n(published)']
obs=[ov['pairs']['s1-s2']['inter'],ov['pairs']['s1-s3']['inter'],ov['pairs']['s2-s3']['inter'],ov['triple'],797965,1140547]
exp=[ov['pairs']['s1-s2']['exp_inter'],ov['pairs']['s1-s3']['exp_inter'],ov['pairs']['s2-s3']['exp_inter'],ov['triple_exp'],14882,263270.5]
fig,ax=plt.subplots(figsize=(6.4,2.6)); x=np.arange(len(names))
ax.bar(x-w/2,exp,w,color=B,label='expected by chance'); ax.bar(x+w/2,obs,w,color=C,label='observed')
ax.set_yscale('log'); ax.set_xticks(x); ax.set_xticklabels(names,fontsize=7); ax.set_ylabel('shared deleted weights')
for i,(o,e) in enumerate(zip(obs,exp)): ax.text(i+w/2,o*1.25,f'{o/e:.1f}×' if o/e<100 else f'{o/e:,.0f}×',ha='center',fontsize=7)
ax.set_ylim(100,1e7); ax.legend(frameon=False,fontsize=7,loc='upper right'); fig.savefig('figures/fig2_overlap.pdf'); fig.savefig('figures/fig2_overlap.png')
# Fig 3: stacking on Qwen2.5-7B (published masks) and Qwen3.5-9B (ours)
fig,axs=plt.subplots(2,2,figsize=(6.4,4.6))
cats=['base','tool','science','union','inter-\nsection']
cols={'inter-\nsection':'#e9c46a','base':B,'tool':C,'science':'#2a9d8f','union':'#6a4c93'}
panels=[
    (axs[0,0],[41,68,49,64,44],'7B ToolAlpaca (of 97)'),
    (axs[0,1],[155,160,344,340,166],'7B Science (of 507)'),
    (axs[1,0],[45,69,49,66,53],'9B ToolAlpaca (of 97)'),
    (axs[1,1],[120,204,316,298,257],'9B Science (of 507)'),
]
for a,vs,title in panels:
    a.bar(cats,vs,color=[cols[c] for c in cats])
    a.set_title(title,fontsize=8)
    for i,v in enumerate(vs):
        a.text(i,v+max(vs)*0.02,str(v),ha='center',fontsize=7)
    a.tick_params(axis='x',labelsize=7)
    a.set_ylim(0,max(vs)*1.18)
fig.tight_layout()
fig.savefig('figures/fig3_stacking.pdf'); fig.savefig('figures/fig3_stacking.png')
# Fig 4: per-layer deletion fraction, 9B seeds
fig,ax=plt.subplots(figsize=(6.4,2.2))
for i,f in enumerate(ov['per_layer_frac']): ax.plot(np.arange(len(f)),np.array(f)*100,marker='o',ms=2.5,lw=1,label=f'seed {[1,2,3][i]}')
ax.set_xlabel('decoder layer'); ax.set_ylabel('% of gate_proj deleted'); ax.legend(frameon=False,fontsize=7)
fig.savefig('figures/fig4_per_layer.pdf'); fig.savefig('figures/fig4_per_layer.png')
# Fig 5: other cartridges paired outcomes
fig,ax=plt.subplots(figsize=(7.2,2.6))
tasks=['Memory recall\n(200)','Memory 2x-len\n(100)','Mem seed2\nrecall (200)','Mem seed2\n2x-len (100)','LFM3 ToolAlpaca\n(97)','Code MBPP\n(500)','Code HumanEval\n(164)']
b=[89,8,90,8,11,349,139]; m=[200,100,200,99,59,293,114]; n=[200,100,200,100,97,500,164]
x=np.arange(len(tasks)); ax.bar(x-w/2,np.array(b)/np.array(n)*100,w,color=B,label='base'); ax.bar(x+w/2,np.array(m)/np.array(n)*100,w,color=[C]*5+[R]*2,label='with cartridge')
for i in range(len(tasks)): ax.text(i+w/2,m[i]/n[i]*100+2,f'{m[i]}',ha='center',fontsize=6.5); ax.text(i-w/2,b[i]/n[i]*100+2,f'{b[i]}',ha='center',fontsize=6.5)
ax.set_xticks(x); ax.set_xticklabels(tasks,fontsize=6); ax.set_ylabel('% correct'); ax.set_ylim(0,125); ax.legend(frameon=False,fontsize=7,loc='upper left',ncol=2)
fig.savefig('figures/fig5_other_cartridges.pdf'); fig.savefig('figures/fig5_other_cartridges.png')
print('ok')
