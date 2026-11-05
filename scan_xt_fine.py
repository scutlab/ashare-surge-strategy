# 细扫 + 验证集确认
import pandas as pd, numpy as np, glob, os, json, time, itertools
BASE=os.path.expanduser("~/workspace/ashare"); t0=time.time()
files=sorted(glob.glob(BASE+"/full_flow_new_*.pkl"))
codes=sorted({c for f in files for c in pd.read_pickle(f)["code"].unique()})
cmap={c:i%3 for i,c in enumerate(codes)}
def load(split):
    out=[]
    for f in files:
        d=pd.read_pickle(f); d=d[d["code"].map(cmap).values==split]
        out.append((d["signal_close"].values,d["buy_low"].values,d["sell_low"].values,d["sell_high"].values,d["sell_close"].values,len(d)))
        del d
    return out
te=load(2); N=sum(b[5] for b in te)
def ev(batches,x,t,s):
    nf=sr=ntp=nsl=npr=0; Nl=sum(b[5] for b in batches)
    for sig,bl,sl,sh,sc,n in batches:
        entry=sig*(1-x); filled=bl<=entry
        slh=sl<=entry*(1-s); tph=~slh&(sh>=entry*(1+t)); real=np.where(slh,-s,np.where(tph,t,sc/entry-1))
        f=filled; nf+=f.sum(); r=real[f]; sr+=r.sum(); ntp+=(f&tph).sum(); nsl+=(f&slh).sum(); npr+=(r>0).sum()
    return {"x":x,"t":t,"s":s,"fill_rate":nf/Nl,"mean":sr/max(nf,1),"n_fill":int(nf),
            "p_tp":ntp/max(nf,1),"p_sl":nsl/max(nf,1),"p_prof":npr/max(nf,1)}
XS=[.06,.07,.08,.09,.10]; TS=[.12,.15,.18,.20,.25,.30]; SS=[.015,.02,.025,.03]
res=[ev(te,x,t,s) for x,t,s in itertools.product(XS,TS,SS)]
json.dump(res,open(BASE+"/scan_xt_fine.json","w"))
top=sorted(res,key=lambda r:r["mean"],reverse=True)[:8]
print(f"{'x':>5} {'t':>5} {'s':>5} {'fill%':>7} {'mean%':>8} {'p_tp':>6} {'p_sl':>6} {'p_prof':>6}   [test]")
for r in top: print(f"{r['x']*100:>4.0f}% {r['t']*100:>4.0f}% {r['s']*100:>4.1f}% {r['fill_rate']*100:>6.2f}% {r['mean']*100:>7.3f}% {r['p_tp']:>6.3f} {r['p_sl']:>6.3f} {r['p_prof']:>6.3f}")
va=load(1)
print("--- validation confirm ---")
for r in top[:5]:
    v=ev(va,r["x"],r["t"],r["s"])
    print(f"x={r['x']*100:.0f}% t={r['t']*100:.0f}% s={r['s']*100:.1f}% -> val mean {v['mean']*100:+.3f}% fill {v['fill_rate']*100:.2f}% p_prof {v['p_prof']:.3f}")
print(f"elapsed {time.time()-t0:.0f}s",flush=True)
