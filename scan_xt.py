# 二维扫描：买入折扣x × 止盈t × 止损s（None/0.02/0.03/0.05），目标：平均每笔成交收益最大且>0
# 先在测试集(i%3==2)粗扫
import pandas as pd, numpy as np, glob, os, json, time, itertools
BASE=os.path.expanduser("~/workspace/ashare"); t0=time.time()
files=sorted(glob.glob(BASE+"/full_flow_new_*.pkl"))
codes=sorted({c for f in files for c in pd.read_pickle(f)["code"].unique()})
cmap={c:i%3 for i,c in enumerate(codes)}
XS=[0,.01,.02,.03,.04,.06,.08,.10]; TS=[.02,.03,.05,.08,.10,.15,.20]; SS=[None,.02,.03,.05]
batches=[]
for f in files:
    d=pd.read_pickle(f); d=d[d["code"].map(cmap).values==2]
    batches.append((d["signal_close"].values,d["buy_low"].values,d["sell_low"].values,d["sell_high"].values,d["sell_close"].values,len(d)))
    del d
N=sum(b[5] for b in batches); print(f"test rows {N}",flush=True)
res=[]
for x,t,s in itertools.product(XS,TS,SS):
    nf=sr=ntp=nsl=npr=0
    for sig,bl,sl,sh,sc,n in batches:
        entry=sig*(1-x); filled=bl<=entry
        if s is None:
            tp=sh>=entry*(1+t); real=np.where(tp,t,sct:=sc/entry-1)
        else:
            slh=sl<=entry*(1-s); tph=~slh&(sh>=entry*(1+t)); real=np.where(slh,-s,np.where(tph,t,sc/entry-1))
        f=filled; nf+=f.sum(); r=real[f]; sr+=r.sum(); ntp+=(f&(sh>=entry*(1+t))).sum()
        if s is not None: nsl+=(f&(sl<=entry*(1-s))).sum()
        npr+=(r>0).sum()
    mean=sr/max(nf,1)
    res.append({"x":x,"t":t,"s":s,"fill_rate":nf/N,"mean":mean,"n_fill":int(nf),
                "p_tp":ntp/max(nf,1),"p_sl":nsl/max(nf,1),"p_prof":npr/max(nf,1)})
json.dump(res,open(BASE+"/scan_xt_coarse.json","w"))
res_sorted=sorted(res,key=lambda r:r["mean"],reverse=True)
print(f"{'x':>5} {'t':>5} {'s':>6} {'fill%':>7} {'mean%':>8} {'p_tp':>6} {'p_sl':>6} {'p_prof':>6}")
for r in res_sorted[:15]:
    print(f"{r['x']*100:>4.0f}% {r['t']*100:>4.0f}% {str(None if r['s'] is None else f'{r['s']*100:.0f}%'):>6} {r['fill_rate']*100:>6.2f}% {r['mean']*100:>7.3f}% {r['p_tp']:>6.3f} {r['p_sl']:>6.3f} {r['p_prof']:>6.3f}")
print(f"elapsed {time.time()-t0:.0f}s",flush=True)
