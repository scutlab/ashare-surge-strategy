# 扫描买入折扣 x：entry=sig*(1-x)，+2%止盈/收盘卖出；目标：平均每笔成交收益>1%
import pandas as pd, numpy as np, glob, os, json, time
BASE=os.path.expanduser("~/workspace/ashare"); t0=time.time()
files=sorted(glob.glob(BASE+"/full_flow_new_*.pkl"))
codes=sorted({c for f in files for c in pd.read_pickle(f)["code"].unique()})
cmap={c:i%3 for i,c in enumerate(codes)}
xs=np.arange(0,10.001,0.25)/100.0
acc=np.zeros((len(xs),3))  # n_fill, sum_real, n_sig
for f in files:
    d=pd.read_pickle(f); d=d[d["code"].map(cmap).values==2]
    sig=d["signal_close"].values; bl=d["buy_low"].values; sh=d["sell_high"].values; sc=d["sell_close"].values
    n=len(d); del d
    for i,x in enumerate(xs):
        entry=sig*(1-x); filled=bl<=entry
        tp=sh>=entry*1.02; real=np.where(tp,0.02,sc/entry-1)
        acc[i,0]+=filled.sum(); acc[i,1]+=real[filled].sum(); acc[i,2]+=n
mean=acc[:,1]/np.maximum(acc[:,0],1)
out=[{"x":round(float(x*100),2),"fill_rate":float(acc[i,0]/acc[i,2]),"mean_per_fill":float(mean[i]),"n_fill":int(acc[i,0])} for i,x in enumerate(xs)]
json.dump(out,open(BASE+"/scan_x.json","w"),indent=2)
print(f"{'x%':>6} {'fill%':>7} {'mean/fill%':>10} {'n_fill':>10}")
for r in out[::2]: print(f"{r['x']:>6.2f} {r['fill_rate']*100:>6.2f}% {r['mean_per_fill']*100:>9.3f}% {r['n_fill']:>10}")
cross=[r for r in out if r["mean_per_fill"]>0.01]
print("first x with mean>1%:", cross[0] if cross else None)
print(f"elapsed {time.time()-t0:.0f}s",flush=True)
