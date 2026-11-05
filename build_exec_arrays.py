# 执行规则(x=9%,t=30%,s=1.5%)标签数组：y=1 当且仅当 T日按sig*0.91成交 且 T+1日收益>0
# 卖出：先止损(sell_low<=entry*0.985 -> -1.5%)，再止盈(sell_high>=entry*1.30 -> +30%)，否则收盘
import pandas as pd, numpy as np, glob, os, json, time, gc, importlib.util, sys
X=float(sys.argv[1]) if len(sys.argv)>1 else 0.09
T=float(sys.argv[2]) if len(sys.argv)>2 else 0.30
S=float(sys.argv[3]) if len(sys.argv)>3 else 0.015
BASE=os.path.expanduser("~/workspace/ashare"); t0=time.time()
OUT=BASE+f"/exec_arrays_{int(X*100)}_{int(T*100)}_{S}"; os.makedirs(OUT,exist_ok=True)
spec=importlib.util.spec_from_file_location("ft",BASE+"/full_train.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS=[f for f in m.FEATS if f!="weekday"]
flow_files=sorted(glob.glob(BASE+"/full_flow_new_*.pkl")); v3_files=sorted(glob.glob(BASE+"/full_features_batch_v3_*.pkl"))
codes=sorted({c for f in flow_files for c in pd.read_pickle(f)["code"].unique()})
cid={c:i for i,c in enumerate(codes)}; NC=len(codes)
batch_of=np.empty(NC,dtype=np.int32); flow_small={}
for bi,f in enumerate(flow_files):
    d=pd.read_pickle(f)
    ci=d["code"].map(cid).values.astype(np.int32); batch_of[ci]=bi
    sig=d["signal_close"].values; entry=sig*(1-X)
    filled=d["buy_low"].values<=entry
    slh=d["sell_low"].values<=entry*(1-S); tph=(~slh)&(d["sell_high"].values>=entry*(1+T))
    real=np.where(slh,-S,np.where(tph,T,d["sell_close"].values/entry-1))
    y=(filled&(real>0)).astype(np.int8)
    dk=d["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    flow_small[bi]=pd.DataFrame({"ci":ci,"dk":dk,"y":y,"sig":sig.astype(np.float32),
        "bl":d["buy_low"].values.astype(np.float32),"sl":d["sell_low"].values.astype(np.float32),
        "sh":d["sell_high"].values.astype(np.float32),"sc":d["sell_close"].values.astype(np.float32)})
    del d; gc.collect()
print(f"x={X} t={T} s={S} codes {NC}",flush=True)
cmap=np.arange(NC,dtype=np.int32)%3
acc={"tr":[],"va":[],"te":[]}; accy={"tr":[],"va":[],"te":[]}; te_meta=[]; tot=0; match=0
for vf in v3_files:
    v=pd.read_pickle(vf)
    ci=v["code"].map(cid).values.astype(np.int32)
    dk=v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    b=batch_of[ci]; Xf=v[FEATS].values.astype(np.float32); tot+=len(v); del v; gc.collect()
    for bi in np.unique(b):
        sel=b==bi
        left=pd.DataFrame({"ci":ci[sel],"dk":dk[sel],"xi":np.where(sel)[0]})
        mg=left.merge(flow_small[bi],on=["ci","dk"],how="inner"); match+=len(mg)
        if len(mg)==0: continue
        Xa=Xf[mg["xi"].values]; yv=mg["y"].values; sp=cmap[mg["ci"].values]
        for s_,c in (("tr",0),("va",1),("te",2)):
            mm=sp==c
            if mm.sum(): acc[s_].append(Xa[mm]); accy[s_].append(yv[mm])
        tm=sp==2
        if tm.sum(): te_meta.append(mg[["dk","sig","bl","sl","sh","sc"]].values[tm].astype(np.float32))
        del left,mg,Xa,yv
    del ci,dk,b,Xf; gc.collect()
print(f"coverage {match}/{tot}={match/tot:.4f}",flush=True); assert match/tot>0.99
meta={"x":X,"t":T,"s":S}
for s_ in ["tr","va","te"]:
    Xa=np.concatenate(acc[s_]); yv=np.concatenate(accy[s_])
    np.save(f"{OUT}/{s_}_X.npy",Xa); np.save(f"{OUT}/{s_}_y.npy",yv)
    meta[s_]={"n":int(len(yv)),"pos_rate":float(yv.mean())}; print(s_,Xa.shape,meta[s_]["pos_rate"],flush=True)
te_meta=np.concatenate(te_meta); np.save(f"{OUT}/te_exec.npy",te_meta)
meta["te_exec_n"]=int(len(te_meta))
json.dump(meta,open(f"{OUT}/meta.json","w"),indent=2); print(f"DONE {time.time()-t0:.0f}s",flush=True)
