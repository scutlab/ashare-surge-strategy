# 新规则标签数组构建：y=1 当且仅当 T日按(T-1收盘*0.98)成交 且 T+1日收益>0（触及+2%按+2%，否则收盘）
# flow按code分区(25批)，v3特征按old/new分区(25批) -> 按(code_id,dk)归并
import pandas as pd, numpy as np, glob, os, json, time, gc, importlib.util
BASE=os.path.expanduser("~/workspace/ashare"); t0=time.time(); OUT=BASE+"/newlabel_arrays"; os.makedirs(OUT,exist_ok=True)
spec=importlib.util.spec_from_file_location("ft",BASE+"/full_train.py"); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS=[f for f in m.FEATS if f!="weekday"]; print("n feats",len(FEATS),flush=True)
flow_files=sorted(glob.glob(BASE+"/full_flow_new_*.pkl")); v3_files=sorted(glob.glob(BASE+"/full_features_batch_v3_*.pkl"))
assert len(flow_files)==len(v3_files)==25
codes=sorted({c for f in flow_files for c in pd.read_pickle(f)["code"].unique()})
cid={c:i for i,c in enumerate(codes)}; NC=len(codes)
batch_of=np.empty(NC,dtype=np.int32)
flow_small={}
for bi,f in enumerate(flow_files):
    d=pd.read_pickle(f)
    ci=d["code"].map(cid).values.astype(np.int32); batch_of[ci]=bi
    sig=d["signal_close"].values; entry=sig*0.98
    filled=d["buy_low"].values<=entry; tp=d["sell_high"].values>=entry*1.02
    y=(filled&((tp)|(d["sell_close"].values>entry))).astype(np.int8)
    dk=d["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    flow_small[bi]=pd.DataFrame({"ci":ci,"dk":dk,"y":y,"sig":sig.astype(np.float32),
        "bl":d["buy_low"].values.astype(np.float32),"sh":d["sell_high"].values.astype(np.float32),
        "sc":d["sell_close"].values.astype(np.float32)})
    del d; gc.collect()
print(f"codes {NC} pass1 {time.time()-t0:.0f}s",flush=True)
cmap=np.arange(NC,dtype=np.int32)%3
acc={"tr":[],"va":[],"te":[]}; accy={"tr":[],"va":[],"te":[]}; te_meta=[]; tot=0; match=0
for vf in v3_files:
    v=pd.read_pickle(vf)
    assert all(c in v.columns for c in FEATS), "feat missing"
    ci=v["code"].map(cid).values.astype(np.int32)
    dk=v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    b=batch_of[ci]; Xf=v[FEATS].values.astype(np.float32); tot+=len(v); del v; gc.collect()
    for bi in np.unique(b):
        sel=b==bi; fs=flow_small[bi]
        left=pd.DataFrame({"ci":ci[sel],"dk":dk[sel],"xi":np.where(sel)[0]})
        mg=left.merge(fs,on=["ci","dk"],how="inner"); match+=len(mg)
        if len(mg)==0: continue
        X=Xf[mg["xi"].values]; yv=mg["y"].values; sp=cmap[mg["ci"].values]
        for s,c in (("tr",0),("va",1),("te",2)):
            mm=sp==c
            if mm.sum(): acc[s].append(X[mm]); accy[s].append(yv[mm])
        tm=sp==2
        if tm.sum(): te_meta.append(mg[["dk","sig","bl","sh","sc"]].values[tm].astype(np.float32))
        del left,mg,X,yv
    del ci,dk,b,Xf; gc.collect()
    print(f"{os.path.basename(vf)} cum_match {match}/{tot} {time.time()-t0:.0f}s",flush=True)
print(f"coverage {match}/{tot}={match/tot:.4f}",flush=True)
assert match/tot>0.99, "merge coverage too low"
meta={}
for s in ["tr","va","te"]:
    X=np.concatenate(acc[s]); y=np.concatenate(accy[s])
    np.save(f"{OUT}/{s}_X.npy",X); np.save(f"{OUT}/{s}_y.npy",y)
    meta[s]={"n":int(len(y)),"pos_rate":float(y.mean())}
    print(s,X.shape,"pos",meta[s]["pos_rate"],flush=True); del X,y
te_meta=np.concatenate(te_meta)
np.save(f"{OUT}/te_exec.npy",te_meta)  # dk,sig,bl,sh,sc
meta["te_exec_n"]=int(len(te_meta))
json.dump(meta,open(f"{OUT}/meta.json","w"),indent=2)
print(json.dumps(meta,indent=2)); print(f"DONE {time.time()-t0:.0f}s",flush=True)
