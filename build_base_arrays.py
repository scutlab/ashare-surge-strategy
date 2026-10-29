import pickle, glob, os, gc, numpy as np, pandas as pd
BASE="/home/hatch/workspace/ashare"; OUT=os.path.join(BASE,"daily_base_arrays"); os.makedirs(OUT, exist_ok=True)
FEATS=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
codes=set()
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d; gc.collect()
codes=sorted(codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
arr={s:{"X":[],"y":[]} for s in ["train","val"]}
for fi,f in enumerate(sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))):
    d=pickle.load(open(f,"rb")); sp=d["code"].map(cmap).to_numpy(np.int8); X=d[FEATS].to_numpy(np.float32); y=d["y_strategy"].to_numpy(np.int8); del d; gc.collect()
    for s,sid in [("train",0),("val",1)]:
        m=sp==sid
        if m.any(): arr[s]["X"].append(X[m]); arr[s]["y"].append(y[m])
    del X,y,sp; gc.collect(); print(f"batch {fi} train={sum(len(x) for x in arr['train']['X'])}", flush=True)
for s in arr:
    for k in arr[s]:
        a=np.concatenate(arr[s][k]); np.save(os.path.join(OUT,f"{s}_{k}.npy"), a); print(s,k,a.shape,a.dtype, flush=True)
print("BASE ARRAYS DONE", flush=True)
