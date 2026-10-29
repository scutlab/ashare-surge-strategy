# 高效建v4数组：每批只转一次numpy，按split掩码切片，不建分集DataFrame；存npy供训练复用
import pickle, glob, os, gc, numpy as np, pandas as pd
BASE="/home/hatch/workspace/ashare"; OUT=os.path.join(BASE,"v4_arrays"); os.makedirs(OUT, exist_ok=True)
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps","days_since_disclosure"]
FEATS=ORIG+FUND
codes=set()
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_v4/full_flow_v4_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d; gc.collect()
codes=sorted(codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
arr={s:{"X":[],"y":[],"date":[],"bought":[],"ret":[]} for s in ["train","val","test"]}
for fi, f in enumerate(sorted(glob.glob(os.path.join(BASE,"full_flow_v4/full_flow_v4_*.pkl")))):
    d=pickle.load(open(f,"rb"))
    split=d["code"].map(cmap).to_numpy(np.int8)
    X=d[FEATS].to_numpy(np.float32); y=d["y_strategy"].to_numpy(np.int8)
    date=d["date"].values.astype("datetime64[D]").astype(int); bought=d["bought"].to_numpy(bool); ret=d["realized_ret"].to_numpy(np.float32)
    del d; gc.collect()
    for s, sid in [("train",0),("val",1),("test",2)]:
        m=split==sid
        if m.any():
            arr[s]["X"].append(X[m]); arr[s]["y"].append(y[m]); arr[s]["date"].append(date[m]); arr[s]["bought"].append(bought[m]); arr[s]["ret"].append(ret[m])
    del X,y,date,bought,ret,split; gc.collect(); print(f"batch {fi} train={sum(len(x) for x in arr['train']['X'])}", flush=True)
for s in arr:
    for k in arr[s]:
        a=np.concatenate(arr[s][k]); np.save(os.path.join(OUT,f"{s}_{k}.npy"), a); print(s,k,a.shape,a.dtype, flush=True)
print("V4 ARRAYS DONE", flush=True)
