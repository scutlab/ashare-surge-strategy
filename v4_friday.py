import pickle, glob, os, json, numpy as np, pandas as pd
BASE="/home/hatch/workspace/ashare"
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps","days_since_disclosure"]
FEATS=ORIG+FUND
mdl=pickle.load(open(os.path.join(BASE,"v4_model_hgb_trainval.pkl"),"rb"))
rows=[]
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_v4/full_flow_v4_*.pkl"))):
    d=pickle.load(open(f,"rb")); d=d[d["date"]=="2026-10-23"]
    if len(d): rows.append(d[["code","signal_close"]+FEATS].copy())
    del d
W=pd.concat(rows, ignore_index=True); print(f"friday rows {len(W)}", flush=True)
W["prob"]=mdl.predict_proba(W[FEATS].to_numpy(np.float32))[:,1]
top=W.sort_values(["prob","code"], ascending=[False,True]).head(20)
print(top[["code","signal_close","prob"]].to_string(), flush=True)
top[["code","signal_close","prob"]].to_csv(os.path.join(BASE,"v4_friday_top20.csv"), index=False)
with open(os.path.join(BASE,"v4_friday_top10.json"),"w") as f: json.dump(top.head(10)[["code","signal_close","prob"]].to_dict("records"), f, ensure_ascii=False, indent=2)
print("V4 FRIDAY DONE", flush=True)
