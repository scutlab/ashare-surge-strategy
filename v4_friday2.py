import pickle, os, json, numpy as np, pandas as pd
BASE="/home/hatch/workspace/ashare"
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps","days_since_disclosure"]
FEATS=ORIG+FUND
mdl=pickle.load(open(os.path.join(BASE,"v4_model_hgb_trainval.pkl"),"rb"))
W=pickle.load(open(os.path.join(BASE,"friday_20261023_all_v3.pkl"),"rb"))
panel=pd.read_pickle(os.path.join(BASE,"fund_panel.pkl")); panel=panel.sort_values(["code","eff_date"])
sig=pd.Timestamp("2026-10-23")
fund=np.full((len(W), len(FUND)-1), np.nan); notice=np.full(len(W), -1, dtype=int)
pan={c:(pd.to_datetime(g["eff_date"]).values.astype("datetime64[D]").astype(int), g[FUND[:-1]].to_numpy(np.float64), pd.to_datetime(g["notice_date"]).values.astype("datetime64[D]").astype(int)) for c,g in panel.groupby("code")}
sigint=np.datetime64("2026-10-23").astype("datetime64[D]").astype(int)
for i, code in enumerate(W["code"].values):
    if code not in pan: continue
    eff, feats, nd = pan[code]; pos=np.searchsorted(eff, sigint, side="right")-1
    if pos>=0: fund[i]=feats[pos]; notice[i]=nd[pos]
for j,c in enumerate(FUND[:-1]): W[c]=fund[:,j]
W["days_since_disclosure"]=np.where(notice>=0, sigint-notice, np.nan)
print(f"friday rows {len(W)} with_fund {W['fund_leverage'].notna().mean():.3f}", flush=True)
W["prob"]=mdl.predict_proba(W[FEATS].to_numpy(np.float32))[:,1]
top=W.sort_values(["prob","code"], ascending=[False,True]).head(20)
print(top[["code","adjclose","prob"]].to_string(), flush=True)
out=top.head(10)[["code","adjclose","prob"]].rename(columns={"adjclose":"friday_close"}).to_dict("records")
with open(os.path.join(BASE,"v4_friday_top10.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
top[["code","adjclose","prob"]].to_csv(os.path.join(BASE,"v4_friday_top20.csv"), index=False)
print("V4 FRIDAY DONE", flush=True)
