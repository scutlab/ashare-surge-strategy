# v4训练：飙升监测策略29特征+11基本面+披露天数，共41特征；同口径评估
import pickle, glob, os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
BASE="/home/hatch/workspace/ashare"
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps","days_since_disclosure"]
FEATS=ORIG+FUND
arr={s:{"X":[],"y":[],"date":[],"bought":[],"ret":[]} for s in ["train","val","test"]}
codes=set()
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_v4/full_flow_v4_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d
codes=sorted(codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_v4/full_flow_v4_*.pkl"))):
    d=pickle.load(open(f,"rb")); d["split"]=d["code"].map(cmap)
    for s, sid in [("train",0),("val",1),("test",2)]:
        sub=d[d["split"]==sid]
        if len(sub):
            arr[s]["X"].append(sub[FEATS].to_numpy(np.float32)); arr[s]["y"].append(sub["y_strategy"].to_numpy(np.int8)); arr[s]["date"].append(pd.to_datetime(sub["date"]).values.astype("datetime64[D]").astype(int)); arr[s]["bought"].append(sub["bought"].to_numpy(bool)); arr[s]["ret"].append(sub["realized_ret"].to_numpy(np.float32))
    del d; print(f"loaded {f} train={sum(len(x) for x in arr['train']['X'])}", flush=True)
for s in arr:
    for k in arr[s]: arr[s][k]=np.concatenate(arr[s][k])
    print(s, arr[s]["X"].shape, "base", float(arr[s]["y"].mean()), flush=True)
def evaluate(model, s):
    X=arr[s]["X"]; y=arr[s]["y"]; p=model.predict_proba(X)[:,1]
    auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    df=pd.DataFrame({"date":arr[s]["date"],"y":y,"bought":arr[s]["bought"],"ret":arr[s]["ret"],"prob":p}); hits=[]; rets=[]; execs=[]
    for date, g in df.groupby("date"):
        top=g.nlargest(10,"prob"); hits.append(top["y"].mean()); execs.append(top["bought"].mean())
        b=top[top["bought"]]
        if len(b): rets.append(b["ret"].mean())
    del df
    return {"n":int(len(y)), "base_win":float(y.mean()), "auc":float(auc), "pr_auc":float(ap), "top10_hit":float(np.mean(hits)), "top10_days":int(len(hits)), "exec_rate":float(np.mean(execs)), "realized_per_day":float(np.mean(rets)) if rets else 0.0, "ret_days":int(len(rets))}
mdl=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
print("training v4 hgb...", flush=True); mdl.fit(arr["train"]["X"], arr["train"]["y"])
with open(os.path.join(BASE,"v4_model_hgb.pkl"),"wb") as f: pickle.dump(mdl,f)
res={"model":"v4_hgb","target":"y_strategy","features":FEATS,"val":evaluate(mdl,"val"),"test":evaluate(mdl,"test")}
with open(os.path.join(BASE,"v4_result_hgb.json"),"w") as f: json.dump(res,f,indent=2)
print(json.dumps(res, indent=2), flush=True); print("V4 TRAIN DONE", flush=True)
