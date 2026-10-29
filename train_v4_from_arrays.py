import pickle, os, json, gc, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
BASE="/home/hatch/workspace/ashare"; A=os.path.join(BASE,"v4_arrays")
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps","days_since_disclosure"]
FEATS=ORIG+FUND
arr={s:{k:np.load(os.path.join(A,f"{s}_{k}.npy")) for k in ["X","y","date","bought","ret"]} for s in ["train","val","test"]}
for s in arr: print(s, arr[s]["X"].shape, "base", float(arr[s]["y"].mean()), flush=True)
def evaluate(model, s):
    X=arr[s]["X"]; y=arr[s]["y"]; p=model.predict_proba(X)[:,1]
    auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    df=pd.DataFrame({"date":arr[s]["date"],"y":y,"bought":arr[s]["bought"],"ret":arr[s]["ret"],"prob":p}); hits=[]; rets=[]; execs=[]
    for date, g in df.groupby("date"):
        top=g.nlargest(10,"prob"); hits.append(top["y"].mean()); execs.append(top["bought"].mean())
        b=top[top["bought"]]
        if len(b): rets.append(b["ret"].mean())
    del df; gc.collect()
    return {"n":int(len(y)), "base_win":float(y.mean()), "auc":float(auc), "pr_auc":float(ap), "top10_hit":float(np.mean(hits)), "top10_days":int(len(hits)), "exec_rate":float(np.mean(execs)), "realized_per_day":float(np.mean(rets)) if rets else 0.0, "ret_days":int(len(rets))}
mdl=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
print("training v4 hgb...", flush=True); mdl.fit(arr["train"]["X"], arr["train"]["y"])
with open(os.path.join(BASE,"v4_model_hgb.pkl"),"wb") as f: pickle.dump(mdl,f)
res={"model":"v4_hgb","target":"y_strategy","features":FEATS,"val":evaluate(mdl,"val"),"test":evaluate(mdl,"test")}
with open(os.path.join(BASE,"v4_result_hgb.json"),"w") as f: json.dump(res,f,indent=2)
print(json.dumps(res, indent=2), flush=True); print("V4 TRAIN DONE", flush=True)
