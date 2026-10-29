# gs-quant特征模型训练：与原新策略模型同口径
# 特征=gsquant_features批次；标签/执行=full_flow_new批次，按(code,date)内连接
# 划分：flow全部代码排序后i%3，0=train,1=val,2=test；评估：每日Top10信号命中、成交率、实现收益/周期
import pickle, glob, os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
BASE="/home/hatch/workspace/ashare"
GS_FEATS=["gs_vol_20","gs_vol_60","gs_std_ret_20","gs_rsi_14","gs_macd","gs_ma_bias_5","gs_ma_bias_20","gs_ma_bias_60","gs_beta_60","gs_maxdd_60","gs_boll_pos_20","gs_exp_vol","ret_1","ret_5","ret_20","vol_ratio_20","close_vs_high60","is_limit_up","excess_ret_1"]
def load_batches(pat):
    dfs=[]
    for f in sorted(glob.glob(pat)):
        d=pickle.load(open(f,"rb")); dfs.append(d)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
print("loading gs features...", flush=True); gs=load_batches(os.path.join(BASE,"gsquant_features","gsquant_feat_batch_*.pkl")); print("gs", gs.shape, flush=True)
print("loading flow...", flush=True); fl=load_batches(os.path.join(BASE,"full_flow_new_*.pkl")); print("flow", fl.shape, flush=True)
gs["date"]=pd.to_datetime(gs["date"]); fl["date"]=pd.to_datetime(fl["date"])
df=fl.merge(gs, on=["code","date"], how="inner", suffixes=("","_gs")); print("joined", df.shape, flush=True)
codes=sorted(df["code"].unique()); cmap={c:i%3 for i,c in enumerate(codes)}
df["split"]=df["code"].map(cmap)
X=df[GS_FEATS].astype(np.float32); y=df["y_strategy"].astype(int)
def evaluate(model, split_id, name):
    d=df[df["split"]==split_id]; p=model.predict_proba(d[GS_FEATS].astype(np.float32))[:,1]; d=d.assign(prob=p)
    auc=roc_auc_score(d["y_strategy"], p); ap=average_precision_score(d["y_strategy"], p)
    # per-day Top10
    hits=[]; rets=[]; execs=[]
    for date, g in d.groupby("date"):
        top=g.nlargest(10, "prob"); hits.append(top["y_strategy"].mean()); execs.append(top["bought"].mean())
        b=top[top["bought"]]
        if len(b): rets.append(b["realized_ret"].mean())
    return {"n":int(len(d)), "base_win":float(d["y_strategy"].mean()), "auc":float(auc), "pr_auc":float(ap),
            "top10_hit":float(np.mean(hits)), "top10_days":int(len(hits)), "exec_rate":float(np.mean(execs)),
            "realized_per_day":float(np.mean(rets)) if rets else 0.0, "ret_days":int(len(rets))}
for name, model in [
    ("gs_hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)),
    ("gs_logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
]:
    print(f"training {name}...", flush=True)
    tr=df["split"]==0
    model.fit(X[tr], y[tr])
    with open(os.path.join(BASE, f"gsquant_model_{name}.pkl"),"wb") as f: pickle.dump(model,f)
    res={"model":name, "target":"y_strategy", "features":GS_FEATS, "val":evaluate(model,1,name), "test":evaluate(model,2,name)}
    with open(os.path.join(BASE, f"gsquant_result_{name}.json"),"w") as f: json.dump(res,f,indent=2)
    print(name, json.dumps(res, indent=2), flush=True)
print("GS TRAIN DONE", flush=True)
