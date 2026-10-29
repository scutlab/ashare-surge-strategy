# gs-quant训练 v2：内存安全版，分批拼接+int键合并，不再全量DataFrame合并
import pickle, glob, os, json, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
BASE="/home/hatch/workspace/ashare"
GS_FEATS=["gs_vol_20","gs_vol_60","gs_std_ret_20","gs_rsi_14","gs_macd","gs_ma_bias_5","gs_ma_bias_20","gs_ma_bias_60","gs_beta_60","gs_maxdd_60","gs_boll_pos_20","gs_exp_vol","ret_1","ret_5","ret_20","vol_ratio_20","close_vs_high60","is_limit_up","excess_ret_1"]
# 1) 代码表：只读code列
codes=set()
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d
codes=sorted(codes); cmap={c:i for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
# 2) gs特征：只留需要的列，转int键+float32
gs_parts=[]
for f in sorted(glob.glob(os.path.join(BASE,"gsquant_features","gsquant_feat_batch_*.pkl"))):
    d=pickle.load(open(f,"rb")); d=d[["code","date"]+GS_FEATS]
    d["cid"]=d["code"].map(cmap); d["dint"]=pd.to_datetime(d["date"]).values.astype("datetime64[D]").astype(int)
    d=d.drop(columns=["code","date"])
    for c in GS_FEATS: d[c]=d[c].astype(np.float32)
    gs_parts.append(d); del d
gs=pd.concat(gs_parts, ignore_index=True); del gs_parts; gs=gs.sort_values(["cid","dint"]); print(f"gs {gs.shape}", flush=True)
# 3) 分批过flow，合并后只留numpy数组
arr={s:{"X":[],"y":[],"date":[],"bought":[],"ret":[]} for s in ["train","val","test"]}
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb"))
    d=d[["code","date","y_strategy","bought","realized_ret"]]
    d["cid"]=d["code"].map(cmap); d["dint"]=pd.to_datetime(d["date"]).values.astype("datetime64[D]").astype(int); d=d.drop(columns=["code","date"])
    m=d.merge(gs, on=["cid","dint"], how="inner"); del d
    m["split"]=m["cid"]%3
    for s, sid in [("train",0),("val",1),("test",2)]:
        sub=m[m["split"]==sid]
        if len(sub):
            arr[s]["X"].append(sub[GS_FEATS].to_numpy(np.float32)); arr[s]["y"].append(sub["y_strategy"].to_numpy(np.int8))
            arr[s]["date"].append(sub["dint"].to_numpy(np.int32)); arr[s]["bought"].append(sub["bought"].to_numpy(bool)); arr[s]["ret"].append(sub["realized_ret"].to_numpy(np.float32))
    del m; print(f"joined {f} total_train={sum(len(x) for x in arr['train']['X'])}", flush=True)
del gs
for s in arr:
    for k in arr[s]: arr[s][k]=np.concatenate(arr[s][k]) if arr[s][k] else np.array([])
    print(s, arr[s]["X"].shape, "base", float(arr[s]["y"].mean()) if len(arr[s]["y"]) else 0, flush=True)
def evaluate(model, s):
    X=arr[s]["X"]; y=arr[s]["y"]; p=model.predict_proba(X)[:,1]
    auc=roc_auc_score(y,p); ap=average_precision_score(y,p)
    df=pd.DataFrame({"date":arr[s]["date"],"y":y,"bought":arr[s]["bought"],"ret":arr[s]["ret"],"prob":p})
    hits=[]; rets=[]; execs=[]
    for date, g in df.groupby("date"):
        top=g.nlargest(10,"prob"); hits.append(top["y"].mean()); execs.append(top["bought"].mean())
        b=top[top["bought"]]
        if len(b): rets.append(b["ret"].mean())
    del df
    return {"n":int(len(y)), "base_win":float(y.mean()), "auc":float(auc), "pr_auc":float(ap), "top10_hit":float(np.mean(hits)), "top10_days":int(len(hits)), "exec_rate":float(np.mean(execs)), "realized_per_day":float(np.mean(rets)) if rets else 0.0, "ret_days":int(len(rets))}
for name, model in [
    ("gs_hgb", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)),
    ("gs_logreg", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))),
]:
    print(f"training {name}...", flush=True); model.fit(arr["train"]["X"], arr["train"]["y"])
    with open(os.path.join(BASE, f"gsquant_model_{name}.pkl"),"wb") as f: pickle.dump(model,f)
    res={"model":name, "target":"y_strategy", "features":GS_FEATS, "val":evaluate(model,"val"), "test":evaluate(model,"test")}
    with open(os.path.join(BASE, f"gsquant_result_{name}.json"),"w") as f: json.dump(res,f,indent=2)
    print(name, json.dumps(res, indent=2), flush=True)
print("GS TRAIN V2 DONE", flush=True)
