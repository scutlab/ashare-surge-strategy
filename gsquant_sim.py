# gs-quant最近1年模拟：口径与new_sim.py完全一致（卖出日窗口、3份轮换复利、Top10、未成交闲置分母10）
import pickle, glob, os, json, numpy as np, pandas as pd, gc, time
from sklearn.ensemble import HistGradientBoostingClassifier
BASE="/home/hatch/workspace/ashare"
GS_FEATS=["gs_vol_20","gs_vol_60","gs_std_ret_20","gs_rsi_14","gs_macd","gs_ma_bias_5","gs_ma_bias_20","gs_ma_bias_60","gs_beta_60","gs_maxdd_60","gs_boll_pos_20","gs_exp_vol","ret_1","ret_5","ret_20","vol_ratio_20","close_vs_high60","is_limit_up","excess_ret_1"]
t0=time.time()
codes=set()
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d
codes=sorted(codes); cmap={c:i for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
gs_parts=[]
for f in sorted(glob.glob(os.path.join(BASE,"gsquant_features","gsquant_feat_batch_*.pkl"))):
    d=pickle.load(open(f,"rb")); d=d[["code","date"]+GS_FEATS]; d["cid"]=d["code"].map(cmap); d["dint"]=pd.to_datetime(d["date"]).values.astype("datetime64[D]").astype(int); d=d.drop(columns=["code","date"])
    for c in GS_FEATS: d[c]=d[c].astype(np.float32)
    gs_parts.append(d); del d
gs=pd.concat(gs_parts, ignore_index=True); del gs_parts; print(f"gs {gs.shape}", flush=True)
arr={s:{"X":[],"y":[],"date":[],"bought":[],"ret":[],"code":[]} for s in ["train","val","test"]}
for f in sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb")); d=d[["code","date","y_strategy","bought","realized_ret"]]
    d["cid"]=d["code"].map(cmap); d["dint"]=pd.to_datetime(d["date"]).values.astype("datetime64[D]").astype(int)
    m=d.merge(gs, on=["cid","dint"], how="inner"); del d
    m["split"]=m["cid"]%3
    for s, sid in [("train",0),("val",1),("test",2)]:
        sub=m[m["split"]==sid]
        if len(sub):
            arr[s]["X"].append(sub[GS_FEATS].to_numpy(np.float32)); arr[s]["y"].append(sub["y_strategy"].to_numpy(np.int8)); arr[s]["date"].append(sub["dint"].to_numpy(np.int32)); arr[s]["bought"].append(sub["bought"].to_numpy(bool)); arr[s]["ret"].append(sub["realized_ret"].to_numpy(np.float32)); arr[s]["code"].append(sub["cid"].to_numpy(np.int32))
    del m
del gs
for s in arr:
    for k in arr[s]: arr[s][k]=np.concatenate(arr[s][k])
    print(s, arr[s]["X"].shape, flush=True)
# retrain train+val
Xtv=np.concatenate([arr["train"]["X"], arr["val"]["X"]]); ytv=np.concatenate([arr["train"]["y"], arr["val"]["y"]]); del arr["train"], arr["val"]
mdl=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
mdl.fit(Xtv, ytv); del Xtv, ytv
with open(os.path.join(BASE,"gsquant_model_gs_hgb_trainval.pkl"),"wb") as f: pickle.dump(mdl,f)
print(f"retrain done elapsed {time.time()-t0:.0f}s", flush=True)
# window: test rows, sell_date via v3 trading calendar per code
cal={}
for vf in sorted(glob.glob(os.path.join(BASE,"full_features_batch_v3_*.pkl"))):
    v=pickle.load(open(vf,"rb"))[["code","date"]]
    for code, g in v.groupby("code"):
        cal.setdefault(code, []).extend(pd.to_datetime(g["date"]).values.astype("datetime64[D]").astype(int))
    del v
for code in cal: cal[code]=np.sort(np.unique(cal[code]))
test=arr["test"]; del arr
# signal date >= 2025-10-15 (sell window starts 2025-10-23, signal is 2 trading days earlier)
mask=test["date"]>=np.datetime64("2025-10-15").astype("datetime64[D]").astype(int)
idx=np.where(mask)[0]; print(f"test recent rows {len(idx)}", flush=True)
sell=np.full(len(idx), -1, dtype=np.int32)
for j, i in enumerate(idx):
    code=codes[test["code"][i]]; arrd=cal.get(code)
    if arrd is None: continue
    pos=np.searchsorted(arrd, test["date"][i])
    if pos+2 < len(arrd): sell[j]=arrd[pos+2]
win_mask=(sell>np.datetime64("2025-10-23").astype("datetime64[D]").astype(int)) & (sell<=np.datetime64("2026-10-23").astype("datetime64[D]").astype(int))
idx=idx[win_mask]; sell=sell[win_mask]
print(f"window rows {len(idx)} signal_days {len(np.unique(test['date'][idx]))}", flush=True)
p=np.empty(len(idx), dtype=np.float32)
for i in range(0, len(idx), 200000): p[i:i+200000]=mdl.predict_proba(test["X"][idx[i:i+200000]])[:,1]
D=pd.DataFrame({"signal_date":test["date"][idx], "code":test["code"][idx], "p":p, "bought":test["bought"][idx], "R":test["ret"][idx], "y":test["y"][idx]})
days=sorted(D["signal_date"].unique()); sleeves=[1.0,1.0,1.0]; cyc=[]; n_trades=n_exec=n_win=0; n_cycles=0
for di, day in enumerate(days):
    g=D[D["signal_date"]==day]
    if len(g)<10: continue
    top=g.sort_values(["p","code"], ascending=[False,True], kind="mergesort").head(10); n_cycles+=1; ex=top[top["bought"]]
    port=float(ex["R"].astype(np.float64).sum()/10.0) if len(ex) else 0.0; sleeves[di%3]*=(1.0+port); cyc.append(port); n_trades+=10; n_exec+=len(ex); n_win+=int(top["y"].sum())
out={"model":"gs_hgb_trainval","window":"sell_date in (2025-10-23, 2026-10-23]","split":"test stocks only","capital":"3 sleeves rotating, compounding","signal_days":len(days),"cycles":n_cycles,"top10_picks":n_trades,"executed":n_exec,"exec_rate":n_exec/n_trades if n_trades else None,"win_rate_top10":n_win/n_trades if n_trades else None,"per_cycle_port_ret_mean":float(np.mean(cyc)) if cyc else None,"per_cycle_port_ret_median":float(np.median(cyc)) if cyc else None,"sleeve_multipliers":[float(s) for s in sleeves],"total_compound_return":float(np.mean(sleeves)-1.0),"base_win_rate_window":float(D["y"].mean()),"window_rows":int(len(D))}
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
with open(os.path.join(BASE,"gsquant_sim_1y.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
print("GS SIM DONE", flush=True)
