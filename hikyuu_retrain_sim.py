# hikyuu重训(train+val)+最近1年模拟，同new_sim/v4口径：卖出日窗口、测试集、3份轮换复利、Top10
import pickle, glob, os, json, gc, numpy as np, pandas as pd, time
from sklearn.ensemble import HistGradientBoostingClassifier
BASE="/home/hatch/workspace/ashare"; t0=time.time()
FEATS=["hk_ma_bias_5","hk_ma_bias_20","hk_ma_bias_60","hk_ema_bias_12","hk_macd_dif","hk_macd_dea","hk_macd_hist","hk_rsi_14","hk_atr_ratio","hk_boll_pos","hk_wr_10","hk_vol_ratio_5","hk_vol_ratio_20","hk_close_vs_high60","hk_close_vs_low60","hk_std_ret_20","hk_ret_1","hk_ret_5","hk_ret_20","hk_adx","hk_plus_di","hk_minus_di","excess_ret_1","is_limit_up"]
flow_files=sorted(glob.glob(os.path.join(BASE,"hikyuu_features/hikyuu_flow_batch_*.pkl"))); assert len(flow_files)==25
all_codes=set()
for f in flow_files:
    d=pickle.load(open(f,"rb")); all_codes.update(d["code"].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
Xtv=[]; ytv=[]
for f in flow_files:
    d=pickle.load(open(f,"rb")); sp=d["code"].map(cmap).values; m=sp!=2
    if m.any(): Xtv.append(d[FEATS].to_numpy(np.float32)[m]); ytv.append(d["y_strategy"].to_numpy(np.int8)[m])
    del d, sp; gc.collect()
Xtv=np.concatenate(Xtv); ytv=np.concatenate(ytv); print(f"trainval {len(ytv)} elapsed {time.time()-t0:.0f}s", flush=True)
mdl=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
mdl.fit(Xtv, ytv); del Xtv, ytv
with open(os.path.join(BASE,"hikyuu_model_hgb_trainval.pkl"),"wb") as f: pickle.dump(mdl,f)
print(f"retrain done elapsed {time.time()-t0:.0f}s", flush=True)
v3_files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_v3_*.pkl"))); rows=[]; WIN_LO=np.datetime64("2025-10-23"); WIN_HI=np.datetime64("2026-10-23")
for fi,(ff,vf) in enumerate(zip(flow_files, v3_files)):
    flow=pickle.load(open(ff,"rb"))
    v=pickle.load(open(vf,"rb"))[["code","date"]].sort_values(["code","date"]).reset_index(drop=True); v["sell_date"]=v.groupby("code")["date"].shift(-2)
    m=flow.merge(v, on=["code","date"], how="inner"); del flow, v; gc.collect()
    m=m[(m["sell_date"]>WIN_LO)&(m["sell_date"]<=WIN_HI)]; m=m[m["code"].map(cmap).values==2]
    if len(m): rows.append(m[["code","date","sell_date","bought","realized_ret","y_strategy","both_touch"]+FEATS].copy())
    del m; gc.collect(); print(f"batch {fi} window_rows {sum(len(r) for r in rows)} elapsed {time.time()-t0:.0f}s", flush=True)
W=pd.concat(rows, ignore_index=True); del rows; gc.collect(); print(f"window total {len(W)} signal_days {W['date'].nunique()}", flush=True)
X=W[FEATS].to_numpy(np.float32); p=np.empty(len(W), dtype=np.float32)
for i in range(0,len(W),200000): p[i:i+200000]=mdl.predict_proba(X[i:i+200000])[:,1]
D=pd.DataFrame({"signal_date":W["date"].values,"code":W["code"].values,"p":p,"bought":W["bought"].values,"R":W["realized_ret"].values,"y":W["y_strategy"].values,"both":W["both_touch"].values}); del W,X,p; gc.collect()
days=sorted(D["signal_date"].unique()); sleeves=[1.0,1.0,1.0]; cyc=[]; cyc_b=[]; n_trades=n_exec=n_win=n_stop=n_take=n_both=n_cycles=0
for di, day in enumerate(days):
    g=D[D["signal_date"]==day]
    if len(g)<10: continue
    top=g.sort_values(["p","code"], ascending=[False,True], kind="mergesort").head(10); n_cycles+=1; ex=top[top["bought"]]
    port=float(ex["R"].astype(np.float64).sum()/10.0) if len(ex) else 0.0; sleeves[di%3]*=(1.0+port); cyc.append(port)
    if len(ex): cyc_b.append(float(ex["R"].astype(np.float64).mean()))
    n_trades+=10; n_exec+=len(ex); n_win+=int(top["y"].sum()); n_stop+=int((ex["R"]==-0.02).sum()); n_take+=int((ex["R"]==0.05).sum()); n_both+=int(ex["both"].sum())
out={"model":"hikyuu_hgb_trainval","window":"sell_date in (2025-10-23, 2026-10-23]","split":"test stocks only","capital":"3 sleeves rotating, compounding","signal_days":len(days),"cycles":n_cycles,"top10_picks":n_trades,"executed":n_exec,"exec_rate":n_exec/n_trades if n_trades else None,"win_rate_top10":n_win/n_trades if n_trades else None,"stop_rate_exec":n_stop/n_exec if n_exec else None,"take_rate_exec":n_take/n_exec if n_exec else None,"both_touch_exec":n_both,"per_cycle_port_ret_mean":float(np.mean(cyc)) if cyc else None,"per_cycle_port_ret_median":float(np.median(cyc)) if cyc else None,"per_cycle_bought_mean_ret":float(np.mean(cyc_b)) if cyc_b else None,"sleeve_multipliers":[float(s) for s in sleeves],"total_compound_return":float(np.mean(sleeves)-1.0),"base_win_rate_window":float(D["y"].mean()),"window_rows":int(len(D))}
with open(os.path.join(BASE,"hikyuu_sim_1y.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=1)
print(json.dumps(out, ensure_ascii=False, indent=2), flush=True); print("SIM_DONE", flush=True)
