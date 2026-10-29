# FMZ候选规则评估：海龟快20/海龟慢55/Dual Thrust，用流水面板的同执行口径（挂牌/-2%/+5%/双触止损）
# 面板合并契约：键(code,date)，先3只股预检再全量；日期精度统一为datetime64[D]
import pickle, os, glob, json, gc, time, numpy as np, pandas as pd
BASE="/home/hatch/workspace/ashare"; t0=time.time()
# ---- 预检：3只股 ----
raw_parts=[]
for f in ["full_raw_partial.pkl","full_new_partial.pkl"]:
    raw_parts.append(pickle.load(open(os.path.join(BASE,f),"rb"))[["code","date","open","high","low","close","adjclose","volume"]])
raw=pd.concat(raw_parts, ignore_index=True); del raw_parts; gc.collect()
raw["date"]=pd.to_datetime(raw["date"]).values.astype("datetime64[D]")
def signals_for(g):
    g=g.sort_values("date").reset_index(drop=True)
    factor=g["adjclose"]/g["close"].replace(0,np.nan)
    o=g["open"].to_numpy(float)*factor.to_numpy(float); h=g["high"].to_numpy(float)*factor.to_numpy(float)
    l=g["low"].to_numpy(float)*factor.to_numpy(float); c=g["adjclose"].to_numpy(float)
    n=len(g)
    def prior_max(x, w):
        # max over prior w days (excluding today): rolling max shifted 1
        s=pd.Series(x); return s.rolling(w).max().shift(1).to_numpy(float)
    def prior_min(x, w):
        s=pd.Series(x); return s.rolling(w).min().shift(1).to_numpy(float)
    hh20, ll20, hc20, lc20 = prior_max(h,20), prior_min(l,20), prior_max(c,20), prior_min(c,20)
    hh55 = prior_max(h,55)
    rng=np.maximum(hh20-lc20, hc20-ll20)
    return pd.DataFrame({"date":g["date"].values,
        "turtle20": h>hh20, "turtle55": h>hh55,
        "dual": c > (o + 0.67*rng)})
pre_codes=["sh600000","sz000001","sz300567"]
for code in pre_codes:
    s=signals_for(raw[raw["code"]==code])
    print(f"preflight {code} rows {len(s)} turtle20 {int(s['turtle20'].sum())} turtle55 {int(s['turtle55'].sum())} dual {int(s['dual'].sum())}", flush=True)
# 手工核对一行：sz300567 最近一个turtle20信号日
s=signals_for(raw[raw["code"]=="sz300567"]); sig=s[s["turtle20"]].tail(1)
print("handcheck sz300567 last turtle20 signal:", sig["date"].values, flush=True)
if len(sig):
    d0=pd.Timestamp(sig["date"].values[0]); g=raw[(raw["code"]=="sz300567")&(raw["date"]<=d0)].sort_values("date").tail(21)
    factor=g["adjclose"]/g["close"]; hh=(g["high"]*factor).to_numpy(float)
    print(f"  signal-day adj high {hh[-1]:.2f} vs prior20 max {hh[:-1].max():.2f} -> breakout {hh[-1]>hh[:-1].max()}", flush=True)
# ---- 全量信号 ----
sig_frames=[]
for code, g in raw.groupby("code"):
    s=signals_for(g); s["code"]=code; sig_frames.append(s)
S=pd.concat(sig_frames, ignore_index=True); del raw, sig_frames; gc.collect()
print(f"signals total rows {len(S)} turtle20 {int(S['turtle20'].sum())} turtle55 {int(S['turtle55'].sum())} dual {int(S['dual'].sum())} elapsed {time.time()-t0:.0f}s", flush=True)
S.to_pickle(os.path.join(BASE,"fmz_signals.pkl"))
# ---- 合并流水 + 卖出日窗口 ----
flow_files=sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl"))); v3_files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_v3_*.pkl")))
all_codes=set()
for f in flow_files:
    d=pickle.load(open(f,"rb")); all_codes.update(d["code"].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}
WIN_LO=np.datetime64("2025-10-23"); WIN_HI=np.datetime64("2026-10-23")
rows=[]
for fi,(ff,vf) in enumerate(zip(flow_files, v3_files)):
    flow=pickle.load(open(ff,"rb"))
    v=pickle.load(open(vf,"rb"))[["code","date"]].sort_values(["code","date"]).reset_index(drop=True); v["sell_date"]=v.groupby("code")["date"].shift(-2)
    m=flow.merge(v, on=["code","date"], how="inner"); del flow, v; gc.collect()
    m=m[(m["sell_date"]>WIN_LO)&(m["sell_date"]<=WIN_HI)]; m=m[m["code"].map(cmap).values==2]
    if len(m): rows.append(m[["code","date","bought","realized_ret","y_strategy","both_touch"]].copy())
    del m; gc.collect()
W=pd.concat(rows, ignore_index=True); del rows; gc.collect()
print(f"window flow rows {len(W)} elapsed {time.time()-t0:.0f}s", flush=True)
W["date"]=pd.to_datetime(W["date"]).values.astype("datetime64[D]")
M=W.merge(S, on=["code","date"], how="inner"); del W, S; gc.collect()
print(f"merged rows {len(M)} (flow window rows matched to signals panel)", flush=True)
def simulate(rule):
    D=M[M[rule]].copy()
    days=sorted(D["date"].unique()); sleeves=[1.0,1.0,1.0]; cyc=[]; n_sig=n_exec=n_win=n_stop=n_take=n_both=0
    for di, day in enumerate(days):
        g=D[D["date"]==day]; ex=g[g["bought"]]
        port=float(ex["realized_ret"].astype(np.float64).sum()/len(g)) if len(g) else 0.0
        sleeves[di%3]*=(1.0+port); cyc.append(port)
        n_sig+=len(g); n_exec+=len(ex); n_win+=int(g["y_strategy"].sum())
        n_stop+=int((ex["realized_ret"]==-0.02).sum()); n_take+=int((ex["realized_ret"]==0.05).sum()); n_both+=int(ex["both_touch"].sum())
    return {"rule":rule,"signal_days":len(days),"signals":int(n_sig),"executed":int(n_exec),
            "exec_rate":n_exec/n_sig if n_sig else None,"win_rate":n_win/n_sig if n_sig else None,
            "stop_rate_exec":n_stop/n_exec if n_exec else None,"take_rate_exec":n_take/n_exec if n_exec else None,
            "both_touch":int(n_both),"per_cycle_port_ret_mean":float(np.mean(cyc)) if cyc else None,
            "per_cycle_port_ret_median":float(np.median(cyc)) if cyc else None,
            "sleeve_multipliers":[float(s) for s in sleeves],"total_compound_return":float(np.mean(sleeves)-1.0)}
res={r:simulate(r) for r in ["turtle20","turtle55","dual"]}
with open(os.path.join(BASE,"fmz_candidates.json"),"w") as f: json.dump(res,f,ensure_ascii=False,indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1), flush=True); print("DONE", flush=True)
