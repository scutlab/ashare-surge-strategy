# 全市场版70%候选：概率>=0.8、95折挂单、止损-2%、止盈+20%、每日Top20；口径与原搜索一致（周期收益=当日已成交单只均值，只计有成交的周期）
# A：诚实版（train-only nowd模型，分集内Top20，验证/测试分别报）B：实战频率版（trainval nowd模型，全股打分，只报触发频率与周期均值并注明含训练股样本内乐观）
import pickle, glob, os, json, gc, numpy as np, pandas as pd, time
BASE="/home/hatch/workspace/ashare"; t0=time.time()
ORIG=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
DSC, STOP, TARG, K, THR = 0.05, 0.02, 0.20, 20, 0.8
files=sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))
codes=set()
for f in files:
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d
codes=sorted(codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(codes)}", flush=True)
def payoff(entry, sl, sh, sc):
    return np.where(sl<=entry*(1-STOP), -STOP, np.where(sh>=entry*(1+TARG), TARG, sc/entry-1))
def run(model_path, tag, splits):
    mdl=pickle.load(open(model_path,"rb")); cand=[]; bench={}  # bench[(split,date)] -> [sumR, n_bought]
    for fi, f in enumerate(files):
        d=pickle.load(open(f,"rb")); sp=d["code"].map(cmap).to_numpy(np.int8)
        X=d[ORIG].to_numpy(np.float32); p=np.empty(len(d), dtype=np.float32)
        for i in range(0,len(d),200000): p[i:i+200000]=mdl.predict_proba(X[i:i+200000])[:,1]
        del X; gc.collect()
        entry=d["signal_close"].to_numpy(np.float64)*(1-DSC); bought=d["buy_low"].to_numpy(np.float64)<=entry
        R=payoff(entry, d["sell_low"].to_numpy(np.float64), d["sell_high"].to_numpy(np.float64), d["sell_close"].to_numpy(np.float64))
        dates=pd.to_datetime(d["date"]).values.astype("datetime64[D]").astype(int)
        for s in splits:
            m=(sp==s)&bought
            if m.any():
                df=pd.DataFrame({"date":dates[m],"R":R[m]})
                for dt, g in df.groupby("date"):
                    k=(s,int(dt)); a=bench.get(k,[0.0,0]); bench[k]=[a[0]+float(g["R"].sum()), a[1]+len(g)]
                del df
        cm=(p>=THR)
        if cm.any():
            sub=d[cm][["code","date","signal_close","buy_low","sell_low","sell_high","sell_close"]].copy()
            sub["prob"]=p[cm]; sub["split"]=sp[cm]; sub["bought"]=bought[cm]; sub["R"]=R[cm]
            cand.append(sub)
        del d, p, entry, bought, R, dates; gc.collect(); print(f"{tag} batch {fi} cand_rows {sum(len(c) for c in cand)} elapsed {time.time()-t0:.0f}s", flush=True)
    C=pd.concat(cand, ignore_index=True) if cand else pd.DataFrame()
    C.to_pickle(os.path.join(BASE,f"search_fullmarket_{tag}.pkl"))
    # candidate evaluation per split
    out={}
    for s in splits:
        sub=C[C["split"]==s]; cyc=[]; trades=[]
        for dt, g in sub.groupby("date"):
            top=g.nlargest(K,"prob"); ex=top[top["bought"]]
            if len(ex)==0: continue
            cyc.append(float(ex["R"].mean())); trades.extend(ex["R"].tolist())
        cyc=np.array(cyc); trades=np.array(trades)
        # benchmark on same candidate cycle dates
        bdates=set(sub["date"].unique()) if len(sub) else set()
        bvals=[]
        for (ss,dt), (sR,nb) in bench.items():
            if ss==s and pd.Timestamp(dt, unit="D").normalize() in bdates: bvals.append(sR/nb)
        # date matching: bench dates are int days, sub dates are datetime; convert
        sub_dates=set(pd.to_datetime(sub["date"]).values.astype("datetime64[D]").astype(int)) if len(sub) else set()
        bvals=[sR/nb for (ss,dt),(sR,nb) in bench.items() if ss==s and dt in sub_dates]
        def stats(a):
            a=np.array(a); n=len(a); mean=float(a.mean()) if n else None; se=float(a.std(ddof=1)/np.sqrt(n)) if n>1 else None
            return {"n":int(n),"mean":mean,"se":se,"lower70_one":float(mean-0.524*se) if se is not None else None,"lower70_two":float(mean-1.036*se) if se is not None else None,"p_cycle_gt2":float((a>0.02).mean()) if n else None}
        out[str(s)]={"cycles":stats(cyc),"trades":{"n":int(len(trades)),"mean":float(trades.mean()) if len(trades) else None,"p_trade_gt2":float((trades>0.02).mean()) if len(trades) else None},"cand_days":int(sub["date"].nunique()) if len(sub) else 0,"cand_rows":int(len(sub)),"benchmark_same_dates":stats(bvals)}
    # last-year frequency (signal date > 2025-10-21, flow max signal 2026-10-21)
    if len(C):
        C["dint"]=pd.to_datetime(C["date"]).values.astype("datetime64[D]").astype(int)
        ly=C[C["dint"]>np.datetime64("2025-10-21").astype("datetime64[D]").astype(int)]
        out["last_year"]={"cand_days":int(ly["date"].nunique()) if len(ly) else 0,"cand_rows":int(len(ly))}
        # per-split last-year cycles
        for s in splits:
            sub=ly[ly["split"]==s]; cyc=[]
            for dt, g in sub.groupby("date"):
                ex=g.nlargest(K,"prob"); ex=ex[ex["bought"]]
                if len(ex): cyc.append(float(ex["R"].mean()))
            out["last_year"][str(s)]={"cycles":len(cyc),"mean":float(np.mean(cyc)) if cyc else None}
    with open(os.path.join(BASE,f"search_fullmarket_{tag}.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
    print(f"{tag} DONE", flush=True); print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)
    return out
resA=run(os.path.join(BASE,"new_model_HGB_nowd.pkl"), "honest_nowd", [1,2])
resB=run(os.path.join(BASE,"new_model_HGB_nowd_trainval.pkl"), "operational_nowd", [0,1,2])
print("SEARCH FULLMARKET DONE", flush=True)
