# gs-quant特征续跑：跳过已有批次里的代码，从下一批号继续，断点安全
import pickle, glob, os, time, pandas as pd, numpy as np
from gs_quant.timeseries import econometrics as ec, technicals as te, statistics as st
OUT="/home/hatch/workspace/ashare/gsquant_features"
existing=sorted(glob.glob(os.path.join(OUT,"gsquant_feat_batch_*.pkl")))
done=set(); start_idx=len(existing)
for f in existing:
    d=pickle.load(open(f,"rb")); done|=set(d["code"].unique())
print(f"resume: existing batches={len(existing)} done codes={len(done)} start_idx={start_idx}", flush=True)
raw=pd.concat([pickle.load(open("/home/hatch/workspace/ashare/full_raw_partial.pkl","rb")),
               pickle.load(open("/home/hatch/workspace/ashare/full_new_partial.pkl","rb"))], ignore_index=True)
idx=pickle.load(open("/home/hatch/workspace/ashare/index_sse.pkl","rb")); idx=idx.set_index("date")["mkt_adjclose"].sort_index()
raw=raw.sort_values(["code","date"]); print("raw", raw.shape, flush=True)
def feats_for(g, code):
    g=g.set_index("date").sort_index(); adj=g["adjclose"]; ret=adj.pct_change(); idf=idx.reindex(g.index).ffill(); mkt_ret=idf.pct_change()
    board="ChiNext" if code.startswith("sz30") else ("STAR" if code.startswith("sh688") else "Main"); thr=0.195 if board in ("ChiNext","STAR") else 0.095
    df=pd.DataFrame(index=g.index)
    df["gs_vol_20"]=ec.volatility(adj, w=20); df["gs_vol_60"]=ec.volatility(adj, w=60); df["gs_std_ret_20"]=st.std(ret, w=20)
    df["gs_rsi_14"]=te.relative_strength_index(adj, w=14); df["gs_macd"]=te.macd(adj)
    df["gs_ma_bias_5"]=adj/te.moving_average(adj, w=5)-1; df["gs_ma_bias_20"]=adj/te.moving_average(adj, w=20)-1; df["gs_ma_bias_60"]=adj/te.moving_average(adj, w=60)-1
    df["gs_beta_60"]=ec.beta(adj, idf, w=60); df["gs_maxdd_60"]=ec.max_drawdown(adj, w=60)
    bb=te.bollinger_bands(adj, w=20, k=2); df["gs_boll_pos_20"]=(adj-bb.iloc[:,0])/(bb.iloc[:,1]-bb.iloc[:,0]); df["gs_exp_vol"]=te.exponential_volatility(adj)
    df["ret_1"]=ret; df["ret_5"]=adj/adj.shift(5)-1; df["ret_20"]=adj/adj.shift(20)-1
    df["vol_ratio_20"]=g["volume"]/g["volume"].shift(1).rolling(20).mean(); df["close_vs_high60"]=adj/adj.shift(1).rolling(60).max()-1
    df["is_limit_up"]=(ret>=thr).astype(float); df["excess_ret_1"]=ret-mkt_ret
    df=df.dropna(); df["code"]=code; df=df.reset_index().rename(columns={"index":"date"}); return df
t0=time.time(); batch=[]; bidx=start_idx; ncode=0
for code, g in raw.groupby("code", sort=True):
    if code in done or len(g)<250: continue
    try: batch.append(feats_for(g, code))
    except Exception as e: print(f"feat fail {code} {type(e).__name__} {str(e)[:150]}", flush=True)
    ncode+=1
    if ncode>=200:
        fn=os.path.join(OUT, f"gsquant_feat_batch_{bidx:02d}.pkl"); out=pd.concat(batch, ignore_index=True)
        with open(fn+".tmp","wb") as f: pickle.dump(out,f)
        os.rename(fn+".tmp", fn); print(f"batch {bidx} done rows={len(out)} elapsed={time.time()-t0:.0f}s -> {fn}", flush=True); batch=[]; bidx+=1; ncode=0
if batch:
    fn=os.path.join(OUT, f"gsquant_feat_batch_{bidx:02d}.pkl"); out=pd.concat(batch, ignore_index=True)
    with open(fn+".tmp","wb") as f: pickle.dump(out,f)
    os.rename(fn+".tmp", fn); print(f"batch {bidx} done rows={len(out)} elapsed={time.time()-t0:.0f}s -> {fn}", flush=True)
print(f"GS FEATURES RESUME DONE elapsed={time.time()-t0:.0f}s", flush=True)
