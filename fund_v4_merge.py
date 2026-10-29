# v4合并：不用merge_asof，改按股searchsorted取最近已生效报告（避开全局排序要求）
import pickle, glob, os, time, pandas as pd, numpy as np
BASE="/home/hatch/workspace/ashare"; OUT=os.path.join(BASE,"full_flow_v4"); os.makedirs(OUT, exist_ok=True)
panel=pd.read_pickle(os.path.join(BASE,"fund_panel.pkl")); panel=panel.sort_values(["code","eff_date"])
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps"]
# panel per code arrays
pan={}
for code, g in panel.groupby("code"):
    pan[code]=(pd.to_datetime(g["eff_date"]).values.astype("datetime64[D]").astype(int), g[FUND].to_numpy(np.float64), pd.to_datetime(g["notice_date"]).values.astype("datetime64[D]").astype(int), pd.to_datetime(g["report_date"]).values.astype("datetime64[D]").astype(int))
del panel; print(f"panel codes {len(pan)}", flush=True)
t0=time.time()
for fi, ff in enumerate(sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))):
    fn=os.path.join(OUT, f"full_flow_v4_{fi:02d}.pkl")
    if os.path.exists(fn):
        try: pickle.load(open(fn,"rb")); print(f"skip v4 batch {fi}", flush=True); continue
        except Exception: pass
    flow=pickle.load(open(ff,"rb")).reset_index(drop=True)
    fund=np.full((len(flow), len(FUND)), np.nan); notice=np.full(len(flow), -1, dtype=np.int32); report=np.full(len(flow), -1, dtype=np.int32)
    dates=pd.to_datetime(flow["date"]).values.astype("datetime64[D]").astype(int)
    for code, idx in flow.groupby("code").groups.items():
        if code not in pan: continue
        eff, feats, nd, rd = pan[code]
        pos=np.searchsorted(eff, dates[idx], side="right")-1
        valid=pos>=0
        ii=np.array(list(idx))[valid]; pp=pos[valid]
        fund[ii]=feats[pp]; notice[ii]=nd[pp]; report[ii]=rd[pp]
    for j, c in enumerate(FUND): flow[c]=fund[:,j]
    flow["notice_date"]=pd.to_datetime(notice, unit="D", origin="unix") if False else pd.Series(np.where(notice>=0, notice, np.nan))
    # days_since: signal date - notice date (in days, using int dates)
    flow["days_since_disclosure"]=np.where(notice>=0, dates-notice, np.nan)
    flow["fund_report_date"]=report
    with open(fn+".tmp","wb") as f: pickle.dump(flow,f)
    os.rename(fn+".tmp", fn)
    print(f"v4 batch {fi} done rows={len(flow)} with_fund={flow['fund_leverage'].notna().mean():.3f} elapsed={time.time()-t0:.0f}s -> {fn}", flush=True)
    del flow, fund
print(f"V4 MERGE DONE elapsed={time.time()-t0:.0f}s", flush=True)
