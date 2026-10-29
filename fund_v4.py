# 基本面v4特征：东财4张表按REPORT_DATE合并，NOTICE_DATE取最大值，生效日=披露日+1天（披露当天收盘不可用，防泄露）
# 信号日用merge_asof向后取最近已生效报告；派生特征全部用同一期或上年同期报告算
import pickle, glob, os, time, pandas as pd, numpy as np
BASE="/home/hatch/workspace/ashare"; OUT=os.path.join(BASE,"full_flow_v4"); os.makedirs(OUT, exist_ok=True)
t0=time.time()
merged={}
for f in sorted(glob.glob(os.path.join(BASE,"eastmoney_full","eastmoney_full_batch_*.pkl"))):
    out=pickle.load(open(f,"rb"))
    for code, rec in out.items(): merged[code]=dict(rec)
for f in sorted(glob.glob(os.path.join(BASE,"eastmoney_repair","eastmoney_repair_batch_*.pkl"))):
    out=pickle.load(open(f,"rb"))
    for code, rec in out.items():
        if code in merged:
            for r, v in rec.items(): merged[code][r]=v
print(f"merged codes {len(merged)} elapsed {time.time()-t0:.0f}s", flush=True)
def panel_for(code, rec):
    def df_of(rep, cols, datecol):
        v=rec[rep]
        if not (isinstance(v, list) and len(v)): return pd.DataFrame()
        d=pd.DataFrame(v); keep=[c for c in cols if c in d.columns]
        if datecol in d.columns: keep.append(datecol)
        if "NOTICE_DATE" in d.columns: keep.append("NOTICE_DATE")
        d=d[keep].copy()
        d["report_date"]=pd.to_datetime(d[datecol]); d["notice_date"]=pd.to_datetime(d["NOTICE_DATE"], errors="coerce")
        drop=[c for c in [datecol, "NOTICE_DATE"] if c in d.columns]
        return d.drop(columns=drop)
    bal=df_of("RPT_DMSK_FN_BALANCE", ["TOTAL_ASSETS","TOTAL_LIABILITIES","TOTAL_EQUITY"], "REPORT_DATE")
    inc=df_of("RPT_DMSK_FN_INCOME", ["PARENT_NETPROFIT","TOTAL_OPERATE_INCOME"], "REPORT_DATE")
    cf=df_of("RPT_DMSK_FN_CASHFLOW", ["NETCASH_OPERATE"], "REPORT_DATE")
    li=df_of("RPT_LICO_FN_CPD", ["BASIC_EPS","BPS","WEIGHTAVG_ROE"], "REPORTDATE")
    for d, tag in [(bal,"bal"),(inc,"inc"),(cf,"cf"),(li,"li")]:
        if "notice_date" in d.columns: d.rename(columns={"notice_date":f"notice_{tag}"}, inplace=True)
    p=bal
    for d in [inc, cf, li]:
        if len(d): p=p.merge(d, on="report_date", how="outer")
    if len(p)==0: return p
    # notice_date: max of available (columns NOTICE_DATE, NOTICE_DATE_r, etc.)
    ncols=[c for c in p.columns if c.startswith("notice_")]
    p["notice_date"]=p[ncols].max(axis=1) if ncols else pd.NaT
    p=p.drop(columns=ncols); p=p.dropna(subset=["notice_date","report_date"]); p["code"]=code
    p=p.sort_values("report_date")
    # prior year same date
    py=p[["report_date","TOTAL_ASSETS","PARENT_NETPROFIT","TOTAL_OPERATE_INCOME"]].copy()
    py["report_date"]=py["report_date"]+pd.DateOffset(years=1)
    py=py.rename(columns={"TOTAL_ASSETS":"TA_py","PARENT_NETPROFIT":"NP_py","TOTAL_OPERATE_INCOME":"REV_py"})
    p=p.merge(py, on="report_date", how="left")
    p["fund_leverage"]=p["TOTAL_LIABILITIES"]/p["TOTAL_ASSETS"]
    p["fund_equity_ratio"]=p["TOTAL_EQUITY"]/p["TOTAL_ASSETS"]
    p["fund_net_margin"]=p["PARENT_NETPROFIT"]/p["TOTAL_OPERATE_INCOME"]
    p["fund_cfo_assets"]=p["NETCASH_OPERATE"]/p["TOTAL_ASSETS"]
    p["fund_accruals"]=(p["PARENT_NETPROFIT"]-p["NETCASH_OPERATE"])/p["TOTAL_ASSETS"]
    p["fund_asset_growth"]=p["TOTAL_ASSETS"]/p["TA_py"]-1
    p["fund_profit_growth"]=p["PARENT_NETPROFIT"]/p["NP_py"]-1
    p["fund_rev_growth"]=p["TOTAL_OPERATE_INCOME"]/p["REV_py"]-1
    p["fund_roe"]=p["WEIGHTAVG_ROE"]; p["fund_eps"]=p["BASIC_EPS"]; p["fund_bps"]=p["BPS"]
    p["eff_date"]=p["notice_date"]+pd.Timedelta(days=1)
    return p[["code","report_date","notice_date","eff_date","fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps"]]
panels=[]
for code, rec in merged.items():
    try:
        p=panel_for(code, rec)
        if len(p): panels.append(p)
    except Exception as e: print(f"panel fail {code} {type(e).__name__} {str(e)[:150]}", flush=True)
panel=pd.concat(panels, ignore_index=True); del panels, merged; panel.to_pickle(os.path.join(BASE,"fund_panel.pkl"))
print(f"panel rows {len(panel)} codes {panel['code'].nunique()} elapsed {time.time()-t0:.0f}s", flush=True)
FUND=["fund_leverage","fund_equity_ratio","fund_net_margin","fund_cfo_assets","fund_accruals","fund_asset_growth","fund_profit_growth","fund_rev_growth","fund_roe","fund_eps","fund_bps"]
panel=panel.sort_values(["code","eff_date"]); panel["eff_date"]=pd.to_datetime(panel["eff_date"]).astype("datetime64[us]")
for fi, ff in enumerate(sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))):
    fn=os.path.join(OUT, f"full_flow_v4_{fi:02d}.pkl")
    if os.path.exists(fn):
        try: pickle.load(open(fn,"rb")); print(f"skip v4 batch {fi}", flush=True); continue
        except Exception: pass
    flow=pickle.load(open(ff,"rb")); flow=flow.sort_values(["code","date"]); flow["date"]=pd.to_datetime(flow["date"]).astype("datetime64[us]")
    m=pd.merge_asof(flow, panel, left_on="date", right_on="eff_date", by="code", direction="backward", suffixes=("","_f"))
    m["days_since_disclosure"]=(pd.to_datetime(m["date"])-pd.to_datetime(m["notice_date"])).dt.days
    with open(fn+".tmp","wb") as f: pickle.dump(m,f)
    os.rename(fn+".tmp", fn)
    print(f"v4 batch {fi} done rows={len(m)} with_fund={m['fund_leverage'].notna().mean():.3f} elapsed={time.time()-t0:.0f}s -> {fn}", flush=True)
    del flow, m
print(f"V4 DONE elapsed={time.time()-t0:.0f}s", flush=True)
