# hikyuu指标特征：用hikyuu 2.8.3 C++引擎在自有行情上计算，与原版/gs/v4同数据、同标签、同执行口径
import pickle, os, glob, gc, time, sys
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings("ignore")
import hikyuu as hku
from hikyuu import Stock, KRecord, Datetime, Query
from hikyuu.indicator import CLOSE, HIGH, LOW, VOL
BASE="/home/hatch/workspace/ashare"; OUT=os.path.join(BASE,"hikyuu_features"); os.makedirs(OUT, exist_ok=True)
HK=["hk_ma_bias_5","hk_ma_bias_20","hk_ma_bias_60","hk_ema_bias_12","hk_macd_dif","hk_macd_dea","hk_macd_hist","hk_rsi_14","hk_kdj_k","hk_kdj_d","hk_kdj_j","hk_atr_ratio","hk_boll_pos","hk_wr_10","hk_vol_ratio_5","hk_vol_ratio_20","hk_close_vs_high60","hk_close_vs_low60","hk_std_ret_20","hk_ret_1","hk_ret_5","hk_ret_20","hk_adx","hk_plus_di","hk_minus_di","excess_ret_1","is_limit_up"]
def limit_thr(code):
    if code.startswith("sh688"): return 0.195
    if code.startswith("sz300") or code.startswith("sz301"): return 0.195
    return 0.095
# 历史行情 -> numpy
t0=time.time(); hist={}
for f in ["full_raw_partial.pkl","full_new_partial.pkl"]:
    d=pickle.load(open(os.path.join(BASE,f),"rb")); d=d.sort_values(["code","date"])
    for code, g in d.groupby("code"):
        factor=(g["adjclose"]/g["close"].replace(0,np.nan)).to_numpy(np.float64)
        rec={"dates": np.array([int(pd.Timestamp(x).strftime("%Y%m%d")) for x in g["date"]], dtype=np.int64),
             "o":(g["open"].to_numpy(np.float64)*factor), "h":(g["high"].to_numpy(np.float64)*factor),
             "l":(g["low"].to_numpy(np.float64)*factor), "c":g["adjclose"].to_numpy(np.float64),
             "v":g["volume"].to_numpy(np.float64)}
        if code in hist:
            for k in rec: hist[code][k]=np.concatenate([hist[code][k], rec[k]])
        else: hist[code]=rec
    del d; gc.collect()
print(f"hist codes {len(hist)} elapsed {time.time()-t0:.0f}s", flush=True)
# 上证市场收益（按指数序列位置）
midx=pickle.load(open(os.path.join(BASE,"index_sse.pkl"),"rb")); midx=midx.sort_values("date")
mdates=[pd.Timestamp(x).strftime("%Y-%m-%d") for x in midx["date"]]; mclose=midx["mkt_adjclose"].to_numpy(np.float64)
mkt={}; 
for i,dt in enumerate(mdates):
    mkt[dt]=(mclose[i]/mclose[i-1]-1 if i>=1 else np.nan, mclose[i]/mclose[i-5]-1 if i>=5 else np.nan)
def compute(code):
    r=hist[code]; n=len(r["c"])
    ks=[KRecord(Datetime(int(dt)), float(o), float(h), float(l), float(c), float(v*c), float(v))
        for dt,o,h,l,c,v in zip(r["dates"], r["o"], r["h"], r["l"], r["c"], r["v"])]
    stk=Stock(code[:2], code[2:], code); stk.set_krecord_list(ks); k=stk.get_kdata(Query(0))
    c,h,l,v=CLOSE(k),HIGH(k),LOW(k),VOL(k)
    ma5=hku.MA(c,5).to_np()["value0"]; ma20=hku.MA(c,20).to_np()["value0"]; ma60=hku.MA(c,60).to_np()["value0"]
    ema12=hku.EMA(c,12).to_np()["value0"]
    macd=hku.MACD(c).to_np(); hist_v, dif, dea = macd["value0"], macd["value1"], macd["value2"]
    rsi=hku.RSI(c,14).to_np()["value0"]
    kdj=hku.KDJ(k); kk, dd, jj = kdj[0].to_np()["value0"], kdj[1].to_np()["value0"], kdj[2].to_np()["value0"]
    atr=hku.ATR(k,14).to_np()["value0"]
    sd20=hku.STDEV(c,20).to_np()["value0"]
    wr=(hku.HHV(h,10).to_np()["value0"]-hku.CLOSE(k).to_np()["value0"])/(hku.HHV(h,10).to_np()["value0"]-hku.LLV(l,10).to_np()["value0"])*-100
    vma5=hku.MA(v,5).to_np()["value0"]; vma20=hku.MA(v,20).to_np()["value0"]; vol=v.to_np()["value0"]
    hhv60=hku.HHV(c,60).to_np()["value0"]; llv60=hku.LLV(c,60).to_np()["value0"]
    rocp1=hku.ROCP(c,1).to_np()["value0"]; rocp5=hku.ROCP(c,5).to_np()["value0"]; rocp20=hku.ROCP(c,20).to_np()["value0"]
    sdret=hku.STDEV(hku.ROCP(c,1),20).to_np()["value0"]
    adx=hku.ADX(k,14).to_np(); ax, pdi, mdi = adx["value0"], adx["value1"], adx["value2"]
    close=c.to_np()["value0"]
    feats={"hk_ma_bias_5":close/ma5-1,"hk_ma_bias_20":close/ma20-1,"hk_ma_bias_60":close/ma60-1,
           "hk_ema_bias_12":close/ema12-1,"hk_macd_dif":dif,"hk_macd_dea":dea,"hk_macd_hist":hist_v,
           "hk_rsi_14":rsi,"hk_kdj_k":kk,"hk_kdj_d":dd,"hk_kdj_j":jj,"hk_atr_ratio":atr/close,
           "hk_boll_pos":(close-(ma20-2*sd20))/(4*sd20),"hk_wr_10":wr,
           "hk_vol_ratio_5":vol/vma5,"hk_vol_ratio_20":vol/vma20,
           "hk_close_vs_high60":close/hhv60-1,"hk_close_vs_low60":close/llv60-1,
           "hk_std_ret_20":sdret,"hk_ret_1":rocp1,"hk_ret_5":rocp5,"hk_ret_20":rocp20,
           "hk_adx":ax,"hk_plus_di":pdi,"hk_minus_di":mdi}
    return r["dates"], feats, close
files=sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))
only=sys.argv[1:] if len(sys.argv)>1 else None
for fi,f in enumerate(files):
    out=os.path.join(OUT, f"hikyuu_flow_batch_{fi:02d}.pkl")
    if os.path.exists(out): print(f"batch {fi} exists, skip", flush=True); continue
    if only and f"{fi:02d}" not in only: continue
    d=pickle.load(open(f,"rb")); bt=time.time()
    for col in HK: d[col]=np.nan
    for code, idx in d.groupby("code").groups.items():
        if code not in hist: continue
        dates, feats, close = compute(code)
        pos={int(dt):i for i,dt in enumerate(dates)}
        rows=d.loc[idx]
        di=np.array([pos.get(int(pd.Timestamp(x).strftime("%Y%m%d")), -1) for x in rows["date"]])
        ok=di>=0
        for col, arr in feats.items():
            vals=np.full(len(rows), np.nan); vals[ok]=arr[di[ok]]; d.loc[idx, col]=vals
        # 市场与涨停（结构特征，与原版同规则）
        mr=np.array([mkt.get(pd.Timestamp(x).strftime("%Y-%m-%d"), (np.nan,np.nan))[0] for x in rows["date"]])
        d.loc[idx, "excess_ret_1"]=np.where(ok, np.array([feats["hk_ret_1"][i] if i>=0 else np.nan for i in di])-mr, np.nan)
        d.loc[idx, "is_limit_up"]=(np.array([feats["hk_ret_1"][i] if i>=0 else np.nan for i in di])>=limit_thr(code)).astype(float)
    for col in HK: d[col]=d[col].astype(np.float32)
    with open(out+".tmp","wb") as fh: pickle.dump(d, fh)
    os.rename(out+".tmp", out); print(f"batch {fi} rows {len(d)} elapsed {time.time()-bt:.0f}s", flush=True)
    del d; gc.collect()
print("HIKYUU FEATURES DONE", flush=True)
