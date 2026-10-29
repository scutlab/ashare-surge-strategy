# 周一信号打分：周一行情 + 历史至周五 -> 29特征 -> 最优模型 -> Top20（周二挂牌买入、周三止损止盈）
import pickle, os, json, gc, numpy as np, pandas as pd, time
BASE="/home/hatch/workspace/ashare"; t0=time.time()
import sys; sys.path.insert(0, BASE)
from daily_pipeline import features_for_day, board, limit_thr
# 上证
midx=pickle.load(open(os.path.join(BASE,"index_sse.pkl"),"rb"))
sse={pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d,v in zip(midx["date"], midx["mkt_adjclose"])}
sse_mon=3697.52  # Yahoo meta regularMarketPrice（周一收盘，chart close字段为空）
fri=sse["2026-10-23"]; mkt_ret=sse_mon/fri-1
# 5个交易日前：交易日序列 ...10-19,10-20,10-21,10-22,10-23,10-26 -> 分母10-19
mkt_ret5=sse_mon/sse["2026-10-19"]-1
print(f"mkt_ret {mkt_ret:.6f} mkt_ret5 {mkt_ret5:.6f}", flush=True)
parts=[]
for f in ["full_raw_partial.pkl","full_new_partial.pkl"]:
    parts.append(pickle.load(open(os.path.join(BASE,f),"rb"))[["code","date","open","high","low","close","adjclose","volume"]])
raw=pd.concat(parts, ignore_index=True); del parts; gc.collect(); raw["date"]=pd.to_datetime(raw["date"])
bars=pickle.load(open(os.path.join(BASE,"monday_20261026.pkl"),"rb"))
# features_for_day按股过滤全表太慢，改用分组dict：临时改写——直接在此实现分组版
by={c: g.sort_values("date") for c,g in raw.groupby("code")}
del raw; gc.collect()
FEATS=["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
rows=[]
day_ts=pd.Timestamp("2026-10-26")
for code, bar in bars.items():
    if not isinstance(bar, dict) or not bar.get("volume"): continue
    g=by.get(code)
    if g is None or len(g)<60: continue
    new=pd.DataFrame([{"code":code,"date":day_ts,"open":bar["open"],"high":bar["high"],"low":bar["low"],"close":bar["close"],"adjclose":bar["adjclose"],"volume":bar["volume"]}])
    gg=pd.concat([g.tail(300), new], ignore_index=True).sort_values("date")
    adj=gg["adjclose"]; factor=adj/gg["close"].replace(0,np.nan); adj_high=gg["high"]*factor; adj_low=gg["low"]*factor; ret=adj.pct_change()
    feat={}
    feat["vol_5"]=ret.rolling(5).std().iloc[-1]; feat["vol_20"]=ret.rolling(20).std().iloc[-1]
    feat["atr_14"]=(np.maximum(adj_high-adj_low, np.maximum((adj_high-adj.shift(1)).abs(), (adj_low-adj.shift(1)).abs())).rolling(14).mean()/adj).iloc[-1]
    big=(ret>0.05).astype(float); feat["big_up_freq_60"]=big.shift(1).rolling(60).mean().iloc[-1]; feat["big_up_freq_250"]=big.shift(1).rolling(250).mean().iloc[-1]
    feat["ret_1"]=ret.iloc[-1]; feat["ret_3"]=adj.pct_change(3).iloc[-1]; feat["ret_5"]=adj.pct_change(5).iloc[-1]; feat["ret_10"]=adj.pct_change(10).iloc[-1]; feat["ret_20"]=adj.pct_change(20).iloc[-1]
    feat["ma_bias_5"]=(adj/adj.rolling(5).mean()-1).iloc[-1]; feat["ma_bias_20"]=(adj/adj.rolling(20).mean()-1).iloc[-1]; feat["ma_bias_60"]=(adj/adj.rolling(60).mean()-1).iloc[-1]
    feat["close_vs_high60"]=(adj/adj.rolling(60).max()-1).iloc[-1]; feat["close_vs_low60"]=(adj/adj.rolling(60).min()-1).iloc[-1]
    feat["breakout_20"]=float(adj.iloc[-1]>adj.shift(1).rolling(20).max().iloc[-1])
    up=(ret>0); feat["consec_up"]=up.groupby((~up).cumsum()).cumsum().iloc[-1]
    feat["vol_ratio_5"]=(gg["volume"]/gg["volume"].shift(1).rolling(5).mean()).iloc[-1]; feat["vol_ratio_20"]=(gg["volume"]/gg["volume"].shift(1).rolling(20).mean()).iloc[-1]
    lim=(ret>=limit_thr(code)).astype(float); feat["is_limit_up"]=lim.iloc[-1]; feat["limit_cnt_5"]=lim.shift(1).rolling(5).sum().iloc[-1]; feat["limit_cnt_20"]=lim.shift(1).rolling(20).sum().iloc[-1]
    delta=adj.diff(); ag=delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); al=(-delta).clip(lower=0).ewm(alpha=1/14, adjust=False).mean(); feat["rsi_14"]=(100-100/(1+ag/al.replace(0,np.nan))).iloc[-1]
    ema12=adj.ewm(span=12, adjust=False).mean(); ema26=adj.ewm(span=26, adjust=False).mean(); dif=ema12-ema26; dea=dif.ewm(span=9, adjust=False).mean(); feat["macd_diff"]=(dif/adj).iloc[-1]; feat["macd_hist"]=((dif-dea)/adj).iloc[-1]
    feat["mkt_ret"]=mkt_ret; feat["mkt_ret5"]=mkt_ret5; feat["excess_ret_1"]=feat["ret_1"]-mkt_ret; feat["excess_ret_5"]=feat["ret_5"]-mkt_ret5
    feat["code"]=code; feat["close"]=float(bar["adjclose"]); feat["board"]=board(code); rows.append(feat)
W=pd.DataFrame(rows); print(f"feature rows {len(W)} elapsed {time.time()-t0:.0f}s", flush=True)
W=W.dropna(subset=FEATS); print(f"after dropna {len(W)}", flush=True)
W.to_pickle(os.path.join(BASE,"monday_20261026_features.pkl"))
mdl=pickle.load(open(os.path.join(BASE,"new_model_HGB_nowd_trainval.pkl"),"rb"))
W["prob"]=mdl.predict_proba(W[FEATS].to_numpy(np.float32))[:,1]
names={}
try:
    lst=pd.read_csv(os.path.join(BASE,"sina_hs_a_20261025.csv")); names=dict(zip(lst["symbol"], lst["name"]))
except Exception: pass
top=W.sort_values(["prob","code"], ascending=[False,True]).head(20)
for i,(_,r) in enumerate(top.iterrows(),1):
    c=float(r["close"]); print(f"{i}. {names.get(r['code'],'')} {r['code']} close {c:.2f} stop {c*0.98:.2f} target {c*1.05:.2f} prob {r['prob']:.4f}", flush=True)
out=[{"rank":i,"code":r["code"],"name":names.get(r["code"],""),"monday_close":float(r["close"]),"stop_price":round(float(r["close"])*0.98,2),"target_price":round(float(r["close"])*1.05,2),"prob_ge5":float(r["prob"])} for i,(_,r) in enumerate(top.iterrows(),1)]
with open(os.path.join(BASE,"monday_20261026_top20.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=2)
print(f"SCORE MONDAY DONE elapsed {time.time()-t0:.0f}s", flush=True)
