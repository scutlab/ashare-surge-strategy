# m=2 Top3 按周/月聚合实际盈亏（扣费后）
import numpy as np, pandas as pd, os, json, pickle, glob, gc, time, sys
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import strategy_config as SC
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
TOPN = SC.N_STRATEGY; X = 0.10; FEE = SC.FEE
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64)
ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
dte = np.load(f"{PP}/te_day.npy"); cte = np.load(f"{PP}/te_code.npy")
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
codes = json.load(open(f"{PP}/codes.json"))["codes"]; cid = {c: i for i, c in enumerate(codes)}
# 用已重训的含费模型打分（排名与之前一致）
reg = pickle.load(open(f"{BASE}/ev_model_10_2.pkl", "rb"))
s = reg.predict(Xte).astype(np.float32); del reg; gc.collect()
order = np.lexsort((-s, dte)); dk = dte[order]
upos = np.sort(np.unique(dk, return_index=True)[1])
picks = []
for i, st in enumerate(upos):
    e = upos[i+1] if i+1 < len(upos) else len(order)
    picks.extend(order[st:st+TOPN])
picks = np.array(picks)
# opens
need = pd.DataFrame({"ci": cte[picks].astype(np.int32), "dk": dte[picks].astype(np.int32),
                     "pi": np.arange(len(picks))})
bo = np.full(len(picks), np.nan); o2 = np.full(len(picks), np.nan); o3 = np.full(len(picks), np.nan)
for vf in sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl")):
    v = pd.read_pickle(vf)[["code", "date", "adj_open"]]
    v["ci"] = v["code"].map(cid).astype(np.int32)
    v["dk"] = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    v = v.sort_values(["ci", "dk"]).reset_index(drop=True)
    g = v.groupby("ci")["adj_open"]
    v["bo"] = g.shift(-1); v["o2"] = g.shift(-2); v["o3"] = g.shift(-3)
    mg = need.merge(v[["ci", "dk", "bo", "o2", "o3"]], on=["ci", "dk"], how="inner")
    pi = mg["pi"].values
    bo[pi] = mg["bo"].values; o2[pi] = mg["o2"].values; o3[pi] = mg["o3"].values
    del v, mg; gc.collect()
P = pr[picks]; E = ex[picks]
sig = P[:, 0]; bl = P[:, 1]; sh2 = P[:, 3]
h3 = E[:, 0]; c3 = E[:, 1]
sig_dk = dte[picks]
good = (sig > 0) & (bl > 0) & (bo > 0) & np.isfinite(bo) & (o2 > 0) & np.isfinite(o2) & (o3 > 0) & np.isfinite(o3) & (sh2 > 0) & np.isfinite(sh2) & (h3 > 0) & np.isfinite(h3)
n0 = len(picks)
limit = sig * (1 - X); filled = good & (bl <= limit)
entry = np.minimum(limit, bo)
r = np.full(n0, np.nan)
sell_dk = np.full(n0, -1, dtype=np.int64)
# T+2
un = np.isnan(r)
gt = un & (o2 >= entry * 1.02); r[gt] = o2[gt]/entry[gt]-1; sell_dk[gt] = sig_dk[gt]+2
un = np.isnan(r)
th = un & (sh2 >= entry * 1.02); r[th] = 0.02; sell_dk[th] = sig_dk[gt]+2 if False else sig_dk[th]+2
# T+3
un = np.isnan(r)
gt = un & (o3 >= entry * 1.02); r[gt] = o3[gt]/entry[gt]-1; sell_dk[gt] = sig_dk[gt]+3
un = np.isnan(r)
th = un & (h3 >= entry * 1.02); r[th] = 0.02; sell_dk[th] = sig_dk[th]+3
un = np.isnan(r)
r[un] = c3[un]/entry[un]-1; sell_dk[un] = sig_dk[un]+3
r = r - FEE
f = filled
df = pd.DataFrame({"sell_dk": sell_dk[f], "r": r[f]})
df["date"] = pd.to_datetime(df["sell_dk"] * 86400, unit="s")
df["week"] = df["date"].dt.to_period("W").astype(str)
df["month"] = df["date"].dt.to_period("M").astype(str)
w = df.groupby("week")["r"].agg(["sum","count"])
m = df.groupby("month")["r"].agg(["sum","count"])
print(f"总笔数={len(df)}, 总收益={df['r'].sum():+.2%}", flush=True)
print(f"\n按周: 共{len(w)}周, 正收益{int((w['sum']>0).sum())}周 ({(w['sum']>0).mean():.1%}), 负收益{int((w['sum']<0).sum())}周", flush=True)
print(f"  最好一周: {w['sum'].max():+.2%} ({w['sum'].idxmax()}, {int(w.loc[w['sum'].idxmax(),'count'])}笔)", flush=True)
print(f"  最差一周: {w['sum'].min():+.2%} ({w['sum'].idxmin()}, {int(w.loc[w['sum'].idxmin(),'count'])}笔)", flush=True)
print(f"\n按月: 共{len(m)}月, 正收益{int((m['sum']>0).sum())}月 ({(m['sum']>0).mean():.1%}), 负收益{int((m['sum']<0).sum())}月", flush=True)
print(f"  最好一月: {m['sum'].max():+.2%} ({m['sum'].idxmax()}, {int(m.loc[m['sum'].idxmax(),'count'])}笔)", flush=True)
print(f"  最差一月: {m['sum'].min():+.2%} ({m['sum'].idxmin()}, {int(m.loc[m['sum'].idxmin(),'count'])}笔)", flush=True)
# 输出每周明细供检查
w.to_csv(f"{BASE}/weekly_pnl_m2_top3.csv")
m.to_csv(f"{BASE}/monthly_pnl_m2_top3.csv")
print(f"\nDONE {time.time()-t0:.0f}s", flush=True)
