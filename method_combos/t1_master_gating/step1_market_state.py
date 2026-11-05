# T1 step1: 构建市场状态向量（MASTER 式）
# 价格：上证指数（index_sse.pkl，与项目现有 mkt_ret 同源）过去 5/10/20/30/60 日收益均值/标准差
# 成交量：上证指数无成交量数据，改用全市场等权聚合（25 个 v3 批次的日均成交量取 log）
#   过去 5/10/20/30/60 日均值/标准差。共 20 维 + 门控标量 mkt_gate。
# 门控：g = 1 + tanh(z)，z 为 mkt_px_mean_20 的 expanding z-score（min_periods=60）
# 输出：market_state.pkl（date -> 21 列），market_daily.csv（检查用）
import os, glob, pickle, time, numpy as np, pandas as pd
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()

# --- 价格：上证指数 ---
sse = pickle.load(open(os.path.join(BASE, "index_sse.pkl"), "rb"))
sse["date"] = pd.to_datetime(sse["date"]).dt.normalize()
sse = sse.sort_values("date").reset_index(drop=True)
sse["px_ret"] = sse["mkt_adjclose"].pct_change()
print(f"sse rows {len(sse)} range {sse['date'].min().date()}~{sse['date'].max().date()}", flush=True)

# --- 成交量：全市场等权日均（v3 批次聚合，不重拉数据） ---
v3_files = sorted(glob.glob(os.path.join(BASE, "full_features_batch_v3_*.pkl")))
assert len(v3_files) == 25, f"v3 files {len(v3_files)}"
acc = {}
for fi, vf in enumerate(v3_files):
    v = pd.read_pickle(vf)[["date", "volume"]]
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    g = v.groupby("date")["volume"].agg(["sum", "count"])
    for d, row in g.iterrows():
        a = acc.get(d, [0.0, 0])
        a[0] += float(row["sum"]); a[1] += int(row["count"]); acc[d] = a
    del v, g
    if fi % 5 == 0: print(f"  vol agg {fi+1}/25 {time.time()-t0:.0f}s", flush=True)
vol = pd.DataFrame([{"date": d, "mkt_vol": s / max(c, 1)} for d, (s, c) in acc.items()])
vol = vol.sort_values("date").reset_index(drop=True)
print(f"vol days {len(vol)} range {vol['date'].min().date()}~{vol['date'].max().date()}", flush=True)

# --- 合并为交易日历 ---
mkt = sse[["date", "px_ret"]].merge(vol, on="date", how="inner").sort_values("date").reset_index(drop=True)
mkt["log_vol"] = np.log(mkt["mkt_vol"].clip(lower=1.0))
print(f"market days {len(mkt)} (inner join)", flush=True)

WINS = [5, 10, 20, 30, 60]
for w in WINS:
    mkt[f"mkt_px_mean_{w}"] = mkt["px_ret"].rolling(w, min_periods=1).mean()
    mkt[f"mkt_px_std_{w}"] = mkt["px_ret"].rolling(w, min_periods=2).std().fillna(0.0)
    mkt[f"mkt_vol_mean_{w}"] = mkt["log_vol"].rolling(w, min_periods=1).mean()
    mkt[f"mkt_vol_std_{w}"] = mkt["log_vol"].rolling(w, min_periods=2).std().fillna(0.0)

# --- 门控标量：mkt_px_mean_20 的 expanding z-score ---
mu = mkt["mkt_px_mean_20"].expanding(min_periods=60).mean()
sd = mkt["mkt_px_mean_20"].expanding(min_periods=60).std().fillna(0.0).replace(0.0, np.nan)
z = (mkt["mkt_px_mean_20"] - mu) / sd
z = z.fillna(0.0).clip(-5, 5)
mkt["mkt_gate"] = 1.0 + np.tanh(z)
print(f"gate describe: min={mkt['mkt_gate'].min():.3f} max={mkt['mkt_gate'].max():.3f} mean={mkt['mkt_gate'].mean():.3f}", flush=True)

state_cols = ([f"mkt_px_mean_{w}" for w in WINS] + [f"mkt_px_std_{w}" for w in WINS] +
              [f"mkt_vol_mean_{w}" for w in WINS] + [f"mkt_vol_std_{w}" for w in WINS] + ["mkt_gate"])
st = mkt[["date"] + state_cols].copy()
assert not st[state_cols].isna().any().any(), "NaN in market state!"
st.to_pickle(os.path.join(OUT, "market_state.pkl"))
mkt[["date", "px_ret", "mkt_vol"] + state_cols].to_csv(os.path.join(OUT, "market_daily.csv"), index=False)
print(f"DONE market_state {st.shape} -> market_state.pkl, elapsed {time.time()-t0:.0f}s", flush=True)
print("NaN check passed; date range:", st["date"].min().date(), "~", st["date"].max().date())
