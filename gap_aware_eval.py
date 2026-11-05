# Gap-aware 真实执行评估：补开盘价，跳空穿过止损/止盈按开盘价成交
# 买入：limit=sig*(1-x)；若buy_low<=limit则成交，成交价=min(limit, buy_open)（跳空低开以开盘价成交，更优）
# 卖出：stop=entry*0.985, take=entry*(1+t)
#   sell_open<=stop -> 按开盘价止损（跳空穿止损）；elif sell_low<=stop -> -1.5%；
#   sell_open>=take -> 按开盘价止盈（跳空穿止盈）；elif sell_high>=take -> +t；else 收盘卖出
# 同时过滤脏数据（任一价格<=0或NaN）
import numpy as np, pandas as pd, os, json, pickle, glob, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; S = float(os.environ.get("STOP_S", "0.015")); t0 = __import__("time").time()
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
pol = json.load(open(f"{BASE}/adaptive_policy.json"))
ki = [FEATS.index(f) for f in pol["key_features"]]
a_raw = np.array(pol["a_raw"]); x0_raw = pol["x0_raw"]
b_raw = np.array(pol["b_raw"]); t0_raw = pol["t0_raw"]
codes = json.load(open(f"{PP}/codes.json"))["codes"]
cid = {c: i for i, c in enumerate(codes)}
clf = pickle.load(open(f"{BASE}/exec_model_10_30_0.015.pkl", "rb"))
X = np.load(f"{IN}/te_X.npy").astype(np.float32)
pr = np.load(f"{PP}/te_prices.npy"); day = np.load(f"{PP}/te_day.npy"); cc = np.load(f"{PP}/te_code.npy")
prob = clf.predict_proba(X)[:, 1].astype(np.float32)
order = np.lexsort((-prob, day)); dk = day[order]
upos = np.sort(np.unique(dk, return_index=True)[1])
picks = []
for i, s in enumerate(upos):
    e = upos[i + 1] if i + 1 < len(upos) else len(order)
    picks.extend(order[s:s + 2])
picks = np.array(picks)
W = X[picks][:, ki].astype(np.float64)
P = pr[picks]; sig, bl, sl, sh, sc = (P[:, j].astype(np.float64) for j in range(5))
pc = cc[picks]; pd_ = day[picks]
del X, pr, day, cc, prob; gc.collect()
print(f"picks={len(picks)}", flush=True)
# ---- 从 v3 批次取 buy_open / sell_open（adj）----
need = pd.DataFrame({"ci": pc.astype(np.int32), "dk": pd_.astype(np.int32),
                     "pi": np.arange(len(picks))})
buyo = np.full(len(picks), np.nan); sello = np.full(len(picks), np.nan)
for vf in sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl")):
    v = pd.read_pickle(vf)[["code", "date", "adj_open"]]
    v["ci"] = v["code"].map(cid).astype(np.int32)
    v["dk"] = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    v = v.sort_values(["ci", "dk"]).reset_index(drop=True)
    g = v.groupby("ci")["adj_open"]
    v["buy_open"] = g.shift(-1); v["sell_open"] = g.shift(-2)
    mg = need.merge(v[["ci", "dk", "buy_open", "sell_open"]], on=["ci", "dk"], how="inner")
    buyo[mg["pi"].values] = mg["buy_open"].values
    sello[mg["pi"].values] = mg["sell_open"].values
    del v, mg; gc.collect()
n_miss = int(np.isnan(buyo).sum())
print(f"opens matched {len(picks)-n_miss}/{len(picks)} miss={n_miss}", flush=True)
buyo = buyo.astype(np.float64); sello = sello.astype(np.float64)
# ---- 脏数据过滤 ----
good = (sig > 0) & (bl > 0) & (sl > 0) & (sh > 0) & (sc > 0) & (buyo > 0) & (sello > 0) \
    & np.isfinite(sig) & np.isfinite(buyo) & np.isfinite(sello)
print(f"good={good.sum()}/{len(picks)} bad={(~good).sum()}", flush=True)
idx = np.where(good)[0]
sig, bl, sl, sh, sc, buyo, sello = sig[idx], bl[idx], sl[idx], sh[idx], sc[idx], buyo[idx], sello[idx]
W = W[idx]
xA = np.clip(x0_raw + W @ a_raw, 0.0, 0.12); tA = np.clip(t0_raw + W @ b_raw, 0.05, 0.40)

def evaluate(name, x, t):
    limit = sig * (1 - x)
    filled = bl <= limit
    entry = np.where(filled, np.minimum(limit, buyo), np.nan)
    stop_px = entry * (1 - S); take_px = entry * (1 + t)
    gap_stop = filled & (sello <= stop_px)
    stop = filled & (~gap_stop) & (sl <= stop_px)
    gap_take = filled & (~gap_stop) & (~stop) & (sello >= take_px)
    take = filled & (~gap_stop) & (~stop) & (~gap_take) & (sh >= take_px)
    close = filled & (~gap_stop) & (~stop) & (~gap_take) & (~take)
    real = np.full(len(sig), np.nan)
    real[gap_stop] = sello[gap_stop] / entry[gap_stop] - 1
    real[stop] = -S
    real[gap_take] = sello[gap_take] / entry[gap_take] - 1
    real[take] = t[take]
    real[close] = sc[close] / entry[close] - 1
    r = real[filled]; n = len(r); mean = float(r.mean()); se = float(r.std(ddof=1) / np.sqrt(n))
    # 原模拟口径（同一样本，用于公平对比）
    e2 = limit; f2 = filled
    s2 = sl <= e2 * (1 - S); p2 = (~s2) & (sh >= e2 * (1 + t))
    r2v = np.where(s2, -S, np.where(p2, t, sc / e2 - 1))[f2]
    m2 = float(r2v.mean()); se2 = float(r2v.std(ddof=1) / np.sqrt(len(r2v)))
    return {"name": name, "n_picks": int(len(sig)), "n_filled": int(n),
        "gap_aware_mean": mean, "gap_aware_lower70": mean - 0.524 * se,
        "sim_mean_same_sample": m2, "sim_lower70_same_sample": m2 - 0.524 * se2,
        "p_profit": float((r > 0).mean()),
        "gap_stop_share": float(gap_stop.sum() / n), "stop_share": float(stop.sum() / n),
        "gap_take_share": float(gap_take.sum() / n), "take_share": float(take.sum() / n),
        "close_share": float(close.sum() / n),
        "buy_gap_improved": float(((buyo < limit) & filled).sum() / n)}

results = [evaluate("adaptive_x(w)_t(w)", xA, tA),
           evaluate("const_x0.10_t0.30", np.full(len(sig), 0.10), np.full(len(sig), 0.30))]
for r_ in results:
    print(json.dumps(r_, ensure_ascii=False), flush=True)
json.dump(results, open(f"{BASE}/gap_aware_eval_S{S}.json", "w"), indent=2, ensure_ascii=False)
print(f"DONE {__import__('time').time()-t0:.0f}s", flush=True)
