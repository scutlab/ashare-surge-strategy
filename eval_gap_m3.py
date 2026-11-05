# m=3 EV gap-aware 终评（与 eval_gap_newexec.py 同口径，m=3）
# 买入：limit=sig*(1-x)，x=0.10；buy_low<=limit成交，entry=min(limit,buy_open)
# 持有至多3天：open>=entry*1.02 -> 按开盘价止盈；elif high>=entry*1.02 -> +2%；到期收盘卖出；无止损
# P基线跳过（p_model_10_3.pkl 不存在，未训练；EV模型 ev_model_10_3.pkl 已有）
import numpy as np, pandas as pd, os, json, pickle, glob, gc, time
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
tag, x, mm = "10_3", 0.10, 3
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64); ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
dte = np.load(f"{PP}/te_day.npy"); cte = np.load(f"{PP}/te_code.npy")
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
reg = pickle.load(open(f"{BASE}/ev_model_{tag}.pkl", "rb"))
ev_pred = reg.predict(Xte).astype(np.float32); del Xte, reg; gc.collect()
def top2(score):
    order = np.lexsort((-score, dte)); dk = dte[order]
    upos = np.sort(np.unique(dk, return_index=True)[1])
    picks = []
    for i, s in enumerate(upos):
        e = upos[i + 1] if i + 1 < len(upos) else len(order)
        picks.extend(order[s:s + 2])
    return np.array(picks)
picks = top2(ev_pred); print(f"picks ev={len(picks)}", flush=True)
codes = json.load(open(f"{PP}/codes.json"))["codes"]; cid = {c: i for i, c in enumerate(codes)}
need = pd.DataFrame({"ci": cte[picks].astype(np.int32), "dk": dte[picks].astype(np.int32),
                     "pi": np.arange(len(picks))})
bo = np.full(len(picks), np.nan); o2 = np.full(len(picks), np.nan)
o3 = np.full(len(picks), np.nan); o4 = np.full(len(picks), np.nan)
for vf in sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl")):
    v = pd.read_pickle(vf)[["code", "date", "adj_open"]]
    v["ci"] = v["code"].map(cid).astype(np.int32)
    v["dk"] = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    v = v.sort_values(["ci", "dk"]).reset_index(drop=True)
    g = v.groupby("ci")["adj_open"]
    v["bo"] = g.shift(-1); v["o2"] = g.shift(-2); v["o3"] = g.shift(-3); v["o4"] = g.shift(-4)
    mg = need.merge(v[["ci", "dk", "bo", "o2", "o3", "o4"]], on=["ci", "dk"], how="inner")
    pi = mg["pi"].values
    bo[pi] = mg["bo"].values; o2[pi] = mg["o2"].values; o3[pi] = mg["o3"].values; o4[pi] = mg["o4"].values
    del v, mg; gc.collect()
np.savez_compressed(f"{BASE}/newexec_opens_ev_{tag}.npz", bo=bo, o2=o2, o3=o3, o4=o4, picks=picks)
print(f"opens miss={int(np.isnan(bo).sum())}/{len(picks)}", flush=True)
P = pr[picks]; sig, bl = P[:, 0], P[:, 1]
sh2, sc2 = P[:, 3], P[:, 4]
E = ex[picks]; h3, c3, h4, c4 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
days_o = [o2, o3, o4]; days_h = [sh2, h3, h4]; cl = c4
good = (sig > 0) & (bl > 0) & (bo > 0) & (cl > 0)
for o, h in zip(days_o, days_h): good = good & (o > 0) & (h > 0) & np.isfinite(o) & np.isfinite(h)
good = good & np.isfinite(sig) & np.isfinite(bo) & np.isfinite(cl)
n0 = len(picks)
sig, bl, bo, cl = sig[good], bl[good], bo[good], cl[good]
days_o = [o[good] for o in days_o]; days_h = [h[good] for h in days_h]
limit = sig * (1 - x); filled = bl <= limit
entry = np.minimum(limit, bo)
r = np.full(len(sig), np.nan); exitt = np.full(len(sig), "", dtype=object)
rem = filled.copy()
for o, h in zip(days_o, days_h):
    gt = rem & (o >= entry * 1.02)
    r[gt] = o[gt] / entry[gt] - 1; exitt[gt] = "gap_take"; rem = rem & (~gt)
    tk = rem & (h >= entry * 1.02)
    r[tk] = 0.02; exitt[tk] = "take"; rem = rem & (~tk)
r[rem] = cl[rem] / entry[rem] - 1; exitt[rem] = "close"
rf = r[filled]; n = len(rf); mean = float(rf.mean()); se = float(rf.std(ddof=1) / np.sqrt(n))
res = {"ranker": "ev", "tag": tag, "x": x, "m": mm, "n_picks": int(n0), "n_good": int(good.sum()),
    "n_filled": int(n), "fill_rate": float(n / n0), "mean": mean, "lower70": mean - 0.524 * se,
    "p_profit": float((rf > 0).mean()),
    "gap_take_share": float((exitt[filled] == "gap_take").mean()),
    "take_share": float((exitt[filled] == "take").mean()),
    "buy_gap_improved": float(((bo < limit) & filled).mean() if n else 0)}
print(json.dumps(res, ensure_ascii=False), flush=True)
json.dump(res, open(f"{BASE}/eval_gap_m3.json", "w"), indent=2, ensure_ascii=False)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
