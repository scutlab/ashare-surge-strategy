# x=0新策略回测（Top3，gap-aware，已扣费）
import numpy as np, pandas as pd, os, json, pickle, glob, gc, time, sys
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import strategy_config as SC
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
TOPN = SC.N_STRATEGY; X = 0.05; TAKE = 0.05; FEE = SC.FEE
print(f"TOPN={TOPN}, X={X}, TAKE={TAKE}", flush=True)
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64)
ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
dte = np.load(f"{PP}/te_day.npy"); cte = np.load(f"{PP}/te_code.npy")
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
codes = json.load(open(f"{PP}/codes.json"))["codes"]; cid = {c: i for i, c in enumerate(codes)}
results = []
for tag, mm in [("5t5_2", 2), ("5t5_3", 3)]:
    reg = pickle.load(open(f"{BASE}/ev_model_{tag}.pkl", "rb"))
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
    P = pr[picks]; E = ex[picks]
    sig = P[:, 0]; bl = P[:, 1]
    sh2 = P[:, 3]; h3 = E[:, 0]; c3 = E[:, 1]; h4 = E[:, 2]; c4 = E[:, 3]
    sig_dk = dte[picks]
    if mm == 2:
        days_o = [o2, o3]; days_h = [sh2, h3]; cl = c3
    else:
        days_o = [o2, o3, o4]; days_h = [sh2, h3, h4]; cl = c4
    good = (sig > 0) & (bl > 0) & (bo > 0) & np.isfinite(bo)
    for o, h in zip(days_o, days_h):
        good = good & (o > 0) & (h > 0) & np.isfinite(o) & np.isfinite(h)
    n0 = len(picks)
    limit = sig * (1 - X); filled = good & (bl <= limit)
    entry = np.minimum(limit, bo)
    r = np.full(n0, np.nan)
    for o, h in zip(days_o, days_h):
        un = np.isnan(r)
        gt = un & (o >= entry * (1 + TAKE))
        r[gt] = o[gt] / entry[gt] - 1
        un = np.isnan(r)
        th = un & (h >= entry * (1 + TAKE))
        r[th] = TAKE
    un = np.isnan(r)
    r[un] = cl[un] / entry[un] - 1
    r = r - FEE
    rf = r[filled]; n = len(rf)
    mean = float(rf.mean()); se = float(rf.std(ddof=1) / np.sqrt(n))
    res = {"tag": tag, "m": mm, "topn": TOPN, "n_filled": int(n),
           "mean": round(mean, 4), "lower70": round(mean - 0.524*se, 4),
           "p_profit": round(float((rf > 0).mean()), 4),
           "fill_rate": round(float(filled.mean()), 4)}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    results.append(res)
json.dump(results, open(f"{BASE}/eval_x5t5_top3.json", "w"), ensure_ascii=False, indent=2)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
