# 新执行 gap-aware 模拟（用已存的 picks+opens）
import numpy as np, os, json, time, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
sel = json.load(open(f"{BASE}/ev_select.json")); tag = sel["best"]
x = sel["results"][tag]["x"]; mm = sel["results"][tag]["m"]
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64); ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
results = []
for name in ["ev", "p"]:
    z = np.load(f"{BASE}/newexec_opens_{name}.npz")
    picks = z["picks"]; bo, o2, o3, o4 = (z[k].astype(np.float64) for k in ["bo", "o2", "o3", "o4"])
    P = pr[picks]; sig, bl = P[:, 0], P[:, 1]
    sh2, sc2 = P[:, 3], P[:, 4]
    E = ex[picks]; h3, c3, h4, c4 = E[:, 0], E[:, 1], E[:, 2], E[:, 3]
    if mm == 2: days_o = [o2, o3]; days_h = [sh2, h3]; cl = c3
    else: days_o = [o2, o3, o4]; days_h = [sh2, h3, h4]; cl = c4
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
    results.append({"ranker": name, "tag": tag, "n_picks": int(n0), "n_good": int(good.sum()),
        "n_filled": int(n), "fill_rate": float(n / n0), "mean": mean, "lower70": mean - 0.524 * se,
        "p_profit": float((rf > 0).mean()),
        "gap_take_share": float((exitt[filled] == "gap_take").mean()),
        "take_share": float((exitt[filled] == "take").mean()),
        "buy_gap_improved": float(((bo < limit) & filled).mean() if n else 0)})
    del z; gc.collect()
for r_ in results: print(json.dumps(r_, ensure_ascii=False), flush=True)
json.dump(results, open(f"{BASE}/eval_gap_newexec.json", "w"), indent=2, ensure_ascii=False)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
