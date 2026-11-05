# EV排序器选型：对每个(x,m)训练 E[r|filled] 回归（train），val 上 Top2 评估选最优(x,m)
import numpy as np, os, json, time, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
from sklearn.ensemble import HistGradientBoostingRegressor
Xtr = np.load(f"{IN}/tr_X.npy").astype(np.float32)
Xva = np.load(f"{IN}/va_X.npy").astype(np.float32)
dva = np.load(f"{PP}/va_day.npy")
print(f"Xtr {Xtr.shape} Xva {Xva.shape}", flush=True)
results = {}
for x in [0.06, 0.08, 0.10]:
    for mm in [2, 3]:
        tag = f"{int(x*100)}_{mm}"
        z = np.load(f"{BASE}/newexec_labels_{tag}.npz")
        ftr = z["tr_f"]; rtr = z["tr_r"].astype(np.float64)
        fva = z["va_f"]; rva = z["va_r"].astype(np.float64)
        reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
            min_samples_leaf=100, early_stopping=False, max_bins=128)
        reg.fit(Xtr[ftr], rtr[ftr])
        pred = reg.predict(Xva).astype(np.float32)
        order = np.lexsort((-pred, dva)); dk = dva[order]
        upos = np.sort(np.unique(dk, return_index=True)[1])
        picks = []
        for i, s in enumerate(upos):
            e = upos[i + 1] if i + 1 < len(upos) else len(order)
            picks.extend(order[s:s + 2])
        picks = np.array(picks)
        rp = rva[picks]; rf = rp[np.isfinite(rp)]
        mean = float(rf.mean()) if len(rf) else 0; se = float(rf.std(ddof=1) / np.sqrt(len(rf))) if len(rf) > 1 else 0
        results[tag] = {"x": x, "m": mm, "val_n_filled": int(len(rf)),
                        "val_mean": mean, "val_lower70": mean - 0.524 * se,
                        "train_n_filled": int(ftr.sum())}
        print(f"[{tag}] train_filled={ftr.sum()} val_filled={len(rf)} val_mean={mean:+.4f} lower70={mean-0.524*se:+.4f}",
              flush=True)
        del z, reg, pred; gc.collect()
best = max(results, key=lambda k: results[k]["val_mean"])
print(f"BEST: {best} {results[best]}", flush=True)
json.dump({"results": results, "best": best}, open(f"{BASE}/ev_select.json", "w"), indent=2)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
