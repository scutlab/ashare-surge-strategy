# EV终训：在train+val上重训最优(x,m)的EV回归器 + P分类器，测试集对比两种排序（simple口径）
import numpy as np, os, json, time, pickle, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
sel = json.load(open(f"{BASE}/ev_select.json")); tag = sel["best"]
x = sel["results"][tag]["x"]; mm = sel["results"][tag]["m"]
print(f"BEST (x,m)=({x},{mm})", flush=True)
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
Xtr = np.load(f"{IN}/tr_X.npy").astype(np.float32); Xva = np.load(f"{IN}/va_X.npy").astype(np.float32)
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
dte = np.load(f"{PP}/te_day.npy")
z = np.load(f"{BASE}/newexec_labels_{tag}.npz")
Xtv = np.concatenate([Xtr, Xva]); del Xtr, Xva; gc.collect()
f_tv = np.concatenate([z["tr_f"], z["va_f"]]); r_tv = np.concatenate([z["tr_r"], z["va_r"]]).astype(np.float64)
y_tv = np.concatenate([z["tr_y"], z["va_y"]])
r_te = z["te_r"].astype(np.float64); del z; gc.collect()
print(f"train+val filled={f_tv.sum()} pos_rate={y_tv.mean():.4f}", flush=True)
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, early_stopping=False, max_bins=128)
reg.fit(Xtv[f_tv], r_tv[f_tv]); print(f"reg fit {time.time()-t0:.0f}s", flush=True)
pickle.dump(reg, open(f"{BASE}/ev_model_{tag}.pkl", "wb"))
clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
clf.fit(Xtv, y_tv); print(f"clf fit {time.time()-t0:.0f}s", flush=True)
pickle.dump(clf, open(f"{BASE}/p_model_{tag}.pkl", "wb"))
del Xtv, f_tv, r_tv, y_tv; gc.collect()
ev_pred = reg.predict(Xte).astype(np.float32); p_pred = clf.predict_proba(Xte)[:, 1].astype(np.float32)
del Xte; gc.collect()
def eval_rank(name, score):
    order = np.lexsort((-score, dte)); dk = dte[order]
    upos = np.sort(np.unique(dk, return_index=True)[1])
    picks = []
    for i, s in enumerate(upos):
        e = upos[i + 1] if i + 1 < len(upos) else len(order)
        picks.extend(order[s:s + 2])
    picks = np.array(picks)
    rp = r_te[picks]; rf = rp[np.isfinite(rp)]; n = len(rf)
    mean = float(rf.mean()) if n else 0; se = float(rf.std(ddof=1) / np.sqrt(n)) if n > 1 else 0
    take = float(((rf >= 0.0199) & (rf <= 0.0201)).sum() / n) if n else 0
    return {"name": name, "tag": tag, "n_picks": int(len(picks)), "n_filled": int(n),
            "fill_rate": float(n / len(picks)), "mean": mean, "lower70": mean - 0.524 * se,
            "p_profit": float((rf > 0).mean()) if n else 0, "take_share": take}
res = [eval_rank("EV_rank(E[r|fill])", ev_pred), eval_rank("P_rank(P(fill&profit))", p_pred)]
for r_ in res: print(json.dumps(r_, ensure_ascii=False), flush=True)
json.dump(res, open(f"{BASE}/ev_final_simple.json", "w"), indent=2, ensure_ascii=False)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
