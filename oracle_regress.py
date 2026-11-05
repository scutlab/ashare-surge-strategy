# Oracle 第二部分：HGB 回归 best_x ~ w，特征重要性 + 分位数表
import numpy as np, os, json, time, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
rng = np.random.default_rng(0)
Xs, ys, ws = [], [], []
for split in ["tr", "va"]:
    z = np.load(f"{BASE}/oracle_{split}.npz")
    X = np.load(f"{IN}/{split}_X.npy").astype(np.float32)
    sel = z["has_fill"] & (z["wgt"] > 1e-9)
    idx = np.where(sel)[0]
    take = rng.choice(idx, size=min(600_000, len(idx)), replace=False)
    Xs.append(X[take]); ys.append(z["best_xi"][take].astype(np.float32) / 100.0); ws.append(z["wgt"][take])
    del z, X; gc.collect()
X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
del Xs, ys, ws; gc.collect()
print(f"sample {X.shape} wgt mean {w.mean():.4f}", flush=True)
n = len(X); perm = rng.permutation(n); cut = int(n * 0.8)
Xtr, ytr, wtr = X[perm[:cut]], y[perm[:cut]], w[perm[:cut]]
Xva2, yva2, wva2 = X[perm[cut:]], y[perm[cut:]], w[perm[cut:]]
del X, y, w; gc.collect()
from sklearn.ensemble import HistGradientBoostingRegressor
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, early_stopping=False, max_bins=128)
reg.fit(Xtr, ytr, sample_weight=wtr)
pred = reg.predict(Xva2)
ss_res = np.sum(wva2 * (yva2 - pred) ** 2); ss_tot = np.sum(wva2 * (yva2 - np.average(yva2, weights=wva2)) ** 2)
r2 = 1 - ss_res / ss_tot
print(f"weighted R2 on holdout = {r2:.4f}", flush=True)
# permutation importance（100k 样本，2 repeats）
from sklearn.inspection import permutation_importance
pi = permutation_importance(reg, Xva2[:100_000], yva2[:100_000], n_repeats=2,
    random_state=0, scoring="r2", n_jobs=2)
imp = sorted(zip(FEATS, pi.importances_mean), key=lambda t: -t[1])
print("top15 importance:", [(f, round(float(v), 4)) for f, v in imp[:15]], flush=True)
# 分位数表：top6 特征，每档加权平均 best_x
res = {"weighted_R2": float(r2), "importance": [(f, float(v)) for f, v in imp],
       "quintiles": {}, "feat_mean": {}, "feat_std": {}}
Xq = Xva2  # 用验证样本做分位数表
for f, _ in imp[:6]:
    j = FEATS.index(f)
    col = Xq[:, j]
    qs = np.quantile(col, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    tabs = []
    for a in range(5):
        mm = (col >= qs[a]) & (col <= qs[a + 1]) if a == 4 else (col >= qs[a]) & (col < qs[a + 1])
        tabs.append([float(qs[a]), float(qs[a + 1]),
                     float(np.average(yva2[mm], weights=wva2[mm])) if mm.sum() else None])
    res["quintiles"][f] = tabs
for j, f in enumerate(FEATS):
    res["feat_mean"][f] = float(Xtr[:, j].mean()); res["feat_std"][f] = float(Xtr[:, j].std())
json.dump(res, open(f"{BASE}/oracle_regress.json", "w"), indent=2)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
