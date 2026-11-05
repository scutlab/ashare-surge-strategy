# 训练 m=3 EV 排序器（train+val），与 ev_model_10_2.pkl 同口径
import numpy as np, os, time, pickle, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
t0 = time.time()
from sklearn.ensemble import HistGradientBoostingRegressor
Xtr = np.load(f"{IN}/tr_X.npy").astype(np.float32); Xva = np.load(f"{IN}/va_X.npy").astype(np.float32)
z = np.load(f"{BASE}/newexec_labels_10_3.npz")
Xtv = np.concatenate([Xtr, Xva]); del Xtr, Xva; gc.collect()
f_tv = z["tr_f"]; f_tv = np.concatenate([f_tv, z["va_f"]])
r_tv = np.concatenate([z["tr_r"], z["va_r"]]).astype(np.float64); del z; gc.collect()
print(f"filled={f_tv.sum()}", flush=True)
reg = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, early_stopping=False, max_bins=128)
reg.fit(Xtv[f_tv], r_tv[f_tv]); print(f"fit {time.time()-t0:.0f}s", flush=True)
pickle.dump(reg, open(f"{BASE}/ev_model_10_3.pkl", "wb"))
print("saved ev_model_10_3.pkl", flush=True)
