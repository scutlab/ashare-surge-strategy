# 训练 x=0 的 EV 模型（m=2/m=3，+5%止盈）
import numpy as np, os, pickle, gc, time, sys
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import strategy_config as SC
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
from sklearn.ensemble import HistGradientBoostingRegressor
Xtr = np.load(f"{IN}/tr_X.npy").astype(np.float32); Xva = np.load(f"{IN}/va_X.npy").astype(np.float32)
Xtv = np.concatenate([Xtr, Xva]); del Xtr, Xva; gc.collect()
for tag in ["5_2", "5_3"]:
    t0 = time.time()
    z = np.load(f"{BASE}/newexec_labels_{tag}.npz")
    f_tv = np.concatenate([z["tr_f"], z["va_f"]])
    r_tv = np.concatenate([z["tr_r"], z["va_r"]]).astype(np.float64) - SC.FEE
    del z; gc.collect()
    reg = HistGradientBoostingRegressor(**SC.HGB_REG)
    reg.fit(Xtv[f_tv], r_tv[f_tv])
    pickle.dump(reg, open(f"{BASE}/ev_model_{tag}.pkl", "wb"))
    print(f"{tag}: trained, {time.time()-t0:.0f}s (n_filled={f_tv.sum()})", flush=True)
print("DONE", flush=True)
