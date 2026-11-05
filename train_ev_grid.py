# 网格搜索：训练 take ∈ {2%,3%,4%,5%} 的模型
import numpy as np, os, pickle, gc, time, sys
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import strategy_config as SC
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
from sklearn.ensemble import HistGradientBoostingRegressor
Xtr = np.load(f"{IN}/tr_X.npy").astype(np.float32); Xva = np.load(f"{IN}/va_X.npy").astype(np.float32)
Xtv = np.concatenate([Xtr, Xva]); del Xtr, Xva; gc.collect()
for ttag in [2, 3, 4, 5]:
    for mm in [2, 3]:
        tag = f"5t{ttag}_{mm}"
        if os.path.exists(f"{BASE}/ev_model_{tag}.pkl"):
            print(f"{tag}: 已存在，跳过", flush=True); continue
        t0 = time.time()
        z = np.load(f"{BASE}/newexec_labels_{tag}.npz")
        f_tv = np.concatenate([z["tr_f"], z["va_f"]])
        r_tv = np.concatenate([z["tr_r"], z["va_r"]]).astype(np.float64) - SC.FEE
        del z; gc.collect()
        reg = HistGradientBoostingRegressor(**SC.HGB_REG)
        reg.fit(Xtv[f_tv], r_tv[f_tv])
        pickle.dump(reg, open(f"{BASE}/ev_model_{tag}.pkl", "wb"))
        print(f"{tag}: trained {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
