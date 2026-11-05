# T1 step3b: 只训练门控模型（独立进程，重试；step3 中基线已训完，门控因 OOM 中断）
import os, time, json, gc, pickle, numpy as np
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
import importlib.util
spec = importlib.util.spec_from_file_location("strategy_config", os.path.join(BASE, "strategy_config.py"))
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time()
manifest = json.load(open(os.path.join(OUT, "t1_build_manifest.json")))
n_tr = manifest["offsets"]["train"]
yb = np.load(os.path.join(BASE, "daily_base_arrays", "train_y.npy")).astype(np.int64)
Xg = np.empty((n_tr, 79), dtype=np.float32)
o = 0
for fi in range(25):
    c = np.load(os.path.join(CHK, f"t1_X_aug_train_{fi:02d}.npy"))
    Xg[o:o + len(c)] = c; o += len(c); del c
assert o == n_tr
print(f"Xg {Xg.shape} y_pos={yb.mean():.4f}", flush=True)
gc.collect()
t1 = time.time()
m = HistGradientBoostingClassifier(**SC.HGB_CLS)
m.fit(Xg, yb)
p = os.path.join(OUT, "t1_model_gated.pkl")
with open(p, "wb") as f:
    pickle.dump(m, f)
m2 = pickle.load(open(p, "rb"))
assert hasattr(m2, "predict_proba")
del m, m2, Xg, yb; gc.collect()
dt = time.time() - t1
log = json.load(open(os.path.join(OUT, "t1_train_log.json")))
log["gated_time_s"] = round(dt, 1)
log["gated_retry_note"] = "step3 同进程训练门控时 OOM(137)，改独立进程 step3b 重训成功"
log["elapsed_s"] = round(log.get("elapsed_s", 0) + time.time() - t0, 1)
json.dump(log, open(os.path.join(OUT, "t1_train_log.json"), "w"), indent=1)
print(f"TRAIN GATED DONE in {dt:.0f}s", flush=True)
