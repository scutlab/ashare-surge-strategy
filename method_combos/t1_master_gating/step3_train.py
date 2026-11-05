# T1 step3: HGB 训练（基线 29 特征 vs 门控 79 特征，同超参、同标签、同行）
# 超参：strategy_config.HGB_CLS（300 树 / lr 0.05 / depth 3 / leaf 100 / balanced / 128 bins）
import os, time, json, gc, pickle, numpy as np
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
import importlib.util
spec = importlib.util.spec_from_file_location("strategy_config", os.path.join(BASE, "strategy_config.py"))
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time(); log = {}

manifest = json.load(open(os.path.join(OUT, "t1_build_manifest.json")))
n_tr = manifest["offsets"]["train"]
print(f"n_train {n_tr}", flush=True)

# --- 基线（现有基数组，原样只读） ---
Xb = np.load(os.path.join(BASE, "daily_base_arrays", "train_X.npy"))
yb = np.load(os.path.join(BASE, "daily_base_arrays", "train_y.npy")).astype(np.int64)
print(f"baseline X {Xb.shape} {Xb.dtype} y_pos={yb.mean():.4f}", flush=True)

# --- 门控：拼接 tr chunks（预分配，避免 concat 峰值翻倍） ---
Xg = np.empty((n_tr, 79), dtype=np.float32)
o = 0
for fi in range(25):
    c = np.load(os.path.join(CHK, f"t1_X_aug_train_{fi:02d}.npy"))
    Xg[o:o + len(c)] = c; o += len(c); del c
assert o == n_tr
assert (Xg[:, :29] == Xb).all(), "gated first-29 cols != baseline!"
print("gated X assembled, first-29 verified equal to baseline", flush=True)
del Xb; gc.collect()

def train(X, y, name):
    t1 = time.time()
    m = HistGradientBoostingClassifier(**SC.HGB_CLS)
    m.fit(X, y)
    p = os.path.join(OUT, name)
    with open(p, "wb") as f:
        pickle.dump(m, f)
    # 加载验证
    m2 = pickle.load(open(p, "rb"))
    assert hasattr(m2, "predict_proba")
    dt = time.time() - t1
    print(f"{name}: trained+saved+reloaded in {dt:.0f}s", flush=True)
    return dt

log["hgb_params"] = SC.HGB_CLS
log["n_train"] = int(n_tr)
log["baseline_time_s"] = round(train(np.load(os.path.join(BASE, "daily_base_arrays", "train_X.npy")), yb, "t1_model_baseline.pkl"), 1)
gc.collect()
log["gated_time_s"] = round(train(Xg, yb, "t1_model_gated.pkl"), 1)
log["elapsed_s"] = round(time.time() - t0, 1)
json.dump(log, open(os.path.join(OUT, "t1_train_log.json"), "w"), indent=1)
print("TRAIN DONE", log, flush=True)
