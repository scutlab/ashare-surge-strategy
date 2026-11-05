# T1 step3c: 训练（50% 系统抽样对照组）
# 背景：4.8M×79 全量训练门控模型两次 OOM（exit=137）；实测 400K 行峰值 610MB，
# 线性外推全量约 7.3GB > 7.7GB 可用内存。改用每 2 行取 1 行的系统抽样（2,403,406 行），
# 两臂（基线/门控）用完全相同的行子集、同标签、同超参，对照依然干净。
# 基线需在相同子集上重训（全量基线模型保留作参考，不用于对照）。
import os, time, json, gc, pickle, numpy as np
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
import importlib.util
spec = importlib.util.spec_from_file_location("strategy_config", os.path.join(BASE, "strategy_config.py"))
SC = importlib.util.module_from_spec(spec); spec.loader.exec_module(SC)
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time(); log = {}

yb = np.load(os.path.join(BASE, "daily_base_arrays", "train_y.npy")).astype(np.int64)
ys = yb[::2].copy(); del yb; gc.collect()
print(f"y_sub {ys.shape} y_pos={ys.mean():.4f}", flush=True)

def train(X, name):
    t1 = time.time()
    m = HistGradientBoostingClassifier(**SC.HGB_CLS)
    m.fit(X, ys)
    p = os.path.join(OUT, name)
    with open(p, "wb") as f:
        pickle.dump(m, f)
    m2 = pickle.load(open(p, "rb"))
    assert hasattr(m2, "predict_proba")
    del m, m2; gc.collect()
    dt = time.time() - t1
    print(f"{name}: done in {dt:.0f}s", flush=True)
    return round(dt, 1)

# --- 基线臂（相同子集重训） ---
Xb = np.load(os.path.join(BASE, "daily_base_arrays", "train_X.npy"))
Xbs = Xb[::2].copy(); del Xb; gc.collect()
log["baseline_sub_time_s"] = train(Xbs, "t1_model_baseline_sub.pkl")
del Xbs; gc.collect()

# --- 门控臂（相同子集） ---
manifest = json.load(open(os.path.join(OUT, "t1_build_manifest.json")))
n_tr = manifest["offsets"]["train"]
Xg = np.empty((n_tr, 79), dtype=np.float32)
o = 0
for fi in range(25):
    c = np.load(os.path.join(CHK, f"t1_X_aug_train_{fi:02d}.npy"))
    Xg[o:o + len(c)] = c; o += len(c); del c
Xgs = Xg[::2].copy(); del Xg; gc.collect()
print(f"Xgs {Xgs.shape}", flush=True)
log["gated_sub_time_s"] = train(Xgs, "t1_model_gated_sub.pkl")
del Xgs; gc.collect()

log["hgb_params"] = SC.HGB_CLS
log["n_train_sub"] = int(len(ys))
log["subsample"] = "every 2nd row (systematic 50%), identical rows for both arms"
log["elapsed_s"] = round(time.time() - t0, 1)
json.dump(log, open(os.path.join(OUT, "t1_train_log.json"), "w"), indent=1)
print("TRAIN SUB DONE", log, flush=True)
