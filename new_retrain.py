# 新策略最终模型：训练+验证合并重训（测试集不碰）
# 输出 new_model_HGB_{wd,nowd}_trainval.pkl + new_retrain_{wd,nowd}.json
# 内存注意：分批读、只保留训练+验证行、每批释放；原 train-only 模型文件不动
import pandas as pd, numpy as np, json, os, glob, time, gc, sys, pickle
BASE = os.path.expanduser("~/workspace/ashare")
import importlib.util
spec = importlib.util.spec_from_file_location("full_train", os.path.join(BASE, "full_train.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS = mod.FEATS; FEATS_NO_WD = [f for f in FEATS if f != 'weekday']
name = sys.argv[1]  # HGB_wd | HGB_nowd
feats = FEATS if name == 'HGB_wd' else FEATS_NO_WD
t0 = time.time()
files = sorted(glob.glob(os.path.join(BASE, "full_flow_new_*.pkl")))
print(f"retrain {name} feats {len(feats)} batches {len(files)}", flush=True)
# pass 1: codes
all_codes = set()
for f in files:
    d = pd.read_pickle(f)
    all_codes.update(d['code'].unique())
    del d
codes = sorted(all_codes)
cmap = {c: i % 3 for i, c in enumerate(codes)}
print(f"codes {len(codes)} elapsed {time.time()-t0:.0f}s", flush=True)
# pass 2: accumulate train+val only
Xs, ys = [], []
n_tr = n_va = 0
for fi, f in enumerate(files):
    d = pd.read_pickle(f)
    sp = d['code'].map(cmap).values
    X = d[feats].to_numpy(dtype=np.float32)
    y = d['y_strategy'].to_numpy(dtype=np.int8)
    for s in (0, 1):
        m = sp == s
        if m.any():
            Xs.append(X[m]); ys.append(y[m])
            if s == 0: n_tr += int(m.sum())
            else: n_va += int(m.sum())
    del d, sp, X, y; gc.collect()
    print(f"batch {fi} train {n_tr} val {n_va} elapsed {time.time()-t0:.0f}s", flush=True)
Xall = np.concatenate(Xs); yall = np.concatenate(ys)
del Xs, ys; gc.collect()
print(f"stacked {len(yall)} win_rate {yall.mean():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
    min_samples_leaf=100, class_weight='balanced', random_state=42,
    early_stopping=False, max_bins=128)
m.fit(Xall, yall)
print(f"fitted elapsed {time.time()-t0:.0f}s", flush=True)
del Xall, yall; gc.collect()
out_model = os.path.join(BASE, f"new_model_{name}_trainval.pkl")
pickle.dump(m, open(out_model, "wb"))
res = {'model': name, 'train_plus_val': True, 'n_train': n_tr, 'n_val': n_va,
       'n_total': n_tr + n_va, 'fit_seconds': round(time.time() - t0, 1),
       'features': feats, 'target': 'y_strategy'}
with open(os.path.join(BASE, f"new_retrain_{name}.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True)
print("RETRAIN_DONE", flush=True)
