# 消融：x(w)自适应 + t=0.30常数 vs 全自适应（测试集）
import numpy as np, os, json, pickle, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; S = 0.015
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
pol = json.load(open(f"{BASE}/adaptive_policy.json"))
ki = [FEATS.index(f) for f in pol["key_features"]]
a_raw = np.array(pol["a_raw"]); x0_raw = pol["x0_raw"]
b_raw = np.array(pol["b_raw"]); t0_raw = pol["t0_raw"]
clf = pickle.load(open(f"{BASE}/exec_model_10_30_0.015.pkl", "rb"))
X = np.load(f"{IN}/te_X.npy").astype(np.float32)
pr = np.load(f"{PP}/te_prices.npy"); day = np.load(f"{PP}/te_day.npy")
prob = clf.predict_proba(X)[:, 1].astype(np.float32)
order = np.lexsort((-prob, day)); dk = day[order]
upos = np.sort(np.unique(dk, return_index=True)[1])
picks = []
for i, s in enumerate(upos):
    e = upos[i + 1] if i + 1 < len(upos) else len(order)
    picks.extend(order[s:s + 2])
picks = np.array(picks)
W = X[picks][:, ki].astype(np.float64)
sig, bl, sl, sh, sc = (pr[picks][:, j].astype(np.float64) for j in range(5))
del X, pr, day, prob; gc.collect()
x = np.clip(x0_raw + W @ a_raw, 0.0, 0.12)
for name, t in [("x(w)+t=0.30const", np.full(len(sig), 0.30)),
                ("x(w)+t(w)full", np.clip(t0_raw + W @ b_raw, 0.05, 0.40))]:
    entry = sig * (1 - x); filled = bl <= entry
    slh = sl <= entry * (1 - S); tph = (~slh) & (sh >= entry * (1 + t))
    real = np.where(slh, -S, np.where(tph, t, sc / entry - 1))
    r = real[filled]; n = len(r); mean = r.mean(); se = r.std(ddof=1) / np.sqrt(n)
    print(f"{name}: n={n} mean={mean:+.4f} lower70={mean-0.524*se:+.4f} "
          f"p_profit={(r>0).mean():.3f} stop={slh[filled].mean():.3f} take={tph[filled].mean():.3f} "
          f"t_mean={t[filled].mean():.3f} t_std={t[filled].std():.3f}", flush=True)
print("DONE", flush=True)
