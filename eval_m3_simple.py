# m=3 在测试集上的快速评估（simple口径）
import numpy as np, os, pickle, json, time
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"; PP = BASE + "/price_panel"
t0 = time.time()
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
z = np.load(f"{BASE}/newexec_labels_10_3.npz")
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64); ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
ev3 = pickle.load(open(f"{BASE}/ev_model_10_3.pkl", "rb"))
s = ev3.predict(Xte); print(f"predict {time.time()-t0:.0f}s", flush=True)
order = np.argsort(-s, kind="stable")
dte = np.load(f"{PP}/te_day.npy"); ud = np.unique(dte); ud = ud[ud > 0]
picks = []
for d in ud:
    idx = np.where(dte == d)[0]
    ii = idx[np.argsort(-s[idx], kind="stable")[:2]]
    picks.extend(ii.tolist())
picks = np.array(picks)
P = pr[picks]; E = ex[picks]
sig = P[:, 0]; bl = P[:, 1]; sh2 = P[:, 3]; sc2 = P[:, 4]; h3 = E[:, 0]; c3 = E[:, 1]; h4 = E[:, 2]; c4 = E[:, 3]
x = 0.10; limit = sig * 0.90; filled = bl <= limit
entry = limit
r = np.full(len(picks), np.nan)
t2 = sh2 >= entry * 1.02
r[t2] = 0.02
nt = ~t2
t3 = nt & (h3 >= entry * 1.02)
r[t3] = 0.02
nt2 = nt & ~t3
t4 = nt2 & (h4 >= entry * 1.02)
r[t4] = 0.02
rest = nt2 & ~t4
r[rest] = c4[rest] / entry[rest] - 1
rf = r[filled]
mean = rf.mean(); se = rf.std(ddof=1) / np.sqrt(len(rf)); lower = mean - 0.524 * se
print(json.dumps({"tag": "m3_simple", "n_picks": len(picks), "n_filled": int(filled.sum()),
                  "mean": round(float(mean), 4), "lower70": round(float(lower), 4),
                  "p_profit": round(float((rf > 0).mean()), 4)}, ensure_ascii=False))
