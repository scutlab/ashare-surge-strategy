# 抽查 gap_take 实例
import numpy as np, os, json, pickle
BASE = '/home/hatch/workspace/ashare'; IN = BASE + '/exec_arrays_10_30_0.015'
PP = BASE + '/price_panel'
sel = json.load(open(f"{BASE}/ev_select.json")); tag = sel["best"]
x = sel["results"][tag]["x"]; mm = sel["results"][tag]["m"]
pr = np.load(f"{PP}/te_prices.npy").astype(np.float64); ex = np.load(f"{PP}/te_ext.npy").astype(np.float64)
codes = json.load(open(f"{PP}/codes.json"))["codes"]
z = np.load(f"{BASE}/newexec_opens_ev.npz")
picks = z["picks"]; bo, o2, o3 = (z[k].astype(np.float64) for k in ["bo", "o2", "o3"])
P = pr[picks]; sig, bl = P[:, 0], P[:, 1]; sh2 = P[:, 3]
E = ex[picks]; h3 = E[:, 0]
cte = np.load(f"{PP}/te_code.npy"); dte = np.load(f"{PP}/te_day.npy")
limit = sig * (1 - x); filled = bl <= limit
entry = np.minimum(limit, bo)
gt = filled & (o2 >= entry * 1.02)
idx = np.where(gt)[0][:3]
for i in idx:
    dstr = str(np.array([dte[picks[i]]], dtype='datetime64[D]')[0])
    print(f"code={codes[cte[picks[i]]]} signal={dstr} sig={sig[i]:.2f} x={x}")
    print(f"  buy: limit={limit[i]:.2f} open={bo[i]:.2f} low={bl[i]:.2f} -> entry={entry[i]:.2f}")
    print(f"  T+2: open={o2[i]:.2f} high={sh2[i]:.2f} | take_px={entry[i]*1.02:.2f}")
    print(f"  -> gap_take exit {(o2[i]/entry[i]-1)*100:+.2f}% (simple口径只记 +2.00%)")
print("n_gap_take:", gt.sum(), "of", filled.sum())
