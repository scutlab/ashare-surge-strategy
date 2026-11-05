# 新执行标签：x=0.05（8折买入）、+2%止盈、无止损、m=2/3
import numpy as np, os, time, gc
BASE = os.path.expanduser("~/workspace/ashare"); PP = BASE + "/price_panel"; t0 = time.time()
X = 0.05; TAKE = 0.05; MS = [2, 3]
data = {}
for s_ in ["tr", "va", "te"]:
    pr = np.load(f"{PP}/{s_}_prices.npy").astype(np.float64)
    ex = np.load(f"{PP}/{s_}_ext.npy").astype(np.float64)
    data[s_] = {"sig": pr[:, 0], "bl": pr[:, 1], "sh2": pr[:, 3], "sc2": pr[:, 4],
                "h3": ex[:, 0], "c3": ex[:, 1], "h4": ex[:, 2], "c4": ex[:, 3], "n": len(pr)}
    del pr, ex; gc.collect()
    print(f"{s_}: n={data[s_]['n']}", flush=True)
for mm in MS:
    tag = f"5t5_{mm}"
    out = {}
    for s_, d in data.items():
        sig, bl = d["sig"], d["bl"]
        limit = sig * (1 - X)
        filled = (bl <= limit) & (sig > 0) & (bl > 0)
        entry = limit
        if mm == 2:
            highs = [d["sh2"], d["h3"]]; closes_last = d["c3"]
            ok = np.isfinite(d["h3"]) & np.isfinite(d["c3"]) & (d["h3"] > 0) & (d["c3"] > 0)
        else:
            highs = [d["sh2"], d["h3"], d["h4"]]; closes_last = d["c4"]
            ok = np.isfinite(d["h3"]) & np.isfinite(d["c3"]) & np.isfinite(d["h4"]) & np.isfinite(d["c4"]) \
                & (d["h3"] > 0) & (d["c3"] > 0) & (d["h4"] > 0) & (d["c4"] > 0)
        ok = ok & np.isfinite(d["sh2"]) & (d["sh2"] > 0) & np.isfinite(d["sc2"]) & (d["sc2"] > 0)
        filled = filled & ok
        take = np.zeros(d["n"], dtype=bool)
        for h in highs:
            take = take | (h >= entry * (1 + TAKE))
        take = take & filled
        r = np.full(d["n"], np.nan)
        r[take] = TAKE
        rest = filled & (~take)
        r[rest] = closes_last[rest] / entry[rest] - 1
        y = (filled & (r > 0)).astype(np.int8)
        out[f"{s_}_y"] = y; out[f"{s_}_r"] = r.astype(np.float32); out[f"{s_}_f"] = filled
        rf = r[filled]
        print(f"{tag} {s_}: fill={filled.mean():.4f}, mean_r={rf.mean():+.4f} (n={len(rf)})", flush=True)
    np.savez_compressed(f"{BASE}/newexec_labels_{tag}.npz", **out)
    print(f"saved newexec_labels_{tag}.npz", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
