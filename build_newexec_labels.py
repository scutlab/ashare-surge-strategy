# 新执行标签（m天持有、+2%止盈、无止损、到期收盘卖；simple口径，不用开盘价，偏保守）
# 时间线：T信号 -> T+1买入(limit=sig*(1-x), buy_low<=limit成交, entry=limit)
#        -> T+2..T+1+m 持有：任一日 high>=entry*1.02 则 +2% 止盈；否则最后收盘卖出
# y = filled & (r>0)；r=NaN 表示未成交或数据缺失
import numpy as np, os, json, time, gc
BASE = os.path.expanduser("~/workspace/ashare"); PP = BASE + "/price_panel"; t0 = time.time()
XS = [0.06, 0.08, 0.10]; MS = [2, 3]
data = {}
for s_ in ["tr", "va", "te"]:
    pr = np.load(f"{PP}/{s_}_prices.npy").astype(np.float64)  # sig,bl,sl,sh,sc
    ex = np.load(f"{PP}/{s_}_ext.npy").astype(np.float64)     # h3,c3,h4,c4
    data[s_] = {"sig": pr[:, 0], "bl": pr[:, 1], "sh2": pr[:, 3], "sc2": pr[:, 4],
                "h3": ex[:, 0], "c3": ex[:, 1], "h4": ex[:, 2], "c4": ex[:, 3], "n": len(pr)}
    del pr, ex; gc.collect()
    print(f"{s_}: n={data[s_]['n']}", flush=True)
summary = {}
for x in XS:
    for mm in MS:
        out = {}; tag = f"{int(x*100)}_{mm}"
        for s_, d in data.items():
            sig, bl = d["sig"], d["bl"]
            limit = sig * (1 - x)
            filled = (bl <= limit) & (sig > 0) & (bl > 0)
            entry = limit
            # holding highs/closes
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
                take = take | (h >= entry * 1.02)
            take = take & filled
            r = np.full(d["n"], np.nan)
            r[take] = 0.02
            rest = filled & (~take)
            r[rest] = closes_last[rest] / entry[rest] - 1
            y = (filled & (r > 0)).astype(np.int8)
            out[f"{s_}_y"] = y; out[f"{s_}_r"] = r.astype(np.float32); out[f"{s_}_f"] = filled
            rf = r[filled]
            summary[tag + "_" + s_] = {"fill": float(filled.mean()),
                "p_profit_filled": float((rf > 0).mean()) if len(rf) else 0,
                "mean_r_filled": float(rf.mean()) if len(rf) else 0,
                "take_share": float(take.sum() / max(1, filled.sum()))}
        np.savez_compressed(f"{BASE}/newexec_labels_{tag}.npz", **out)
        print(f"x={x} m={mm}: " + " ".join(
            f"{s_} fill={summary[tag+'_'+s_]['fill']:.3f} E[r|fill]={summary[tag+'_'+s_]['mean_r_filled']:+.4f}"
            for s_ in ["tr", "va", "te"]), flush=True)
json.dump(summary, open(f"{BASE}/newexec_labels_summary.json", "w"), indent=2)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
