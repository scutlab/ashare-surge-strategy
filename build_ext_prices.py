# 扩展价格面板：为 price_panel 的每行补 T+3/T+4 的 high/close（adj），用于 m=2,3 新执行标签
# 行序与 price_panel/{tr,va,te}_prices.npy 完全一致；缺失记 NaN
import numpy as np, pandas as pd, os, json, glob, gc, time
BASE = os.path.expanduser("~/workspace/ashare"); PP = BASE + "/price_panel"; t0 = time.time()
codes = json.load(open(f"{PP}/codes.json"))["codes"]; cid = {c: i for i, c in enumerate(codes)}
splits = {}
for s_ in ["tr", "va", "te"]:
    ci = np.load(f"{PP}/{s_}_code.npy").astype(np.int32)
    dk = np.load(f"{PP}/{s_}_day.npy").astype(np.int32)
    splits[s_] = {"ci": ci, "dk": dk, "n": len(ci),
                  "h3": np.full(len(ci), np.nan, dtype=np.float32),
                  "c3": np.full(len(ci), np.nan, dtype=np.float32),
                  "h4": np.full(len(ci), np.nan, dtype=np.float32),
                  "c4": np.full(len(ci), np.nan, dtype=np.float32)}
    print(f"{s_}: n={len(ci)}", flush=True)
for vf in sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl")):
    v = pd.read_pickle(vf)[["code", "date", "adj_high", "adjclose"]]
    v["ci"] = v["code"].map(cid).astype(np.int32)
    v["dk"] = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    v = v.sort_values(["ci", "dk"]).reset_index(drop=True)
    g = v.groupby("ci")
    v["h3"] = g["adj_high"].shift(-3).astype(np.float32); v["c3"] = g["adjclose"].shift(-3).astype(np.float32)
    v["h4"] = g["adj_high"].shift(-4).astype(np.float32); v["c4"] = g["adjclose"].shift(-4).astype(np.float32)
    vv = v[["ci", "dk", "h3", "c3", "h4", "c4"]]
    for s_, d in splits.items():
        need = pd.DataFrame({"ci": d["ci"], "dk": d["dk"], "ri": np.arange(d["n"])})
        mg = need.merge(vv, on=["ci", "dk"], how="inner")
        ri = mg["ri"].values
        d["h3"][ri] = mg["h3"].values; d["c3"][ri] = mg["c3"].values
        d["h4"][ri] = mg["h4"].values; d["c4"][ri] = mg["c4"].values
        del need, mg
    del v, vv; gc.collect()
    print(f"  {os.path.basename(vf)} done {time.time()-t0:.0f}s", flush=True)
for s_, d in splits.items():
    ext = np.stack([d["h3"], d["c3"], d["h4"], d["c4"]], axis=1)
    np.save(f"{PP}/{s_}_ext.npy", ext)
    cov3 = float(np.isfinite(d["h3"]).mean()); cov4 = float(np.isfinite(d["h4"]).mean())
    print(f"{s_}: ext {ext.shape} cov_h3={cov3:.4f} cov_h4={cov4:.4f}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
