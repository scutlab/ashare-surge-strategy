# 价格面板：与 exec_arrays_* 完全相同的 merge 行序，输出每行 (sig,bl,sl,sh,sc,day)
# 对齐验证：用 x=0.10,t=0.30,s=0.015 重算 y，必须与 exec_arrays_10_30_0.015/{tr,va,te}_y.npy 逐位相等；
# 且 te 价格必须与 te_exec.npy[:,1:6] 逐位相等。两项都过才算对齐成功。
import pandas as pd, numpy as np, glob, os, json, time, gc, sys
X, T, S = 0.10, 0.30, 0.015
BASE = os.path.expanduser("~/workspace/ashare"); t0 = time.time()
OUT = BASE + "/price_panel"; os.makedirs(OUT, exist_ok=True)
IN = BASE + "/exec_arrays_10_30_0.015"
flow_files = sorted(glob.glob(BASE + "/full_flow_new_*.pkl"))
v3_files = sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl"))
codes = sorted({c for f in flow_files for c in pd.read_pickle(f)["code"].unique()})
cid = {c: i for i, c in enumerate(codes)}; NC = len(codes)
batch_of = np.empty(NC, dtype=np.int32); flow_small = {}
for bi, f in enumerate(flow_files):
    d = pd.read_pickle(f)
    ci = d["code"].map(cid).values.astype(np.int32); batch_of[ci] = bi
    sig = d["signal_close"].values; entry = sig * (1 - X)
    filled = d["buy_low"].values <= entry
    slh = d["sell_low"].values <= entry * (1 - S); tph = (~slh) & (d["sell_high"].values >= entry * (1 + T))
    real = np.where(slh, -S, np.where(tph, T, d["sell_close"].values / entry - 1))
    y = (filled & (real > 0)).astype(np.int8)
    dk = d["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    flow_small[bi] = pd.DataFrame({"ci": ci, "dk": dk, "y": y, "sig": sig.astype(np.float32),
        "bl": d["buy_low"].values.astype(np.float32), "sl": d["sell_low"].values.astype(np.float32),
        "sh": d["sell_high"].values.astype(np.float32), "sc": d["sell_close"].values.astype(np.float32)})
    del d; gc.collect()
print(f"codes {NC}", flush=True)
cmap = np.arange(NC, dtype=np.int32) % 3
acc = {"tr": [], "va": [], "te": []}; accy = {"tr": [], "va": [], "te": []}
accd = {"tr": [], "va": [], "te": []}; accc = {"tr": [], "va": [], "te": []}; tot = 0; match = 0
for vf in v3_files:
    v = pd.read_pickle(vf)
    ci = v["code"].map(cid).values.astype(np.int32)
    dk = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
    b = batch_of[ci]; n = len(v); tot += n; del v; gc.collect()
    xi_all = np.arange(n)
    for bi in np.unique(b):
        sel = b == bi
        left = pd.DataFrame({"ci": ci[sel], "dk": dk[sel], "xi": xi_all[sel]})
        mg = left.merge(flow_small[bi], on=["ci", "dk"], how="inner"); match += len(mg)
        if len(mg) == 0: continue
        sp = cmap[mg["ci"].values]
        pr = mg[["sig", "bl", "sl", "sh", "sc"]].values.astype(np.float32)
        dy = mg["dk"].values.astype(np.int32); yv = mg["y"].values; cc = mg["ci"].values.astype(np.int32)
        for s_, c in (("tr", 0), ("va", 1), ("te", 2)):
            mm = sp == c
            if mm.sum():
                acc[s_].append(pr[mm]); accy[s_].append(yv[mm]); accd[s_].append(dy[mm]); accc[s_].append(cc[mm])
        del left, mg, pr, dy, yv
    del ci, dk, b, xi_all; gc.collect()
print(f"coverage {match}/{tot}={match/tot:.4f}", flush=True)
ok = True
for s_ in ["tr", "va", "te"]:
    pr = np.concatenate(acc[s_]); yv = np.concatenate(accy[s_]); dy = np.concatenate(accd[s_])
    cc = np.concatenate(accc[s_])
    yref = np.load(f"{IN}/{s_}_y.npy")
    eq = (len(yv) == len(yref)) and bool((yv == yref).all())
    print(f"{s_}: n={len(yv)} y_equal={eq}", flush=True); ok &= eq
    np.save(f"{OUT}/{s_}_prices.npy", pr); np.save(f"{OUT}/{s_}_day.npy", dy); np.save(f"{OUT}/{s_}_code.npy", cc)
json.dump({"codes": codes}, open(f"{OUT}/codes.json", "w"))
te_ex = np.load(f"{IN}/te_exec.npy")
te_pr = np.load(f"{OUT}/te_prices.npy")
eq2 = te_pr.shape == te_ex[:, 1:6].shape and bool((te_pr == te_ex[:, 1:6]).all())
print(f"te prices equal te_exec.npy: {eq2}", flush=True); ok &= eq2
json.dump({"ok": bool(ok), "elapsed": time.time() - t0}, open(f"{OUT}/build_check.json", "w"))
print("ALIGN_OK" if ok else "ALIGN_FAIL", flush=True)
assert ok, "alignment verification failed"
print(f"DONE {time.time()-t0:.0f}s", flush=True)
