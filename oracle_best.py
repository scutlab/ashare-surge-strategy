# Oracle 分析：对 train+val 每行，求事后最优 (x,t)（33 格点），回归找驱动特征
# 网格：x=0.00..0.10 步长0.01（11），t={0.10,0.20,0.30}（3），s=0.015 固定
# 输出：best_x 分布、HGB 回归特征重要性、分位数表、t* 分析 -> oracle_best.json
import numpy as np, os, json, time, gc, pickle
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
xg = np.arange(0.0, 0.101, 0.01, dtype=np.float32)   # 11
tg = np.array([0.10, 0.20, 0.30], dtype=np.float32)  # 3
S = 0.015
res = {"xgrid": xg.tolist(), "tgrid": tg.tolist(), "s": S}
all_imp = []
for split in ["tr", "va"]:
    pr = np.load(f"{PP}/{split}_prices.npy")          # sig,bl,sl,sh,sc float32
    X = np.load(f"{IN}/{split}_X.npy").astype(np.float32)
    n = len(pr)
    sig = pr[:, 0].astype(np.float32); bl = pr[:, 1]; sl = pr[:, 2]; sh = pr[:, 3]; sc = pr[:, 4]
    entry = sig[:, None] * (1.0 - xg[None, :])        # (n,11)
    filled = bl[:, None] <= entry
    slh = sl[:, None] <= entry * (1 - S)
    has_fill = filled.any(axis=1)
    # 对每个 t 求 real(n,11)，联合 argmax
    best_real = np.full(n, -np.inf, dtype=np.float32)
    best_xi = np.zeros(n, dtype=np.int8); best_ti = np.zeros(n, dtype=np.int8)
    real = np.empty((n, 11), dtype=np.float32)
    for ti, t in enumerate(tg):
        tph = (~slh) & (sh[:, None] >= entry * (1 + t))
        np.divide(sc[:, None], entry, out=real)
        real -= 1.0
        np.copyto(real, t, where=tph); np.copyto(real, -S, where=slh)
        real[~filled] = -np.inf
        upd = real.max(axis=1) > best_real
        best_real[upd] = real.max(axis=1)[upd]
        am = real.argmax(axis=1)
        best_xi[upd] = am[upd].astype(np.int8); best_ti[upd] = np.int8(ti)
    best_real[~has_fill] = np.nan
    # 权重：各 x 下（t=0.30）已成交收益的极差，衡量“选 x 有多重要”
    tph30 = (~slh) & (sh[:, None] >= entry * 1.30)
    r30 = sc[:, None] / entry - 1.0
    r30[tph30] = 0.30; r30[slh] = -S; r30[~filled] = np.nan
    with np.errstate(all="ignore"):
        wgt = np.nanmax(r30, axis=1) - np.nanmin(r30, axis=1)
    wgt[~has_fill] = 0.0; wgt = np.nan_to_num(wgt, nan=0.0).astype(np.float32)
    np.savez_compressed(f"{BASE}/oracle_{split}.npz", best_xi=best_xi, best_ti=best_ti,
                        wgt=wgt, best_real=best_real.astype(np.float32), has_fill=has_fill)
    hx, _ = np.histogram(best_xi[has_fill], bins=np.arange(12) - 0.5)
    ht, _ = np.histogram(best_ti[has_fill], bins=np.arange(4) - 0.5)
    res[f"{split}_bestx_hist"] = hx.tolist(); res[f"{split}_bestt_hist"] = ht.tolist()
    res[f"{split}_fill_any"] = float(has_fill.mean())
    # t* 分析：只看 t 确实影响结果的行（在 x* 处 t 的极差>0.01）
    xi = best_xi[has_fill]
    r_t = []
    for t in tg:
        ee = sig * (1 - xg[best_xi])
        f2 = bl <= ee; s2 = sl <= ee * (1 - S); p2 = (~s2) & (sh >= ee * (1 + t))
        rr = np.where(s2, -S, np.where(p2, t, sc / ee - 1)); rr[~f2] = np.nan
        r_t.append(rr)
    r_t = np.stack(r_t)
    with np.errstate(all="ignore"):
        trange = np.nanmax(r_t, axis=0) - np.nanmin(r_t, axis=0)
    matters = (trange > 0.01) & has_fill
    res[f"{split}_t_matters_rate"] = float(matters.mean())
    if matters.sum() > 0:
        htm, _ = np.histogram(best_ti[matters], bins=np.arange(4) - 0.5)
        res[f"{split}_bestt_hist_matters"] = htm.tolist()
    print(f"[{split}] n={n} fill_any={has_fill.mean():.3f} t_matters={matters.mean():.4f}", flush=True)
    del pr, X, entry, filled, slh, real, r30, best_real, best_xi, best_ti, wgt, xi; gc.collect()
json.dump(res, open(f"{BASE}/oracle_best.json", "w"), indent=2)
print(f"PART1 DONE {time.time()-t0:.0f}s", flush=True)
