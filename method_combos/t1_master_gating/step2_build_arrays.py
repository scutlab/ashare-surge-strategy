# T1 step2: 构建增补特征数组（MASTER 式市场门控，纯特征工程）
# 只读原始文件；在基数组拷贝语义上操作（新文件写 T1 目录，不碰原始）。
# 流程（每 flow 文件 fi 独立一批，可断点续跑）：
#   flow fi (full_flow_new_{fi}) <-> v3_sorted[fi]（与 new_flow.py 相同的 sorted glob 顺序）
#   取 buy_open/sell_open（v3 adj_open 按 code shift -1/-2），按 (code,date) merge
#   按 date join market_state（20 维 + mkt_gate）；市场状态缺失的早期日期用 bfill
#   X_aug = [29 基线, 20 市场状态, 1 门控标量, 29 门控特征] = 79 列 float32
#   按 code 的 i%3 分流（与 build_base_arrays.py 完全相同的 codes/cmap）-> chunks
# 校验：每批 tr/va chunk 的前 29 列必须与 daily_base_arrays/{tr,va}_X.npy 对应行块逐位相等
import os, glob, time, gc, numpy as np, pandas as pd
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks"); os.makedirs(CHK, exist_ok=True)
FEATS29 = ["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3",
 "ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60",
 "close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1",
 "excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5",
 "limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
WINS = [5, 10, 20, 30, 60]
MKT_COLS = ([f"mkt_px_mean_{w}" for w in WINS] + [f"mkt_px_std_{w}" for w in WINS] +
            [f"mkt_vol_mean_{w}" for w in WINS] + [f"mkt_vol_std_{w}" for w in WINS])
GATE_COL = "mkt_gate"
NEW_FEAT_NAMES = FEATS29 + MKT_COLS + [GATE_COL] + [f"g_{f}" for f in FEATS29]
assert len(NEW_FEAT_NAMES) == 79
t0 = time.time()

# --- 与 build_base_arrays.py 完全相同的 codes/cmap ---
flow_files = sorted(glob.glob(os.path.join(BASE, "full_flow_new_*.pkl")))
v3_files = sorted(glob.glob(os.path.join(BASE, "full_features_batch_v3_*.pkl")))
assert len(flow_files) == 25 and len(v3_files) == 25
codes = set()
for f in flow_files:
    dd = pd.read_pickle(f)
    codes |= set(dd["code"].unique()); del dd; gc.collect()
codes = sorted(codes); cmap = {c: i % 3 for i, c in enumerate(codes)}
print(f"codes {len(codes)}", flush=True)

# --- 市场状态（date -> 21 列） ---
ms = pd.read_pickle(os.path.join(OUT, "market_state.pkl"))
ms["date"] = pd.to_datetime(ms["date"]).dt.normalize()
ms = ms.sort_values("date").reset_index(drop=True)
ms_idx = ms.set_index("date")
print(f"market_state {ms.shape} {ms['date'].min().date()}~{ms['date'].max().date()}", flush=True)

# --- 基线数组（mmap 只读，用于行序校验） ---
base = {}
for s in ["train", "val"]:
    base[s] = np.load(os.path.join(BASE, "daily_base_arrays", f"{s}_X.npy"), mmap_mode="r")
    print(s, base[s].shape, base[s].dtype, flush=True)
offset = {"train": 0, "val": 0}
SPLIT = {0: "train", 1: "val", 2: "test"}

done_path = os.path.join(CHK, "done.txt")
done = set()
if os.path.exists(done_path):
    done = set(int(x) for x in open(done_path).read().split())
# 恢复 offset（已完成的 chunk 行数）
for s in ["train", "val"]:
    n = 0
    for fi in sorted(done):
        p = os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy")
        if os.path.exists(p): n += np.load(p, mmap_mode="r").shape[0]
    offset[s] = n
print(f"resume: done={sorted(done)} offsets={offset}", flush=True)

bfill_used = 0
for fi, ff in enumerate(flow_files):
    if fi in done:
        continue
    f = pd.read_pickle(ff)
    f["date"] = pd.to_datetime(f["date"]).dt.normalize()
    # --- opens from v3 ---
    v = pd.read_pickle(v3_files[fi])[["code", "date", "adj_open"]]
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    v = v.sort_values(["code", "date"]).reset_index(drop=True)
    g = v.groupby("code", sort=False)["adj_open"]
    v["buy_open"] = g.shift(-1).astype(np.float32); v["sell_open"] = g.shift(-2).astype(np.float32)
    m = f[["code", "date"]].merge(v[["code", "date", "buy_open", "sell_open"]],
                                  on=["code", "date"], how="left")
    assert len(m) == len(f) and m["buy_open"].notna().all(), f"open merge fail fi={fi}"
    f["buy_open"] = m["buy_open"].to_numpy(); f["sell_open"] = m["sell_open"].to_numpy()
    del v, g, m; gc.collect()
    # --- market state join ---
    f = f.join(ms_idx[MKT_COLS + [GATE_COL]], on="date")
    na_rows = f[MKT_COLS[0]].isna().sum()
    if na_rows:
        bfill_used += int(na_rows)
        f[MKT_COLS + [GATE_COL]] = f[MKT_COLS + [GATE_COL]].bfill()
    assert not f[MKT_COLS + [GATE_COL]].isna().any().any(), f"market NaN fi={fi}"
    # --- 增补特征 ---
    X29 = f[FEATS29].to_numpy(np.float32)
    M20 = f[MKT_COLS].to_numpy(np.float32)
    gt = f[[GATE_COL]].to_numpy(np.float32)
    Xaug = np.concatenate([X29, M20, gt, X29 * gt], axis=1).astype(np.float32)
    assert Xaug.shape[1] == 79 and np.isfinite(Xaug).all(), f"Xaug bad fi={fi}"
    sp = f["code"].map(cmap).to_numpy()
    meta_cols = ["code", "date", "signal_close", "buy_low", "buy_open", "sell_low",
                 "sell_high", "sell_close", "sell_open", "bought", "y_strategy"]
    for sid, sname in SPLIT.items():
        msk = sp == sid
        if not msk.any():
            continue
        xa = Xaug[msk]
        np.save(os.path.join(CHK, f"t1_X_aug_{sname}_{fi:02d}.npy"), xa)
        mf = f.loc[msk, meta_cols]
        np.savez_compressed(os.path.join(CHK, f"t1_meta_{sname}_{fi:02d}.npz"),
            code=mf["code"].to_numpy(), date=mf["date"].to_numpy().astype("datetime64[D]"),
            signal_close=mf["signal_close"].to_numpy(np.float32),
            buy_low=mf["buy_low"].to_numpy(np.float32), buy_open=mf["buy_open"].to_numpy(np.float32),
            sell_low=mf["sell_low"].to_numpy(np.float32), sell_high=mf["sell_high"].to_numpy(np.float32),
            sell_close=mf["sell_close"].to_numpy(np.float32), sell_open=mf["sell_open"].to_numpy(np.float32),
            bought=mf["bought"].to_numpy(np.int8), y=mf["y_strategy"].to_numpy(np.int8))
        # --- 行序校验（tr/va 前29列 == 基数组对应行块） ---
        if sname in ("train", "val"):
            o = offset[sname]; n = xa.shape[0]
            ref = np.asarray(base[sname][o:o + n])
            assert ref.shape == xa[:, :29].shape and (ref == xa[:, :29]).all(), \
                f"row-order mismatch fi={fi} split={sname}"
            offset[sname] += n
    del f, X29, M20, gt, Xaug, sp; gc.collect()
    with open(done_path, "a") as fh:
        fh.write(f"{fi}\n")
    done.add(fi)
    print(f"fi={fi:02d} rows={len(xa) if 'xa' in dir() else '?'} offsets={offset} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

manifest = {"n_files": len(done), "offsets": offset, "bfill_rows": bfill_used,
            "n_features": 79, "feature_names": NEW_FEAT_NAMES,
            "row_order_verified": True, "elapsed_s": round(time.time() - t0, 1)}
import json
json.dump(manifest, open(os.path.join(OUT, "t1_build_manifest.json"), "w"), indent=1)
print("BUILD DONE", manifest, flush=True)
