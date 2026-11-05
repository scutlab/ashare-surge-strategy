#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""method_combos 公共模块：只读现有数据；统一评估口径；结果保存与加载校验。

全局约束：
- 只读 ~/workspace/ashare 下现有数据，绝不修改/删除原始与分批文件（PRESERVATION.md）。
- 评估口径与现行配置一致：x=0.05 限价买入、+7%止盈/-2%止损、0.5%往返费、
  gap-aware 真实执行、Top10 名单、平均每笔按前 N_STRATEGY=3 只统计。
"""
import os, sys, json, time, glob
import numpy as np
import pandas as pd

BASE = os.path.expanduser("~/workspace/ashare")
sys.path.insert(0, BASE)
import strategy_config as SC  # noqa: E402

COMBO_DIR = os.path.join(BASE, "method_combos")
T1_DIR = os.path.join(COMBO_DIR, "t1_master_gating")
PANEL_DIR = os.path.join(BASE, "price_panel")
BASE_ARR = os.path.join(BASE, "daily_base_arrays")
EXEC_DIR = os.path.join(BASE, "exec_arrays_10_30_0.015")

# ---- 现行口径（strategy_config.py，2026-11-02 锁定）----
X_DISC = float(SC.ORIG_X_DISC)   # 0.05 买入折扣：信号收盘×0.95 限价
STOP = float(SC.ORIG_STOP)       # 0.02 止损
TAKE = float(SC.ORIG_TAKE)       # 0.07 止盈
FEE = float(SC.FEE)              # 0.005 往返手续费
N_LIST = int(SC.N_LIST)          # 10 名单数量
N_STRATEGY = int(SC.N_STRATEGY)  # 3 平均每笔按前3只统计
FEAT29 = list(SC.FEATS)

# T1 市场状态列（market_daily.csv）：20 个市场统计 + mkt_gate
MKT_FEAT_COLS = (
    [f"mkt_px_mean_{w}" for w in (5, 10, 20, 30, 60)]
    + [f"mkt_px_std_{w}" for w in (5, 10, 20, 30, 60)]
    + [f"mkt_vol_mean_{w}" for w in (5, 10, 20, 30, 60)]
    + [f"mkt_vol_std_{w}" for w in (5, 10, 20, 30, 60)]
    + ["mkt_gate"]
)


# ---------------------------------------------------------------- 数据加载
def load_features(split):
    """读 29 维基数组。split: tr/va/te。返回 (X float32 memmap, y int8)。"""
    if split == "tr":
        X = np.load(f"{BASE_ARR}/train_X.npy", mmap_mode="r")
        y = np.load(f"{BASE_ARR}/train_y.npy", mmap_mode="r")
    elif split == "va":
        X = np.load(f"{BASE_ARR}/val_X.npy", mmap_mode="r")
        y = np.load(f"{BASE_ARR}/val_y.npy", mmap_mode="r")
    elif split == "te":
        X = np.load(f"{EXEC_DIR}/te_X.npy", mmap_mode="r")
        y = np.load(f"{EXEC_DIR}/te_y.npy", mmap_mode="r")
    else:
        raise ValueError(f"unknown split {split}")
    assert X.shape[1] == 29, f"feature dim changed: {X.shape}"
    return X, y


def load_exec_bars(split):
    """读执行价格条。优先用 T1 meta（含买卖开盘价，真 gap-aware）；
    否则退回 price_panel（sig/bl/sl/sh/sc，无开盘价，入口按限价保守估计）。

    返回 dict: sig, bl, sl, sh, sc, bo, so(可为None), day(int32), code(int32), codes(list), has_open(bool)
    行序与同 split 基数组一致（T1 manifest row_order_verified）。
    """
    n_exp = {"tr": 4806811, "va": 4781715, "te": 4807515}[split]
    t1_split = {"tr": "train", "va": "val", "te": "test"}[split]  # T1 文件用 train/val/test 命名
    bars = {}
    meta_files = sorted(glob.glob(f"{T1_DIR}/chunks/t1_meta_{t1_split}_*.npz"))
    use_meta = False
    if meta_files:
        try:
            parts = {k: [] for k in ("signal_close", "buy_low", "buy_open", "sell_low",
                                     "sell_high", "sell_close", "sell_open", "date", "code")}
            for f in meta_files:
                z = np.load(f, allow_pickle=True)  # code 列为 object 字符串，需 allow_pickle
                for k in parts:
                    parts[k].append(z[k])
                z.close()
            code_names = json.load(open(f"{PANEL_DIR}/codes.json"))["codes"]
            cid = {c: i for i, c in enumerate(code_names)}
            date_i = np.concatenate(parts["date"]).astype("datetime64[D]").astype(np.int64).astype(np.int32)
            code_i = pd.Series(np.concatenate(parts["code"]).astype(str)).map(cid)
            if len(date_i) == n_exp and code_i.notna().all():
                code_i = code_i.to_numpy(np.int32)
                panel_day = np.load(f"{PANEL_DIR}/{split}_day.npy", mmap_mode="r")
                panel_code = np.load(f"{PANEL_DIR}/{split}_code.npy", mmap_mode="r")
                if (len(panel_day) == n_exp and bool((date_i == panel_day).all())
                        and bool((code_i == panel_code).all())):
                    use_meta = True
                    bars = {"sig": np.concatenate(parts["signal_close"]).astype(np.float64),
                            "bl": np.concatenate(parts["buy_low"]).astype(np.float64),
                            "sl": np.concatenate(parts["sell_low"]).astype(np.float64),
                            "sh": np.concatenate(parts["sell_high"]).astype(np.float64),
                            "sc": np.concatenate(parts["sell_close"]).astype(np.float64),
                            "bo": np.concatenate(parts["buy_open"]).astype(np.float64),
                            "so": np.concatenate(parts["sell_open"]).astype(np.float64),
                            "day": date_i,
                            "code": code_i,
                            "has_open": True}
        except Exception as e:
            print(f"[common] T1 meta 读取失败，回退 price_panel: {type(e).__name__}: {e}", flush=True)
    if not use_meta:
        pr = np.load(f"{PANEL_DIR}/{split}_prices.npy", mmap_mode="r")  # sig,bl,sl,sh,sc
        assert len(pr) == n_exp, f"panel rows {len(pr)} != {n_exp}"
        bars = {"sig": pr[:, 0].astype(np.float64), "bl": pr[:, 1].astype(np.float64),
                "sl": pr[:, 2].astype(np.float64), "sh": pr[:, 3].astype(np.float64),
                "sc": pr[:, 4].astype(np.float64), "bo": None, "so": None,
                "day": np.load(f"{PANEL_DIR}/{split}_day.npy", mmap_mode="r").astype(np.int32),
                "code": np.load(f"{PANEL_DIR}/{split}_code.npy", mmap_mode="r").astype(np.int32),
                "has_open": False}
        print("[common] 使用 price_panel 近似口径（无开盘价：入口按限价保守估计，无跳空止盈/止损）", flush=True)
    else:
        print("[common] 使用 T1 meta 真 gap-aware 口径（含买卖开盘价）", flush=True)
    bars["codes"] = json.load(open(f"{PANEL_DIR}/codes.json"))["codes"]
    return bars


def load_market_daily():
    """读 T1 的 market_daily.csv（date -> 21 维市场状态），无前视偏差（当日值只用当日及以前数据）。"""
    p = os.path.join(T1_DIR, "market_daily.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"T1 市场状态文件缺失: {p}（T1 测试尚未产出？）")
    df = pd.read_csv(p)
    return df


def ensure_market_daily():
    """优先复用 T1 的 market_daily.csv；缺失则按 MASTER 口径自建：
    上证指数价格（index_sse.pkl）+ 全市场等权日均成交量（v3 批次聚合，不重拉数据），
    过去 5/10/20/30/60 日均值/标准差 + mkt_gate 门控标量。
    自建结果缓存到 method_combos/market_daily_built.csv（不碰 T1 目录、不改原始文件）。"""
    p = os.path.join(T1_DIR, "market_daily.csv")
    if os.path.exists(p):
        print("[common] 复用 T1 市场状态 market_daily.csv", flush=True)
        return pd.read_csv(p)
    print("[common] T1 市场状态缺失，按 MASTER 口径自建（只读现有数据）…", flush=True)
    t0 = time.time()
    cache = os.path.join(COMBO_DIR, "market_daily_built.csv")
    if os.path.exists(cache):
        print("[common] 使用已缓存的自建市场状态", flush=True)
        return pd.read_csv(cache)
    sse = pd.read_pickle(os.path.join(BASE, "index_sse.pkl"))
    sse["date"] = pd.to_datetime(sse["date"]).dt.normalize()
    sse = sse.sort_values("date").reset_index(drop=True)
    sse["px_ret"] = sse["mkt_adjclose"].pct_change()
    acc = {}
    v3_files = sorted(glob.glob(os.path.join(BASE, "full_features_batch_v3_*.pkl")))
    if not v3_files:
        raise FileNotFoundError("v3 特征批次缺失，无法自建市场成交量")
    for vf in v3_files:
        v = pd.read_pickle(vf)[["date", "volume"]]
        v["date"] = pd.to_datetime(v["date"]).dt.normalize()
        g = v.groupby("date")["volume"].agg(["sum", "count"])
        for d, row in g.iterrows():
            a = acc.get(d, [0.0, 0])
            a[0] += float(row["sum"]); a[1] += int(row["count"]); acc[d] = a
        del v, g
    vol = pd.DataFrame([{"date": d, "mkt_vol": s / max(c, 1)} for d, (s, c) in acc.items()])
    mkt = sse[["date", "px_ret"]].merge(vol, on="date", how="inner").sort_values("date").reset_index(drop=True)
    mkt["log_vol"] = np.log(mkt["mkt_vol"].clip(lower=1.0))
    WINS = [5, 10, 20, 30, 60]
    for w in WINS:
        mkt[f"mkt_px_mean_{w}"] = mkt["px_ret"].rolling(w, min_periods=1).mean()
        mkt[f"mkt_px_std_{w}"] = mkt["px_ret"].rolling(w, min_periods=2).std().fillna(0.0)
        mkt[f"mkt_vol_mean_{w}"] = mkt["log_vol"].rolling(w, min_periods=1).mean()
        mkt[f"mkt_vol_std_{w}"] = mkt["log_vol"].rolling(w, min_periods=2).std().fillna(0.0)
    mu = mkt["mkt_px_mean_20"].expanding(min_periods=60).mean()
    sd = mkt["mkt_px_mean_20"].expanding(min_periods=60).std().fillna(0.0).replace(0.0, np.nan)
    z = ((mkt["mkt_px_mean_20"] - mu) / sd).fillna(0.0).clip(-5, 5)
    mkt["mkt_gate"] = 1.0 + np.tanh(z)
    cols = [c for c in MKT_FEAT_COLS if c in mkt.columns]
    out = mkt[["date"] + cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    assert not out[cols].isna().any().any(), "自建市场状态含 NaN"
    os.makedirs(COMBO_DIR, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"[common] 市场状态自建完成 {len(out)} 天, {time.time()-t0:.0f}s -> {cache}", flush=True)
    return out


def market_matrix(day_ints, mkt_df):
    """把每行的 day(int, unix epoch 天) 映射为 21 维市场状态向量。"""
    dstr = pd.to_datetime(day_ints.astype(np.int64), unit="D", origin="unix").strftime("%Y-%m-%d")
    tab = mkt_df.set_index("date")
    cols = [c for c in MKT_FEAT_COLS if c in tab.columns]
    mat = tab[cols]
    out = np.empty((len(dstr), len(cols)), dtype=np.float32)
    # 向量化映射：date -> 位置
    idx = tab.index
    pos = idx.get_indexer(dstr)
    if (pos < 0).any():
        missing = np.unique(dstr[pos < 0])[:5]
        raise ValueError(f"market_daily 缺少日期: {missing}")
    out[:] = mat.to_numpy(np.float32)[pos]
    return out, cols


# ---------------------------------------------------------------- 实现收益
def realized_current_cfg(bars, x=X_DISC, stop=STOP, take=TAKE, fee=FEE):
    """按现行口径从执行价格条算实现收益（向量化）。
    有开盘价时 = build_gap_review 单持有日逻辑：买入跳空低开按开盘成交；
    卖出日开盘穿过止盈/止损按开盘成交；同日双触按止损。
    无开盘价时：入口按限价保守估计（=真实执行的下界）。
    返回 (filled bool, realized_net float64)：未成交计 0（不交易无费用）。
    """
    sig, bl, sl, sh, sc = bars["sig"], bars["bl"], bars["sl"], bars["sh"], bars["sc"]
    limit = sig * (1 - x)
    bo, so = bars["bo"], bars["so"]
    n = len(sig)
    filled = np.zeros(n, dtype=bool)
    entry = np.zeros(n, dtype=np.float64)
    if bo is not None:
        gap_buy = bo <= limit
        norm_buy = (~gap_buy) & (bl <= limit)
        filled = gap_buy | norm_buy
        entry = np.where(gap_buy, bo, np.where(norm_buy, limit, 0.0))
    else:
        filled = bl <= limit
        entry = np.where(filled, limit, 0.0)
    r = np.zeros(n, dtype=np.float64)
    fe = filled & (entry > 0)
    if so is not None:
        gap_stop = fe & (so <= entry * (1 - stop))
        gap_take = fe & (~gap_stop) & (so >= entry * (1 + take))
        r[gap_stop] = -stop
        r[gap_take] = take
        rest = fe & (~gap_stop) & (~gap_take)
    else:
        rest = fe
    if rest.any():
        e = entry[rest]
        hit_stop = sl[rest] <= e * (1 - stop)
        hit_take = (~hit_stop) & (sh[rest] >= e * (1 + take))
        rr = np.where(hit_stop, -stop, np.where(hit_take, take, sc[rest] / e - 1))
        r[rest] = rr
    realized_net = np.where(filled, r - fee, 0.0)
    return filled, realized_net


# ---------------------------------------------------------------- 评估
def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float(a.dot(b) / denom) if denom > 0 else float("nan")


def _rank(a):
    return np.argsort(np.argsort(a)).astype(np.float64)


def evaluate(scores, bars, tag="eval", verbose=True):
    """同口径评估：排序指标（IC/RankIC，按日分组，只看成交行）+
    Top10 模拟（每日按分取 Top10，前 N_STRATEGY 只中成交的平均每笔）。
    scores: (n,) float；bars: load_exec_bars 输出。
    返回 metrics dict（含可存盘的 per-day 数组）。
    """
    t0 = time.time()
    filled, realized = realized_current_cfg(bars)
    days = bars["day"]
    udays = np.unique(days)
    ics, rics, top3_rs, top3_n = [], [], [], []
    n_traded_days = 0
    for d in udays:
        m = (days == d) & filled
        if m.sum() < 20:
            continue
        s = scores[m].astype(np.float64); r = realized[m]
        ics.append(_pearson(s, r))
        rics.append(_pearson(_rank(s), _rank(r)))
        order = np.argsort(-s, kind="stable")
        top = order[:N_LIST]
        tb = top[:N_STRATEGY]
        top3_rs.append(r[tb].sum())
        top3_n.append(len(tb))
        n_traded_days += 1
    ics = np.array(ics); rics = np.array(rics)
    top3_rs = np.array(top3_rs); top3_n = np.array(top3_n)
    metrics = {
        "tag": tag,
        "n_days": int(len(udays)),
        "n_traded_days": int(n_traded_days),
        "IC": float(np.nanmean(ics)) if len(ics) else None,
        "RankIC": float(np.nanmean(rics)) if len(rics) else None,
        "IC_std": float(np.nanstd(ics)) if len(ics) else None,
        "top3_mean": float(top3_rs.sum() / top3_n.sum()) if top3_n.sum() else None,
        "top3_n": int(top3_n.sum()),
        "fill_rate": float(filled.mean()),
        "has_open": bool(bars["has_open"]),
        "x_disc": X_DISC, "stop": STOP, "take": TAKE, "fee": FEE,
        "n_list": N_LIST, "n_strategy": N_STRATEGY,
        "elapsed_s": round(time.time() - t0, 1),
    }
    arrays = {"per_day_ic": ics, "per_day_rankic": rics,
              "per_day_top3_sum": top3_rs, "per_day_top3_n": top3_n}
    if verbose:
        ic = f"{metrics['IC']:+.4f}" if metrics["IC"] is not None else "n/a"
        ric = f"{metrics['RankIC']:+.4f}" if metrics["RankIC"] is not None else "n/a"
        tp3 = f"{metrics['top3_mean']:+.4%}" if metrics["top3_mean"] is not None else "n/a"
        print(f"[{tag}] IC={ic} RankIC={ric} 前{N_STRATEGY}只平均每笔={tp3} "
              f"(n={metrics['top3_n']}, {metrics['n_traded_days']}天, "
              f"{'真gap-aware' if metrics['has_open'] else '近似口径'})", flush=True)
    return metrics, arrays


def evaluate_with_adjusted_scores(scores, bars, adjust_fn=None, tag="eval", verbose=True):
    """evaluate 的变体：先用 adjust_fn(scores, bars, days) 调整分数再评估（如置信度加权）。"""
    if adjust_fn is not None:
        scores = adjust_fn(np.asarray(scores, dtype=np.float64), bars)
    return evaluate(scores, bars, tag=tag, verbose=verbose)


# ---------------------------------------------------------------- 训练与保存
def train_hgb(X, r, filled, tiny=False):
    """HGB 回归器（EV 式：只在成交行上拟合 E[return|filled]）。"""
    from sklearn.ensemble import HistGradientBoostingRegressor
    kw = dict(SC.HGB_REG)
    if tiny:
        kw.update(max_iter=30, min_samples_leaf=50)
    reg = HistGradientBoostingRegressor(**kw)
    reg.fit(np.asarray(X)[filled], np.asarray(r)[filled].astype(np.float64))
    return reg


def save_summary(outdir, summary, arrays=None, model=None, model_name="model.pkl"):
    """保存结果：summary.json + arrays.npz + 可选模型；全部做加载回校验。"""
    os.makedirs(outdir, exist_ok=True)
    sp = os.path.join(outdir, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    ap = None
    if arrays:
        ap = os.path.join(outdir, "arrays.npz")
        np.savez_compressed(ap, **arrays)
    mp = None
    if model is not None:
        import pickle
        mp = os.path.join(outdir, model_name)
        tmp = mp + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(model, f)
        os.rename(tmp, mp)
    # 加载回校验（项目规则：以产出文件可加载为准）
    json.load(open(sp))
    if ap:
        z = np.load(ap); _ = [z[k] for k in z.files]; z.close()
    if mp:
        import pickle
        pickle.load(open(mp, "rb"))
    print(f"[common] 结果已保存并通过加载校验: {outdir}", flush=True)
    return {"summary": sp, "arrays": ap, "model": mp}


def spot_check_alignment(X, days, n=2000, tol=2e-4):
    """抽查行序对齐：用 index_sse 重算 mkt_ret，与 X[:,24] 对比。只告警不中断。"""
    try:
        df = pd.read_pickle(os.path.join(BASE, "index_sse.pkl"))
        d = pd.to_datetime(df["date"])
        di = (d - pd.Timestamp("1970-01-01")).dt.days.to_numpy()
        px = df["mkt_adjclose"].to_numpy(np.float64)
        ret = np.empty_like(px); ret[0] = 0; ret[1:] = px[1:] / px[:-1] - 1
        lut = dict(zip(di.tolist(), ret.tolist()))
        rng = np.random.default_rng(0)
        idx = rng.choice(len(days), size=min(n, len(days)), replace=False)
        exp = np.array([lut.get(int(dd), np.nan) for dd in days[idx]])
        got = np.asarray(X[idx, 24], dtype=np.float64)
        m = np.isfinite(exp) & np.isfinite(got)
        err = np.abs(exp[m] - got[m]).max() if m.sum() else float("nan")
        ok = bool(m.sum() > 100 and err < tol)
        print(f"[common] 行序抽查: {m.sum()}行, mkt_ret最大偏差={err:.2e} -> {'通过' if ok else '告警，请核查'}", flush=True)
        return {"align_check_n": int(m.sum()), "align_check_maxerr": float(err), "align_check_ok": ok}
    except Exception as e:
        print(f"[common] 行序抽查跳过: {type(e).__name__}: {e}", flush=True)
        return {"align_check_ok": None}
