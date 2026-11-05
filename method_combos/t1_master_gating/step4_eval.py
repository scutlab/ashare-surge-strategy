# T1 step4: 评估（同口径对照）
# 口径（与现行配置/测试方案一致）：
#   买入=信号收盘×0.95 限价（x=0.05），+7% 止盈 / -2% 止损，0.5% 往返费，
#   gap-aware 真实执行（买入跳空低开按开盘价成交；卖出日开盘穿过止盈/止损按开盘价成交；
#   盘中双触按止损），Top10 名单、平均每笔按前 3 只中成交的统计。
# 验证集：IC/RankIC/ICIR（日度截面，相对 gap-aware 净收益）+ AUC（相对训练标签 y）
# 测试集：组合回测（Top10->前3只成交每笔净收益），70% 单侧下界 = 均值-0.524*SE（日度周期为单位）
import os, time, json, gc, pickle, numpy as np
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
t0 = time.time()

X_DISC, STOP, TAKE, FEE = 0.05, 0.02, 0.07, 0.005
manifest = json.load(open(os.path.join(OUT, "t1_build_manifest.json")))
m_base = pickle.load(open(os.path.join(OUT, "t1_model_baseline_sub.pkl"), "rb"))
m_gate = pickle.load(open(os.path.join(OUT, "t1_model_gated_sub.pkl"), "rb"))
print("models loaded", flush=True)

def load_split(s):
    if s in manifest["offsets"]:
        n = manifest["offsets"][s]
    else:  # test：从 chunk 文件推算行数
        n = sum(np.load(os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy"),
                        mmap_mode="r").shape[0]
                for fi in range(25) if os.path.exists(os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy")))
    X = np.empty((n, 79), dtype=np.float32)
    o = 0
    metas = []
    for fi in range(25):
        p = os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy")
        if not os.path.exists(p):
            continue
        c = np.load(p); X[o:o + len(c)] = c; o += len(c); del c
        # npz 为本脚本自产（含 pandas str 列），allow_pickle=True 安全
        metas.append(np.load(os.path.join(CHK, f"t1_meta_{s}_{fi:02d}.npz"), allow_pickle=True))
    X = X[:o]
    meta = {k: np.concatenate([m[k] for m in metas]) for k in metas[0].files}
    assert len(meta["code"]) == o
    return X, meta

def gap_aware_net(meta):
    """向量化 gap-aware 净收益（镜像 daily_pipeline.build_gap_review，hold=[sell_day]）。
    返回 net（未成交=NaN）与 status（0未成交 1止损 2止盈/跳空止盈 3收盘卖出）。"""
    sig = meta["signal_close"].astype(np.float64)
    limit = sig * (1 - X_DISC)
    bought = meta["buy_low"] <= limit
    n = len(sig)
    net = np.full(n, np.nan); st = np.zeros(n, np.int8)
    idx = np.where(bought)[0]
    bo = meta["buy_open"][idx].astype(np.float64)
    entry = np.minimum(limit[idx], bo)
    so = meta["sell_open"][idx].astype(np.float64)
    sl = meta["sell_low"][idx].astype(np.float64); sh = meta["sell_high"][idx].astype(np.float64)
    sc = meta["sell_close"][idx].astype(np.float64)
    r = np.empty(len(idx)); s = np.empty(len(idx), np.int8)
    m_gsl = so <= entry * (1 - STOP)
    m_gtp = (~m_gsl) & (so >= entry * (1 + TAKE))
    r[m_gsl] = so[m_gsl] / entry[m_gsl] - 1; s[m_gsl] = 1
    r[m_gtp] = so[m_gtp] / entry[m_gtp] - 1; s[m_gtp] = 2
    rest = ~(m_gsl | m_gtp)
    ri = np.where(rest)[0]
    hsl = sl[ri] <= entry[ri] * (1 - STOP)
    htp = (~hsl) & (sh[ri] >= entry[ri] * (1 + TAKE))
    r[ri[hsl]] = -STOP; s[ri[hsl]] = 1
    r[ri[htp]] = TAKE; s[ri[htp]] = 2
    neither = ~(hsl | htp)
    r[ri[neither]] = sc[ri[neither]] / entry[ri[neither]] - 1; s[ri[neither]] = 3
    net[idx] = r - FEE; st[idx] = s
    return net, st

def gap_aware_loop(meta, ii):
    """独立循环实现（抽查用），逐行镜像 build_gap_review。"""
    out = []
    for i in ii:
        sig = float(meta["signal_close"][i]); limit = sig * (1 - X_DISC)
        if not (float(meta["buy_low"][i]) <= limit):
            out.append((np.nan, 0)); continue
        entry = min(limit, float(meta["buy_open"][i]))
        so, sl, sh, sc = (float(meta[k][i]) for k in ("sell_open", "sell_low", "sell_high", "sell_close"))
        if so <= entry * (1 - STOP): r, s = so / entry - 1, 1
        elif so >= entry * (1 + TAKE): r, s = so / entry - 1, 2
        elif sl <= entry * (1 - STOP): r, s = -STOP, 1
        elif sh >= entry * (1 + TAKE): r, s = TAKE, 2
        else: r, s = sc / entry - 1, 3
        out.append((r - FEE, s))
    a = np.array(out)
    return a[:, 0], a[:, 1].astype(np.int8)

def predict_batched(model, X, bs=500000):
    ps = []
    for i in range(0, len(X), bs):
        ps.append(model.predict_proba(X[i:i + bs])[:, 1].astype(np.float64))
        if i % (bs * 4) == 0: print(f"  pred {i}/{len(X)}", flush=True)
    return np.concatenate(ps)

def ranks(x):
    r = np.empty(len(x)); r[np.argsort(np.argsort(x))] = np.arange(len(x))
    return r

res = {"spec": {"x_disc": X_DISC, "stop": STOP, "take": TAKE, "fee": FEE,
                "rule": "买入=信号收盘×0.95限价; gap-aware: 买入跳空低开按开盘价成交, 卖出日开盘穿过止盈/止损按开盘价成交, 盘中双触按止损; Top10名单, 平均每笔按前3只中成交的统计",
                "train": "HGB(300树/lr0.05/深3/叶100/balanced/128bins), 同标签 y_strategy(买入成交且旧口径收益≥+5%), 同行(50%系统抽样, 两臂行完全一致)"}}

# ============ 验证集：IC / RankIC / ICIR ============
Xv, mv = load_split("val")
print(f"val X {Xv.shape}", flush=True)
net_v, _ = gap_aware_net(mv)
# 独立抽查：循环实现 vs 向量化
rng = np.random.default_rng(7)
ii = rng.choice(len(Xv), 2000, replace=False)
nv2, st2 = gap_aware_loop(mv, ii)
assert np.allclose(net_v[ii], nv2, equal_nan=True), "gap_aware vectorized != loop!"
res["spotcheck"] = "passed: vectorized gap_aware == independent loop on 2000 rows"
print("spotcheck passed", flush=True)
del nv2, st2; gc.collect()

pv_b = predict_batched(m_base, Xv[:, :29]); gc.collect()
pv_g = predict_batched(m_gate, Xv); del Xv; gc.collect()

day = mv["date"].astype("datetime64[D]").astype(np.int64)
bought_m = np.isfinite(net_v)
res["val"] = {"n_rows": int(len(mv["code"])), "n_bought": int(bought_m.sum())}
pairs = []  # (ic_b, ric_b, ic_g, ric_g) per day, both arms finite
for d in np.unique(day):
    m_ = (day == d) & bought_m
    if m_.sum() < 10: continue
    y_ = net_v[m_]
    ib = float(np.corrcoef(pv_b[m_], y_)[0, 1]); rb = float(np.corrcoef(ranks(pv_b[m_]), ranks(y_))[0, 1])
    ig = float(np.corrcoef(pv_g[m_], y_)[0, 1]); rg = float(np.corrcoef(ranks(pv_g[m_]), ranks(y_))[0, 1])
    if all(np.isfinite([ib, rb, ig, rg])):
        pairs.append((ib, rb, ig, rg))
pairs = np.array(pairs)
for arm, ic, ric in (("baseline", pairs[:, 0], pairs[:, 1]), ("gated", pairs[:, 2], pairs[:, 3])):
    auc = float(roc_auc_score(mv["y"].astype(int), pv_b if arm == "baseline" else pv_g))
    res["val"][arm] = {"n_days": int(len(ic)),
        "IC": round(float(np.mean(ic)), 5), "RankIC": round(float(np.mean(ric)), 5),
        "ICIR": round(float(np.mean(ic) / (np.std(ic, ddof=1) + 1e-12)), 4),
        "AUC_vs_y": round(auc, 5)}
    print(f"val {arm}: IC={np.mean(ic):.5f} RankIC={np.mean(ric):.5f} "
          f"ICIR={np.mean(ic)/np.std(ic, ddof=1):.4f} AUC={auc:.5f} days={len(ic)}", flush=True)
t_ic, p_ic = ttest_rel(pairs[:, 2], pairs[:, 0])
t_ric, p_ric = ttest_rel(pairs[:, 3], pairs[:, 1])
res["val"]["delta_IC"] = round(res["val"]["gated"]["IC"] - res["val"]["baseline"]["IC"], 5)
res["val"]["delta_RankIC"] = round(res["val"]["gated"]["RankIC"] - res["val"]["baseline"]["RankIC"], 5)
res["val"]["paired_t_IC"] = {"t": round(float(t_ic), 3), "p": round(float(p_ic), 4)}
res["val"]["paired_t_RankIC"] = {"t": round(float(t_ric), 3), "p": round(float(p_ric), 4)}
json.dump({"val": res["val"], "spec": res["spec"], "spotcheck": res["spotcheck"]},
          open(os.path.join(OUT, "t1_val_partial.json"), "w"), indent=1)
print("val checkpoint saved", flush=True)
del mv, net_v, pv_b, pv_g, day, bought_m, pairs; gc.collect()

# ============ 测试集：组合回测 ============
Xt, mt = load_split("test")
print(f"test X {Xt.shape}", flush=True)
pt_b = predict_batched(m_base, Xt[:, :29]); gc.collect()
pt_g = predict_batched(m_gate, Xt); del Xt; gc.collect()
dayt = mt["date"].astype("datetime64[D]").astype(np.int64)
sig = mt["signal_close"].astype(np.float64)
limit = sig * (1 - X_DISC)
bought_t = mt["buy_low"] <= limit
bo = mt["buy_open"].astype(np.float64); entry_all = np.minimum(limit, bo)
so = mt["sell_open"].astype(np.float64); sl = mt["sell_low"].astype(np.float64)
sh = mt["sell_high"].astype(np.float64); sc = mt["sell_close"].astype(np.float64)

def trade_outcome(ii):
    """对给定行索引（已按排名排好）做 gap-aware 逐笔，返回 (net, status)。"""
    b = bought_t[ii]
    net = np.full(len(ii), np.nan); st = np.zeros(len(ii), np.int8)
    j = np.where(b)[0]
    en = entry_all[ii][j]
    so_, sl_, sh_, sc_ = so[ii][j], sl[ii][j], sh[ii][j], sc[ii][j]
    r = np.empty(len(j)); s = np.empty(len(j), np.int8)
    m_gsl = so_ <= en * (1 - STOP); m_gtp = (~m_gsl) & (so_ >= en * (1 + TAKE))
    r[m_gsl] = so_[m_gsl] / en[m_gsl] - 1; s[m_gsl] = 1
    r[m_gtp] = so_[m_gtp] / en[m_gtp] - 1; s[m_gtp] = 2
    rest = ~(m_gsl | m_gtp); ri = np.where(rest)[0]
    hsl = sl_[ri] <= en[ri] * (1 - STOP); htp = (~hsl) & (sh_[ri] >= en[ri] * (1 + TAKE))
    r[ri[hsl]] = -STOP; s[ri[hsl]] = 1; r[ri[htp]] = TAKE; s[ri[htp]] = 2
    neither = ~(hsl | htp); r[ri[neither]] = sc_[ri[neither]] / en[ri[neither]] - 1; s[ri[neither]] = 3
    net[j] = r - FEE; st[j] = s
    return net, st

bt_rows = []
daily_series = {}
for arm, p in (("baseline", pt_b), ("gated", pt_g)):
    daily = []
    for d in np.unique(dayt):
        m_ = np.where(dayt == d)[0]
        if len(m_) < 10: continue
        top10 = m_[np.argsort(-p[m_], kind="stable")[:10]]
        net10, st10 = trade_outcome(top10)
        top3 = np.arange(10)[:3]
        fb = top3[np.isfinite(net10[top3])]
        dm = float(np.mean(net10[fb])) if len(fb) else np.nan
        daily.append({"day": int(d), "mean_top3": dm,
                      "n_top3_filled": int(len(fb)),
                      "tp": int(((st10[:10] == 2)).sum()), "sl": int((st10[:10] == 1).sum()),
                      "close": int((st10[:10] == 3).sum()), "unfilled": int((st10[:10] == 0).sum())})
    import pandas as pd
    dd = pd.DataFrame(daily)
    dd.to_csv(os.path.join(OUT, f"t1_backtest_daily_{arm}.csv"), index=False)
    v = dd["mean_top3"].dropna().to_numpy()
    n = len(v); mean = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n))
    ntr = int(dd["n_top3_filled"].sum())
    stat = {"n_days": n, "n_trades_top3": ntr,
            "pooled_mean_per_trade": None,  # 用日度均值口径（与 search_70pct 一致）
            "daily_mean_top3": round(mean, 5), "se": round(se, 6),
            "lower70_one_sided": round(mean - 0.524 * se, 5),
            "tp_rate_top10": round(float((dd["tp"] / 10).mean()), 4),
            "sl_rate_top10": round(float((dd["sl"] / 10).mean()), 4),
            "unfilled_rate_top10": round(float((dd["unfilled"] / 10).mean()), 4)}
    bt_rows.append((arm, stat))
    daily_series[arm] = v
    print(f"backtest {arm}: days={n} trades={ntr} mean={mean:.5f} lower70={mean-0.524*se:.5f} "
          f"tp={stat['tp_rate_top10']:.3f} sl={stat['sl_rate_top10']:.3f} unfilled={stat['unfilled_rate_top10']:.3f}", flush=True)

res["test_backtest"] = {"baseline": bt_rows[0][1], "gated": bt_rows[1][1]}
gb, gg = bt_rows[0][1], bt_rows[1][1]
res["test_backtest"]["delta_daily_mean"] = round(gg["daily_mean_top3"] - gb["daily_mean_top3"], 5)
res["test_backtest"]["delta_lower70"] = round(gg["lower70_one_sided"] - gb["lower70_one_sided"], 5)
vb, vg = daily_series["baseline"], daily_series["gated"]
assert len(vb) == len(vg), "day series length mismatch"
mk = np.isfinite(vb) & np.isfinite(vg)
t_bt, p_bt = ttest_rel(vg[mk], vb[mk])
res["test_backtest"]["paired_t_daily_mean"] = {"t": round(float(t_bt), 3), "p": round(float(p_bt), 4),
                                               "n_paired_days": int(mk.sum())}
res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(OUT, "t1_result.json"), "w"), indent=1)
print("EVAL DONE", json.dumps(res, indent=1)[:2000], flush=True)
