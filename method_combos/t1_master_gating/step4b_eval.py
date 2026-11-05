# T1 step4b: 评估（只读 probs + meta，轻量）
# 口径：买入=信号收盘×0.95 限价（x=0.05），+7% 止盈 / -2% 止损，0.5% 往返费，
# gap-aware（买入跳空低开按开盘价成交；卖出日开盘穿过止盈/止损按开盘价成交；盘中双触按止损），
# Top10 名单、平均每笔按前 3 只中成交的统计。
# 验证集：IC/RankIC/ICIR（日度截面，相对 gap-aware 净收益）+ AUC（相对训练标签 y）
# 测试集：组合回测；70% 单侧下界 = 均值-0.524*SE（日度周期为单位）
import os, time, json, gc, pickle, numpy as np, pandas as pd
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
from sklearn.metrics import roc_auc_score
from scipy.stats import ttest_rel
t0 = time.time()

X_DISC, STOP, TAKE, FEE = 0.05, 0.02, 0.07, 0.005

def load_meta(s):
    ms = [np.load(os.path.join(CHK, f"t1_meta_{s}_{fi:02d}.npz"), allow_pickle=True)
          for fi in range(25) if os.path.exists(os.path.join(CHK, f"t1_meta_{s}_{fi:02d}.npz"))]
    return {k: np.concatenate([m[k] for m in ms]) for k in ms[0].files}

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

def ranks(x):
    r = np.empty(len(x)); r[np.argsort(np.argsort(x))] = np.arange(len(x))
    return r

res = {"spec": {"x_disc": X_DISC, "stop": STOP, "take": TAKE, "fee": FEE,
    "rule": "买入=信号收盘×0.95限价; gap-aware: 买入跳空低开按开盘价成交, 卖出日开盘穿过止盈/止损按开盘价成交, 盘中双触按止损; Top10名单, 平均每笔按前3只中成交的统计",
    "train": "HGB(300树/lr0.05/深3/叶100/balanced/128bins), 同标签 y_strategy(买入成交且旧口径收益≥+5%), 同行(50%系统抽样, 两臂行完全一致)",
    "note": "全量 4.8M×79 训练两次 OOM(137)，改 50% 系统抽样（两臂同行）后训练成功"}}

# ============ 验证集 ============
mv = load_meta("val")
pv_b = np.load(os.path.join(OUT, "t1_prob_val_baseline.npy")).astype(np.float64)
pv_g = np.load(os.path.join(OUT, "t1_prob_val_gated.npy")).astype(np.float64)
assert len(pv_b) == len(mv["code"]) == len(pv_g)
net_v, _ = gap_aware_net(mv)
day = pd.to_datetime(mv["date"]).values.astype("datetime64[D]").astype(np.int64)
bought_m = np.isfinite(net_v)
pairs = []
for d in np.unique(day):
    m_ = (day == d) & bought_m
    if m_.sum() < 10: continue
    y_ = net_v[m_]
    xb, xg = pv_b[m_], pv_g[m_]
    if np.std(xb) == 0 or np.std(xg) == 0 or np.std(y_) == 0: continue
    ib = float(np.corrcoef(xb, y_)[0, 1]); rb = float(np.corrcoef(ranks(xb), ranks(y_))[0, 1])
    ig = float(np.corrcoef(xg, y_)[0, 1]); rg = float(np.corrcoef(ranks(xg), ranks(y_))[0, 1])
    if all(np.isfinite([ib, rb, ig, rg])):
        pairs.append((ib, rb, ig, rg))
pairs = np.array(pairs)
res["val"] = {"n_rows": int(len(mv["code"])), "n_bought": int(bought_m.sum())}
for arm, ic, ric, p in (("baseline", pairs[:, 0], pairs[:, 1], pv_b),
                        ("gated", pairs[:, 2], pairs[:, 3], pv_g)):
    auc = float(roc_auc_score(mv["y"].astype(int), p))
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
del mv, net_v, pv_b, pv_g, day, bought_m, pairs; gc.collect()

# ============ 测试集回测 ============
mt = load_meta("test")
pt_b = np.load(os.path.join(OUT, "t1_prob_test_baseline.npy")).astype(np.float64)
pt_g = np.load(os.path.join(OUT, "t1_prob_test_gated.npy")).astype(np.float64)
dayt = pd.to_datetime(mt["date"]).values.astype("datetime64[D]").astype(np.int64)
sig = mt["signal_close"].astype(np.float64)
limit = sig * (1 - X_DISC)
bought_t = mt["buy_low"] <= limit
bo = mt["buy_open"].astype(np.float64); entry_all = np.minimum(limit, bo)
so = mt["sell_open"].astype(np.float64); sl = mt["sell_low"].astype(np.float64)
sh = mt["sell_high"].astype(np.float64); sc = mt["sell_close"].astype(np.float64)

def trade_outcome(ii):
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

bt_rows = []; daily_series = {}
for arm, p in (("baseline", pt_b), ("gated", pt_g)):
    daily = []
    for d in np.unique(dayt):
        m_ = np.where(dayt == d)[0]
        if len(m_) < 10: continue
        top10 = m_[np.argsort(-p[m_], kind="stable")[:10]]
        net10, st10 = trade_outcome(top10)
        fb = np.arange(3)[np.isfinite(net10[:3])]
        dm = float(np.mean(net10[fb])) if len(fb) else np.nan
        daily.append({"day": int(d), "mean_top3": dm, "n_top3_filled": int(len(fb)),
                      "tp": int((st10 == 2).sum()), "sl": int((st10 == 1).sum()),
                      "close": int((st10 == 3).sum()), "unfilled": int((st10 == 0).sum())})
    dd = pd.DataFrame(daily)
    dd.to_csv(os.path.join(OUT, f"t1_backtest_daily_{arm}.csv"), index=False)
    v = dd["mean_top3"].dropna().to_numpy()
    n = len(v); mean = float(v.mean()); se = float(v.std(ddof=1) / np.sqrt(n))
    stat = {"n_days": n, "n_trades_top3": int(dd["n_top3_filled"].sum()),
            "daily_mean_top3": round(mean, 5), "se": round(se, 6),
            "lower70_one_sided": round(mean - 0.524 * se, 5),
            "tp_rate_top10": round(float((dd["tp"] / 10).mean()), 4),
            "sl_rate_top10": round(float((dd["sl"] / 10).mean()), 4),
            "unfilled_rate_top10": round(float((dd["unfilled"] / 10).mean()), 4)}
    bt_rows.append((arm, stat)); daily_series[arm] = v
    print(f"backtest {arm}: days={n} trades={stat['n_trades_top3']} mean={mean:.5f} "
          f"lower70={mean-0.524*se:.5f} tp={stat['tp_rate_top10']:.3f} "
          f"sl={stat['sl_rate_top10']:.3f} unfilled={stat['unfilled_rate_top10']:.3f}", flush=True)

res["test_backtest"] = {"baseline": bt_rows[0][1], "gated": bt_rows[1][1]}
gb, gg = bt_rows[0][1], bt_rows[1][1]
res["test_backtest"]["delta_daily_mean"] = round(gg["daily_mean_top3"] - gb["daily_mean_top3"], 5)
res["test_backtest"]["delta_lower70"] = round(gg["lower70_one_sided"] - gb["lower70_one_sided"], 5)
vb, vg = daily_series["baseline"], daily_series["gated"]
assert len(vb) == len(vg)
mk = np.isfinite(vb) & np.isfinite(vg)
t_bt, p_bt = ttest_rel(vg[mk], vb[mk])
res["test_backtest"]["paired_t_daily_mean"] = {"t": round(float(t_bt), 3), "p": round(float(p_bt), 4),
                                               "n_paired_days": int(mk.sum())}
res["elapsed_s"] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(OUT, "t1_result.json"), "w"), indent=1)
print("EVAL DONE", flush=True)
print(json.dumps(res, indent=1))
