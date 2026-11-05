# 2026-11-04 日报续跑脚本：定时 run 在 PE 抓取后被运行时重启终止。
# 已完成阶段（抓行情/提特征）不再重做；从打分开始续跑，并补做增量标签+重建。
# 顺序与 daily_pipeline.py main() 完全一致：打分（用旧指针模型）→PE→增量标签→重建→复核→m_compare→output JSON→email txt。
import pickle, os, sys, json, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
os.chdir(os.path.expanduser("~/workspace/ashare"))
import strategy_config as SC
import daily_pipeline as DP

BASE = os.path.expanduser("~/workspace/ashare")
FEATS = DP.FEATS
day = "2026-11-04"
t0 = time.time()

bars = pickle.load(open(os.path.join(BASE, "daily_bars", f"{day}.pkl"), "rb"))
ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
print(f"day {day} bars ok {len(ok)}", flush=True)
assert len(ok) >= 1000, "NOT_TRADING_DAY"

feat = pickle.load(open(os.path.join(BASE, "daily_features", f"{day}.pkl"), "rb"))
print(f"features {day} rows {len(feat)}", flush=True)

# 当前模型指针（重建前打分必须用旧模型，与原脚本顺序一致）
ptr = os.path.join(BASE, "daily_model_pointer.txt")
model_path = open(ptr).read().strip() if os.path.exists(ptr) else os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
mdl = pickle.load(open(model_path, "rb"))
print(f"scoring with {model_path}", flush=True)
feat["prob"] = mdl.predict_proba(feat[FEATS].to_numpy(np.float32))[:, 1]

names = {}
try:
    lst = pd.read_csv(os.path.join(BASE, "sina_hs_a_20261025.csv"))
    names = dict(zip(lst["symbol"], lst["name"]))
except Exception:
    pass

_sorted = feat.sort_values(["prob", "code"], ascending=[False, True])
_keep = []
for _, _r in _sorted.iterrows():
    _nm = str(names.get(_r["code"], ""))
    if ("退" in _nm) or ("ST" in _nm):
        continue
    _keep.append(_r)
    if len(_keep) >= 10:
        break
top = pd.DataFrame(_keep)

m2_top10, m3_top10 = [], []
try:
    ev2 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_2.pkl"), "rb"))
    feat["_ev2"] = ev2.predict(feat[FEATS].to_numpy(np.float32))
    del ev2
    _s2 = feat.sort_values(["_ev2", "code"], ascending=[False, True])
    for _, _r in _s2.iterrows():
        _nm = str(names.get(_r["code"], ""))
        if ("退" in _nm) or ("ST" in _nm):
            continue
        c = float(_r["close"])
        m2_top10.append({"code": _r["code"], "name": names.get(_r["code"], ""),
                         "signal_close": c, "limit_price": round(c * 0.95, 2),
                         "ev_score": float(_r["_ev2"]), "pe": None})
        if len(m2_top10) >= 10:
            break
except Exception as e:
    print(f"m2 scoring failed: {type(e).__name__}", flush=True)
try:
    ev3 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_3.pkl"), "rb"))
    feat["_ev3"] = ev3.predict(feat[FEATS].to_numpy(np.float32))
    del ev3
    _s3 = feat.sort_values(["_ev3", "code"], ascending=[False, True])
    for _, _r in _s3.iterrows():
        _nm = str(names.get(_r["code"], ""))
        if ("退" in _nm) or ("ST" in _nm):
            continue
        c = float(_r["close"])
        m3_top10.append({"code": _r["code"], "name": names.get(_r["code"], ""),
                         "signal_close": c, "limit_price": round(c * 0.95, 2),
                         "ev_score": float(_r["_ev3"]), "pe": None})
        if len(m3_top10) >= 10:
            break
except Exception as e:
    print(f"m3 scoring failed: {type(e).__name__}", flush=True)

top_list = []
try:
    pe_map = DP.fetch_pe_robust([str(r["code"]) for _, r in top.iterrows()] +
                                [t["code"] for t in m2_top10] + [t["code"] for t in m3_top10])
except Exception:
    pe_map = {}
for t in m2_top10 + m3_top10:
    _pe = pe_map.get(t["code"])
    t["pe"] = float(_pe) if _pe is not None else None
for _, r in top.iterrows():
    c = float(r["close"])
    _pe = pe_map.get(str(r["code"]))
    _lim = round(c * (1 - SC.ORIG_X_DISC), 2)
    top_list.append({"code": r["code"], "name": names.get(r["code"], ""), "signal_close": c,
                     "limit_price": _lim, "stop_price": round(_lim * (1 - SC.ORIG_STOP), 2),
                     "target_price": round(_lim * (1 + SC.ORIG_TAKE), 2),
                     "prob_ge5": float(r["prob"]),
                     "pe": (float(_pe) if _pe is not None else None)})
print(f"top10={len(top_list)} m2={len(m2_top10)} m3={len(m3_top10)}", flush=True)

# 增量标签（与原脚本逐字一致）：信号日=2个交易日前
sse_saved = json.load(open(os.path.join(BASE, "sse_daily.json")))
trading_days = sorted([d for d in sse_saved.keys() if d <= day])
idx = trading_days.index(day)
print(f"trading_days idx={idx} sig={trading_days[idx-2]} buy={trading_days[idx-1]}", flush=True)

inc_path = os.path.join(BASE, "incremental_flow.pkl")
inc = pickle.load(open(inc_path, "rb")) if os.path.exists(inc_path) else pd.DataFrame()
inc_new = 0
if idx >= 2:
    sig_day = trading_days[idx-2]; buy_day = trading_days[idx-1]; sell_day = day
    def bars_of(d):
        if d == day:
            return bars
        p = os.path.join(BASE, "daily_bars", f"{d}.pkl")
        return pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    sb, bb, lb = bars_of(sig_day), bars_of(buy_day), bars_of(sell_day)
    sf_path = os.path.join(BASE, "daily_features", f"{sig_day}.pkl")
    if os.path.exists(sf_path) and sb and bb and lb:
        sf = pickle.load(open(sf_path, "rb"))
        recs = []
        for _, r in sf.iterrows():
            code = r["code"]
            if code not in sb or code not in bb or code not in lb:
                continue
            _sb, _bb, _lb = sb[code], bb[code], lb[code]
            if _sb is None or _bb is None or _lb is None:
                continue
            entry = float(r["close"]); buy_low = float(_bb["low"]); sell_low = float(_lb["low"]); sell_high = float(_lb["high"]); sell_close = float(_lb["adjclose"])
            bought = buy_low <= entry
            if not bought:
                realized = 0.0
            elif sell_low <= entry * 0.98:
                realized = -0.02
            elif sell_high >= entry * 1.05:
                realized = 0.05
            else:
                realized = sell_close / entry - 1
            rec = {f: r[f] for f in FEATS}
            rec.update({"code": code, "date": sig_day, "signal_close": entry, "buy_low": buy_low,
                        "sell_low": sell_low, "sell_high": sell_high, "sell_close": sell_close,
                        "bought": bool(bought), "realized_ret": float(realized),
                        "y_strategy": int(bool(bought and realized >= 0.05))})
            recs.append(rec)
        if recs:
            newdf = pd.DataFrame(recs)
            if len(inc):
                inc = pd.concat([inc, newdf], ignore_index=True)
            else:
                inc = newdf
            inc_new = len(newdf)
            inc.to_pickle(inc_path)
print(f"incremental_new_rows={inc_new} incremental_total_rows={len(inc)}", flush=True)

# 重建模型（与原脚本逐字一致）：基数组+增量行；失败则保留旧模型
retrain_status = "skipped_no_new_labels" if inc_new == 0 else "failed_keep_old"
base_dir = os.path.join(BASE, "daily_base_arrays")
if inc_new > 0 and os.path.exists(os.path.join(base_dir, "train_X.npy")):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtv = np.concatenate([np.load(os.path.join(base_dir, "train_X.npy")),
                              np.load(os.path.join(base_dir, "val_X.npy")),
                              inc[FEATS].to_numpy(np.float32)])
        ytv = np.concatenate([np.load(os.path.join(base_dir, "train_y.npy")),
                              np.load(os.path.join(base_dir, "val_y.npy")),
                              inc["y_strategy"].to_numpy(np.int8)])
        m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3,
                                            min_samples_leaf=100, class_weight="balanced",
                                            early_stopping=False, max_bins=128)
        m2.fit(Xtv, ytv)
        del Xtv, ytv
        new_path = os.path.join(BASE, f"daily_model_{day}.pkl")
        with open(new_path + ".tmp", "wb") as f:
            pickle.dump(m2, f)
        os.rename(new_path + ".tmp", new_path)
        pickle.load(open(new_path, "rb"))  # load-back校验
        open(ptr, "w").write(new_path)
        model_path = new_path
        retrain_status = "ok"
    except Exception as e:
        retrain_status = f"failed_keep_old: {type(e).__name__}"
print(f"model {model_path} retrain_status={retrain_status}", flush=True)

# 复核（与原脚本一致）
review = {"status": "no_archive"}
if idx >= 2:
    sig_day = trading_days[idx-2]; buy_day = trading_days[idx-1]
    arch = os.path.join(BASE, "daily_output", f"{sig_day}.json")
    if os.path.exists(arch):
        try:
            old_top = json.load(open(arch))["top10"]
            bp = os.path.join(BASE, "daily_bars", f"{buy_day}.pkl")
            bars_buy = pickle.load(open(bp, "rb")) if os.path.exists(bp) else {}
            try:
                rev_pe = DP.fetch_pe_robust([t["code"] for t in old_top])
            except Exception:
                rev_pe = {}
            review = DP.build_review(sig_day, buy_day, day, old_top, bars_buy, bars, rev_pe)
        except Exception as e:
            review = {"status": f"failed: {type(e).__name__}"}
    else:
        review = {"status": "no_archive", "signal_day": sig_day}
print(f"review: sig={review.get('signal_day')} win={review.get('win_count')} bought={review.get('bought_count')}", flush=True)

def _bars_of(d):
    if d == day:
        return bars
    p = os.path.join(BASE, "daily_bars", f"{d}.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}

m2_review, m3_review = {"status": "no_archive"}, {"status": "no_archive"}
if idx >= 3:
    _s2, _b2 = trading_days[idx-3], trading_days[idx-2]
    _h2 = [trading_days[idx-1], day]
    _a2 = os.path.join(BASE, "daily_output", f"{_s2}.json")
    if os.path.exists(_a2):
        try:
            _old2 = json.load(open(_a2)).get("m2_top10", [])
            try:
                _pe2 = DP.fetch_pe_robust([t["code"] for t in _old2])
            except Exception:
                _pe2 = {}
            m2_review = DP.build_gap_review(_s2, _b2, _h2, _old2,
                                            {d: _bars_of(d) for d in [_b2] + _h2},
                                            SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe2, tag="m2")
        except Exception as e:
            m2_review = {"status": f"failed: {type(e).__name__}"}
    else:
        m2_review = {"status": "no_archive", "signal_day": _s2}
if idx >= 4:
    _s3, _b3 = trading_days[idx-4], trading_days[idx-3]
    _h3 = [trading_days[idx-2], trading_days[idx-1], day]
    _a3 = os.path.join(BASE, "daily_output", f"{_s3}.json")
    if os.path.exists(_a3):
        try:
            _old3 = json.load(open(_a3)).get("m3_top10", [])
            try:
                _pe3 = DP.fetch_pe_robust([t["code"] for t in _old3])
            except Exception:
                _pe3 = {}
            m3_review = DP.build_gap_review(_s3, _b3, _h3, _old3,
                                            {d: _bars_of(d) for d in [_b3] + _h3},
                                            SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe3, tag="m3")
        except Exception as e:
            m3_review = {"status": f"failed: {type(e).__name__}"}
    else:
        m3_review = {"status": "no_archive", "signal_day": _s3}
print(f"m2_review bought={m2_review.get('bought_count')} m3_review bought={m3_review.get('bought_count')}", flush=True)

cmp_path = os.path.join(BASE, "m_compare.json")
cmpd = json.load(open(cmp_path)) if os.path.exists(cmp_path) else \
    {"orig": {"n": 0, "sum_r": 0.0}, "m2": {"n": 0, "sum_r": 0.0}, "m3": {"n": 0, "sum_r": 0.0}}
for _key, _rv in (("orig", review), ("m2", m2_review), ("m3", m3_review)):
    if _rv.get("items"):
        for _it in _rv["items"][:SC.N_STRATEGY]:
            if _it.get("status") not in ("未成交", "数据缺失") and _it.get("realized") is not None:
                cmpd[_key]["n"] += 1
                cmpd[_key]["sum_r"] += float(_it["realized"])
json.dump(cmpd, open(cmp_path, "w"))

# 尾部：直接取 daily_pipeline.py main() 尾部源码（_cmp_line 定义 → email 写盘）exec 执行，
# 保证 output JSON 与邮件正文与原脚本逐字一致。
import textwrap
_src = open(os.path.join(BASE, "daily_pipeline.py"), encoding="utf-8").read()
_start = _src.index("    def _cmp_line(key, label):")
_end_marker = "    open(os.path.join(BASE, \"daily_output\", f\"{day}_email.txt\"), \"w\", encoding=\"utf-8\").write(\"\\n\".join(lines))"
_end = _src.index(_end_marker) + len(_end_marker)
_tail = textwrap.dedent(_src[_start:_end])
_ns = {"os": os, "json": json, "SC": SC, "BASE": BASE, "day": day,
       "model_path": model_path, "retrain_status": retrain_status,
       "ok": ok, "feat": feat, "inc_new": inc_new, "inc": inc,
       "top_list": top_list, "m2_top10": m2_top10, "m3_top10": m3_top10,
       "review": review, "m2_review": m2_review, "m3_review": m3_review,
       "cmpd": cmpd}
exec(compile(_tail, "daily_pipeline_tail", "exec"), _ns)
print(f"RESUME DONE {day} top10={len(top_list)} retrain={retrain_status} elapsed {time.time()-t0:.0f}s", flush=True)
