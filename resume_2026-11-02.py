"""续跑 2026-11-02 日报：复用 daily_pipeline 的函数与配置。
背景：当日 17:00 定时任务首次运行在复核阶段被 SIGTERM 杀死（非脚本错误）。
已完成且不再重做的步骤：行情抓取、特征计算、当日打分、增量标签追加（10-29
批次 4931 行，已写入 incremental_flow.pkl）、HGB 重建（daily_model_2026-11-02.pkl
已生成并通过 load-back 校验，daily_model_pointer.txt 已更新）。
本脚本只做：打分复算（与死掉的运行完全相同的旧模型与流程）-> 市盈率抓取
（接口已恢复）-> 三策略复核 -> m_compare 累计 -> output JSON + 邮件文本。
输出生成块从 daily_pipeline.py 按标记原文提取并执行，保证格式与 main() 一致。
"""
import pickle, os, sys, json, time, datetime, textwrap
import numpy as np, pandas as pd

BASE_DIR = os.path.expanduser("~/workspace/ashare")
sys.path.insert(0, BASE_DIR)
import daily_pipeline as P
import strategy_config as SC

BASE = P.BASE
day = "2026-11-02"
t0 = time.time()

# --- 行情 / ok（与 main 一致） ---
bars = pickle.load(open(os.path.join(BASE, "daily_bars", f"{day}.pkl"), "rb"))
ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
print(f"resume: bars ok {len(ok)}", flush=True)
assert len(ok) >= 1000, "not a trading day?!"

# --- 特征 ---
feat = pickle.load(open(os.path.join(BASE, "daily_features", f"{day}.pkl"), "rb"))
print(f"resume: features rows {len(feat)}", flush=True)

# --- 当日打分：死掉的运行在指针更新前打分，用的是旧模型 ---
model_path = os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
mdl = pickle.load(open(model_path, "rb"))
feat["prob"] = mdl.predict_proba(feat[P.FEATS].to_numpy(np.float32))[:, 1]

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

# --- m=2/m=3 EV 打分（与 main 逐行一致） ---
m2_top10, m3_top10 = [], []
try:
    ev2 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_2.pkl"), "rb"))
    feat["_ev2"] = ev2.predict(feat[P.FEATS].to_numpy(np.float32)); del ev2
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
    feat["_ev3"] = ev3.predict(feat[P.FEATS].to_numpy(np.float32)); del ev3
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

# --- 市盈率（与 main 一致；死掉的运行当时接口故障取回 0/30，现已恢复） ---
top_list = []
try:
    pe_map = P.fetch_pe_robust([str(r["code"]) for _, r in top.iterrows()] +
                      [t["code"] for t in m2_top10] + [t["code"] for t in m3_top10])
except Exception:
    pe_map = {}
print(f"resume: PE fetched {len(pe_map)}/30", flush=True)
for t in m2_top10 + m3_top10:
    _pe = pe_map.get(t["code"])
    t["pe"] = float(_pe) if _pe is not None else None
for _, r in top.iterrows():
    c = float(r["close"])
    _pe = pe_map.get(str(r["code"]))
    top_list.append({"code": r["code"], "name": names.get(r["code"], ""), "signal_close": c,
                     "stop_price": round(c * 0.98, 2), "target_price": round(c * 1.05, 2),
                     "prob_ge5": float(r["prob"]), "pe": (float(_pe) if _pe is not None else None)})

# --- 增量标签：死掉的运行已追加，不再重做，只统计 ---
inc = pickle.load(open(os.path.join(BASE, "incremental_flow.pkl"), "rb"))
inc_new = int((inc["date"].astype(str) == "2026-10-29").sum())
print(f"resume: incremental total {len(inc)}, new(2026-10-29) {inc_new}", flush=True)
assert inc_new > 0, "incremental rows for 2026-10-29 missing!"

# --- 重建状态：死掉的运行已完成（模型文件+指针+load-back校验） ---
new_model = os.path.join(BASE, f"daily_model_{day}.pkl")
assert os.path.exists(new_model), "rebuilt model missing!"
pickle.load(open(new_model, "rb"))  # load-back 校验
ptr = os.path.join(BASE, "daily_model_pointer.txt")
assert open(ptr).read().strip() == new_model, "pointer not updated!"
retrain_status = "ok"
print("resume: retrain ok, pointer ->", new_model, flush=True)

# --- 交易日序列 ---
sse_saved = json.load(open(os.path.join(BASE, "sse_daily.json")))
trading_days = sorted([d for d in sse_saved.keys() if d <= day])
idx = trading_days.index(day)

def _bars_of(d):
    if d == day:
        return bars
    p = os.path.join(BASE, "daily_bars", f"{d}.pkl")
    return pickle.load(open(p, "rb")) if os.path.exists(p) else {}

# --- 原策略复核（与 main 一致） ---
review = {"status": "no_archive"}
if idx >= 2:
    sig_day = trading_days[idx - 2]; buy_day = trading_days[idx - 1]
    arch = os.path.join(BASE, "daily_output", f"{sig_day}.json")
    if os.path.exists(arch):
        try:
            old_top = json.load(open(arch))["top10"]
            bp = os.path.join(BASE, "daily_bars", f"{buy_day}.pkl")
            bars_buy = pickle.load(open(bp, "rb")) if os.path.exists(bp) else {}
            try:
                rev_pe = P.fetch_pe_robust([t["code"] for t in old_top])
            except Exception:
                rev_pe = {}
            review = P.build_review(sig_day, buy_day, day, old_top, bars_buy, bars, rev_pe)
        except Exception as e:
            review = {"status": f"failed: {type(e).__name__}"}
    else:
        review = {"status": "no_archive", "signal_day": sig_day}

# --- m2/m3 复核（与 main 一致） ---
m2_review, m3_review = {"status": "no_archive"}, {"status": "no_archive"}
if idx >= 3:
    _s2, _b2 = trading_days[idx - 3], trading_days[idx - 2]
    _h2 = [trading_days[idx - 1], day]
    _a2 = os.path.join(BASE, "daily_output", f"{_s2}.json")
    if os.path.exists(_a2):
        try:
            _old2 = json.load(open(_a2)).get("m2_top10", [])
            try:
                _pe2 = P.fetch_pe_robust([t["code"] for t in _old2])
            except Exception:
                _pe2 = {}
            m2_review = P.build_gap_review(_s2, _b2, _h2, _old2,
                                           {d: _bars_of(d) for d in [_b2] + _h2},
                                           SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe2, tag="m2")
        except Exception as e:
            m2_review = {"status": f"failed: {type(e).__name__}"}
    else:
        m2_review = {"status": "no_archive", "signal_day": _s2}
if idx >= 4:
    _s3, _b3 = trading_days[idx - 4], trading_days[idx - 3]
    _h3 = [trading_days[idx - 2], trading_days[idx - 1], day]
    _a3 = os.path.join(BASE, "daily_output", f"{_s3}.json")
    if os.path.exists(_a3):
        try:
            _old3 = json.load(open(_a3)).get("m3_top10", [])
            try:
                _pe3 = P.fetch_pe_robust([t["code"] for t in _old3])
            except Exception:
                _pe3 = {}
            m3_review = P.build_gap_review(_s3, _b3, _h3, _old3,
                                           {d: _bars_of(d) for d in [_b3] + _h3},
                                           SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe3, tag="m3")
        except Exception as e:
            m3_review = {"status": f"failed: {type(e).__name__}"}
    else:
        m3_review = {"status": "no_archive", "signal_day": _s3}

print("resume: reviews done, executing verbatim output block from daily_pipeline.py", flush=True)

# --- 输出生成块：从 daily_pipeline.py 按标记原文提取执行（与 main 完全一致） ---
_src = open(os.path.join(BASE, "daily_pipeline.py"), encoding="utf-8").read().splitlines()
_start = next(i for i, l in enumerate(_src) if "# 三策略累计对比" in l)
_end = next(i for i, l in enumerate(_src) if "DAILY DONE" in l)
_tail = textwrap.dedent("\n".join(_src[_start:_end + 1]))
exec(compile(_tail, "daily_pipeline_tail", "exec"))
print("resume: DONE", flush=True)
