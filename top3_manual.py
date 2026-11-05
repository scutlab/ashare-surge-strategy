"""用户手动触发：拉取最新数据、全盘评估、取Top3、发163邮箱。"""
import pickle, glob, os, sys, json, gc, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategy_config as SC
from daily_pipeline import features_for_day, fetch_pe_robust, sse_close

BASE = os.path.dirname(os.path.abspath(__file__))
day = "2026-11-02"
FEATS = SC.FEATS

print("[%s] 加载最新行情..." % day, flush=True)
bars = pickle.load(open(os.path.join(BASE, "daily_bars", "%s.pkl" % day), "rb"))
ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
print("bars ok %d/%d" % (len(ok), len(bars)), flush=True)

midx = pickle.load(open(os.path.join(BASE, "index_sse.pkl"), "rb"))
sse_hist = {pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d, v in zip(midx["date"], midx["mkt_adjclose"])}
sse_today = sse_close(day)
sse_hist[day] = sse_today
sse_path = os.path.join(BASE, "sse_daily.json")
sse_saved = json.load(open(sse_path)) if os.path.exists(sse_path) else {}
sse_hist.update(sse_saved)
sse_hist[day] = sse_today
trading_days = sorted([d for d in sse_hist.keys() if d <= day])
idx = trading_days.index(day) if day in trading_days else -1
mkt_ret = sse_today / sse_hist[trading_days[idx-1]] - 1 if idx >= 1 else 0.0
mkt_ret5 = sse_today / sse_hist[trading_days[idx-5]] - 1 if idx >= 5 else 0.0
print("mkt_ret=%.4f mkt_ret5=%.4f" % (mkt_ret, mkt_ret5), flush=True)

print("加载历史行情...", flush=True)
parts = []
for fp in [os.path.join(BASE, "full_raw_partial.pkl"), os.path.join(BASE, "full_new_partial.pkl")]:
    if os.path.exists(fp):
        _df = pickle.load(open(fp, "rb"))
        parts.append(_df[["code", "date", "open", "high", "low", "close", "adjclose", "volume"]])
        del _df
for bf in sorted(glob.glob(os.path.join(BASE, "daily_bars", "*.pkl"))):
    bday = os.path.basename(bf)[:-4]
    if bday >= day:
        continue
    bd = pickle.load(open(bf, "rb"))
    rows = []
    for c, v in bd.items():
        if isinstance(v, dict):
            _r = {"code": c, "date": pd.Timestamp(bday)}
            _r.update(v)
            rows.append(_r)
    if rows:
        parts.append(pd.DataFrame(rows))
    del bd
raw = pd.concat(parts, ignore_index=True)
del parts
gc.collect()
raw["date"] = pd.to_datetime(raw["date"])

print("计算29特征...", flush=True)
t0 = time.time()
feat = features_for_day(raw, bars, day, mkt_ret, mkt_ret5)
del raw
gc.collect()
print("features rows %d elapsed %.0fs" % (len(feat), time.time()-t0), flush=True)

names = {}
try:
    lst = pd.read_csv(os.path.join(BASE, "sina_hs_a_20261025.csv"))
    names = dict(zip(lst["symbol"], lst["name"]))
except Exception:
    pass

def is_excluded(code):
    nm = str(names.get(code, ""))
    return ("退" in nm) or ("ST" in nm)

print("三策略打分...", flush=True)
ptr = os.path.join(BASE, "daily_model_pointer.txt")
if os.path.exists(ptr):
    model_path = open(ptr).read().strip()
else:
    model_path = os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
print("原策略模型: %s" % os.path.basename(model_path), flush=True)
mdl = pickle.load(open(model_path, "rb"))
feat["prob"] = mdl.predict_proba(feat[FEATS].to_numpy(np.float32))[:, 1]
del mdl
gc.collect()

ev2 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_2.pkl"), "rb"))
feat["_ev2"] = ev2.predict(feat[FEATS].to_numpy(np.float32))
del ev2
gc.collect()
ev3 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_3.pkl"), "rb"))
feat["_ev3"] = ev3.predict(feat[FEATS].to_numpy(np.float32))
del ev3
gc.collect()

def top3_by(col):
    s = feat.sort_values([col, "code"], ascending=[False, True])
    out = []
    for _, r in s.iterrows():
        if is_excluded(r["code"]):
            continue
        out.append(r)
        if len(out) >= 3:
            break
    return out

orig_top3 = top3_by("prob")
m2_top3 = top3_by("_ev2")
m3_top3 = top3_by("_ev3")

all_codes = list(set([str(r["code"]) for r in orig_top3 + m2_top3 + m3_top3]))
print("抓取市盈率 (%d只)..." % len(all_codes), flush=True)
pe_map = fetch_pe_robust(all_codes)

def fmt_stock(r, extra):
    code = str(r["code"])
    name = names.get(code, "")
    c = float(r["close"])
    pe = pe_map.get(code)
    pe_s = ("%.1f倍" % float(pe)) if pe is not None else "暂无"
    return {"code": code, "name": name, "signal_close": round(c, 2),
            "limit_price": round(c * 0.95, 2), "pe": pe_s, "extra": extra}

result = {
    "date": day,
    "orig_top3": [fmt_stock(r, "prob=%.4f" % float(r["prob"])) for r in orig_top3],
    "m2_top3": [fmt_stock(r, "ev=%.4f" % float(r["_ev2"])) for r in m2_top3],
    "m3_top3": [fmt_stock(r, "ev=%.4f" % float(r["_ev3"])) for r in m3_top3],
}
out_json = os.path.join(BASE, "daily_output", "top3_%s.json" % day)
json.dump(result, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("saved %s" % out_json, flush=True)

lines = []
lines.append("A股三策略Top3（信号日%s，收盘后评估）" % day)
lines.append("")
lines.append("统一规则：按信号收盘x0.95（95折）限价买入；+7%止盈；已扣0.5%往返手续费；剔除ST/退市股。")
lines.append("")
lines.append("【原策略Top3】（分类器；T+1买入，T+2卖出，-2%止损/+7%止盈）：")
for i, s in enumerate(result["orig_top3"], 1):
    lines.append("%d. %s %s，信号收盘%s元，限价%s元，市盈率%s（%s）" % (i, s["name"], s["code"], s["signal_close"], s["limit_price"], s["pe"], s["extra"]))
lines.append("")
lines.append("【m=2策略Top3】（EV回归器；持有至多2天，+7%止盈，无止损）：")
for i, s in enumerate(result["m2_top3"], 1):
    lines.append("%d. %s %s，信号收盘%s元，限价%s元，市盈率%s（%s）" % (i, s["name"], s["code"], s["signal_close"], s["limit_price"], s["pe"], s["extra"]))
lines.append("")
lines.append("【m=3策略Top3】（EV回归器；持有至多3天，+7%止盈，无止损）：")
for i, s in enumerate(result["m3_top3"], 1):
    lines.append("%d. %s %s，信号收盘%s元，限价%s元，市盈率%s（%s）" % (i, s["name"], s["code"], s["signal_close"], s["limit_price"], s["pe"], s["extra"]))
lines.append("")
lines.append("说明：以上为模型按最新行情全盘评估结果，仅供研究参考，不构成投资建议。限价单可能不成交；涨跌停排队、部分成交未建模。")

email_txt = "\n".join(lines)
out_email = os.path.join(BASE, "daily_output", "top3_%s_email.txt" % day)
open(out_email, "w", encoding="utf-8").write(email_txt)
print("saved %s" % out_email, flush=True)
print("DONE", flush=True)
