# 续跑脚本（2026-10-29）：复用当日已算好的 bars/features（daily_pipeline.py
# 在增量标签步骤因 None 检查缺失崩溃，特征文件完好：4933 行）。
# 本脚本精确复刻 daily_pipeline.py main() 中 features 写出之后
# （第 213 行起：打分→PE→增量标签→重建→复核→写出）的全部逻辑，
# 仅跳过“抓当日行情”和“算 29 特征”两步（改从当日 pkl 加载）。
import sys, os, time, json, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import daily_pipeline as dp
from daily_pipeline import BASE, FEATS, fetch_pe, build_review

day = "2026-10-29"
t0 = time.time()

# —— 与 main() 相同的输入，只是从已存盘的文件加载 ——
bars = pickle.load(open(os.path.join(BASE, "daily_bars", f"{day}.pkl"), "rb"))
ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
feat = pickle.load(open(os.path.join(BASE, "daily_features", f"{day}.pkl"), "rb"))

# —— 上证当日收盘/交易日序列（与 main() 同；sse_daily.json 已在崩溃前更新过，此处幂等重写） ——
midx = pickle.load(open(os.path.join(BASE, "index_sse.pkl"), "rb"))
sse_hist = {pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d, v in zip(midx["date"], midx["mkt_adjclose"])}
sse_today = dp.sse_close(day)
sse_hist[day] = sse_today
sse_path = os.path.join(BASE, "sse_daily.json"); sse_saved = json.load(open(sse_path)) if os.path.exists(sse_path) else {}
sse_hist.update(sse_saved); sse_hist[day] = sse_today
sse_saved.update(sse_hist); json.dump(sse_saved, open(sse_path, "w"))
trading_days = sorted([d for d in sse_saved.keys() if d <= day])
idx = trading_days.index(day) if day in trading_days else -1

# —— 以下与 main() 第 213 行起完全一致 ——
ptr = os.path.join(BASE, "daily_model_pointer.txt")
model_path = open(ptr).read().strip() if os.path.exists(ptr) else os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
mdl = pickle.load(open(model_path, "rb"))
feat["prob"] = mdl.predict_proba(feat[FEATS].to_numpy(np.float32))[:, 1]
names = {}
try:
    lst = pd.read_csv(os.path.join(BASE, "sina_hs_a_20261025.csv"))
    names = dict(zip(lst["symbol"], lst["name"]))
except Exception: pass
_sorted = feat.sort_values(["prob", "code"], ascending=[False, True])
_keep = []
for _, _r in _sorted.iterrows():
    _nm = str(names.get(_r["code"], ""))
    if ("退" in _nm) or ("ST" in _nm): continue
    _keep.append(_r)
    if len(_keep) >= 10: break
top = pd.DataFrame(_keep)
top_list = []
try:
    pe_map = fetch_pe([str(r["code"]) for _, r in top.iterrows()])
except Exception:
    pe_map = {}
for _, r in top.iterrows():
    c = float(r["close"])
    _pe = pe_map.get(str(r["code"]))
    top_list.append({"code": r["code"], "name": names.get(r["code"], ""), "signal_close": c, "stop_price": round(c * 0.98, 2), "target_price": round(c * 1.05, 2), "prob_ge5": float(r["prob"]), "pe": (float(_pe) if _pe is not None else None)})
# 增量标签（含 None 修复）
inc_path = os.path.join(BASE, "incremental_flow.pkl")
inc = pickle.load(open(inc_path, "rb")) if os.path.exists(inc_path) else pd.DataFrame()
inc_new = 0
if idx >= 2:
    sig_day = trading_days[idx-2]; buy_day = trading_days[idx-1]; sell_day = day
    def bars_of(d):
        if d == day: return bars
        p = os.path.join(BASE, "daily_bars", f"{d}.pkl")
        return pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    sb, bb, lb = bars_of(sig_day), bars_of(buy_day), bars_of(sell_day)
    sf_path = os.path.join(BASE, "daily_features", f"{sig_day}.pkl")
    if os.path.exists(sf_path) and sb and bb and lb:
        sf = pickle.load(open(sf_path, "rb"))
        recs = []
        for _, r in sf.iterrows():
            code = r["code"]
            if code not in sb or code not in bb or code not in lb: continue
            _sb, _bb, _lb = sb[code], bb[code], lb[code]
            if _sb is None or _bb is None or _lb is None: continue
            entry = float(r["close"]); buy_low = float(_bb["low"]); sell_low = float(_lb["low"]); sell_high = float(_lb["high"]); sell_close = float(_lb["adjclose"])
            bought = buy_low <= entry
            if not bought: realized = 0.0
            elif sell_low <= entry * 0.98: realized = -0.02
            elif sell_high >= entry * 1.05: realized = 0.05
            else: realized = sell_close / entry - 1
            rec = {f: r[f] for f in FEATS}; rec.update({"code": code, "date": sig_day, "signal_close": entry, "buy_low": buy_low, "sell_low": sell_low, "sell_high": sell_high, "sell_close": sell_close, "bought": bool(bought), "realized_ret": float(realized), "y_strategy": int(bool(bought and realized >= 0.05))})
            recs.append(rec)
        if recs:
            newdf = pd.DataFrame(recs)
            if len(inc): inc = pd.concat([inc, newdf], ignore_index=True)
            else: inc = newdf
            inc_new = len(newdf); inc.to_pickle(inc_path)
print(f"incremental new rows {inc_new}", flush=True)
# 重建模型：基数组（train+val）+ 增量行；失败则保留旧模型
retrain_status = "skipped_no_new_labels" if inc_new == 0 else "failed_keep_old"
base_dir = os.path.join(BASE, "daily_base_arrays")
if inc_new > 0 and os.path.exists(os.path.join(base_dir, "train_X.npy")):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xtv = np.concatenate([np.load(os.path.join(base_dir, "train_X.npy")), np.load(os.path.join(base_dir, "val_X.npy")), inc[FEATS].to_numpy(np.float32)])
        ytv = np.concatenate([np.load(os.path.join(base_dir, "train_y.npy")), np.load(os.path.join(base_dir, "val_y.npy")), inc["y_strategy"].to_numpy(np.int8)])
        m2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight="balanced", early_stopping=False, max_bins=128)
        m2.fit(Xtv, ytv); del Xtv, ytv
        new_path = os.path.join(BASE, f"daily_model_{day}.pkl")
        with open(new_path + ".tmp", "wb") as f: pickle.dump(m2, f)
        os.rename(new_path + ".tmp", new_path)
        pickle.load(open(new_path, "rb"))  # load-back校验
        open(ptr, "w").write(new_path); model_path = new_path; retrain_status = "ok"
    except Exception as e:
        retrain_status = f"failed_keep_old: {type(e).__name__}"
print(f"retrain {retrain_status} model {os.path.basename(model_path)}", flush=True)
# 复核：卖出日就是当日的那批推荐（信号日=2个交易日前），与当日Top10一起进邮件
review = {"status": "no_archive"}
if idx >= 2:
    sig_day = trading_days[idx-2]; buy_day = trading_days[idx-1]
    arch = os.path.join(BASE, "daily_output", f"{sig_day}.json")
    if os.path.exists(arch):
        try:
            old_top = json.load(open(arch))["top10"]
            bp = os.path.join(BASE, "daily_bars", f"{buy_day}.pkl")
            bars_buy = pickle.load(open(bp, "rb")) if os.path.exists(bp) else {}
            review = build_review(sig_day, buy_day, day, old_top, bars_buy, bars)
        except Exception as e:
            review = {"status": f"failed: {type(e).__name__}"}
    else:
        review = {"status": "no_archive", "signal_day": sig_day}
out = {"date": day, "trading_day": True, "bars_ok": len(ok), "features_rows": int(len(feat)), "model_path": model_path, "retrain_status": retrain_status, "incremental_new_rows": int(inc_new), "incremental_total_rows": int(len(inc)), "top10": top_list, "review": review}
with open(os.path.join(BASE, "daily_output", f"{day}.json"), "w") as f: json.dump(out, f, ensure_ascii=False, indent=2)
lines = [f"A股下一交易日Top10（信号日{day}，买入日为下一交易日、按信号收盘价挂牌，卖出日为再下一交易日-2%止损/+5%止盈，同日双触按止损算）", f"模型：{os.path.basename(model_path)}；重建状态：{retrain_status}；当日有效行情{len(ok)}只，特征{len(feat)}只。", "概率口径：模型预测的买入成交且卖出收益≥+5%的概率，不是正收益概率（正收益还含0~+5%区间，模型概率是其下界）；以下收益均为扣费前，未计费用、滑点和税，不构成投资建议。", "市盈率为新浪当日快照值（负值表示亏损），仅供参考，非模型特征；取不到时显示“市盈率暂无”。", ""]
for i, t in enumerate(top_list, 1):
    _pe_txt = f"市盈率{t['pe']:.1f}倍" if t.get("pe") is not None else "市盈率暂无"
    lines.append(f"{i}. {t['name']} {t['code']}，信号收盘{t['signal_close']:.2f}元，止损{t['stop_price']:.2f}元，止盈{t['target_price']:.2f}元，模型概率{t['prob_ge5']:.3f}，{_pe_txt}")
lines.append("")
if review.get("items"):
    lines.append(f"前两个交易日推荐复核（信号日{review['signal_day']}，买入日{review['buy_day']}，卖出日{review['sell_day']}，按-2%止损/+5%止盈、同日双触按止损算）：止盈{review['win_count']}只，成交{review['bought_count']}只。")
    for i in review["items"]:
        if i["status"] in ("止盈", "止损"):
            lines.append(f"- {i['name']} {i['code']}，信号收盘{i['signal_close']:.2f}元，{i['status']}（{i['realized']:+.0%}）")
        elif i["status"] == "收盘卖出":
            lines.append(f"- {i['name']} {i['code']}，信号收盘{i['signal_close']:.2f}元，收盘卖出（{i['realized']:+.2%}）")
        else:
            lines.append(f"- {i['name']} {i['code']}，信号收盘{i['signal_close']:.2f}元，{i['status']}")
elif review.get("signal_day"):
    lines.append(f"前两个交易日推荐复核：信号日{review['signal_day']}无存档（日报自10月27日开始），暂无可复核的推荐。")
else:
    lines.append("前两个交易日推荐复核：暂无（复核失败或数据不足，不影响当日Top10）。")
open(os.path.join(BASE, "daily_output", f"{day}_email.txt"), "w", encoding="utf-8").write("\n".join(lines))
print(f"DAILY DONE {day} top10={len(top_list)} retrain={retrain_status} elapsed {time.time()-t0:.0f}s", flush=True)
