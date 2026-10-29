# 工作日17点日线任务：取当日行情 -> 提当日信号特征 -> 增量标签 -> 重建模型 -> 下一交易日Top10
# 口径（与主项目一致）：
# - 信号日D收盘打分，买入日=下一交易日按D收盘价挂牌，卖出日=再下一交易日-2%止损/+5%止盈，同日双触按止损算
# - 模型概率 = 买入成交且卖出收益>=+5%的概率，不是正收益概率（正收益还含0~+5%区间，模型概率是其下界，不可等同）
# - 标签滞后2个交易日：当日只能为信号日=D-2个交易日的周期补标签（其卖出日就是D），不能用当日未完成周期训练
# - 非交易日（当日有效行情股数<1000）直接跳过，不发邮件
# - 重建失败时保留旧模型，邮件中注明模型版本未更新，绝不以失败模型打分
import pickle, glob, os, sys, json, gc, time, datetime, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASE = os.path.expanduser("~/workspace/ashare")
FEATS = ["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
CST = datetime.timezone(datetime.timedelta(hours=8))

def board(code):
    c = code[2:]
    if c.startswith("688"): return "STAR"
    if c.startswith("300") or c.startswith("301"): return "ChiNext"
    return "Main"

def limit_thr(code):
    c = code[2:]
    if c.startswith("688") or c.startswith("300") or c.startswith("301"): return 0.195
    return 0.095

def epoch(s):
    return int(datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.timezone.utc).timestamp())

def fetch_pe(codes):
    """当日市盈率（新浪行情中心快照 per 字段，负值表示亏损）。只取Top10的10只股；
    沪深快照每页100只、10线程并行拉取；任何失败返回缺失，绝不因此阻断Top10邮件。"""
    want = set(codes); out = {}
    def one_page(page):
        url = ("https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/"
               f"Market_Center.getHQNodeData?page={page}&num=100&sort=symbol&asc=1&node=hs_a")
        for _ in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"})
                rows = json.load(urllib.request.urlopen(req, timeout=20))
                d = {}
                for r in rows:
                    sym = r.get("symbol", "")
                    if sym in want and r.get("per") is not None:
                        try: d[sym] = float(r["per"])
                        except (TypeError, ValueError): pass
                return d
            except Exception:
                time.sleep(1)
        return {}
    try:
        with ThreadPoolExecutor(max_workers=10) as ex:
            for d in ex.map(one_page, range(1, 61)):
                out.update(d)
                if len(out) >= len(want): break
    except Exception:
        pass
    return {c: out.get(c) for c in codes}

def build_review(sig_day, buy_day, sell_day, top10, bars_buy, bars_sell):
    """复核信号日sig_day的推荐：买入日buy_day按信号收盘挂牌，卖出日sell_day按-2%/+5%（双触按止损）。
    返回review dict；缺行情的个股标数据缺失，不影响其他个股。"""
    items = []
    for t in top10:
        code = t["code"]; entry = float(t["signal_close"])
        bb, lb = bars_buy.get(code), bars_sell.get(code)
        if not bb or not lb:
            items.append({"code": code, "name": t.get("name", ""), "signal_close": entry, "status": "数据缺失"})
            continue
        bought = float(bb["low"]) <= entry
        if not bought:
            items.append({"code": code, "name": t.get("name", ""), "signal_close": entry, "status": "未成交"})
        elif float(lb["low"]) <= entry * 0.98:
            items.append({"code": code, "name": t.get("name", ""), "signal_close": entry, "status": "止损", "realized": -0.02})
        elif float(lb["high"]) >= entry * 1.05:
            items.append({"code": code, "name": t.get("name", ""), "signal_close": entry, "status": "止盈", "realized": 0.05})
        else:
            rc = float(lb["adjclose"]) / entry - 1
            items.append({"code": code, "name": t.get("name", ""), "signal_close": entry, "status": "收盘卖出", "realized": round(rc, 4)})
    winners = [i["code"] for i in items if i["status"] == "止盈"]
    return {"signal_day": sig_day, "buy_day": buy_day, "sell_day": sell_day, "items": items,
            "winners": winners, "win_count": len(winners),
            "bought_count": sum(1 for i in items if i["status"] != "未成交" and i["status"] != "数据缺失")}

def fetch_day(codes, day):
    """抓day当日的Yahoo日线，返回 {code: dict或None}；day格式YYYY-MM-DD"""
    d0 = datetime.date.fromisoformat(day)
    p1, p2 = epoch(str(d0 - datetime.timedelta(days=10))), epoch(str(d0 + datetime.timedelta(days=3)))
    def fetch(code):
        sym = code[2:] + (".SS" if code.startswith("sh") else ".SZ")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d"
        for attempt in range(2):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                dd = json.load(urllib.request.urlopen(req, timeout=25))
                r = dd["chart"]["result"][0]; ts = r.get("timestamp", []); q = r["indicators"]["quote"][0]
                adj = r["indicators"].get("adjclose", [{}])[0].get("adjclose", [])
                for t, o, h, l, c, v, a in zip(ts, q["open"], q["high"], q["low"], q["close"], q["volume"], adj):
                    dt = datetime.datetime.fromtimestamp(t, tz=CST).strftime("%Y-%m-%d")
                    if dt == day and c is not None:
                        return (code, {"open": o, "high": h, "low": l, "close": c, "adjclose": a, "volume": v})
                return (code, None)
            except Exception:
                if attempt == 1: return (code, None)
                time.sleep(2)
    out = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        for code, res in ex.map(fetch, codes):
            out[code] = res
    return out

def sse_close(day):
    """上证指数当日收盘：chart close为空时用meta regularMarketPrice（收盘后即为当日收盘）"""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/000001.SS?period1={epoch('2026-01-01')}&period2={epoch(str(datetime.date.fromisoformat(day) + datetime.timedelta(days=3)))}&interval=1d"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    dd = json.load(urllib.request.urlopen(req, timeout=25))
    r = dd["chart"]["result"][0]; ts = r.get("timestamp", []); q = r["indicators"]["quote"][0]
    for t, c in zip(ts, q["close"]):
        dt = datetime.datetime.fromtimestamp(t, tz=CST).strftime("%Y-%m-%d")
        if dt == day and c is not None: return float(c)
    meta = r.get("meta", {})
    if meta.get("regularMarketPrice") is not None:
        return float(meta["regularMarketPrice"])
    return None

def features_for_day(raw_df, day_bars, day, mkt_ret, mkt_ret5):
    """raw_df: 历史行情（code/date/open/high/low/close/adjclose/volume）；day_bars: 当日行情dict
    每股取最近300行+当日行，按full_train.py同公式算当日特征，只返回当日行"""
    rows = []
    day_ts = pd.Timestamp(day)
    for code, bar in day_bars.items():
        if bar is None: continue
        g = raw_df[raw_df["code"] == code].sort_values("date")
        if len(g) < 60: continue
        new = pd.DataFrame([{"code": code, "date": day_ts, "open": bar["open"], "high": bar["high"], "low": bar["low"], "close": bar["close"], "adjclose": bar["adjclose"], "volume": bar["volume"]}])
        g = pd.concat([g.tail(300), new], ignore_index=True).sort_values("date")
        if g["date"].iloc[-1] != day_ts: continue
        adj = g["adjclose"]; factor = adj / g["close"].replace(0, np.nan)
        adj_high = g["high"] * factor; adj_low = g["low"] * factor
        ret = adj.pct_change()
        feat = {}
        feat["vol_5"] = ret.rolling(5).std().iloc[-1]; feat["vol_20"] = ret.rolling(20).std().iloc[-1]
        feat["atr_14"] = (np.maximum(adj_high - adj_low, np.maximum((adj_high - adj.shift(1)).abs(), (adj_low - adj.shift(1)).abs())).rolling(14).mean() / adj).iloc[-1]
        big = (ret > 0.05).astype(float)
        feat["big_up_freq_60"] = big.shift(1).rolling(60).mean().iloc[-1]; feat["big_up_freq_250"] = big.shift(1).rolling(250).mean().iloc[-1]
        feat["ret_1"] = ret.iloc[-1]; feat["ret_3"] = adj.pct_change(3).iloc[-1]; feat["ret_5"] = adj.pct_change(5).iloc[-1]
        feat["ret_10"] = adj.pct_change(10).iloc[-1]; feat["ret_20"] = adj.pct_change(20).iloc[-1]
        feat["ma_bias_5"] = (adj / adj.rolling(5).mean() - 1).iloc[-1]; feat["ma_bias_20"] = (adj / adj.rolling(20).mean() - 1).iloc[-1]; feat["ma_bias_60"] = (adj / adj.rolling(60).mean() - 1).iloc[-1]
        feat["close_vs_high60"] = (adj / adj.rolling(60).max() - 1).iloc[-1]; feat["close_vs_low60"] = (adj / adj.rolling(60).min() - 1).iloc[-1]
        feat["breakout_20"] = float(adj.iloc[-1] > adj.shift(1).rolling(20).max().iloc[-1])
        up = (ret > 0); feat["consec_up"] = up.groupby((~up).cumsum()).cumsum().iloc[-1]
        feat["vol_ratio_5"] = (g["volume"] / g["volume"].shift(1).rolling(5).mean()).iloc[-1]; feat["vol_ratio_20"] = (g["volume"] / g["volume"].shift(1).rolling(20).mean()).iloc[-1]
        lim = (ret >= limit_thr(code)).astype(float)
        feat["is_limit_up"] = lim.iloc[-1]; feat["limit_cnt_5"] = lim.shift(1).rolling(5).sum().iloc[-1]; feat["limit_cnt_20"] = lim.shift(1).rolling(20).sum().iloc[-1]
        delta = adj.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        ag = gain.ewm(alpha=1/14, adjust=False).mean(); al = loss.ewm(alpha=1/14, adjust=False).mean()
        feat["rsi_14"] = (100 - 100 / (1 + ag / al.replace(0, np.nan))).iloc[-1]
        ema12 = adj.ewm(span=12, adjust=False).mean(); ema26 = adj.ewm(span=26, adjust=False).mean(); dif = ema12 - ema26; dea = dif.ewm(span=9, adjust=False).mean()
        feat["macd_diff"] = (dif / adj).iloc[-1]; feat["macd_hist"] = ((dif - dea) / adj).iloc[-1]
        feat["mkt_ret"] = mkt_ret; feat["mkt_ret5"] = mkt_ret5
        feat["excess_ret_1"] = feat["ret_1"] - mkt_ret; feat["excess_ret_5"] = feat["ret_5"] - mkt_ret5
        feat["code"] = code; feat["date"] = day; feat["close"] = float(bar["adjclose"]); feat["board"] = board(code)
        rows.append(feat)
    df = pd.DataFrame(rows)
    if len(df): df = df.dropna(subset=FEATS)
    return df

def main():
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now(CST).strftime("%Y-%m-%d")
    t0 = time.time(); os.makedirs(os.path.join(BASE, "daily_bars"), exist_ok=True); os.makedirs(os.path.join(BASE, "daily_features"), exist_ok=True); os.makedirs(os.path.join(BASE, "daily_output"), exist_ok=True)
    codes = set()
    for f in sorted(glob.glob(os.path.join(BASE, "full_flow_new_*.pkl"))):
        d = pickle.load(open(f, "rb")); codes |= set(d["code"].unique()); del d
    codes = sorted(codes)
    bars = fetch_day(codes, day)
    ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
    print(f"day {day} bars ok {len(ok)}/{len(codes)}", flush=True)
    with open(os.path.join(BASE, "daily_bars", f"{day}.pkl"), "wb") as f: pickle.dump(bars, f)
    if len(ok) < 1000:
        print(f"NOT_TRADING_DAY {day} ok={len(ok)}", flush=True)
        with open(os.path.join(BASE, "daily_output", f"{day}.json"), "w") as f: json.dump({"date": day, "trading_day": False, "bars_ok": len(ok)}, f, ensure_ascii=False, indent=2)
        return
    # 上证当日与5日前收盘（5日前取历史指数文件+已存daily sse）
    midx = pickle.load(open(os.path.join(BASE, "index_sse.pkl"), "rb"))
    sse_hist = {pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d, v in zip(midx["date"], midx["mkt_adjclose"])}
    sse_today = sse_close(day)
    sse_hist[day] = sse_today
    sse_path = os.path.join(BASE, "sse_daily.json"); sse_saved = json.load(open(sse_path)) if os.path.exists(sse_path) else {}
    # 2026-10-28补丁：sse_hist此前只含历史指数文件+当日，前一交易日若只在sse_saved（如2026-10-27）则KeyError；先把已存每日收盘并入，当日为准
    sse_hist.update(sse_saved); sse_hist[day] = sse_today
    sse_saved.update(sse_hist); json.dump(sse_saved, open(sse_path, "w"))
    trading_days = sorted([d for d in sse_saved.keys() if d <= day])
    # 注意：sse_saved含全历史指数日，交易日序列以指数日为准；mkt_ret5用5个交易日前收盘
    idx = trading_days.index(day) if day in trading_days else -1
    mkt_ret = sse_today / sse_hist[trading_days[idx-1]] - 1 if idx >= 1 else 0.0
    mkt_ret5 = sse_today / sse_hist[trading_days[idx-5]] - 1 if idx >= 5 else 0.0
    # 历史行情：原始两文件 + 此前daily_bars（不含当日，当日单独拼）
    parts = []
    for f in [os.path.join(BASE, "full_raw_partial.pkl"), os.path.join(BASE, "full_new_partial.pkl")]:
        if os.path.exists(f): parts.append(pickle.load(open(f, "rb"))[["code", "date", "open", "high", "low", "close", "adjclose", "volume"]])
    for bf in sorted(glob.glob(os.path.join(BASE, "daily_bars", "*.pkl"))):
        bday = os.path.basename(bf)[:-4]
        if bday >= day: continue
        bd = pickle.load(open(bf, "rb"))
        rows = [{"code": c, "date": pd.Timestamp(bday), **v} for c, v in bd.items() if isinstance(v, dict)]
        if rows: parts.append(pd.DataFrame(rows))
    raw = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    raw["date"] = pd.to_datetime(raw["date"])
    feat = features_for_day(raw, bars, day, mkt_ret, mkt_ret5)
    feat.to_pickle(os.path.join(BASE, "daily_features", f"{day}.pkl"))
    print(f"features {day} rows {len(feat)} elapsed {time.time()-t0:.0f}s", flush=True)
    # 当前模型指针
    ptr = os.path.join(BASE, "daily_model_pointer.txt")
    model_path = open(ptr).read().strip() if os.path.exists(ptr) else os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
    mdl = pickle.load(open(model_path, "rb"))
    feat["prob"] = mdl.predict_proba(feat[FEATS].to_numpy(np.float32))[:, 1]
    # 名称映射
    names = {}
    try:
        lst = pd.read_csv(os.path.join(BASE, "sina_hs_a_20261025.csv"))
        names = dict(zip(lst["symbol"], lst["name"]))
    except Exception: pass
    # 剔除退市/ST股（名称含“退”或“ST”）：原始宇宙只按2026-10-23快照剔ST，漏了退市股，2026-10-27周一打分曾把元道退排进Top20，须在此处拦截
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
    # 增量标签：信号日=2个交易日前（其卖出日=当日），需买入日与卖出日行情均已存
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
        # 信号日特征来自daily_features（若信号日在启动前则用原flow文件中的当日行近似，缺失则跳过该日）
        sf_path = os.path.join(BASE, "daily_features", f"{sig_day}.pkl")
        if os.path.exists(sf_path) and sb and bb and lb:
            sf = pickle.load(open(sf_path, "rb"))
            recs = []
            for _, r in sf.iterrows():
                code = r["code"]
                if code not in sb or code not in bb or code not in lb: continue
                entry = float(r["close"]); buy_low = float(bb[code]["low"]); sell_low = float(lb[code]["low"]); sell_high = float(lb[code]["high"]); sell_close = float(lb[code]["adjclose"])
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

if __name__ == "__main__":
    main()
