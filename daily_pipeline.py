# 工作日17点日线任务：取当日行情 -> 提当日信号特征 -> 增量标签 -> 重建模型 -> 下一交易日Top10
# 2026-11-01起增加两条对比线：m=2/m=3 新策略（EV排序Top10）；2026-11-02 16:00起三策略统一x=0.05、+7%止盈（网格最优），
# 与原策略并行输出、并行复核（gap-aware真实执行口径），三策略累计对比记入 m_compare.json，原策略每日增量重建不变。
# 口径（与主项目一致）：
# - 信号日D收盘打分，买入日=下一交易日按D收盘×0.95限价挂牌，卖出日=再下一交易日-2%止损/+7%止盈（止损止盈按实际成交价），同日双触按止损算
# - 模型概率 = 买入成交且卖出收益>=+5%的概率，不是正收益概率（正收益还含0~+5%区间，模型概率是其下界，不可等同）
# - 标签滞后2个交易日：当日只能为信号日=D-2个交易日的周期补标签（其卖出日就是D），不能用当日未完成周期训练
# - 非交易日（当日有效行情股数<1000）直接跳过，不发邮件
# - 重建失败时保留旧模型，邮件中注明模型版本未更新，绝不以失败模型打分
import pickle, glob, os, sys, json, gc, time, datetime, urllib.request
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import strategy_config as SC

BASE = os.path.expanduser("~/workspace/ashare")
FEATS = ["vol_5","vol_20","atr_14","big_up_freq_60","big_up_freq_250","ret_1","ret_3","ret_5","ret_10","ret_20","ma_bias_5","ma_bias_20","ma_bias_60","close_vs_high60","close_vs_low60","breakout_20","consec_up","excess_ret_1","excess_ret_5","vol_ratio_5","vol_ratio_20","is_limit_up","limit_cnt_5","limit_cnt_20","mkt_ret","mkt_ret5","rsi_14","macd_diff","macd_hist"]
CST = datetime.timezone(datetime.timedelta(hours=8))

# ---- 阶段标记与续跑（2026-11-05 加固；不动策略口径，只加保命）----
# 8 个阶段：抓行情(bars)→提特征(features)→打分(scoring)→市盈率(pe)→增量标签(inc_labels)→重建(retrain)→复核(review)→落盘(output)
# 每个阶段产物落盘后即写标记 daily_output/{day}.stages.json（tmp+rename 原子写）。
# 续跑：python daily_pipeline.py --resume [YYYY-MM-DD]，读标记、只跑缺失阶段；
# 增量标签追加带幂等校验（库中已有 sig_day 的行则跳过），m_compare 累计带 counted_keys 防重复加。
STAGES = ["bars", "features", "scoring", "pe", "inc_labels", "retrain", "review", "output"]
RESUME_MODE = False

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
    """当日市盈率（新浪行情中心快照 per 字段，负值表示亏损）。取传入的全部代码；
    沪深快照每页100只、10线程并行拉取；任何失败返回缺失，绝不因此阻断邮件。"""
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

def fetch_pe_robust(codes, max_rounds=4):
    """完整信息保障：多轮重试直到拿全或耗尽轮数。缺失的code逐轮重试，
    每轮间隔3秒；某轮拿到的不再重抓。返回 {code: pe或None}。"""
    want = [c for c in codes if c]
    out = {}
    for rnd in range(max_rounds):
        missing = [c for c in want if c not in out]
        if not missing:
            break
        if rnd > 0:
            time.sleep(3)
            print(f"PE retry round {rnd+1}/{max_rounds}, still missing {len(missing)}: {missing[:5]}", flush=True)
        try:
            got = fetch_pe(missing)
        except Exception:
            got = {}
        out.update({k: v for k, v in got.items() if v is not None})
    if len(out) < len(want):
        print(f"PE incomplete after {max_rounds} rounds: {len(out)}/{len(want)}", flush=True)
    return {c: out.get(c) for c in codes}

def build_review(sig_day, buy_day, sell_day, top10, bars_buy, bars_sell, pe_map=None):
    """复核信号日sig_day的推荐：买入日buy_day按信号收盘×0.90限价买入，卖出日sell_day按-2%/+2%（双触按止损）。
    2026-11-02修正为gap-aware真实执行：买入跳空低开按开盘价成交（更优）；卖出日开盘跳空穿过
    止盈/止损按开盘价成交；盘中双触按止损。买卖价均为复权价。返回格式与旧版兼容（新增"跳空止盈"状态）。
    2026-11-02 15:08起：三策略统一x=0.10、+2%止盈（原为x=0、+5%止盈）。"""
    r = build_gap_review(sig_day, buy_day, [sell_day], top10,
                         {buy_day: bars_buy, sell_day: bars_sell},
                         SC.ORIG_X_DISC, SC.ORIG_STOP, SC.ORIG_TAKE, pe_map, tag="orig")
    winners = [i["code"] for i in r["items"] if i["status"] in ("止盈", "跳空止盈")]
    return {"signal_day": r["signal_day"], "buy_day": r["buy_day"], "sell_day": sell_day,
            "items": r["items"], "winners": winners, "win_count": len(winners),
            "bought_count": r["bought_count"], "top_bought_count": r["top_bought_count"],
            "mean_top_n": r["mean_top_n"], "mean_realized": r["mean_realized"]}

def _adj(bar):
    """把Yahoo未复权OHLC按adjclose/close换算为复权价；bar为None或缺字段返回None"""
    if not bar: return None
    try:
        c, a = float(bar["close"]), float(bar["adjclose"])
        if not (c > 0 and a > 0): return None
        f = a / c
        return {"open": float(bar["open"]) * f, "high": float(bar["high"]) * f,
                "low": float(bar["low"]) * f, "close": a}
    except (KeyError, TypeError, ValueError):
        return None

def build_gap_review(sig_day, buy_day, hold_days, picks, bars_by_day, x_disc, stop, take, pe_map=None, tag="", mean_top_n=None):
    """通用 gap-aware 复核（真实执行口径，供策略对比用）。
    picks: [{code, signal_close, ...}]（按排名顺序）；bars_by_day: {day: bars_dict} 需含买入日与全部持有日；
    x_disc: 买入折扣（0=按信号收盘买入；0.10=m策略）；stop: 止损比例（None=无止损）；
    take: 止盈比例。买入跳空低开按开盘价成交（更优）；持有日跳空穿过止盈/止损按开盘价成交；
    止损与止盈同日触发按止损算（与原口径一致）。
    mean_top_n: "平均每笔"按前N只中成交的统计（默认 strategy_config.N_STRATEGY）。"""
    if mean_top_n is None:
        mean_top_n = SC.N_STRATEGY
    pe_map = pe_map or {}
    items = []
    for t in picks:
        code = t["code"]; sig = float(t["signal_close"])
        _pe = pe_map.get(code); _pe = float(_pe) if _pe is not None else None
        base = {"code": code, "name": t.get("name", ""), "signal_close": sig, "pe": _pe,
                "buy_price": None, "sell_price": None}
        limit = sig * (1 - x_disc)
        bb = _adj(bars_by_day.get(buy_day, {}).get(code))
        holds = [_adj(bars_by_day.get(d, {}).get(code)) for d in hold_days]
        if bb is None or any(h is None for h in holds):
            items.append({**base, "status": "数据缺失"}); continue
        if bb["low"] > limit:
            items.append({**base, "status": "未成交"}); continue
        entry = min(limit, bb["open"])
        done = None
        for h in holds:
            if stop is not None and h["open"] <= entry * (1 - stop):
                done = ("止损", h["open"] / entry - 1, h["open"]); break
            if h["open"] >= entry * (1 + take):
                done = ("跳空止盈", h["open"] / entry - 1, h["open"]); break
            hit_stop = stop is not None and h["low"] <= entry * (1 - stop)
            hit_take = h["high"] >= entry * (1 + take)
            if hit_stop:  # 同日双触按止损
                done = ("止损", -stop, entry * (1 - stop)); break
            if hit_take:
                done = ("止盈", take, entry * (1 + take)); break
        if done is None:
            rc = holds[-1]["close"] / entry - 1
            done = ("收盘卖出", rc, holds[-1]["close"])
        st, r, sp = done
        r_net = float(r) - SC.FEE  # 扣除0.5%手续费（往返）
        items.append({**base, "status": st, "realized": round(r_net, 4),
                      "buy_price": round(entry, 2), "sell_price": round(float(sp), 2)})
    bought = [i for i in items if i["status"] not in ("未成交", "数据缺失")]
    top_bought = [i for i in items[:mean_top_n] if i["status"] not in ("未成交", "数据缺失")]
    mean_r = round(sum(i["realized"] for i in top_bought) / len(top_bought), 4) if top_bought else None
    return {"tag": tag, "signal_day": sig_day, "buy_day": buy_day, "hold_days": hold_days,
            "x_disc": x_disc, "stop": stop, "take": take, "items": items,
            "bought_count": len(bought), "top_bought_count": len(top_bought),
            "mean_top_n": mean_top_n, "mean_realized": mean_r}

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

def _stages_path(day):
    return os.path.join(BASE, "daily_output", f"{day}.stages.json")

def load_stages(day):
    p = _stages_path(day)
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}
    return {}

def mark_stage(day, stage, **meta):
    """阶段完成后写标记（tmp+rename 原子写，防被杀写坏）"""
    p = _stages_path(day)
    cur = load_stages(day)
    cur["day"] = day
    cur[stage] = True
    cur.update(meta)
    cur["updated_at"] = datetime.datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S")
    tmp = p + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f, ensure_ascii=False, indent=2)
    os.rename(tmp, p)
    print(f"stage done: {stage}", flush=True)

def _load_pkl_ok(path):
    try:
        pickle.load(open(path, "rb"))
        return True
    except Exception:
        return False

def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.rename(tmp, path)

def _atomic_pickle_dump(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.rename(tmp, path)

def _atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.rename(tmp, path)

def valid_output(day):
    """output 阶段完成判定：JSON 存在、可加载、top10 恰 10 条、trading_day 为 True，且 email txt 存在"""
    jp = os.path.join(BASE, "daily_output", f"{day}.json")
    tp = os.path.join(BASE, "daily_output", f"{day}_email.txt")
    if not (os.path.exists(jp) and os.path.exists(tp)):
        return False
    try:
        d = json.load(open(jp))
    except Exception:
        return False
    return bool(d.get("trading_day")) and isinstance(d.get("top10"), list) and len(d["top10"]) == 10

def main(day=None):
    if day is None:
        day = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.now(CST).strftime("%Y-%m-%d")
    t0 = time.time(); os.makedirs(os.path.join(BASE, "daily_bars"), exist_ok=True); os.makedirs(os.path.join(BASE, "daily_features"), exist_ok=True); os.makedirs(os.path.join(BASE, "daily_output"), exist_ok=True)
    stages = load_stages(day) if RESUME_MODE else {}
    if RESUME_MODE:
        print(f"resume mode for {day}, markers: {stages}", flush=True)
        if valid_output(day):
            print(f"resume: {day} output already complete, nothing to do", flush=True)
            return
    codes = set()
    for f in sorted(glob.glob(os.path.join(BASE, "full_flow_new_*.pkl"))):
        d = pickle.load(open(f, "rb")); codes |= set(d["code"].unique()); del d
    codes = sorted(codes)
    bars_path = os.path.join(BASE, "daily_bars", f"{day}.pkl")
    if RESUME_MODE and stages.get("bars") and _load_pkl_ok(bars_path):
        bars = pickle.load(open(bars_path, "rb"))
        print(f"resume: skip fetch_day, loaded {bars_path}", flush=True)
    else:
        bars = fetch_day(codes, day)
        _atomic_pickle_dump(bars_path, bars)
        mark_stage(day, "bars")
    ok = {k: v for k, v in bars.items() if isinstance(v, dict)}
    print(f"day {day} bars ok {len(ok)}/{len(codes)}", flush=True)
    if len(ok) < 1000:
        print(f"NOT_TRADING_DAY {day} ok={len(ok)}", flush=True)
        _atomic_write_json(os.path.join(BASE, "daily_output", f"{day}.json"),
                           {"date": day, "trading_day": False, "bars_ok": len(ok)})
        mark_stage(day, "output", trading_day=False)
        return
    # 上证当日与5日前收盘（5日前取历史指数文件+已存daily sse）
    midx = pickle.load(open(os.path.join(BASE, "index_sse.pkl"), "rb"))
    sse_hist = {pd.Timestamp(d).strftime("%Y-%m-%d"): float(v) for d, v in zip(midx["date"], midx["mkt_adjclose"])}
    sse_today = sse_close(day)
    sse_hist[day] = sse_today
    sse_path = os.path.join(BASE, "sse_daily.json"); sse_saved = json.load(open(sse_path)) if os.path.exists(sse_path) else {}
    # 2026-10-28补丁：sse_hist此前只含历史指数文件+当日，前一交易日若只在sse_saved（如2026-10-27）则KeyError；先把已存每日收盘并入，当日为准
    sse_hist.update(sse_saved); sse_hist[day] = sse_today
    sse_saved.update(sse_hist); _atomic_write_json(sse_path, sse_saved)
    trading_days = sorted([d for d in sse_saved.keys() if d <= day])
    # 注意：sse_saved含全历史指数日，交易日序列以指数日为准；mkt_ret5用5个交易日前收盘
    idx = trading_days.index(day) if day in trading_days else -1
    mkt_ret = sse_today / sse_hist[trading_days[idx-1]] - 1 if idx >= 1 else 0.0
    mkt_ret5 = sse_today / sse_hist[trading_days[idx-5]] - 1 if idx >= 5 else 0.0
    feat_path = os.path.join(BASE, "daily_features", f"{day}.pkl")
    if RESUME_MODE and stages.get("features") and _load_pkl_ok(feat_path):
        feat = pickle.load(open(feat_path, "rb"))
        print(f"resume: skip features_for_day, loaded {feat_path} rows {len(feat)}", flush=True)
    else:
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
        del raw; gc.collect()
        _atomic_pickle_dump(feat_path, feat)
        mark_stage(day, "features")
        print(f"features {day} rows {len(feat)} elapsed {time.time()-t0:.0f}s", flush=True)
    # 当前模型指针
    ptr = os.path.join(BASE, "daily_model_pointer.txt")
    model_path = open(ptr).read().strip() if os.path.exists(ptr) else os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl")
    if RESUME_MODE and stages.get("model_used") and _load_pkl_ok(stages["model_used"]):
        # 续跑必须用原 run 打分时的模型（指针在原 run 重建后已前移，直接读指针会错）
        model_path = stages["model_used"]
        print(f"resume: scoring with original model {model_path}", flush=True)
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
    # m=2/m=3 新策略：EV排序器取Top10（x=0.05限价买入、+7%止盈，三策略统一），作为与原策略的并行对比线；
    # 模型为回测期固定模型（ev_model_5t7_2/5t7_3.pkl），不参与每日增量重建
    m2_top10, m3_top10 = [], []
    try:
        ev2 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_2.pkl"), "rb"))
        feat["_ev2"] = ev2.predict(feat[FEATS].to_numpy(np.float32)); del ev2
        _s2 = feat.sort_values(["_ev2", "code"], ascending=[False, True])
        for _, _r in _s2.iterrows():
            _nm = str(names.get(_r["code"], ""))
            if ("退" in _nm) or ("ST" in _nm): continue
            c = float(_r["close"])
            m2_top10.append({"code": _r["code"], "name": names.get(_r["code"], ""),
                            "signal_close": c, "limit_price": round(c * 0.95, 2),
                            "ev_score": float(_r["_ev2"]), "pe": None})
            if len(m2_top10) >= 10: break
    except Exception as e:
        print(f"m2 scoring failed: {type(e).__name__}", flush=True)
    try:
        ev3 = pickle.load(open(os.path.join(BASE, "ev_model_5t7_3.pkl"), "rb"))
        feat["_ev3"] = ev3.predict(feat[FEATS].to_numpy(np.float32)); del ev3
        _s3 = feat.sort_values(["_ev3", "code"], ascending=[False, True])
        for _, _r in _s3.iterrows():
            _nm = str(names.get(_r["code"], ""))
            if ("退" in _nm) or ("ST" in _nm): continue
            c = float(_r["close"])
            m3_top10.append({"code": _r["code"], "name": names.get(_r["code"], ""),
                            "signal_close": c, "limit_price": round(c * 0.95, 2),
                            "ev_score": float(_r["_ev3"]), "pe": None})
            if len(m3_top10) >= 10: break
    except Exception as e:
        print(f"m3 scoring failed: {type(e).__name__}", flush=True)
    mark_stage(day, "scoring", model_used=model_path)
    top_list = []
    try:
        pe_map = fetch_pe_robust([str(r["code"]) for _, r in top.iterrows()] +
                          [t["code"] for t in m2_top10] + [t["code"] for t in m3_top10])
    except Exception:
        pe_map = {}
    for t in m2_top10 + m3_top10:
        _pe = pe_map.get(t["code"])
        t["pe"] = float(_pe) if _pe is not None else None
    for _, r in top.iterrows():
        c = float(r["close"])
        _pe = pe_map.get(str(r["code"]))
        _lim = round(c * (1 - SC.ORIG_X_DISC), 2)  # 现行口径：信号收盘×0.95限价买入（strategy_config.ORIG_X_DISC）
        top_list.append({"code": r["code"], "name": names.get(r["code"], ""), "signal_close": c, "limit_price": _lim, "stop_price": round(_lim * (1 - SC.ORIG_STOP), 2), "target_price": round(_lim * (1 + SC.ORIG_TAKE), 2), "prob_ge5": float(r["prob"]), "pe": (float(_pe) if _pe is not None else None)})
    mark_stage(day, "pe")
    # 增量标签：信号日=2个交易日前（其卖出日=当日），需买入日与卖出日行情均已存
    inc_path = os.path.join(BASE, "incremental_flow.pkl")
    inc = pickle.load(open(inc_path, "rb")) if os.path.exists(inc_path) else pd.DataFrame()
    inc_new = 0
    sig_day = None
    _inc_new_out = 0  # 输出展示用：续跑命中幂等时，还原标记中记录的当日实际追加数
    if idx >= 2:
        sig_day = trading_days[idx-2]; buy_day = trading_days[idx-1]; sell_day = day
        # 幂等校验：库中已有 sig_day 的行则跳过追加（防重复跑/续跑重复加）
        _already = len(inc) > 0 and "date" in inc.columns and bool((inc["date"] == sig_day).any())
        if _already:
            print(f"inc_labels for {sig_day} already in store ({int((inc['date'] == sig_day).sum())} rows), skip append [idempotent]", flush=True)
            _inc_new_out = int(stages.get("inc_new", 0) or 0)
        else:
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
                    inc_new = len(newdf); _inc_new_out = inc_new; _atomic_pickle_dump(inc_path, inc)
    mark_stage(day, "inc_labels", inc_new=int(_inc_new_out), sig_day=sig_day)
    # 重建模型：基数组（train+val）+ 增量行；失败则保留旧模型
    retrain_status = "skipped_no_new_labels" if inc_new == 0 else "failed_keep_old"
    do_retrain = inc_new > 0
    _model_file = os.path.join(BASE, f"daily_model_{day}.pkl")
    if RESUME_MODE:
        _rss = stages.get("retrain_status", "")
        if stages.get("retrain") and _rss == "ok" and _load_pkl_ok(_model_file):
            do_retrain = False; retrain_status = "ok"
            if os.path.exists(ptr): model_path = open(ptr).read().strip()
            print("resume: skip retrain (already done)", flush=True)
        elif stages.get("retrain") and _rss == "skipped_no_new_labels":
            do_retrain = False; retrain_status = _rss
            print("resume: skip retrain (no new labels, as before)", flush=True)
        elif (not stages.get("retrain")) and sig_day and len(inc) > 0 and "date" in inc.columns \
                and bool((inc["date"] == sig_day).any()) and not _load_pkl_ok(_model_file):
            # 标签已入库但模型未生成（上一轮死在追加后、重建前）→ 补做重建
            do_retrain = True
            print("resume: labels exist but model missing, retrain now", flush=True)
    base_dir = os.path.join(BASE, "daily_base_arrays")
    if do_retrain and os.path.exists(os.path.join(base_dir, "train_X.npy")):
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
    mark_stage(day, "retrain", retrain_status=retrain_status)
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
                try:
                    rev_pe = fetch_pe_robust([t["code"] for t in old_top])
                except Exception:
                    rev_pe = {}
                review = build_review(sig_day, buy_day, day, old_top, bars_buy, bars, rev_pe)
            except Exception as e:
                review = {"status": f"failed: {type(e).__name__}"}
        else:
            review = {"status": "no_archive", "signal_day": sig_day}
    # 三策略 gap-aware 复核（真实执行口径，供公平对比；原策略的日报复核口径保持不变）
    def _bars_of(d):
        if d == day: return bars
        p = os.path.join(BASE, "daily_bars", f"{d}.pkl")
        return pickle.load(open(p, "rb")) if os.path.exists(p) else {}
    m2_review, m3_review = {"status": "no_archive"}, {"status": "no_archive"}
    # orig_gap 已并入 build_review（2026-11-02起原复核即gap-aware），累计对比直接用 review
    if idx >= 3:
        _s2, _b2 = trading_days[idx-3], trading_days[idx-2]; _h2 = [trading_days[idx-1], day]
        _a2 = os.path.join(BASE, "daily_output", f"{_s2}.json")
        if os.path.exists(_a2):
            try:
                _old2 = json.load(open(_a2)).get("m2_top10", [])
                try: _pe2 = fetch_pe_robust([t["code"] for t in _old2])
                except Exception: _pe2 = {}
                m2_review = build_gap_review(_s2, _b2, _h2, _old2, {d: _bars_of(d) for d in [_b2] + _h2}, SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe2, tag="m2")
            except Exception as e:
                m2_review = {"status": f"failed: {type(e).__name__}"}
        else:
            m2_review = {"status": "no_archive", "signal_day": _s2}
    if idx >= 4:
        _s3, _b3 = trading_days[idx-4], trading_days[idx-3]; _h3 = [trading_days[idx-2], trading_days[idx-1], day]
        _a3 = os.path.join(BASE, "daily_output", f"{_s3}.json")
        if os.path.exists(_a3):
            try:
                _old3 = json.load(open(_a3)).get("m3_top10", [])
                try: _pe3 = fetch_pe_robust([t["code"] for t in _old3])
                except Exception: _pe3 = {}
                m3_review = build_gap_review(_s3, _b3, _h3, _old3, {d: _bars_of(d) for d in [_b3] + _h3}, SC.M_X_DISC, SC.M_STOP, SC.M_TAKE, _pe3, tag="m3")
            except Exception as e:
                m3_review = {"status": f"failed: {type(e).__name__}"}
        else:
            m3_review = {"status": "no_archive", "signal_day": _s3}
    mark_stage(day, "review")
    # 三策略累计对比（gap-aware口径、已扣0.5%手续费；m_compare.json 持续累积）
    cmp_path = os.path.join(BASE, "m_compare.json")
    cmpd = json.load(open(cmp_path)) if os.path.exists(cmp_path) else \
        {"orig": {"n": 0, "sum_r": 0.0}, "m2": {"n": 0, "sum_r": 0.0}, "m3": {"n": 0, "sum_r": 0.0}}
    cmpd.setdefault("counted_keys", [])  # 幂等：已计入的 (signal_day:strategy) 不重复加
    for _key, _rv in (("orig", review), ("m2", m2_review), ("m3", m3_review)):
        _ck = f"{_rv.get('signal_day', '?')}:{_key}"
        if _ck in cmpd["counted_keys"]:
            print(f"m_compare: {_ck} already counted, skip", flush=True)
            continue
        if _rv.get("items"):
            cmpd["counted_keys"].append(_ck)
            for _it in _rv["items"][:SC.N_STRATEGY]:  # 平均每笔按前N_STRATEGY只统计
                if _it.get("status") not in ("未成交", "数据缺失") and _it.get("realized") is not None:
                    cmpd[_key]["n"] += 1; cmpd[_key]["sum_r"] += float(_it["realized"])
    _atomic_write_json(cmp_path, cmpd)
    def _cmp_line(key, label):
        n, s = cmpd[key]["n"], cmpd[key]["sum_r"]
        return f"- {label}：累计{n}笔（前{SC.N_STRATEGY}只），平均每笔{(s/n):+.2%}" if n else f"- {label}：暂无成交记录"
    out = {"date": day, "trading_day": True, "bars_ok": len(ok), "features_rows": int(len(feat)), "model_path": model_path, "retrain_status": retrain_status, "incremental_new_rows": int(_inc_new_out), "incremental_total_rows": int(len(inc)), "top10": top_list, "review": review,
           "m2_top10": m2_top10, "m3_top10": m3_top10, "m2_review": m2_review, "m3_review": m3_review,
           "strategy_compare": {k: {"n": v["n"], "mean": (v["sum_r"]/v["n"] if v["n"] else None)} for k, v in cmpd.items() if k in ("orig", "m2", "m3")}}
    _atomic_write_json(os.path.join(BASE, "daily_output", f"{day}.json"), out)
    lines = [f"A股下一交易日Top10（信号日{day}，买入日为下一交易日、按信号收盘×0.95限价买入，卖出日为再下一交易日-2%止损/+7%止盈，止损止盈按实际成交价算、跳空低开更优时相应下移，同日双触按止损算）", f"模型：{os.path.basename(model_path)}；重建状态：{retrain_status}；当日有效行情{len(ok)}只，特征{len(feat)}只。", "概率口径：模型预测的买入成交且卖出收益≥+5%的概率，不是正收益概率（正收益还含0~+5%区间，模型概率是其下界）；以下收益已扣除0.5%交易手续费（往返），未计滑点和税，不构成投资建议。", "市盈率为新浪当日快照值（负值表示亏损），仅供参考，非模型特征；取不到时显示“市盈率暂无”。", ""]
    # 开头：前3只推荐与收益对比（便于对比分析）
    lines.append("【前3只推荐与收益对比】（各策略实际按前3只交易；累计为前瞻纸面跟踪，gap-aware、已扣费）")
    def _top3_line(label, picks, key, bench):
        _nms = "、".join(t["name"] for t in picks[:SC.N_STRATEGY])
        _n, _s = cmpd[key]["n"], cmpd[key]["sum_r"]
        _fwd = f"累计{_n}笔、平均{(_s/_n):+.2%}" if _n else "暂无前瞻记录"
        _b = f"；回测基准{bench}" if bench else ""
        return f"- {label}前3只：{_nms}｜{_fwd}{_b}"
    lines.append(_top3_line("原策略", top_list, "orig", None))
    lines.append(_top3_line("m=2", m2_top10, "m2", SC.BENCH_M2))
    lines.append(_top3_line("m=3", m3_top10, "m3", SC.BENCH_M3))
    lines.append("")
    for i, t in enumerate(top_list, 1):
        _pe_txt = f"市盈率{t['pe']:.1f}倍" if t.get("pe") is not None else "市盈率暂无"
        lines.append(f"{i}. {t['name']} {t['code']}，信号收盘{t['signal_close']:.2f}元，限价买入{t['limit_price']:.2f}元，止损{t['stop_price']:.2f}元，止盈{t['target_price']:.2f}元（止损/止盈按实际成交价算，表中为按限价成交的参考值），模型概率{t['prob_ge5']:.3f}，{_pe_txt}")
    lines.append("")
    lines.append(f"【对比线】m=2 新策略 Top10（EV排序，信号日{day}；买入日为下一交易日、按信号收盘×0.95限价买入；持有至多2天、+7%止盈、无止损，到期收盘卖出；回测期固定模型，不参与每日重建）")
    if m2_top10:
        for i, t in enumerate(m2_top10, 1):
            _pe_txt = f"市盈率{t['pe']:.1f}倍" if t.get("pe") is not None else "市盈率暂无"
            lines.append(f"{i}. {t['name']} {t['code']}，信号收盘{t['signal_close']:.2f}元，限价买入{t['limit_price']:.2f}元，EV期望{t['ev_score']:+.2%}，{_pe_txt}")
    else:
        lines.append("（本日 m=2 打分失败或无有效名单）")
    lines.append("")
    lines.append(f"【对比线】m=3 新策略 Top10（EV排序，信号日{day}；买入规则同 m=2；持有至多3天、+7%止盈、无止损，到期收盘卖出；回测期固定模型）")
    if m3_top10:
        for i, t in enumerate(m3_top10, 1):
            _pe_txt = f"市盈率{t['pe']:.1f}倍" if t.get("pe") is not None else "市盈率暂无"
            lines.append(f"{i}. {t['name']} {t['code']}，信号收盘{t['signal_close']:.2f}元，限价买入{t['limit_price']:.2f}元，EV期望{t['ev_score']:+.2%}，{_pe_txt}")
    else:
        lines.append("（本日 m=3 打分失败或无有效名单）")
    lines.append("")
    if review.get("items"):
        _rm = review.get("mean_realized")
        _rmt = f"{_rm:+.2%}" if _rm is not None else "—"
        lines.append(f"前两个交易日推荐复核（信号日{review['signal_day']}，买入日{review['buy_day']}，卖出日{review['sell_day']}，-2%止损/+7%止盈、同日双触按止损；gap-aware真实执行：跳空穿过按开盘价成交）：止盈{review['win_count']}只，成交{review['bought_count']}只，前{SC.N_STRATEGY}只平均每笔{_rmt}；买入价为实际成交价（含跳空低开更优价），卖出价为实际卖出价，市盈率为当前值、仅供参考。")
        for i in review["items"]:
            _pe = f"市盈率{i['pe']:.1f}倍" if i.get("pe") is not None else "市盈率暂无"
            if i["status"] in ("止盈", "跳空止盈", "止损"):
                lines.append(f"- {i['name']} {i['code']}，买入价{i['buy_price']:.2f}元，卖出价{i['sell_price']:.2f}元，{i['status']}（{i['realized']:+.2%}），{_pe}")
            elif i["status"] == "收盘卖出":
                lines.append(f"- {i['name']} {i['code']}，买入价{i['buy_price']:.2f}元，卖出价{i['sell_price']:.2f}元，收盘卖出（{i['realized']:+.2%}），{_pe}")
            else:
                lines.append(f"- {i['name']} {i['code']}，{i['status']}，{_pe}")
    elif review.get("signal_day"):
        lines.append(f"前两个交易日推荐复核：信号日{review['signal_day']}无存档（日报自10月27日开始），暂无可复核的推荐。")
    else:
        lines.append("前两个交易日推荐复核：暂无（复核失败或数据不足，不影响当日Top10）。")
    def _m_review_lines(rv, label, rule_txt):
        _ls = []
        if rv.get("items"):
            _mr = rv["mean_realized"]
            _mrt = f"{_mr:+.2%}" if _mr is not None else "—"
            _ls.append(f"{label}复核（信号日{rv['signal_day']}，买入日{rv['buy_day']}，持有至{rv['hold_days'][-1]}，{rule_txt}；gap-aware真实执行）：成交{rv['bought_count']}只，前{SC.N_STRATEGY}只平均每笔{_mrt}；买入价为实际成交价（含跳空低开更优价），卖出价为实际卖出价，市盈率为当前值、仅供参考。")
            for i in rv["items"]:
                _pe = f"市盈率{i['pe']:.1f}倍" if i.get("pe") is not None else "市盈率暂无"
                if i["status"] in ("止盈", "跳空止盈", "止损"):
                    _ls.append(f"- {i['name']} {i['code']}，买入价{i['buy_price']:.2f}元，卖出价{i['sell_price']:.2f}元，{i['status']}（{i['realized']:+.2%}），{_pe}")
                elif i["status"] == "收盘卖出":
                    _ls.append(f"- {i['name']} {i['code']}，买入价{i['buy_price']:.2f}元，卖出价{i['sell_price']:.2f}元，收盘卖出（{i['realized']:+.2%}），{_pe}")
                else:
                    _ls.append(f"- {i['name']} {i['code']}，{i['status']}，{_pe}")
        elif rv.get("signal_day"):
            _ls.append(f"{label}复核：信号日{rv['signal_day']}无存档，暂无可复核的推荐。")
        else:
            _ls.append(f"{label}复核：暂无（复核失败或数据不足）。")
        return _ls
    lines.append("")
    lines.extend(_m_review_lines(m2_review, "m=2", "按信号收盘×0.95限价买入、+7%止盈、无止损"))
    lines.append("")
    lines.extend(_m_review_lines(m3_review, "m=3", "按信号收盘×0.95限价买入、+7%止盈、无止损"))
    lines.append("")
    lines.append("【三策略对比累计】gap-aware真实执行口径、已扣0.5%手续费：")
    lines.append(_cmp_line("orig", "原策略"))
    lines.append(_cmp_line("m2", "m=2"))
    lines.append(_cmp_line("m3", "m=3"))
    _atomic_write_text(os.path.join(BASE, "daily_output", f"{day}_email.txt"), "\n".join(lines))
    mark_stage(day, "output")
    print(f"DAILY DONE {day} top10={len(top_list)} retrain={retrain_status} elapsed {time.time()-t0:.0f}s", flush=True)

def resume(day=None):
    """通用续跑入口：读阶段标记，只跑缺失阶段。用法：python daily_pipeline.py --resume [YYYY-MM-DD]"""
    global RESUME_MODE
    RESUME_MODE = True
    if day is None:
        day = datetime.datetime.now(CST).strftime("%Y-%m-%d")
    main(day)

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--resume":
        resume(sys.argv[2] if len(sys.argv) >= 3 else None)
    else:
        main()
