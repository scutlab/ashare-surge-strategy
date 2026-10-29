# 全量训练与周一打分脚本（待 full_raw.pkl 到齐后用 venv python 运行）
# 口径与试点完全一致，区别：
# -  universe: 沪深非 ST（2026-10-23 快照），ST 已在续传合并时剔除，此处再校验一次
# -  特征：与 pilot_features.py 相同的 30 个特征（weekday 保留为可对比变量）
# -  划分：按股票代码排序 i%3 = 训练/验证/测试
# -  模型对比：HGB（飙升监测策略，即去 weekday）、HGB（含 weekday）、LogReg；验证集选模型，测试集只报一次
# -  周一打分：验证选出的模型用训练+验证重训，对 2026-10-23 周五收盘行打分，
#    口径为周一收盘 / 周五收盘 >5%；周五无成交、特征缺失者不进入候选；
#    输出 Top20（含名称、板块、周五收盘价、昨日涨幅、是否涨停、量比、概率），最终交付取 Top10
# -  注意 T+1：周一开盘买入不能周一收盘卖出；兑现周一收盘口径的合规做法是周五收盘买入、
#    周一收盘卖出，或周一开盘买入持有到周二再评估（见 two_strategies.md）
# 运行前检查：full_raw.pkl 存在且代码数接近 5029；否则不要用 partial 文件冒充全量。
import pandas as pd, numpy as np, json, os, time
BASE = os.path.expanduser("~/workspace/ashare")
FEATS = ['vol_5','vol_20','atr_14','big_up_freq_60','big_up_freq_250','ret_1','ret_3','ret_5','ret_10','ret_20','ma_bias_5','ma_bias_20','ma_bias_60','close_vs_high60','close_vs_low60','breakout_20','consec_up','excess_ret_1','excess_ret_5','vol_ratio_5','vol_ratio_20','is_limit_up','limit_cnt_5','limit_cnt_20','mkt_ret','mkt_ret5','rsi_14','macd_diff','macd_hist','weekday']
FEATS_NO_WD = [f for f in FEATS if f != 'weekday']

def board(code):
    c = code[2:]
    if c.startswith('688'): return 'STAR'
    if c.startswith('300') or c.startswith('301'): return 'ChiNext'
    return 'Main'

def limit_thr(code):
    c = code[2:]
    if c.startswith('688') or c.startswith('300') or c.startswith('301'): return 0.195
    return 0.095

def build_features(raw, midx):
    raw['date'] = pd.to_datetime(raw['date']); midx['date'] = pd.to_datetime(midx['date'])
    midx['mkt_ret'] = midx['mkt_adjclose'].pct_change(); midx['mkt_ret5'] = midx['mkt_adjclose'].pct_change(5)
    mkt = midx[['date','mkt_ret','mkt_ret5']]
    frames = []
    for code, g in raw.sort_values(['code','date']).groupby('code'):
        g = g.copy(); adj = g['adjclose']
        g['factor'] = g['adjclose'] / g['close'].replace(0, np.nan)
        g['adj_high'] = g['high'] * g['factor']; g['adj_low'] = g['low'] * g['factor']
        ret = adj.pct_change()
        g['ret_1'] = ret; g['ret_3'] = adj.pct_change(3); g['ret_5'] = adj.pct_change(5); g['ret_10'] = adj.pct_change(10); g['ret_20'] = adj.pct_change(20)
        g['ma_bias_5'] = adj / adj.rolling(5).mean() - 1; g['ma_bias_20'] = adj / adj.rolling(20).mean() - 1; g['ma_bias_60'] = adj / adj.rolling(60).mean() - 1
        g['close_vs_high60'] = adj / adj.rolling(60).max() - 1; g['close_vs_low60'] = adj / adj.rolling(60).min() - 1
        g['breakout_20'] = (adj > adj.shift(1).rolling(20).max()).astype(float)
        prev_adj = adj.shift(1)
        tr = np.maximum(g['adj_high'] - g['adj_low'], np.maximum((g['adj_high'] - prev_adj).abs(), (g['adj_low'] - prev_adj).abs()))
        g['atr_14'] = tr.rolling(14).mean() / adj
        g['vol_5'] = ret.rolling(5).std(); g['vol_20'] = ret.rolling(20).std()
        g['vol_ratio_5'] = g['volume'] / g['volume'].shift(1).rolling(5).mean()
        g['vol_ratio_20'] = g['volume'] / g['volume'].shift(1).rolling(20).mean()
        big = (ret > 0.05).astype(float)
        g['big_up_freq_60'] = big.shift(1).rolling(60).mean(); g['big_up_freq_250'] = big.shift(1).rolling(250).mean()
        lim = (ret >= limit_thr(code)).astype(float)
        g['is_limit_up'] = lim; g['limit_cnt_5'] = lim.shift(1).rolling(5).sum(); g['limit_cnt_20'] = lim.shift(1).rolling(20).sum()
        up = (ret > 0); g['consec_up'] = up.groupby((~up).cumsum()).cumsum()
        delta = adj.diff(); gain = delta.clip(lower=0); loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
        g['rsi_14'] = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
        ema12 = adj.ewm(span=12, adjust=False).mean(); ema26 = adj.ewm(span=26, adjust=False).mean()
        dif = ema12 - ema26; dea = dif.ewm(span=9, adjust=False).mean()
        g['macd_diff'] = dif / adj; g['macd_hist'] = (dif - dea) / adj
        g['weekday'] = g['date'].dt.dayofweek; g['board'] = board(code)
        g['next_ret'] = adj.shift(-1) / adj - 1; g['next_volume'] = g['volume'].shift(-1)
        frames.append(g)
    df = pd.concat(frames, ignore_index=True).merge(mkt, on='date', how='left')
    df['excess_ret_1'] = df['ret_1'] - df['mkt_ret']; df['excess_ret_5'] = df['ret_5'] - df['mkt_ret5']
    return df

if __name__ == '__main__':
    t0 = time.time()
    raw = pd.read_pickle(os.path.join(BASE, 'full_raw.pkl'))
    lst = pd.read_csv(os.path.join(BASE, 'sina_hs_a_20261025.csv'))
    st_codes = set(lst[lst['name'].str.contains('ST', na=False)]['symbol'])
    raw = raw[~raw['code'].isin(st_codes)]
    print(f"raw codes {raw['code'].nunique()} rows {len(raw)}", flush=True)
    midx = pd.read_pickle(os.path.join(BASE, 'index_sse.pkl'))
    df = build_features(raw, midx)
    df = df.replace([np.inf, -np.inf], np.nan)
    # 周五打分用行：保留特征完整但 next_ret 缺失的行，单独存
    friday = df[(df['date'] == '2026-10-23') & (df['volume'] > 0)].dropna(subset=FEATS).copy()
    friday.to_pickle(os.path.join(BASE, 'friday_20261023_features.pkl'))
    print(f"friday candidates {len(friday)}", flush=True)
    df = df.dropna(subset=FEATS + ['next_ret'])
    df = df[(df['volume'] > 0) & (df['next_volume'] > 0)]
    df['y'] = (df['next_ret'] > 0.05).astype(int)
    df.to_pickle(os.path.join(BASE, 'full_features.pkl'))
    print(f"features rows {len(df)} stocks {df['code'].nunique()} y {df['y'].mean():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
    # 训练/验证/测试与周一打分在下一步脚本 full_evaluate.py 中做，避免本文件一次跑太久导致中断丢失特征
    print("features saved, run full_evaluate.py next", flush=True)
