# 最近1年模拟（新策略）：卖出日在 (2025-10-23, 2026-10-23]、仅测试集股票
# 口径：信号日打分取当日 Top10；买入日按信号收盘挂牌（最低价<=信号收盘才成交）；
# 卖出日先止损-2%后止盈+5%（同日双触按止损）；未触及按卖出收盘；未成交资金闲置
# 资金：3 份轮换复利（按交易信号日顺序轮换，周期 3 交易日）
# 内存注意：分批读、只保留窗口内测试行、每批释放
import pandas as pd, numpy as np, json, os, glob, time, gc, pickle
BASE = os.path.expanduser("~/workspace/ashare")
import importlib.util
spec = importlib.util.spec_from_file_location("full_train", os.path.join(BASE, "full_train.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS = mod.FEATS; FEATS_NO_WD = [f for f in FEATS if f != 'weekday']
t0 = time.time()
flow_files = sorted(glob.glob(os.path.join(BASE, "full_flow_new_*.pkl")))
v3_files = sorted(glob.glob(os.path.join(BASE, "full_features_batch_v3_*.pkl")))
assert len(flow_files) == len(v3_files) == 25, (len(flow_files), len(v3_files))
print(f"batches {len(flow_files)}", flush=True)
# pass 1: codes（与训练一致：全局排序后 %3，测试=2）
all_codes = set()
for f in flow_files:
    d = pd.read_pickle(f)
    all_codes.update(d['code'].unique())
    del d
codes = sorted(all_codes)
cmap = {c: i % 3 for i, c in enumerate(codes)}
print(f"codes {len(codes)} test {sum(1 for v in cmap.values() if v==2)} elapsed {time.time()-t0:.0f}s", flush=True)
# pass 2: 窗口行收集（卖出日需 v3 日期映射：同批同序）
rows = []
WIN_LO = np.datetime64('2025-10-23'); WIN_HI = np.datetime64('2026-10-23')
for fi, (ff, vf) in enumerate(zip(flow_files, v3_files)):
    flow = pd.read_pickle(ff)
    v = pd.read_pickle(vf)[['code', 'date']].sort_values(['code', 'date']).reset_index(drop=True)
    v['sell_date'] = v.groupby('code')['date'].shift(-2)
    m = flow.merge(v, on=['code', 'date'], how='inner')
    del flow, v; gc.collect()
    m = m[(m['sell_date'] > WIN_LO) & (m['sell_date'] <= WIN_HI)]
    m = m[m['code'].map(cmap).values == 2]
    if len(m):
        rows.append(m[['code', 'date', 'sell_date', 'bought', 'realized_ret',
                       'y_strategy', 'both_touch'] + FEATS].copy())
    del m; gc.collect()
    print(f"batch {fi} window_rows {sum(len(r) for r in rows)} elapsed {time.time()-t0:.0f}s", flush=True)
W = pd.concat(rows, ignore_index=True)
del rows; gc.collect()
print(f"window total {len(W)} signal_days {W['date'].nunique()} elapsed {time.time()-t0:.0f}s", flush=True)
W.to_pickle(os.path.join(BASE, "new_sim_1y_window.pkl"))

def simulate(model_path, feats, tag):
    with open(model_path, "rb") as f:
        mdl = pickle.load(f)
    X = W[feats].to_numpy(dtype=np.float32)
    p = np.empty(len(W), dtype=np.float32)
    for i in range(0, len(W), 200000):
        p[i:i+200000] = mdl.predict_proba(X[i:i+200000])[:, 1]
    del X; gc.collect()
    D = pd.DataFrame({'signal_date': W['date'].values, 'code': W['code'].values,
                      'p': p, 'bought': W['bought'].values,
                      # realized_ret 在流水里是 float32，保留 float32 才能用 == -0.02/0.05 精确识别止损/止盈
                      'R': W['realized_ret'].values,
                      'y': W['y_strategy'].values,
                      'both': W['both_touch'].values})
    del p; gc.collect()
    days = sorted(D['signal_date'].unique())
    sleeves = [1.0, 1.0, 1.0]
    cyc_port, cyc_bought_mean = [], []
    n_trades = n_exec = n_win = n_stop = n_take = n_both = 0
    n_cycles = 0
    for di, day in enumerate(days):
        g = D[D['signal_date'] == day]
        if len(g) < 10:
            continue
        top = g.sort_values(['p', 'code'], ascending=[False, True], kind='mergesort').head(10)
        n_cycles += 1
        ex = top[top['bought']]
        port_ret = float(ex['R'].astype(np.float64).sum() / 10.0) if len(ex) else 0.0  # 未成交资金闲置
        sleeves[di % 3] *= (1.0 + port_ret)
        cyc_port.append(port_ret)
        if len(ex):
            cyc_bought_mean.append(float(ex['R'].astype(np.float64).mean()))
        n_trades += 10; n_exec += len(ex)
        n_win += int(top['y'].sum())
        n_stop += int((ex['R'] == -0.02).sum())
        n_take += int((ex['R'] == 0.05).sum())
        n_both += int(ex['both'].sum())
    total_ret = float(np.mean(sleeves) - 1.0)
    out = {
        'model': tag, 'window': 'sell_date in (2025-10-23, 2026-10-23]',
        'split': 'test stocks only', 'capital': '3 sleeves rotating, compounding',
        'signal_days': len(days), 'cycles': n_cycles,
        'top10_picks': n_trades, 'executed': n_exec,
        'exec_rate': n_exec / n_trades if n_trades else None,
        'win_rate_top10': n_win / n_trades if n_trades else None,
        'stop_rate_exec': n_stop / n_exec if n_exec else None,
        'take_rate_exec': n_take / n_exec if n_exec else None,
        'both_touch_exec': n_both, 'both_touch_rate_exec': n_both / n_exec if n_exec else None,
        'per_cycle_port_ret_mean': float(np.mean(cyc_port)) if cyc_port else None,
        'per_cycle_port_ret_median': float(np.median(cyc_port)) if cyc_port else None,
        'per_cycle_bought_mean_ret': float(np.mean(cyc_bought_mean)) if cyc_bought_mean else None,
        'sleeve_multipliers': [float(s) for s in sleeves],
        'total_compound_return': total_ret,
        'base_win_rate_window': float(W['y_strategy'].mean()),
        'window_rows': int(len(W)),
    }
    print(f"{tag} done elapsed {time.time()-t0:.0f}s", flush=True)
    print(json.dumps(out, ensure_ascii=False), flush=True)
    del D; gc.collect()
    return out

res = {}
res['HGB_wd'] = simulate(os.path.join(BASE, "new_model_HGB_wd_trainval.pkl"), FEATS, 'HGB_wd')
res['HGB_nowd'] = simulate(os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl"), FEATS_NO_WD, 'HGB_nowd')
with open(os.path.join(BASE, "new_sim_1y.json"), "w") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("SIM_DONE", flush=True)
