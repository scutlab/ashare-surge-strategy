# 10交易日复盘（信号日10-13至10-26），三策略前3只，gap-aware+扣费
import sys; sys.path.insert(0, '.')
import daily_pipeline as dp, strategy_config as SC, pickle, json, glob
import pandas as pd, numpy as np
BASE = '.'
names = dict(zip(pd.read_csv('sina_hs_a_20261025.csv')['symbol'], pd.read_csv('sina_hs_a_20261025.csv')['name']))
clf = pickle.load(open('daily_model_2026-10-30.pkl','rb'))
ev2 = pickle.load(open('ev_model_5t7_2.pkl','rb'))
ev3 = pickle.load(open('ev_model_5t7_3.pkl','rb'))
# 加载v3特征（只取需要的列）
print("加载v3特征...", flush=True)
dfs = []
for vf in sorted(glob.glob(BASE + '/full_features_batch_v3_*.pkl')):
    v = pd.read_pickle(vf)
    # 只保留 10-12 至 10-30 的日期（需要前后各几天）
    v['dstr'] = v['date'].dt.strftime('%Y-%m-%d')
    v = v[v['dstr'].between('2026-09-07', '2026-10-16')]
    if len(v):
        dfs.append(v[['code','dstr','adj_open','adj_high','adj_low','adjclose'] + dp.FEATS])
allv = pd.concat(dfs, ignore_index=True)
print(f"特征行数: {len(allv)}", flush=True)
# 按日期组织
feat_by_day = {d: g for d, g in allv.groupby('dstr')}
bars_by_day = {}
for d, g in feat_by_day.items():
    bd = {}
    for _, r in g.iterrows():
        # 用复权价直接构造（_adj会处理）
        bd[r['code']] = {'open': float(r['adj_open']), 'high': float(r['adj_high']),
                         'low': float(r['adj_low']), 'close': float(r['adjclose']),
                         'adjclose': float(r['adjclose'])}
    bars_by_day[d] = bd
sig_days = ['2026-09-08','2026-09-09','2026-09-10','2026-09-11','2026-09-14',
            '2026-09-15','2026-09-16','2026-09-17','2026-09-18','2026-09-21',
            '2026-09-22','2026-09-23','2026-09-24','2026-09-25','2026-09-28',
            '2026-09-29','2026-09-30','2026-10-08','2026-10-09','2026-10-12']
def get_top3(day, model, col):
    f = feat_by_day[day].copy()
    f[col] = model.predict(f[dp.FEATS].to_numpy('float32'))
    s = f.sort_values([col,'code'], ascending=[False,True])
    lst = []
    for _, r in s.iterrows():
        nm = str(names.get(r['code'],''))
        if ('退' in nm) or ('ST' in nm): continue
        lst.append({'code': r['code'], 'name': nm, 'signal_close': float(r['adjclose'])})
        if len(lst) >= 3: break
    return lst
results = {}
for sig in sig_days:
    if sig not in feat_by_day:
        print(f"{sig}: 无特征数据", flush=True); continue
    rd = {'picks': {}, 'reviews': {}}
    # 前3只
    o3 = get_top3(sig, clf, '_p')
    m2_3 = get_top3(sig, ev2, '_e2')
    m3_3 = get_top3(sig, ev3, '_e3')
    rd['picks'] = {'orig': [(t['name'],t['code']) for t in o3],
                   'm2': [(t['name'],t['code']) for t in m2_3],
                   'm3': [(t['name'],t['code']) for t in m3_3]}
    # 复核：需要后续交易日
    # 用交易日历推算（简化：按日期顺序找）
    all_days = sorted(bars_by_day.keys())
    try:
        idx = all_days.index(sig)
    except ValueError:
        continue
    # 原策略：buy idx+1, sell idx+2
    if idx+2 < len(all_days):
        bd, sd = all_days[idx+1], all_days[idx+2]
        # 确保是连续交易日（跳过周末已在all_days中体现）
        r = dp.build_review(sig, bd, sd, o3, bars_by_day[bd], bars_by_day[sd], {})
        rd['reviews']['orig'] = {'mean': r['mean_realized'], 'n': r['top_bought_count'],
                                 'items': [(i['name'],i['status'],i.get('realized')) for i in r['items'][:3]]}
    # m2: buy idx+1, hold idx+2, idx+3
    if idx+3 < len(all_days):
        bd, hd = all_days[idx+1], [all_days[idx+2], all_days[idx+3]]
        bbd = {d: bars_by_day[d] for d in [bd]+hd}
        r = dp.build_gap_review(sig, bd, hd, m2_3, bbd, 0.05, None, 0.07, tag='m2')
        rd['reviews']['m2'] = {'mean': r['mean_realized'], 'n': r['top_bought_count'],
                               'items': [(i['name'],i['status'],i.get('realized')) for i in r['items'][:3]]}
    # m3: buy idx+1, hold idx+2, idx+3, idx+4
    if idx+4 < len(all_days):
        bd, hd = all_days[idx+1], [all_days[idx+2], all_days[idx+3], all_days[idx+4]]
        bbd = {d: bars_by_day[d] for d in [bd]+hd}
        r = dp.build_gap_review(sig, bd, hd, m3_3, bbd, 0.05, None, 0.07, tag='m3')
        rd['reviews']['m3'] = {'mean': r['mean_realized'], 'n': r['top_bought_count'],
                               'items': [(i['name'],i['status'],i.get('realized')) for i in r['items'][:3]]}
    results[sig] = rd
    print(f"{sig}: 完成", flush=True)
json.dump(results, open('review_20days_x5t7.json','w'), ensure_ascii=False, indent=2)
# 汇总打印
print("\n" + "="*60)
for sig in sig_days:
    if sig not in results: continue
    r = results[sig]
    print(f"\n信号日 {sig}:")
    for k, label in [('orig','原策略'),('m2','m=2'),('m3','m=3')]:
        pks = ', '.join(n for n,c in r['picks'][k])
        if k in r['reviews']:
            rv = r['reviews'][k]
            mn = f"{rv['mean']:+.2%}" if rv['mean'] is not None else "无成交"
            print(f"  {label}前3: {pks} -> 平均{mn} ({rv['n']}只成交)")
        else:
            print(f"  {label}前3: {pks} -> (数据未齐)")
print("\nDONE")
