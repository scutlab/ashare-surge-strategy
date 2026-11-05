# 4交易日复盘（信号日10-27至10-30），三策略前3只，gap-aware+扣费
import sys; sys.path.insert(0, '.')
import daily_pipeline as dp, strategy_config as SC, pickle, json
import pandas as pd
BASE = '.'
names = dict(zip(pd.read_csv('sina_hs_a_20261025.csv')['symbol'], pd.read_csv('sina_hs_a_20261025.csv')['name']))
# 加载模型
clf = pickle.load(open('daily_model_2026-10-30.pkl','rb'))
ev2 = pickle.load(open('ev_model_10_2.pkl','rb'))
ev3 = pickle.load(open('ev_model_10_3.pkl','rb'))
bars = {d: pickle.load(open(f'daily_bars/{d}.pkl','rb')) for d in ['2026-10-27','2026-10-28','2026-10-29','2026-10-30']}
def top3_from_features(day, model, score_col):
    feat = pickle.load(open(f'daily_features/{day}.pkl','rb')).copy()
    feat[score_col] = model.predict(feat[dp.FEATS].to_numpy('float32'))
    s = feat.sort_values([score_col,'code'], ascending=[False,True])
    lst = []
    for _, r in s.iterrows():
        nm = str(names.get(r['code'],''))
        if ('退' in nm) or ('ST' in nm): continue
        c = float(r['close'])
        lst.append({'code': r['code'], 'name': nm, 'signal_close': c, score_col: float(r[score_col])})
        if len(lst) >= 3: break
    return lst
def top3_orig_arch(day):
    arch = json.load(open(f'daily_output/{day}.json'))
    return arch['top10'][:3]
results = {}
for sig in ['2026-10-27','2026-10-28','2026-10-29','2026-10-30']:
    results[sig] = {}
    # 原策略前3（用存档）
    o3 = top3_orig_arch(sig)
    results[sig]['orig_picks'] = [(t['name'], t['code']) for t in o3]
    # m2/m3前3（现场打分）
    m2_3 = top3_from_features(sig, ev2, '_ev2')
    m3_3 = top3_from_features(sig, ev3, '_ev3')
    results[sig]['m2_picks'] = [(t['name'], t['code']) for t in m2_3]
    results[sig]['m3_picks'] = [(t['name'], t['code']) for t in m3_3]
    # 复核（只算数据齐的）
    # 原策略：buy S+1, sell S+2
    buy_d = {'2026-10-27':'2026-10-28','2026-10-28':'2026-10-29','2026-10-29':'2026-10-30'}.get(sig)
    sell_d = {'2026-10-27':'2026-10-29','2026-10-28':'2026-10-30'}.get(sig)
    if buy_d and sell_d:
        r = dp.build_review(sig, buy_d, sell_d, o3, bars[buy_d], bars[sell_d], {})
        results[sig]['orig'] = {'mean': r['mean_realized'], 'items': [(i['name'], i['status'], i['realized']) for i in r['items']]}
    # m2: buy S+1, hold S+2,S+3
    m2_sched = {'2026-10-27': ('2026-10-28', ['2026-10-29','2026-10-30'])}
    if sig in m2_sched:
        bd, hd = m2_sched[sig]
        bbd = {bd: bars[bd]}; 
        for h in hd: bbd[h] = bars[h]
        # 构造 bars_by_day
        r = dp.build_gap_review(sig, bd, hd, [{'code':t['code'],'signal_close':t['signal_close'],'name':t['name']} for t in m2_3],
                                bbd, 0.10, None, 0.02, tag='m2')
        results[sig]['m2'] = {'mean': r['mean_realized'], 'items': [(i['name'], i['status'], i['realized']) for i in r['items']]}
    # m3: 需要4天，无完整
json.dump(results, open('review_4days.json','w'), ensure_ascii=False, indent=2)
# 打印
for sig in ['2026-10-27','2026-10-28','2026-10-29','2026-10-30']:
    print(f"\n=== 信号日 {sig} ===")
    r = results[sig]
    print(f"原策略前3: {', '.join(n for n,c in r['orig_picks'])}")
    if 'orig' in r:
        print(f"  -> 复核(买{ {'2026-10-27':'10-28','2026-10-28':'10-29'}[sig] }卖{ {'2026-10-27':'10-29','2026-10-28':'10-30'}[sig] }): 平均{r['orig']['mean']:+.2%}")
        for n,s,rt in r['orig']['items']: print(f"     {n} {s} {rt:+.2%}")
    else:
        print(f"  -> 复核: 数据未齐（卖出日在未来）")
    print(f"m=2前3: {', '.join(n for n,c in r['m2_picks'])}")
    if 'm2' in r:
        print(f"  -> 复核(买10-28持有至10-30): 平均{r['m2']['mean']:+.2%}")
        for n,s,rt in r['m2']['items']: print(f"     {n} {s} {rt:+.2%}")
    else:
        print(f"  -> 复核: 数据未齐（持有期未完）")
    print(f"m=3前3: {', '.join(n for n,c in r['m3_picks'])}")
    print(f"  -> 复核: 数据未齐（需要4天行情）")
print("\nDONE")
