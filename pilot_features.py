import pandas as pd, numpy as np, json, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

t0=time.time()
raw=pd.read_pickle("~/workspace/ashare/pilot_raw.pkl")
midx=pd.read_pickle("~/workspace/ashare/index_sse.pkl")
uni=pd.read_csv("~/workspace/ashare/pilot_universe.csv")
st_map={r.symbol: ('ST' in str(r.name)) for r in uni.itertuples()}
raw['date']=pd.to_datetime(raw['date'])
midx['date']=pd.to_datetime(midx['date'])
midx['mkt_ret']=midx['mkt_adjclose'].pct_change()
midx['mkt_ret5']=midx['mkt_adjclose'].pct_change(5)
mkt=midx[['date','mkt_ret','mkt_ret5']]

def board(code):
    c=code[2:]
    if c.startswith('688'): return 'STAR'
    if c.startswith('300') or c.startswith('301'): return 'ChiNext'
    return 'Main'
def limit_thr(code):
    c=code[2:]
    if c.startswith('688') or c.startswith('300') or c.startswith('301'): return 0.195
    return 0.095

frames=[]
for code, g in raw.sort_values(['code','date']).groupby('code'):
    g=g.copy()
    g['factor']=g['adjclose']/g['close'].replace(0, np.nan)
    g['adj_high']=g['high']*g['factor']; g['adj_low']=g['low']*g['factor']
    adj=g['adjclose']
    ret=adj.pct_change()
    g['ret_1']=ret; g['ret_3']=adj.pct_change(3); g['ret_5']=adj.pct_change(5); g['ret_10']=adj.pct_change(10); g['ret_20']=adj.pct_change(20); g['ret_60']=adj.pct_change(60)
    g['ma_bias_5']=adj/g['adjclose'].rolling(5).mean()-1; g['ma_bias_20']=adj/g['adjclose'].rolling(20).mean()-1; g['ma_bias_60']=adj/g['adjclose'].rolling(60).mean()-1
    g['close_vs_high60']=adj/g['adjclose'].rolling(60).max()-1; g['close_vs_low60']=adj/g['adjclose'].rolling(60).min()-1
    g['breakout_20']=(adj>g['adjclose'].shift(1).rolling(20).max()).astype(float)
    prev_adj=adj.shift(1)
    tr=np.maximum(g['adj_high']-g['adj_low'], np.maximum((g['adj_high']-prev_adj).abs(), (g['adj_low']-prev_adj).abs()))
    g['atr_14']=tr.rolling(14).mean()/adj
    g['vol_5']=ret.rolling(5).std(); g['vol_20']=ret.rolling(20).std()
    g['vol_ratio_5']=g['volume']/g['volume'].shift(1).rolling(5).mean()
    g['vol_ratio_20']=g['volume']/g['volume'].shift(1).rolling(20).mean()
    big=(ret>0.05).astype(float)
    g['big_up_freq_60']=big.shift(1).rolling(60).mean(); g['big_up_freq_250']=big.shift(1).rolling(250).mean()
    thr=limit_thr(code)
    lim=(ret>=thr).astype(float)
    g['is_limit_up']=lim
    g['limit_cnt_5']=lim.shift(1).rolling(5).sum(); g['limit_cnt_20']=lim.shift(1).rolling(20).sum()
    up=(ret>0)
    g['consec_up']=up.groupby((~up).cumsum()).cumsum()
    # RSI Wilder
    delta=adj.diff(); gain=delta.clip(lower=0); loss=(-delta).clip(lower=0)
    avg_gain=gain.ewm(alpha=1/14, adjust=False).mean(); avg_loss=loss.ewm(alpha=1/14, adjust=False).mean()
    g['rsi_14']=100-100/(1+avg_gain/avg_loss.replace(0, np.nan))
    ema12=adj.ewm(span=12, adjust=False).mean(); ema26=adj.ewm(span=26, adjust=False).mean()
    dif=ema12-ema26; dea=dif.ewm(span=9, adjust=False).mean()
    g['macd_diff']=dif/adj; g['macd_hist']=(dif-dea)/adj
    g['weekday']=g['date'].dt.weekday
    g['board']=board(code); g['is_st']=float(st_map.get(code, False))
    g['next_ret']=adj.shift(-1)/adj-1
    g['next_volume']=g['volume'].shift(-1)
    frames.append(g)
df=pd.concat(frames, ignore_index=True)
df=df.merge(mkt, on='date', how='left')
df['excess_ret_1']=df['ret_1']-df['mkt_ret']
df['excess_ret_5']=df['ret_5']-df['mkt_ret5']
print("built", len(df), "elapsed", round(time.time()-t0,1))

FEATS_VOL=['vol_5','vol_20','atr_14','big_up_freq_60','big_up_freq_250']
FEATS_MOM=['ret_1','ret_3','ret_5','ret_10','ret_20','ma_bias_5','ma_bias_20','ma_bias_60','close_vs_high60','close_vs_low60','breakout_20','consec_up','excess_ret_1','excess_ret_5']
FEATS_VOLUME=['vol_ratio_5','vol_ratio_20']
FEATS_LIMIT=['is_limit_up','limit_cnt_5','limit_cnt_20']
FEATS_MKT=['mkt_ret','mkt_ret5']
FEATS_AUX=['rsi_14','macd_diff','macd_hist','weekday','is_st']
GROUPS={'vol':FEATS_VOL,'mom':FEATS_MOM,'volume':FEATS_VOLUME,'limit':FEATS_LIMIT,'mkt':FEATS_MKT,'aux':FEATS_AUX}
ALL=sum(GROUPS.values(), [])
df=df.replace([np.inf,-np.inf], np.nan)
df=df.dropna(subset=ALL+['next_ret'])
df=df[(df['volume']>0)&(df['next_volume']>0)]
df['y']=(df['next_ret']>0.05).astype(int)
print("after filter", len(df), "y rate", round(df['y'].mean(),4), "date", df['date'].min(), df['date'].max(), "stocks", df['code'].nunique())
df.to_pickle("~/workspace/ashare/pilot_features.pkl")
# split by stock sorted
codes=sorted(df['code'].unique())
split={c: (i%3) for i,c in enumerate(codes)}  # 0 train 1 val 2 test
df['split']=df['code'].map(split)
for s,name in [(0,'train'),(1,'val'),(2,'test')]:
    sub=df[df['split']==s]; print(name, len(sub), "stocks", sub['code'].nunique(), "y", round(sub['y'].mean(),4))

def metrics(y, p, dates):
    out={}
    out['n']=len(y); out['base']=float(y.mean())
    if y.nunique()<2: return out
    out['auc']=float(roc_auc_score(y,p)); out['pr_auc']=float(average_precision_score(y,p))
    order=np.argsort(-p)
    for K in [100,500,1000]:
        if len(y)>=K: out[f'prec@{K}']=float(y.iloc[order[:K]].mean())
    # per-day top10
    d=pd.DataFrame({'date':dates,'y':y.values,'p':p}); hits=[]
    for _,grp in d.groupby('date'):
        if len(grp)>=10:
            hits.append(grp.nlargest(10,'p')['y'].mean())
    out['top10_hit_mean']=float(np.mean(hits)) if hits else None; out['top10_days']=len(hits)
    return out

def eval_model(name, feats, model_fn, train, val, test):
    m=model_fn(); m.fit(train[feats], train['y'])
    res={'model':name,'feats':feats}
    for split_name, sub in [('val',val),('test',test)]:
        p=m.predict_proba(sub[feats])[:,1]
        res[split_name]=metrics(sub['y'], p, sub['date'])
    print(json.dumps(res, ensure_ascii=False, indent=1), flush=True)
    return res, m

train=df[df['split']==0]; val=df[df['split']==1]; test=df[df['split']==2]
results=[]
# baselines on test
for rule, col in [('today_limit_up','is_limit_up'),('today_ret>5%','ret_1')]:
    pred=(test[col]>= (0.195 if False else 0))  # placeholder
    # actual rule: limit flag or ret>0.05
    if rule=='today_limit_up': pred=test['is_limit_up']>0
    else: pred=test['ret_1']>0.05
    print("BASELINE", rule, "n_pred", int(pred.sum()), "precision", round(float(test.loc[pred,'y'].mean()) if pred.sum() else 0,4), "recall", round(float(test.loc[pred,'y'].sum()/test['y'].sum()),4), flush=True)

def hgb(): return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42)
def logreg(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0))

# full models
for name, fn in [('HGB_full',hgb),('LogReg_full',logreg)]:
    r,_=eval_model(name, ALL, fn, train, val, test); results.append(r)
# group-only HGB
for gname, feats in GROUPS.items():
    r,_=eval_model(f'HGB_only_{gname}', feats, hgb, train, val, test); results.append(r)
# all-minus-one HGB
for gname, feats in GROUPS.items():
    rest=[f for f in ALL if f not in feats]
    r,_=eval_model(f'HGB_minus_{gname}', rest, hgb, train, val, test); results.append(r)

with open("~/workspace/ashare/pilot_results.json","w") as f: json.dump(results,f,ensure_ascii=False,indent=1)
# feature importance from full HGB refit
m=hgb(); m.fit(train[ALL], train['y'])
# permutation importance on val sample (limit to 20000 for speed)
from sklearn.inspection import permutation_importance
rng=np.random.RandomState(42); idx=rng.choice(len(val), size=min(20000,len(val)), replace=False)
vs=val.iloc[idx]
pi=permutation_importance(m, vs[ALL], vs['y'], n_repeats=3, random_state=42, scoring='average_precision')
imp=sorted(zip(ALL, pi.importances_mean), key=lambda x:-x[1])
print("PERMUTATION_IMPORTANCE_PR_AUC")
for f,v in imp: print(f, round(float(v),4))
with open("~/workspace/ashare/pilot_importance.json","w") as f: json.dump(imp,f,ensure_ascii=False,indent=1)
print("total elapsed", round(time.time()-t0,1))
