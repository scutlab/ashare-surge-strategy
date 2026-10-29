import pandas as pd, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
df=pd.read_pickle("~/workspace/ashare/pilot_features.pkl")
print("inf counts")
print(np.isinf(df.select_dtypes(include=[float])).sum()[lambda s: s>0])
df=df.replace([np.inf,-np.inf], np.nan)
FEATS=['vol_5','vol_20','atr_14','big_up_freq_60','big_up_freq_250','ret_1','ret_3','ret_5','ret_10','ret_20','ma_bias_5','ma_bias_20','ma_bias_60','close_vs_high60','close_vs_low60','breakout_20','consec_up','excess_ret_1','excess_ret_5','vol_ratio_5','vol_ratio_20','is_limit_up','limit_cnt_5','limit_cnt_20','mkt_ret','mkt_ret5','rsi_14','macd_diff','macd_hist','weekday','is_st']
df=df.dropna(subset=FEATS)
codes=sorted(df['code'].unique()); split={c:i%3 for i,c in enumerate(codes)}; df['split']=df['code'].map(split)
train=df[df['split']==0]; val=df[df['split']==1]
m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42)
m.fit(train[FEATS], train['y'])
val=val.copy(); val['p']=m.predict_proba(val[FEATS])[:,1]
top=val.nlargest(100,'p')
print("top100 y mean", top['y'].mean(), "dates", top['date'].nunique())
print(top['date'].value_counts().head(20))
print(top[['code','date','ret_1','is_limit_up','vol_ratio_5','vol_5','big_up_freq_60','next_ret','y','board']].head(30).to_string())
print("\ntop100 by board", top['board'].value_counts().to_dict(), "val board base", val.groupby('board')['y'].mean().to_dict())
print("\nlimit_up in top100", top['is_limit_up'].mean(), "ret1 mean", top['ret_1'].mean(), "ret1>0.09", (top['ret_1']>0.09).mean())
# What if we remove is_limit_up and ret_1 etc? Check correlation of p with next day market return?
# Per-date: for top100 dates, what was next day market return? Merge index next day?
midx=pd.read_pickle("~/workspace/ashare/index_sse.pkl"); midx['date']=pd.to_datetime(midx['date']); midx['next_mkt_ret']=midx['mkt_adjclose'].shift(-1)/midx['mkt_adjclose']-1
top2=top.merge(midx[['date','next_mkt_ret']], on='date', how='left')
print("top100 next_mkt_ret mean", top2['next_mkt_ret'].mean(), "median", top2['next_mkt_ret'].median())
print("val next_mkt_ret overall approx: check dates in top100 vs all")
# Check: are top100 mostly a handful of stocks?
print("top100 stocks", top['code'].value_counts().head(20).to_dict())
# Check leakage test: shuffle? Train on train, evaluate val but with features shifted? Simpler: check if any feature equals next_ret proxy: correlation of each feature with y in val
for f in FEATS:
    print(f, round(float(np.corrcoef(val[f].fillna(0), val['y'])[0,1]),3))
