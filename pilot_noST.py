import pandas as pd, numpy as np, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
df=pd.read_pickle("~/workspace/ashare/pilot_features.pkl")
print("before ST filter stocks", df['code'].nunique(), "rows", len(df), "ST rows", int((df['is_st']>0).sum()))
df=df[df['is_st']==0].copy()
df=df.replace([np.inf,-np.inf], np.nan)
FEATS=['vol_5','vol_20','atr_14','big_up_freq_60','big_up_freq_250','ret_1','ret_3','ret_5','ret_10','ret_20','ma_bias_5','ma_bias_20','ma_bias_60','close_vs_high60','close_vs_low60','breakout_20','consec_up','excess_ret_1','excess_ret_5','vol_ratio_5','vol_ratio_20','is_limit_up','limit_cnt_5','limit_cnt_20','mkt_ret','mkt_ret5','rsi_14','macd_diff','macd_hist','weekday']
df=df.dropna(subset=FEATS)
print("after ST filter stocks", df['code'].nunique(), "rows", len(df), "y rate", round(df['y'].mean(),4))
codes=sorted(df['code'].unique()); split={c:i%3 for i,c in enumerate(codes)}; df['split']=df['code'].map(split)
for s,n in [(0,'train'),(1,'val'),(2,'test')]:
    sub=df[df['split']==s]; print(n, len(sub), "stocks", sub['code'].nunique(), "y", round(sub['y'].mean(),4))
def metrics(y,p,dates):
    out={'n':len(y),'base':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
    order=np.argsort(-p)
    for K in [100,500,1000]:
        if len(y)>=K: out[f'prec@{K}']=float(y.iloc[order[:K]].mean())
    d=pd.DataFrame({'date':dates,'y':y.values,'p':p}); hits=[]
    for _,grp in d.groupby('date'):
        if len(grp)>=10: hits.append(grp.nlargest(10,'p')['y'].mean())
    out['top10_hit_mean']=float(np.mean(hits)) if hits else None; out['top10_days']=len(hits)
    return out
train=df[df['split']==0]; val=df[df['split']==1]; test=df[df['split']==2]
# baseline on test
pred=test['is_limit_up']>0
print("BASELINE limit_up test n", int(pred.sum()), "precision", round(float(test.loc[pred,'y'].mean()),4), "recall", round(float(test.loc[pred,'y'].sum()/test['y'].sum()),4))
pred2=test['ret_1']>0.05
print("BASELINE ret>5% test n", int(pred2.sum()), "precision", round(float(test.loc[pred2,'y'].mean()),4), "recall", round(float(test.loc[pred2,'y'].sum()/test['y'].sum()),4))
results=[]
for name, model in [('HGB_noST', HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42)),
                    ('LogReg_noST', make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0)))]:
    model.fit(train[FEATS], train['y'])
    res={'model':name}
    for sn, sub in [('val',val),('test',test)]:
        p=model.predict_proba(sub[FEATS])[:,1]; res[sn]=metrics(sub['y'], p, sub['date'])
    print(json.dumps(res, ensure_ascii=False, indent=1)); results.append(res)
    if name.startswith('HGB'):
        # permutation importance on val sample 10000, fast-ish
        from sklearn.inspection import permutation_importance
        rng=np.random.RandomState(42); idx=rng.choice(len(val), size=min(10000,len(val)), replace=False); vs=val.iloc[idx]
        pi=permutation_importance(model, vs[FEATS], vs['y'], n_repeats=2, random_state=42, scoring='average_precision')
        imp=sorted(zip(FEATS, pi.importances_mean), key=lambda x:-x[1])
        print("IMPORTANCE")
        for f,v in imp: print(f, round(float(v),4))
        with open("~/workspace/ashare/pilot_importance_noST.json","w") as f: json.dump(imp,f,ensure_ascii=False,indent=1)
with open("~/workspace/ashare/pilot_results_noST.json","w") as f: json.dump(results,f,ensure_ascii=False,indent=1)
df.to_pickle("~/workspace/ashare/pilot_features_noST.pkl")
print("saved pilot_features_noST.pkl")
