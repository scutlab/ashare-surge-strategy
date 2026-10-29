import pandas as pd, numpy as np, json, os, glob, time
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS; FEATS_NO_WD=[f for f in FEATS if f!='weekday']
t0=time.time()
files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_*.pkl")))
print(f"batches {len(files)}", flush=True)
df=pd.concat([pd.read_pickle(f) for f in files], ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
print(f"features rows {len(df)} stocks {df['code'].nunique()} dates {df['date'].min()} to {df['date'].max()} elapsed {time.time()-t0:.0f}s", flush=True)
for col, shift in [('buy_close','adjclose'),('buy_is_limit','is_limit_up'),('buy_vol_ratio_5','vol_ratio_5'),('buy_ret','ret_1'),('sell_close','adjclose'),('sell_high','adj_high'),('sell_volume','volume')]:
    pass
g=df.groupby('code')
df['buy_close']=g['adjclose'].shift(-1); df['buy_is_limit']=g['is_limit_up'].shift(-1)
df['buy_vol_ratio_5']=g['vol_ratio_5'].shift(-1); df['buy_ret']=g['ret_1'].shift(-1)
df['sell_close']=g['adjclose'].shift(-2); df['sell_high']=g['adj_high'].shift(-2); df['sell_volume']=g['volume'].shift(-2)
flow=df.dropna(subset=FEATS+['buy_close','sell_close','sell_high'])
flow=flow[(flow['volume']>0)&(flow['sell_volume']>0)]
flow['y_close']=(flow['sell_close']/flow['buy_close']-1>0.05).astype(int)
flow['y_touch']=(flow['sell_high']/flow['buy_close']-1>0.05).astype(int)
flow['buyable']=(flow['buy_is_limit']<=0)
flow.to_pickle(os.path.join(BASE,"full_flow.pkl"))
print(f"flow rows {len(flow)} buyable {int(flow['buyable'].sum())} y_close {flow['y_close'].mean():.4f} y_touch {flow['y_touch'].mean():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
del df; import gc; gc.collect()
# Train on buyable only (tradable population)
tr=flow[flow['buyable']].copy()
codes=sorted(tr['code'].unique()); split={c:i%3 for i,c in enumerate(codes)}; tr['split']=tr['code'].map(split)
train=tr[tr['split']==0]; val=tr[tr['split']==1]; test=tr[tr['split']==2]
print(f"buyable train {len(train)} val {len(val)} test {len(test)}", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
def metrics(y,p,dates):
    out={'n':int(len(y)),'base':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
    order=np.argsort(-p)
    for K in [100,500]:
        if len(y)>=K: out[f'prec@{K}']=float(y.iloc[order[:K]].mean())
    d=pd.DataFrame({'date':dates,'y':y.values,'p':p}); hits=[]
    for _,grp in d.groupby('date'):
        if len(grp)>=10: hits.append(grp.nlargest(10,'p')['y'].mean())
    out['top10_hit_mean']=float(np.mean(hits)) if hits else None; out['top10_days']=len(hits)
    return out
results={}
for target in ['y_close','y_touch']:
    for name, feats in [('HGB_wd',FEATS),('HGB_nowd',FEATS_NO_WD)]:
        m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42)
        m.fit(train[feats], train[target])
        res={'model':name,'target':target}
        for sn, sub in [('val',val),('test',test)]:
            res[sn]=metrics(sub[target], m.predict_proba(sub[feats])[:,1], sub['date'])
        print(json.dumps(res, ensure_ascii=False), flush=True); results[f"{name}_{target}"]=res
        if name=='HGB_wd' and target=='y_close':
            import pickle; pickle.dump(m, open(os.path.join(BASE,"full_model_hgb_wd_y_close.pkl"),"wb"))
with open(os.path.join(BASE,"full_flow_results.json"),"w") as f: json.dump(results,f,ensure_ascii=False,indent=1)
print("FULL TRAIN DONE", flush=True)
