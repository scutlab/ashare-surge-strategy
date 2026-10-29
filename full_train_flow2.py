import pandas as pd, numpy as np, json, os, glob, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS; FEATS_NO_WD=[f for f in FEATS if f!='weekday']
t0=time.time()
files=sorted(glob.glob(os.path.join(BASE,"full_flow_batch_*.pkl")))
print(f"flow batches {len(files)}", flush=True)
tr=pd.concat([pd.read_pickle(f) for f in files], ignore_index=True)
print(f"flow rows {len(tr)} buyable {int(tr['buyable'].sum())} y_close {tr['y_close'].mean():.4f} y_touch {tr['y_touch'].mean():.4f} elapsed {time.time()-t0:.0f}s", flush=True)
tr=tr[tr['buyable']].copy()
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
        if name=='HGB_wd':
            import pickle; pickle.dump(m, open(os.path.join(BASE,f"full_model_hgb_wd_{target}.pkl"),"wb"))
with open(os.path.join(BASE,"full_flow_results.json"),"w") as f: json.dump(results,f,ensure_ascii=False,indent=1)
print("FULL TRAIN DONE", flush=True)
