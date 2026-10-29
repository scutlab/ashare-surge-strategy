# 单模型训练：一次只训一个模型，训完立即存盘，避免多模型连续跑被中断后全丢
# 用法：python full_train_one.py HGB_wd y_close  /  HGB_nowd y_close  /  HGB_wd y_touch  /  HGB_nowd y_touch
import pandas as pd, numpy as np, json, os, glob, time, gc, sys
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS; FEATS_NO_WD=[f for f in FEATS if f!='weekday']
name=sys.argv[1]; target=sys.argv[2]
feats=FEATS if name=='HGB_wd' else FEATS_NO_WD
t0=time.time()
files=sorted(glob.glob(os.path.join(BASE,"full_flow_batch_*.pkl")))
parts=[]
for f in files:
    d=pd.read_pickle(f)
    d=d[d['buyable']][['code','date']+feats+[target]].copy()
    parts.append(d)
tr=pd.concat(parts, ignore_index=True); del parts; gc.collect()
print(f"loaded buyable rows {len(tr)} elapsed {time.time()-t0:.0f}s", flush=True)
codes=sorted(tr['code'].unique()); cmap={c:i%3 for i,c in enumerate(codes)}
tr['split']=tr['code'].map(cmap); del tr['code']; gc.collect()
train=tr[tr['split']==0]; val=tr[tr['split']==1]; test=tr[tr['split']==2]
print(f"train {len(train)} val {len(val)} test {len(test)}", flush=True)
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
m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42)
print("fitting...", flush=True)
m.fit(train[feats], train[target])
print(f"fitted elapsed {time.time()-t0:.0f}s", flush=True)
import pickle; pickle.dump(m, open(os.path.join(BASE,f"full_model_{name}_{target}.pkl"),"wb"))
res={'model':name,'target':target}
for sn, sub in [('val',val),('test',test)]:
    res[sn]=metrics(sub[target], m.predict_proba(sub[feats])[:,1], sub['date'])
    print(f"{sn} done elapsed {time.time()-t0:.0f}s", flush=True)
with open(os.path.join(BASE,f"full_result_{name}_{target}.json"),"w") as f: json.dump(res,f,ensure_ascii=False,indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True)
print("DONE", flush=True)
