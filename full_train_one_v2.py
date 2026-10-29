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
# code -> split mapping first (small)
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}; del all_codes, codes
print(f"codes {len(cmap)}", flush=True)
parts={'train':[],'val':[],'test':[]}
for f in files:
    d=pd.read_pickle(f); d=d[d['buyable']]
    sp=d['code'].map(cmap).values
    dates=d['date'].values.astype('datetime64[D]').astype(int)
    X=d[feats].to_numpy(dtype=np.float32); y=d[target].to_numpy(dtype=np.int8)
    for s, key in [(0,'train'),(1,'val'),(2,'test')]:
        m=sp==s
        if m.any(): parts[key].append((X[m], y[m], dates[m]))
    del d, sp, dates, X, y; gc.collect()
def stack(key):
    X=np.concatenate([p[0] for p in parts[key]]); y=np.concatenate([p[1] for p in parts[key]]); dt=np.concatenate([p[2] for p in parts[key]])
    return X, y, dt
Xtr, ytr, _ = stack('train'); print(f"train {len(ytr)} elapsed {time.time()-t0:.0f}s", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42, early_stopping=False, max_bins=128)
m.fit(Xtr, ytr); print(f"fitted elapsed {time.time()-t0:.0f}s", flush=True)
del Xtr, ytr; gc.collect()
import pickle; pickle.dump(m, open(os.path.join(BASE,f"full_model_{name}_{target}.pkl"),"wb"))
def metrics(key):
    X,y,dt=stack(key); p=m.predict_proba(X)[:,1]; del X; gc.collect()
    out={'n':int(len(y)),'base':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
    order=np.argsort(-p)
    for K in [100,500]:
        if len(y)>=K: out[f'prec@{K}']=float(y[order[:K]].mean())
    df=pd.DataFrame({'date':dt,'y':y,'p':p}); del y, dt, p; gc.collect(); hits=[]
    for _,grp in df.groupby('date'):
        if len(grp)>=10: hits.append(grp.nlargest(10,'p')['y'].mean())
    out['top10_hit_mean']=float(np.mean(hits)) if hits else None; out['top10_days']=len(hits)
    print(f"{key} done elapsed {time.time()-t0:.0f}s", flush=True); return out
res={'model':name,'target':target,'val':metrics('val'),'test':metrics('test')}
with open(os.path.join(BASE,f"full_result_{name}_{target}.json"),"w") as f: json.dump(res,f,ensure_ascii=False,indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True); print("DONE", flush=True)
