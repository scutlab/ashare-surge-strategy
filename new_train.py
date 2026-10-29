# 新策略训练：单模型、numpy、省内存，与 full_train_one_v2 同结构，目标 y_strategy
import pandas as pd, numpy as np, json, os, glob, time, gc, sys, pickle
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS; FEATS_NO_WD=[f for f in FEATS if f!='weekday']
name=sys.argv[1]; feats=FEATS if name=='HGB_wd' else FEATS_NO_WD
t0=time.time(); files=sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(cmap)}", flush=True)
parts={'train':[],'val':[],'test':[]}
for f in files:
    d=pd.read_pickle(f)
    sp=d['code'].map(cmap).values; dates=d['date'].values.astype('datetime64[D]').astype(int)
    X=d[feats].to_numpy(dtype=np.float32); y=d['y_strategy'].to_numpy(dtype=np.int8); bought=d['bought'].to_numpy(); R=d['realized_ret'].to_numpy(dtype=np.float32)
    for s,key in [(0,'train'),(1,'val'),(2,'test')]:
        m=sp==s
        if m.any(): parts[key].append((X[m], y[m], dates[m], bought[m], R[m]))
    del d, sp, dates, X, y, bought, R; gc.collect()
def stack(key):
    return (np.concatenate([p[0] for p in parts[key]]), np.concatenate([p[1] for p in parts[key]]), np.concatenate([p[2] for p in parts[key]]), np.concatenate([p[3] for p in parts[key]]), np.concatenate([p[4] for p in parts[key]]))
Xtr, ytr, _, _, _ = stack('train'); print(f"train {len(ytr)} elapsed {time.time()-t0:.0f}s", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42, early_stopping=False, max_bins=128)
m.fit(Xtr, ytr); print(f"fitted elapsed {time.time()-t0:.0f}s", flush=True); del Xtr, ytr; gc.collect()
pickle.dump(m, open(os.path.join(BASE,f"new_model_{name}.pkl"),"wb"))
def evaluate(key):
    X,y,dt,bought,R=stack(key); p=m.predict_proba(X)[:,1]; del X; gc.collect()
    out={'n':int(len(y)),'base_win':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
    order=np.argsort(-p)
    for K in [100,500]:
        if len(y)>=K: out[f'prec@{K}']=float(y[order[:K]].mean())
    # per-day Top10 signal hit + realized return over executed
    df=pd.DataFrame({'date':dt,'y':y,'p':p,'bought':bought,'R':R}); del y, dt, bought, R, p; gc.collect()
    hits=[]; rets=[]; exec_rates=[]; days=0
    for _,grp in df.groupby('date'):
        if len(grp)<10: continue
        top=grp.nlargest(10,'p'); days+=1
        hits.append(top['y'].mean()); exec_rates.append(top['bought'].mean())
        ex=top[top['bought']]
        if len(ex): rets.append(ex['R'].mean())
    out['top10_signal_hit_mean']=float(np.mean(hits)); out['top10_exec_rate']=float(np.mean(exec_rates)); out['top10_realized_ret_perday_mean']=float(np.mean(rets)) if rets else None; out['top10_days']=days; out['top10_ret_days']=len(rets)
    print(f"{key} done elapsed {time.time()-t0:.0f}s", flush=True); return out
res={'model':name,'target':'y_strategy','val':evaluate('val'),'test':evaluate('test')}
with open(os.path.join(BASE,f"new_result_{name}.json"),"w") as f: json.dump(res,f,ensure_ascii=False,indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True); print("DONE", flush=True)
