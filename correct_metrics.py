import pandas as pd, numpy as np, json, os, glob, pickle, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
m=pickle.load(open(os.path.join(BASE,"full_model_HGB_wd_y_close.pkl"),"rb"))
files=sorted(glob.glob(os.path.join(BASE,"full_flow_batch_*.pkl")))
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}
from sklearn.metrics import roc_auc_score, average_precision_score
for split_name, split_id in [('val',1),('test',2)]:
    ys=[]; ps=[]; dts=[]
    for f in files:
        d=pd.read_pickle(f); d=d[d['buyable']]
        sp=d['code'].map(cmap).values; mask=sp==split_id
        if not mask.any(): continue
        sub=d[mask]
        p=m.predict_proba(sub[FEATS].to_numpy(dtype=np.float32))[:,1]
        ys.append(sub['y_close'].to_numpy()); ps.append(p); dts.append(sub['date'].values.astype('datetime64[D]').astype(int))
        del d, sub, p; gc.collect()
    y=np.concatenate(ys); p=np.concatenate(ps); dt=np.concatenate(dts)
    order=np.argsort(-p)
    out={'n':int(len(y)),'base':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p)),
         'prec@100':float(y[order[:100]].mean()),'prec@500':float(y[order[:500]].mean())}
    df=pd.DataFrame({'date':dt,'y':y,'p':p}); hits=[]
    for _,grp in df.groupby('date'):
        if len(grp)>=10: hits.append(grp.nlargest(10,'p')['y'].mean())
    out['top10_hit_mean']=float(np.mean(hits)); out['top10_days']=len(hits)
    print(split_name, json.dumps(out, ensure_ascii=False), flush=True)
    with open(os.path.join(BASE,f"full_result_HGB_wd_y_close_corrected_{split_name}.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=1)
print("CORRECTED DONE", flush=True)
