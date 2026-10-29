# 测试集后验：概率>=0.8 候选（5%折扣、-2%/+20%、Top20），以及无门槛对照
import pandas as pd, numpy as np, json, os, glob, pickle, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
files=sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}
m=pickle.load(open(os.path.join(BASE,"new_model_HGB_wd.pkl"),"rb"))
print("model loaded, building test Top20", flush=True)
rows=[]
for f in files:
    d=pd.read_pickle(f); sp=d['code'].map(cmap).values; mask=sp==2
    if not mask.any(): continue
    sub=d[mask]
    X=sub[FEATS].to_numpy(dtype=np.float32); ps=[]
    for i in range(0, len(X), 200000): ps.append(m.predict_proba(X[i:i+200000])[:,1])
    p=np.concatenate(ps); del X, ps
    sub=sub.assign(prob=p)
    for dt, grp in sub.groupby('date'):
        top=grp.nlargest(20,'prob')
        rows.append(top[['date','code','prob','signal_close','buy_low','sell_low','sell_high','sell_close']])
    del d, sub; gc.collect()
top20=pd.concat(rows, ignore_index=True); top20.to_pickle(os.path.join(BASE,"search_test_top20.pkl"))
print(f"test Top20 rows {len(top20)} days {top20['date'].nunique()}", flush=True)
def ev(dsc, stop, targ, K, pm):
    cr=[]; tr=[]
    for dt, grp in top20.groupby('date'):
        g=grp[grp['prob']>=pm]
        if len(g)==0: continue
        tp=g.nlargest(K,'prob'); entry=tp['signal_close'].to_numpy()*(1-dsc); bought=tp['buy_low'].to_numpy()<=entry
        if bought.sum()==0: continue
        ex=tp[bought]; e=entry[bought]; sl=ex['sell_low'].to_numpy(); sh=ex['sell_high'].to_numpy(); sc=ex['sell_close'].to_numpy()
        is_stop=sl<=e*(1-stop); is_targ=(~is_stop)&(sh>=e*(1+targ))
        R=np.where(is_stop, -stop, np.where(is_targ, targ, sc/e-1))
        cr.append(R.mean()); tr.extend(R.tolist())
    cr=np.array(cr); tr=np.array(tr); se=cr.std(ddof=1)/np.sqrt(len(cr)) if len(cr)>1 else float('nan')
    return {'discount':dsc,'stop':stop,'target':targ,'K':K,'prob_min':pm,'mean_cycle':float(cr.mean()),'n_cycles':len(cr),'se':float(se),
            'lower70_one_sided':float(cr.mean()-0.524*se) if len(cr)>1 else None,'p_cycle_gt2':float((cr>0.02).mean()),'p_trade_gt2':float((tr>0.02).mean()),'n_trades':len(tr),'mean_trade':float(tr.mean())}
out=[ev(0.05,0.02,0.20,20,0.8), ev(0.05,0.02,0.20,20,0.0), ev(0.05,0.02,0.20,20,0.7), ev(0.02,0.02,0.20,10,0.0)]
with open(os.path.join(BASE,"search_test_grid.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=1)
for r in out: print(json.dumps(r, ensure_ascii=False), flush=True)
print("SEARCH TEST DONE", flush=True)
