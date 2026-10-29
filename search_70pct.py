# 寻找平均盈利70%把握>=2%的策略：先在验证集上搜，测试集后验
# 定义：单位=每个信号日的TopK组合收益（等权、只算实际买入股），N=周期数
# 70%把握：单侧70%下限 = 均值 - 0.524*SE >= 0.02；同时报双侧70%下限 = 均值 - 1.036*SE
# 搜：入场折扣 d、止损 s、止盈 t、TopK；先从 new_flow 数据取验证集每信号日 Top20（按 train-only wd 模型概率），再在 Top20 上算所有组合
import pandas as pd, numpy as np, json, os, glob, pickle, gc, itertools
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
print("model loaded, building val Top20", flush=True)
rows=[]
for f in files:
    d=pd.read_pickle(f); sp=d['code'].map(cmap).values; mask=sp==1
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
top20=pd.concat(rows, ignore_index=True); top20.to_pickle(os.path.join(BASE,"search_val_top20.pkl"))
print(f"val Top20 rows {len(top20)} days {top20['date'].nunique()}", flush=True)
# grid search on Top20
results=[]
for dsc, stop, targ, K in itertools.product([0.0,0.02,0.05],[0.02,0.05,0.10],[0.05,0.10,0.20],[5,10,20]):
    cycle_rets=[]; exec_rates=[]; win_rates=[]
    for dt, grp in top20.groupby('date'):
        top=grp.nlargest(K,'prob')
        entry=top['signal_close'].to_numpy()*(1-dsc)
        bought=top['buy_low'].to_numpy()<=entry
        if bought.sum()==0: continue
        ex=top[bought]; entry_ex=entry[bought]
        sl=ex['sell_low'].to_numpy(); sh=ex['sell_high'].to_numpy(); sc=ex['sell_close'].to_numpy()
        R=np.where(sl<=entry_ex*(1-stop), -stop, np.where(sh>=entry_ex*(1+targ), targ, sc/entry_ex-1))
        cycle_rets.append(R.mean()); exec_rates.append(bought.mean()); win_rates.append((R>=targ-1e-9).mean() if targ<=0.20 else 0)
    cycle_rets=np.array(cycle_rets); mean=float(cycle_rets.mean()); std=float(cycle_rets.std(ddof=1)); n=len(cycle_rets); se=std/np.sqrt(n)
    results.append({'discount':dsc,'stop':stop,'target':targ,'K':K,'mean':mean,'std':std,'n_cycles':n,'se':float(se),
                    'lower70_one_sided':float(mean-0.524*se),'lower70_two_sided':float(mean-1.036*se),
                    'exec_rate':float(np.mean(exec_rates)),'win_rate_exec_mean':float(np.mean(win_rates))})
results=sorted(results, key=lambda x: x['lower70_one_sided'], reverse=True)
with open(os.path.join(BASE,"search_val_grid.json"),"w") as f: json.dump(results,f,ensure_ascii=False,indent=1)
for r in results[:15]: print(json.dumps(r, ensure_ascii=False), flush=True)
print("SEARCH VAL DONE", flush=True)
