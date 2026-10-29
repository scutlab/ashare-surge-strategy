# 平均收益验证：用已保存模型，对测试集（和验证集）按信号日取 Top10，买入日收盘买入
# 两种卖出：A 持有到卖出日收盘；B 盘中触及超5%时按5%卖出（保守，不计超额），否则持有到收盘
# 不计费用、滑点、税；涨停买不到股已在可买入子集中剔除
import pandas as pd, numpy as np, json, os, glob, pickle, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
files=sorted(glob.glob(os.path.join(BASE,"full_flow_batch_*.pkl")))
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}
models={'HGB_wd_y_close': pickle.load(open(os.path.join(BASE,"full_model_HGB_wd_y_close.pkl"),"rb"))}
if os.path.exists(os.path.join(BASE,"full_model_HGB_wd_y_touch.pkl")):
    models['HGB_wd_y_touch']=pickle.load(open(os.path.join(BASE,"full_model_HGB_wd_y_touch.pkl"),"rb"))
for split_name, split_id in [('val',1),('test',2)]:
    # per-date structures: base sum/count, and for each model top10 list
    base={}; tops={m:{} for m in models}
    for f in files:
        d=pd.read_pickle(f); d=d[d['buyable']]
        sp=d['code'].map(cmap).values; mask=sp==split_id
        if not mask.any(): continue
        sub=d[mask]
        close_ret=(sub['sell_close'].to_numpy()/sub['buy_close'].to_numpy()-1)
        touch_ret=(sub['sell_high'].to_numpy()/sub['buy_close'].to_numpy()-1)
        trig=np.where(touch_ret>0.05, 0.05, close_ret)  # B
        dates=sub['date'].values.astype('datetime64[D]').astype(int)
        for dt, cr, tr in zip(dates, close_ret, trig):
            b=base.setdefault(int(dt), [0.0,0]); b[0]+=cr; b[1]+=1
        for mname, m in models.items():
            # predict in chunks of 100k to limit memory
            X=sub[FEATS].to_numpy(dtype=np.float32); ps=[]
            for i in range(0, len(X), 100000):
                ps.append(m.predict_proba(X[i:i+100000])[:,1])
            p=np.concatenate(ps); del X, ps
            for dt, prob, cr, tr in zip(dates, p, close_ret, trig):
                lst=tops[mname].setdefault(int(dt), [])
                lst.append((float(prob), float(cr), float(tr)))
                if len(lst)>50:  # keep only top 50 by prob to bound memory, Top10 derived later
                    lst.sort(key=lambda x: -x[0]); del lst[50:]
        del d, sub, close_ret, touch_ret, trig, dates; gc.collect()
    out={}
    # base per-day mean
    base_days=np.array([v[0]/v[1] for v in base.values() if v[1]>=10])
    out['base_perday_mean']=float(base_days.mean()); out['base_perday_std']=float(base_days.std()); out['base_days']=int(len(base_days)); out['base_pct_days_positive']=float((base_days>0).mean())
    for mname in models:
        for K in [1,5,10]:
            close_means=[]; trig_means=[]
            for dt, lst in tops[mname].items():
                if len(lst)<K: continue
                lst.sort(key=lambda x: -x[0]); top=lst[:K]
                close_means.append(np.mean([x[1] for x in top])); trig_means.append(np.mean([x[2] for x in top]))
            close_means=np.array(close_means); trig_means=np.array(trig_means)
            out[f'{mname}_Top{K}_close_perday_mean']=float(close_means.mean()); out[f'{mname}_Top{K}_close_perday_std']=float(close_means.std()); out[f'{mname}_Top{K}_close_pct_days_positive']=float((close_means>0).mean()); out[f'{mname}_Top{K}_close_pct_days_over5']=float((close_means>0.05).mean())
            out[f'{mname}_Top{K}_trig_perday_mean']=float(trig_means.mean()); out[f'{mname}_Top{K}_trig_perday_std']=float(trig_means.std()); out[f'{mname}_Top{K}_trig_pct_days_positive']=float((trig_means>0).mean())
            out[f'{mname}_Top{K}_days']=int(len(close_means))
    print(split_name, json.dumps(out, ensure_ascii=False, indent=1), flush=True)
    with open(os.path.join(BASE,f"return_verify_{split_name}.json"),"w") as f: json.dump(out,f,ensure_ascii=False,indent=1)
print("RETURN VERIFY DONE", flush=True)
