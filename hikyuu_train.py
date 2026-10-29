# hikyuu训练：与new_train.py同协议（HGB参数、code%3划分、Top10评估），KDJ除外（hikyuu空值传播致76%缺失）
import pandas as pd, numpy as np, json, os, glob, time, gc, pickle
BASE=os.path.expanduser("~/workspace/ashare")
FEATS=["hk_ma_bias_5","hk_ma_bias_20","hk_ma_bias_60","hk_ema_bias_12","hk_macd_dif","hk_macd_dea","hk_macd_hist","hk_rsi_14","hk_atr_ratio","hk_boll_pos","hk_wr_10","hk_vol_ratio_5","hk_vol_ratio_20","hk_close_vs_high60","hk_close_vs_low60","hk_std_ret_20","hk_ret_1","hk_ret_5","hk_ret_20","hk_adx","hk_plus_di","hk_minus_di","excess_ret_1","is_limit_up"]
t0=time.time(); files=sorted(glob.glob(os.path.join(BASE,"hikyuu_features/hikyuu_flow_batch_*.pkl")))
assert len(files)==25, len(files)
all_codes=set()
for f in files:
    d=pd.read_pickle(f); all_codes.update(d['code'].unique()); del d
codes=sorted(all_codes); cmap={c:i%3 for i,c in enumerate(codes)}; print(f"codes {len(cmap)}", flush=True)
parts={'train':[],'val':[],'test':[]}
for f in files:
    d=pd.read_pickle(f); sp=d['code'].map(cmap).values; dates=d['date'].values.astype('datetime64[D]').astype(int)
    X=d[FEATS].to_numpy(dtype=np.float32); y=d['y_strategy'].to_numpy(dtype=np.int8); bought=d['bought'].to_numpy(); R=d['realized_ret'].to_numpy(dtype=np.float32)
    for s,key in [(0,'train'),(1,'val'),(2,'test')]:
        m=sp==s
        if m.any(): parts[key].append((X[m], y[m], dates[m], bought[m], R[m]))
    del d, sp, dates, X, y, bought, R; gc.collect()
def stack(key): return tuple(np.concatenate([p[i] for p in parts[key]]) for i in range(5))
Xtr, ytr, _, _, _ = stack('train'); print(f"train {len(ytr)} elapsed {time.time()-t0:.0f}s", flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
m=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=3, min_samples_leaf=100, class_weight='balanced', random_state=42, early_stopping=False, max_bins=128)
m.fit(Xtr, ytr); print(f"fitted elapsed {time.time()-t0:.0f}s", flush=True); del Xtr, ytr; gc.collect()
pickle.dump(m, open(os.path.join(BASE,"hikyuu_model_hgb.pkl"),"wb"))
def evaluate(key):
    X,y,dt,bought,R=stack(key); p=m.predict_proba(X)[:,1]; del X; gc.collect()
    out={'n':int(len(y)),'base_win':float(y.mean()),'auc':float(roc_auc_score(y,p)),'pr_auc':float(average_precision_score(y,p))}
    order=np.argsort(-p)
    for K in [100,500]:
        if len(y)>=K: out[f'prec@{K}']=float(y[order[:K]].mean())
    df=pd.DataFrame({'date':dt,'y':y,'p':p,'bought':bought,'R':R}); del y, dt, bought, R, p; gc.collect()
    hits=[]; rets=[]; exec_rates=[]; days=0
    for _,grp in df.groupby('date'):
        if len(grp)<10: continue
        top=grp.nlargest(10,'p'); days+=1; hits.append(top['y'].mean()); exec_rates.append(top['bought'].mean())
        ex=top[top['bought']]
        if len(ex): rets.append(ex['R'].mean())
    out['top10_signal_hit_mean']=float(np.mean(hits)); out['top10_exec_rate']=float(np.mean(exec_rates)); out['top10_realized_ret_perday_mean']=float(np.mean(rets)) if rets else None; out['top10_days']=days; out['top10_ret_days']=len(rets)
    print(f"{key} done elapsed {time.time()-t0:.0f}s", flush=True); return out
res={'model':'hikyuu_hgb','target':'y_strategy','feats':FEATS,'val':evaluate('val'),'test':evaluate('test')}
with open(os.path.join(BASE,"hikyuu_result_hgb.json"),"w") as f: json.dump(res,f,ensure_ascii=False,indent=1)
print(json.dumps(res, ensure_ascii=False), flush=True); print("DONE", flush=True)
