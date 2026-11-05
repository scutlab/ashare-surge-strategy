# 新执行规则全历史评估（测试集股票）：T日按(T-1收盘*0.98)限价买，T+1日触及买入价*1.02按+2%止盈，否则收盘卖出
import pandas as pd, numpy as np, glob, os, json, time
BASE = os.path.expanduser("~/workspace/ashare"); t0=time.time()
files = sorted(glob.glob(os.path.join(BASE,"full_flow_new_*.pkl")))
codes = sorted({c for f in files for c in pd.read_pickle(f)['code'].unique()})
cmap = {c:i%3 for i,c in enumerate(codes)}
S = dict(n=0, fill=0, tp=0, prof=0, ret=0.0)
for f in files:
    d = pd.read_pickle(f)
    d = d[d['code'].map(cmap).values==2]
    entry = d['signal_close'].values*0.98
    filled = d['buy_low'].values<=entry
    tp = d['sell_high'].values>=entry*1.02
    rc = d['sell_close'].values/entry-1
    S['n']+=len(d); S['fill']+=filled.sum(); S['tp']+=(filled&tp).sum()
    rf = np.where(tp,0.02,rc)[filled]
    S['prof']+=(rf>0).sum(); S['ret']+=rf.sum()
    del d
json.dump({k:(float(v) if isinstance(v,np.floating) else int(v)) for k,v in S.items()},
          open(os.path.join(BASE,'eval_new_exec_A.json'),'w'), indent=2)
print(json.dumps({k:(float(v) if isinstance(v,np.floating) else int(v)) for k,v in S.items()}, indent=2))
print(f"elapsed {time.time()-t0:.0f}s")
