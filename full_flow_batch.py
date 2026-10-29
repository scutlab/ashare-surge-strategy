# 按特征分批文件逐批建三日循环，避免一次性对 1440 万行做 groupby shift 导致内存溢出
# 每个特征分批文件已包含该批 200 只股票的完整历史，批内 shift 不会跨股票
import pandas as pd, numpy as np, os, glob, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_*.pkl")))
print(f"batches {len(files)}", flush=True)
t0=time.time(); total=0
for fi, f in enumerate(files):
    df=pd.read_pickle(f).sort_values(['code','date']).reset_index(drop=True)
    g=df.groupby('code')
    df['buy_close']=g['adjclose'].shift(-1); df['buy_is_limit']=g['is_limit_up'].shift(-1)
    df['buy_vol_ratio_5']=g['vol_ratio_5'].shift(-1); df['buy_ret']=g['ret_1'].shift(-1)
    df['sell_close']=g['adjclose'].shift(-2); df['sell_high']=g['adj_high'].shift(-2); df['sell_volume']=g['volume'].shift(-2)
    flow=df.dropna(subset=FEATS+['buy_close','sell_close','sell_high'])
    flow=flow[(flow['volume']>0)&(flow['sell_volume']>0)]
    flow['y_close']=(flow['sell_close']/flow['buy_close']-1>0.05).astype(int)
    flow['y_touch']=(flow['sell_high']/flow['buy_close']-1>0.05).astype(int)
    flow['buyable']=(flow['buy_is_limit']<=0)
    keep=['code','date','board']+FEATS+['buy_close','buy_is_limit','buy_vol_ratio_5','buy_ret','sell_close','sell_high','y_close','y_touch','buyable']
    flow=flow[keep].copy()
    out=os.path.join(BASE,f"full_flow_batch_{fi:02d}.pkl")
    flow.to_pickle(out); total+=len(flow)
    print(f"batch {fi} {os.path.basename(f)} -> rows {len(flow)} buyable {int(flow['buyable'].sum())} elapsed {time.time()-t0:.0f}s", flush=True)
    del df, flow; gc.collect()
print(f"FLOW BATCH DONE total {total}", flush=True)
