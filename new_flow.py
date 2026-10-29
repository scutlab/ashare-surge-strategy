# 新策略流水最终版：同日既触止损又触止盈时按止损算（用户 2026-10-25 20:34 明确：无盘中时间数据时用此规则）
# 其他不变：买入价=信号日收盘，买入日最低价<=买入价才成交；卖出日止损-2%、止盈+5%，否则收盘
import pandas as pd, numpy as np, os, glob, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_v3_*.pkl")))
print(f"v3 batches {len(files)}", flush=True)
t0=time.time(); total=0; both_total=0
for fi, f in enumerate(files):
    df=pd.read_pickle(f).sort_values(['code','date']).reset_index(drop=True)
    g=df.groupby('code')
    df['signal_close']=df['adjclose']
    df['buy_low']=g['adj_low'].shift(-1); df['buy_close']=g['adjclose'].shift(-1); df['buy_volume']=g['volume'].shift(-1)
    df['sell_low']=g['adj_low'].shift(-2); df['sell_high']=g['adj_high'].shift(-2); df['sell_close']=g['adjclose'].shift(-2); df['sell_volume']=g['volume'].shift(-2)
    flow=df.dropna(subset=FEATS+['buy_low','sell_low','sell_high','sell_close'])
    flow=flow[(flow['volume']>0)&(flow['buy_volume']>0)&(flow['sell_volume']>0)]
    flow['bought']=(flow['buy_low']<=flow['signal_close'])
    sig=flow['signal_close'].to_numpy(); sl=flow['sell_low'].to_numpy(); sh=flow['sell_high'].to_numpy(); sc=flow['sell_close'].to_numpy()
    both=(sl<=0.98*sig)&(sh>=1.05*sig)
    R=np.where(sl<=0.98*sig, -0.02, np.where(sh>=1.05*sig, 0.05, sc/sig-1))
    flow['realized_ret']=np.where(flow['bought'].to_numpy(), R, np.nan)
    flow['both_touch']=both&flow['bought'].to_numpy()
    flow['y_strategy']=(flow['bought'] & (flow['realized_ret']>=0.05)).astype(int)
    both_total+=int(flow['both_touch'].sum())
    keep=['code','date','board','signal_close','buy_low','buy_close','sell_low','sell_high','sell_close','bought','both_touch','realized_ret','y_strategy']+FEATS
    flow=flow[keep].copy()
    out=os.path.join(BASE,f"full_flow_new_{fi:02d}.pkl"); flow.to_pickle(out); total+=len(flow)
    print(f"new flow batch {fi} rows {len(flow)} bought {int(flow['bought'].sum())} win {int(flow['y_strategy'].sum())} both_touch {int(flow['both_touch'].sum())} elapsed {time.time()-t0:.0f}s", flush=True)
    del df, flow; gc.collect()
print(f"NEW FLOW DONE total {total} both_touch_total {both_total}", flush=True)
