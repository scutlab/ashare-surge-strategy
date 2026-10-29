# 全量三日循环评估（待特征分批文件到齐后用 venv python 运行，不用管道 tail，避免假完成）
# 信号日 S 特征 -> 买入日 S+1 收盘价买入 -> 卖出日 S+2 判定
# y_close = 卖出日收盘/买入日收盘-1 > 5%；y_touch = 卖出日最高价/买入日收盘-1 > 5%
# 可交易评估必须先剔除买入日涨停股（尾盘买不到）；变体 B（加买入日字段）只在可买入子集上重算
import pandas as pd, numpy as np, json, os, glob
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS=mod.FEATS
files=sorted(glob.glob(os.path.join(BASE,"full_features_batch_*.pkl")))
print("batches", len(files), flush=True)
df=pd.concat([pd.read_pickle(f) for f in files], ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
print(f"features rows {len(df)} stocks {df['code'].nunique()} dates {df['date'].min()} to {df['date'].max()}", flush=True)
# 构建三日对齐（按股票 shift，禁止跨股票）
df['buy_close']=df.groupby('code')['adjclose'].shift(-1)
df['buy_is_limit']=df.groupby('code')['is_limit_up'].shift(-1)
df['buy_vol_ratio_5']=df.groupby('code')['vol_ratio_5'].shift(-1)
df['buy_ret']=df.groupby('code')['ret_1'].shift(-1)
df['sell_close']=df.groupby('code')['adjclose'].shift(-2)
df['sell_high']=df.groupby('code')['adj_high'].shift(-2)
df['sell_volume']=df.groupby('code')['volume'].shift(-2)
flow=df.dropna(subset=FEATS+['buy_close','sell_close','sell_high'])
flow=flow[(flow['volume']>0)&(flow['sell_volume']>0)]
flow['y_close']=(flow['sell_close']/flow['buy_close']-1>0.05).astype(int)
flow['y_touch']=(flow['sell_high']/flow['buy_close']-1>0.05).astype(int)
flow['buyable']=(flow['buy_is_limit']<=0)
flow.to_pickle(os.path.join(BASE,"full_flow.pkl"))
print(f"flow rows {len(flow)} buyable {int(flow['buyable'].sum())} y_close {flow['y_close'].mean():.4f} y_touch {flow['y_touch'].mean():.4f}", flush=True)
