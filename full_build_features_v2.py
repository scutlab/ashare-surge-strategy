# 分批建特征，避免一次性持有全部股票的特征帧导致内存溢出（上一版在老部分 2378 只时无声中断）
import pandas as pd, numpy as np, os, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
build_features=mod.build_features; FEATS=mod.FEATS
t0=time.time(); midx=pd.read_pickle(os.path.join(BASE,"index_sse.pkl"))
lst=pd.read_csv(os.path.join(BASE,"sina_hs_a_20261025.csv"))
st=set(lst[lst['name'].str.contains('ST', na=False)]['symbol'])
friday_frames=[]
for name, path in [("old","full_raw_partial.pkl"),("new","full_new_partial.pkl")]:
    raw=pd.read_pickle(os.path.join(BASE,path))
    raw=raw[~raw['code'].isin(st)].sort_values(['code','date'])
    codes=sorted(raw['code'].unique())
    print(f"{name} codes {len(codes)} rows {len(raw)}", flush=True)
    for bi in range(0, len(codes), 200):
        batch_codes=codes[bi:bi+200]
        sub=raw[raw['code'].isin(batch_codes)]
        df=build_features(sub, midx.copy())
        df=df.replace([np.inf,-np.inf], np.nan)
        fri=df[(df['date']=='2026-10-23') & (df['volume']>0)].dropna(subset=FEATS)
        if len(fri): friday_frames.append(fri[['code','date','board','adjclose','adj_high','volume']+FEATS].copy())
        df=df.dropna(subset=FEATS); df=df[df['volume']>0]
        keep=['code','date','board','adjclose','adj_high','volume']+FEATS+['next_ret']
        df=df[keep].copy()
        for c in FEATS+['adjclose','adj_high']: df[c]=df[c].astype(np.float32)
        out=os.path.join(BASE,f"full_features_batch_{name}_{bi//200:02d}.pkl")
        df.to_pickle(out)
        print(f"{name} batch {bi//200} codes {len(batch_codes)} rows {len(df)} elapsed {time.time()-t0:.0f}s -> {out}", flush=True)
        del sub, df; gc.collect()
    del raw; gc.collect()
if friday_frames:
    pd.concat(friday_frames, ignore_index=True).to_pickle(os.path.join(BASE,"friday_20261023_all.pkl"))
    print(f"friday total {sum(len(f) for f in friday_frames)}", flush=True)
print("FEATURE BUILD V2 DONE", flush=True)
