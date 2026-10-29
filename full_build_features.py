import pandas as pd, numpy as np, os, time, gc
BASE=os.path.expanduser("~/workspace/ashare")
# Reuse feature logic from full_train.py by importing its functions (without running main)
import importlib.util
spec=importlib.util.spec_from_file_location("full_train", os.path.join(BASE,"full_train.py"))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
build_features=mod.build_features
t0=time.time()
midx=pd.read_pickle(os.path.join(BASE,"index_sse.pkl"))
out_chunks=[]
for name, path in [("old","full_raw_partial.pkl"),("new","full_new_partial.pkl")]:
    raw=pd.read_pickle(os.path.join(BASE,path))
    lst=pd.read_csv(os.path.join(BASE,"sina_hs_a_20261025.csv"))
    st=set(lst[lst['name'].str.contains('ST', na=False)]['symbol'])
    raw=raw[~raw['code'].isin(st)]
    print(f"{name} codes {raw['code'].nunique()} rows {len(raw)}", flush=True)
    df=build_features(raw, midx)
    del raw; gc.collect()
    df=df.replace([np.inf,-np.inf], np.nan)
    # Friday rows
    friday=df[(df['date']=='2026-10-23') & (df['volume']>0)].dropna(subset=mod.FEATS).copy()
    friday.to_pickle(os.path.join(BASE,f"friday_20261023_{name}.pkl"))
    print(f"{name} friday {len(friday)}", flush=True)
    df=df.dropna(subset=mod.FEATS+['next_ret'])
    df=df[(df['volume']>0)&(df['next_volume']>0)]
    df['y']=(df['next_ret']>0.05).astype(int)
    # keep only needed cols, float32 for features
    keep=['code','date','board']+mod.FEATS+['next_ret','y']
    df=df[keep].copy()
    for c in mod.FEATS: df[c]=df[c].astype(np.float32)
    df.to_pickle(os.path.join(BASE,f"full_features_{name}.pkl"))
    print(f"{name} features rows {len(df)} stocks {df['code'].nunique()} elapsed {time.time()-t0:.0f}s", flush=True)
    del df; gc.collect()
print("FEATURE BUILD DONE", flush=True)
