import pandas as pd, urllib.request, json, time, concurrent.futures, os, shutil
BASE=os.path.expanduser("~/workspace/ashare")
lst=pd.read_csv(os.path.join(BASE,"sina_hs_a_20261025.csv"))
lst=lst[lst['symbol'].str.startswith(('sh','sz'))].sort_values('symbol')
st_codes=set(lst[lst['name'].str.contains('ST', na=False)]['symbol'])
target=[c for c in lst['symbol'] if c not in st_codes]
print(f"target non-ST {len(target)}", flush=True)
def yahoo_sym(s): return s[2:]+".SS" if s.startswith("sh") else s[2:]+".SZ"
def fetch(sym_code):
    ysym=yahoo_sym(sym_code)
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}?period1=820454400&period2=1792800000&interval=1d"
    for attempt in range(3):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r: j=json.load(r)
            res=j['chart']['result'][0]; ts=res['timestamp']; q=res['indicators']['quote'][0]; adj=res['indicators']['adjclose'][0]['adjclose']
            df=pd.DataFrame({'code':sym_code,'date':pd.to_datetime(ts, unit='s').tz_localize('UTC').tz_convert('Asia/Shanghai').date,'open':q['open'],'high':q['high'],'low':q['low'],'close':q['close'],'adjclose':adj,'volume':q['volume']})
            return df.dropna(subset=['adjclose'])
        except Exception:
            if attempt==2: return None
            time.sleep(1+attempt)
    return None
# old partial, non-ST only
old=pd.read_pickle(os.path.join(BASE,"full_raw_partial.pkl"))
old=old[~old['code'].isin(st_codes)]
have=set(old['code'].unique())
missing=[c for c in target if c not in have]
print(f"partial non-ST codes {len(have)} missing {len(missing)}", flush=True)
# incremental new file: if a previous v2 partial exists, resume from it too
new_path=os.path.join(BASE,"full_new_partial.pkl")
new_frames=[]; new_have=set()
if os.path.exists(new_path):
    try:
        nd=pd.read_pickle(new_path); new_frames=[nd]; new_have=set(nd['code'].unique()); print(f"v2 partial exists codes {len(new_have)} rows {len(nd)}", flush=True)
    except Exception as e:
        print(f"v2 partial corrupt {e}, ignoring", flush=True)
missing=[c for c in missing if c not in new_have]
print(f"to download now {len(missing)}", flush=True)
def save_atomic(df, path):
    tmp=path+".tmp"
    df.to_pickle(tmp); os.replace(tmp, path)
failed_path=os.path.join(BASE,"full_download_failed_v2.json")
failed=[]
if os.path.exists(failed_path):
    try: failed=json.load(open(failed_path))
    except Exception: failed=[]
t0=time.time(); done=0
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs={ex.submit(fetch,c):c for c in missing}
    for fut in concurrent.futures.as_completed(futs):
        done+=1; df=fut.result()
        if df is not None and len(df)>200: new_frames.append(df)
        else: failed.append(futs[fut])
        if done%500==0 or done==len(missing):
            allnew=pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
            save_atomic(allnew, new_path)
            with open(failed_path,"w") as f: json.dump(failed,f)
            print(f"{done}/{len(missing)} new_codes={allnew['code'].nunique() if len(allnew) else 0} failed={len(failed)} elapsed={time.time()-t0:.0f}s checkpoint saved", flush=True)
# final merge only via temp+rename
allnew=pd.concat(new_frames, ignore_index=True) if new_frames else pd.DataFrame()
out=pd.concat([old, allnew], ignore_index=True) if len(allnew) else old
save_atomic(out, os.path.join(BASE,"full_raw.pkl"))
with open(os.path.join(BASE,"full_download_failed.json"),"w") as f: json.dump(failed,f)
print(f"FINAL rows {len(out)} codes {out['code'].nunique()} date {out['date'].min()} to {out['date'].max()} failed {failed}", flush=True)
