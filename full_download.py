import pandas as pd, urllib.request, json, time, concurrent.futures, os
lst=pd.read_csv("~/workspace/ashare/sina_hs_a_20261025.csv")
lst=lst[lst['symbol'].str.startswith(('sh','sz'))].sort_values('symbol').reset_index(drop=True)
print("universe", len(lst), flush=True)
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
        except Exception as e:
            if attempt==2: return None
            time.sleep(1+attempt)
    return None
out_path="~/workspace/ashare/full_raw.pkl"
# resume: if chunk files exist, skip? Simple: run all, save chunks every 500
t0=time.time(); all_dfs=[]; done=0; failed=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs={ex.submit(fetch,c):c for c in lst['symbol']}
    for fut in concurrent.futures.as_completed(futs):
        done+=1; df=fut.result()
        if df is not None and len(df)>200: all_dfs.append(df)
        else: failed.append(futs[fut])
        if done%500==0:
            print(f"{done}/{len(futs)} ok={len(all_dfs)} failed={len(failed)} elapsed={time.time()-t0:.0f}s", flush=True)
            pd.concat(all_dfs, ignore_index=True).to_pickle("~/workspace/ashare/full_raw_partial.pkl")
print("done", len(all_dfs), "failed", len(failed), failed[:20], "elapsed", time.time()-t0, flush=True)
out=pd.concat(all_dfs, ignore_index=True); out.to_pickle(out_path)
print("rows", len(out), out['date'].min(), out['date'].max(), flush=True)
