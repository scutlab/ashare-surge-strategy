import pandas as pd, urllib.request, json, time, concurrent.futures, sys
lst=pd.read_csv("~/workspace/ashare/sina_hs_a_20261025.csv")
lst=lst[lst['symbol'].str.startswith(('sh','sz'))].sort_values('symbol').reset_index(drop=True)
# deterministic stratified pilot: every 27th
pilot=lst.iloc[::27].copy()
print("universe sh/sz", len(lst), "pilot", len(pilot))
print(pilot[['symbol','name']].to_string())
pilot.to_csv("~/workspace/ashare/pilot_universe.csv", index=False)

def yahoo_sym(s):
    # sh600000 -> 600000.SS, sz000001 -> 000001.SZ
    return s[2:]+".SS" if s.startswith("sh") else s[2:]+".SZ"

def fetch(sym_code):
    ysym=yahoo_sym(sym_code)
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}?period1=820454400&period2=1792800000&interval=1d"
    for attempt in range(3):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                j=json.load(r)
            res=j['chart']['result'][0]
            ts=res['timestamp']; q=res['indicators']['quote'][0]; adj=res['indicators']['adjclose'][0]['adjclose']
            df=pd.DataFrame({'code':sym_code,'date':pd.to_datetime(ts, unit='s').tz_localize('UTC').tz_convert('Asia/Shanghai').date,
                             'open':q['open'],'high':q['high'],'low':q['low'],'close':q['close'],'adjclose':adj,'volume':q['volume']})
            df=df.dropna(subset=['adjclose'])
            return df
        except Exception as e:
            if attempt==2:
                print("FAIL", sym_code, ysym, e, flush=True); return None
            time.sleep(1+attempt)
    return None

t0=time.time()
all_dfs=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs={ex.submit(fetch, c):c for c in pilot['symbol']}
    done=0
    for fut in concurrent.futures.as_completed(futs):
        done+=1
        df=fut.result()
        if df is not None and len(df)>200:
            all_dfs.append(df)
        if done%20==0:
            print(f"{done}/{len(futs)} ok={len(all_dfs)} elapsed={time.time()-t0:.0f}s", flush=True)
print("done stocks", len(all_dfs), "elapsed", time.time()-t0)
if all_dfs:
    out=pd.concat(all_dfs, ignore_index=True)
    out.to_pickle("~/workspace/ashare/pilot_raw.pkl")
    print("rows", len(out), "date range", out['date'].min(), out['date'].max())
    print(out.groupby('code').size().describe())
# index
idx=fetch("sh000001") if False else None
# fetch SSE composite directly: yahoo 000001.SS
def fetch_index():
    url="https://query1.finance.yahoo.com/v8/finance/chart/000001.SS?period1=820454400&period2=1792800000&interval=1d"
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r: j=json.load(r)
    res=j['chart']['result'][0]; ts=res['timestamp']; q=res['indicators']['quote'][0]; adj=res['indicators']['adjclose'][0]['adjclose']
    return pd.DataFrame({'date':pd.to_datetime(ts, unit='s').tz_localize('UTC').tz_convert('Asia/Shanghai').date,'mkt_adjclose':adj,'mkt_close':q['close']})
try:
    midx=fetch_index(); midx.to_pickle("~/workspace/ashare/index_sse.pkl"); print("index rows", len(midx), midx['date'].min(), midx['date'].max())
except Exception as e:
    print("index FAIL", e)
