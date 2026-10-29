# 抓2026-10-26周一行情（Yahoo日线），供周二买入信号打分
import pickle, os, json, time, datetime, urllib.request
from concurrent.futures import ThreadPoolExecutor
BASE="/home/hatch/workspace/ashare"
codes=set()
for f in sorted(__import__("glob").glob(os.path.join(BASE,"full_flow_new_*.pkl"))):
    d=pickle.load(open(f,"rb")); codes|=set(d["code"].unique()); del d
codes=sorted(codes); print(f"codes {len(codes)}", flush=True)
P1, P2 = 1792713600, 1793145600  # 2026-10-23 to 2026-10-28 UTC
def fetch(code):
    sym=code[2:]+(".SS" if code.startswith("sh") else ".SZ")
    url=f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={P1}&period2={P2}&interval=1d"
    for attempt in range(2):
        try:
            req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
            d=json.load(urllib.request.urlopen(req, timeout=25))
            r=d["chart"]["result"][0]; ts=r.get("timestamp",[]); q=r["indicators"]["quote"][0]
            adj=r["indicators"].get("adjclose",[{}])[0].get("adjclose",[])
            for t,o,h,l,c,v,a in zip(ts,q["open"],q["high"],q["low"],q["close"],q["volume"],adj):
                dt=datetime.datetime.fromtimestamp(t, tz=datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y-%m-%d")
                if dt=="2026-10-26" and c is not None:
                    return (code, {"open":o,"high":h,"low":l,"close":c,"adjclose":a,"volume":v})
            return (code, None)
        except Exception as e:
            if attempt==1: return (code, f"ERR {type(e).__name__}")
            time.sleep(2)
out={}
with ThreadPoolExecutor(max_workers=20) as ex:
    for i,(code,res) in enumerate(ex.map(fetch, codes)):
        out[code]=res
        if (i+1)%500==0: print(f"fetched {i+1} ok={sum(1 for v in out.values() if isinstance(v,dict))}", flush=True)
ok={k:v for k,v in out.items() if isinstance(v,dict)}; bad={k:v for k,v in out.items() if not isinstance(v,dict)}
print(f"monday ok {len(ok)} missing {len(bad)}", flush=True)
with open(os.path.join(BASE,"monday_20261026.pkl"),"wb") as f: pickle.dump(out,f)
print("FETCH MONDAY DONE", flush=True)
