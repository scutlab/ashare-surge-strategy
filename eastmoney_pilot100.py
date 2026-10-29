# 东财100只试点：资产负债+利润+现金流+主要指标，全部带NOTICE_DATE，按披露日对齐用
import urllib.request, urllib.parse, json, time, pickle, pandas as pd
BASE="https://datacenter-web.eastmoney.com/api/data/v1/get"
codes=[c.strip() for c in open("/home/hatch/workspace/ashare/eastmoney_pilot_codes.csv") if c.strip()]
# codes are like sh600000 -> Eastmoney SECURITY_CODE is 600000 (no prefix)
def fetch(rep, code6):
    rows=[]; page=1
    while True:
        p={"reportName":rep,"columns":"ALL","filter":f'(SECURITY_CODE="{code6}")',"pageNumber":page,"pageSize":200,"source":"WEB","client":"WEB"}
        url=BASE+"?"+urllib.parse.urlencode(p)
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/"})
        with urllib.request.urlopen(req, timeout=30) as r: j=json.loads(r.read().decode())
        res=j.get("result") or {}
        rows.extend(res.get("data",[]))
        if page>=res.get("pages",1) or not res.get("data"): break
        page+=1
    return rows
reports=["RPT_DMSK_FN_BALANCE","RPT_DMSK_FN_INCOME","RPT_DMSK_FN_CASHFLOW","RPT_LICO_FN_CPD"]
out={}; t0=time.time()
for i, full in enumerate(codes):
    code6=full[2:]; rec={}
    for rep in reports:
        try: rec[rep]=fetch(rep, code6)
        except Exception as e: rec[rep]=f"FAIL {type(e).__name__} {str(e)[:200]}"
        time.sleep(0.2)
    out[full]=rec
    if (i+1)%20==0: print(f"pilot {i+1}/100 elapsed {time.time()-t0:.0f}s", flush=True)
with open("/home/hatch/workspace/ashare/eastmoney_pilot.pkl","wb") as f: pickle.dump(out,f)
print(f"PILOT DONE stocks={len(out)} elapsed={time.time()-t0:.0f}s", flush=True)
for rep in reports:
    ns=[len(v[rep]) if isinstance(v[rep], list) else 0 for v in out.values()]
    print(rep, f"stocks_with_data={sum(1 for n in ns if n>0)}/100 mean_reports={sum(ns)/len(ns):.1f} min={min(ns)} max={max(ns)}", flush=True)
