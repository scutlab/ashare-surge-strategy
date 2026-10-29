# 东财全量：5029只非ST，分批200只/文件，断点续跑（已存在且可加载的批次直接跳过）
import urllib.request, urllib.parse, json, time, pickle, os, pandas as pd
BASE="https://datacenter-web.eastmoney.com/api/data/v1/get"
OUT="/home/hatch/workspace/ashare/eastmoney_full"
os.makedirs(OUT, exist_ok=True)
s=pd.read_csv("/home/hatch/workspace/ashare/sina_hs_a_20261025.csv", dtype=str)
s=s[~s['name'].str.contains('ST', na=False)]; s=s[s['symbol'].str.startswith(('sh','sz'))]
codes=sorted(s['symbol'].tolist()); print(f"full codes {len(codes)}", flush=True)
def fetch(rep, code6):
    rows=[]; page=1
    while True:
        p={"reportName":rep,"columns":"ALL","filter":f'(SECURITY_CODE="{code6}")',"pageNumber":page,"pageSize":200,"source":"WEB","client":"WEB"}
        url=BASE+"?"+urllib.parse.urlencode(p)
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/"})
        with urllib.request.urlopen(req, timeout=30) as r: j=json.loads(r.read().decode())
        res=j.get("result") or {}; rows.extend(res.get("data",[]))
        if page>=res.get("pages",1) or not res.get("data"): break
        page+=1
    return rows
reports=["RPT_DMSK_FN_BALANCE","RPT_DMSK_FN_INCOME","RPT_DMSK_FN_CASHFLOW","RPT_LICO_FN_CPD"]
t0=time.time()
for bi in range(0, len(codes), 50):
    batch=codes[bi:bi+50]; fn=os.path.join(OUT, f"eastmoney_full_batch_{bi//50:02d}.pkl")
    if os.path.exists(fn):
        try: pickle.load(open(fn,"rb")); print(f"skip batch {bi//50} exists", flush=True); continue
        except Exception: print(f"redo batch {bi//50} corrupt", flush=True)
    out={}
    for full in batch:
        rec={}
        for rep in reports:
            try: rec[rep]=fetch(rep, full[2:])
            except Exception as e: rec[rep]=f"FAIL {type(e).__name__} {str(e)[:200]}"
            time.sleep(0.2)
        out[full]=rec
    with open(fn+".tmp","wb") as f: pickle.dump(out,f)
    os.rename(fn+".tmp", fn)
    print(f"batch {bi//50} done {len(out)} stocks elapsed {time.time()-t0:.0f}s -> {fn}", flush=True)
print(f"FULL DONE elapsed {time.time()-t0:.0f}s", flush=True)
