# 东财补跑：只重试失败的(code, report)对，最多3次，超时60秒，断点续跑
import urllib.request, urllib.parse, json, time, pickle, os, glob
BASE="https://datacenter-web.eastmoney.com/api/data/v1/get"
OUT="/home/hatch/workspace/ashare/eastmoney_repair"; os.makedirs(OUT, exist_ok=True)
reports=["RPT_DMSK_FN_BALANCE","RPT_DMSK_FN_INCOME","RPT_DMSK_FN_CASHFLOW","RPT_LICO_FN_CPD"]
def fetch(rep, code6):
    rows=[]; page=1
    while True:
        p={"reportName":rep,"columns":"ALL","filter":f'(SECURITY_CODE="{code6}")',"pageNumber":page,"pageSize":200,"source":"WEB","client":"WEB"}
        url=BASE+"?"+urllib.parse.urlencode(p)
        req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/"})
        with urllib.request.urlopen(req, timeout=60) as r: j=json.loads(r.read().decode())
        res=j.get("result") or {}; rows.extend(res.get("data",[]))
        if page>=res.get("pages",1) or not res.get("data"): break
        page+=1
    return rows
# collect fails per original batch
fails={}
for f in sorted(glob.glob("/home/hatch/workspace/ashare/eastmoney_full/eastmoney_full_batch_*.pkl")):
    bi=f.split("_")[-1].split(".")[0]; out=pickle.load(open(f,"rb"))
    for code, rec in out.items():
        for r in reports:
            v=rec[r]
            if not (isinstance(v, list) and len(v)>0): fails.setdefault(bi, []).append((code, r))
print(f"repair batches {len(fails)} total pairs {sum(len(v) for v in fails.values())}", flush=True)
t0=time.time()
for bi, pairs in sorted(fails.items()):
    fn=os.path.join(OUT, f"eastmoney_repair_batch_{bi}.pkl")
    if os.path.exists(fn):
        try: pickle.load(open(fn,"rb")); print(f"skip repair batch {bi}", flush=True); continue
        except Exception: pass
    out={}
    for code, rep in pairs:
        ok=None
        for attempt in range(3):
            try: ok=fetch(rep, code[2:]); break
            except Exception as e:
                ok=f"FAIL {type(e).__name__} {str(e)[:200]}"; time.sleep(1.0)
        out.setdefault(code, {})[rep]=ok; time.sleep(0.5)
    with open(fn+".tmp","wb") as f: pickle.dump(out,f)
    os.rename(fn+".tmp", fn)
    succ=sum(1 for v in out.values() for x in v.values() if isinstance(x, list) and len(x)>0)
    print(f"repair batch {bi} done pairs={len(pairs)} success={succ} elapsed={time.time()-t0:.0f}s -> {fn}", flush=True)
print(f"REPAIR DONE elapsed={time.time()-t0:.0f}s", flush=True)
