import urllib.request, urllib.parse, json, time
BASE="https://datacenter-web.eastmoney.com/api/data/v1/get"
def fetch(report, cols, filt, pagesize=100):
    p={"reportName":report,"columns":",".join(cols),"filter":filt,"pageNumber":1,"pageSize":pagesize,"sortColumns":"NOTICE_DATE","sortTypes":-1,"source":"WEB","client":"WEB"}
    url=BASE+"?"+urllib.parse.urlencode(p)
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0","Referer":"https://data.eastmoney.com/"})
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read().decode())
tests=[
 ("RPT_F10_FN_MAIN", ["SECURITY_CODE","SECURITY_NAME_ABBR","REPORT_DATE","REPORT_TYPE","NOTICE_DATE","TOTAL_ASSETS","NET_ASSETS","NET_PROFIT","OPERATE_CASH_FLOW"]),
 ("RPT_LICO_FN_CPD", ["SECURITY_CODE","SECURITY_NAME_ABBR","REPORT_DATE","REPORT_TYPE","NOTICE_DATE"]),
]
for rep, cols in tests:
    try:
        j=fetch(rep, cols, '(SECURITY_CODE="600000")')
        res=j.get("result",{}); print(rep, "success", res.get("success"), "total", res.get("total"), "data_rows", len(res.get("data",[])), flush=True)
        if res.get("data"): print(" sample:", res["data"][0], flush=True)
    except Exception as e:
        print(rep, "FAIL", type(e).__name__, str(e)[:500], flush=True)
