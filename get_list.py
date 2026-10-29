import urllib.request, json, time
rows=[]
for page in range(1,80):
    url=f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page={page}&num=100&sort=code&asc=1&node=hs_a&symbol=&_s_r_a=page"
    req=urllib.request.Request(url, headers={"User-Agent":"Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data=json.loads(r.read().decode('utf-8', errors='ignore'))
    except Exception as e:
        print("page",page,"ERR",e); break
    if not data: print("page",page,"empty"); break
    rows.extend(data)
    print(page, len(data), data[0]['symbol'], data[-1]['symbol'], "total", len(rows))
    if len(data)<100: break
import pandas as pd
df=pd.DataFrame(rows)
print(df.shape)
print(df.head().to_string())
df.to_csv("~/workspace/ashare/sina_hs_a_20261025.csv", index=False)
# board counts
def board(s):
    c=s[2:]
    if c.startswith('688'): return '科创板'
    if c.startswith('300'): return '创业板'
    if c.startswith('60'): return '沪主板/科创外'
    if c.startswith('000') or c.startswith('001'): return '深主板'
    if c.startswith('002') or c.startswith('003'): return '中小板/深主板'
    return '其他'
df['board']=df['symbol'].apply(board)
print(df['board'].value_counts())
print("ST count", df['name'].str.contains('ST', na=False).sum())
