import baostock as bs, time, pandas as pd
lg = bs.login()
print("login", lg.error_code, lg.error_msg)
t0=time.time()
rs = bs.query_stock_basic()
rows=[]
while rs.next():
    rows.append(rs.get_row_data())
print("basic rows", len(rows), "time", round(time.time()-t0,1))
df=pd.DataFrame(rows, columns=rs.fields)
print(df.head(10).to_string())
print(df['type'].value_counts().head() if 'type' in df.columns else df.columns.tolist())
df.to_csv("~/workspace/ashare/stock_basic_raw.csv", index=False)
# test 3 stocks history
for code in ["sh.600000","sz.000001","sh.600519"]:
    t0=time.time()
    rs = bs.query_history_k_data_plus(code, "code,date,open,high,low,close,preclose,volume,amount,turn,pctChg", start_date="1996-01-01", end_date="2026-10-23", frequency="d", adjustflag="2")
    n=0
    while rs.next():
        n+=1
    print(code, "rows", n, "sec", round(time.time()-t0,2), "err", rs.error_code)
bs.logout()
