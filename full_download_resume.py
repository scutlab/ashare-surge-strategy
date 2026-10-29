# 全量下载断点续传脚本（2026-10-25 运行时重启后使用）
# 逻辑：
# 1. 读取 full_raw_partial.pkl（若可读），统计已下载代码。
# 2. 目标 = 沪深全部代码 - 当前 ST 代码（2026-10-23 快照名称含 ST/*ST，共 194 只）。
# 3. 只并行下载缺失的非 ST 代码（8 线程，与原脚本相同 Yahoo 接口）。
# 4. 合并已下载 + 新下载，剔除 ST，保存为 full_raw.pkl；同时保存缺失/失败清单。
# 5. 若 partial 文件损坏无法读取，则回退到全量重新下载，并在日志中明确说明。
# 注意：不要直接用系统 python3 读 pickle（pandas 版本与 venv 不一致会报错），必须用 ~/workspace/ashare_venv/bin/python。
import pandas as pd, urllib.request, json, time, concurrent.futures, os
BASE = os.path.expanduser("~/workspace/ashare")
lst = pd.read_csv(os.path.join(BASE, "sina_hs_a_20261025.csv"))
lst = lst[lst["symbol"].str.startswith(("sh", "sz"))].sort_values("symbol")
st_codes = set(lst[lst["name"].str.contains("ST", na=False)]["symbol"])
target = [c for c in lst["symbol"] if c not in st_codes]
print(f"target non-ST {len(target)} ST excluded {len(st_codes)}", flush=True)

def yahoo_sym(s):
    return s[2:] + ".SS" if s.startswith("sh") else s[2:] + ".SZ"

def fetch(sym_code):
    ysym = yahoo_sym(sym_code)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ysym}?period1=820454400&period2=1792800000&interval=1d"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=25) as r:
                j = json.load(r)
            res = j["chart"]["result"][0]
            ts = res["timestamp"]; q = res["indicators"]["quote"][0]; adj = res["indicators"]["adjclose"][0]["adjclose"]
            df = pd.DataFrame({"code": sym_code, "date": pd.to_datetime(ts, unit="s").tz_localize("UTC").tz_convert("Asia/Shanghai").date, "open": q["open"], "high": q["high"], "low": q["low"], "close": q["close"], "adjclose": adj, "volume": q["volume"]})
            return df.dropna(subset=["adjclose"])
        except Exception:
            if attempt == 2:
                return None
            time.sleep(1 + attempt)
    return None

partial_path = os.path.join(BASE, "full_raw_partial.pkl")
have = set()
frames = []
try:
    old = pd.read_pickle(partial_path)
    old = old[~old["code"].isin(st_codes)]
    have = set(old["code"].unique())
    frames.append(old)
    print(f"partial ok rows {len(old)} codes {len(have)}", flush=True)
except Exception as e:
    print(f"partial load FAIL {type(e).__name__} {e} -> full redownload", flush=True)
    have = set()

missing = [c for c in target if c not in have]
print(f"missing non-ST {len(missing)}", flush=True)
t0 = time.time(); done = 0; failed = []
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(fetch, c): c for c in missing}
    for fut in concurrent.futures.as_completed(futs):
        done += 1; df = fut.result()
        if df is not None and len(df) > 200:
            frames.append(df)
        else:
            failed.append(futs[fut])
        if done % 500 == 0:
            print(f"{done}/{len(missing)} ok={len(frames)-1} failed={len(failed)} elapsed={time.time()-t0:.0f}s", flush=True)
out = pd.concat(frames, ignore_index=True)
out.to_pickle(os.path.join(BASE, "full_raw.pkl"))
print(f"FINAL rows {len(out)} codes {out['code'].nunique()} date {out['date'].min()} to {out['date'].max()} failed {failed}", flush=True)
with open(os.path.join(BASE, "full_download_failed.json"), "w") as f:
    json.dump(failed, f)
