# 隔离测试：build_gap_review + m2/m3打分（不写真实输出）
import sys, os, pickle, json
sys.path.insert(0, os.path.expanduser("~/workspace/ashare"))
import daily_pipeline as dp
import pandas as pd

# --- 测试1：build_gap_review 合成数据 ---
# 信号收盘100，x=0.10 -> limit=90；买入日：open=92, low=89 -> 成交，entry=min(90,92)=90
# 持有日1：open=95 (>=90*1.02=91.8) -> 跳空止盈，r=95/90-1=+5.56%
bars = {
    "buy": {"sh600000": {"open": 92, "high": 93, "low": 89, "close": 91, "adjclose": 91, "volume": 100}},
    "h1": {"sh600000": {"open": 95, "high": 96, "low": 94, "close": 95.5, "adjclose": 95.5, "volume": 100}},
    "h2": {"sh600000": {"open": 96, "high": 97, "low": 95, "close": 96, "adjclose": 96, "volume": 100}},
}
picks = [{"code": "sh600000", "name": "测试股", "signal_close": 100.0}]
r = dp.build_gap_review("2026-10-28", "buy", ["h1", "h2"], picks, bars, 0.10, None, 0.02, tag="m2")
it = r["items"][0]
assert it["status"] == "跳空止盈", it
assert abs(it["realized"] - 0.0556) < 1e-3, it
assert abs(it["buy_price"] - 90.0) < 1e-9, it
print("test1 跳空止盈 OK:", it["status"], f"{it['realized']:+.2%}")

# 测试2：未触发止盈，收盘卖出
bars2 = {
    "buy": {"sh600001": {"open": 88, "high": 89, "low": 87, "close": 88, "adjclose": 88, "volume": 100}},  # entry=88（跳空低开更优）
    "h1": {"sh600001": {"open": 88.5, "high": 89, "low": 87.5, "close": 88.2, "adjclose": 88.2, "volume": 100}},
    "h2": {"sh600001": {"open": 88.3, "high": 89.5, "low": 87, "close": 87.5, "adjclose": 87.5, "volume": 100}},
}
picks2 = [{"code": "sh600001", "name": "测试股2", "signal_close": 100.0}]
r2 = dp.build_gap_review("2026-10-28", "buy", ["h1", "h2"], picks2, bars2, 0.10, None, 0.02, tag="m2")
it2 = r2["items"][0]
assert it2["status"] == "收盘卖出", it2
assert abs(it2["buy_price"] - 88.0) < 1e-9, it2  # 跳空低开按开盘价
assert abs(it2["realized"] - (87.5/88-1)) < 1e-4, it2
print("test2 收盘卖出 OK:", it2["status"], f"{it2['realized']:+.2%}", "buy=", it2["buy_price"])

# 测试3：未成交
bars3 = {"buy": {"sh600002": {"open": 95, "high": 96, "low": 91, "close": 95, "adjclose": 95, "volume": 100}},
         "h1": {"sh600002": {"open": 95, "high": 96, "low": 94, "close": 95, "adjclose": 95, "volume": 100}},
         "h2": {"sh600002": {"open": 95, "high": 96, "low": 94, "close": 95, "adjclose": 95, "volume": 100}}}
picks3 = [{"code": "sh600002", "name": "测试股3", "signal_close": 100.0}]
r3 = dp.build_gap_review("2026-10-28", "buy", ["h1", "h2"], picks3, bars3, 0.10, None, 0.02, tag="m2")
assert r3["items"][0]["status"] == "未成交", r3["items"][0]
print("test3 未成交 OK")

# 测试4：原策略 gap-aware（含止损），跳空穿止损按开盘价
bars4 = {"buy": {"sh600003": {"open": 100, "high": 101, "low": 99, "close": 100, "adjclose": 100, "volume": 100}},
         "h1": {"sh600003": {"open": 96, "high": 97, "low": 95, "close": 96, "adjclose": 96, "volume": 100}}}
picks4 = [{"code": "sh600003", "name": "测试股4", "signal_close": 100.0}]
r4 = dp.build_gap_review("2026-10-28", "buy", ["h1"], picks4, bars4, 0.0, 0.02, 0.05, tag="orig")
it4 = r4["items"][0]
assert it4["status"] == "止损", it4
assert abs(it4["realized"] - (-0.04)) < 1e-9, it4  # 开盘96 <= 98，按开盘价 -4%
print("test4 跳空穿止损 OK:", it4["status"], f"{it4['realized']:+.2%}")

# --- 测试5：m2/m3打分（用2026-10-30真实特征） ---
BASE = os.path.expanduser("~/workspace/ashare")
feat = pickle.load(open(f"{BASE}/daily_features/2026-10-30.pkl", "rb"))
ev2 = pickle.load(open(f"{BASE}/ev_model_10_2.pkl", "rb"))
ev3 = pickle.load(open(f"{BASE}/ev_model_10_3.pkl", "rb"))
s2 = ev2.predict(feat[dp.FEATS].to_numpy(dtype="float32"))
s3 = ev3.predict(feat[dp.FEATS].to_numpy(dtype="float32"))
assert len(s2) == len(feat) and len(s3) == len(feat)
assert not (s2 != s2).any() and not (s3 != s3).any()
i2 = s2.argsort()[::-1][:2]; i3 = s3.argsort()[::-1][:2]
print("test5 打分 OK: m2 top2 =", list(feat.iloc[i2]["code"]), "ev=", [f"{v:+.2%}" for v in s2[i2]])
print("test5 打分 OK: m3 top2 =", list(feat.iloc[i3]["code"]), "ev=", [f"{v:+.2%}" for v in s3[i3]])
print("ALL TESTS PASSED")
