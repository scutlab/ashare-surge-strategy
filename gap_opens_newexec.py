# 新执行 gap-aware 终评：m天持有、+2%止盈、无止损；开盘价跳空如实处理
# 买入：limit=sig*(1-x)，buy_low<=limit成交，entry=min(limit,buy_open)
# 持有日：open>=entry*1.02 -> 按开盘价止盈；elif high>=entry*1.02 -> +2%；m天未触 -> 最后收盘卖出
import numpy as np, pandas as pd, os, json, pickle, glob, gc, time
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; t0 = time.time()
sel = json.load(open(f"{BASE}/ev_select.json")); tag = sel["best"]
x = sel["results"][tag]["x"]; mm = sel["results"][tag]["m"]
print(f"BEST (x,m)=({x},{mm})", flush=True)
reg = pickle.load(open(f"{BASE}/ev_model_{tag}.pkl", "rb"))
clf = pickle.load(open(f"{BASE}/p_model_{tag}.pkl", "rb"))
Xte = np.load(f"{IN}/te_X.npy").astype(np.float32)
dte = np.load(f"{PP}/te_day.npy"); cte = np.load(f"{PP}/te_code.npy")
pr = np.load(f"{PP}/te_prices.npy"); ex = np.load(f"{PP}/te_ext.npy")
codes = json.load(open(f"{PP}/codes.json"))["codes"]; cid = {c: i for i, c in enumerate(codes)}
ev_pred = reg.predict(Xte).astype(np.float32); p_pred = clf.predict_proba(Xte)[:, 1].astype(np.float32)
del Xte; gc.collect()
def top2(score):
    order = np.lexsort((-score, dte)); dk = dte[order]
    upos = np.sort(np.unique(dk, return_index=True)[1])
    picks = []
    for i, s in enumerate(upos):
        e = upos[i + 1] if i + 1 < len(upos) else len(order)
        picks.extend(order[s:s + 2])
    return np.array(picks)
picks_ev = top2(ev_pred); picks_p = top2(p_pred)
print(f"picks ev={len(picks_ev)} p={len(picks_p)}", flush=True)
# ---- opens for test picks: buy_open, o2..o4 ----
for name, picks in (("ev", picks_ev), ("p", picks_p)):
    need = pd.DataFrame({"ci": cte[picks].astype(np.int32), "dk": dte[picks].astype(np.int32),
                         "pi": np.arange(len(picks))})
    bo = np.full(len(picks), np.nan); o2 = np.full(len(picks), np.nan)
    o3 = np.full(len(picks), np.nan); o4 = np.full(len(picks), np.nan)
    for vf in sorted(glob.glob(BASE + "/full_features_batch_v3_*.pkl")):
        v = pd.read_pickle(vf)[["code", "date", "adj_open"]]
        v["ci"] = v["code"].map(cid).astype(np.int32)
        v["dk"] = v["date"].values.astype("datetime64[D]").astype(np.int64).astype(np.int32)
        v = v.sort_values(["ci", "dk"]).reset_index(drop=True)
        g = v.groupby("ci")["adj_open"]
        v["bo"] = g.shift(-1); v["o2"] = g.shift(-2); v["o3"] = g.shift(-3); v["o4"] = g.shift(-4)
        mg = need.merge(v[["ci", "dk", "bo", "o2", "o3", "o4"]], on=["ci", "dk"], how="inner")
        pi = mg["pi"].values
        bo[pi] = mg["bo"].values; o2[pi] = mg["o2"].values; o3[pi] = mg["o3"].values; o4[pi] = mg["o4"].values
        del v, mg; gc.collect()
    np.savez_compressed(f"{BASE}/newexec_opens_{name}.npz", bo=bo, o2=o2, o3=o3, o4=o4,
                        picks=picks)
    print(f"{name}: opens miss={int(np.isnan(bo).sum())}/{len(picks)}", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
