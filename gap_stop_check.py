# 跳空跌破止损的影响定量：悲观下界（止损单全部按当日最低价成交，这是理论最差情形）
# 真实情形一定介于“模拟口径（恰好-1.5%）”与“悲观下界”之间
import numpy as np, os, json, pickle, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; S = 0.015
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
pol = json.load(open(f"{BASE}/adaptive_policy.json"))
ki = [FEATS.index(f) for f in pol["key_features"]]
a_raw = np.array(pol["a_raw"]); x0_raw = pol["x0_raw"]
b_raw = np.array(pol["b_raw"]); t0_raw = pol["t0_raw"]
clf = pickle.load(open(f"{BASE}/exec_model_10_30_0.015.pkl", "rb"))
X = np.load(f"{IN}/te_X.npy").astype(np.float32)
pr = np.load(f"{PP}/te_prices.npy"); day = np.load(f"{PP}/te_day.npy")
prob = clf.predict_proba(X)[:, 1].astype(np.float32)
order = np.lexsort((-prob, day)); dk = day[order]
upos = np.sort(np.unique(dk, return_index=True)[1])
picks = []
for i, s in enumerate(upos):
    e = upos[i + 1] if i + 1 < len(upos) else len(order)
    picks.extend(order[s:s + 2])
picks = np.array(picks)
W = X[picks][:, ki].astype(np.float64)
sig, bl, sl, sh, sc = (pr[picks][:, j].astype(np.float64) for j in range(5))
del X, pr, day, prob; gc.collect()

def run(name, x, t):
    entry = sig * (1 - x); filled = bl <= entry
    slh = sl <= entry * (1 - S); tph = (~slh) & (sh >= entry * (1 + t))
    real_sim = np.where(slh, -S, np.where(tph, t, sc / entry - 1))
    real_pes = np.where(slh, sl / entry - 1, np.where(tph, t, sc / entry - 1))
    out = {}
    for tag, real in (("sim", real_sim), ("pessimistic", real_pes)):
        r = real[filled]; n = len(r); mean = r.mean(); se = r.std(ddof=1) / np.sqrt(n)
        out[tag] = {"n": int(n), "mean": float(mean), "lower70": float(mean - 0.524 * se)}
    # 止损单中，最低价相对止损价的深度分布
    f = filled & slh
    depth = sl[f] / (entry[f] * (1 - S)) - 1  # 负数表示最低价比止损价还低多少
    out["stop_depth_quantiles"] = [float(q) for q in np.quantile(depth, [0, .1, .25, .5, .75, .9, 1])]
    out["stop_below_3pct"] = float((depth < -0.03).mean())
    out["stop_below_5pct"] = float((depth < -0.05).mean())
    out["name"] = name
    return out

xA = np.clip(x0_raw + W @ a_raw, 0.0, 0.12); tA = np.clip(t0_raw + W @ b_raw, 0.05, 0.40)
for name, x, t in [("adaptive", xA, tA), ("const_x0.10_t0.30", np.full(len(sig), .10), np.full(len(sig), .30))]:
    r = run(name, x, t)
    print(json.dumps(r, ensure_ascii=False), flush=True)
print("DONE", flush=True)
