# 测试集最终评估：n=2，自适应 x(w)/t(w) vs 常数基线
# 含独立交叉验证：前 3000 个 pick 用纯 Python 循环重算，与向量化结果逐笔比对
import numpy as np, os, json, time, pickle, gc
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; S = 0.015; t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
pol = json.load(open(f"{BASE}/adaptive_policy.json"))
keyf = pol["key_features"]; ki = [FEATS.index(f) for f in keyf]
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
print(f"test picks={len(picks)}", flush=True)

def apply_policy(kind, cx=None, ct=None):
    if kind == "adaptive":
        x = np.clip(x0_raw + W @ a_raw, 0.0, 0.12); t = np.clip(t0_raw + W @ b_raw, 0.05, 0.40)
    else:
        x = np.full(len(sig), cx); t = np.full(len(sig), ct)
    entry = sig * (1 - x)
    filled = bl <= entry
    slh = sl <= entry * (1 - S); tph = (~slh) & (sh >= entry * (1 + t))
    real = np.where(slh, -S, np.where(tph, t, sc / entry - 1))
    return x, t, filled, real, slh, tph

def metrics(name, x, t, filled, real, slh, tph):
    r = real[filled]; n = len(r)
    out = {"name": name, "n_picks": int(len(real)), "n_filled": int(n),
           "fill_rate": float(n / len(real))}
    if n == 0: return out
    mean = float(r.mean()); sd = float(r.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n)
    out.update({"mean_per_trade": mean, "std": sd, "se": float(se),
        "lower70_onesided": mean - 0.524 * se, "lower70_twosided": mean - 1.036 * se,
        "p_profit": float((r > 0).mean()),
        "stop_share": float(slh[filled].mean()), "take_share": float(tph[filled].mean()),
        "x_mean": float(x[filled].mean()), "x_std": float(x[filled].std()),
        "t_mean": float(t[filled].mean()), "t_std": float(t[filled].std())})
    return out

results = []
xA, tA, fA, rA, sA, pA = apply_policy("adaptive")
results.append(metrics("adaptive_x(w)_t(w)", xA, tA, fA, rA, sA, pA))
for cx, ct in [(0.10, 0.30), (0.08, 0.30), (0.06, 0.30), (0.06, 0.20), (0.10, 0.10), (0.04, 0.30)]:
    x, t, f, r, s_, p_ = apply_policy("const", cx, ct)
    results.append(metrics(f"const_x{cx:.2f}_t{ct:.2f}", x, t, f, r, s_, p_))

# 独立交叉验证：前 3000 pick 纯循环重算
def loop_sim(x, t, lim=3000):
    rr = []
    for i in range(min(lim, len(sig))):
        e = sig[i] * (1 - x[i])
        if bl[i] > e: continue
        if sl[i] <= e * (1 - S): rr.append(-S); continue
        if sh[i] >= e * (1 + t[i]): rr.append(t[i]); continue
        rr.append(sc[i] / e - 1)
    return np.array(rr)
rl = loop_sim(xA, tA)
mask_first = np.zeros(len(sig), bool); mask_first[:3000] = True
rv2 = rA[mask_first & fA]
diff = float(np.max(np.abs(rl - rv2))) if len(rl) == len(rv2) and len(rl) > 0 else -1
print(f"loop-check n={len(rl)} maxdiff={diff:.2e}", flush=True)
assert diff >= 0 and diff < 1e-9, "independent recomputation mismatch"
for r_ in results:
    print(json.dumps(r_, ensure_ascii=False), flush=True)
json.dump({"results": results, "policy_adopt_on_val": pol["adopt"],
           "loop_check_maxdiff": diff, "elapsed": time.time() - t0},
          open(f"{BASE}/eval_adaptive_test.json", "w"), indent=2, ensure_ascii=False)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
