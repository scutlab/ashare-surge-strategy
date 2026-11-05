# 自适应策略拟合：x(w)=clip(x0+a.z), t(w)=clip(t0+b.z)，z 为关键特征标准化值
# n=2：每天取排序模型 Top2，直接优化“平均每笔成交收益”（train 随机搜索+坐标上升，val 选型）
# 用法：python fit_adaptive_policy.py [K=5]
import numpy as np, os, json, time, pickle, gc, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BASE = os.path.expanduser("~/workspace/ashare"); IN = BASE + "/exec_arrays_10_30_0.015"
PP = BASE + "/price_panel"; S = 0.015; t0 = time.time()
import importlib.util
spec = importlib.util.spec_from_file_location("ft", BASE + "/full_train.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != "weekday"]
oreg = json.load(open(f"{BASE}/oracle_regress.json"))
keyf = [f for f, _ in oreg["importance"][:K]]
ki = [FEATS.index(f) for f in keyf]
print("key features:", keyf, flush=True)
clf = pickle.load(open(f"{BASE}/exec_model_10_30_0.015.pkl", "rb"))

def load_split(split):
    X = np.load(f"{IN}/{split}_X.npy").astype(np.float32)
    pr = np.load(f"{PP}/{split}_prices.npy"); day = np.load(f"{PP}/{split}_day.npy")
    prob = clf.predict_proba(X)[:, 1].astype(np.float32)
    order = np.lexsort((-prob, day))
    dk = day[order]; upos = np.unique(dk, return_index=True)[1]; upos = np.sort(upos)
    picks = []
    for s_i, e_i in zip(upos, list(upos[1:]) + [len(order)]):
        picks.extend(order[s_i:s_i + 2])
    picks = np.array(picks)
    Z = X[picks][:, ki]
    P = pr[picks]  # sig,bl,sl,sh,sc
    D = {"Z": Z, "sig": P[:, 0].astype(np.float64), "bl": P[:, 1].astype(np.float64),
         "sl": P[:, 2].astype(np.float64), "sh": P[:, 3].astype(np.float64),
         "sc": P[:, 4].astype(np.float64)}
    del X, pr, day, prob; gc.collect()
    return D

Dtr = load_split("tr"); Dva = load_split("va")
mu = Dtr["Z"].mean(axis=0); sd = Dtr["Z"].std(axis=0) + 1e-12
Ztr = (Dtr["Z"] - mu) / sd; Zva = (Dva["Z"] - mu) / sd
print(f"picks tr={len(Ztr)} va={len(Zva)}", flush=True)

def sim_mean(D, Z, x0, a, t0_, b):
    x = np.clip(x0 + Z @ a, 0.0, 0.12); t = np.clip(t0_ + Z @ b, 0.05, 0.40)
    entry = D["sig"] * (1 - x)
    filled = D["bl"] <= entry
    if filled.sum() == 0: return -np.inf, 0
    slh = D["sl"] <= entry * (1 - S); tph = (~slh) & (D["sh"] >= entry * (1 + t))
    real = np.where(slh, -S, np.where(tph, t, D["sc"] / entry - 1))
    r = real[filled]
    return float(r.mean()), int(len(r))

def sim_const(D, x, t):
    return sim_mean(D, np.zeros((len(D["sig"]), K)), x, np.zeros(K), t, np.zeros(K))

# 常数基线（train/val）
for cx, ct in [(0.10, 0.30), (0.08, 0.30), (0.06, 0.20), (0.10, 0.10), (0.06, 0.30)]:
    mtr, ntr = sim_const(Dtr, cx, ct); mva, nva = sim_const(Dva, cx, ct)
    print(f"const x={cx:.2f} t={ct:.2f}: train {mtr:+.4f} (n={ntr})  val {mva:+.4f} (n={nva})", flush=True)

rng = np.random.default_rng(1)
cands = []
for _ in range(4000):
    x0 = rng.uniform(0, 0.12); a = rng.uniform(-0.06, 0.06, K)
    t0_ = rng.uniform(0.05, 0.40); b = rng.uniform(-0.15, 0.15, K)
    mv, _ = sim_mean(Dtr, Ztr, x0, a, t0_, b)
    cands.append((mv, x0, a, t0_, b))
cands.sort(key=lambda c: -c[0])
print("train random search top5:", [round(c[0], 4) for c in cands[:5]], flush=True)

def refine(mv, x0, a, t0_, b):
    # 坐标上升
    params = [x0] + list(a) + [t0_] + list(b)
    best = mv
    for sweep in range(6):
        improved = False
        for j in range(len(params)):
            pj = params[j]
            step = 0.01 if j == 0 else (0.005 if j <= K else (0.02 if j == K + 1 else 0.01))
            for dj in (step, -step, step * 0.25, -step * 0.25):
                q = params.copy(); q[j] = pj + dj
                v, _ = sim_mean(Dtr, Ztr, q[0], np.array(q[1:K + 1]), q[K + 1], np.array(q[K + 2:]))
                if v > best + 1e-6:
                    best, params = v, q; improved = True; break
            else:
                continue
            break
        if not improved: break
    return best, params[0], np.array(params[1:K + 1]), params[K + 1], np.array(params[K + 2:])

refined = []
for mv, x0, a, t0_, b in cands[:10]:
    bv, bx0, ba, bt0, bb = refine(mv, x0, a, t0_, b)
    mv_va, nva = sim_mean(Dva, Zva, bx0, ba, bt0, bb)
    mv_tr, ntr = sim_mean(Dtr, Ztr, bx0, ba, bt0, bb)
    refined.append((mv_va, mv_tr, ntr, nva, bx0, ba, bt0, bb))
    print(f"refined: train {mv_tr:+.4f} (n={ntr}) val {mv_va:+.4f} (n={nva})", flush=True)
refined.sort(key=lambda c: -c[0])
bv_va, bv_tr, ntr, nva, bx0, ba, bt0, bb = refined[0]
# 与常数基线比较（val）
base_va = max(sim_const(Dva, cx, ct)[0] for cx, ct in
              [(0.10, 0.30), (0.08, 0.30), (0.06, 0.20), (0.10, 0.10), (0.06, 0.30), (0.04, 0.30)])
print(f"best adaptive val {bv_va:+.4f} vs best const val {base_va:+.4f}", flush=True)
# 化为非标准化系数，便于解读
a_raw = ba / sd; x0_raw = bx0 - float(ba @ (mu / sd))
b_raw = bb / sd; t0_raw = bt0 - float(bb @ (mu / sd))
out = {"K": K, "key_features": keyf, "x0": float(bx0), "a_std": ba.tolist(),
       "t0": float(bt0), "b_std": bb.tolist(), "x0_raw": float(x0_raw),
       "a_raw": a_raw.tolist(), "t0_raw": float(t0_raw), "b_raw": b_raw.tolist(),
       "feat_mean": mu.tolist(), "feat_std": sd.tolist(),
       "train_mean": float(bv_tr), "train_n": int(ntr), "val_mean": float(bv_va),
       "val_n": int(nva), "best_const_val": float(base_va),
       "adopt": bool(bv_va > base_va + 0.001)}
json.dump(out, open(f"{BASE}/adaptive_policy.json", "w"), indent=2)
print("ADOPT" if out["adopt"] else "FALLBACK_TO_CONST", flush=True)
print(f"DONE {time.time()-t0:.0f}s", flush=True)
