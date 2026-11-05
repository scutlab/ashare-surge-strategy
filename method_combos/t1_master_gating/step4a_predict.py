# T1 step4a: 分块预测并落盘（轻量，可断点续跑）
# 输出：t1_prob_{val,test}_{baseline,gated}.npy（float32）
import os, time, gc, pickle, numpy as np
BASE = os.path.expanduser("~/workspace/ashare")
OUT = os.path.join(BASE, "method_combos", "t1_master_gating")
CHK = os.path.join(OUT, "chunks")
t0 = time.time()
m_base = pickle.load(open(os.path.join(OUT, "t1_model_baseline_sub.pkl"), "rb"))
m_gate = pickle.load(open(os.path.join(OUT, "t1_model_gated_sub.pkl"), "rb"))
print("models loaded", flush=True)

def n_rows(s):
    return sum(np.load(os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy"), mmap_mode="r").shape[0]
               for fi in range(25)
               if os.path.exists(os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy")))

BS = 200000
for s in ["val", "test"]:
    n = n_rows(s)
    for arm, model, cols in (("baseline", m_base, slice(0, 29)), ("gated", m_gate, slice(0, 79))):
        outp = os.path.join(OUT, f"t1_prob_{s}_{arm}.npy")
        tmp = outp + ".tmp"
        if os.path.exists(outp):
            print(f"skip {outp} (exists)", flush=True); continue
        prob = np.empty(n, dtype=np.float32)
        o = 0
        for fi in range(25):
            p = os.path.join(CHK, f"t1_X_aug_{s}_{fi:02d}.npy")
            if not os.path.exists(p): continue
            Xc = np.load(p, mmap_mode="r")
            for i in range(0, len(Xc), BS):
                b = np.asarray(Xc[i:i + BS, cols])
                prob[o:o + len(b)] = model.predict_proba(b)[:, 1].astype(np.float32)
                o += len(b)
            del Xc; gc.collect()
            print(f"  {s}/{arm} fi={fi} {o}/{n}", flush=True)
        assert o == n
        np.save(tmp, prob); os.replace(tmp, outp)
        del prob; gc.collect()
        print(f"saved {outp}", flush=True)
print(f"PREDICT DONE {time.time()-t0:.0f}s", flush=True)
