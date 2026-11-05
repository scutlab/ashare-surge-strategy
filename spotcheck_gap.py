# 抽查3个跳空止损实例
import numpy as np, pandas as pd, json, pickle, glob
BASE = '/home/hatch/workspace/ashare'; IN = BASE + '/exec_arrays_10_30_0.015'
PP = BASE + '/price_panel'; S = 0.015
import importlib.util
spec = importlib.util.spec_from_file_location('ft', BASE + '/full_train.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
FEATS = [f for f in m.FEATS if f != 'weekday']
pol = json.load(open(BASE + '/adaptive_policy.json')); ki = [FEATS.index(f) for f in pol['key_features']]
a_raw = np.array(pol['a_raw']); x0_raw = pol['x0_raw']
codes = json.load(open(PP + '/codes.json'))['codes']
clf = pickle.load(open(BASE + '/exec_model_10_30_0.015.pkl', 'rb'))
X = np.load(IN + '/te_X.npy').astype(np.float32); pr = np.load(PP + '/te_prices.npy')
day = np.load(PP + '/te_day.npy'); cc = np.load(PP + '/te_code.npy')
prob = clf.predict_proba(X)[:, 1].astype(np.float32)
order = np.lexsort((-prob, day)); dk = day[order]; upos = np.sort(np.unique(dk, return_index=True)[1])
picks = []
for i, s in enumerate(upos):
    e = upos[i + 1] if i + 1 < len(upos) else len(order); picks.extend(order[s:s + 2])
picks = np.array(picks)
W = X[picks][:, ki].astype(np.float64); P = pr[picks]
sig, bl, sl, sh, sc = (P[:, j].astype(np.float64) for j in range(5)); pc = cc[picks]; pd_ = day[picks]
need = pd.DataFrame({'ci': pc.astype(np.int32), 'dk': pd_.astype(np.int32), 'pi': np.arange(len(picks))})
buyo = np.full(len(picks), np.nan); sello = np.full(len(picks), np.nan)
cid = {c: i for i, c in enumerate(codes)}
for vf in sorted(glob.glob(BASE + '/full_features_batch_v3_*.pkl')):
    v = pd.read_pickle(vf)[['code', 'date', 'adj_open']]
    v['ci'] = v['code'].map(cid).astype(np.int32)
    v['dk'] = v['date'].values.astype('datetime64[D]').astype(np.int64).astype(np.int32)
    v = v.sort_values(['ci', 'dk']).reset_index(drop=True)
    g = v.groupby('ci')['adj_open']; v['buy_open'] = g.shift(-1); v['sell_open'] = g.shift(-2)
    mg = need.merge(v[['ci', 'dk', 'buy_open', 'sell_open']], on=['ci', 'dk'], how='inner')
    buyo[mg['pi'].values] = mg['buy_open'].values; sello[mg['pi'].values] = mg['sell_open'].values
x = np.clip(x0_raw + W @ a_raw, 0, 0.12)
limit = sig * (1 - x); filled = bl <= limit
entry = np.minimum(limit, buyo); stop_px = entry * (1 - S)
gap_stop = filled & (sello <= stop_px)
for i in np.where(gap_stop)[0][:3]:
    dstr = str(np.array([pd_[i]], dtype='datetime64[D]')[0])
    print(f"code={codes[pc[i]]} signal_date={dstr}")
    print(f"  sig={sig[i]:.2f} x={x[i]:.3f} limit={limit[i]:.2f} | buy: open={buyo[i]:.2f} low={bl[i]:.2f} -> entry={entry[i]:.2f}")
    print(f"  stop_px={stop_px[i]:.2f} | sell: open={sello[i]:.2f} low={sl[i]:.2f} close={sc[i]:.2f}")
    print(f"  -> gap-stop exit {(sello[i]/entry[i]-1)*100:+.2f}%  (原模拟记 -1.50%)")
