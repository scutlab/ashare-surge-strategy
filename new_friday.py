# 周五 2026-10-23 候选 Top10：两个训练+验证重训模型分别打分，标明差异
import pandas as pd, numpy as np, json, os, pickle
BASE = os.path.expanduser("~/workspace/ashare")
import importlib.util
spec = importlib.util.spec_from_file_location("full_train", os.path.join(BASE, "full_train.py"))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
FEATS = mod.FEATS; FEATS_NO_WD = [f for f in FEATS if f != 'weekday']
df = pd.read_pickle(os.path.join(BASE, "friday_20261023_all_v3.pkl"))
print(f"friday rows {len(df)}", flush=True)

def top10(model_path, feats):
    with open(model_path, "rb") as f:
        mdl = pickle.load(f)
    p = mdl.predict_proba(df[feats].to_numpy(dtype=np.float32))[:, 1]
    d = df[['code', 'adjclose', 'is_limit_up']].copy()
    d['prob'] = p
    d = d.sort_values(['prob', 'code'], ascending=[False, True], kind='mergesort').head(10)
    return [{'rank': i + 1, 'code': r.code, 'prob': float(r.prob),
             'friday_close': float(r.adjclose), 'friday_limit_up': bool(r.is_limit_up)}
            for i, r in enumerate(d.itertuples())]

wd = top10(os.path.join(BASE, "new_model_HGB_wd_trainval.pkl"), FEATS)
nowd = top10(os.path.join(BASE, "new_model_HGB_nowd_trainval.pkl"), FEATS_NO_WD)
set_wd = {x['code'] for x in wd}; set_nowd = {x['code'] for x in nowd}
rank_wd = {x['code']: x['rank'] for x in wd}; rank_nowd = {x['code']: x['rank'] for x in nowd}
diff = {'overlap': len(set_wd & set_nowd),
        'only_wd': [c for c in sorted(set_wd - set_nowd)],
        'only_nowd': [c for c in sorted(set_nowd - set_wd)],
        'rank_shift_overlap': {c: {'wd_rank': rank_wd[c], 'nowd_rank': rank_nowd[c]}
                               for c in sorted(set_wd & set_nowd)}}
out = {'signal_date': '2026-10-23', 'models': 'train+val retrained',
       'top10_wd': wd, 'top10_nowd': nowd, 'difference': diff}
with open(os.path.join(BASE, "new_friday_top10.json"), "w") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False), flush=True)
print("FRIDAY_DONE", flush=True)
