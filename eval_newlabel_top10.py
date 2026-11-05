# 用已训好的 newlabel_model_HGB.pkl 做 Top10 模拟（修正版2：lexsort保证同日连续）
import numpy as np, os, json, time, pickle, pandas as pd
from sklearn.metrics import roc_auc_score
BASE=os.path.expanduser("~/workspace/ashare"); IN=BASE+"/newlabel_arrays"; t0=time.time()
clf=pickle.load(open(BASE+"/newlabel_model_HGB.pkl","rb"))
Xte=np.load(f"{IN}/te_X.npy"); yte=np.load(f"{IN}/te_y.npy"); ex=np.load(f"{IN}/te_exec.npy")
prob=clf.predict_proba(Xte)[:,1]; del Xte
dk=ex[:,0].astype(np.int64); sig=ex[:,1].astype(np.float64); bl=ex[:,2]; sh=ex[:,3]; sc=ex[:,4]
entry=sig*0.98; filled=bl<=entry; tp=sh>=entry*1.02
real=np.where(tp,0.02,sc/entry-1)
# 方法1：lexsort (dk asc, prob desc) -> 每日连续且按概率降序
order=np.lexsort((-prob,dk))
udk,idx=np.unique(dk[order],return_index=True); idx=np.sort(idx)
cyc=[]; pf_=0; ptp=0; pprof=0; npick=0
for s,e in zip(idx,list(idx[1:])+[len(order)]):
    top=order[s:e][:10]; f=filled[top]; npick+=len(top); k=int(f.sum()); pf_+=k
    if k==0: continue
    r=real[top][f]; cyc.append(r.mean()); ptp+=int(tp[top][f].sum()); pprof+=int((r>0).sum())
cyc=np.array(cyc)
# 方法2（独立）：pandas groupby
df=pd.DataFrame({"dk":dk,"p":prob,"f":filled,"r":real})
df["rk"]=df.groupby("dk")["p"].rank(method="first",ascending=False)
g=df[df["rk"]<=10].groupby("dk")
cyc2=g.apply(lambda x:x.loc[x["f"],"r"].mean(),include_groups=False).dropna().values
assert abs(cyc2.mean()-cyc.mean())<1e-12 and len(cyc2)==len(cyc), f"xcheck {cyc2.mean()} vs {cyc.mean()}"
se=cyc.std(ddof=1)/np.sqrt(len(cyc))
top10={"n_cycles":int(len(cyc)),"mean_cycle":float(cyc.mean()),"se":float(se),
    "lower70_onesided":float(cyc.mean()-0.524*se),"p_cycle_pos":float((cyc>0).mean()),
    "fill_rate":pf_/npick,"p_tp_given_fill":ptp/pf_,"p_prof_given_fill":pprof/pf_,
    "n_picks":npick,"n_filled":pf_,"xcheck":"ok"}
res={"n_test":int(len(yte)),"base_rate":float(yte.mean()),"auc":float(roc_auc_score(yte,prob)),
    "mean_prob":float(prob.mean()),"top10":top10}
json.dump(res,open(BASE+"/newlabel_result.json","w"),indent=2)
print(json.dumps(top10,indent=2)); print(f"DONE {time.time()-t0:.0f}s",flush=True)
