# 参数化：训练(x,t,s)标签模型 + Top1/Top3模拟
import numpy as np, os, json, time, pickle, pandas as pd, sys
X_=float(sys.argv[1]); T_=float(sys.argv[2]); S_=float(sys.argv[3])
tag=f"{int(X_*100)}_{int(T_*100)}_{S_}"
BASE=os.path.expanduser("~/workspace/ashare"); IN=BASE+f"/exec_arrays_{tag}"; t0=time.time()
Xtr=np.load(f"{IN}/tr_X.npy"); ytr=np.load(f"{IN}/tr_y.npy"); Xva=np.load(f"{IN}/va_X.npy"); yva=np.load(f"{IN}/va_y.npy")
X=np.concatenate([Xtr,Xva]); y=np.concatenate([ytr,yva]); del Xtr,ytr,Xva,yva
print(f"[{tag}] train+val {X.shape} pos {y.mean():.5f}",flush=True)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_depth=3,
    min_samples_leaf=100,class_weight="balanced",early_stopping=False,max_bins=128)
clf.fit(X,y); del X,y
pickle.dump(clf,open(f"{BASE}/exec_model_{tag}.pkl","wb")); print(f"[{tag}] fit {time.time()-t0:.0f}s",flush=True)
Xte=np.load(f"{IN}/te_X.npy"); yte=np.load(f"{IN}/te_y.npy"); ex=np.load(f"{IN}/te_exec.npy")
prob=clf.predict_proba(Xte)[:,1]; del Xte
dk=ex[:,0].astype(np.int64); sig=ex[:,1].astype(np.float64); bl=ex[:,2]; sl=ex[:,3]; sh=ex[:,4]; sc=ex[:,5]
entry=sig*(1-X_); filled=bl<=entry
slh=sl<=entry*(1-S_); tph=(~slh)&(sh>=entry*(1+T_))
real=np.where(slh,-S_,np.where(tph,T_,sc/entry-1))
res={"tag":tag,"x":X_,"t":T_,"s":S_,"n_test":int(len(yte)),"base_rate":float(yte.mean()),"auc":float(roc_auc_score(yte,prob))}
print(f"[{tag}] base {res['base_rate']:.5f} auc {res['auc']:.4f}",flush=True)
order=np.lexsort((-prob,dk)); udk,idx=np.unique(dk[order],return_index=True); idx=np.sort(idx)
bounds=[(s,e) for s,e in zip(idx,list(idx[1:])+[len(order)])]
def sim(K):
    cyc=[]; nf=ntp=nsl=npr=npick=0
    for bs,be in bounds:
        top=np.arange(bs,be)[:K]; f=filled[order[top]]; rr=real[order[top]]
        npick+=len(top); k=int(f.sum()); nf+=k
        if k==0: continue
        r=rr[f]; cyc.append(r.mean()); ntp+=int(tph[order[top]][f].sum()); nsl+=int(slh[order[top]][f].sum()); npr+=int((r>0).sum())
    cyc=np.array(cyc); se=cyc.std(ddof=1)/np.sqrt(max(len(cyc),2))
    return {"K":K,"n_cycles":int(len(cyc)),"mean":float(cyc.mean()),"se":float(se),
        "lower70":float(cyc.mean()-0.524*se),"p_pos":float((cyc>0).mean()),
        "fill_rate":nf/max(npick,1),"p_tp":ntp/max(nf,1),"p_sl":nsl/max(nf,1),"p_prof":npr/max(nf,1)}
res["top1"]=sim(1); res["top3"]=sim(3)
df=pd.DataFrame({"dk":dk,"p":prob,"f":filled,"r":real})
df["rk"]=df.groupby("dk")["p"].rank(method="first",ascending=False)
cyc2=df[df["rk"]<=1].groupby("dk").apply(lambda x:x.loc[x["f"],"r"].mean(),include_groups=False).dropna().values
assert abs(cyc2.mean()-res["top1"]["mean"])<1e-12 and len(cyc2)==res["top1"]["n_cycles"], "xcheck fail"
res["top1"]["xcheck"]="ok"
json.dump(res,open(f"{BASE}/exec_result_{tag}.json","w"),indent=2)
print(f"[{tag}] top1:",json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in res['top1'].items() if k!='xcheck'}))
print(f"[{tag}] top3 mean {res['top3']['mean']:.4f} n {res['top3']['n_cycles']}")
print(f"[{tag}] DONE {time.time()-t0:.0f}s",flush=True)
