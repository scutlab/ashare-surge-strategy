# 新规则模型训练 + TopK模拟（K=3/5/10），执行：-9%限价买，-1.5%止损/+30%止盈/收盘卖出
import numpy as np, os, json, time, pickle, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
X_,T_,S_=0.09,0.30,0.015
BASE=os.path.expanduser("~/workspace/ashare"); IN=BASE+f"/exec_arrays_{int(X_*100)}_{int(T_*100)}_{S_}"; t0=time.time()
Xtr=np.load(f"{IN}/tr_X.npy"); ytr=np.load(f"{IN}/tr_y.npy"); Xva=np.load(f"{IN}/va_X.npy"); yva=np.load(f"{IN}/va_y.npy")
X=np.concatenate([Xtr,Xva]); y=np.concatenate([ytr,yva]); del Xtr,ytr,Xva,yva
print(f"train+val {X.shape} pos {y.mean():.5f}",flush=True)
clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_depth=3,
    min_samples_leaf=100,class_weight="balanced",early_stopping=False,max_bins=128)
clf.fit(X,y); del X,y
mpk=f"{BASE}/exec_model_{int(X_*100)}_{int(T_*100)}_{S_}.pkl"; pickle.dump(clf,open(mpk,"wb"))
print(f"fit {time.time()-t0:.0f}s -> {mpk}",flush=True)
Xte=np.load(f"{IN}/te_X.npy"); yte=np.load(f"{IN}/te_y.npy"); ex=np.load(f"{IN}/te_exec.npy")
prob=clf.predict_proba(Xte)[:,1]; del Xte
dk=ex[:,0].astype(np.int64); sig=ex[:,1].astype(np.float64); bl=ex[:,2]; sl=ex[:,3]; sh=ex[:,4]; sc=ex[:,5]
entry=sig*(1-X_); filled=bl<=entry
slh=sl<=entry*(1-S_); tph=(~slh)&(sh>=entry*(1+T_))
real=np.where(slh,-S_,np.where(tph,T_,sc/entry-1))
res={"n_test":int(len(yte)),"base_rate":float(yte.mean()),"auc":float(roc_auc_score(yte,prob)),"x":X_,"t":T_,"s":S_}
print("base %.5f auc %.4f"%(res["base_rate"],res["auc"]),flush=True)
order=np.lexsort((-prob,dk)); udk,idx=np.unique(dk[order],return_index=True); idx=np.sort(idx)
day_top=[order[s:e] for s,e in zip(idx,list(idx[1:])+[len(order)])]
def sim(K, mask=None):
    cyc=[]; nf=ntp=nsl=npr=npick=0
    for top_all in day_top:
        top=top_all[:K]
        if mask is not None:
            top=top[mask[top]]
            if len(top)==0: continue
        f=filled[top]; npick+=len(top); k=int(f.sum()); nf+=k
        if k==0: continue
        r=real[top][f]; cyc.append(r.mean()); ntp+=int(tph[top][f].sum()); nsl+=int(slh[top][f].sum()); npr+=int((r>0).sum())
    cyc=np.array(cyc); se=cyc.std(ddof=1)/np.sqrt(max(len(cyc),2))
    return {"K":K,"n_cycles":int(len(cyc)),"mean":float(cyc.mean()),"se":float(se),
        "lower70":float(cyc.mean()-0.524*se),"p_pos":float((cyc>0).mean()),
        "fill_rate":nf/max(npick,1),"p_tp":ntp/max(nf,1),"p_sl":nsl/max(nf,1),"p_prof":npr/max(nf,1),
        "n_picks":npick,"n_filled":nf}
res["sims"]={f"top{k}":sim(k) for k in (3,5,10)}
# 独立复算 Top3（pandas）
df=pd.DataFrame({"dk":dk,"p":prob,"f":filled,"r":real})
df["rk"]=df.groupby("dk")["p"].rank(method="first",ascending=False)
cyc2=df[df["rk"]<=3].groupby("dk").apply(lambda x:x.loc[x["f"],"r"].mean(),include_groups=False).dropna().values
assert abs(cyc2.mean()-res["sims"]["top3"]["mean"])<1e-12 and len(cyc2)==res["sims"]["top3"]["n_cycles"], "xcheck fail"
res["sims"]["top3"]["xcheck"]="ok"
# 近一年切片（信号日>=2025-10-21，近似口径）
cut=np.datetime64("2025-10-21").astype("datetime64[D]").astype(int)
res["last1y_approx"]=sim(3, dk>=cut)
json.dump(res,open(f"{BASE}/exec_result_{int(X_*100)}_{int(T_*100)}_{S_}.json","w"),indent=2)
for k,v in res["sims"].items(): print(k,json.dumps({kk:(round(vv,4) if isinstance(vv,float) else vv) for kk,vv in v.items() if kk!="xcheck"}))
print("last1y",json.dumps({k:(round(v,4) if isinstance(v,float) else v) for k,v in res["last1y_approx"].items()}))
print(f"DONE {time.time()-t0:.0f}s",flush=True)
