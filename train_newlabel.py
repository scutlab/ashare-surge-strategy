# 新规则标签 HGB 训练 + 测试集评估 + Top10 模拟（执行规则：-2%限价买，+2%止盈/收盘卖）
import numpy as np, os, json, time, pickle
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
BASE=os.path.expanduser("~/workspace/ashare"); IN=BASE+"/newlabel_arrays"; t0=time.time()
Xtr=np.load(f"{IN}/tr_X.npy"); ytr=np.load(f"{IN}/tr_y.npy")
Xva=np.load(f"{IN}/va_X.npy"); yva=np.load(f"{IN}/va_y.npy")
X=np.concatenate([Xtr,Xva]); y=np.concatenate([ytr,yva]); del Xtr,ytr,Xva,yva
print(f"train+val {X.shape} pos {y.mean():.4f}",flush=True)
clf=HistGradientBoostingClassifier(max_iter=300,learning_rate=0.05,max_depth=3,
    min_samples_leaf=100,class_weight="balanced",early_stopping=False,max_bins=128)
clf.fit(X,y); del X,y
pickle.dump(clf,open(BASE+"/newlabel_model_HGB.pkl","wb"))
print(f"fit {time.time()-t0:.0f}s",flush=True)
# 测试集
Xte=np.load(f"{IN}/te_X.npy"); yte=np.load(f"{IN}/te_y.npy"); ex=np.load(f"{IN}/te_exec.npy")
prob=clf.predict_proba(Xte)[:,1]; del Xte
res={"n_test":int(len(yte)),"base_rate":float(yte.mean()),"auc":float(roc_auc_score(yte,prob)),
     "mean_prob":float(prob.mean())}
print("test base %.4f auc %.4f"%(res["base_rate"],res["auc"]),flush=True)
dk=ex[:,0].astype(np.int64); sig=ex[:,1].astype(np.float64); bl=ex[:,2]; sh=ex[:,3]; sc=ex[:,4]
entry=sig*0.98; filled=bl<=entry; tp=sh>=entry*1.02
real=np.where(tp,0.02,sc/entry-1)
# Top10/日
order=np.argsort(-prob,kind="stable")
sdays,idx=np.unique(dk[order],return_index=True)
cyc=[]; picks_fill=0; picks_tp=0; picks_prof=0; picks=0
for s,e in zip(idxs:=idx, list(idx[1:])+[len(order)]):
    top=order[s:e][:10]; f=filled[top]
    picks+=len(top); pf=int(f.sum()); picks_fill+=pf
    if pf==0: continue
    r=real[top][f]; cyc.append(r.mean())
    picks_tp+=int(tp[top][f].sum()); picks_prof+=int((r>0).sum())
cyc=np.array(cyc); se=cyc.std(ddof=1)/np.sqrt(len(cyc))
res["top10"]={"n_cycles":int(len(cyc)),"mean_cycle":float(cyc.mean()),"se":float(se),
    "lower70_onesided":float(cyc.mean()-0.524*se),"p_cycle_pos":float((cyc>0).mean()),
    "fill_rate":picks_fill/picks,"p_tp_given_fill":picks_tp/picks_fill,"p_prof_given_fill":picks_prof/picks_fill,
    "n_picks":picks,"n_filled":picks_fill}
# 独立复算：pandas groupby 交叉验证均值
import pandas as pd
df=pd.DataFrame({"dk":dk[order],"p":prob[order],"f":filled[order],"r":real[order]})
df["rk"]=df.groupby("dk")["p"].rank(method="first",ascending=False)
g=df[df["rk"]<=10].groupby("dk")
cyc2=g.apply(lambda x:x.loc[x["f"],"r"].mean(),include_groups=False).dropna().values
res["top10"]["xcheck_mean"]=float(cyc2.mean()); res["top10"]["xcheck_n"]=int(len(cyc2))
assert abs(cyc2.mean()-cyc.mean())<1e-9 and len(cyc2)==len(cyc), "xcheck failed"
json.dump(res,open(BASE+"/newlabel_result.json","w"),indent=2)
print(json.dumps(res["top10"],indent=2)); print(f"DONE {time.time()-t0:.0f}s",flush=True)
