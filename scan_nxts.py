# 联合扫描 (n,x,t,s)：用已训模型(标签x9/t30/s1.5%)的排序，在多种执行参数下评估Top-n组合
# 为最终候选点精确重训做准备
import numpy as np, os, json, time, pickle, itertools
X_,T_,S_=0.09,0.30,0.015
BASE=os.path.expanduser("~/workspace/ashare"); IN=BASE+f"/exec_arrays_{int(X_*100)}_{int(T_*100)}_{S_}"; t0=time.time()
clf=pickle.load(open(f"{BASE}/exec_model_{int(X_*100)}_{int(T_*100)}_{S_}.pkl","rb"))
Xte=np.load(f"{IN}/te_X.npy"); ex=np.load(f"{IN}/te_exec.npy")
prob=clf.predict_proba(Xte)[:,1]; del Xte,clf
dk=ex[:,0].astype(np.int64); sig=ex[:,1].astype(np.float64); bl=ex[:,2]; sl=ex[:,3]; sh=ex[:,4]; sc=ex[:,5]
del ex
order=np.lexsort((-prob,dk)); udk,idx=np.unique(dk[order],return_index=True); idx=np.sort(idx)
bounds=[(s,e) for s,e in zip(idx,list(idx[1:])+[len(order)])]
NS=[1,2,3,5,8,10,15,20,30]; XS=[.06,.08,.09,.10]; TS=[.15,.20,.30]; SS=[.015,.02]
execs={}
for x,t,s in itertools.product(XS,TS,SS):
    entry=sig*(1-x); filled=bl<=entry
    slh=sl<=entry*(1-s); tph=(~slh)&(sh>=entry*(1+t))
    real=np.where(slh,-s,np.where(tph,t,sc/entry-1))
    execs[(x,t,s)]=(filled,real)
    del entry,filled,slh,tph,real
print(f"exec combos {len(execs)} days {len(bounds)}",flush=True)
res=[]
for (x,t,s),(filled,real) in execs.items():
    fr=filled[order]; rr=real[order]
    for n in NS:
        cyc=[]; nf=npick=0
        for bs,be in bounds:
            top=np.arange(bs,be)[:n]; f=fr[top]; npick+=len(top); k=int(f.sum()); nf+=k
            if k: cyc.append(rr[top][f].mean())
        cyc=np.array(cyc)
        if len(cyc)<30: continue
        se=cyc.std(ddof=1)/np.sqrt(len(cyc))
        res.append({"n":n,"x":x,"t":t,"s":s,"ncyc":int(len(cyc)),"mean":float(cyc.mean()),"se":float(se),
            "lower70":float(cyc.mean()-0.524*se),"p_pos":float((cyc>0).mean()),"fill_rate":nf/max(npick,1)})
json.dump(res,open(BASE+"/scan_nxts.json","w"))
top=sorted(res,key=lambda r:r["mean"],reverse=True)[:12]
print(f"{'n':>3} {'x':>4} {'t':>4} {'s':>5} {'ncyc':>6} {'mean%':>8} {'p_pos':>6} {'fill%':>6}")
for r in top: print(f"{r['n']:>3} {r['x']*100:>3.0f}% {r['t']*100:>3.0f}% {r['s']*100:>4.1f}% {r['ncyc']:>6} {r['mean']*100:>7.3f}% {r['p_pos']:>6.3f} {r['fill_rate']*100:>5.2f}%")
print(f"DONE {time.time()-t0:.0f}s",flush=True)
