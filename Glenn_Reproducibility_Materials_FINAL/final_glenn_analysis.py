import numpy as np, pandas as pd, json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix

SEED=20260812
CLASSES=["Elevated PVR","VVC shunting","Pulmonary gas-exchange impairment","Increased metabolic demand","Ventricular dysfunction"]
FEATURES=['SaO2','HR','MAP','Hb','Glenn_pressure','SVC_sat']

def truncnorm(rng, mean, sd, low, high, n):
    x=rng.normal(mean,sd,n)
    for _ in range(20):
        m=(x<low)|(x>high)
        if not m.any(): break
        x[m]=rng.normal(mean,sd,m.sum())
    return np.clip(x,low,high)

def simulate_class(mech,n,seed,noise_mult=1.0,compensated=True,selective_noise=None,
                   hb_shift=0,pvr_mult=1,vo2_mult=1):
    rng=np.random.default_rng(seed)
    Hb=truncnorm(rng,15+hb_shift,1.4,9.5,20.5,n)
    Pat=truncnorm(rng,5,1,2.5,8,n)
    Rp=truncnorm(rng,3*pvr_mult,0.65*pvr_mult,1.0,7.0,n)
    VO2=truncnorm(rng,160*vo2_mult,24*vo2_mult,70,300,n)
    Psource=truncnorm(rng,68,6.5,52,84,n)
    Rvent=truncnorm(rng,1.8,0.35,0.9,2.8,n)
    Rup=truncnorm(rng,28,4.0,18,40,n)
    Rlow=truncnorm(rng,32,4.5,20,46,n)
    Spv=truncnorm(rng,.98,.006,.96,.995,n)
    Rvvc=np.full(n,np.inf)
    sev=rng.uniform(0,1,n)
    if mech=="Elevated PVR":
        Rp*=1.5+1.7*sev
    elif mech=="VVC shunting":
        Rvvc=45-35*sev
    elif mech=="Pulmonary gas-exchange impairment":
        Spv=.94-.17*sev
    elif mech=="Increased metabolic demand":
        VO2*=1.2+.6*sev
    elif mech=="Ventricular dysfunction":
        if compensated:
            pf=.97-.34*(sev**1.30)
            rvf=1+.65*(sev**1.15)
            mult=np.where(sev<=.72,1+1.45*(sev/.72),2.45-(2.45-1.50)*((sev-.72)/.28))
            Psource*=pf; Rvent*=rvf; Rup*=mult; Rlow*=mult
        else:
            Psource*=.86-.28*sev
            Rvent*=1+.25*sev
    Rpar=np.where(np.isfinite(Rvvc),1/(1/Rp+1/Rvvc),Rp)
    G=1/(Rup+Rpar)+1/Rlow
    delta=Psource/(1+Rvent*G)
    MAPt=Pat+delta
    Qsvc=delta/(Rup+Rpar)
    Psvct=Pat+Qsvc*Rpar
    Qp=(Psvct-Pat)/Rp
    Qvvc=np.where(np.isfinite(Rvvc),(Psvct-Pat)/Rvvc,0)
    Qivc=delta/Rlow
    Qs=Qsvc+Qivc
    fup=.40
    Cpv=1.34*Hb*Spv
    Ca=Cpv-(VO2/(10*Qp))*((1-fup)+fup*np.divide(Qvvc,Qsvc,out=np.zeros(n),where=Qsvc>0))
    Sat=Ca/(1.34*Hb)
    Csvc=Ca-(fup*VO2)/(10*Qsvc)
    SVCt=Csvc/(1.34*Hb)
    HR0=truncnorm(rng,125,14,85,170,n)
    HRt=HR0+17*np.maximum(VO2/160-1,0)+.35*np.maximum(0,62-MAPt)+rng.normal(0,2.5,n)
    sig={'SaO2':1.2,'Glenn_pressure':1.0,'MAP':2.0,'Hb':.25,'SVC_sat':1.8,'HR':3.0}
    if selective_noise: sig.update(selective_noise)
    obs={
        'SaO2':Sat*100+rng.normal(0,sig['SaO2']*noise_mult,n),
        'HR':HRt+rng.normal(0,sig['HR']*noise_mult,n),
        'MAP':MAPt+rng.normal(0,sig['MAP']*noise_mult,n),
        'Hb':Hb+rng.normal(0,sig['Hb']*noise_mult,n),
        'Glenn_pressure':Psvct+rng.normal(0,sig['Glenn_pressure']*noise_mult,n),
        'SVC_sat':SVCt*100+rng.normal(0,sig['SVC_sat']*noise_mult,n),
    }
    NIRS=.72*obs['SVC_sat']+.28*obs['SaO2']+rng.normal(0,2.2*noise_mult,n)
    return pd.DataFrame({'mechanism':mech,**obs,'NIRS':NIRS,'Qp':Qp,'Qs':Qs,'severity':sev})

def generate(noise_mult=1.0,compensated=True,selective_noise=None,seed_shift=0,hb_shift=0,pvr_mult=1,vo2_mult=1):
    df=pd.concat([simulate_class(c,30000,SEED+seed_shift+i*1000,noise_mult,compensated,selective_noise,hb_shift,pvr_mult,vo2_mult) for i,c in enumerate(CLASSES)],ignore_index=True)
    broad=df[df.Qp.between(.3,5)&df.Qs.between(1,8)&df.Glenn_pressure.between(3,30)&df.MAP.between(25,100)&df.SaO2.between(40,95)&df.SVC_sat.between(20,90)].copy()
    strict=broad[broad.SaO2.between(65,80)&broad.Qp.between(.8,3.2)&broad.Qs.between(1.8,6)&broad.Glenn_pressure.between(5,24)].copy()
    counts=strict.mechanism.value_counts().reindex(CLASSES)
    m=int(counts.min())
    bal=pd.concat([strict[strict.mechanism==c].sample(m,random_state=SEED+seed_shift+i) for i,c in enumerate(CLASSES)],ignore_index=True)
    return df,broad,strict,bal,counts

def fit_eval(df,features=FEATURES,seed=SEED,rf=False):
    X=df[features]; y=df.mechanism
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=seed,stratify=y)
    if rf:
        mod=RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=seed,n_jobs=-1)
    else:
        mod=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(max_iter=3000,random_state=seed))])
    mod.fit(Xtr,ytr)
    pred=mod.predict(Xte); prob=mod.predict_proba(Xte); classes=list(mod.classes_)
    Y=label_binarize(yte,classes=classes)
    auc=roc_auc_score(Y,prob,average='macro',multi_class='ovr')
    recalls=dict(zip(classes,recall_score(yte,pred,labels=classes,average=None,zero_division=0)))
    cm=confusion_matrix(yte,pred,labels=classes,normalize='true')
    return {'accuracy':accuracy_score(yte,pred),'auc':auc,'recalls':recalls,'n':len(df),'n_test':len(yte),'classes':classes,'cm':cm,'yte':np.asarray(yte),'pred':pred,'prob':prob,'model':mod}

def bootstrap(res,B=1000):
    rng=np.random.default_rng(SEED+919)
    y=res['yte']; pred=res['pred']; prob=res['prob']; classes=res['classes']; out=[]
    for _ in range(B):
        idx=rng.integers(0,len(y),len(y)); yy=y[idx]; pp=pred[idx]; pr=prob[idx]
        Y=label_binarize(yy,classes=classes)
        if Y.shape[1] != len(classes): continue
        try: auc=roc_auc_score(Y,pr,average='macro',multi_class='ovr')
        except: continue
        rec=recall_score(yy,pp,labels=classes,average=None,zero_division=0)
        out.append([accuracy_score(yy,pp),auc,*rec])
    a=np.array(out)
    return np.quantile(a,[.025,.975],axis=0)

# Primary
full,broad,strict,bal,counts=generate()
primary=fit_eval(bal); ci=bootstrap(primary)
rf=fit_eval(bal,rf=True)
sp=fit_eval(bal,['SaO2'])
noninv=fit_eval(bal,['SaO2','HR','MAP','Hb'])
nirs=fit_eval(bal,FEATURES+['NIRS'])
abl={}
for f in FEATURES:
    abl[f]=fit_eval(bal,[x for x in FEATURES if x!=f])
# calibration
classes=primary['classes']; Y=label_binarize(primary['yte'],classes=classes); prob=primary['prob']; pred=primary['pred']; y=primary['yte']
brier=float(np.mean(np.sum((prob-Y)**2,axis=1)))
conf=prob.max(axis=1); correct=(pred==y); bins=np.linspace(0,1,11); ece=0
for lo,hi in zip(bins[:-1],bins[1:]):
    m=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-9))
    if m.any(): ece += m.mean()*abs(correct[m].mean()-conf[m].mean())
# sensitivity scenarios
scenarios=[
 ('Baseline reproducibility',dict(seed_shift=100)),
 ('2x measurement noise',dict(noise_mult=2,seed_shift=200)),
 ('3x measurement noise',dict(noise_mult=3,seed_shift=300)),
 ('High catheter variability',dict(selective_noise={'Glenn_pressure':3.0,'SVC_sat':5.0},seed_shift=400)),
 ('Hb -2 g/dL',dict(hb_shift=-2,seed_shift=500)),
 ('Hb +2 g/dL',dict(hb_shift=2,seed_shift=600)),
 ('Baseline PVR -20%',dict(pvr_mult=.8,seed_shift=700)),
 ('Baseline PVR +25%',dict(pvr_mult=1.25,seed_shift=800)),
 ('VO2 -15%',dict(vo2_mult=.85,seed_shift=900)),
 ('VO2 +20%',dict(vo2_mult=1.2,seed_shift=1000)),
]
sens=[]
for name,kw in scenarios:
    f,b,s,ba,c=generate(**kw); met=fit_eval(ba,seed=SEED+kw.get('seed_shift',0))
    sens.append({'name':name,'generated':len(f),'broad':len(b),'strict':len(s),'smallest_class':int(c.min()),'balanced':len(ba),'accuracy':met['accuracy'],'auc':met['auc']})
# noncompensated controlled comparison
f0,b0,s0,ba0,c0=generate(compensated=False,seed_shift=1100); noncomp=fit_eval(ba0,seed=SEED+1100)
# unbalanced strict
unbal=fit_eval(strict,seed=SEED+1200)
# table 2 summary
sumrows=[]
for c in CLASSES:
    d=bal[bal.mechanism==c]; row={'Mechanism':c}
    for v in FEATURES:
        q=d[v].quantile([.025,.5,.975]); row[v]=[float(q.loc[.5]),float(q.loc[.025]),float(q.loc[.975])]
    sumrows.append(row)
# ventricular stages
vd=bal[bal.mechanism=='Ventricular dysfunction'].copy()
vd['stage']=pd.cut(vd.severity,[-.01,.33,.72,1.01],labels=['Mild','Moderate','Severe'])
vstage=vd.groupby('stage',observed=True)[['MAP','Qs','SVC_sat','HR']].median().round(2).to_dict('index')
# Save data / outputs
os.makedirs('/mnt/data/final_glenn',exist_ok=True)
bal.to_csv('/mnt/data/final_glenn/synthetic_glenn_balanced_final.csv',index=False)
strict.to_csv('/mnt/data/final_glenn/synthetic_glenn_strict_unbalanced_final.csv',index=False)
summary={
 'counts':{'generated':len(full),'broad':len(broad),'strict':len(strict),'strict_by_class':{k:int(v) for k,v in counts.items()},'balanced':len(bal),'per_class':int(len(bal)/5),'train':int(round(len(bal)*.75)),'test':primary['n_test']},
 'primary':{'accuracy':primary['accuracy'],'auc':primary['auc'],'acc_ci':ci[:,0].tolist(),'auc_ci':ci[:,1].tolist(),'recalls':primary['recalls'],'recall_ci':{c:[float(ci[0,2+i]),float(ci[1,2+i])] for i,c in enumerate(primary['classes'])}},
 'spO2':{'accuracy':sp['accuracy'],'auc':sp['auc']},'noninvasive':{'accuracy':noninv['accuracy'],'auc':noninv['auc']},'nirs':{'accuracy':nirs['accuracy'],'auc':nirs['auc']},'rf':{'accuracy':rf['accuracy'],'auc':rf['auc']},
 'ablation':{f:{'accuracy':m['accuracy'],'auc':m['auc']} for f,m in abl.items()},
 'calibration':{'brier':brier,'ece10':float(ece)},
 'sensitivity':sens,
 'noncompensated':{'n':len(ba0),'accuracy':noncomp['accuracy'],'auc':noncomp['auc'],'recalls':noncomp['recalls']},
 'unbalanced':{'n':len(strict),'accuracy':unbal['accuracy'],'auc':unbal['auc'],'recalls':unbal['recalls']},
 'table2':sumrows,'ventricular_stages':vstage,
 'cm_classes':primary['classes'],'cm':primary['cm'].tolist()
}
with open('/mnt/data/final_glenn/results.json','w') as f: json.dump(summary,f,indent=2)
print(json.dumps(summary,indent=2))
