import numpy as np, pandas as pd, json, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix

SEED=20260812
CLASSES=["Elevated PVR","VVC shunting","Pulmonary gas-exchange impairment","Increased metabolic demand","Ventricular dysfunction"]
FEATURES=['SaO2','HR','MAP','Hb','Glenn_pressure','SVC_sat']

# Load the exact final frozen primary cohort generated for the revised manuscript.
bal=pd.read_csv('/mnt/data/final_glenn/synthetic_glenn_balanced_final.csv')
X=bal[FEATURES]; y=bal.mechanism
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=SEED,stratify=y)
primary=Pipeline([('sc',StandardScaler()),('lr',LogisticRegression(max_iter=3000,random_state=SEED))])
primary.fit(Xtr,ytr)
primary_classes=list(primary.classes_)

# ---------------- Simple physiologic rule comparator ----------------
# Thresholds are derived only from TRAINING-set class medians, not from the held-out test set.
train=pd.concat([Xtr.reset_index(drop=True), ytr.reset_index(drop=True).rename('mechanism')],axis=1)
med=train.groupby('mechanism')[FEATURES].median()
# PVR threshold: midpoint between PVR median Glenn pressure and highest non-PVR median.
pvr_gp=(med.loc['Elevated PVR','Glenn_pressure'] + med.drop(index='Elevated PVR')['Glenn_pressure'].max())/2
# Lung SpO2 threshold: midpoint between lung median and the lowest median among remaining non-PVR states.
lung_sp=(med.loc['Pulmonary gas-exchange impairment','SaO2'] + med.drop(index=['Elevated PVR','Pulmonary gas-exchange impairment'])['SaO2'].min())/2
# Metabolic HR threshold: midpoint between metabolic median and highest non-metabolic median.
met_hr=(med.loc['Increased metabolic demand','HR'] + med.drop(index='Increased metabolic demand')['HR'].max())/2
# Venous saturation threshold to separate metabolic/ventricular from VVC; midpoint between ventricular and VVC medians.
svc_low=(med.loc['Ventricular dysfunction','SVC_sat'] + med.loc['VVC shunting','SVC_sat'])/2
# MAP threshold to distinguish ventricular from metabolic among low-SVC states; midpoint of ventricular and metabolic medians.
vd_map=(med.loc['Ventricular dysfunction','MAP'] + med.loc['Increased metabolic demand','MAP'])/2

def phys_rule(df):
    out=[]
    for _,r in df.iterrows():
        if r.Glenn_pressure >= pvr_gp:
            c='Elevated PVR'
        elif r.SaO2 <= lung_sp and r.SVC_sat > svc_low:
            c='Pulmonary gas-exchange impairment'
        elif r.SVC_sat <= svc_low:
            if r.HR >= met_hr and r.MAP >= vd_map:
                c='Increased metabolic demand'
            else:
                c='Ventricular dysfunction'
        else:
            c='VVC shunting'
        out.append(c)
    return np.array(out)

rule_pred=phys_rule(Xte)
rule_acc=accuracy_score(yte,rule_pred)
rule_rec=dict(zip(CLASSES,recall_score(yte,rule_pred,labels=CLASSES,average=None,zero_division=0)))
rule_cm=confusion_matrix(yte,rule_pred,labels=CLASSES,normalize='true')

# ---------------- Exact simulator for reviewer-requested sensitivity analyses ----------------
def truncnorm(rng, mean, sd, low, high, n):
    x=rng.normal(mean,sd,n)
    for _ in range(20):
        m=(x<low)|(x>high)
        if not m.any(): break
        x[m]=rng.normal(mean,sd,m.sum())
    return np.clip(x,low,high)

def simulate_vd_atrial(n,seed,atrial_rise_max=0.0):
    rng=np.random.default_rng(seed)
    Hb=truncnorm(rng,15,1.4,9.5,20.5,n)
    Pat0=truncnorm(rng,5,1,2.5,8,n)
    Rp=truncnorm(rng,3,0.65,1.0,7.0,n)
    VO2=truncnorm(rng,160,24,70,300,n)
    Psource=truncnorm(rng,68,6.5,52,84,n)
    Rvent=truncnorm(rng,1.8,0.35,0.9,2.8,n)
    Rup=truncnorm(rng,28,4.0,18,40,n)
    Rlow=truncnorm(rng,32,4.5,20,46,n)
    Spv=truncnorm(rng,.98,.006,.96,.995,n)
    sev=rng.uniform(0,1,n)
    # Final compensated ventricular phenotype
    pf=.97-.34*(sev**1.30); rvf=1+.65*(sev**1.15)
    mult=np.where(sev<=.72,1+1.45*(sev/.72),2.45-(2.45-1.50)*((sev-.72)/.28))
    Psource*=pf; Rvent*=rvf; Rup*=mult; Rlow*=mult
    # Reviewer-requested hidden filling-pressure sensitivity: progressive rise with severity.
    Pat=Pat0 + atrial_rise_max*sev
    Rpar=Rp
    G=1/(Rup+Rpar)+1/Rlow
    delta=Psource/(1+Rvent*G)
    MAPt=Pat+delta
    Qsvc=delta/(Rup+Rpar); Psvct=Pat+Qsvc*Rpar
    Qp=(Psvct-Pat)/Rp; Qivc=delta/Rlow; Qs=Qsvc+Qivc
    Cpv=1.34*Hb*Spv; fup=.40
    Ca=Cpv-(VO2/(10*Qp))*((1-fup))
    Sat=Ca/(1.34*Hb)
    Csvc=Ca-(fup*VO2)/(10*Qsvc); SVCt=Csvc/(1.34*Hb)
    HR0=truncnorm(rng,125,14,85,170,n)
    HRt=HR0+17*np.maximum(VO2/160-1,0)+.35*np.maximum(0,62-MAPt)+rng.normal(0,2.5,n)
    d=pd.DataFrame({
        'mechanism':'Ventricular dysfunction','SaO2':Sat*100+rng.normal(0,1.2,n),
        'HR':HRt+rng.normal(0,3,n),'MAP':MAPt+rng.normal(0,2,n),'Hb':Hb+rng.normal(0,.25,n),
        'Glenn_pressure':Psvct+rng.normal(0,1,n),'SVC_sat':SVCt*100+rng.normal(0,1.8,n),
        'Qp':Qp,'Qs':Qs,'severity':sev,'Pat_true':Pat
    })
    broad=d[d.Qp.between(.3,5)&d.Qs.between(1,8)&d.Glenn_pressure.between(3,30)&d.MAP.between(25,100)&d.SaO2.between(40,95)&d.SVC_sat.between(20,90)]
    strict=broad[broad.SaO2.between(65,80)&broad.Qp.between(.8,3.2)&broad.Qs.between(1.8,6)&broad.Glenn_pressure.between(5,24)].copy()
    return d,broad,strict

atrial_results=[]
for rise in [0,3,6]:
    full,broad,strict=simulate_vd_atrial(30000,SEED+1500,atrial_rise_max=rise)
    if len(strict):
        pred=primary.predict(strict[FEATURES]); prob=primary.predict_proba(strict[FEATURES])
        vc=pd.Series(pred).value_counts(normalize=True)
        pvr_idx=primary_classes.index('Elevated PVR'); vd_idx=primary_classes.index('Ventricular dysfunction')
        atrial_results.append({
            'atrial_rise_max_mmHg':rise,'generated':len(full),'broad':len(broad),'strict':len(strict),
            'median_atrial_pressure':float(strict.Pat_true.median()),
            'median_glenn_pressure':float(strict.Glenn_pressure.median()),
            'median_qs':float(strict.Qs.median()),
            'predicted_as_vd_pct':100*float(vc.get('Ventricular dysfunction',0)),
            'predicted_as_pvr_pct':100*float(vc.get('Elevated PVR',0)),
            'mean_prob_vd':float(prob[:,vd_idx].mean()),'mean_prob_pvr':float(prob[:,pvr_idx].mean())
        })

# ---------------- Mixed mechanism: elevated PVR + gas-exchange impairment ----------------
def simulate_mixed_pvr_gas(n,seed):
    rng=np.random.default_rng(seed)
    Hb=truncnorm(rng,15,1.4,9.5,20.5,n); Pat=truncnorm(rng,5,1,2.5,8,n)
    Rp=truncnorm(rng,3,0.65,1.0,7.0,n); VO2=truncnorm(rng,160,24,70,300,n)
    Psource=truncnorm(rng,68,6.5,52,84,n); Rvent=truncnorm(rng,1.8,0.35,0.9,2.8,n)
    Rup=truncnorm(rng,28,4.0,18,40,n); Rlow=truncnorm(rng,32,4.5,20,46,n)
    sev_pvr=rng.uniform(0,1,n); sev_gas=rng.uniform(0,1,n)
    Rp*=1.5+1.7*sev_pvr
    Spv=.94-.17*sev_gas
    Rpar=Rp; G=1/(Rup+Rpar)+1/Rlow; delta=Psource/(1+Rvent*G)
    MAPt=Pat+delta; Qsvc=delta/(Rup+Rpar); Psvct=Pat+Qsvc*Rpar; Qp=(Psvct-Pat)/Rp; Qivc=delta/Rlow; Qs=Qsvc+Qivc
    Cpv=1.34*Hb*Spv; fup=.40; Ca=Cpv-(VO2/(10*Qp))*(1-fup); Sat=Ca/(1.34*Hb)
    Csvc=Ca-(fup*VO2)/(10*Qsvc); SVCt=Csvc/(1.34*Hb)
    HR0=truncnorm(rng,125,14,85,170,n); HRt=HR0+17*np.maximum(VO2/160-1,0)+.35*np.maximum(0,62-MAPt)+rng.normal(0,2.5,n)
    d=pd.DataFrame({'SaO2':Sat*100+rng.normal(0,1.2,n),'HR':HRt+rng.normal(0,3,n),'MAP':MAPt+rng.normal(0,2,n),
        'Hb':Hb+rng.normal(0,.25,n),'Glenn_pressure':Psvct+rng.normal(0,1,n),'SVC_sat':SVCt*100+rng.normal(0,1.8,n),
        'Qp':Qp,'Qs':Qs,'sev_pvr':sev_pvr,'sev_gas':sev_gas})
    broad=d[d.Qp.between(.3,5)&d.Qs.between(1,8)&d.Glenn_pressure.between(3,30)&d.MAP.between(25,100)&d.SaO2.between(40,95)&d.SVC_sat.between(20,90)]
    strict=broad[broad.SaO2.between(65,80)&broad.Qp.between(.8,3.2)&broad.Qs.between(1.8,6)&broad.Glenn_pressure.between(5,24)].copy()
    return d,broad,strict

mf,mb,ms=simulate_mixed_pvr_gas(30000,SEED+1700)
mp=primary.predict_proba(ms[FEATURES]); mlabels=primary.predict(ms[FEATURES])
vc=pd.Series(mlabels).value_counts(normalize=True)
pvr_idx=primary_classes.index('Elevated PVR'); gas_idx=primary_classes.index('Pulmonary gas-exchange impairment')
top2=np.argsort(-mp,axis=1)[:,:2]
both_top2=np.mean([set([primary_classes[i] for i in inds]) >= {'Elevated PVR','Pulmonary gas-exchange impairment'} for inds in top2])
either_top1=np.mean(np.isin(mlabels,['Elevated PVR','Pulmonary gas-exchange impairment']))
mixed_result={
    'generated':len(mf),'broad':len(mb),'strict':len(ms),
    'median_spo2':float(ms.SaO2.median()),'median_glenn_pressure':float(ms.Glenn_pressure.median()),'median_svc_sat':float(ms.SVC_sat.median()),
    'top1_either_component_pct':100*float(either_top1),'both_components_top2_pct':100*float(both_top2),
    'predicted_class_pct':{c:100*float(vc.get(c,0)) for c in CLASSES},
    'mean_prob_pvr':float(mp[:,pvr_idx].mean()),'mean_prob_gas':float(mp[:,gas_idx].mean())
}

out={
    'physiologic_rule':{
        'thresholds':{'pvr_glenn_pressure':pvr_gp,'lung_spo2':lung_sp,'metabolic_hr':met_hr,'low_svc_sat':svc_low,'ventricular_map':vd_map},
        'accuracy':rule_acc,'recalls':rule_rec,'cm':rule_cm.tolist(),
        'logistic_accuracy':accuracy_score(yte,primary.predict(Xte))
    },
    'atrial_pressure_sensitivity':atrial_results,
    'mixed_pvr_gas_exchange':mixed_result
}
os.makedirs('/mnt/data/reviewer2_analysis',exist_ok=True)
with open('/mnt/data/reviewer2_analysis/results_reviewer2.json','w') as f: json.dump(out,f,indent=2)
pd.DataFrame(atrial_results).to_csv('/mnt/data/reviewer2_analysis/atrial_pressure_sensitivity.csv',index=False)
pd.DataFrame([mixed_result]).to_csv('/mnt/data/reviewer2_analysis/mixed_pvr_gas_exchange.csv',index=False)
pd.DataFrame([{'metric':'Accuracy','Physiologic rule':rule_acc,'Multinomial logistic regression':accuracy_score(yte,primary.predict(Xte))}]).to_csv('/mnt/data/reviewer2_analysis/physiologic_rule_comparator.csv',index=False)
print(json.dumps(out,indent=2))
