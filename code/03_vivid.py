"""
03 - VIVID: importance (permutation) + interactions (partial dependence, Friedman H)
     at the level of the ORIGINAL VARIABLES (not one-hot). Output: csv/vivi_v2_{sdm,ldm,gdm}.csv
"""
import os, numpy as np, pandas as pd, json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
np.random.seed(7)
BASE=os.path.dirname(os.path.abspath(__file__))
meta=json.load(open(f"{BASE}/meta_v2.json")); CONT,CAT,RESP=meta['CONT'],meta['CAT'],meta['RESP']
feat=meta['feat_names']; cmin=pd.Series(meta['cmin']); cmax=pd.Series(meta['cmax'])
model=tf.keras.models.load_model(f"{BASE}/nn_v2.keras",compile=False)
orig=pd.read_csv(f"{BASE}/csv/orig_v2.csv"); VARS=CONT+CAT
LABEL={'gdd_cum':'Thermal time','tmean':'Tmean','tmax':'Tmax','srad':'Solar rad.','diff_rad':'Diffuse rad.',
       'q':'Humidity','cultivar':'Cultivar','county':'County','treatment':'Treatment'}
def enc(df):
    Xc=(df[CONT].astype(float)-cmin[CONT])/(cmax[CONT]-cmin[CONT]); Xd=pd.get_dummies(df[CAT].astype(str),prefix=CAT)
    X=pd.concat([Xc.reset_index(drop=True),Xd.reset_index(drop=True)],axis=1)
    for f in feat:
        if f not in X.columns: X[f]=0.0
    return X[feat].astype(float).values
def pred(df): return model.predict(enc(df),verbose=0,batch_size=512)
BG=orig.sample(min(160,len(orig)),random_state=1).reset_index(drop=True)
gv=lambda v: list(pd.unique(orig[v].astype(str))) if v in CAT else list(np.quantile(orig[v].astype(float),np.linspace(.05,.95,6)))
G={};P1={}
for v in VARS:
    g=gv(v); acc=np.zeros((len(g),3))
    for k,val in enumerate(g):
        t=BG.copy(); t[v]=val; acc[k]=pred(t).mean(0)
    G[v]=g; P1[v]=acc-acc.mean(0,keepdims=True)
base=pred(orig); IMP={}
for v in VARS:
    inc=np.zeros(3)
    for _ in range(5):
        t=orig.copy(); t[v]=np.random.permutation(t[v].values); inc+=np.sqrt(np.mean((pred(t)-base)**2,0))
    IMP[v]=inc/5
M={r:np.zeros((len(VARS),len(VARS))) for r in RESP}
for i in range(len(VARS)):
    for j in range(i+1,len(VARS)):
        vi,vj=VARS[i],VARS[j]; gi,gj=G[vi],G[vj]; P2=np.zeros((len(gi),len(gj),3))
        for a,x in enumerate(gi):
            for b,y in enumerate(gj):
                t=BG.copy(); t[vi]=x; t[vj]=y; P2[a,b]=pred(t).mean(0)
        inter=(P2-P2.mean((0,1),keepdims=True))-P1[vi][:,None,:]-P1[vj][None,:,:]
        val=np.sqrt(np.mean(inter**2,(0,1)))
        for r in range(3): M[RESP[r]][i,j]=M[RESP[r]][j,i]=val[r]
for r in RESP:
    for i,v in enumerate(VARS): M[r][i,i]=IMP[v][RESP.index(r)]
    pd.DataFrame(M[r],index=[LABEL[v] for v in VARS],columns=[LABEL[v] for v in VARS]).to_csv(f"{BASE}/csv/vivi_v2_{r}.csv")
print("VIVID ok. Vimp/Vint por saida:")
for r in RESP:
    off=M[r][~np.eye(len(VARS),bool)]; print(f"  {r}: Vimp max {np.diag(M[r]).max():.3f}  Vint max {off.max():.3f}")
