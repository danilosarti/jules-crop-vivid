"""
02 - Train the emulator (same architecture as the paper) and evaluate (R2/RMSE in t/ha).
Input : csv/model_data_v2.csv    Output: nn_v2.keras, eval_v2.npz, meta_v2.json, csv/orig_v2.csv
"""
import os, numpy as np, pandas as pd, json
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'; os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
tf.keras.utils.set_random_seed(123)
BASE=os.path.dirname(os.path.abspath(__file__))
d=pd.read_csv(f"{BASE}/csv/model_data_v2.csv")
CONT=['gdd_cum','tmean','tmax','srad','diff_rad','q']; CAT=['cultivar','county','treatment']; RESP=['sdm','ldm','gdm']
Xc=d[CONT].astype(float); cmin,cmax=Xc.min(),Xc.max(); Xc_n=(Xc-cmin)/(cmax-cmin)
Xd=pd.get_dummies(d[CAT].astype(str),prefix=CAT)
X=pd.concat([Xc_n.reset_index(drop=True),Xd.reset_index(drop=True)],axis=1).astype(float); feat=list(X.columns)
Y=d[RESP].astype(float); ymin,ymax=Y.min(),Y.max(); Yn=(Y-ymin)/(ymax-ymin)
rng=np.random.RandomState(123); idx=rng.permutation(len(X)); ntr=int(0.75*len(X)); tr,te=idx[:ntr],idx[ntr:]
inp=Input(shape=(X.shape[1],))
h=layers.Dense(256,activation='relu')(inp); h=layers.Dropout(0.5)(h)
h=layers.Dense(128,activation='relu')(h); h=layers.Dropout(0.4)(h)
h=layers.Dense(64,activation='relu')(h); h=layers.Dropout(0.3)(h)
model=Model(inp, layers.Dense(3,activation='sigmoid')(h))
model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(),metrics=['mse'])
es=tf.keras.callbacks.EarlyStopping(patience=30,restore_best_weights=True,monitor='val_loss')
model.fit(X.values[tr],Yn.values[tr],epochs=400,batch_size=16,validation_split=0.2,verbose=0,callbacks=[es])
model.save(f"{BASE}/nn_v2.keras")
dn=lambda a: a*(ymax.values-ymin.values)+ymin.values
pte=dn(model.predict(X.values[te],verbose=0)); ate=dn(Yn.values[te]); m={}
for i,c in enumerate(RESP):
    a,p=ate[:,i],pte[:,i]; m[c]={'R2':round(float(1-np.sum((a-p)**2)/np.sum((a-a.mean())**2)),3),'RMSE':round(float(np.sqrt(np.mean((a-p)**2))),3)}
print(json.dumps(m,indent=2))
np.savez(f"{BASE}/eval_v2.npz",pred_te=pte,act_te=ate)
d[CONT+CAT+RESP].to_csv(f"{BASE}/csv/orig_v2.csv",index=False)
json.dump(dict(CONT=CONT,CAT=CAT,RESP=RESP,feat_names=feat,cmin=cmin.to_dict(),cmax=cmax.to_dict(),
    ymin=ymin.to_dict(),ymax=ymax.to_dict(),metrics=m),open(f"{BASE}/meta_v2.json","w"))
