#!/usr/bin/env python3

# CoughGate

import subprocess, sys, os, tempfile

def _pip(*a):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *a], stdout=subprocess.DEVNULL)

print("Installing packages...")
_pip(
    "mindspore==2.8.0",
    "numpy==1.26.4",
    "pandas==2.2.2",
    "scipy==1.13.1",
    "librosa==0.10.2",
    "soundfile",
    "scikit-learn",
    "matplotlib",
    "onnx",
    "onnxruntime",
    "tqdm",
)
_pip("--force-reinstall", "--no-deps", "numpy==1.26.4", "pandas==2.2.2")
print("Done.\n")

_PIPE = r'''
import os, sys, json, random, warnings, logging, time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import scipy.signal
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("CoughGateV3")

# CONFIG
class CFG:
    SR=16000; DUR=4.0; N=64000
    MELS=64; FFT=1024; HOP=256; FMIN=50; FMAX=8000
    NC=2

    COUGH_POS_THRESH=0.80
    COUGH_NEG_THRESH=0.30

    SEED=42; BATCH=32; EPOCHS=45; LR=4e-4; LR_MIN=1e-5
    VAL_RATIO=0.20; PATIENCE=14
    OVERSAMPLE_TARGET=6000

    FREQ_MASK=10; TIME_MASK=25; N_MASKS=3
    LABEL_SMOOTH=0.05; FOCAL_GAMMA=2.5
    MIXUP_ALPHA=0.3; TIME_SHIFT_MAX=8000

    SNR_MIN_DB=3.0; SNR_MAX_DB=20.0; SNR_PROB=0.5

    N_PHONE_BLOWS=1200
    N_WORD_BURSTS=1200
    N_BREATH_SYNTH=600

    ESC_HARD_CATS={22,38,36,9,40,34,5,26,29,30,27,13,0,1,24,7,2}
    ESC_COUGH_CAT=10
    ESC_HARD_OVERSAMPLE=3

    DEMO_CHILD=0; DEMO_ADULT_M=1; DEMO_ADULT_F=2; DEMO_ELDERLY=3; DEMO_UNKNOWN=4
    N_DEMO=5

    MAX_POS=None; MAX_NEG=None

    OUTPUT=Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./output")

for d in [CFG.OUTPUT]: d.mkdir(parents=True, exist_ok=True)

np.random.seed(CFG.SEED); random.seed(CFG.SEED)

# AUDIO

def load_audio(fp, sr=CFG.SR):
    try:
        y,_=librosa.load(str(fp), sr=sr, mono=True)
        return y if len(y)>sr*0.3 else None
    except: return None

def audio_to_mel(y, sr=CFG.SR):
    y=y[:CFG.N] if len(y)>=CFG.N else np.pad(y,(0,CFG.N-len(y)))
    S=librosa.feature.melspectrogram(y=y,sr=sr,n_fft=CFG.FFT,hop_length=CFG.HOP,n_mels=CFG.MELS,fmin=CFG.FMIN,fmax=CFG.FMAX)
    return librosa.power_to_db(S,ref=np.max).astype(np.float32)

# DEMO

def age_gender_to_demo(age, gender):
    if age is None or (isinstance(age,float) and np.isnan(age)): return CFG.DEMO_UNKNOWN
    age=int(age)
    if age<=18: return CFG.DEMO_CHILD
    if age>=60: return CFG.DEMO_ELDERLY
    g=str(gender or '').lower()
    if g in ('male','m','1'): return CFG.DEMO_ADULT_M
    if g in ('female','f','0','woman'): return CFG.DEMO_ADULT_F
    return CFG.DEMO_UNKNOWN

def demo_onehot(i):
    v=np.zeros(CFG.N_DEMO,dtype=np.float32); v[i]=1.0; return v

# LOAD COUGHVID
csv_path=None
for dp,_,fns in os.walk("/kaggle/input"):
    if "coughvid_v3.csv" in fns:
        csv_path=Path(dp)/"coughvid_v3.csv"; break

if csv_path is None: sys.exit(1)

df=pd.read_csv(csv_path)

audio_dir=None
for dp,_,fns in os.walk("/kaggle/input"):
    if any(fn.endswith(".webm") for fn in fns[:10]):
        audio_dir=Path(dp); break

files=set(os.listdir(audio_dir))
df=df[df["audio_name"].isin(files)]

df_pos=df[df["cough_detected"]>=CFG.COUGH_POS_THRESH]
df_neg=df[df["cough_detected"]<CFG.COUGH_NEG_THRESH]

# EXTRACT

def extract(df,label):
    X,Y,D=[],[],[]
    for _,r in tqdm(df.iterrows(), total=len(df)):
        y=load_audio(audio_dir/r["audio_name"])
        if y is None: continue
        X.append(audio_to_mel(y)); Y.append(label)
        D.append(demo_onehot(age_gender_to_demo(r.get("age"),r.get("gender"))))
    return np.array(X),np.array(Y),np.array(D)

X_pos,Y_pos,D_pos=extract(df_pos,1)
X_neg,Y_neg,D_neg=extract(df_neg,0)

# MERGE
X=np.concatenate([X_pos,X_neg])
Y=np.concatenate([Y_pos,Y_neg])
D=np.concatenate([D_pos,D_neg])

mel_mean=X.mean(); mel_std=X.std()+1e-8
X=(X-mel_mean)/mel_std
np.save(CFG.OUTPUT/"mel_norm.npy",[mel_mean,mel_std])

Xt,Xv,Yt,Yv,Dt,Dv=train_test_split(X,Y,D,test_size=CFG.VAL_RATIO,random_state=CFG.SEED,stratify=Y)

# MODEL
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor

ms.set_context(mode=ms.PYNATIVE_MODE)

class Model(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv=nn.SequentialCell([
            nn.Conv2d(1,32,3,pad_mode="same"),nn.ReLU(),
            nn.Conv2d(32,64,3,pad_mode="same"),nn.ReLU(),
        ])
        self.gap=ops.ReduceMean(keep_dims=False)
        self.fc=nn.Dense(64+CFG.N_DEMO,2)
    def construct(self,x,d):
        x=self.conv(x)
        x=self.gap(x,(2,3))
        x=ops.concat((x,d),1)
        return self.fc(x)

model=Model()

loss_fn=nn.SoftmaxCrossEntropyWithLogits(sparse=True)
opt=nn.Adam(model.trainable_params(),learning_rate=CFG.LR)

def step(x,d,y):
    def f(m,d,y): return loss_fn(model(m,d),y)
    grad=ms.value_and_grad(f,None,opt.parameters)
    l,g=grad(x,d,y); opt(g); return l

# TRAIN
for ep in range(CFG.EPOCHS):
    for i in range(0,len(Xt),CFG.BATCH):
        xb=Tensor(Xt[i:i+CFG.BATCH][:,None,:,:])
        db=Tensor(Dt[i:i+CFG.BATCH])
        yb=Tensor(Yt[i:i+CFG.BATCH])
        step(xb,db,yb)

# EVAL
preds=[]
for i in range(0,len(Xv),CFG.BATCH):
    xb=Tensor(Xv[i:i+CFG.BATCH][:,None,:,:])
    db=Tensor(Dv[i:i+CFG.BATCH])
    p=model(xb,db).asnumpy().argmax(1)
    preds.extend(p)

print(classification_report(Yv,preds))

'''

_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
_tmp.write(_PIPE)
_tmp.close()

subprocess.run([sys.executable, _tmp.name])
os.unlink(_tmp.name)

