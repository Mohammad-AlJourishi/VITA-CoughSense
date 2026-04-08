#!/usr/bin/env python3

# DeepCoughNet

# === Setup ===
import subprocess, sys, os

def _pip(*a):
    subprocess.check_call([sys.executable,"-m","pip","install","-q",*a], stdout=subprocess.DEVNULL)

print("Installing packages...")
_pip("mindspore==2.8.0","numpy==1.26.4","scipy==1.13.1","librosa==0.10.2",
     "soundfile","pydub","scikit-learn","matplotlib","seaborn","tqdm")
_pip("--force-reinstall","--no-deps","numpy==1.26.4")
print("Done.")


_PIPE = r'''#!/usr/bin/env python3
from __future__ import annotations

# === Imports ===
import os, sys, json, random, warnings, logging, time, shutil, glob
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
from tqdm import tqdm

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger("DCN4")

# === Config ===
class CFG:
    COUGHVID_BASE = Path("/kaggle/input/datasets/mohammadjourishi/vita2026/archive (2)/tabular_form/tabular_form")
    COUGHVID_CSV = COUGHVID_BASE / "coughvid_v3.csv"
    COUGHVID_LOCAL = Path("/mnt/user-data/uploads/coughvid_v3.csv")

    COVID_SOUNDS_BASE = Path("/kaggle/input/datasets/pranaynandan63/covid-19-cough-sounds/cleaned_data")

    COSWARA_BASE = Path("/kaggle/input/datasets/sarabhian/coswara-dataset-heavy-cough")
    COSWARA_CSVS = COSWARA_BASE / "csvs"
    COSWARA_AUDIO = COSWARA_BASE / "coswara_data" / "kaggle_data"

    OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./output")
    CACHE_DIR = OUTPUT_DIR / "cache"; CKPT_DIR = OUTPUT_DIR / "checkpoints"; PLOT_DIR = OUTPUT_DIR / "plots"

    SR = 16000; DUR = 4.0; N_SAMPLES = 64000
    N_MELS = 64; N_FFT = 1024; HOP = 256; FMIN = 50; FMAX = 8000

    SEED = 42; BATCH_SIZE = 32; EPOCHS = 35; LR = 5e-4
    VAL_RATIO = 0.20; PATIENCE = 10; NC = 3

    L2N = {0: "healthy", 1: "symptomatic", 2: "COVID-19-positive"}
    L2C = {0: "green", 1: "orange", 2: "red"}

for d in [CFG.OUTPUT_DIR, CFG.CACHE_DIR, CFG.CKPT_DIR, CFG.PLOT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === MindSpore Setup ===
import mindspore as ms
from mindspore import context, Tensor, nn, ops
from mindspore.dataset import GeneratorDataset

try:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    Tensor(np.ones((2,2), dtype=np.float32))
    gpu_ok = True
except:
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    gpu_ok = False

ms.set_seed(CFG.SEED); np.random.seed(CFG.SEED); random.seed(CFG.SEED)

import librosa

# === Audio Utils ===
def load_audio_safe(fp, sr=CFG.SR):
    try:
        y, _ = librosa.load(str(fp), sr=sr, mono=True)
        return y
    except:
        return None

def audio_to_mel(y):
    y = y[:CFG.N_SAMPLES] if len(y) >= CFG.N_SAMPLES else np.pad(y, (0, CFG.N_SAMPLES - len(y)))
    S = librosa.feature.melspectrogram(
        y=y, sr=CFG.SR, n_fft=CFG.N_FFT, hop_length=CFG.HOP,
        n_mels=CFG.N_MELS, fmin=CFG.FMIN, fmax=CFG.FMAX)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)

# === Data placeholders (same logic as original, shortened here) ===
X_all, Y_all = [], []

# (You keep your dataset loading logic here — unchanged)

# === Normalize ===
X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)

mel_mean = X_all.mean(); mel_std = X_all.std()
X_all = (X_all - mel_mean) / (mel_std + 1e-8)

# === Split ===
X_train, X_val, Y_train, Y_val = train_test_split(
    X_all, Y_all, test_size=CFG.VAL_RATIO, random_state=CFG.SEED, stratify=Y_all)

# === Model ===
class DeepCoughNet(nn.Cell):
    def __init__(s, nc=CFG.NC):
        super().__init__()
        s.c1 = nn.SequentialCell([nn.Conv2d(1,32,3,pad_mode="same"), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2)])
        s.c2 = nn.SequentialCell([nn.Conv2d(32,64,3,pad_mode="same"), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2)])
        s.c3 = nn.SequentialCell([nn.Conv2d(64,128,3,pad_mode="same"), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2,2)])
        s.c4 = nn.SequentialCell([nn.Conv2d(128,256,3,pad_mode="same"), nn.BatchNorm2d(256), nn.ReLU()])
        s.gap = ops.ReduceMean(keep_dims=False); s.drop = nn.Dropout(p=0.4); s.fc = nn.Dense(256, nc)

    def construct(s, x):
        x = s.c1(x); x = s.c2(x); x = s.c3(x); x = s.c4(x)
        x = s.gap(x, (2, 3)); x = s.drop(x); return s.fc(x)

model = DeepCoughNet()

# === Loss ===
class FocalLoss(nn.Cell):
    def __init__(self, gamma=2.0, num_classes=3):
        super().__init__()
        self.gamma = gamma; self.nc = num_classes
        self.softmax = ops.Softmax(axis=1)
        self.onehot = ops.OneHot()
        self.log = ops.Log()
        self.on = Tensor(1.0, ms.float32); self.off = Tensor(0.0, ms.float32)

    def construct(self, logits, labels):
        probs = self.softmax(logits)
        one_hot = self.onehot(labels, self.nc, self.on, self.off)
        log_probs = self.log(probs + 1e-8)
        ce = -(one_hot * log_probs).sum(axis=1)
        p_correct = (probs * one_hot).sum(axis=1)
        return ((1.0 - p_correct) ** self.gamma * ce).mean()

loss_fn = FocalLoss()
optimizer = nn.Adam(model.trainable_params(), learning_rate=CFG.LR)

# === Training Loop ===
def fwd(f, l): return loss_fn(model(f), l)
gfn = ms.value_and_grad(fwd, None, optimizer.parameters)

def step(f, l):
    lo, g = gfn(f, l); optimizer(g); return lo

for ep in range(1, CFG.EPOCHS + 1):
    model.set_train(True)
    losses = []
    for f, l in zip(X_train, Y_train):
        losses.append(float(step(Tensor(f), Tensor(l)).asnumpy()))

    log.info(f"Epoch {ep} loss={np.mean(losses):.4f}")

# === Save ===
ms.save_checkpoint(model, str(CFG.CKPT_DIR / "deepcoughnet_best.ckpt"))

log.info("Done.")
'''

# === Run ===
_out = "/kaggle/working" if os.path.exists("/kaggle/working") else "./output"
os.makedirs(_out, exist_ok=True)
_s = os.path.join(_out, "pipeline_v4.py")

with open(_s, "w") as f:
    f.write(_PIPE)

print(f"Pipeline saved: {_s}")

result = subprocess.run([sys.executable, _s], cwd=_out)

if result.returncode != 0:
    sys.exit(result.returncode)

print("ALL DONE")