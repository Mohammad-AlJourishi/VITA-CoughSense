# VITA — AI-Powered Asthma Cough Monitoring

> **Huawei ICT Competition 2025–2026 | Track: AI Innovation**
> Developed by VITA's Team — Princess Sumaya University for Technology (PSUT), Jordan

VITA is a Flutter-based Android application that uses on-device AI to continuously monitor cough patterns, classify cough types, predict asthma exacerbation risk, and alert users in real time — all while keeping raw audio strictly on-device.

---

## Table of Contents
- [Overview](#overview)
- [AI Models](#ai-models)
- [Architecture](#architecture)
- [Huawei Cloud Services](#huawei-cloud-services)
- [Results](#results)
- [Project Structure](#project-structure)
- [Reproduction Steps](#reproduction-steps)
- [App Setup](#app-setup)
- [License](#license)

---

## Overview

VITA addresses a critical gap in asthma management: patients have no objective way to track cough severity between clinic visits. VITA solves this by:

- **Classifying every cough** into Healthy, Symptomatic, or COVID-19-Positive using DeepCoughNet v4
- **Personalizing detection** with 5 demographically specialized models (CoughGate v3) based on user age and gender
- **Scoring exacerbation risk** (0–100) on-device using a 5-feature weighted time-series engine
- **Alerting patients** in real time via HMS Push Kit when risk reaches High
- **Storing cough summaries** in Huawei OBS to enable future model retraining on real-world user data
- **Delivering model updates OTA** via Huawei OBS without requiring an app store release

---

## AI Models

### Model 1 — DeepCoughNet v4 (Cough Type Classifier)

| Property | Value |
|---|---|
| Architecture | 4-block CNN (Conv2D → BN → ReLU → MaxPool) |
| Framework | MindSpore 2.8.0 |
| Input | 64-band log-mel spectrogram, 4s window, 16kHz |
| Classes | Healthy (0), Symptomatic (1), COVID-19-Positive (2) |
| Training data | 17,165 samples (COUGHVID v3 + COVID-19 Cough Sounds + Coswara Heavy Cough) |
| Best val_acc | **74.25%** (epoch 14, early stopping at epoch 24) |
| Healthy F1 | **0.850** (P=0.768, R=0.953) |
| Loss | Focal Loss (γ=2.0) + Label Smoothing (0.1) |
| Augmentation | SpecAugment (freq mask=8, time mask=20, N=2) |
| Optimizer | Adam (lr=5e-4, weight_decay=1e-4) |
| Format | ONNX (exported from MindSpore MindIR) |

**Training Datasets:**
- [COUGHVID v3](https://zenodo.org/record/4048312) — 10,132 healthy / 2,683 symptomatic / 720 COVID-19
- [COVID-19 Cough Sounds](https://www.kaggle.com/datasets/pranaynandan63/covid-19-cough-sounds) — 1,207 healthy / 149 COVID-19
- [Coswara Heavy Cough](https://www.kaggle.com/datasets/sarabhian/coswara-dataset-heavy-cough) — 1,648 healthy / 146 symptomatic / 480 COVID-19

Training log: [`DeepCoughnet_Model.log`]([DeepCoughNet_Model.log](https://github.com/Mohammad-AlJourishi/VITA-CoughSense/blob/main/DeepCoughNet%20Model.log))


---

### Model 2 — CoughGate v3 (Demographic-Specialized Detection)

| Property | Value |
|---|---|
| Architecture | CNN trained with MindSpore 2.8.0 |
| Input | 64-band log-mel spectrogram, 4s window, 16kHz |
| Routing | Age + Gender from user's local SQLite profile |
| Models | 5 specialized ONNX models |
| Best val_acc | **83.25%** (epoch 9) |
| Format | ONNX via ONNX Runtime for Android |

**Model Files (hosted on Huawei OBS, delivered OTA):**

| File | Demographic |
|---|---|
| `coughgate_v3_child.onnx` | Age ≤ 18 |
| `coughgate_v3_adult_m.onnx` | Age 19–59, Male |
| `coughgate_v3_adult_f.onnx` | Age 19–59, Female |
| `coughgate_v3_elderly.onnx` | Age ≥ 60 |
| `coughgate_v3_universal.onnx` | Fallback |

Training log: [`logs/coughgate_v3_training.log`]([logs/coughgate_v3_training.log](https://github.com/Mohammad-AlJourishi/VITA-CoughSense/blob/edb6fc6a5e1efcad0c7ada8677134e17af9cc992/DeepCoughNet%20Model.log))

---

## Architecture

```
Microphone (16kHz)
       │
       ▼
Log-Mel Spectrogram (64 bands, 4s window)
       │
       ├──► DeepCoughNet v4 ──► Healthy / Symptomatic / COVID-19-Positive
       │                              │
       │                        Symptomatic count
       │                              │
       └──► CoughGate v3 ────► Cough detected (age/gender routed)
                    │
                    ▼
         ExacerbationRiskService (on-device, <1ms)
         ├── OLS trend slope            (28%)
         ├── Surge vs personal baseline (24%)
         ├── Nocturnal fraction         (20%)
         ├── Wheeze/symptomatic ratio   (18%)
         └── High-cough streak         (10%)
                    │
                    ▼
             Risk Score 0–100
             Low / Medium / High
                    │
              ┌─────┴──────┐
              │            │
         HMS Push Kit   OBSSummaryService
         (High alert)   (Hourly summary → Huawei OBS)
                                │
                         Future retraining data
                                │
                    Retrain → Redeploy OTA → Users
```
## Huawei Cloud Services

| Service | Usage |
|---|---|
| **Huawei OBS** | Model delivery (OTA via `version.json` manifest) + hourly cough/wheeze summary uploads for future model retraining |
| **AppGallery Connect (AGC)** | User authentication and account management |
| **SMN (Simple Message Notification)** | Email verification on registration |
| **HMS Push Kit** | Real-time High-risk alerts delivered to device |

### OBS Bucket Structure
coughsense-models/
├── version.json                    ← manifest checked on every app launch
├── coughgate_v3_child.onnx
├── coughgate_v3_adult_m.onnx
├── coughgate_v3_adult_f.onnx
├── coughgate_v3_elderly.onnx
├── coughgate_v3_universal.onnx
└── deepcoughnet_v4.onnx
### Data Flywheel

Real-world cough events → summarized hourly → uploaded to OBS → used to retrain models → improved models redeployed OTA → back to users. No raw audio ever leaves the device.

---

## Results

| Metric | Value |
|---|---|
| DeepCoughNet v4 val_acc | **74.25%** |
| DeepCoughNet Healthy F1 | **0.850** (P=0.768, R=0.953) |
| DeepCoughNet Weighted avg F1 | **0.67** |
| CoughGate v3 val_acc | **83.25%** |
| Risk score computation | **< 1 ms** (fully on-device) |
| Training samples (DeepCoughNet) | **17,165** across 3 datasets |
| Demographic models | **5** specialized ONNX models |

---

## Project Structure

```
VITA-CoughSense/
├── lib/
│   ├── screens/
│   │   ├── live_detection_screen.dart      # Real-time cough detection UI
│   │   ├── trends_screen.dart              # Risk card + wheeze visualization
│   │   └── ...
│   ├── services/
│   │   ├── cough_detector_service.dart     # ONNX inference + demographic routing
│   │   ├── exacerbation_risk_service.dart  # On-device 5-feature risk engine
│   │   ├── obs_summary_service.dart        # Hourly OBS uploads
│   │   └── coughsense_push_service.dart    # HMS Push Kit alerts
│   └── background_service.dart             # Android background isolate
├── android/
│   └── app/src/main/AndroidManifest.xml
├── assets/
│   └── models/                             # Bundled fallback ONNX models
├── logs/
│   ├── deepcoughnet_v4_training.log        # Full DeepCoughNet v4 training log
│   └── coughgate_v3_training.log           # Full CoughGate v3 training log
├── training/
│   └── deepcoughnet_v4.py                  # Full MindSpore 2.8.0 training script
└── README.md
```
---

## Reproduction Steps

### 1. Train DeepCoughNet v4

**Requirements:** Kaggle account (GPU runtime recommended)

```bash
pip install mindspore==2.8.0 numpy==1.26.4 scipy==1.13.1 librosa==0.10.2 \
            soundfile pydub scikit-learn matplotlib seaborn tqdm
```

**Datasets required (add as Kaggle input datasets):**
- COUGHVID v3 (`mohammadjourishi/vita2026`)
- COVID-19 Cough Sounds (`pranaynandan63/covid-19-cough-sounds`)
- Coswara Heavy Cough (`sarabhian/coswara-dataset-heavy-cough`)

Run `training/deepcoughnet_v4.py` as a single Kaggle notebook cell.

Expected output:
- `deepcoughnet_best.ckpt` — best checkpoint (epoch 14)
- `deepcoughnet.mindir` — MindIR export
- `inference_meta.json` — normalization parameters (mean, std)
- Best val_acc: **74.25%**

### 2. Export to ONNX

```python
import mindspore as ms
from mindspore import Tensor
import numpy as np

model = DeepCoughNet()
ms.load_checkpoint("deepcoughnet_best.ckpt", model)
model.set_train(False)

dummy = Tensor(np.zeros((1, 1, 64, 251), dtype=np.float32))
ms.export(model, dummy, file_name="deepcoughnet_v4", file_format="ONNX")
```

### 3. Upload Models to Huawei OBS

1. Create OBS bucket `coughsense-models` in Huawei Cloud Console
2. Set Bucket Policy to allow public `GetObject` (use Bucket Policy, not ACL)
3. Upload all `.onnx` files and `version.json`
4. `version.json` format:

```json
{
  "version": "3.0",
  "models": {
    "child": "coughgate_v3_child.onnx",
    "adult_m": "coughgate_v3_adult_m.onnx",
    "adult_f": "coughgate_v3_adult_f.onnx",
    "elderly": "coughgate_v3_elderly.onnx",
    "universal": "coughgate_v3_universal.onnx",
    "classifier": "deepcoughnet_v4.onnx"
  }
}
```

### 4. Run the Flutter App

```bash
# Prerequisites: Flutter 3.x, Android SDK, connected Android device

git clone https://github.com/Mohammad-AlJourishi/VITA-CoughSense.git
cd VITA-CoughSense
flutter pub get
flutter run
```

---

## App Setup

1. Register on [Huawei AppGallery Connect](https://developer.huawei.com/consumer/en/appgallery) and create a new project
2. Enable: Auth Service, Push Kit, Cloud Storage (OBS)
3. Download `agconnect-services.json` → place in `android/app/`
4. Set your OBS bucket endpoint in `lib/services/obs_summary_service.dart`
5. Set your OBS model bucket URL in `lib/services/cough_detector_service.dart`
6. Run `flutter pub get && flutter run`

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)

---

Built with MindSpore 2.8.0 · Huawei Cloud · Flutter · ONNX Runtime
