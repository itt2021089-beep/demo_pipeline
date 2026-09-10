# Safe Zone AI — two-stage mosquito larva classification (runnable package)

Everything needed to run the trained pipeline on another machine. No dataset
and no training code is required.

```
INPUT IMAGE
    |
Stage 1 - ResNet50 (PyTorch)        Larva / Non_larva
    |  if Non_larva -> STOP, Stage 2 is not executed
    |  if Larva
Stage 2 - EfficientNet-B0 (Keras)   Aedes / Anopheles / Culex / Non_larvae
    |
FINAL RESULT
```

## 1. Install

Use **Python 3.11**. TensorFlow 2.21 has no Windows wheel for Python 3.13+,
so a newer interpreter will fail during install.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

The install is large (~2-3 GB) because it pulls both PyTorch and TensorFlow.
`requirements.txt` pins CPU builds; both models run fine on CPU. Stage 1 uses
CUDA automatically if a compatible GPU is present.

## 2. Run

Web UI:

```bash
streamlit run demo/app.py
```

or double-click `run_demo.bat` (Windows) / `bash run_demo.sh` (macOS/Linux).

Command line, no UI:

```bash
python -m demo.inference sample_images/Aedes/<one-of-the-files>.jpg
```

## 3. Verify the files arrived intact

```bash
python verify_package.py
```

Re-hashes every file and compares against `MANIFEST.json`. Worth running after
any transfer, because the two checkpoints are ~113 MB together and a truncated
upload produces confusing load errors rather than an obvious failure.

Note: `demo/test_pipeline.py` is included for reference but will **not** run
here — it reads the research datasets, which are deliberately not shipped.

## What is in here

| Path | What it is |
|---|---|
| `demo/` | UI, inference, per-stage preprocessing, config |
| `stage1/` | ResNet50 model definition (imported by `demo/inference.py`) and the training/eval code it came from |
| `stage2/` | Only `dedup.py` — `stage1/dataset.py` imports it at module load. The rest of Stage 2's training code is not part of the runtime. |
| `preprocessing.py` | Project transforms; `GammaCorrectionTransform` is used by Stage 1 |
| `demo_models/stage1_resnet50/best_model.pth` | Trained Stage 1 checkpoint (~94 MB) |
| `demo_models/stage2_efficientnet/best_tuning_run.h5` | Stage 2 Keras model (~19 MB) |
| `sample_images/` | 15 images for a quick smoke test |
| `requirements.txt` | Pinned dependency versions |
| `MANIFEST.json` | SHA-256 of every file in the package |

## Two things a reviewer should know

**The two stages do NOT share preprocessing.** They both take 224x224 RGB, which
makes it tempting to reuse one tensor — doing so is a silent bug.

- Stage 1: `gamma(1.5) -> resize 224 -> ToTensor -> ImageNet normalise -> NCHW`
- Stage 2: `resize 224 -> float32 in [0, 255] -> NHWC` (**raw pixels, not normalised**)

The Keras model has `Rescaling(1/255) -> Normalization -> Rescaling(1/std)` baked
into its own graph. Normalising before handing it the array double-normalises and
quietly corrupts predictions. `demo/preprocessing.py` keeps them separate.

**Stage 2's class mapping is user-confirmed, not source-verified.** The Keras
training code that produced `best_tuning_run.h5` is not available, and the `.h5`
stores only weights — no `class_indices`. The order
`0=Aedes, 1=Anopheles, 2=Culex, 3=Non_larvae` was confirmed by the project owner
and is corroborated by a prediction diagnostic (clean 1-to-1 mapping, 88-100%
agreement, matching sorted folder order) — but corroboration is not proof. The UI
labels it accordingly. Stage 1's mapping *is* source-verified (written at
training time from the ImageFolder ordering).

## Accuracy figures — do not misquote these

| | Value | What it actually is |
|---|---|---|
| Stage 1, this checkpoint | 99.92% test accuracy | 1,245 held-out images, verified leakage-free split. Checkpoint chosen on validation macro F1; test evaluated once. |
| Stage 2, this checkpoint | not independently evaluated | This package makes no accuracy claim for it. |
| Live demo output | per-image | A runtime prediction. Never a research metric. |

Earlier presentation figures (99.88% Stage 1, 99.43% Stage 2) belong to different
experiments and are **not** attributable to these checkpoints.

Built 2026-09-10 · ICT 4808 Group 08 · Rajarata University of Sri Lanka
