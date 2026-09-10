# Two-stage mosquito larva classification — demo

```
INPUT IMAGE
    ↓
Stage 1 — ResNet50 (PyTorch)      Larva / Non-larva
    ↓  if Non-larva → STOP (Stage 2 is not executed)
    ↓  if Larva
Stage 2 — EfficientNet-B0 (Keras) Aedes / Anopheles / Culex / Non_larvae
    ↓
FINAL RESULT
```

## Run it

```bash
streamlit run demo/app.py
```

Command line, no UI:

```bash
python -m demo.inference path/to/image.jpg
```

Verification suite (end-to-end test + automated checks):

```bash
python -m demo.test_pipeline
```

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit UI — upload, Stage 1 card, Stage 2 card, final result, timings |
| `inference.py` | Model loading (cached), `predict_stage1`, `predict_stage2`, `run_pipeline` |
| `preprocessing.py` | Per-stage preprocessing — the two stages differ, see below |
| `config.py` | Model paths, class mappings **and their provenance** |
| `test_pipeline.py` | End-to-end test on known images + automated verification |

## Models

| | Stage 1 | Stage 2 |
|---|---|---|
| Architecture | ResNet50 | EfficientNet-B0 |
| Framework | PyTorch | Keras 3 / TensorFlow |
| Checkpoint | `demo_models/stage1_resnet50/best_model.pth` | `demo_models/stage2_efficientnet/best_tuning_run.h5` |
| Input | 224×224 RGB, NCHW | 224×224 RGB, NHWC |
| Output | 2 logits → softmax | Dense(4, softmax) |
| Classes | `0 Larva`, `1 Non_larva` | `0 Aedes`, `1 Anopheles`, `2 Culex`, `3 Non_larvae` |

## Preprocessing — the two stages are NOT the same

They share a 224×224 input size, which makes it tempting to reuse one tensor.
Doing so would be a silent bug.

**Stage 1** expects a normalised tensor:

```
gamma(1.5) → resize 224 → ToTensor → ImageNet normalise → NCHW
```

**Stage 2** expects **raw 0–255 pixels**:

```
resize 224 → float32 in [0, 255] → NHWC
```

The Keras model has `Rescaling(1/255) → Normalization → Rescaling(1/std)` baked
into its own graph (Keras' `efficientnet.preprocess_input`). Normalising before
handing it the array would double-normalise and quietly corrupt predictions.
`preprocessing.py` keeps these separate.

## The Stage 1 gate

When Stage 1 predicts `Non_larva`, `run_pipeline` returns immediately with
`stage2_executed = False` and the message:

> Non-larva detected — Stage 2 was not executed.

Stage 2 is never invoked in that path. This is enforced in code, not just in
the UI, and is covered by the automated checks.

## Class-mapping provenance — read this before citing results

| Stage | Mapping | Provenance |
|---|---|---|
| 1 | `0 Larva`, `1 Non_larva` | **Source-verified.** Derived from the training dataset's folder order (`larvae` < `non_larvae`), written at training time by `stage1/train.py`. |
| 2 | `0 Aedes`, `1 Anopheles`, `2 Culex`, `3 Non_larvae` | **User-confirmed, NOT source-verified.** |

The Keras training code that produced `best_tuning_run.h5` does not exist on
this machine, and the `.h5` stores only weights — no `class_indices`. The order
above was confirmed by the project owner. It is corroborated by a prediction
diagnostic that produced a clean one-to-one mapping over all four classes
(88–100% agreement) matching sorted folder order, and by both Keras loaders
indexing classes in sorted order by default — but corroboration is not proof.
Full detail: `results/stage2_demo/STAGE2_INTEGRATION_VERIFICATION.md`.

If the original notebook turns up, verify `train_generator.class_indices`
against `config.STAGE2_CLASSES` and set `STAGE2_MAPPING_SOURCE_VERIFIED = True`.

## Research integrity

The demo separates three different things, and the sidebar shows all three:

| | Value | What it is |
|---|---|---|
| Historical Stage 1 (presentation) | 99.88% | Earlier ResNet50 experiment on a split later found to leak. **Not this model.** |
| Historical Stage 2 (presentation) | 99.43% | **Cannot be linked to this checkpoint** — the `.h5` carries no run identifier, and the presentation describes that experiment as 3-species while this model has 4 outputs. |
| Current Stage 1 (this checkpoint) | 99.92% test accuracy | Measured on a verified leakage-free split (1,245 images). |
| Current Stage 2 (this checkpoint) | not independently evaluated | This demo makes no accuracy claim for it. |
| Live demo output | per-image | Runtime prediction only. Never a research metric. |

The demo never displays 99.88% or 99.43% as if they belonged to the running
models.

## Requirements

Already installed in the project venv: `torch`, `torchvision`, `tensorflow`,
`streamlit`, `pillow`, `numpy`.

Runs on CPU; uses CUDA for Stage 1 automatically when available. Both models
are loaded once and cached (`st.cache_resource` in the UI, a module-level cache
in `inference.py`), so repeated predictions do not reload weights.
