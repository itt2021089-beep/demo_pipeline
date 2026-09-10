"""
Demo configuration: model paths and class mappings.

Every mapping here records HOW it was established, because the two stages have
very different provenance and that difference matters for the research write-up.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


# ─────────────────────────────────────────────
# Stage 1 — ResNet50 (PyTorch)
# ─────────────────────────────────────────────
STAGE1_CHECKPOINT = _p("demo_models", "stage1_resnet50", "best_model.pth")
STAGE1_CONFIG = _p("demo_models", "stage1_resnet50", "model_config.json")
STAGE1_MAPPING_FILE = _p("demo_models", "stage1_resnet50", "class_mapping.json")

STAGE1_CLASSES = ["Larva", "Non_larva"]
STAGE1_MAPPING_PROVENANCE = (
    "SOURCE-VERIFIED. Derived from the training dataset's folder ordering "
    "(torchvision ImageFolder sorts alphabetically: 'larvae' < 'non_larvae'), "
    "written by stage1/train.py at training time and re-checked by "
    "stage1/demo_test.py."
)

STAGE1_INPUT_SIZE = 224
STAGE1_GAMMA = 1.5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ─────────────────────────────────────────────
# Stage 2 — EfficientNet-B0 (Keras)
# ─────────────────────────────────────────────
STAGE2_MODEL = _p("demo_models", "stage2_efficientnet", "best_tuning_run.h5")
STAGE2_MODEL_FALLBACK = _p("efficient net b0 trained.h5")   # byte-identical copy

STAGE2_CLASSES = ["Aedes", "Anopheles", "Culex", "Non_larvae"]
STAGE2_MAPPING_PROVENANCE = (
    "USER-CONFIRMED, NOT source-verified. The Keras training code that produced "
    "this checkpoint does not exist on this machine, and the .h5 stores only "
    "model_weights/optimizer_weights with no class_indices. The order "
    "0=Aedes, 1=Anopheles, 2=Culex, 3=Non_larvae was confirmed by the project "
    "owner on 2026-09-10. It is corroborated by a prediction diagnostic that "
    "produced a clean 1-to-1 mapping over all four classes (88-100% agreement) "
    "matching sorted folder order, and by the fact that both Keras loaders "
    "index classes in sorted order by default -- but corroboration is not "
    "proof. See results/stage2_demo/STAGE2_INTEGRATION_VERIFICATION.md."
)
STAGE2_MAPPING_SOURCE_VERIFIED = False

STAGE2_INPUT_SIZE = 224
# The model embeds Rescaling(1/255) -> Normalization -> Rescaling(1/std)
# (Keras efficientnet.preprocess_input). It must be fed RAW 0-255 RGB.
STAGE2_EXPECTS_RAW_0_255 = True

# ─────────────────────────────────────────────
# Pipeline behaviour
# ─────────────────────────────────────────────
GATE_CLASS = "Non_larva"          # Stage 1 result that halts the pipeline
GATE_MESSAGE = "Non-larva detected — Stage 2 was not executed."

# ─────────────────────────────────────────────
# Preprocessing description — must match what preprocessing.py ACTUALLY does.
# Nothing is listed here that is not implemented.
# ─────────────────────────────────────────────
PREPROCESSING_STEPS = {
    "Stage 1 — ResNet50 (PyTorch)": [
        ("Gamma correction", "γ = 1.5 (project's GammaCorrectionTransform)"),
        ("Resize", "224 × 224"),
        ("Tensor conversion", "ToTensor → float [0,1], NCHW"),
        ("Normalization", "ImageNet mean/std"),
    ],
    "Stage 2 — EfficientNet-B0 (Keras)": [
        ("Resize", "224 × 224"),
        ("Tensor conversion", "float32 array, NHWC"),
        ("Normalization", "none applied here — the model performs its own "
                          "rescaling/normalization internally"),
    ],
}

# ─────────────────────────────────────────────
# Research status — every claim below is traceable to a project file.
# Historical presentation figures are deliberately NOT surfaced as headline
# numbers, because they cannot be attributed to these checkpoints.
# ─────────────────────────────────────────────
LIVE_DEMO_STATUS = [
    {
        "stage": "Stage 1 — ResNet50",
        "status": "Trained model",
        "detail": "Loaded from demo_models/stage1_resnet50/best_model.pth. "
                  "Binary larva / non-larva classifier.",
    },
    {
        "stage": "Stage 2 — EfficientNet-B0",
        "status": "Trained model",
        "detail": "Loaded from demo_models/stage2_efficientnet/best_tuning_run.h5. "
                  "Four-class species classifier.",
    },
    {
        "stage": "Current prediction",
        "status": "Live inference",
        "detail": "Computed at request time from the uploaded image. It is a "
                  "single-image demonstration, not an evaluation metric.",
    },
]

# Only figures that exist in this project's documentation AND were measured on
# the exact checkpoint in use.
EVALUATION_STATUS = [
    {
        "model": "Stage 1 — ResNet50 (this checkpoint)",
        "result": "99.92% test accuracy",
        "context": "1,245 held-out test images, verified leakage-free split. "
                   "Checkpoint selected on validation macro F1; test evaluated "
                   "once afterwards.",
        "source": "results/stage1/resnet50/FINAL_STAGE1_REPORT.md",
        "verified": True,
    },
    {
        "model": "Stage 2 — EfficientNet-B0 (this checkpoint)",
        "result": "Not independently evaluated in this project",
        "context": "This demo does not re-evaluate the Keras model, and no "
                   "accuracy figure is claimed for it here.",
        "source": "results/stage2_demo/STAGE2_INTEGRATION_VERIFICATION.md",
        "verified": False,
    },
]
