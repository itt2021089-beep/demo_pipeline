"""
Two-stage inference: ResNet50 gate -> EfficientNet-B0 species classifier.

    from demo.inference import run_pipeline
    result = run_pipeline("image.jpg")

Pipeline contract (from the research presentation):

    INPUT -> Stage 1 (Larva / Non-larva)
          -> if Non_larva: STOP, Stage 2 is not executed
          -> if Larva:     Stage 2 (Aedes / Anopheles / Culex / Non_larvae)

Both models are loaded once and cached. Nothing is hard-coded or faked: every
class name comes from demo/config.py, and every probability comes from a real
forward pass.
"""

import os
import threading
import time

import numpy as np

from demo import config
from demo.preprocessing import load_image, preprocess_stage1, preprocess_stage2

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

_S1 = None          # (model, device)
_S2 = None          # keras model
_LOCK = threading.Lock()


# ─────────────────────────────────────────────
# Loading (once)
# ─────────────────────────────────────────────
def load_stage1(device=None):
    global _S1
    with _LOCK:
        if _S1 is not None and device is None:
            return _S1
        import torch
        from stage1 import model as s1model

        dev = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        ckpt = config.STAGE1_CHECKPOINT
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Stage 1 checkpoint missing: {ckpt}")
        m = s1model.build_resnet50(num_classes=len(config.STAGE1_CLASSES),
                                   dropout=0.4, pretrained=False, device=dev)
        m.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=True))
        m.eval()
        if device is None:
            _S1 = (m, dev)
            return _S1
        return (m, dev)


def load_stage2():
    global _S2
    with _LOCK:
        if _S2 is not None:
            return _S2
        from tensorflow import keras
        path = config.STAGE2_MODEL
        if not os.path.exists(path):
            path = config.STAGE2_MODEL_FALLBACK
        if not os.path.exists(path):
            raise FileNotFoundError(f"Stage 2 model missing: {config.STAGE2_MODEL}")
        _S2 = keras.models.load_model(path, compile=False)
        return _S2


def warmup():
    """Load both models and run one dummy pass, so the UI's first real
    prediction is not dominated by lazy initialisation."""
    import torch
    m1, dev = load_stage1()
    with torch.no_grad():
        m1(torch.zeros(1, 3, config.STAGE1_INPUT_SIZE, config.STAGE1_INPUT_SIZE,
                       device=dev))
    m2 = load_stage2()
    m2.predict(np.zeros((1, config.STAGE2_INPUT_SIZE, config.STAGE2_INPUT_SIZE, 3),
                        dtype=np.float32), verbose=0)
    return True


# ─────────────────────────────────────────────
# Stages
# ─────────────────────────────────────────────
def predict_stage1(img, device=None):
    import torch
    model, dev = load_stage1(device)
    x = preprocess_stage1(img).to(dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        probs = torch.softmax(model(x).float(), dim=1)[0].cpu().numpy()
    ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    names = config.STAGE1_CLASSES
    return {
        "model": "ResNet50",
        "class": names[idx],
        "class_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "is_larva": names[idx] != config.GATE_CLASS,
        "inference_ms": ms,
    }


def predict_stage2(img):
    model = load_stage2()
    x = preprocess_stage2(img)
    t0 = time.perf_counter()
    probs = model.predict(x, verbose=0)[0]
    ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    names = config.STAGE2_CLASSES
    return {
        "model": "EfficientNet-B0",
        "class": names[idx],
        "class_index": idx,
        "confidence": float(probs[idx]),
        "probabilities": {n: float(p) for n, p in zip(names, probs)},
        "mapping_source_verified": config.STAGE2_MAPPING_SOURCE_VERIFIED,
        "inference_ms": ms,
    }


# ─────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────
def run_pipeline(source, device=None):
    """
    Returns:
        {
          "stage1": {...},
          "stage2": {...} or None,
          "stage2_executed": bool,
          "final_class": str,
          "final_confidence": float,
          "message": str,
          "total_ms": float,
        }
    """
    t0 = time.perf_counter()
    img = load_image(source)

    s1 = predict_stage1(img, device=device)

    if not s1["is_larva"]:
        # GATE: Stage 2 must not run.
        return {
            "stage1": s1, "stage2": None, "stage2_executed": False,
            "final_class": s1["class"], "final_confidence": s1["confidence"],
            "message": config.GATE_MESSAGE,
            "total_ms": (time.perf_counter() - t0) * 1000,
        }

    s2 = predict_stage2(img)
    return {
        "stage1": s1, "stage2": s2, "stage2_executed": True,
        "final_class": s2["class"], "final_confidence": s2["confidence"],
        "message": f"Larva detected — classified as {s2['class']}.",
        "total_ms": (time.perf_counter() - t0) * 1000,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("usage: python -m demo.inference <image> [<image> ...]")
        raise SystemExit(1)
    for p in sys.argv[1:]:
        r = run_pipeline(p)
        s1 = r["stage1"]
        print(f"\n{p}")
        print(f"  Stage 1: {s1['class']} ({s1['confidence']:.4f})")
        if r["stage2_executed"]:
            s2 = r["stage2"]
            print(f"  Stage 2: index {s2['class_index']} -> {s2['class']} "
                  f"({s2['confidence']:.4f})")
        else:
            print(f"  Stage 2: NOT EXECUTED")
        print(f"  FINAL  : {r['final_class']}  |  {r['message']}")
        print(f"  time   : {r['total_ms']:.1f} ms")
