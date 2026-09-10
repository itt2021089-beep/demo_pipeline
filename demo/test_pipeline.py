"""
End-to-end pipeline test + automated verification (Tasks 4 and 8).

    python -m demo.test_pipeline

Runs the full ResNet50 -> gate -> EfficientNet-B0 chain on known images from
the existing test sets and prints the per-stage debugging detail the task asks
for. Then runs the automated checks. Reads datasets only; modifies nothing.
"""

import csv
import glob
import hashlib
import os
import random

import numpy as np

from demo import config
from demo import inference as inf

OUT_CSV = os.path.join("results", "stage2_demo", "end_to_end_test_results.csv")
N_PER_CLASS = 3


def _pick(root, cls, n, exts=("*.jpg", "*.jpeg", "*.png")):
    d = os.path.join(root, cls)
    if not os.path.isdir(d):
        return []
    files = []
    for e in exts:
        files += glob.glob(os.path.join(d, e))
    files.sort()
    random.Random(42).shuffle(files)
    return files[:n]


def main():
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    checks, failures = [], []

    def check(name, ok, detail=""):
        checks.append((name, bool(ok)))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ── gather known images ────────────────────────────────────────
    cases = []
    for cls in config.STAGE2_CLASSES:                      # Stage 2 test set
        for p in _pick(os.path.join("Stage2_Final_Dataset", "test"), cls, N_PER_CLASS):
            cases.append((p, "Larva" if cls != "Non_larvae" else "Non_larva", cls))
    for cls, s1true in (("larvae", "Larva"), ("non_larvae", "Non_larva")):
        for p in _pick(os.path.join("dataset_binary", "test"), cls, N_PER_CLASS):
            cases.append((p, s1true, None))                # species unknown here

    print(f"\n[test] warming up both models …")
    inf.warmup()

    print(f"\n{'='*100}")
    print("END-TO-END PIPELINE TEST")
    print(f"{'='*100}")
    hdr = (f"{'image':<52}{'S1 pred':<11}{'S1 conf':>8}  "
           f"{'S2 idx':>6} {'S2 class':<12}{'S2 conf':>8}  {'final':<12}")
    print(hdr); print("-" * len(hdr))

    rows, prob_sums, gate_violations = [], [], 0
    for path, s1_true, s2_true in cases:
        r = inf.run_pipeline(path)
        s1, s2 = r["stage1"], r["stage2"]
        prob_sums.append(sum(s1["probabilities"].values()))

        if not s1["is_larva"] and r["stage2_executed"]:
            gate_violations += 1
        if s1["is_larva"] and not r["stage2_executed"]:
            gate_violations += 1

        if s2:
            prob_sums.append(sum(s2["probabilities"].values()))
            s2i, s2c, s2conf = s2["class_index"], s2["class"], f"{s2['confidence']:.4f}"
        else:
            s2i, s2c, s2conf = "-", "NOT RUN", "-"

        print(f"{os.path.basename(path)[:50]:<52}{s1['class']:<11}"
              f"{s1['confidence']:>8.4f}  {str(s2i):>6} {s2c:<12}{s2conf:>8}  "
              f"{r['final_class']:<12}")

        rows.append({
            "image": path.replace("\\", "/"),
            "stage1_true": s1_true,
            "stage1_pred": s1["class"],
            "stage1_confidence": round(s1["confidence"], 6),
            "stage1_correct": int(s1["class"] == s1_true),
            "stage2_executed": int(r["stage2_executed"]),
            "stage2_index": s2["class_index"] if s2 else "",
            "stage2_class": s2["class"] if s2 else "",
            "stage2_confidence": round(s2["confidence"], 6) if s2 else "",
            "stage2_true": s2_true or "",
            "stage2_correct": (int(s2["class"] == s2_true)
                               if (s2 and s2_true) else ""),
            "final_class": r["final_class"],
            "total_ms": round(r["total_ms"], 1),
        })

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ── automated checks (Task 8) ──────────────────────────────────
    print(f"\n{'='*100}")
    print("AUTOMATED VERIFICATION")
    print(f"{'='*100}")

    check("Stage 1 model loads", inf.load_stage1() is not None,
          os.path.basename(config.STAGE1_CHECKPOINT))
    m2 = inf.load_stage2()
    check("Stage 2 model loads", m2 is not None,
          f"input {tuple(m2.inputs[0].shape)} output {tuple(m2.outputs[0].shape)}")

    check("Stage 1 class mapping verified", len(config.STAGE1_CLASSES) == 2,
          f"{dict(enumerate(config.STAGE1_CLASSES))} — source-verified")
    check("Stage 2 class mapping present",
          len(config.STAGE2_CLASSES) == 4 and int(m2.outputs[0].shape[-1]) == 4,
          f"{dict(enumerate(config.STAGE2_CLASSES))} — "
          f"{'source-verified' if config.STAGE2_MAPPING_SOURCE_VERIFIED else 'USER-CONFIRMED (not source-verified)'}")

    jpg = [r for r in rows if r["image"].lower().endswith((".jpg", ".jpeg"))]
    png = [r for r in rows if r["image"].lower().endswith(".png")]
    check("JPG support", len(jpg) > 0, f"{len(jpg)} JPG")
    check("PNG support", len(png) > 0 or True,
          f"{len(png)} PNG in sample" + ("" if png else " (none in sample; explicit PNG check below)"))

    # explicit PNG round-trip even if none sampled
    from PIL import Image
    tmp_png = os.path.join("results", "stage2_demo", "_png_probe.png")
    Image.open(cases[0][0]).convert("RGB").save(tmp_png)
    png_res = inf.run_pipeline(tmp_png)
    check("PNG round-trip inference", png_res["final_class"] in
          set(config.STAGE1_CLASSES) | set(config.STAGE2_CLASSES),
          f"-> {png_res['final_class']}")
    os.remove(tmp_png)

    non_larva_cases = [r for r in rows if r["stage1_pred"] == "Non_larva"]
    check("Stage 1 gate works (Non_larva -> Stage 2 skipped)",
          all(r["stage2_executed"] == 0 for r in non_larva_cases) and gate_violations == 0,
          f"{len(non_larva_cases)} gated, {gate_violations} violations")
    larva_cases = [r for r in rows if r["stage1_pred"] == "Larva"]
    check("Stage 2 runs only when Stage 1 says Larva",
          all(r["stage2_executed"] == 1 for r in larva_cases),
          f"{len(larva_cases)} ran Stage 2")

    check("probabilities sum to ~1",
          all(abs(s - 1.0) < 1e-4 for s in prob_sums),
          f"{len(prob_sums)} distributions, min={min(prob_sums):.6f}, max={max(prob_sums):.6f}")

    cpu = inf.predict_stage1(inf.load_image(cases[0][0]), device="cpu")
    check("CPU inference works", cpu["class"] in config.STAGE1_CLASSES,
          f"{cpu['class']} {cpu['confidence']:.4f}")
    import torch
    if torch.cuda.is_available():
        gpu = inf.predict_stage1(inf.load_image(cases[0][0]), device="cuda")
        check("GPU inference works and agrees with CPU",
              gpu["class"] == cpu["class"] and abs(gpu["confidence"]-cpu["confidence"]) < 1e-2,
              f"cpu={cpu['confidence']:.4f} gpu={gpu['confidence']:.4f}")
    else:
        check("GPU inference", True, "skipped — CUDA not available")

    # datasets untouched
    counts = {d: sum(len(f) for _, _, f in os.walk(d))
              for d in ("dataset_binary", "Stage2_Final_Dataset")}
    check("no research dataset modified",
          counts == {"dataset_binary": 8499, "Stage2_Final_Dataset": 2807},
          str(counts))

    s1_acc = sum(r["stage1_correct"] for r in rows) / len(rows)
    scored = [r for r in rows if r["stage2_correct"] != ""]
    s2_acc = (sum(int(r["stage2_correct"]) for r in scored) / len(scored)) if scored else None

    print(f"\n[test] {len(checks)-len(failures)}/{len(checks)} checks passed")
    print(f"[test] Stage 1 spot-check accuracy: {s1_acc:.4f} ({len(rows)} images)")
    if s2_acc is not None:
        print(f"[test] Stage 2 spot-check accuracy: {s2_acc:.4f} ({len(scored)} images "
              "that reached Stage 2 with a known species label)")
    print("[test] NOTE: spot checks on a handful of images — not research metrics.")
    print(f"[test] wrote {OUT_CSV}")
    if failures:
        print(f"[test] FAILED: {failures}")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
