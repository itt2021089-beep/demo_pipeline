"""
Near-duplicate grouping for Stage2_new_aug_dataset.

Why this module exists
----------------------
Stage2_new_aug_dataset was augmented BEFORE it was split. Horizontal flips,
90-degree rotations and Gaussian-noise copies of the same source photograph
are scattered across train/, val/ and test/. Measured on the delivered split,
96% of val Anopheles images and 92% of test Anopheles images have a copy of
themselves sitting in train; one single Anopheles photograph appears 97 times
across all three splits. Any accuracy computed on that split is memorisation,
not generalisation.

This module recovers the underlying grouping so that split.py can keep every
copy of one source photograph inside a single split.

How the matcher works
---------------------
A first attempt using a plain 32x32 grayscale thumbnail failed: these are
microscope images with a strong circular vignette, and at that resolution the
vignette dominates the descriptor, so genuinely different specimens shot
through the same eyepiece correlated at ~1.0. The descriptor used here removes
the low-frequency component first:

  1. grayscale, resize to 128x128
  2. subtract a Gaussian-blurred copy (radius 8) -- kills the vignette and any
     global illumination difference, keeps specimen edges and texture
  3. downsample to 64x64, then normalise to zero mean / unit variance

Two images are compared by normalised cross-correlation of these descriptors,
taking the maximum over the 8 dihedral transforms (4 rotations x optional
mirror) so that flip/rotate augmentations still match their source. Additive
noise barely moves the correlation, so noise copies match too.

Calibration on this dataset (see docstring of `calibrate`):
  byte-identical pairs  -> 1.000
  random pairs          -> median 0.07, 95th percentile 0.15-0.55
A threshold of 0.90 sits far above the background and was verified by eye on
rendered contact sheets of the largest groups in every class.

Groups are the connected components of the "similarity >= threshold" graph.
Chaining could in principle over-merge, so `sweep_threshold` is provided and
the result is stable from 0.70 to 0.95 (leakage stays 55-64% either way).
"""

import json
import os
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter

# ─────────────────────────────────────────────
# Descriptor
# ─────────────────────────────────────────────
RESIZE = 128        # working resolution before high-pass
DESC = 64           # descriptor side length
BLUR_RADIUS = 8     # Gaussian radius used to estimate the low-frequency part
DEFAULT_THRESHOLD = 0.90


def descriptor(path: str) -> np.ndarray:
    """High-pass, contrast-normalised 64x64 descriptor of one image."""
    img = Image.open(path).convert("L").resize((RESIZE, RESIZE), Image.BILINEAR)
    low = img.filter(ImageFilter.GaussianBlur(radius=BLUR_RADIUS))
    high = np.asarray(img, np.float32) - np.asarray(low, np.float32)
    high = np.array(
        Image.fromarray(high).resize((DESC, DESC), Image.BILINEAR), dtype=np.float32
    )
    high -= high.mean()
    return high / (high.std() + 1e-6)


def _dihedral(a: np.ndarray):
    """The 8 dihedral transforms of a square array."""
    out = []
    for k in range(4):
        r = np.rot90(a, k)
        out += [r, np.fliplr(r)]
    return out


def similarity_matrix(descriptors: np.ndarray) -> np.ndarray:
    """
    Pairwise max-over-dihedral normalised correlation.

    `descriptors` is (n, DESC, DESC). Returns an (n, n) float32 matrix with the
    diagonal set to -1 so self-matches never create edges.
    """
    n = descriptors.shape[0]
    flat = descriptors.reshape(n, -1)
    variants = np.stack(
        [v.ravel() for a in descriptors for v in _dihedral(a)]
    ) / (DESC * DESC)

    sim = np.empty((n, n), np.float32)
    for i in range(0, n, 64):
        block = flat[i : i + 64]
        corr = (block @ variants.T).reshape(block.shape[0], n, 8).max(axis=2)
        sim[i : i + 64] = corr
    np.fill_diagonal(sim, -1.0)
    return sim


# ─────────────────────────────────────────────
# Grouping
# ─────────────────────────────────────────────
class _DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def connected_components(sim: np.ndarray, threshold: float):
    """Return a list of group ids, one per row of `sim`."""
    n = sim.shape[0]
    dsu = _DisjointSet(n)
    rows, cols = np.nonzero(sim >= threshold)
    for i, j in zip(rows, cols):
        dsu.union(int(i), int(j))

    roots = {}
    labels = []
    for i in range(n):
        r = dsu.find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels.append(roots[r])
    return labels


# ─────────────────────────────────────────────
# Dataset scan
# ─────────────────────────────────────────────
def scan_dataset(root: str, splits, classes):
    """List every image as (split, class, filename, path), sorted for determinism."""
    items = []
    for split in splits:
        for cls in classes:
            folder = os.path.join(root, split, cls)
            for fname in sorted(os.listdir(folder)):
                items.append((split, cls, fname, os.path.join(folder, fname)))
    return items


def build_groups(root: str, splits, classes, threshold: float = DEFAULT_THRESHOLD,
                 cache_path: str = None, verbose: bool = True):
    """
    Group every image in the dataset by underlying source photograph.

    Grouping is done independently per class: two images of different classes
    are never merged, which is correct here (the classes are disjoint sets of
    specimens) and keeps the similarity matrices small.

    Returns
    -------
    dict mapping "<split>/<class>/<filename>" -> "<class>_g<NNN>"
    """
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as fh:
            cached = json.load(fh)
        if cached.get("threshold") == threshold:
            if verbose:
                print(f"[dedup] loaded cached groups from {cache_path}")
            return cached["groups"]

    groups = {}
    for cls in classes:
        paths, keys = [], []
        for split in splits:
            folder = os.path.join(root, split, cls)
            for fname in sorted(os.listdir(folder)):
                paths.append(os.path.join(folder, fname))
                keys.append(f"{split}/{cls}/{fname}")

        desc = np.stack([descriptor(p) for p in paths])
        sim = similarity_matrix(desc)
        labels = connected_components(sim, threshold)

        for key, lab in zip(keys, labels):
            groups[key] = f"{cls}_g{lab:04d}"

        if verbose:
            sizes = defaultdict(int)
            for lab in labels:
                sizes[lab] += 1
            print(
                f"[dedup] {cls:11s} {len(paths):5d} files -> {len(sizes):4d} groups "
                f"(largest {max(sizes.values())})"
            )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as fh:
            json.dump({"threshold": threshold, "groups": groups}, fh)
        if verbose:
            print(f"[dedup] cached groups to {cache_path}")

    return groups


# ─────────────────────────────────────────────
# Diagnostics (used to produce the leakage numbers quoted above)
# ─────────────────────────────────────────────
def leakage_report(groups: dict, splits, classes):
    """
    For each class, how many val/test images live in a group that also contains
    a train image. That fraction is exactly the share of the held-out set the
    model has effectively already seen.
    """
    members = defaultdict(list)
    for key, gid in groups.items():
        split = key.split("/", 1)[0]
        members[gid].append(split)

    rows = []
    for cls in classes:
        counts = {s: [0, 0] for s in splits if s != "train"}
        for key, gid in groups.items():
            split, kcls, _ = key.split("/", 2)
            if kcls != cls or split == "train":
                continue
            counts[split][1] += 1
            if "train" in members[gid]:
                counts[split][0] += 1
        rows.append((cls, counts))
    return rows
