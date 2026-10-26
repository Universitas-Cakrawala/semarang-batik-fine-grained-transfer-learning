"""Dataset-validity diagnostics reported alongside the main results.

1. naive_split_leakage: how many test images would have a rotated/flipped copy
   of the same capture in training under the proposal's plain stratified split.
2. handcrafted_baselines: non-deep classifiers on colour / intensity / LBP texture
   histograms, on RGB and on the grayscale-equalised protocol. Every class in this
   snapshot is a single cloth with a distinctive colourway, so these measure how
   much of the task is solvable without learning motif shape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import resolve
from .data.dataset import load_labels, load_split
from .data.split import _d8, dihedral_distance
from .metrics import compute_metrics
from .utils import write_json


def naive_split_leakage(cfg: dict) -> dict:
    df = load_split(cfg)
    inv = pd.read_csv(resolve(cfg["paths"]["manifests_dir"]) / "inventory.csv").set_index("relpath")
    df = df.join(inv[["phash_d8", "datetime_original"]], on="relpath")
    tr, te = train_test_split(np.arange(len(df)), test_size=0.15, stratify=df.label, random_state=42)
    d8 = _d8(df.phash_d8)
    near = dihedral_distance(d8[te], d8[tr]).min(1) <= cfg["data"]["leakage"]["phash_max_distance"]
    key = df["class"] + "|" + df.datetime_original.astype(str)
    same_capture = key.iloc[te].isin(set(key.iloc[tr])).to_numpy()
    grp = df.group_id.iloc[te].isin(set(df.group_id.iloc[tr])).to_numpy()
    return {
        "procedure": "sklearn stratified 85/15 split, random_state=42 (proposal's plain split)",
        "n_test": int(len(te)),
        "test_with_rotated_or_flipped_copy_in_train": int(near.sum()),
        "test_with_same_capture_second_in_train": int(same_capture.sum()),
        "test_whose_near_duplicate_group_is_in_train": int(grp.sum()),
        "fraction_leaked": round(float(grp.mean()), 4),
    }


def _hsv_hist(im: Image.Image, bins=(16, 8, 8)) -> np.ndarray:
    hsv = np.asarray(im.convert("HSV")).reshape(-1, 3)
    h, _ = np.histogramdd(hsv, bins=bins, range=((0, 256), (0, 256), (0, 256)))
    return (h.ravel() / h.sum()).astype(np.float32)


def _gray_hist(im: Image.Image) -> np.ndarray:
    a = np.asarray(im.convert("L"))
    return (np.histogram(a, 32, (0, 256))[0] / a.size).astype(np.float32)


def _lbp_hist(im: Image.Image) -> np.ndarray:
    """Basic 8-neighbour local binary pattern histogram (texture only)."""
    a = np.asarray(im.convert("L"), dtype=np.float32)
    c, code = a[1:-1, 1:-1], np.zeros((a.shape[0] - 2, a.shape[1] - 2), dtype=np.int32)
    for k, (dy, dx) in enumerate([(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]):
        code |= (a[1 + dy:a.shape[0] - 1 + dy, 1 + dx:a.shape[1] - 1 + dx] >= c).astype(np.int32) << k
    return (np.bincount(code.ravel(), minlength=256) / code.size).astype(np.float32)


FEATURES = {
    "hsv_colour_histogram": ("RGB", _hsv_hist),
    "gray_intensity_histogram": ("RGB", _gray_hist),
    "lbp_texture": ("RGB", _lbp_hist),
    "gray_eq_intensity_histogram": ("GRAY_EQ", _gray_hist),
    "gray_eq_lbp_texture": ("GRAY_EQ", _lbp_hist),
}


def handcrafted_baselines(cfg: dict) -> dict:
    """Non-deep baselines (logistic regression, C chosen on val). They bound how much of
    the task is solvable by colour, global brightness, or local texture statistics."""
    root = resolve(cfg["paths"]["processed_dir"])
    classes = load_labels(cfg)
    images = {}
    for split in ("train", "val", "test"):
        f = load_split(cfg, split)
        ims = []
        for r in f.relpath:
            im = Image.open(root / f"{r.rsplit('.', 1)[0]}.jpg").convert("RGB")
            im.thumbnail((160, 160))
            ims.append(im)
        images[split] = (ims, f.label.values)
    out = {}
    for name, (view, fn) in FEATURES.items():
        prep = (lambda im: ImageOps.equalize(ImageOps.grayscale(im))) if view == "GRAY_EQ" else (lambda im: im)
        X = {s: np.stack([fn(prep(im)) for im in images[s][0]]) for s in images}
        y = {s: images[s][1] for s in images}
        best = None
        for C in (0.1, 1.0, 10.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000)).fit(X["train"], y["train"])
            val = compute_metrics(clf.predict_proba(X["val"]), y["val"], classes)["macro_f1"]
            if best is None or val > best[0]:
                best = (val, C, clf)
        test = compute_metrics(best[2].predict_proba(X["test"]), y["test"], classes)
        out[name] = {"input": view, "C": best[1], "val_macro_f1": best[0], "test_macro_f1": test["macro_f1"],
                     "test_accuracy": test["accuracy"]}
        print(f"[diag] {name:30s} val {best[0]:.4f} test {test['macro_f1']:.4f}", flush=True)
    return out


def run_diagnostics(cfg: dict) -> dict:
    out = {"naive_split_leakage": naive_split_leakage(cfg), "handcrafted_baselines": handcrafted_baselines(cfg)}
    write_json(out, resolve(cfg["paths"]["metrics_dir"]) / "dataset_diagnostics.json")
    return out
