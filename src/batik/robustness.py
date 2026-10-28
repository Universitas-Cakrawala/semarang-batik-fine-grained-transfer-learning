"""Supplementary robustness evaluation.

Deterministic, evaluation-only corruptions of the *standard test split* at three
severities. Every image gets its own seeded RNG, so all models see exactly the
same corrupted pixels. Results support statements about robustness only for the
listed transformations -- not a general "fabric deformation" claim.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.transforms import v2

from .config import resolve
from .data.dataset import BatikDataset, load_labels, load_split
from .data.transforms import _fill, eval_transform
from .metrics import compute_metrics
from .models import build_model
from .train import predict
from .utils import pick_device, write_json

SEVERITIES = {
    "rotation": [15, 30, 45],                 # degrees, sign alternates per image
    "perspective": [0.10, 0.20, 0.30],        # torchvision distortion scale
    "elastic": [40.0, 80.0, 120.0],           # elastic alpha (sigma=6): non-rigid cloth-like warps
    "brightness": [0.25, 0.45, 0.65],         # +/- relative change, sign alternates
    "contrast": [0.25, 0.45, 0.65],
    "occlusion": [0.10, 0.20, 0.30],          # fraction of image area covered by a grey patch
    "blur": [1.0, 2.0, 3.0],                  # Gaussian sigma in cache pixels (~320 px short side)
    # Colour-shortcut probes: each class is one cloth with its own colourway, so a
    # large drop here means the model keys on colour/brightness rather than motif.
    "grayscale": [1.0],
    "grayscale_equalized": [1.0],
}


def corrupt(img: Image.Image, kind: str, severity: float, rng: np.random.Generator, fill) -> Image.Image:
    sign = 1 if rng.random() < 0.5 else -1
    if kind == "rotation":
        return F.rotate(img, sign * severity, interpolation=F.InterpolationMode.BILINEAR, fill=list(fill))
    if kind == "perspective":
        w, h = img.size
        dx, dy = severity * w / 2, severity * h / 2
        start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
        end = [[int(rng.uniform(0, dx)), int(rng.uniform(0, dy))],
               [int(w - 1 - rng.uniform(0, dx)), int(rng.uniform(0, dy))],
               [int(w - 1 - rng.uniform(0, dx)), int(h - 1 - rng.uniform(0, dy))],
               [int(rng.uniform(0, dx)), int(h - 1 - rng.uniform(0, dy))]]
        return F.perspective(img, start, end, interpolation=F.InterpolationMode.BILINEAR, fill=list(fill))
    if kind == "elastic":
        torch.manual_seed(int(rng.integers(0, 2**31 - 1)))
        return v2.ElasticTransform(alpha=severity, sigma=6.0, fill=list(fill))(img)
    if kind == "brightness":
        return ImageEnhance.Brightness(img).enhance(1 + sign * severity)
    if kind == "contrast":
        return ImageEnhance.Contrast(img).enhance(1 + sign * severity)
    if kind == "occlusion":
        w, h = img.size
        side = int((severity * w * h) ** 0.5)
        x0, y0 = int(rng.integers(0, w - side)), int(rng.integers(0, h - side))
        out = img.copy()
        ImageDraw.Draw(out).rectangle([x0, y0, x0 + side, y0 + side], fill=tuple(fill))
        return out
    if kind == "blur":
        return img.filter(ImageFilter.GaussianBlur(severity))
    if kind == "grayscale":
        return ImageOps.grayscale(img).convert("RGB")
    if kind == "grayscale_equalized":
        return ImageOps.equalize(ImageOps.grayscale(img)).convert("RGB")
    raise ValueError(kind)


class CorruptedDataset(Dataset):
    def __init__(self, base: BatikDataset, kind: str, severity: float, cfg: dict, seed: int = 2024):
        self.base, self.kind, self.severity, self.seed = base, kind, severity, seed
        self.tf, self.fill = eval_transform(cfg), _fill(cfg["preprocess"]["mean"])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        img = corrupt(self.base.load_image(i), self.kind, self.severity, rng, self.fill)
        return self.tf(img), int(self.base.frame.at[i, "label"]), i


def evaluate_run(run_dir: Path, cfg: dict, device: str | None = None, num_workers: int = min(4, os.cpu_count() or 1)) -> pd.DataFrame:
    dev = pick_device(device)
    classes = load_labels(cfg)
    state = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    run_cfg = state["config"]
    model = build_model(run_cfg, len(classes)).to(dev)
    model.load_state_dict(state["model"])
    base = BatikDataset(load_split(cfg, "test"), resolve(cfg["paths"]["processed_dir"]), None)

    def score(ds):
        loader = DataLoader(ds, batch_size=64, num_workers=num_workers)
        probs, labels, _, _ = predict(model, loader, dev, amp=False)
        return compute_metrics(probs, labels, classes)

    # Preprocessing comes from the run's own config (e.g. grayscale protocol G1g-G3g).
    clean = score(BatikDataset(base.frame, base.image_root, eval_transform(run_cfg)))
    rows = [{"corruption": "clean", "severity_level": 0, "severity": 0.0,
             "macro_f1": clean["macro_f1"], "accuracy": clean["accuracy"]}]
    for kind, levels in SEVERITIES.items():
        for lvl, sev in enumerate(levels, 1):
            m = score(CorruptedDataset(base, kind, sev, run_cfg))
            rows.append({"corruption": kind, "severity_level": lvl, "severity": sev,
                         "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]})
            print(f"[robust] {run_dir.name} {kind:<11} s{lvl} F1 {m['macro_f1']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df["relative_drop"] = 1 - df.macro_f1 / clean["macro_f1"]
    df.insert(0, "run_id", run_dir.name)
    df.to_csv(run_dir / "robustness.csv", index=False)
    write_json({"clean_macro_f1": clean["macro_f1"], "mean_corrupted_macro_f1": float(df.macro_f1[1:].mean())},
               run_dir / "robustness_summary.json")
    return df
