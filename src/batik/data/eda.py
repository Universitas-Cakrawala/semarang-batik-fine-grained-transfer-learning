"""EDA figures for the data card: class balance, samples per class, native sizes."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from .audit import CLASSES


def make_figures(inventory_csv: Path, processed_dir: Path, fig_dir: Path, per_class: int = 6, seed: int = 0) -> None:
    fig_dir.mkdir(parents=True, exist_ok=True)
    inv = pd.read_csv(inventory_csv)
    inc = inv[inv.included]

    counts = inc["class"].value_counts().reindex(CLASSES, fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(range(len(counts)), counts.values, color="#0E7490")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontsize=7)
    ax.set_xticks(range(len(counts)), counts.index, rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="images", title=f"Class distribution after audit (n={len(inc)})", ylim=(0, counts.max() * 1.15))
    fig.tight_layout()
    fig.savefig(fig_dir / "class_distribution.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(CLASSES), per_class, figsize=(1.5 * per_class, 1.5 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        sample = inc[inc["class"] == cls].sample(min(per_class, (inc["class"] == cls).sum()), random_state=seed)
        for c in range(per_class):
            ax = axes[r, c]
            ax.set_xticks([]), ax.set_yticks([])
            if c < len(sample):
                ax.imshow(Image.open(processed_dir / Path(sample.relpath.iloc[c]).with_suffix(".jpg")))
        axes[r, 0].set_ylabel(cls, fontsize=7, rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(fig_dir / "samples_per_class.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    for cls in CLASSES:
        s = inc[inc["class"] == cls]
        ax.scatter(s.oriented_width, s.oriented_height, s=6, alpha=0.6, label=cls)
    ax.set(xlabel="width (px, EXIF-oriented)", ylabel="height (px)", title="Native image sizes")
    ax.legend(fontsize=6, markerscale=2)
    fig.tight_layout()
    fig.savefig(fig_dir / "native_sizes.png", dpi=130)
    plt.close(fig)
