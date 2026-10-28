"""Dataset backed by the frozen split manifest and the processed image cache."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ..config import resolve
from ..utils import read_json, seed_worker
from .transforms import eval_transform, train_transform


def manifest_path(cfg: dict) -> Path:
    return resolve(cfg["paths"]["manifests_dir"]) / "split.csv"


def load_labels(cfg: dict) -> list[str]:
    return read_json(resolve(cfg["paths"]["manifests_dir"]) / "labels.json")["classes"]


def load_split(cfg: dict, split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(manifest_path(cfg))
    return df if split is None else df[df.split == split].reset_index(drop=True)


class BatikDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, transform: Callable | None):
        self.frame = frame.reset_index(drop=True)
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def load_image(self, i: int) -> Image.Image:
        rel = Path(self.frame.at[i, "relpath"]).with_suffix(".jpg")
        with Image.open(self.image_root / rel) as im:
            return im.convert("RGB")

    def __getitem__(self, i: int):
        img = self.load_image(i)
        x = self.transform(img) if self.transform else img
        return x, int(self.frame.at[i, "label"]), i


def build_loaders(cfg: dict, seed: int, subset: int | None = None) -> dict[str, DataLoader]:
    root = resolve(cfg["paths"]["processed_dir"])
    tc = cfg["train"]
    loaders = {}
    for split in ("train", "val", "test"):
        frame = load_split(cfg, split)
        if subset:
            frame = frame.groupby("label", group_keys=False).head(max(1, subset // 10)).reset_index(drop=True)
        is_train = split == "train"
        ds = BatikDataset(frame, root, train_transform(cfg) if is_train else eval_transform(cfg))
        g = torch.Generator()
        g.manual_seed(seed)
        loaders[split] = DataLoader(
            ds,
            batch_size=tc["batch_size"],
            shuffle=is_train,
            drop_last=False,
            num_workers=tc["num_workers"],
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
            persistent_workers=tc["num_workers"] > 0,
        )
    return loaders
