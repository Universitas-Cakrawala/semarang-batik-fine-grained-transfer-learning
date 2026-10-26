"""Train / eval transforms."""

from __future__ import annotations

from PIL import ImageOps
from torchvision import transforms as T


class GrayEqualize:
    """Shortcut-controlled protocol: luminance only + histogram equalisation,
    replicated to 3 channels so architectures and ImageNet weights are unchanged."""

    def __call__(self, img):
        return ImageOps.equalize(ImageOps.grayscale(img)).convert("RGB")

    def __repr__(self):
        return "GrayEqualize()"


def _prefix(cfg: dict) -> list:
    return [GrayEqualize()] if cfg["preprocess"].get("grayscale_equalize") else []


def _fill(mean) -> tuple[int, int, int]:
    # Pixels exposed by the affine warp take the dataset-normalisation mean, so
    # they become ~0 after Normalize instead of an out-of-distribution black.
    return tuple(int(round(m * 255)) for m in mean)


def eval_transform(cfg: dict) -> T.Compose:
    p = cfg["preprocess"]
    return T.Compose(
        _prefix(cfg)
        + [
            T.Resize(p["resize_short"], interpolation=T.InterpolationMode.BILINEAR, antialias=True),
            T.CenterCrop(p["crop"]),
            T.ToTensor(),
            T.Normalize(p["mean"], p["std"]),
        ]
    )


def train_transform(cfg: dict) -> T.Compose:
    if not cfg["augmentation"]["enabled"]:
        return eval_transform(cfg)
    p, a = cfg["preprocess"], cfg["augmentation"]
    rrc, aff, cj = a["random_resized_crop"], a["affine"], a["color_jitter"]
    return T.Compose(
        _prefix(cfg)
        + [
            T.RandomResizedCrop(
                p["crop"], scale=tuple(rrc["scale"]), ratio=tuple(rrc["ratio"]), antialias=True
            ),
            T.RandomHorizontalFlip(a["hflip_p"]),
            T.RandomAffine(
                degrees=aff["degrees"],
                translate=(aff["translate"], aff["translate"]),
                scale=tuple(aff["scale"]),
                shear=aff["shear"],
                interpolation=T.InterpolationMode.BILINEAR,
                fill=_fill(p["mean"]),
            ),
            T.ColorJitter(**cj),
            T.ToTensor(),
            T.Normalize(p["mean"], p["std"]),
        ]
    )


def denormalize(x, cfg: dict):
    import torch

    mean = torch.tensor(cfg["preprocess"]["mean"]).view(-1, 1, 1)
    std = torch.tensor(cfg["preprocess"]["std"]).view(-1, 1, 1)
    return (x * std + mean).clamp(0, 1)
