"""Single-image inference with a saved checkpoint (used by the demo)."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image, ImageOps

from .data.transforms import eval_transform
from .models import build_model


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu"):
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        self.cfg, self.classes = state["config"], state["classes"]
        self.model = build_model(self.cfg, len(self.classes)).to(device).eval()
        self.model.load_state_dict(state["model"])
        self.tf, self.device = eval_transform(self.cfg), device
        self.short = self.cfg["data"]["cache_short_side"]

    def prepare(self, img: Image.Image) -> Image.Image:
        # Same steps as the training cache: EXIF orientation, RGB, shorter side -> 320.
        img = ImageOps.exif_transpose(img).convert("RGB")
        scale = self.short / min(img.size)
        if scale < 1:
            img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
        return img

    @torch.no_grad()
    def __call__(self, img: Image.Image | str | Path, k: int = 3) -> list[tuple[str, float]]:
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        x = self.tf(self.prepare(img))[None].to(self.device)
        p = torch.softmax(self.model(x), 1)[0]
        top = torch.topk(p, k)
        return [(self.classes[i], float(v)) for v, i in zip(top.values, top.indices)]
