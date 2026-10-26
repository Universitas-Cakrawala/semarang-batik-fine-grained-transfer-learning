"""M1 Custom 4-block CNN and EfficientNet-B0 (random / ImageNet init)."""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tvm

EFFNET_WEIGHTS = "IMAGENET1K_V1"


class CustomCNN(nn.Module):
    """4 x [Conv3x3-BN-ReLU-MaxPool2] -> GAP -> Dropout -> Linear."""

    def __init__(self, num_classes: int = 10, channels=(32, 64, 128, 256), dropout: float = 0.3):
        super().__init__()
        layers, c_in = [], 3
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c_in, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def efficientnet_b0(num_classes: int, pretrained: bool, dropout: float = 0.2) -> nn.Module:
    model = tvm.efficientnet_b0(weights=EFFNET_WEIGHTS if pretrained else None, dropout=dropout)
    in_features = model.classifier[1].in_features
    head = nn.Linear(in_features, num_classes)
    # Same head init for scratch and pretrained so only the backbone init differs.
    nn.init.uniform_(head.weight, -1.0 / in_features**0.5, 1.0 / in_features**0.5)
    nn.init.zeros_(head.bias)
    model.classifier[1] = head
    return model


def build_model(cfg: dict, num_classes: int = 10) -> nn.Module:
    m = cfg["model"]
    if m["name"] == "custom_cnn":
        if m.get("pretrained"):
            raise ValueError("custom_cnn has no pretrained weights")
        return CustomCNN(num_classes, dropout=m.get("dropout", 0.3))
    if m["name"] == "efficientnet_b0":
        return efficientnet_b0(num_classes, bool(m.get("pretrained")), m.get("dropout", 0.2))
    raise ValueError(f"unknown model {m['name']}")


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def embed(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Penultimate (pooled) features, used for leakage audit and error analysis."""
    if isinstance(model, CustomCNN):
        return model.forward_features(x)
    return torch.flatten(model.avgpool(model.features(x)), 1)


# ---- two-stage fine-tuning helpers (G5, supplementary) -----------------------


def freeze_backbone(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False


def unfreeze_from(model: nn.Module, start: int) -> None:
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= start


def set_frozen_bn_eval(model: nn.Module) -> None:
    """Keep running statistics of fully-frozen blocks fixed (call after model.train())."""
    for block in model.features:
        if not any(p.requires_grad for p in block.parameters()):
            for m in block.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
