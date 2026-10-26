"""Phase 2 pipeline validation (sanity checks). Not reported as results."""

from __future__ import annotations

import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import resolve
from .data.audit import CLASSES
from .data.dataset import BatikDataset, load_labels, load_split
from .data.split import assign_splits, verify_split
from .data.transforms import denormalize, eval_transform, train_transform
from .models import build_model
from .utils import seed_everything, write_json


def check_labels(cfg) -> dict:
    classes = load_labels(cfg)
    train = load_split(cfg)
    mapping_ok = classes == [c for c in CLASSES if c in classes] and all(
        classes[l] == c for l, c in zip(train.label, train["class"])
    )
    present = {s: sorted(train[train.split == s]["class"].unique()) == sorted(classes) for s in ("train", "val", "test")}
    return {"label_mapping_consistent": bool(mapping_ok), "all_classes_in_every_split": present,
            "passed": bool(mapping_ok and all(present.values()))}


def check_loader(cfg) -> dict:
    frame = load_split(cfg, "train")
    root = resolve(cfg["paths"]["processed_dir"])
    one_per_class = frame.groupby("label").head(1).reset_index(drop=True)
    ds = BatikDataset(one_per_class, root, train_transform(cfg))
    x, y, _ = next(iter(DataLoader(ds, batch_size=len(ds))))
    ok = tuple(x.shape[1:]) == (3, 224, 224) and sorted(y.tolist()) == list(range(len(load_labels(cfg))))
    return {"batch_shape": list(x.shape), "labels": sorted(y.tolist()), "finite": bool(torch.isfinite(x).all()),
            "mean": float(x.mean()), "std": float(x.std()), "passed": bool(ok and torch.isfinite(x).all())}


def check_split_reproducible(cfg) -> dict:
    df = load_split(cfg)
    sp = cfg["data"]["split"]
    again = assign_splits(df.label.values, df.group_id.values, sp["seed"], sp["n_folds"], sp["fractions"])
    same = bool((again == df.split.values).all())
    checksum = verify_split(resolve(cfg["paths"]["manifests_dir"]))
    leaks = int(df.groupby("group_id").split.nunique().gt(1).sum())
    return {"resplit_identical": same, "manifest_checksum_ok": checksum, "groups_spanning_splits": leaks,
            "passed": same and checksum and leaks == 0}


def check_transforms(cfg) -> dict:
    frame = load_split(cfg, "val").head(1)
    ds = BatikDataset(frame, resolve(cfg["paths"]["processed_dir"]), None)
    img = ds.load_image(0)
    ev = eval_transform(cfg)
    deterministic_eval = bool(torch.equal(ev(img), ev(img)))
    tr = train_transform(cfg)
    random_train = not torch.equal(tr(img), tr(img)) if cfg["augmentation"]["enabled"] else None
    return {"eval_deterministic": deterministic_eval, "train_random": random_train,
            "passed": deterministic_eval and random_train is not False}


def augmentation_grid(cfg, out_path, n_classes: int = 10, n_aug: int = 6, seed: int = 0) -> None:
    torch.manual_seed(seed)
    frame = load_split(cfg, "train").groupby("label").head(1).reset_index(drop=True).head(n_classes)
    ds = BatikDataset(frame, resolve(cfg["paths"]["processed_dir"]), None)
    aug_cfg = copy.deepcopy(cfg)
    aug_cfg["augmentation"]["enabled"] = True
    ev, tr = eval_transform(cfg), train_transform(aug_cfg)
    fig, axes = plt.subplots(len(frame), n_aug + 1, figsize=(1.6 * (n_aug + 1), 1.6 * len(frame)))
    for r in range(len(frame)):
        img = ds.load_image(r)
        tiles = [ev(img)] + [tr(img) for _ in range(n_aug)]
        for c, t in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(denormalize(t, cfg).permute(1, 2, 0).numpy())
            ax.set_xticks([]), ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(frame.at[r, "class"], fontsize=7, rotation=0, ha="right", va="center")
        axes[0, 0].set_title("eval", fontsize=8)
        for c in range(1, n_aug + 1):
            axes[0, c].set_title(f"aug {c}", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def check_overfit(cfg, model_name: str, pretrained: bool, n_images: int = 32) -> dict:
    """A model that cannot fit 32 images has a pipeline bug, not a capacity problem."""
    seed_everything(0)
    c = copy.deepcopy(cfg)
    c["model"] = {"name": model_name, "pretrained": pretrained, "dropout": 0.0}
    frame = load_split(cfg, "train").groupby("label", group_keys=False).head(n_images // 10 + 1).head(n_images)
    ds = BatikDataset(frame.reset_index(drop=True), resolve(cfg["paths"]["processed_dir"]), eval_transform(cfg))
    loader = DataLoader(ds, batch_size=c["train"]["batch_size"], shuffle=True)
    model = build_model(c, len(load_labels(cfg)))
    opt = torch.optim.AdamW(model.parameters(), lr=c["train"]["lr"], weight_decay=0.0)
    ce = torch.nn.CrossEntropyLoss()
    losses = []
    for _ in range(c["train"]["epochs"]):
        model.train()
        for x, y, _ in loader:
            opt.zero_grad()
            loss = ce(model(x), y)
            loss.backward()
            opt.step()
        losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            acc = np.mean([(model(x).argmax(1) == y).float().mean().item() for x, y, _ in DataLoader(ds, batch_size=64)])
        if acc == 1.0:
            break
    return {"model": model_name, "pretrained": pretrained, "n_images": len(ds), "epochs_used": len(losses),
            "final_train_acc": float(acc), "first_loss": losses[0], "last_loss": losses[-1], "passed": bool(acc >= 0.95)}


def check_checkpoint_roundtrip(cfg, tmp_dir) -> dict:
    seed_everything(0)
    results = {}
    for name in ("custom_cnn", "efficientnet_b0"):
        c = copy.deepcopy(cfg)
        c["model"] = {"name": name, "pretrained": False, "dropout": 0.2}
        m = build_model(c).eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            ref = m(x)
        path = tmp_dir / f"roundtrip_{name}.pt"
        torch.save({"model": m.state_dict()}, path)
        m2 = build_model(c).eval()
        m2.load_state_dict(torch.load(path, weights_only=False)["model"])
        with torch.no_grad():
            results[name] = bool(torch.equal(ref, m2(x)))
        path.unlink()
    return {**results, "passed": all(results.values())}


def run_sanity(cfg) -> dict:
    out_dir = resolve("artifacts/sanity")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "labels": check_labels(cfg),
        "loader": check_loader(cfg),
        "split_reproducible": check_split_reproducible(cfg),
        "transforms": check_transforms(cfg),
        "checkpoint_roundtrip": check_checkpoint_roundtrip(cfg, out_dir),
    }
    augmentation_grid(cfg, resolve("figures") / "augmentation_grid.png")
    for name, pre in (("custom_cnn", False), ("efficientnet_b0", False), ("efficientnet_b0", True)):
        key = f"overfit_{name}{'_pretrained' if pre else ''}"
        report[key] = check_overfit(cfg, name, pre)
        print(f"[sanity] {key}: {report[key]}", flush=True)
    report["all_passed"] = all(v["passed"] for v in report.values() if isinstance(v, dict))
    write_json(report, out_dir / "sanity_report.json")
    print(json.dumps({k: bool(v["passed"]) if isinstance(v, dict) else bool(v) for k, v in report.items()}, indent=2))
    return report
