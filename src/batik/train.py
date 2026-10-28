"""Training / evaluation engine for one (config, seed) run.

Protocol: AdamW, per-step cosine annealing over the whole
epoch budget, checkpoint = best validation Macro F1, no early stopping, and a
single test evaluation of that checkpoint after training finishes.
"""

from __future__ import annotations

import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .config import dump_config, resolve
from .data.dataset import build_loaders, load_labels
from .metrics import compute_metrics
from .models import build_model, count_parameters, freeze_backbone, set_frozen_bn_eval, unfreeze_from
from .utils import environment_info, pick_device, read_json, seed_everything, write_json


def run_id_for(cfg: dict, seed: int, tag: str | None = None) -> str:
    m = cfg["model"]
    model = {"custom_cnn": "customcnn", "efficientnet_b0": "effb0"}[m["name"]]
    init = "pt" if m.get("pretrained") else "scratch"
    aug = "augon" if cfg["augmentation"]["enabled"] else "augoff"
    strat = "-2stage" if cfg["train"]["strategy"] == "two_stage" else ""
    rid = f"{cfg.get('group', 'X')}_{model}-{init}{strat}_{aug}_s{seed}"
    return f"{tag}_{rid}" if tag else rid


# ---- optimisation -------------------------------------------------------------


class Stage:
    """One optimisation stage: which params train, optimiser, per-step cosine LR."""

    def __init__(self, model, lr: float, weight_decay: float, epochs: int, steps_per_epoch: int):
        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, epochs * steps_per_epoch)
        )


def stage_plan(cfg: dict) -> list[dict]:
    t = cfg["train"]
    if t["strategy"] == "single_stage":
        return [{"name": "full", "epochs": t["epochs"], "lr": t["lr"], "unfreeze_from": 0}]
    if t["strategy"] == "two_stage":
        ts = t["two_stage"]
        return [
            {"name": "linear_probe", "epochs": ts["probe_epochs"], "lr": ts["probe_lr"], "unfreeze_from": None},
            {
                "name": "finetune_top",
                "epochs": t["epochs"] - ts["probe_epochs"],
                "lr": ts["finetune_lr"],
                "unfreeze_from": ts["unfreeze_from"],
            },
        ]
    raise ValueError(t["strategy"])


def configure_stage(model, cfg: dict, stage: dict, steps_per_epoch: int) -> Stage:
    if cfg["train"]["strategy"] == "two_stage":
        if stage["unfreeze_from"] is None:
            freeze_backbone(model)
        else:
            unfreeze_from(model, stage["unfreeze_from"])
    return Stage(model, stage["lr"], cfg["train"]["weight_decay"], stage["epochs"], steps_per_epoch)


# ---- loops -----------------------------------------------------------------------


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    probs, labels, idxs, loss_sum = [], [], [], 0.0
    ce = nn.CrossEntropyLoss(reduction="sum")
    for x, y, i in loader:
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
        logits = logits.float()
        loss_sum += ce(logits, y).item()
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        labels.append(y.cpu().numpy())
        idxs.append(i.numpy())
    probs, labels, idxs = np.concatenate(probs), np.concatenate(labels), np.concatenate(idxs)
    return probs, labels, idxs, loss_sum / len(labels)


def train_one_epoch(model, loader, stage: Stage, scaler, device, amp: bool, cfg: dict, desc: str) -> dict:
    model.train()
    if cfg["train"]["strategy"] == "two_stage":
        set_frozen_bn_eval(model)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg["train"].get("label_smoothing", 0.0))
    loss_sum, correct, n = 0.0, 0, 0
    for x, y, _ in tqdm(loader, desc=desc, leave=False, disable=not os.isatty(2)):
        x = x.to(device, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(device, non_blocking=True)
        stage.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp):
            logits = model(x)
            loss = ce(logits, y)
        # Under AMP an fp16 overflow is expected occasionally: GradScaler skips that step.
        if not amp and not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss {loss.item()}")
        scaler.scale(loss).backward()
        scaler.step(stage.optimizer)
        scaler.update()
        stage.scheduler.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return {"train_loss": loss_sum / n, "train_acc": correct / n}


def predictions_frame(frame: pd.DataFrame, probs, labels, idxs, classes) -> pd.DataFrame:
    order = np.argsort(idxs)
    probs, labels, idxs = probs[order], labels[order], idxs[order]
    top = np.argsort(-probs, 1)[:, :3]
    out = pd.DataFrame(
        {
            "image_id": frame.loc[idxs, "image_id"].values,
            "relpath": frame.loc[idxs, "relpath"].values,
            "label": labels,
            "true_class": [classes[l] for l in labels],
            "pred": probs.argmax(1),
            "pred_class": [classes[p] for p in probs.argmax(1)],
            "confidence": probs.max(1),
            "top3": ["|".join(classes[j] for j in row) for row in top],
            "top3_probs": ["|".join(f"{probs[k, j]:.4f}" for j in row) for k, row in enumerate(top)],
        }
    )
    for j, c in enumerate(classes):
        out[f"p_{j}"] = probs[:, j]
    out["correct"] = out.label == out.pred
    return out


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---- one run ------------------------------------------------------------------------


def run(cfg: dict, seed: int, tag: str | None = None, subset: int | None = None, device: str | None = None,
        overwrite: bool = False) -> dict:
    run_id = run_id_for(cfg, seed, tag)
    out = resolve(cfg["paths"]["runs_dir"]) / run_id
    if (out / "summary.json").exists() and not overwrite:
        print(f"[train] {run_id} already complete -> skipping")
        return read_json(out / "summary.json")
    if overwrite:  # retrain from scratch: never resume a stale interrupted run
        for name in ("last.pt", "best.pt", "summary.json"):
            (out / name).unlink(missing_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    dev = pick_device(device)
    amp = cfg["train"]["amp"] == "auto" and dev.type == "cuda"
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)  # peak memory is reported per run
    classes = load_labels(cfg)
    loaders = build_loaders(cfg, seed, subset=subset)
    # channels_last is ~1.4x faster on both oneDNN (CPU) and cuDNN (tensor cores).
    model = build_model(cfg, len(classes)).to(dev, memory_format=torch.channels_last)
    scaler = torch.amp.GradScaler(dev.type, enabled=amp)
    stages = stage_plan(cfg)
    steps = len(loaders["train"])

    cfg_record = {**cfg, "seed": seed, "run_id": run_id, "subset": subset}
    dump_config(cfg_record, out / "config.yaml")
    env = environment_info(dev)
    env["amp"] = amp
    write_json(env, out / "env.json")
    params = count_parameters(model)
    counts = {s: len(l.dataset) for s, l in loaders.items()}
    print(f"[train] {run_id} | device={dev} amp={amp} | params={params['total']:,} | counts={counts}")

    # Resume state ---------------------------------------------------------------
    history, best, start_epoch, elapsed = [], {"val_macro_f1": -1.0, "val_loss": math.inf, "epoch": 0}, 0, 0.0
    last_ckpt = out / "last.pt"
    # Load on CPU: RNG states must stay CPU ByteTensors; load_state_dict moves the rest.
    resume_state = torch.load(last_ckpt, map_location="cpu", weights_only=False) if last_ckpt.exists() else None
    if resume_state:
        model.load_state_dict(resume_state["model"])
        history, best, start_epoch, elapsed = (
            resume_state["history"], resume_state["best"], resume_state["epoch"], resume_state["elapsed"]
        )
        print(f"[train] resuming {run_id} from epoch {start_epoch}")

    epoch = 0
    for stage_idx, stage_cfg in enumerate(stages):
        stage_start, stage_end = epoch, epoch + stage_cfg["epochs"]
        epoch = stage_end
        if start_epoch >= stage_end:
            continue
        stage = configure_stage(model, cfg, stage_cfg, steps)
        if resume_state:
            # Optimiser/scheduler belong to a stage; a crash right at a stage boundary
            # (two-stage G5) starts the next stage fresh, exactly as an uninterrupted run would.
            if resume_state["stage_idx"] == stage_idx:
                stage.optimizer.load_state_dict(resume_state["optimizer"])
                stage.scheduler.load_state_dict(resume_state["scheduler"])
            scaler.load_state_dict(resume_state["scaler"])
            torch.set_rng_state(resume_state["rng_cpu"])
            if torch.cuda.is_available() and resume_state.get("rng_cuda") is not None:
                torch.cuda.set_rng_state_all(resume_state["rng_cuda"])
            np.random.set_state(resume_state["rng_numpy"])
            loaders["train"].generator.set_state(resume_state["rng_loader"])
        for ep in range(max(stage_start, start_epoch), stage_end):
            t0 = time.time()
            tr = train_one_epoch(model, loaders["train"], stage, scaler, dev, amp, cfg,
                                 desc=f"{run_id} ep{ep + 1}")
            probs, labels, _, val_loss = predict(model, loaders["val"], dev, amp)
            vm = compute_metrics(probs, labels, classes)
            dt = time.time() - t0
            elapsed += dt
            row = {
                "epoch": ep + 1, "stage": stage_cfg["name"], "lr": stage.optimizer.param_groups[0]["lr"], **tr,
                "val_loss": val_loss, "val_macro_f1": vm["macro_f1"], "val_acc": vm["accuracy"],
                "val_top3": vm["top3_accuracy"], "epoch_seconds": dt,
            }
            history.append(row)
            improved = vm["macro_f1"] > best["val_macro_f1"] or (
                vm["macro_f1"] == best["val_macro_f1"] and val_loss < best["val_loss"]
            )
            if improved:
                best = {"val_macro_f1": vm["macro_f1"], "val_loss": val_loss, "epoch": ep + 1}
                _atomic_save({"model": model.state_dict(), "epoch": ep + 1, "classes": classes,
                              "config": cfg_record}, out / "best.pt")
            print(
                f"[train] {run_id} ep {ep + 1:02d}/{cfg['train']['epochs']} {stage_cfg['name']:<12} "
                f"loss {tr['train_loss']:.3f} acc {tr['train_acc']:.3f} | val loss {val_loss:.3f} "
                f"F1 {vm['macro_f1']:.4f} acc {vm['accuracy']:.4f} | {dt:.0f}s{' *' if improved else ''}",
                flush=True,
            )
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
            _atomic_save(
                {
                    "model": model.state_dict(), "optimizer": stage.optimizer.state_dict(),
                    "scheduler": stage.scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": ep + 1, "stage_idx": stage_idx, "history": history, "best": best, "elapsed": elapsed,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "rng_numpy": np.random.get_state(), "rng_loader": loaders["train"].generator.get_state(),
                },
                last_ckpt,
            )
        resume_state = None

    # Final evaluation of the validation-selected checkpoint (test touched once) ---
    state = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(state["model"])
    results = {}
    for split in ("val", "test"):
        probs, labels, idxs, loss = predict(model, loaders[split], dev, amp)
        m = compute_metrics(probs, labels, classes)
        m["loss"] = loss
        results[split] = m
        write_json(m, out / f"metrics_{split}.json")
        predictions_frame(loaders[split].dataset.frame, probs, labels, idxs, classes).to_csv(
            out / f"predictions_{split}.csv", index=False
        )

    peak_mem = (
        torch.cuda.max_memory_allocated(dev) / 2**20 if dev.type == "cuda"
        else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    )
    summary = {
        "run_id": run_id, "group": cfg.get("group"), "seed": seed, "model": cfg["model"]["name"],
        "pretrained": bool(cfg["model"].get("pretrained")), "augmentation": cfg["augmentation"]["enabled"],
        "strategy": cfg["train"]["strategy"], "config_source": cfg.get("_source"), "subset": subset,
        "git_commit": env["git_commit"], "device": env["device_name"] or env["device"],
        "params_total": params["total"], "counts": counts,
        "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
        "val_macro_f1": results["val"]["macro_f1"],
        "test_macro_f1": results["test"]["macro_f1"], "test_accuracy": results["test"]["accuracy"],
        "test_top3_accuracy": results["test"]["top3_accuracy"], "test_nll": results["test"]["nll"],
        "train_seconds": elapsed, "seconds_per_epoch": elapsed / max(1, len(history)),
        "peak_memory_mb": peak_mem, "checkpoint": os.path.relpath(out / "best.pt", resolve(".")),
    }
    write_json(summary, out / "summary.json")
    last_ckpt.unlink(missing_ok=True)  # resume state no longer needed; best.pt is the artefact
    print(f"[train] {run_id} DONE | best ep {best['epoch']} val F1 {best['val_macro_f1']:.4f} | "
          f"test F1 {summary['test_macro_f1']:.4f} acc {summary['test_accuracy']:.4f} | {elapsed / 60:.1f} min")
    return summary
