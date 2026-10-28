"""Error analysis evidence for the final (validation-selected) model.

For each misclassified test image this produces the evidence a causal argument
needs, beyond a confusion matrix:
  * Grad-CAM of the predicted class (where the model looked),
  * nearest training images of the true and predicted class in the model's own
    embedding space (what it thinks the image resembles),
  * controlled input interventions -- full frame instead of centre crop,
    grayscale, horizontal flip, brightness normalised to the training mean,
    5-crop averaging -- and whether each one flips the prediction back.
    An intervention that fixes the error is evidence for the corresponding cause.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms as T

from .config import resolve
from .data.dataset import BatikDataset, load_labels, load_split
from .data.transforms import _prefix, denormalize, eval_transform
from .models import build_model, embed
from .utils import write_json


class GradCAM:
    def __init__(self, model, layer):
        self.model, self.acts, self.grads = model, None, None
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x: torch.Tensor, cls: int) -> np.ndarray:
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, cls].backward()
        w = self.grads.mean((2, 3), keepdim=True)
        cam = Fnn.relu((w * self.acts).sum(1, keepdim=True))
        cam = Fnn.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        return (cam / cam.max().clamp(min=1e-8)).detach().numpy()


def _interventions(cfg, train_brightness: float):
    p = cfg["preprocess"]
    norm = [T.ToTensor(), T.Normalize(p["mean"], p["std"])]
    ev = eval_transform(cfg)
    pre = _prefix(cfg)  # grayscale protocol runs see grayscale-equalised input everywhere
    full = T.Compose([*pre, T.Resize((p["crop"], p["crop"]), antialias=True), *norm])  # whole frame, aspect squashed

    def bright(img):
        cur = np.asarray(img.convert("L"), dtype=np.float32).mean()
        return ev(ImageEnhance.Brightness(img).enhance(train_brightness / max(cur, 1.0)))

    five = T.Compose([*pre, T.Resize(p["resize_short"], antialias=True), T.FiveCrop(p["crop"]),
                      T.Lambda(lambda cs: torch.stack([T.Compose(norm)(c) for c in cs]))])
    return {
        "full_frame": lambda im: full(im)[None],
        "grayscale": lambda im: ev(ImageOps.grayscale(im).convert("RGB"))[None],
        "hflip": lambda im: ev(ImageOps.mirror(im))[None],
        "brightness_norm": lambda im: bright(im)[None],
        "five_crop": lambda im: five(im),
    }


def analyse(cfg: dict, run_id: str, max_cases: int = 10, k_neighbours: int = 3) -> pd.DataFrame:
    run_dir = resolve(cfg["paths"]["runs_dir"]) / run_id
    out_dir = resolve("artifacts/error_analysis") / run_id
    fig_dir = resolve("figures/errors") / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    classes = load_labels(cfg)
    state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(state["config"], len(classes))
    model.load_state_dict(state["model"])
    model.eval()
    root = resolve(cfg["paths"]["processed_dir"])
    run_cfg = state["config"]
    ev = eval_transform(run_cfg)

    preds = pd.read_csv(run_dir / "predictions_test.csv")
    test = load_split(cfg, "test").set_index("image_id").loc[preds.image_id].reset_index()
    train = load_split(cfg, "train")
    inv = pd.read_csv(resolve(cfg["paths"]["manifests_dir"]) / "inventory.csv").set_index("relpath")
    train_brightness = float(inv.loc[train.relpath, "brightness"].mean())
    class_stats = inv.loc[train.relpath].groupby("class")[["brightness", "contrast", "sharpness"]].median()

    # Embeddings of the training set with the *final* model (for nearest neighbours).
    train_ds = BatikDataset(train, root, ev)
    with torch.no_grad():
        feats = []
        for i in range(0, len(train_ds), 64):
            x = torch.stack([train_ds[j][0] for j in range(i, min(i + 64, len(train_ds)))])
            feats.append(Fnn.normalize(embed(model, x), dim=1))
        train_emb = torch.cat(feats)

    cam = GradCAM(model, model.features[-1]) if hasattr(model, "features") else None
    interventions = _interventions(run_cfg, train_brightness)
    wrong = preds[~preds.correct].sort_values("confidence", ascending=False)
    pair_counts = wrong.groupby(["true_class", "pred_class"]).size().rename("pair_count")
    wrong = wrong.join(pair_counts, on=["true_class", "pred_class"])
    # Prefer variety: round-robin over confusion pairs (most confident case of each pair first),
    # frequent pairs first within a round.
    wrong["pair_rank"] = wrong.groupby(["true_class", "pred_class"]).cumcount()
    wrong = wrong.sort_values(["pair_rank", "pair_count", "confidence"],
                              ascending=[True, False, False]).head(max_cases)

    cases = []
    test_ds = BatikDataset(test, root, None)
    for n, (_, r) in enumerate(wrong.iterrows(), 1):
        i = int(test.index[test.image_id == r.image_id][0])
        img = test_ds.load_image(i)
        x = ev(img)[None]
        with torch.no_grad():
            q = Fnn.normalize(embed(model, x), dim=1)[0]
        sims = (train_emb @ q).numpy()
        nn_true = train.assign(sim=sims)[train.label == r.label].nlargest(k_neighbours, "sim")
        nn_pred = train.assign(sim=sims)[train.label == r.pred].nlargest(k_neighbours, "sim")

        effects = {}
        for name, fn in interventions.items():
            with torch.no_grad():
                pr = torch.softmax(model(fn(img)), 1).mean(0).numpy()
            effects[name] = {"pred_class": classes[int(pr.argmax())], "p_true": float(pr[r.label]),
                             "fixed": bool(pr.argmax() == r.label)}
        heat = cam(x.clone().requires_grad_(True), int(r.pred)) if cam else None
        stats = inv.loc[r.relpath, ["brightness", "contrast", "sharpness"]]
        case = {
            "case": n, "image_id": r.image_id, "relpath": r.relpath, "true_class": r.true_class,
            "pred_class": r.pred_class, "confidence": float(r.confidence),
            "p_true": float(r[f"p_{int(r.label)}"]), "top3": r.top3, "top3_probs": r.top3_probs,
            "confusion_pair_count_in_test": int(r.pair_count),
            "nn_true_sim": float(nn_true.sim.max()), "nn_pred_sim": float(nn_pred.sim.max()),
            "nn_true": nn_true.relpath.tolist(), "nn_pred": nn_pred.relpath.tolist(),
            "brightness": float(stats.brightness), "class_median_brightness": float(class_stats.loc[r.true_class, "brightness"]),
            "sharpness": float(stats.sharpness), "class_median_sharpness": float(class_stats.loc[r.true_class, "sharpness"]),
            "interventions": effects,
        }
        cases.append(case)
        _case_figure(case, img, x, heat, nn_true, nn_pred, root, cfg, fig_dir / f"case_{n:02d}.png")
        print(f"[errors] case {n}: {r.true_class} -> {r.pred_class} ({r.confidence:.2f}) "
              f"fixed_by={[k for k, v in effects.items() if v['fixed']]}", flush=True)

    write_json({"run_id": run_id, "n_test_errors": int((~preds.correct).sum()), "n_test": len(preds),
                "confusion_pairs": {f"{a} -> {b}": int(c) for (a, b), c in pair_counts.sort_values(ascending=False).items()},
                "cases": cases}, out_dir / "cases.json")
    flat = pd.json_normalize(cases, sep=".")
    flat.to_csv(out_dir / "cases.csv", index=False)
    return flat


def _case_figure(case, img, x, heat, nn_true, nn_pred, root, cfg, path) -> None:
    k = len(nn_true)
    fig = plt.figure(figsize=(12, 5.4))
    gs = fig.add_gridspec(2, 3 + k, width_ratios=[1.25, 1.25, 1.25] + [1] * k)
    ax = fig.add_subplot(gs[:, 0])
    ax.imshow(img)
    ax.set_title("full image (cache)", fontsize=8)
    crop = denormalize(x[0], cfg).permute(1, 2, 0).numpy()
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(crop)
    ax2.set_title("model input (centre crop)", fontsize=8)
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(crop)
    if heat is not None:
        ax3.imshow(heat, cmap="jet", alpha=0.45)
    ax3.set_title(f"Grad-CAM for '{case['pred_class']}'", fontsize=8)
    ax4 = fig.add_subplot(gs[:, 2])
    names, probs = case["top3"].split("|"), [float(p) for p in case["top3_probs"].split("|")]
    ax4.barh(range(3)[::-1], probs, color=["#B91C1C" if nm != case["true_class"] else "#047857" for nm in names])
    ax4.set_yticks(range(3)[::-1], names, fontsize=7)
    ax4.set_xlim(0, 1)
    fixed = [kk for kk, v in case["interventions"].items() if v["fixed"]] or ["none"]
    ax4.set_title(f"top-3 (p_true={case['p_true']:.2f})\nfixed by: {', '.join(fixed)}", fontsize=8)
    for row, (frame, label) in enumerate(((nn_true, "true"), (nn_pred, "pred"))):
        for j, (_, nb) in enumerate(frame.iterrows()):
            a = fig.add_subplot(gs[row, 3 + j])
            a.imshow(Image.open(root / Path(nb.relpath).with_suffix(".jpg")))
            a.set_title(f"{label}: {nb['class'][:14]}\ncos {nb.sim:.2f}", fontsize=7)
    for a in fig.axes:
        if a is not ax4:
            a.set_xticks([])
            a.set_yticks([])
    fig.suptitle(f"Case {case['case']}: true '{case['true_class']}' → predicted '{case['pred_class']}' "
                 f"(conf {case['confidence']:.2f}) — {case['relpath']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
