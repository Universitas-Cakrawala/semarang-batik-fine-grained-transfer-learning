"""Command-line entry point: `batik <command>` (or `python -m batik <command>`)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, resolve

BASE = "configs/base.yaml"
EXPERIMENTS = {
    "G1": "configs/experiments/G1_custom_cnn_scratch_aug.yaml",
    "G2": "configs/experiments/G2_effnetb0_scratch_aug.yaml",
    "G3": "configs/experiments/G3_effnetb0_pretrained_aug.yaml",
    "G4": "configs/experiments/G4_effnetb0_pretrained_noaug.yaml",
    "G5": "configs/experiments/G5_effnetb0_pretrained_twostage.yaml",
    "G1g": "configs/experiments/G1g_custom_cnn_scratch_aug_gray.yaml",
    "G2g": "configs/experiments/G2g_effnetb0_scratch_aug_gray.yaml",
    "G3g": "configs/experiments/G3g_effnetb0_pretrained_aug_gray.yaml",
}


def matrix_order(groups: list[str], seeds: list[int]) -> list[tuple[str, int]]:
    """Priority: G1-G3 all seeds, G4 seed 42, remaining G4 seeds,
    then the grayscale shortcut-controlled protocol (G1g-G3g), then G5."""
    order = [(g, s) for s in seeds for g in ("G1", "G2", "G3") if g in groups]
    if "G4" in groups:
        order += [("G4", seeds[0])] + [("G4", s) for s in seeds[1:]]
    order += [(g, s) for s in seeds for g in ("G1g", "G2g", "G3g") if g in groups]
    if "G5" in groups:
        order += [("G5", s) for s in seeds]
    return order


def cmd_download(a):
    from .data.download import download

    cfg = load_config(BASE)
    download(cfg["data"]["kaggle_ref"], cfg["data"]["kaggle_version"], resolve(cfg["paths"]["raw_dir"]),
             resolve(cfg["paths"]["manifests_dir"]), keep_zip=not a.delete_zip)


def cmd_audit(a):
    from .data.audit import run_audit
    from .data.eda import make_figures

    cfg = load_config(BASE)
    p = cfg["paths"]
    run_audit(resolve(p["raw_dir"]), resolve(p["processed_dir"]), resolve(p["manifests_dir"]),
              cfg["data"]["cache_short_side"], workers=a.workers)
    make_figures(resolve(p["manifests_dir"]) / "inventory.csv", resolve(p["processed_dir"]), resolve("figures"))


def cmd_split(a):
    from .data.split import make_split

    cfg = load_config(BASE)
    p = cfg["paths"]
    if (resolve(p["manifests_dir"]) / "split.csv").exists() and not a.force:
        sys.exit("split.csv already exists and is frozen. Use `batik verify-split`; "
                 "pass --force only to deliberately create a new split version.")
    make_split(resolve(p["manifests_dir"]) / "inventory.csv", resolve(p["processed_dir"]),
               resolve(p["manifests_dir"]), resolve("artifacts/cache"), cfg)


def cmd_verify(a):
    from .data.split import verify_split

    sys.exit(0 if verify_split(resolve(load_config(BASE)["paths"]["manifests_dir"])) else 1)


def cmd_sanity(a):
    from .sanity import run_sanity

    report = run_sanity(load_config("configs/sanity.yaml"))
    sys.exit(0 if report["all_passed"] else 1)


def cmd_diagnostics(a):
    import json

    from .diagnostics import run_diagnostics

    print(json.dumps(run_diagnostics(load_config(BASE)), indent=2))


def _overrides(a) -> dict:
    o: dict = {"train": {}}
    if a.epochs:
        o["train"]["epochs"] = a.epochs
    if a.num_workers is not None:
        o["train"]["num_workers"] = a.num_workers
    if a.batch_size:
        o["train"]["batch_size"] = a.batch_size
    return o


def cmd_train(a):
    from .train import run

    path = EXPERIMENTS.get(a.config, a.config)
    cfg = load_config(path, _overrides(a))
    if (a.subset or a.epochs or a.batch_size) and not a.tag:
        a.tag = "smoke"  # non-protocol runs are always tagged and never aggregated
    for seed in a.seeds or cfg["seeds"]:
        run(cfg, seed, tag=a.tag, subset=a.subset, device=a.device, overwrite=a.overwrite)


def _refuse_protocol_overrides(a, cmd: str) -> None:
    """Protocol runs keep the frozen epoch budget and batch size (they share run ids)."""
    if a.epochs or a.batch_size:
        sys.exit(f"`{cmd}` produces protocol runs; --epochs/--batch-size are not allowed. "
                 "Use `batik train --config G3 --epochs N` for a tagged smoke run instead.")


def cmd_matrix(a):
    from .train import run

    _refuse_protocol_overrides(a, "matrix")
    order = matrix_order(a.groups, a.seeds)
    print("[matrix] order:", ", ".join(f"{g}/s{s}" for g, s in order))
    for g, s in order:
        run(load_config(EXPERIMENTS[g], _overrides(a)), s, device=a.device)


def cmd_aggregate(a):
    import json

    from .analysis import aggregate, write_final_selection

    cfg = load_config(BASE)
    res = aggregate(cfg)
    if res["headline"]:
        res["final_model"] = write_final_selection(cfg)
    print(json.dumps(res["headline"], indent=2))


def cmd_robustness(a):
    import pandas as pd

    from .analysis import load_runs, select_final_run
    from .robustness import evaluate_run

    cfg = load_config(BASE)
    runs_dir = resolve(cfg["paths"]["runs_dir"])
    run_ids = a.runs or [select_final_run(cfg, g) for g in a.groups
                         if g in set(load_runs(runs_dir).group)]
    if not run_ids:
        sys.exit("no completed runs to evaluate")
    frames = [evaluate_run(runs_dir / r, cfg, device=a.device) for r in run_ids]
    df = pd.concat(frames)
    out = resolve(cfg["paths"]["metrics_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "robustness.csv", index=False)
    piv = df.pivot_table(index=["corruption", "severity_level"], columns="run_id", values="macro_f1", sort=False)
    lines = ["## Robustness: test Macro F1 (%) under evaluation-only corruptions", "",
             "| Corruption | Level | " + " | ".join(piv.columns) + " |", "|---|---:|" + "---:|" * len(piv.columns)]
    lines += [f"| {k} | {lvl} | " + " | ".join(f"{v * 100:.2f}" for v in row) + " |" for (k, lvl), row in piv.iterrows()]
    (out / "robustness.md").write_text("\n".join(lines) + "\n")
    print(piv.round(4).to_string())


def cmd_errors(a):
    """Error-analysis evidence for the final model. The brief needs >= 3 concrete test
    failures; the RGB model may be (near) perfect because of the colour shortcut, so we
    fall back to the final runs of G3g and then G1 until at least 3 failures are covered."""
    import pandas as pd

    from .analysis import load_runs, select_final_run
    from .errors import analyse

    cfg = load_config(BASE)
    if a.run:
        analyse(cfg, a.run, max_cases=a.max_cases)
        return
    runs_dir = resolve(cfg["paths"]["runs_dir"])
    available = set(load_runs(runs_dir).group)
    covered = 0
    for group in ("G3", "G3g", "G1"):
        if group not in available:
            continue
        run_id = select_final_run(cfg, group)
        n_err = int((~pd.read_csv(runs_dir / run_id / "predictions_test.csv").correct).sum())
        print(f"[errors] {group} final run {run_id}: {n_err} test errors")
        if n_err:
            analyse(cfg, run_id, max_cases=a.max_cases)
            covered += min(n_err, a.max_cases)
        if covered >= 3:
            break
    if covered < 3:
        print(f"[errors] WARNING: only {covered} test failures exist across G3/G3g/G1; "
              "use robustness failures (batik robustness) as additional evidence")


def cmd_release(a):
    """Copy the final model (best *validation* G3 run) to artifacts/release/ for the demo."""
    import shutil

    from .analysis import select_final_run
    from .utils import read_json, write_json

    cfg = load_config(BASE)
    run_id = a.run or select_final_run(cfg)
    src = resolve(cfg["paths"]["runs_dir"]) / run_id
    dst = resolve("artifacts/release")
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src / "best.pt", dst / "batik_effb0_final.pt")
    summary = read_json(src / "summary.json")
    write_json({"run_id": run_id, "selection_rule": "G3 run with highest validation Macro F1",
                "test_macro_f1": summary["test_macro_f1"], "best_epoch": summary["best_epoch"],
                "file": "batik_effb0_final.pt"}, dst / "release.json")
    print(f"[release] {run_id} -> {dst / 'batik_effb0_final.pt'}")


def cmd_predict(a):
    from .predict import Predictor

    pred = Predictor(resolve(a.checkpoint))
    for img in a.images:
        print(img, " | ".join(f"{c}: {p:.3f}" for c, p in pred(img)))


def cmd_reproduce(a):
    """Headline number: G3 test Macro F1 (mean ± std over seeds 42/43/44)."""
    import numpy as np
    import torch

    from .data.dataset import build_loaders, load_labels
    from .data.split import verify_split
    from .metrics import compute_metrics
    from .models import build_model
    from .train import predict, run, run_id_for
    from .utils import pick_device, read_json

    _refuse_protocol_overrides(a, "reproduce")
    cfg = load_config(EXPERIMENTS["G3"], _overrides(a))
    if not verify_split(resolve(cfg["paths"]["manifests_dir"])):
        sys.exit("split manifest checksum mismatch -- refusing to report a headline number")
    dev, classes, scores = pick_device(a.device), load_labels(cfg), []
    for seed in cfg["seeds"]:
        rid = run_id_for(cfg, seed)
        rdir = resolve(cfg["paths"]["runs_dir"]) / rid
        if a.retrain:
            scores.append(run(cfg, seed, device=a.device, overwrite=True)["test_macro_f1"])
            continue
        ckpt = rdir / "best.pt"
        if not ckpt.exists():
            sys.exit(f"missing {ckpt}: download the released checkpoints (README) or pass --retrain")
        state = torch.load(ckpt, map_location=dev, weights_only=False)
        model = build_model(state["config"], len(classes)).to(dev)
        model.load_state_dict(state["model"])
        probs, labels, _, _ = predict(model, build_loaders(cfg, seed)["test"], dev, amp=False)
        f1 = compute_metrics(probs, labels, classes)["macro_f1"]
        recorded = read_json(rdir / "summary.json")["test_macro_f1"]
        print(f"[reproduce] {rid}: test Macro F1 {f1:.4f} (recorded {recorded:.4f}, |Δ|={abs(f1 - recorded):.4f})")
        scores.append(f1)
    s = np.array(scores)
    print(f"\nHEADLINE  G3 EfficientNet-B0 (ImageNet, aug ON) test Macro F1 = "
          f"{s.mean() * 100:.2f} ± {s.std(ddof=1) * 100:.2f} % over seeds {cfg['seeds']}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="batik", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("download", help="download + checksum the Kaggle snapshot")
    s.add_argument("--delete-zip", action="store_true")
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("audit", help="inventory, validation, dedup, cache, EDA figures")
    s.add_argument("--workers", type=int, default=6)
    s.set_defaults(fn=cmd_audit)

    s = sub.add_parser("split", help="leakage-aware group-stratified split (refuses to overwrite)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=cmd_split)
    sub.add_parser("verify-split", help="check split manifest checksum").set_defaults(fn=cmd_verify)
    sub.add_parser("sanity", help="Phase 2 pipeline checks").set_defaults(fn=cmd_sanity)
    sub.add_parser("diagnostics", help="naive-split leakage + colour-histogram baseline").set_defaults(fn=cmd_diagnostics)

    def train_args(p):
        p.add_argument("--device", default=None)
        p.add_argument("--epochs", type=int, default=None, help="override (marks run as smoke)")
        p.add_argument("--batch-size", type=int, default=None)
        p.add_argument("--num-workers", type=int, default=None)

    s = sub.add_parser("train", help="train one config for one or more seeds")
    s.add_argument("--config", required=True, help="G1..G5 or a YAML path")
    s.add_argument("--seeds", type=int, nargs="*")
    s.add_argument("--tag", default=None)
    s.add_argument("--subset", type=int, default=None, help="images per split (smoke test)")
    s.add_argument("--overwrite", action="store_true")
    train_args(s)
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("matrix", help="run the experiment matrix in priority order (resumable)")
    s.add_argument("--groups", nargs="*", default=["G1", "G2", "G3", "G4", "G1g", "G2g", "G3g", "G5"])
    s.add_argument("--seeds", type=int, nargs="*", default=[42, 43, 44])
    train_args(s)
    s.set_defaults(fn=cmd_matrix)

    sub.add_parser("aggregate", help="tables, figures, final model selection").set_defaults(fn=cmd_aggregate)

    s = sub.add_parser("robustness", help="evaluation-only corruption benchmark")
    s.add_argument("--runs", nargs="*")
    s.add_argument("--groups", nargs="*", default=["G1", "G2", "G3", "G4", "G1g", "G2g", "G3g"])
    s.add_argument("--device", default=None)
    s.set_defaults(fn=cmd_robustness)

    s = sub.add_parser("errors", help="error-analysis evidence for the final model")
    s.add_argument("--run", default=None)
    s.add_argument("--max-cases", type=int, default=10)
    s.set_defaults(fn=cmd_errors)

    s = sub.add_parser("release", help="copy the final model to artifacts/release/")
    s.add_argument("--run", default=None)
    s.set_defaults(fn=cmd_release)

    s = sub.add_parser("predict", help="classify image files")
    s.add_argument("images", nargs="+")
    s.add_argument("--checkpoint", default="artifacts/release/batik_effb0_final.pt")
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("reproduce", help="reproduce the headline number")
    s.add_argument("--retrain", action="store_true", help="retrain G3 x 3 seeds instead of evaluating checkpoints")
    train_args(s)
    s.set_defaults(fn=cmd_reproduce)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
