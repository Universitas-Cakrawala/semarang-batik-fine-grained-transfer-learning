"""Phase 4: aggregate run summaries into comparison/ablation tables and figures."""

from __future__ import annotations

import itertools

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import resolve
from .metrics import paired_bootstrap
from .utils import read_json, write_json

GROUP_LABELS = {
    "G1": "M1 Custom CNN, scratch, aug ON",
    "G2": "M2 EfficientNet-B0, scratch, aug ON",
    "G3": "M3 EfficientNet-B0, ImageNet, aug ON",
    "G4": "M3 EfficientNet-B0, ImageNet, aug OFF",
    "G5": "EfficientNet-B0, ImageNet, two-stage (suppl.)",
    "G1g": "M1 Custom CNN, scratch, aug ON, grayscale-eq",
    "G2g": "M2 EfficientNet-B0, scratch, aug ON, grayscale-eq",
    "G3g": "M3 EfficientNet-B0, ImageNet, aug ON, grayscale-eq",
}
COMPARISONS = [  # (a, b, question answered)
    ("G3", "G1", "Transfer learning vs from-scratch baseline (headline)"),
    ("G2", "G1", "Architecture effect (both random init)"),
    ("G3", "G2", "Pretrained-initialisation effect (same architecture)"),
    ("G3", "G4", "Ablation: augmentation ON vs OFF"),
    ("G5", "G3", "Training strategy: two-stage vs full fine-tuning (suppl.)"),
    ("G3g", "G1g", "Grayscale protocol: transfer learning vs from-scratch baseline"),
    ("G2g", "G1g", "Grayscale protocol: architecture effect"),
    ("G3g", "G2g", "Grayscale protocol: pretrained-initialisation effect"),
    ("G3", "G3g", "Cost of removing colour for the transfer model"),
]


def load_runs(runs_dir) -> pd.DataFrame:
    rows = [read_json(p) for p in sorted(runs_dir.glob("*/summary.json"))]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"no completed runs in {runs_dir}")
    df = df[df.subset.isna()] if "subset" in df else df  # smoke runs never enter tables
    # Protocol run ids start with the group (G1_, G3g_, ...); tagged runs (smoke_, pilot_, ...) never enter tables.
    return df[df.run_id.str.match(r"^G\d+g?_")].sort_values(["group", "seed"]).reset_index(drop=True)


def fmt(mean: float, std: float, pct: bool = True) -> str:
    k = 100 if pct else 1
    return f"{mean * k:.2f} ± {std * k:.2f}" if not np.isnan(std) else f"{mean * k:.2f}"


def aggregate(cfg: dict) -> dict:
    runs_dir = resolve(cfg["paths"]["runs_dir"])
    out = resolve(cfg["paths"]["metrics_dir"])
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(runs_dir)
    cols = ["run_id", "group", "seed", "params_total", "best_epoch", "best_val_macro_f1", "test_macro_f1",
            "test_accuracy", "test_top3_accuracy", "train_seconds", "seconds_per_epoch", "peak_memory_mb", "device"]
    runs[cols].to_csv(out / "all_runs.csv", index=False)

    metrics = ["test_macro_f1", "test_accuracy", "test_top3_accuracy", "best_val_macro_f1"]
    g = runs.groupby("group")
    agg = g[metrics].agg(["mean", "std"])
    agg[("n_seeds", "")] = g.size()
    agg[("train_minutes", "mean")] = g.train_seconds.mean() / 60
    agg[("params", "")] = g.params_total.first()
    agg.to_csv(out / "group_summary.csv")

    comps = []
    for a, b, question in COMPARISONS:
        if a not in agg.index or b not in agg.index:
            continue
        pa, pb = runs[runs.group == a], runs[runs.group == b]
        seeds = sorted(set(pa.seed) & set(pb.seed))
        boots = []
        for s in seeds:
            da = pd.read_csv(runs_dir / pa[pa.seed == s].run_id.iloc[0] / "predictions_test.csv").sort_values("image_id")
            db = pd.read_csv(runs_dir / pb[pb.seed == s].run_id.iloc[0] / "predictions_test.csv").sort_values("image_id")
            assert (da.image_id.values == db.image_id.values).all(), "test sets differ"
            boots.append(paired_bootstrap(da.label.values, da.pred.values, db.pred.values, 10, seed=s))
        diff = [pa[pa.seed == s].test_macro_f1.iloc[0] - pb[pb.seed == s].test_macro_f1.iloc[0] for s in seeds]
        comps.append({
            "a": a, "b": b, "question": question, "paired_seeds": seeds,
            "mean_diff_pp": float(np.mean(diff) * 100) if diff else None,
            "std_diff_pp": float(np.std(diff, ddof=1) * 100) if len(diff) > 1 else None,
            "per_seed_bootstrap": boots,
            "all_seed_ci_exclude_zero": bool(boots) and all(bt["ci95_low"] > 0 or bt["ci95_high"] < 0 for bt in boots),
        })

    headline = None
    if "G3" in agg.index:
        g3 = runs[runs.group == "G3"]
        headline = {
            "description": "G3 EfficientNet-B0 ImageNet + aug ON, test Macro F1 (mean ± std over seeds)",
            "test_macro_f1_mean": float(g3.test_macro_f1.mean()),
            "test_macro_f1_std": float(g3.test_macro_f1.std(ddof=1)) if len(g3) > 1 else None,
            "seeds": g3.seed.tolist(),
            "per_seed": dict(zip(g3.seed.astype(str), g3.test_macro_f1.round(6))),
        }
        if "G1" in agg.index:
            g1 = runs[runs.group == "G1"].test_macro_f1.mean()
            headline["target_macro_f1_ge_0.85"] = bool(g3.test_macro_f1.mean() >= 0.85)
            headline["gain_over_baseline_pp"] = float((g3.test_macro_f1.mean() - g1) * 100)
            headline["target_gain_ge_10pp"] = bool((g3.test_macro_f1.mean() - g1) * 100 >= 10)

    result = {"headline": headline, "comparisons": comps, "n_runs": int(len(runs)),
              "groups": {k: int(v) for k, v in g.size().items()}}
    write_json(result, out / "summary.json")
    write_markdown_tables(runs, agg, comps, out / "tables.md")
    plot_curves(runs, runs_dir, resolve("figures") / "training_curves.png")
    plot_group_bars(runs, resolve("figures") / "group_macro_f1.png")
    return result


def write_markdown_tables(runs, agg, comps, path) -> None:
    lines = ["## Comparison of all run groups (test split, mean ± std over seeds)", "",
             "| Group | Configuration | Params | Seeds | Test Macro F1 (%) | Test Acc (%) | Test Top-3 (%) | Val Macro F1 (%) | Train time / run (min) |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for grp, r in agg.iterrows():
        lines.append(
            f"| {grp} | {GROUP_LABELS.get(grp, grp)} | {int(r[('params', '')]):,} | {int(r[('n_seeds', '')])} | "
            f"{fmt(r[('test_macro_f1', 'mean')], r[('test_macro_f1', 'std')])} | "
            f"{fmt(r[('test_accuracy', 'mean')], r[('test_accuracy', 'std')])} | "
            f"{fmt(r[('test_top3_accuracy', 'mean')], r[('test_top3_accuracy', 'std')])} | "
            f"{fmt(r[('best_val_macro_f1', 'mean')], r[('best_val_macro_f1', 'std')])} | "
            f"{r[('train_minutes', 'mean')]:.1f} |"
        )
    lines += ["", "## Every run", "", "| Run | Seed | Best epoch | Val Macro F1 | Test Macro F1 | Test Acc | Test Top-3 | Train min | Device |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for _, r in runs.iterrows():
        lines.append(f"| {r.run_id} | {r.seed} | {r.best_epoch} | {r.best_val_macro_f1:.4f} | {r.test_macro_f1:.4f} | "
                     f"{r.test_accuracy:.4f} | {r.test_top3_accuracy:.4f} | {r.train_seconds / 60:.1f} | {r.device} |")
    lines += ["", "## Paired comparisons (Δ Macro F1 = A − B, percentage points)", "",
              "| A vs B | Question | Seeds | Mean Δ ± std (pp) | Per-seed 95% bootstrap CI (pp) |", "|---|---|---|---:|---|"]
    for c in comps:
        cis = "; ".join(f"s{s}: [{b['ci95_low'] * 100:.1f}, {b['ci95_high'] * 100:.1f}]"
                        for s, b in zip(c["paired_seeds"], c["per_seed_bootstrap"]))
        if c["mean_diff_pp"] is None:
            continue  # no paired seeds yet
        sd = f" ± {c['std_diff_pp']:.2f}" if c["std_diff_pp"] is not None else ""
        lines.append(f"| {c['a']} vs {c['b']} | {c['question']} | {','.join(map(str, c['paired_seeds']))} | "
                     f"{c['mean_diff_pp']:.2f}{sd} | {cis} |")
    path.write_text("\n".join(lines) + "\n")


def plot_curves(runs, runs_dir, path) -> None:
    groups = sorted(runs.group.unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = dict(zip(groups, plt.cm.tab10.colors))
    for _, r in runs.iterrows():
        h = pd.read_csv(runs_dir / r.run_id / "history.csv")
        axes[0].plot(h.epoch, h.train_loss, color=colors[r.group], alpha=0.7, lw=1)
        axes[0].plot(h.epoch, h.val_loss, color=colors[r.group], alpha=0.7, lw=1, ls="--")
        axes[1].plot(h.epoch, h.val_macro_f1, color=colors[r.group], alpha=0.8, lw=1,
                     label=f"{r.group}" if r.seed == runs[runs.group == r.group].seed.min() else None)
    axes[0].set(title="Loss (solid = train, dashed = val)", xlabel="epoch", ylabel="cross-entropy", yscale="log")
    axes[1].set(title="Validation Macro F1", xlabel="epoch", ylabel="Macro F1", ylim=(0, 1))
    axes[1].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_group_bars(runs, path) -> None:
    groups = sorted(runs.group.unique())
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for i, grp in enumerate(groups):
        v = runs[runs.group == grp].test_macro_f1.values * 100
        ax.bar(i, v.mean(), color="#0E7490", alpha=0.85, width=0.6)
        ax.scatter(np.full(len(v), i) + np.linspace(-0.12, 0.12, len(v)), v, color="#17324D", s=14, zorder=3)
    ax.axhline(85, color="#B45309", ls="--", lw=1, label="target 85%")
    ax.set_xticks(range(len(groups)), groups)
    ax.set(ylabel="Test Macro F1 (%)", ylim=(0, 100), title="Test Macro F1 per run group (bars = mean, dots = seeds)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(metrics: dict, classes: list[str], path, title: str) -> None:
    cm = np.array(metrics["confusion_matrix"])
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i, j in itertools.product(range(len(classes)), repeat=2):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "#17324D")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(classes)), classes, fontsize=7)
    ax.set(xlabel="predicted", ylabel="true", title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def select_final_run(cfg: dict, group: str = "G3") -> str:
    """Final model = the run of `group` with the best *validation* Macro F1 (never test)."""
    runs = load_runs(resolve(cfg["paths"]["runs_dir"]))
    g = runs[runs.group == group]
    return g.sort_values(["best_val_macro_f1", "seed"], ascending=[False, True]).run_id.iloc[0]


def write_final_selection(cfg: dict) -> dict:
    run_id = select_final_run(cfg)
    run_dir = resolve(cfg["paths"]["runs_dir"]) / run_id
    m = read_json(run_dir / "metrics_test.json")
    classes = list(m["per_class"].keys())
    plot_confusion(m, classes, resolve("figures") / "confusion_matrix_final.png", f"Test confusion matrix — {run_id}")
    sel = {"final_run": run_id, "rule": "G3 run with highest validation Macro F1", "test_macro_f1": m["macro_f1"],
           "per_class_f1": {c: v["f1"] for c, v in m["per_class"].items()}}
    write_json(sel, resolve(cfg["paths"]["metrics_dir"]) / "final_model.json")
    return sel
