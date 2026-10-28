"""Classification metrics shared by training, evaluation and analysis."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support


def topk_accuracy(probs: np.ndarray, labels: np.ndarray, k: int) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == labels[:, None]).any(axis=1).mean())


def compute_metrics(probs: np.ndarray, labels: np.ndarray, class_names: list[str]) -> dict:
    preds = probs.argmax(1)
    n = len(class_names)
    p, r, f, s = precision_recall_fscore_support(labels, preds, labels=range(n), zero_division=0)
    return {
        "n": int(len(labels)),
        "macro_f1": float(f1_score(labels, preds, average="macro", labels=range(n), zero_division=0)),
        "accuracy": float(accuracy_score(labels, preds)),
        "top3_accuracy": topk_accuracy(probs, labels, 3),
        "nll": float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, None)).mean()),
        "per_class": {
            c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
            for i, c in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(labels, preds, labels=range(n)).tolist(),
    }


def macro_f1(labels: np.ndarray, preds: np.ndarray, n_classes: int) -> float:
    return float(f1_score(labels, preds, average="macro", labels=range(n_classes), zero_division=0))


def paired_bootstrap(labels, preds_a, preds_b, n_classes: int, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of macro-F1(a) - macro-F1(b) by resampling test images (paired)."""
    rng = np.random.default_rng(seed)
    labels, preds_a, preds_b = map(np.asarray, (labels, preds_a, preds_b))
    n = len(labels)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = macro_f1(labels[idx], preds_a[idx], n_classes) - macro_f1(labels[idx], preds_b[idx], n_classes)
    observed = macro_f1(labels, preds_a, n_classes) - macro_f1(labels, preds_b, n_classes)
    return {
        "observed_diff": float(observed),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_diff_le_0": float((diffs <= 0).mean()),
        "n_boot": n_boot,
    }
