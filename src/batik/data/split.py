"""Leakage-aware, group-stratified 70/15/15 split.

Groups are connected components of a within-class "near-duplicate" graph whose
edges come from (a) 64-bit pHash Hamming distance and (b) cosine similarity of
ImageNet EfficientNet-B0 embeddings. Every group is placed in exactly one split
via StratifiedGroupKFold (20 folds of ~5%: 3 test, 3 val, 14 train), so near-
identical photographs cannot sit on both sides of the train/test boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from ..utils import read_json, sha256_file, write_json
from .audit import CLASSES


def _hamming_matrix(hashes: list[str]) -> np.ndarray:
    h = np.array([int(x, 16) for x in hashes], dtype=np.uint64)
    return np.bitwise_count(h[:, None] ^ h[None, :]).astype(np.int16)


def _d8(hashes_d8) -> np.ndarray:
    return np.array([[int(x, 16) for x in row.split("|")] for row in hashes_d8], dtype=np.uint64)


def dihedral_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """min over the 8 rotations/flips of b of the pHash Hamming distance to a's identity view."""
    out = np.full((len(a), len(b)), 64, dtype=np.int16)
    for k in range(b.shape[1]):
        out = np.minimum(out, np.bitwise_count(a[:, :1] ^ b[None, :, k]).astype(np.int16))
    return out


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@torch.no_grad()
def compute_embeddings(frame: pd.DataFrame, processed_dir: Path, cache: Path, batch_size: int = 64) -> np.ndarray:
    """L2-normalised ImageNet EfficientNet-B0 pooled features (audit only, never trained)."""
    key = hashlib.sha256("".join(frame.sha256).encode()).hexdigest()[:16]
    cache_file = cache / f"embeddings_effb0_{key}.npy"
    if cache_file.exists():
        return np.load(cache_file)
    from torchvision import models as tvm
    from torchvision import transforms as T

    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = tvm.efficientnet_b0(weights=weights).eval()
    tf = T.Compose([T.Resize(256, antialias=True), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    feats = []
    paths = [processed_dir / Path(r).with_suffix(".jpg") for r in frame.relpath]
    for i in range(0, len(paths), batch_size):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]])
        f = torch.flatten(model.avgpool(model.features(x)), 1)
        feats.append(torch.nn.functional.normalize(f, dim=1).numpy())
        print(f"\r[split] embeddings {min(i + batch_size, len(paths))}/{len(paths)}", end="", flush=True)
    print()
    emb = np.concatenate(feats).astype(np.float32)
    cache.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, emb)
    return emb


def center(emb: np.ndarray) -> np.ndarray:
    """Mean-centre then L2-normalise. Post-SiLU features are all positive, so raw
    cosines are ~0.95+ even for unrelated images; centring makes cosine meaningful."""
    e = emb - emb.mean(0, keepdims=True)
    return e / np.linalg.norm(e, axis=1, keepdims=True).clip(min=1e-8)


def build_groups(frame: pd.DataFrame, emb: np.ndarray, phash_max: int, cos_min: float,
                 same_timestamp: bool = True) -> tuple[np.ndarray, list[dict]]:
    """Connected components of the within-class near-duplicate graph.

    Edges: (1) identical EXIF capture second (the snapshot stores each capture as
    several rotated/flipped files, plus burst shots), (2) rotation/flip-invariant
    pHash distance <= phash_max, (3) centred-embedding cosine >= cos_min.
    """
    uf = _UnionFind(len(frame))
    edges = []
    d8 = _d8(frame.phash_d8) if "phash_d8" in frame else None
    for cls, idx in frame.groupby("class").indices.items():
        if d8 is not None:
            ham = dihedral_distance(d8[idx], d8[idx])
            ham = np.minimum(ham, ham.T)
        else:
            ham = _hamming_matrix(frame.phash.iloc[idx].tolist())
        cos = emb[idx] @ emb[idx].T
        use_ts = same_timestamp and "datetime_original" in frame
        ts = frame.datetime_original.iloc[idx].fillna("").to_numpy(dtype=object) if use_ts else None
        iu = np.triu_indices(len(idx), 1)
        for a, b in zip(*iu):
            reasons = []
            if ts is not None and ts[a] and ts[a] == ts[b]:
                reasons.append("same_timestamp")
            if ham[a, b] <= phash_max:
                reasons.append("phash_d8")
            if cos[a, b] >= cos_min:
                reasons.append("embedding")
            if reasons:
                uf.union(idx[a], idx[b])
                edges.append({"a": frame.relpath.iloc[idx[a]], "b": frame.relpath.iloc[idx[b]], "class": cls,
                              "phash_d8_dist": int(ham[a, b]), "cosine": round(float(cos[a, b]), 4),
                              "reason": "+".join(reasons)})
    roots = np.array([uf.find(i) for i in range(len(frame))])
    _, group_ids = np.unique(roots, return_inverse=True)
    return group_ids, edges


def assign_splits(labels: np.ndarray, groups: np.ndarray, seed: int, n_folds: int, fractions: dict) -> np.ndarray:
    n_test = round(fractions["test"] * n_folds)
    n_val = round(fractions["val"] * n_folds)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.empty(len(labels), dtype=int)
    for k, (_, test_idx) in enumerate(sgkf.split(np.zeros(len(labels)), labels, groups)):
        fold[test_idx] = k
    split = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))
    return split


def cross_split_report(frame: pd.DataFrame, emb: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Nearest *training* neighbour of every val/test image (rotation-invariant pHash + embedding)."""
    tr = np.where(frame.split == "train")[0]
    ev = np.where(frame.split != "train")[0]
    d8 = _d8(frame.phash_d8)
    ham = dihedral_distance(d8[ev], d8[tr])
    cos = emb[ev] @ emb[tr].T
    ts, cls = frame.datetime_original.to_numpy(dtype=object), frame["class"].to_numpy(dtype=object)
    ts_tr, cls_tr = ts[tr], cls[tr]
    rows = []
    for r, i in enumerate(ev):
        j, k = tr[int(np.argmax(cos[r]))], tr[int(np.argmin(ham[r]))]
        rows.append({
            "relpath": frame.relpath.iloc[i], "split": frame.split.iloc[i], "class": frame["class"].iloc[i],
            "nn_emb_relpath": frame.relpath.iloc[j], "nn_emb_class": frame["class"].iloc[j],
            "nn_emb_cosine": float(cos[r].max()),
            "nn_phash_relpath": frame.relpath.iloc[k], "nn_phash_d8_dist": int(ham[r].min()),
            "same_timestamp_in_train": bool(((ts_tr == ts[i]) & (cls_tr == cls[i])).any()),
        })
    df = pd.DataFrame(rows).sort_values("nn_emb_cosine", ascending=False)
    summary = {
        "n_eval_images": int(len(df)),
        "min_phash_d8_dist_to_train": int(df.nn_phash_d8_dist.min()),
        "n_phash_d8_dist_le_8": int((df.nn_phash_d8_dist <= 8).sum()),
        "n_same_timestamp_in_train": int(df.same_timestamp_in_train.sum()),
        "nn_cosine_quantiles": {str(q): round(float(df.nn_emb_cosine.quantile(q)), 4) for q in (0.5, 0.9, 0.99, 1.0)},
        "nn_same_class_rate": round(float((df["class"] == df.nn_emb_class).mean()), 4),
    }
    return summary, df


def make_split(inventory_csv: Path, processed_dir: Path, manifests_dir: Path, cache_dir: Path, cfg: dict) -> pd.DataFrame:
    inv = pd.read_csv(inventory_csv)
    frame = inv[inv.included].sort_values("relpath").reset_index(drop=True)
    classes = [c for c in CLASSES if c in set(frame["class"])]
    label_of = {c: i for i, c in enumerate(classes)}
    frame["label"] = frame["class"].map(label_of)
    frame["image_id"] = frame.sha256.str[:12]

    lk, sp = cfg["data"]["leakage"], cfg["data"]["split"]
    emb = center(compute_embeddings(frame, processed_dir, cache_dir))
    groups, edges = build_groups(frame, emb, lk["phash_max_distance"], lk["embed_min_cosine"],
                                 lk.get("group_same_timestamp", True))
    frame["group_id"] = groups
    frame["split"] = assign_splits(frame.label.values, groups, sp["seed"], sp["n_folds"], sp["fractions"])

    # Integrity checks -- fail loudly rather than silently leak.
    assert frame.groupby("group_id").split.nunique().max() == 1, "a group spans several splits"
    assert frame.sha256.is_unique, "exact duplicates survived the audit"

    cols = ["image_id", "relpath", "class", "label", "split", "group_id", "sha256"]
    manifests_dir.mkdir(parents=True, exist_ok=True)
    split_csv = manifests_dir / "split.csv"
    frame[cols].to_csv(split_csv, index=False)
    write_json({"classes": classes, "label_of": label_of}, manifests_dir / "labels.json")
    pd.DataFrame(edges).to_csv(manifests_dir / "near_duplicate_edges.csv", index=False)

    leak_summary, nn_df = cross_split_report(frame, emb)
    nn_df.to_csv(manifests_dir / "cross_split_nearest_train.csv", index=False)
    group_sizes = frame.groupby("group_id").size()
    counts = frame.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(columns=["train", "val", "test"])
    meta = {
        "split_csv_sha256": sha256_file(split_csv),
        "seed": sp["seed"], "n_folds": sp["n_folds"], "fractions_target": sp["fractions"],
        "n_images": int(len(frame)),
        "split_counts": {k: int(v) for k, v in frame.split.value_counts().reindex(["train", "val", "test"]).items()},
        "split_fractions": {k: round(float(v), 4) for k, v in frame.split.value_counts(normalize=True).items()},
        "per_class_split_counts": {c: {s: int(v) for s, v in r.items()} for c, r in counts.iterrows()},
        "grouping": {
            "phash_max_distance": lk["phash_max_distance"], "embed_min_cosine": lk["embed_min_cosine"],
            "group_same_timestamp": lk.get("group_same_timestamp", True),
            "edge_reasons": pd.Series([e["reason"] for e in edges]).value_counts().to_dict() if edges else {},
            "n_edges": len(edges), "n_groups": int(group_sizes.size),
            "n_singletons": int((group_sizes == 1).sum()), "largest_group": int(group_sizes.max()),
            "images_in_multi_image_groups": int(group_sizes[group_sizes > 1].sum()),
        },
        "leakage_check": leak_summary,
    }
    write_json(meta, manifests_dir / "split_meta.json")
    print(f"[split] {meta['split_counts']} | groups={meta['grouping']} | sha256={meta['split_csv_sha256'][:12]}")
    return frame


def verify_split(manifests_dir: Path) -> bool:
    meta = read_json(manifests_dir / "split_meta.json")
    ok = sha256_file(manifests_dir / "split.csv") == meta["split_csv_sha256"]
    print(f"[split] manifest checksum {'OK' if ok else 'MISMATCH'}")
    return ok
