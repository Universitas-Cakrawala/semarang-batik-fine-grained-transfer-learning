"""Phase 1 dataset audit.

One pass over the raw Kaggle snapshot that, per file:
  * hashes the raw bytes (exact-duplicate detection, manifest checksum),
  * verifies the JPEG/PNG can be parsed and fully decoded,
  * records format, mode, native size, EXIF orientation/timestamp/camera,
  * writes the deterministic training cache (EXIF-transposed RGB, shorter side
    `cache_short_side`, JPEG q=95) -- decoding 12 MP photos every epoch would
    otherwise dominate training time,
  * computes perceptual hashes and simple image statistics on the cached image.

Corrupt files and exact duplicates are excluded *before* any split is made, and
every exclusion is written to `exclusions.csv` with its reason.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from ..utils import write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Canonical class names (Kaggle metadata, dataset version 2).
CLASSES = [
    "Asem Arang",
    "Asem Sinom",
    "Asem Warak",
    "Blekok",
    "Blekok Warak",
    "Gambang Semarangan",
    "Kembang Sepatu",
    "Semarangan",
    "Tugu Muda",
    "Warak Beras Utah",
]

EXIF_ORIENTATION = 0x0112
EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_IFD = 0x8769
EXIF_DATETIME_ORIGINAL = 0x9003


def dihedral_views(im: Image.Image) -> list[Image.Image]:
    """The 8 rotations/reflections of an image (identity first)."""
    views = []
    for base in (im, ImageOps.mirror(im)):
        views += [base.rotate(90 * k, expand=True) for k in range(4)]
    return views


def canonical_class(folder: str) -> str | None:
    norm = " ".join(folder.replace("_", " ").replace("-", " ").split()).lower()
    norm = norm.removeprefix("batik cap ")  # Kaggle v2 folders are named "Batik Cap <motif>"
    for name in CLASSES:
        if norm == name.lower():
            return name
    return None


def discover(raw_dir: Path) -> list[tuple[Path, str]]:
    """Return (path, class) for every image whose parent folder is a known class."""
    items, unknown = [], Counter()
    for p in sorted(raw_dir.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        cls = canonical_class(p.parent.name)
        if cls is None:
            unknown[p.parent.name] += 1
            continue
        items.append((p, cls))
    if unknown:
        print(f"[audit] skipped images in non-class folders: {dict(unknown)}")
    return items


def _cache_path(processed_dir: Path, rel: Path) -> Path:
    return processed_dir / rel.with_suffix(".jpg")


def _audit_one(args) -> dict:
    path, cls, raw_dir, processed_dir, short_side = args
    rel = path.relative_to(raw_dir)
    rec: dict = {
        "relpath": rel.as_posix(),
        "class": cls,
        "ext": path.suffix.lower(),
        "bytes": path.stat().st_size,
        "status": "ok",
        "error": "",
    }
    data = path.read_bytes()
    rec["sha256"] = hashlib.sha256(data).hexdigest()
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rec["format"] = im.format
            rec["mode"] = im.mode
            rec["width"], rec["height"] = im.size
            exif = im.getexif()
            rec["exif_orientation"] = int(exif.get(EXIF_ORIENTATION, 1) or 1)
            rec["camera"] = f"{exif.get(EXIF_MAKE, '')} {exif.get(EXIF_MODEL, '')}".strip()
            rec["datetime_original"] = str(exif.get_ifd(EXIF_IFD).get(EXIF_DATETIME_ORIGINAL, "") or "")
            # JPEG DCT-domain downscale keeps both sides >= 2x the cache size.
            im.draft("RGB", (2 * short_side, 2 * short_side))
            im.load()
            im = ImageOps.exif_transpose(im)
            if im.mode != "RGB":
                rec["converted_from"] = im.mode
                im = im.convert("RGB")
            w, h = im.size
            rec["oriented_width"], rec["oriented_height"] = (
                (rec["height"], rec["width"]) if rec["exif_orientation"] in (5, 6, 7, 8) else (rec["width"], rec["height"])
            )
            scale = short_side / min(w, h)
            if scale < 1:
                im = im.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
            out = _cache_path(processed_dir, rel)
            out.parent.mkdir(parents=True, exist_ok=True)
            im.save(out, "JPEG", quality=95)
            rec["cache_width"], rec["cache_height"] = im.size
            rec["phash"] = str(imagehash.phash(im))
            # The Kaggle snapshot ships each photo in several rotated/flipped copies,
            # so near-duplicate search must be invariant to the 8 dihedral transforms.
            rec["phash_d8"] = "|".join(str(imagehash.phash(t)) for t in dihedral_views(im))
            rec["dhash"] = str(imagehash.dhash(im))
            gray = np.asarray(im.convert("L"), dtype=np.float32)
            lap = gray[1:-1, 1:-1] * 4 - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
            rec["brightness"] = float(gray.mean())
            rec["contrast"] = float(gray.std())
            rec["sharpness"] = float(lap.var())
    except Exception as e:  # corrupt header, truncated file, unsupported mode...
        rec["status"] = "corrupt"
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
    return rec


def run_audit(raw_dir: Path, processed_dir: Path, manifests_dir: Path, short_side: int, workers: int = 6) -> pd.DataFrame:
    items = discover(raw_dir)
    if not items:
        raise SystemExit(f"No class images found under {raw_dir}. Did you run `batik download`?")
    print(f"[audit] {len(items)} candidate images in {raw_dir}")
    jobs = [(p, c, raw_dir, processed_dir, short_side) for p, c in items]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(_audit_one, jobs, chunksize=8))
    df = pd.DataFrame(records).sort_values("relpath").reset_index(drop=True)

    # Exact duplicates: keep the first copy (sorted path) within a class; if the
    # same bytes appear under different classes the label is ambiguous -> drop all.
    exclusions = []
    df["exclude_reason"] = ""
    df.loc[df.status != "ok", "exclude_reason"] = "corrupt_or_unreadable"
    ok = df[df.status == "ok"]
    for sha, grp in ok.groupby("sha256"):
        if len(grp) < 2:
            continue
        if grp["class"].nunique() > 1:
            df.loc[grp.index, "exclude_reason"] = "exact_duplicate_label_conflict"
        else:
            df.loc[grp.index[1:], "exclude_reason"] = "exact_duplicate_of:" + grp.iloc[0]["relpath"]
    df["included"] = df["exclude_reason"] == ""
    for _, r in df[~df.included].iterrows():
        exclusions.append({"relpath": r.relpath, "class": r["class"], "reason": r.exclude_reason, "error": r.error})

    manifests_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifests_dir / "inventory.csv", index=False)
    pd.DataFrame(exclusions, columns=["relpath", "class", "reason", "error"]).to_csv(
        manifests_dir / "exclusions.csv", index=False
    )
    summary = summarize(df)
    write_json(summary, manifests_dir / "audit_summary.json")
    print(json.dumps({k: summary[k] for k in ("n_files", "n_included", "n_excluded", "exclusion_reasons")}, indent=2))
    return df


def summarize(df: pd.DataFrame) -> dict:
    inc = df[df.included]
    per_class = inc["class"].value_counts().reindex(CLASSES, fill_value=0)
    sizes = inc.apply(lambda r: f"{int(r.oriented_width)}x{int(r.oriented_height)}", axis=1)
    return {
        "n_files": int(len(df)),
        "n_included": int(len(inc)),
        "n_excluded": int((~df.included).sum()),
        "exclusion_reasons": {k.split(":")[0]: int(v) for k, v in df.loc[~df.included, "exclude_reason"].str.split(":").str[0].value_counts().items()},
        "per_class_raw": {c: int(v) for c, v in df["class"].value_counts().reindex(CLASSES, fill_value=0).items()},
        "per_class_included": {c: int(v) for c, v in per_class.items()},
        "class_balance": {
            "min": int(per_class.min()),
            "max": int(per_class.max()),
            "imbalance_ratio": round(float(per_class.max() / max(per_class.min(), 1)), 4),
        },
        "extensions": {k: int(v) for k, v in df["ext"].value_counts().items()},
        "formats": {k: int(v) for k, v in inc["format"].value_counts().items()},
        "modes": {k: int(v) for k, v in inc["mode"].value_counts().items()},
        "exif_orientation": {str(k): int(v) for k, v in inc["exif_orientation"].value_counts().items()},
        "cameras": {k or "<none>": int(v) for k, v in inc["camera"].fillna("").value_counts().items()},
        "has_datetime_original": int((inc["datetime_original"].fillna("") != "").sum()),
        "native_size_top10": {k: int(v) for k, v in sizes.value_counts().head(10).items()},
        "native_short_side": _describe(np.minimum(inc.oriented_width, inc.oriented_height)),
        "native_long_side": _describe(np.maximum(inc.oriented_width, inc.oriented_height)),
        "megabytes_total": round(float(df["bytes"].sum()) / 2**20, 1),
        "sharpness": _describe(inc["sharpness"]),
        "brightness": _describe(inc["brightness"]),
    }


def _describe(s) -> dict:
    s = pd.Series(s, dtype=float)
    return {k: round(float(v), 2) for k, v in s.describe(percentiles=[0.05, 0.5, 0.95]).items() if k != "count"}
