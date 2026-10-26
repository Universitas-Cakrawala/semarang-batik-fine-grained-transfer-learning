"""Fetch the pinned Kaggle snapshot and record its checksum.

Kaggle serves public datasets without credentials through the API download
endpoint; if that ever changes, place `~/.kaggle/kaggle.json` (or set
KAGGLE_USERNAME / KAGGLE_KEY) and the request is retried with basic auth.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import zipfile
from pathlib import Path

from ..utils import read_json, sha256_file, write_json

API = "https://www.kaggle.com/api/v1/datasets/download/{ref}"


def _auth_args() -> list[str]:
    user, key = os.environ.get("KAGGLE_USERNAME"), os.environ.get("KAGGLE_KEY")
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if not (user and key) and cred.exists():
        c = read_json(cred)
        user, key = c.get("username"), c.get("key")
    return ["-u", f"{user}:{key}"] if user and key else []


def download(ref: str, version: int, raw_root: Path, manifests_dir: Path, keep_zip: bool = True) -> Path:
    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = raw_root.parent / f"{ref.split('/')[-1]}.zip"
    if not zip_path.exists():
        if shutil.which("curl") is None:
            raise SystemExit("curl is required to download the dataset")
        print(f"[download] {ref} (~13 GB) -> {zip_path}")
        # The un-versioned endpoint serves the current version; we pin by checksum below.
        cmd = ["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(zip_path), *_auth_args(), API.format(ref=ref)]
        subprocess.run(cmd, check=True)
    print("[download] hashing archive ...")
    digest = sha256_file(zip_path)
    snap_file = manifests_dir / "dataset_snapshot.json"
    if snap_file.exists():
        pinned = read_json(snap_file)
        if pinned.get("archive_sha256") != digest:
            raise SystemExit(
                f"Archive checksum {digest[:12]} differs from pinned snapshot {pinned.get('archive_sha256', '')[:12]}: "
                "Kaggle may have published a new version. Do not mix snapshots; see README."
            )
    if not any(raw_root.iterdir()):
        print(f"[download] extracting -> {raw_root}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_root)
    snapshot = {
        "kaggle_ref": ref, "kaggle_version": version, "archive": zip_path.name,
        "archive_bytes": zip_path.stat().st_size, "archive_sha256": digest,
        "license": "CC BY-NC-SA 4.0",
        "citation": "Winarno, E., Septiarini, A., Hadikurniawati, W., & Hamdani, H. (2024). The Hybrid Features and "
                    "Supervised Learning for Batik Pattern Classification. ACM Journal on Computing and Cultural "
                    "Heritage, 17(2). https://doi.org/10.1145/3631131",
    }
    write_json(snapshot, snap_file)
    if not keep_zip:
        zip_path.unlink()
    return raw_root
