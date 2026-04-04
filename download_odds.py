"""
scripts/download_odds.py

Run this ONCE on your LOCAL machine to download the ODDS .mat files.
Then upload the output folder to Google Drive → SNN-Research/data/

Usage:
    python scripts/download_odds.py
    python scripts/download_odds.py --out /path/to/custom/dir

Requirements (local):
    pip install scipy numpy requests
"""

import argparse
import os
import sys
import hashlib
import requests
import scipy.io
import numpy as np
from pathlib import Path


# ── Dataset registry ──────────────────────────────────────────────────────────
# These are the PyPI-hosted mirrors from the ADBench paper (NeurIPS 2022).
# We use the HuggingFace Hub mirror which is reliably accessible globally.
# If HuggingFace is blocked in your region, the GitHub fallback is tried next.

DATASETS = {
    "thyroid": {
        "hf_url": (
            "https://huggingface.co/datasets/Minqi824/ADBench/resolve/main/"
            "adbench/datasets/Classical/thyroid.mat"
        ),
        "gh_url": (
            "https://github.com/Minqi824/ADBench/raw/main/"
            "adbench/datasets/Classical/thyroid.mat"
        ),
        "expected_shape": (3772, 6),
        "anomaly_rate": 0.025,
    },
    "cardiotocography": {
        "hf_url": (
            "https://huggingface.co/datasets/Minqi824/ADBench/resolve/main/"
            "adbench/datasets/Classical/cardiotocography.mat"
        ),
        "gh_url": (
            "https://github.com/Minqi824/ADBench/raw/main/"
            "adbench/datasets/Classical/cardiotocography.mat"
        ),
        "expected_shape": (1831, 21),
        "anomaly_rate": 0.096,
    },
    "arrhythmia": {
        "hf_url": (
            "https://huggingface.co/datasets/Minqi824/ADBench/resolve/main/"
            "adbench/datasets/Classical/arrhythmia.mat"
        ),
        "gh_url": (
            "https://github.com/Minqi824/ADBench/raw/main/"
            "adbench/datasets/Classical/arrhythmia.mat"
        ),
        "expected_shape": (452, 274),
        "anomaly_rate": 0.146,
    },
}


def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """Download url → dest. Returns True on success."""
    try:
        print(f"    GET {url}")
        r = requests.get(url, timeout=timeout, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"    ✗ {e}")
        if dest.exists():
            dest.unlink()
        return False


def verify_mat(path: Path, expected_shape: tuple, expected_anomaly_rate: float) -> bool:
    """Load and sanity-check a .mat file."""
    try:
        mat = scipy.io.loadmat(str(path))
        X = mat["X"]
        y = mat["y"].ravel()
        actual_rate = y.mean()
        ok_shape = (X.shape[0] == expected_shape[0] and X.shape[1] == expected_shape[1])
        ok_rate  = abs(actual_rate - expected_anomaly_rate) < 0.01
        if ok_shape and ok_rate:
            print(f"    ✓ Verified: {X.shape[0]} samples × {X.shape[1]} features "
                  f"| anomaly rate: {actual_rate:.1%}")
            return True
        else:
            print(f"    ✗ Shape/rate mismatch: got {X.shape}, rate={actual_rate:.3f}")
            return False
    except Exception as e:
        print(f"    ✗ Load failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download ODDS .mat files for SNN-Research project")
    parser.add_argument("--out", default="./data_download",
                        help="Output directory (default: ./data_download)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading ODDS datasets to: {out_dir.resolve()}\n")

    success = []
    failed  = []

    for name, info in DATASETS.items():
        print(f"── {name} ──")
        dest = out_dir / f"{name}.mat"

        if dest.exists():
            print(f"  Already exists, verifying...")
            if verify_mat(dest, info["expected_shape"], info["anomaly_rate"]):
                success.append(name)
                continue
            else:
                print(f"  File corrupt or wrong — re-downloading...")
                dest.unlink()

        # Try HuggingFace first, then GitHub
        downloaded = False
        for url in [info["hf_url"], info["gh_url"]]:
            print(f"  Trying:")
            if download_file(url, dest):
                if verify_mat(dest, info["expected_shape"], info["anomaly_rate"]):
                    downloaded = True
                    success.append(name)
                    break
                else:
                    dest.unlink()

        if not downloaded:
            print(f"  ✗ All mirrors failed for {name}.")
            print(f"    Manual fallback:")
            print(f"      1. Go to https://github.com/Minqi824/ADBench")
            print(f"         → adbench/datasets/Classical/{name}.mat")
            print(f"      2. Download manually and place at: {dest}")
            failed.append(name)

        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"DONE: {len(success)} succeeded, {len(failed)} failed")
    print("=" * 60)

    if success:
        print(f"\n✅ Successfully downloaded: {', '.join(success)}")
        print(f"\nFiles are in: {out_dir.resolve()}")
        print("\nNext step — upload these files to Google Drive:")
        print("  Destination: My Drive → SNN-Research → data →")
        for name in success:
            print(f"    {name}.mat")
        print("\nOr if you're using Drive for Desktop (sync folder):")
        print(f"  Copy {out_dir.resolve()}/*.mat")
        print(f"  → ~/Google Drive/My Drive/SNN-Research/data/")

    if failed:
        print(f"\n⚠  Failed: {', '.join(failed)}")
        print("  See manual fallback instructions above.")

    print()
    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
