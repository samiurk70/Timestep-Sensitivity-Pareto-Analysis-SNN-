"""
data/loader.py

Loads datasets from a persistent directory (local).
All downloads happen ONCE and are cached — no re-downloading per session.

Datasets:
  thyroid  — ADBench Classical .npz  (38_thyroid.npz)
  cardio   — ADBench Classical .npz  (7_Cardiotocography.npz)
  nslkdd   — NSL-KDD network intrusion, auto-downloaded
  unswnb15 — UNSW-NB15, read from local CSV (see instructions below)

UNSW-NB15 setup (one time):
  1. Download from Kaggle: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
  2. Save UNSW_NB15_training-set.csv → data/unswnb15.csv
  3. Run Cell 4 — it caches to unswnb15_cached.npz automatically
"""

import numpy as np
import requests
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


# ── Dataset registry ──────────────────────────────────────────────────────────

_REGISTRY = {
    "thyroid": {
        "adbench_filename": "38_thyroid.npz",
        "format": "adbench_npz",
        "n_samples": 3772, "n_features": 6, "anomaly_rate": 0.025,
        "source": "ODDS via ADBench (Han et al. NeurIPS 2022)",
        "citation": (
            "Rayana, S. (2016). ODDS Library. Stony Brook University.  "
            "Han, S., et al. (2022). ADBench. NeurIPS Datasets & Benchmarks."
        ),
    },
    "cardio": {
        "adbench_filename": "7_Cardiotocography.npz",
        "format": "adbench_npz",
        "n_samples": 1831, "n_features": 21, "anomaly_rate": 0.096,
        "source": "ODDS via ADBench (Han et al. NeurIPS 2022)",
        "citation": (
            "Rayana, S. (2016). ODDS Library. Stony Brook University.  "
            "Han, S., et al. (2022). ADBench. NeurIPS Datasets & Benchmarks."
        ),
    },
    "nslkdd": {
        "filename": "nslkdd.npz",
        "format": "npz_download",
        "download_url": (
            "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt"
        ),
        "n_samples": 10000, "n_features": 38, "anomaly_rate": 0.465,
        "source": "NSL-KDD (Tavallaee et al. 2009)",
        "citation": (
            "Tavallaee, M., et al. (2009). A detailed analysis of the KDD CUP 99 "
            "data set. CISDA 2009."
        ),
    },
    "unswnb15": {
        "local_csv": "unswnb15.csv",        # place this file in data/
        "format": "unswnb15_local",
        "n_features": 42, "anomaly_rate": 0.321,
        "source": "UNSW-NB15 (Moustafa & Slay 2015)",
        "citation": (
            "Moustafa, N., & Slay, J. (2015). UNSW-NB15: a comprehensive data set "
            "for network intrusion detection systems. MilCIS 2015."
        ),
    },
}


# ── DataManager ───────────────────────────────────────────────────────────────

class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, name: str) -> tuple:
        if name not in _REGISTRY:
            raise ValueError(
                f"Unknown dataset '{name}'. Available: {list(_REGISTRY)}")

        info = _REGISTRY[name]
        cached = self.data_dir / f"{name}_cached.npz"

        if cached.exists():
            d = np.load(str(cached))
            X, y = d["X"].astype(np.float32), d["y"].astype(np.float32)
            self._describe(name, X, y)
            return X, y

        X, y = self._acquire(name, info)
        X = MinMaxScaler().fit_transform(X).astype(np.float32)
        y = y.astype(np.float32)
        np.savez(str(cached), X=X, y=y)
        self._describe(name, X, y)
        return X, y

    def split(self, X: np.ndarray, y: np.ndarray,
              test_size: float = 0.2, seed: int = 42) -> tuple:
        """Semi-supervised: train on normal only, test on normal + all anomalies."""
        X_norm, X_anom = X[y == 0], X[y == 1]
        X_tr, X_nte = train_test_split(
            X_norm, test_size=test_size, random_state=seed)
        X_te = np.vstack([X_nte, X_anom])
        y_te = np.concatenate([np.zeros(len(X_nte)), np.ones(len(X_anom))])
        idx = np.random.RandomState(seed).permutation(len(X_te))
        print(f"  Train : {len(X_tr):>5} normal only")
        print(f"  Test  : {len(X_nte):>5} normal + {len(X_anom):>5} anomaly")
        return X_tr, X_te[idx], y_te[idx]

    def dataloaders(self, X_train: np.ndarray, X_test: np.ndarray,
                    batch_size: int = 64) -> tuple:
        tr = DataLoader(
            TensorDataset(torch.tensor(X_train)),
            batch_size=batch_size, shuffle=True, drop_last=False)
        te = DataLoader(
            TensorDataset(torch.tensor(X_test)),
            batch_size=batch_size, shuffle=False, drop_last=False)
        return tr, te

    def info(self, name: str) -> dict:
        return _REGISTRY[name]

    # ── Internal acquisition ──────────────────────────────────────────────────

    def _acquire(self, name: str, info: dict) -> tuple:
        fmt = info["format"]
        if fmt == "adbench_npz":
            return self._load_adbench_npz(name, info)
        if fmt == "npz_download" and name == "nslkdd":
            return self._download_nslkdd(info)
        if fmt == "unswnb15_local":
            return self._load_unswnb15_local(info)
        raise ValueError(f"No acquisition method for '{name}' (format={fmt})")

    # ── ADBench .npz loader ───────────────────────────────────────────────────

    def _load_adbench_npz(self, name: str, info: dict) -> tuple:
        adbench_name = info["adbench_filename"]
        path = self.data_dir / adbench_name

        if not path.exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"Dataset '{name}' not found.\n\n"
                f"Expected : {path}\n\n"
                f"Fix:\n"
                f"  Copy  ADBench/adbench/datasets/Classical/{adbench_name}\n"
                f"  →     {path}\n"
                f"  Then re-run this cell.\n"
                f"{'='*60}\n"
            )

        raw = np.load(str(path), allow_pickle=True)
        data = self._extract_array(raw, path)
        X = data[:, :-1].astype(np.float32)
        y = (data[:, -1] > 0).astype(np.float32)
        return X, y

    @staticmethod
    def _extract_array(raw, path: Path) -> np.ndarray:
        keys = list(raw.files)
        if len(keys) == 1:
            arr = raw[keys[0]]
            if arr.dtype == object:
                arr = arr.item()
            return np.array(arr, dtype=np.float64)
        if "X" in keys and "y" in keys:
            X = np.array(raw["X"], dtype=np.float64)
            y = np.array(raw["y"]).ravel().reshape(-1, 1).astype(np.float64)
            return np.hstack([X, y])
        if "data" in keys:
            return np.array(raw["data"], dtype=np.float64)
        raise RuntimeError(
            f"Unrecognised .npz structure in {path.name}. Keys: {keys}")

    # ── NSL-KDD downloader ────────────────────────────────────────────────────

    def _download_nslkdd(self, info: dict) -> tuple:
        print("  Downloading NSL-KDD...")
        import pandas as pd
        from io import StringIO

        r = requests.get(info["download_url"], timeout=60)
        r.raise_for_status()
        cols = [f"f{i}" for i in range(41)] + ["label", "difficulty"]
        df = pd.read_csv(StringIO(r.text), names=cols)
        numeric = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ["label", "difficulty"]]
        X = df[numeric].fillna(0).values.astype(np.float32)
        y = (df["label"] != "normal").values.astype(np.float32)
        idx = np.random.RandomState(42).choice(
            len(X), min(10_000, len(X)), replace=False)
        print(f"  NSL-KDD: {len(X)} total rows, sampled {len(idx)}")
        return X[idx], y[idx]

    # ── UNSW-NB15 local CSV loader ────────────────────────────────────────────

    def _load_unswnb15_local(self, info: dict) -> tuple:
        """
        Load UNSW-NB15 from a local CSV file placed in data/.

        Setup:
          1. Download UNSW_NB15_training-set.csv from Kaggle:
             https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
          2. Save as: data/unswnb15.csv
          3. Re-run Cell 4 — this function builds the cache automatically.

        The label column is 'label' (0=normal, 1=attack).
        Categorical columns (proto, service, state) are dropped.
        Up to 50,000 rows sampled for speed.
        """
        import pandas as pd

        csv_path = self.data_dir / info["local_csv"]

        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"UNSW-NB15 CSV not found.\n\n"
                f"Expected : {csv_path}\n\n"
                f"Fix:\n"
                f"  1. Download from Kaggle:\n"
                f"     https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15\n"
                f"  2. Save UNSW_NB15_training-set.csv as:\n"
                f"     {csv_path}\n"
                f"  3. Re-run Cell 4.\n"
                f"{'='*60}\n"
            )

        print(f"  Loading UNSW-NB15 from {csv_path.name}...")
        df = pd.read_csv(str(csv_path))
        print(f"  Raw shape: {df.shape}")

        # Drop identifier / categorical columns
        drop_cols = ["id", "attack_cat", "proto", "service", "state",
                     "srcip", "dstip", "sport", "dsport"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns],
                     errors="ignore")

        # Find label column (may be 'label' or 'Label')
        label_col = next((c for c in df.columns if c.lower() == "label"), None)
        if label_col is None:
            raise RuntimeError(
                f"No 'label' column found.\n"
                f"Columns present: {list(df.columns)}\n"
                f"Rename the attack/normal indicator column to 'label'.")

        y = df[label_col].values.astype(np.float32)
        X_df = df.drop(columns=[label_col])
        X_df = X_df.select_dtypes(include=[np.number]).fillna(0)
        X = X_df.values.astype(np.float32)

        # Sample for speed — 50k rows is plenty for benchmarking
        n_sample = min(50_000, len(X))
        idx = np.random.RandomState(42).choice(len(X), n_sample, replace=False)
        X, y = X[idx], y[idx]

        print(f"  UNSW-NB15: {len(X)} samples | "
              f"{X.shape[1]} features | anomaly rate: {y.mean():.1%}")
        return X, y

    # ── Describe ──────────────────────────────────────────────────────────────

    @staticmethod
    def _describe(name: str, X: np.ndarray, y: np.ndarray):
        info = _REGISTRY.get(name, {})
        print(f"[{name.upper()}] {X.shape[0]} samples | "
              f"{X.shape[1]} features | "
              f"anomaly rate: {y.mean():.1%}")
        if "source" in info:
            print(f"  Source : {info['source']}")
