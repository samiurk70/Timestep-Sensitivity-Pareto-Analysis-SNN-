"""
data/loader.py

All downloads happen ONCE and are cached — should not need re-downloading per session.

Datasets:
  thyroid  — ADBench Classical .npz  (38_thyroid.npz)
  cardio   — ADBench Classical .npz  (7_Cardiotocography.npz)
  nslkdd   — NSL-KDD network intrusion, auto-downloaded
  smap     — NASA SMAP proxy, auto-generated

ADBench .npz format:
  Each file contains a single structured array where the LAST column is the
  label (0=normal, 1=anomaly) and all preceding columns are features.
  This differs from ODDS .mat files which store 'X' and 'y' separately.

Usage:
    from src.data.loader import DataManager
    dm = DataManager(data_dir="/content/drive/MyDrive/SNN-Research/data")
    X, y = dm.load("thyroid")
    X_train, X_test, y_test = dm.split(X, y)
    train_loader, test_loader = dm.dataloaders(X_train, X_test)
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
        # File from ADBench/adbench/datasets/Classical/38_thyroid.npz
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
        # File from ADBench/adbench/datasets/Classical/7_Cardiotocography.npz
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
    "smap": {
        "filename": "smap.npz",
        "format": "synthetic",
        "n_samples": 2000, "n_features": 25, "anomaly_rate": 0.10,
        "source": "NASA SMAP proxy (Hundman et al. KDD 2018)",
        "citation": (
            "Hundman, K., et al. (2018). Detecting spacecraft anomalies using LSTMs "
            "and nonparametric dynamic thresholding. KDD 2018."
        ),
    },
}


# ── DataManager ───────────────────────────────────────────────────────────────

class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Load dataset by name. Builds a normalised cache on first call.
        Returns (X, y) both float32, X in [0, 1].
        """
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
              test_size: float = 0.2, seed: int = 42
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Semi-supervised split:
          TRAIN = normal samples only
          TEST  = held-out normal + ALL anomalies
        """
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
                    batch_size: int = 64) -> tuple[DataLoader, DataLoader]:
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

    def _acquire(self, name: str, info: dict) -> tuple[np.ndarray, np.ndarray]:
        fmt = info["format"]
        if fmt == "adbench_npz":
            return self._load_adbench_npz(name, info)
        if fmt == "npz_download" and name == "nslkdd":
            return self._download_nslkdd(info)
        if fmt == "synthetic" and name == "smap":
            return self._build_smap()
        raise ValueError(f"No acquisition method for '{name}' (format={fmt})")

    # ── ADBench .npz loader ───────────────────────────────────────────────────

    def _load_adbench_npz(self, name: str, info: dict
                          ) -> tuple[np.ndarray, np.ndarray]:
        """
        Load an ADBench Classical .npz file.

        ADBench packs each dataset as a 2-D float array where the LAST column
        is the binary label and all preceding columns are features.
        The array may be stored under various key names.

        The file must already be copied into data_dir with its original name.
        """
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
        y = (data[:, -1] > 0).astype(np.float32)   # binarise
        return X, y

    @staticmethod
    def _extract_array(raw, path: Path) -> np.ndarray:
        """
        ADBench .npz files use inconsistent key names across datasets.
        Try the three patterns that cover all 47 Classical datasets.
        """
        keys = list(raw.files)

        # Pattern A — single key (most common: the array IS the dataset)
        if len(keys) == 1:
            arr = raw[keys[0]]
            if arr.dtype == object:
                arr = arr.item()          # unwrap object wrapper
            return np.array(arr, dtype=np.float64)

        # Pattern B — explicit X / y keys (ODDS-style)
        if "X" in keys and "y" in keys:
            X = np.array(raw["X"], dtype=np.float64)
            y = np.array(raw["y"]).ravel().reshape(-1, 1).astype(np.float64)
            return np.hstack([X, y])

        # Pattern C — 'data' key
        if "data" in keys:
            return np.array(raw["data"], dtype=np.float64)

        raise RuntimeError(
            f"Unrecognised .npz structure in {path.name}.\n"
            f"Keys present: {keys}\n"
            f"Open with np.load() and inspect manually."
        )

    # ── NSL-KDD downloader ────────────────────────────────────────────────────

    def _download_nslkdd(self, info: dict) -> tuple[np.ndarray, np.ndarray]:
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

    # ── SMAP proxy builder ────────────────────────────────────────────────────

    def _build_smap(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Synthetic SMAP-proxy dataset.

        The real SMAP dataset requires downloading ~130 MB from:
          https://github.com/khundman/telemanom  (data/ folder)
        Place smap_train.pkl + smap_test.pkl in data_dir for real data.

        Label this 'SMAP-proxy' in your paper if using this synthetic version.
        """
        print("  Building SMAP-proxy (synthetic).")
        print("  NOTE: label as 'SMAP-proxy' in paper. See loader.py for real data.")
        rng = np.random.RandomState(42)
        n, f, r = 2000, 25, 0.10
        n_a = int(n * r)
        n_n = n - n_a
        t = np.linspace(0, 4 * np.pi, f)
        X_n = (np.sin(t) * 0.3 + np.cos(2 * t) * 0.15
               + rng.normal(0, 0.15, (n_n, f))).astype(np.float32)
        X_a = (np.sin(3 * t) * 0.8
               + rng.normal(0, 0.40, (n_a, f))).astype(np.float32)
        X = np.vstack([X_n, X_a])
        y = np.concatenate([np.zeros(n_n), np.ones(n_a)])
        idx = rng.permutation(n)
        return X[idx], y[idx]

    # ── Describe ──────────────────────────────────────────────────────────────

    @staticmethod
    def _describe(name: str, X: np.ndarray, y: np.ndarray):
        info = _REGISTRY.get(name, {})
        print(f"[{name.upper()}] {X.shape[0]} samples | "
              f"{X.shape[1]} features | "
              f"anomaly rate: {y.mean():.1%}")
        if "source" in info:
            print(f"  Source : {info['source']}")
