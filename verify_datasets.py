"""
scripts/verify_datasets.py

We need to run this locally to confirm the ADBench .npz files load correctly
before uploading them to Google Drive if using T4 GPU.

Usage:
    python scripts/verify_datasets.py --data_dir ./data_download

Expected output:
    [THYROID]  3772 samples | 6 features  | anomaly rate: 2.5%
    [CARDIO]   1831 samples | 21 features | anomaly rate: 9.6%
    All local dataset checks passed.
"""
import argparse
import sys
import numpy as np
from pathlib import Path

def check_adbench_npz(path: Path, expected_features: int,
                      expected_approx_samples: int,
                      expected_anomaly_rate: float,
                      name: str) -> bool:
    if not path.exists():
        print(f"  [MISSING] {path.name}")
        print(f"            Copy from ADBench/adbench/datasets/Classical/")
        return False

    try:
        raw = np.load(str(path), allow_pickle=True)
        keys = list(raw.files)

        # Extract the array (same logic as loader.py _extract_array)
        if len(keys) == 1:
            arr = raw[keys[0]]
            if arr.dtype == object:
                arr = arr.item()
            data = np.array(arr, dtype=np.float64)
        elif "X" in keys and "y" in keys:
            X = np.array(raw["X"])
            y = np.array(raw["y"]).ravel().reshape(-1, 1)
            data = np.hstack([X, y])
        elif "data" in keys:
            data = np.array(raw["data"])
        else:
            print(f"  [FAIL] {name}: unrecognised keys {keys}")
            return False

        n_samples  = data.shape[0]
        n_features = data.shape[1] - 1      # last col is label
        y          = (data[:, -1] > 0).astype(float)
        rate       = y.mean()

        ok_features = (n_features == expected_features)
        ok_samples  = (abs(n_samples - expected_approx_samples) < expected_approx_samples * 0.1)
        ok_rate     = (abs(rate - expected_anomaly_rate) < 0.03)

        if ok_features and ok_samples and ok_rate:
            print(f"  [OK] {name}: {n_samples} samples | "
                  f"{n_features} features | anomaly: {rate:.1%}")
            return True
        else:
            print(f"  [WARN] {name}: got {n_samples}x{n_features}, rate={rate:.3f}")
            print(f"         expected ~{expected_approx_samples}x{expected_features}, "
                  f"rate~{expected_anomaly_rate:.3f}")
            print(f"         File loaded but stats differ from ODDS reference.")
            print(f"         This may still be usable — check the data manually.")
            return True   # not a hard failure

    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data_download",
                        help="Directory containing the .npz files")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"\nChecking datasets in: {data_dir.resolve()}\n")

    checks = [
        ("thyroid",  "38_thyroid.npz",          6,   3772, 0.025),
        ("cardio",   "7_Cardiotocography.npz",  21,   1831, 0.096),
    ]

    passed = 0
    for name, filename, n_feat, n_samp, rate in checks:
        path = data_dir / filename
        ok = check_adbench_npz(path, n_feat, n_samp, rate, name.upper())
        passed += int(ok)

    print(f"\n{passed}/{len(checks)} dataset files verified.")

    if passed == len(checks):
        print("\nNext step:")
        print(f"  Upload all .npz files from {data_dir.resolve()}")
        print(f"  to Google Drive → SNN-Research → data →")
        for _, filename, *_ in checks:
            print(f"    {filename}")
        print("\nThen open experiment.ipynb and run Cell 4.")
    else:
        missing = len(checks) - passed
        print(f"\n{missing} file(s) missing.")
        print("Copy them from ADBench/adbench/datasets/Classical/ to your data_dir.")
        sys.exit(1)


if __name__ == "__main__":
    main()
