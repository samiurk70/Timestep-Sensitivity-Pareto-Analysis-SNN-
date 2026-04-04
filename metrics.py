"""
evaluation/metrics.py

All evaluation logic: reconstruction errors, thresholding, F1, AUC,
latency measurement, energy accounting, multi-seed runner.

Key design:
  - Multi-seed evaluation is the DEFAULT path (not an afterthought)
  - Energy figures are labelled by their basis (GPU-actual vs Loihi-projected)
  - Results are returned as plain dicts for easy JSON serialisation
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader


# ── Reconstruction error ──────────────────────────────────────────────────────

def reconstruction_errors(model: nn.Module, loader: DataLoader,
                           device: str) -> np.ndarray:
    model.eval()
    errors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            mse = ((recon - batch) ** 2).mean(dim=1)
            errors.extend(mse.cpu().numpy())
    return np.array(errors)


def optimal_threshold(train_errors: np.ndarray,
                      percentile: float = 95.0) -> float:
    return float(np.percentile(train_errors, percentile))


# ── Detection metrics ─────────────────────────────────────────────────────────

def detection_metrics(errors: np.ndarray, y_true: np.ndarray,
                      threshold: float) -> dict:
    y_pred = (errors > threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, errors)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    return {
        "f1":        float(f1),
        "auc":       float(auc),
        "precision": float(precision),
        "recall":    float(recall),
    }


# ── Latency ───────────────────────────────────────────────────────────────────

def measure_latency(model: nn.Module, sample: torch.Tensor,
                    n_warmup: int = 5, n_runs: int = 50) -> dict:
    """GPU-synchronised latency measurement (actual wall-clock on this hardware)."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(sample)

        if sample.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(sample)
            if sample.device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    return {"mean_ms": float(np.mean(times)), "std_ms": float(np.std(times))}


# ── Single-run evaluation ─────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, train_loader: DataLoader,
                   test_loader: DataLoader, y_test: np.ndarray,
                   device: str, model_type: str = "SNN") -> dict:
    """
    Full evaluation for one trained model.
    model_type: "SNN" or "ANN" — determines which energy counter is called.

    Note: test_errors are NOT stored in the returned dict to keep results
    JSON-serialisable. The threshold IS stored so you can reconstruct
    predictions externally if needed.
    """
    tr_err = reconstruction_errors(model, train_loader, device)
    threshold = optimal_threshold(tr_err)
    te_err = reconstruction_errors(model, test_loader, device)
    det = detection_metrics(te_err, y_test, threshold)

    sample = next(iter(test_loader))[0][:32].to(device)
    lat = measure_latency(model, sample)

    result = {**det, "latency_ms": lat["mean_ms"], "latency_std_ms": lat["std_ms"],
              "threshold": threshold}
    # test_errors kept separately for plotting — not in the main result dict

    if model_type == "SNN":
        ops = model.count_synops(sample)
        result.update({
            "synops_per_sample": ops["synops_per_sample"],
            "energy_nJ_projected": ops["energy_nJ"],  # Loihi 2 projection
            "energy_basis": "Loihi2_4.6fJ_per_synop",
        })
    else:
        ops = model.count_flops(sample)
        result.update({
            "flops_per_sample": ops["flops_per_sample"],
            "energy_nJ_actual": ops["energy_nJ"],  # GPU actual
            "energy_basis": "GPU_200fJ_per_FLOP",
        })

    return result


# ── Multi-seed runner ─────────────────────────────────────────────────────────

def multi_seed_evaluate(
    model_class,
    model_kwargs: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_fn,           # callable(model, loader, device) → None
    device: str,
    seeds: list[int] = (0, 1, 2, 3, 4),
    batch_size: int = 64,
    model_type: str = "SNN",
) -> dict:
    """
    Train + evaluate across multiple seeds.
    Returns mean ± std for all scalar metrics.
    This is the REQUIRED path for any publishable result.
    """
    from torch.utils.data import DataLoader, TensorDataset

    per_seed: list[dict] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Shuffle training data for this seed
        idx = np.random.permutation(len(X_train))
        X_tr_s = X_train[idx]

        tr_loader = DataLoader(
            TensorDataset(torch.tensor(X_tr_s)),
            batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(
            TensorDataset(torch.tensor(X_test)),
            batch_size=batch_size, shuffle=False)

        model = model_class(**model_kwargs).to(device)
        train_fn(model, tr_loader, device)

        result = evaluate_model(model, tr_loader, te_loader, y_test,
                                device, model_type)
        result["seed"] = seed
        per_seed.append(result)
        print(f"  seed={seed} | F1={result['f1']:.4f} | AUC={result['auc']:.4f}")

    # Aggregate
    scalar_keys = ["f1", "auc", "precision", "recall",
                   "latency_ms", "latency_std_ms"]
    if model_type == "SNN":
        scalar_keys += ["synops_per_sample", "energy_nJ_projected"]
    else:
        scalar_keys += ["flops_per_sample", "energy_nJ_actual"]

    agg = {}
    for k in scalar_keys:
        vals = [r[k] for r in per_seed if k in r]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"]  = float(np.std(vals))

    agg["per_seed"] = per_seed
    agg["n_seeds"] = len(seeds)
    return agg


# ── Summary printer ───────────────────────────────────────────────────────────

def print_comparison(snn_results: dict, ann_results: dict, dataset: str):
    print(f"\n{'='*65}")
    print(f"  {dataset.upper()} — Energy-Accuracy Summary")
    print(f"{'='*65}")

    def _row(name, r, energy_key, energy_label):
        f1  = r.get("f1_mean",  r.get("f1",  0))
        auc = r.get("auc_mean", r.get("auc", 0))
        f1s = r.get("f1_std",   0)
        e   = r.get(energy_key, 0)
        lat = r.get("latency_ms_mean", r.get("latency_ms", 0))
        print(f"  {name:<12} F1={f1:.4f}±{f1s:.4f}  AUC={auc:.4f}  "
              f"E={e:.5f} nJ ({energy_label})  lat={lat:.2f}ms")

    _row("SNN", snn_results, "energy_nJ_projected_mean", "Loihi2-projected")
    _row("ANN", ann_results, "energy_nJ_actual_mean",    "GPU-actual")

    # Energy reduction — use means
    snn_e = snn_results.get("energy_nJ_projected_mean",
                             snn_results.get("energy_nJ_projected", 1e-9))
    ann_e = ann_results.get("energy_nJ_actual_mean",
                             ann_results.get("energy_nJ_actual", 1))
    ratio = ann_e / (snn_e + 1e-12)
    print(f"\n  Projected energy reduction (neuromorphic vs GPU): {ratio:.1f}×")
    print(f"  ⚠  SNN is projected on Loihi 2; ANN measured on GPU.")
    print(f"     Direct hardware comparison is future work.")
    print(f"{'='*65}")
