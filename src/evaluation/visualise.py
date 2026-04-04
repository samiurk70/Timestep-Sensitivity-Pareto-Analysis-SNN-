"""
evaluation/visualise.py

All plotting functions used in the notebook.
Each function saves to results_dir and returns the save path.

Plots produced:
  plot_training_loss       — epoch loss curves (SNN vs ANN)
  plot_error_distributions — reconstruction error histogram for one model
  plot_comparison          — F1 / AUC bar charts across datasets
  plot_t_sensitivity       — Pareto frontier: F1 vs projected energy vs T
  plot_sparsity            — per-layer spike firing rates
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


_SNN_COLOR = "#2196F3"
_ANN_COLOR = "#FF5722"
_ALPHA     = 0.85
_DPI       = 150

_MODEL_COLORS = {"SNN": _SNN_COLOR, "ANN": _ANN_COLOR}


def _savefig(fig, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Training loss ─────────────────────────────────────────────────────────────

def plot_training_loss(snn_losses, ann_losses, dataset, save_dir) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(snn_losses, color=_SNN_COLOR, label="SNN", linewidth=2)
    ax.plot(ann_losses, color=_ANN_COLOR, label="ANN", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title(f"Training Loss — {dataset.upper()}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = str(Path(save_dir) / "training_loss.png")
    return _savefig(fig, path)


# ── Error distributions (per-model, matches notebook call signature) ──────────

def plot_error_distributions(train_errors, test_errors, y_test,
                              threshold, model_name,
                              save_dir, dataset=None) -> str:
    """
    Plot reconstruction error histogram for a single model.

    Notebook call signature:
        plot_error_distributions(tr_err, te_err, y_te, thresh, mname,
                                  save_dir=str(ds_dir))

    Parameters
    ----------
    train_errors : 1-D array of reconstruction errors on training data
    test_errors  : 1-D array of reconstruction errors on test data
    y_test       : binary labels for test set (0=normal, 1=anomaly)
    threshold    : anomaly threshold (vertical line)
    model_name   : 'SNN' or 'ANN' (used in title + filename)
    save_dir     : directory to save figure
    dataset      : optional dataset name for title
    """
    color = _MODEL_COLORS.get(model_name, "#607D8B")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    title_suffix = f" — {dataset.upper()}" if dataset else ""

    # Left: training errors (all normal)
    axes[0].hist(train_errors, bins=50, color=color, alpha=_ALPHA)
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold ({threshold:.4f})")
    axes[0].set_title(f"{model_name} Train Errors{title_suffix}")
    axes[0].set_xlabel("MSE")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Right: test errors split by label
    axes[1].hist(test_errors[y_test == 0], bins=50, color=color,
                 alpha=_ALPHA, label="Normal")
    axes[1].hist(test_errors[y_test == 1], bins=50, color="gray",
                 alpha=_ALPHA, label="Anomaly")
    axes[1].axvline(threshold, color="black", linestyle="--", linewidth=1.5,
                    label=f"Threshold ({threshold:.4f})")
    axes[1].set_title(f"{model_name} Test Errors{title_suffix}")
    axes[1].set_xlabel("MSE")
    axes[1].set_ylabel("Count")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fname = f"error_dist_{model_name.lower()}.png"
    path = str(Path(save_dir) / fname)
    return _savefig(fig, path)


# ── Multi-dataset comparison ──────────────────────────────────────────────────

def plot_comparison(all_results, save_dir) -> str:
    """
    Bar chart: F1 and AUC for SNN vs ANN across all datasets.
    all_results: {dataset_name: {'snn': {...}, 'ann': {...}}}
    """
    datasets = list(all_results.keys())
    n = len(datasets)
    x = np.arange(n)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["f1", "auc"]):
        snn_vals = [all_results[d]["snn"].get(f"{metric}_mean",
                    all_results[d]["snn"].get(metric, 0)) for d in datasets]
        snn_stds = [all_results[d]["snn"].get(f"{metric}_std", 0)
                    for d in datasets]
        ann_vals = [all_results[d]["ann"].get(f"{metric}_mean",
                    all_results[d]["ann"].get(metric, 0)) for d in datasets]
        ann_stds = [all_results[d]["ann"].get(f"{metric}_std", 0)
                    for d in datasets]

        ax.bar(x - width/2, snn_vals, width, yerr=snn_stds,
               color=_SNN_COLOR, alpha=_ALPHA, capsize=4, label="SNN")
        ax.bar(x + width/2, ann_vals, width, yerr=ann_stds,
               color=_ANN_COLOR, alpha=_ALPHA, capsize=4, label="ANN")
        ax.set_xticks(x)
        ax.set_xticklabels([d.upper() for d in datasets], fontsize=9)
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} Comparison (mean ± std, 5 seeds)")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = str(Path(save_dir) / "comparison_all_datasets.png")
    return _savefig(fig, path)


# ── T-sensitivity Pareto frontier ─────────────────────────────────────────────

def plot_t_sensitivity(t_results, dataset, save_dir) -> str:
    """
    t_results: {T_int: result_dict}  (output of multi_seed_evaluate)
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    T_vals   = sorted(t_results.keys())
    energies = [t_results[T].get("energy_nJ_projected_mean",
                t_results[T].get("energy_nJ_projected", 0)) for T in T_vals]
    f1s      = [t_results[T].get("f1_mean",
                t_results[T].get("f1", 0)) for T in T_vals]
    f1_stds  = [t_results[T].get("f1_std", 0) for T in T_vals]

    sc = ax.scatter(energies, f1s, c=T_vals, cmap="viridis", s=120, zorder=5)
    ax.errorbar(energies, f1s, yerr=f1_stds,
                fmt="none", color="gray", capsize=4, alpha=0.6)

    for T, e, f in zip(T_vals, energies, f1s):
        ax.annotate(f"T={T}", (e, f), textcoords="offset points",
                    xytext=(6, 4), fontsize=9)

    ax.plot(energies, f1s, color=_SNN_COLOR, alpha=0.4,
            linewidth=1.5, linestyle="--")

    plt.colorbar(sc, ax=ax, label="T (time steps)")
    ax.set_xlabel("Projected Energy (nJ, Loihi 2)")
    ax.set_ylabel("F1 Score (mean ± std)")
    ax.set_title(f"T-Sensitivity: Accuracy–Energy Tradeoff\n{dataset.upper()}")
    ax.grid(True, alpha=0.3)

    path = str(Path(save_dir) / f"t_sensitivity_{dataset}.png")
    return _savefig(fig, path)


# ── Spike sparsity ────────────────────────────────────────────────────────────

def plot_sparsity(sparsity_by_dataset, save_dir) -> str:
    datasets     = list(sparsity_by_dataset.keys())
    layers       = ["enc_hidden", "enc_latent", "dec_hidden"]
    layer_labels = ["Enc Hidden", "Enc Latent", "Dec Hidden"]

    n     = len(datasets)
    x     = np.arange(len(layers))
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, n))

    for i, (ds, rates) in enumerate(sparsity_by_dataset.items()):
        vals   = [rates.get(l, 0) for l in layers]
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=ds.upper(),
               color=colors[i], alpha=_ALPHA)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Mean Firing Rate")
    ax.set_title("Per-Layer Spike Firing Rates\n"
                 "(lower = sparser = more energy-efficient on neuromorphic HW)")
    ax.set_ylim(0, 1.0)
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.4,
               label="50% reference")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    path = str(Path(save_dir) / "sparsity_analysis.png")
    return _savefig(fig, path)
