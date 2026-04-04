"""
models/baselines.py

Classical anomaly detection baselines: OCSVM, IsolationForest, LOF.

These are fit on normal training data only (semi-supervised, same split
as the SNN/ANN experiments) and evaluated on the mixed test set.

Usage (single detector):
    from src.models.baselines import BaselineDetector
    result = BaselineDetector('IsoForest').fit(X_train).evaluate(X_test, y_test)

Usage (all baselines):
    from src.models.baselines import run_all_baselines
    results = run_all_baselines(X_train, X_test, y_test)
    # returns dict: {'IsoForest': {...}, 'OCSVM': {...}, 'LOF': {...}}
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, roc_auc_score


# ── Baseline registry ─────────────────────────────────────────────────────────

_BASELINE_CONFIGS = {
    "IsoForest": dict(
        cls=IsolationForest,
        kwargs=dict(n_estimators=100, contamination="auto", random_state=42),
    ),
    "OCSVM": dict(
        cls=OneClassSVM,
        kwargs=dict(kernel="rbf", nu=0.05, gamma="scale"),
    ),
    "LOF": dict(
        cls=LocalOutlierFactor,
        kwargs=dict(n_neighbors=20, contamination="auto", novelty=True),
    ),
}


# ── Single detector wrapper ───────────────────────────────────────────────────

class BaselineDetector:
    """
    Thin wrapper around sklearn anomaly detectors.
    Provides a consistent fit/evaluate interface matching the SNN/ANN pipeline.
    """

    def __init__(self, name: str):
        if name not in _BASELINE_CONFIGS:
            raise ValueError(
                f"Unknown baseline '{name}'. "
                f"Available: {list(_BASELINE_CONFIGS)}")
        self.name = name
        cfg = _BASELINE_CONFIGS[name]
        self.model = cfg["cls"](**cfg["kwargs"])

    def fit(self, X_train: np.ndarray) -> "BaselineDetector":
        self.model.fit(X_train)
        return self

    def evaluate(self, X_test: np.ndarray,
                 y_test: np.ndarray) -> dict:
        """
        Predict anomaly scores and compute F1 + AUC.

        sklearn convention: decision_function returns HIGHER = more normal.
        We negate so that HIGHER score = more anomalous (consistent with
        reconstruction-error framing used for SNN/ANN).
        """
        scores = -self.model.decision_function(X_test)   # higher = anomalous
        preds  = (self.model.predict(X_test) == -1).astype(int)  # -1 = anomaly

        f1  = float(f1_score(y_test, preds, zero_division=0))
        auc = float(roc_auc_score(y_test, scores))

        tp = int(((preds == 1) & (y_test == 1)).sum())
        fp = int(((preds == 1) & (y_test == 0)).sum())
        fn = int(((preds == 0) & (y_test == 1)).sum())
        precision = tp / (tp + fp + 1e-8)
        recall    = tp / (tp + fn + 1e-8)

        return {
            "f1":        f1,
            "auc":       auc,
            "precision": float(precision),
            "recall":    float(recall),
        }


# ── Run all baselines ─────────────────────────────────────────────────────────

def run_all_baselines(X_train: np.ndarray, X_test: np.ndarray,
                      y_test: np.ndarray) -> dict:
    """
    Fit and evaluate all three baselines.
    Returns dict keyed by baseline name, each value a metrics dict.
    """
    results = {}
    for name in _BASELINE_CONFIGS:
        try:
            result = BaselineDetector(name).fit(X_train).evaluate(X_test, y_test)
            results[name] = result
            print(f"  {name:<12} F1={result['f1']:.4f}  AUC={result['auc']:.4f}")
        except Exception as e:
            print(f"  {name:<12} FAILED: {e}")
            results[name] = {"f1": 0.0, "auc": 0.0, "error": str(e)}
    return results
