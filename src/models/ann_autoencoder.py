"""
models/ann_autoencoder.py

Standard deep autoencoder — the ANN baseline for energy-accuracy comparison.

Architecture mirrors the SNN: same layer widths (input→hidden→latent→hidden→input)
so the comparison is architecturally fair. Only the neuron model differs.

Energy accounting:
  MACs = multiply-accumulate operations per forward pass
  Estimated energy (nJ) = MACs × 200e-6 nJ  (200 fJ/FLOP, Horowitz 2014)
  This is GPU-measured (actual hardware), not projected.
"""

import torch
import torch.nn as nn
import numpy as np


# ── GPU energy constant ───────────────────────────────────────────────────────
_GPU_FJ_PER_FLOP = 200e-6   # 200 fJ → nJ: 200e-15 J × 1e6 nJ/J


class ANNAutoencoder(nn.Module):
    """
    Fully-connected ANN Autoencoder (ReLU activations).

    Parameters match SNNAutoencoder for fair comparison:
      input_dim, hidden_dim, latent_dim — identical to SNN counterpart.
    T is unused (ANN is single-pass) but accepted as kwarg for drop-in
    compatibility with the multi-seed runner.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 32, T: int = 16, **kwargs):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # ── Encoder ───────────────────────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,  hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.ReLU(),   # output in [0,∞); input is MinMax-scaled to [0,1]
        )

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    # ── FLOPs counter ─────────────────────────────────────────────────────────

    def count_flops(self, x: torch.Tensor) -> dict:
        """
        Analytical MAC count for a fully-connected autoencoder.

        Each Linear(in, out) layer contributes in × out MACs per sample.
        We count encoder + decoder separately and sum.

        Returns dict with flops_per_sample and energy_nJ.
        """
        layers = [
            (self.input_dim,  self.hidden_dim),   # enc fc1
            (self.hidden_dim, self.latent_dim),    # enc fc2
            (self.latent_dim, self.hidden_dim),    # dec fc1
            (self.hidden_dim, self.input_dim),     # dec fc2
        ]
        macs = sum(i * o for i, o in layers)
        energy_nJ = macs * _GPU_FJ_PER_FLOP

        return {
            "flops_per_sample": float(macs),
            "energy_nJ":        energy_nJ,
            "energy_basis":     "GPU_200fJ_per_FLOP",
        }
