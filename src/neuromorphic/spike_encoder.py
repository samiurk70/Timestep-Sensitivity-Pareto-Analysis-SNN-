"""
neuromorphic/spike_encoder.py

Rate coding encoder: converts real-valued inputs in [0, 1] to spike trains.

In the main SNN autoencoder, SpikingJelly handles rate coding implicitly
by replicating the input across T time steps. This module provides an
explicit, inspectable encoder used in Cell 3 (demo) and for analysis.

Two encoders are provided:
  RateEncoder  — Bernoulli sampling: spike prob = x (standard rate coding)
  LatencyEncoder — spike time inversely proportional to stimulus intensity
"""

import torch
import numpy as np


class RateEncoder:
    """
    Bernoulli rate encoder.

    For each time step t in {1, ..., T}:
        spike(t) = Bernoulli(x)  where x ∈ [0, 1]

    Higher x → more spikes → higher rate → stronger "signal".
    This is the implicit encoding used inside SNNAutoencoder.forward().

    Parameters
    ----------
    T : number of time steps
    """

    def __init__(self, T: int = 16):
        self.T = T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (batch, features)  values in [0, 1]
        Returns: (T, batch, features)  binary spike tensor
        """
        x_clamped = x.clamp(0.0, 1.0)
        spikes = torch.bernoulli(
            x_clamped.unsqueeze(0).expand(self.T, -1, -1))
        return spikes

    def decode(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        spikes : (T, batch, features)
        Returns: (batch, features) mean firing rate
        """
        return spikes.float().mean(dim=0)

    def spike_rate(self, x: torch.Tensor) -> float:
        """Mean firing rate across all neurons and time steps."""
        spikes = self.encode(x)
        return float(spikes.mean())


class LatencyEncoder:
    """
    Latency (time-to-first-spike) encoder.

    Stronger stimulus (higher x) → earlier spike time.
    Neurons with x=0 never fire within T steps.

    spike_time = floor((1 - x) * T)   for x > 0
    No spike                           for x = 0

    Parameters
    ----------
    T : number of time steps (window)
    """

    def __init__(self, T: int = 16):
        self.T = T

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (batch, features)  values in [0, 1]
        Returns: (T, batch, features)  binary spike tensor (at most 1 spike/neuron)
        """
        x_clamped = x.clamp(0.0, 1.0)
        spike_times = ((1.0 - x_clamped) * self.T).long().clamp(0, self.T - 1)
        # Zero-input neurons: set spike time = T (beyond window, never fires)
        spike_times[x_clamped == 0] = self.T

        spikes = torch.zeros(self.T, *x.shape, device=x.device)
        for t in range(self.T):
            spikes[t] = (spike_times == t).float()
        return spikes


def demo_encoding(save_dir: str | None = None) -> None:
    """
    Visualise rate vs latency encoding for a small example input.
    Useful as a methods figure.
    """
    import matplotlib.pyplot as plt

    x = torch.tensor([[0.1, 0.3, 0.5, 0.7, 0.9]])
    T = 20
    rate_enc    = RateEncoder(T=T)
    latency_enc = LatencyEncoder(T=T)

    rate_spikes    = rate_enc.encode(x).squeeze(1).numpy()     # (T, 5)
    latency_spikes = latency_enc.encode(x).squeeze(1).numpy()  # (T, 5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = [f"x={v:.1f}" for v in x[0].tolist()]

    for ax, spikes, title in [
        (axes[0], rate_spikes,    "Rate Encoding (Bernoulli)"),
        (axes[1], latency_spikes, "Latency Encoding (Time-to-first-spike)"),
    ]:
        for i in range(spikes.shape[1]):
            spike_times = np.where(spikes[:, i])[0]
            ax.scatter(spike_times, np.full_like(spike_times, i),
                       marker="|", s=200, color="#2196F3", linewidths=1.5)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Time step")
        ax.set_title(title)
        ax.set_xlim(-1, T)
        ax.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    if save_dir:
        import os
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/spike_encoding_demo.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
