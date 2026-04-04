"""
models/snn_autoencoder.py

SNN Autoencoder built with SpikingJelly 0.0.0.0.14.

Architecture:
  Encoder: input → Linear → LIF → Linear → LIF   (hidden → latent)
  Decoder: latent → Linear → LIF → Linear → sigmoid (latent → hidden → output)

The decoder output uses sigmoid so reconstruction values are in [0,1],
matching MinMax-scaled inputs. This is critical for MSE loss convergence.

Fixes vs initial version:
  - bias=True on all Linear layers (was False — killed gradients)
  - decoder output: sigmoid not ReLU (bounded output matches [0,1] input)
  - surrogate gradient: ATan (default in SJ 0.0.0.0.14, handles vanishing grads)
  - input replicated across T steps via unsqueeze+expand (explicit rate coding)

Energy accounting:
  SynOps = spike_count × fan_out
  Projected energy (nJ) = SynOps × 4.6e-6 nJ  (Intel Loihi 2, Davies 2021)
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven import neuron, functional, surrogate


_LOIHI2_FJ_PER_SYNOP = 4.6e-6   # 4.6 fJ → nJ


class SNNAutoencoder(nn.Module):
    """
    Fully-connected SNN Autoencoder with LIF neurons (SpikingJelly 0.0.0.0.14).

    Parameters
    ----------
    input_dim  : number of input features
    hidden_dim : encoder/decoder hidden layer width
    latent_dim : bottleneck width
    T          : number of simulation time steps (default 16)
    tau        : LIF membrane time constant (default 2.0)
    threshold  : LIF firing threshold (default 0.5 — lower = more spikes = better gradients)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 32, T: int = 16,
                 tau: float = 2.0, threshold: float = 0.5):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.T          = T

        sg = surrogate.ATan()   # smooth surrogate gradient for backprop

        # ── Encoder ───────────────────────────────────────────────────────────
        self.enc_fc1  = nn.Linear(input_dim,  hidden_dim, bias=True)
        self.enc_lif1 = neuron.LIFNode(tau=tau, v_threshold=threshold,
                                        surrogate_function=sg, detach_reset=True)
        self.enc_fc2  = nn.Linear(hidden_dim, latent_dim, bias=True)
        self.enc_lif2 = neuron.LIFNode(tau=tau, v_threshold=threshold,
                                        surrogate_function=sg, detach_reset=True)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec_fc1  = nn.Linear(latent_dim, hidden_dim, bias=True)
        self.dec_lif1 = neuron.LIFNode(tau=tau, v_threshold=threshold,
                                        surrogate_function=sg, detach_reset=True)
        self.dec_fc2  = nn.Linear(hidden_dim, input_dim,  bias=True)
        # Sigmoid: output in [0,1] matches MinMax-scaled input
        self.dec_out  = nn.Sigmoid()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, input_dim)  values in [0,1]
        Returns reconstructed x, same shape.
        Runs T time steps, accumulates outputs, returns mean.
        """
        functional.reset_net(self)
        out_acc = torch.zeros_like(x)

        for _ in range(self.T):
            h1    = self.enc_lif1(self.enc_fc1(x))
            z     = self.enc_lif2(self.enc_fc2(h1))
            d1    = self.dec_lif1(self.dec_fc1(z))
            recon = self.dec_out(self.dec_fc2(d1))
            out_acc = out_acc + recon

        return out_acc / self.T

    # ── SynOps counter ────────────────────────────────────────────────────────

    def count_synops(self, x: torch.Tensor) -> dict:
        """
        Count synaptic operations and project energy (Loihi 2).
        SynOps = Σ_layer spikes_into_layer × fan_out.
        """
        functional.reset_net(self)
        batch = x.shape[0]
        total_synops = 0.0

        with torch.no_grad():
            for _ in range(self.T):
                h1 = self.enc_lif1(self.enc_fc1(x))
                z  = self.enc_lif2(self.enc_fc2(h1))
                d1 = self.dec_lif1(self.dec_fc1(z))
                _  = self.dec_out(self.dec_fc2(d1))

                total_synops += float(h1.sum()) * self.enc_fc2.weight.shape[0]
                total_synops += float(z.sum())  * self.dec_fc1.weight.shape[0]
                total_synops += float(d1.sum()) * self.dec_fc2.weight.shape[0]

        synops_per_sample = total_synops / batch
        energy_nJ = synops_per_sample * _LOIHI2_FJ_PER_SYNOP

        return {
            "synops_per_sample": synops_per_sample,
            "energy_nJ":         energy_nJ,
            "energy_basis":      "Loihi2_4.6fJ_per_synop",
        }

    # ── Spike rate monitor ────────────────────────────────────────────────────

    def spike_rates(self, x: torch.Tensor) -> dict:
        """Mean firing rate per layer across T steps. Used for sparsity analysis."""
        functional.reset_net(self)
        acc = {"enc_hidden": 0.0, "enc_latent": 0.0, "dec_hidden": 0.0}

        with torch.no_grad():
            for _ in range(self.T):
                h1 = self.enc_lif1(self.enc_fc1(x))
                z  = self.enc_lif2(self.enc_fc2(h1))
                d1 = self.dec_lif1(self.dec_fc1(z))
                _  = self.dec_out(self.dec_fc2(d1))

                acc["enc_hidden"] += float(h1.mean())
                acc["enc_latent"] += float(z.mean())
                acc["dec_hidden"] += float(d1.mean())

        return {k: v / self.T for k, v in acc.items()}
