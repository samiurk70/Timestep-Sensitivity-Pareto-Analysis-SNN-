"""
neuromorphic/lif_neuron.py

Standalone LIF (Leaky Integrate-and-Fire) neuron simulation.
Used in Cell 2 (smoke test) and Cell 3 (demo figure) of the notebook.

Provides:
  LIFNeuron  — class with .simulate() method (used in smoke test Cell 2)
  demo()     — full simulation + plot, saves to save_path (used in Cell 3)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ── LIF class (used in smoke test) ───────────────────────────────────────────

class LIFNeuron:
    """
    Single LIF neuron with Euler integration.

    Parameters
    ----------
    tau       : membrane time constant (ms)
    v_thresh  : firing threshold
    v_reset   : reset potential after spike
    v_rest    : resting potential
    dt        : simulation timestep (ms)
    """

    def __init__(self, tau: float = 10.0, v_thresh: float = 1.0,
                 v_reset: float = 0.0, v_rest: float = 0.0,
                 dt: float = 0.1):
        self.tau      = tau
        self.v_thresh = v_thresh
        self.v_reset  = v_reset
        self.v_rest   = v_rest
        self.dt       = dt

    def simulate(self, current_trace) -> list:
        """
        Simulate neuron given a sequence of input currents.

        Parameters
        ----------
        current_trace : list/array of input current values, one per time step

        Returns
        -------
        List of 0/1 spike indicators, same length as current_trace.
        """
        V      = self.v_rest
        spikes = []

        for I in current_trace:
            dV = self.dt * (-(V - self.v_rest) / self.tau + I)
            V += dV
            if V >= self.v_thresh:
                spikes.append(1)
                V = self.v_reset
            else:
                spikes.append(0)

        return spikes


# ── Full simulation + plot ────────────────────────────────────────────────────

def _run_simulation(tau_m=10.0, v_rest=0.0, v_thresh=1.0, v_reset=0.0,
                    dt=0.1, t_total=100.0, I_const=0.15):
    steps  = int(t_total / dt)
    t      = np.arange(steps) * dt
    V      = np.full(steps, v_rest)
    spikes = []

    for i in range(1, steps):
        dV   = dt * (-(V[i-1] - v_rest) / tau_m + I_const)
        V[i] = V[i-1] + dV
        if V[i] >= v_thresh:
            spikes.append(t[i])
            V[i] = v_reset

    return {"t": t, "V": V, "spikes": np.array(spikes),
            "tau_m": tau_m, "v_thresh": v_thresh, "I_const": I_const}


def demo(save_path=None, save_dir=None):
    """
    Run LIF simulation and plot membrane potential + spike raster.

    Accepts either save_path (full file path) or save_dir (directory,
    saves as lif_demo.png inside it). save_path takes priority.

    Returns path to saved figure, or None.
    """
    result = _run_simulation()
    t, V, spikes = result["t"], result["V"], result["spikes"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)

    ax1.plot(t, V, color="#2196F3", linewidth=1.2, label="V(t)")
    ax1.axhline(result["v_thresh"], color="red", linestyle="--",
                linewidth=1.0, alpha=0.7,
                label=f"Threshold ({result['v_thresh']})")
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5,
                label="Rest")
    ax1.set_ylabel("Membrane Potential")
    ax1.set_ylim(-0.1, 1.3)
    ax1.legend(fontsize=9, loc="upper right")
    if len(spikes) > 1:
        isi = float(np.diff(spikes).mean())
        ax1.set_title(
            f"LIF Neuron — tau={result['tau_m']} ms, "
            f"I={result['I_const']}, thresh={result['v_thresh']}\n"
            f"Spikes: {len(spikes)}  |  Mean ISI: {isi:.1f} ms")
    else:
        ax1.set_title(
            f"LIF Neuron — tau={result['tau_m']} ms, "
            f"I={result['I_const']}, thresh={result['v_thresh']}")
    ax1.grid(True, alpha=0.3)

    if len(spikes) > 0:
        ax2.eventplot(spikes, lineoffsets=0.5, linelengths=0.6,
                      colors="#FF5722", linewidths=1.5)
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Spikes")
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()

    out_path = None
    if save_path is not None:
        out_path = save_path
    elif save_dir is not None:
        out_path = str(Path(save_dir) / "lif_demo.png")

    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  LIF demo saved: {out_path}")
        return out_path

    plt.show()
    return None
