# Timestep-Sensitivity Pareto Analysis of Spiking Neural Network 
# Autoencoders for Energy-Efficient Anomaly Detection
### A Simulation Study with Projected Neuromorphic Energy

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-orange)
![SpikingJelly](https://img.shields.io/badge/SpikingJelly-0.0.0.0.14-green)
![arXiv](https://img.shields.io/badge/arXiv-preprint-red)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

> **Samiur Rahman Khan**
> GitHub: [samiurk70](https://github.com/samiurk70) | 
> Scholar: [ddQa7D4AAAAJ](https://scholar.google.co.uk/citations?user=ddQa7D4AAAAJ)

---

## Abstract

Spiking Neural Networks offer a promising pathway to energy-efficient 
anomaly detection on resource-constrained hardware, yet the timestep 
parameter T — which governs the tradeoff between detection accuracy and 
inference energy in Leaky Integrate-and-Fire architectures — has not been 
systematically characterised as a design variable in prior work. This paper 
presents a Pareto frontier analysis of T across {4, 8, 16, 32} on four 
benchmark datasets spanning anomaly rates from 2.5 to 54.9 percent, 
evaluating a LIF-based SNN autoencoder against an architecturally equivalent 
ANN and three classical baselines under five-seed multi-seed evaluation. 
The primary finding is that T=8 constitutes a Pareto-efficient operating 
point on three of four datasets, achieving near-peak F1 at projected 
inference energies 14.5 to 19.5 times lower than the ANN baseline.

---

## Research Contributions

1. **T-sensitivity Pareto frontier analysis** — first systematic 
   characterisation of T as a design variable on an accuracy-energy 
   frontier for reconstruction-based anomaly detection with SNNs

2. **Per-layer spike sparsity analysis** — mechanistic account of 
   projected energy savings through layer-wise silent neuron fractions 
   (47 to 57 percent across datasets)

3. **Multi-seed statistical evaluation** — five independent seeds with 
   mean and standard deviation reported throughout, addressing the 
   reproducibility weakness in published SNN benchmarks

4. **Honest energy framing** — GPU-measured ANN energy separated from 
   Loihi-2-projected SNN energy with explicit sensitivity analysis across 
   3.2 to 6.0 fJ/SynOp constant assumptions

---

## Results Summary

| Dataset | SNN F1 | ANN F1 | Energy Reduction |
|---|---|---|---|
| Thyroid | 0.7572 ± 0.0235 | 0.6746 ± 0.2929 | 14.5× |
| Cardio | 0.5160 ± 0.0161 | 0.3728 ± 0.0670 | 16.3× |
| NSL-KDD | 0.9468 ± 0.0040 | 0.8823 ± 0.0177 | 19.5× |
| UNSW-NB15 | 0.5481 ± 0.0065 | 0.1863 ± 0.1457 | 16.6× |

*SNN energy: Loihi-2 projected (4.6 fJ/SynOp, Davies et al. 2021)*
*ANN energy: GPU-measured estimate (200 fJ/FLOP, Horowitz 2014)*
*All F1 figures: mean ± std across 5 independent random seeds*

---

## T-Sensitivity Finding

The central finding of this work. T=8 is Pareto-efficient on 3 of 4 
datasets — near-peak F1 at half the projected energy cost of T=16.

| Dataset | T=4 F1 | T=8 F1 | T=16 F1 | T=32 F1 |
|---|---|---|---|---|
| Thyroid | 0.7179 | **0.7556** | 0.7294 | 0.7301 |
| Cardio | 0.5359 | 0.5197 | **0.5542** | 0.5354 |
| NSL-KDD | 0.9402 | **0.9437** | 0.9492 | 0.9420 |
| UNSW-NB15 | 0.5357 | **0.5482** | 0.5549 | 0.5543 |

Projected energy scales approximately linearly with T. T=8 offers 
the best accuracy-per-nanojoule on Thyroid and NSL-KDD. T=4 is 
preferred on UNSW-NB15 where F1 is flat across all T values.

---

## Spike Sparsity

| Dataset | Enc Hidden | Enc Latent | Dec Hidden | Silent Fraction |
|---|---|---|---|---|
| Thyroid | 29.9% | 60.8% | 39.4% | **56.6%** |
| Cardio | 40.7% | 62.7% | 35.8% | **53.6%** |
| NSL-KDD | 46.1% | 47.1% | 35.0% | **57.3%** |
| UNSW-NB15 | 47.8% | 59.7% | 51.6% | **47.0%** |

Silent neurons contribute zero synaptic operations and zero projected 
energy on neuromorphic hardware. The energy reduction ratios reported 
above reflect the combined effect of lower per-operation cost and 
substantially reduced operation count from sparse activation.

---

## Architecture
Input (d) → [LIF Hidden 64] → [LIF Latent 16] → [LIF Hidden 64] → Output (d)
LIF parameters: tau=2.0 | threshold=0.5 | reset=zero
Training:       surrogate gradient (arctangent) | Adam lr=0.001
Reconstruction: mean membrane potential across T timesteps
Loss:           MSE reconstruction on normal-only training split

ANN baseline uses identical skeleton with ReLU activations and no 
temporal dimension. All capacity, objective, and training budget 
parameters are matched exactly to isolate the effect of spiking 
computation.

---

## Datasets

| Dataset | Samples | Features | Anomaly Rate | Source |
|---|---|---|---|---|
| Thyroid | 3,772 | 6 | 2.5% | ODDS via ADBench |
| Cardiotocography | 2,114 | 21 | 22.0% | ODDS via ADBench |
| NSL-KDD | 10,000 | 38 | 46.5% | Public repository |
| UNSW-NB15 | 50,000 | 39 | 54.9% | Moustafa & Slay 2015 |

---

## Repository Structure

Timestep-Sensitivity-Pareto-Analysis-SNN-/

├── src/
│   ├── models/
│   │   ├── snn_autoencoder.py     # LIF autoencoder implementation
│   │   ├── ann_autoencoder.py     # ANN baseline
│   │   └── baselines.py           # IsoForest, OCSVM, LOF
│   ├── data/
│   │   └── loader.py              # DataManager — all four datasets
│   ├── evaluation/
│   │   ├── trainer.py             # Training loop
│   │   └── metrics.py             # F1, AUC, multi-seed evaluation
│   ├── visualise/
│   │   └── visualise.py           # All figure generation functions
│   └── neuromorphic/
│       └── lif_neuron.py          # Standalone LIF neuron demo
├── data/                          # Dataset files (see setup below)
├── results/                       # Saved JSON results and figures
│   ├── results.json               # Main experiment results
│   ├── t_sensitivity.json         # T-sweep results
│   └── *.png                      # Generated figures
├── experiment.ipynb               # Main experimental notebook
├── energy_sensitivity_figure.py   # Standalone Fig. 2 generation script
├── requirements.txt               # Exact package versions
└── README.md

---

## Setup and Reproduction

### Requirements
```bash
pip install -r requirements.txt
```

Core dependencies:

torch==2.6.0+cu124
spikingjelly==0.0.0.0.14
scikit-learn
numpy
pandas
matplotlib
scipy

GPU is recommended. The full experiment was run on an NVIDIA RTX 4070 
Laptop GPU (8GB VRAM) with CUDA 12.4. CPU execution is supported but 
training will be substantially slower.

### Dataset setup

**Thyroid and Cardiotocography** — download via ADBench:
```bash
python scripts/download_odds.py
```

Then copy the generated `.npz` files to `./data/`:
- `38_thyroid.npz`
- `7_Cardiotocography.npz`

**NSL-KDD** — downloads automatically on first run.

**UNSW-NB15** — place `unswnb15.csv` in `./data/`. 
The preprocessed file used in this study contains 50,000 samples 
with 39 features and binary labels. Original dataset available at 
[UNSW Research](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

### Running the experiment

Open `experiment.ipynb` in VS Code or Jupyter and run cells sequentially.

- **Cell 0** — environment setup and GPU verification
- **Cell 1** — imports
- **Cell 2** — smoke test (approximately 15 seconds)
- **Cell 3** — LIF neuron demo (Fig. 1)
- **Cell 4** — dataset loading
- **Cell 5** — experiment configuration
- **Cell 6** — main experiment: SNN + ANN + baselines, 5 seeds
- **Cell 7** — save results JSON and comparison figure
- **Cell 8** — T-sensitivity sweep
- **Cell 9** — spike sparsity analysis
- **Cell 10** — final summary table

To regenerate the energy sensitivity figure (Fig. 2) without 
rerunning the full experiment:
```bash
python energy_sensitivity_figure.py
```

This loads from the saved `results/results.json` directly.

---

## Expected Runtime

Full experiment on RTX 4070 Laptop GPU:

| Component | Approximate time |
|---|---|
| Main experiment (5 seeds × 4 datasets) | 45 to 60 minutes |
| T-sensitivity sweep (3 seeds × 4 T × 4 datasets) | 90 to 120 minutes |
| Sparsity analysis | 5 minutes |
| Energy sensitivity figure | Under 1 minute |

---

## Citation

If you use this code or build on this work please cite the arXiv preprint:
```bibtex
@misc{khan2025snn,
  author    = {Samiur Rahman Khan},
  title     = {Timestep-Sensitivity Pareto Analysis of Spiking Neural 
               Network Autoencoders for Energy-Efficient Anomaly 
               Detection: A Simulation Study with Projected 
               Neuromorphic Energy},
  year      = {2025},
  publisher = {arXiv},
  note      = {arXiv preprint},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## Scope and Limitations

All energy figures in this paper are projections derived from published 
hardware characterisation data applied to GPU-measured operation counts. 
They are not direct measurements of power consumption on neuromorphic 
hardware. The conclusions are presented as a simulation study that 
establishes a projected efficiency advantage large enough to motivate 
hardware validation, not as a demonstration of that advantage on 
deployed systems.

Direct hardware measurement on Loihi-2, architecture ablation across 
hidden dimensions, threshold robustness analysis, and extension to 
temporal anomaly detection datasets are identified as the primary 
extensions for the journal version of this work.

---

## Related Publications

- Khan S. et al. — Blockchain empowered decentralized application 
  development platform. ICCA 2020. [Cited 14×]
- Sohan M., Khan S. et al. — Secured smart IoT using lightweight 
  blockchain. arXiv 2022. [Cited 13×]
- Khan S., Al-Amin M. — Novel identity check using W3C standards 
  and hybrid blockchain. IJIEEB Vol.15 No.4, 2023.

---

## Author

**Samiur Rahman Khan**
