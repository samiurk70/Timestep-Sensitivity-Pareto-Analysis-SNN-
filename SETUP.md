# Setup Guide

## One-time local setup

### 1. Clone the repo and create your branch
```bash
git clone https://github.com/YOUR_USERNAME/SNN-Abnormality-Detection.git
cd SNN-Abnormality-Detection
git checkout -b experiment/neuromorphic-snn
git push -u origin experiment/neuromorphic-snn
```

### 2. Download the ODDS .mat datasets (run locally once)
```bash
git clone --depth 1 https://github.com/Minqi824/ADBench /tmp/adbench
```
Then copy these files to your **Google Drive** under `SNN-Research/data/`:
- `/tmp/adbench/adbench/datasets/Classical/thyroid.mat`
- `/tmp/adbench/adbench/datasets/Classical/cardiotocography.mat`

NSL-KDD and SMAP download automatically inside the notebook.

---

## Per-session setup (VSCode + local RTX 4070)

1. Open VSCode in your project folder
2. Open `experiment.ipynb`
3. Select kernel: `SNN Research (Python 3.10)` (top-right of notebook)
4. Run **Cell 0** — sets paths, verifies CUDA GPU
5. Run **Cell 2** (smoke test) before committing to full training
6. Run cells in order from Cell 3

No Drive mount, no Colab, no git pull on session start.
Results and checkpoints save locally to `PROJECT_ROOT/results/`.

### First-time kernel setup
```bash
# In your terminal (Python 3.10 — where torch is installed)
C:/Users/samiu/AppData/Local/Programs/Python/Python310/python.exe -m pip install jupyter ipykernel spikingjelly pyod scipy scikit-learn pandas
C:/Users/samiu/AppData/Local/Programs/Python/Python310/python.exe -m ipykernel install --user --name snn-research --display-name "SNN Research (Python 3.10)"
```
Then in VSCode: Command Palette → `Python: Select Interpreter` → pick Python 3.10

### Verify CUDA is working
```bash
C:/Users/samiu/AppData/Local/Programs/Python/Python310/python.exe -c "import torch; print(torch.cuda.get_device_name(0))"
# Should print: YOUR LOCAL GPU
```

---

## Directory layout

```
Project root (local + GitHub)
─────────────────────────────────────────────
experiment.ipynb
src/
  models/
    snn_autoencoder.py
    ann_autoencoder.py
    baselines.py
  data/
    loader.py
  evaluation/
    metrics.py
    trainer.py
    visualise.py
  neuromorphic/
    lif_neuron.py
    spike_encoder.py
scripts/
  verify_datasets.py
  download_odds.py
requirements.txt
SETUP.md
.gitignore

data/                    <- local only, gitignored
  38_thyroid.npz         <- copy from ADBench
  7_Cardiotocography.npz <- copy from ADBench
  nslkdd.npz             <- auto-downloaded
  smap.npz               <- auto-generated
  thyroid_cached.npz     <- built on first load
  cardio_cached.npz

results/                 <- local only, gitignored
  lif_demo.png
  thyroid/
    training_loss.png
    snn_seed0.pt
    ann_seed0.pt
  comparison_all_datasets.png
  t_sensitivity_nslkdd.png
  sparsity_analysis.png
  results.json
  t_sensitivity.json
```

---

## Dataset citations (copy into your paper)

**Thyroid / Cardio (ODDS)**  
Rayana, S. (2016). ODDS Library. Stony Brook University, Department of Computer Sciences. http://odds.cs.stonybrook.edu

**NSL-KDD**  
Tavallaee, M., Bagheri, E., Lu, W., & Ghorbani, A. A. (2009). A detailed analysis of the KDD CUP 99 data set. In *2009 IEEE Symposium on Computational Intelligence for Security and Defense Applications* (CISDA).

**SMAP**  
Hundman, K., Constantinou, V., Laporte, C., Colwell, I., & Soderstrom, T. (2018). Detecting spacecraft anomalies using LSTMs and nonparametric dynamic thresholding. In *Proceedings of KDD 2018*.

**ADBench (dataset mirror)**  
Han, S., et al. (2022). ADBench: Anomaly Detection Benchmark. *NeurIPS 2022 Datasets and Benchmarks Track*.

---

## Energy comparison framing 

> SNN energy consumption is estimated as the product of synaptic operations (SynOps) and the per-operation energy of Intel Loihi 2 (4.6 fJ/SynOp; Davies et al., 2021). ANN energy is estimated from multiply-accumulate operations (MACs) at 200 fJ/FLOP, representative of modern GPU inference (Horowitz, 2014). These figures project the advantage of deploying the SNN on neuromorphic silicon rather than a GPU; direct on-hardware comparison remains future work.
