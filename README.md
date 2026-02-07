# Hybrid CNN + Attention Model for EEG Decoding

Publication-ready codebase for ablation experiments comparing CNN-only, CNN+SE, CNN+CBAM,
and CNN+MHA architectures on BCI Competition IV 2a motor imagery data.

---

## Project Structure

```
eeg-hybrid-attention/
├── configs/
│   └── default.yaml          # All hyperparameters & experiment settings
├── data/
│   ├── __init__.py
│   ├── download_bciciv2a.py   # Download + preprocess BCI-IV-2a
│   └── bciciv2a_dataset.py    # PyTorch Dataset for BCI-IV-2a
├── models/
│   ├── __init__.py
│   ├── eegnet.py              # EEGNet baseline
│   ├── deep_convnet.py        # Deep ConvNet baseline
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── se_block.py        # Squeeze-and-Excitation
│   │   ├── cbam.py            # CBAM (channel + spatial)
│   │   └── mha.py             # Lightweight multi-head attention
│   └── hybrid_cnn.py          # Hybrid CNN + pluggable attention
├── trainers/
│   ├── __init__.py
│   └── trainer.py             # Training / evaluation loop
├── utils/
│   ├── __init__.py
│   ├── metrics.py             # Accuracy, balanced acc, ROC-AUC
│   ├── statistics.py          # Wilcoxon, mixed-effects helpers
│   └── reproducibility.py     # Seed fixing, checkpoint helpers
├── visualizations/
│   ├── __init__.py
│   ├── attention_maps.py      # Attention weight visualization
│   ├── saliency.py            # Integrated gradients / Grad-CAM 1D
│   └── summary_plots.py       # Bar charts, box plots for results
├── scripts/
│   ├── run_all_ablations.sh   # One-command ablation sweep
│   └── run_single.py          # Train single model
├── results/
│   ├── figures/
│   ├── logs/
│   └── checkpoints/
├── notebooks/
│   └── analysis.ipynb         # Post-hoc analysis notebook (placeholder)
├── requirements.txt
├── environment.yml            # Conda environment spec
├── Dockerfile                 # For RunPod / reproducibility
├── .gitignore
└── README.md
```

---

## Quick Start (Local or RunPod)

```bash
# 1. Clone & enter
git clone <your-repo-url> && cd eeg-hybrid-attention

# 2. Environment
conda env create -f environment.yml
conda activate eeg-hybrid

# 3. Download data
python -m data.download_bciciv2a --output_dir ./data/raw

# 4. Run ALL ablations (all subjects × all models)
bash scripts/run_all_ablations.sh

# 5. Results land in results/
```

---

## Setting Up RunPod GPU + VSCode (Remote SSH)

### Step 1 — Create a RunPod Pod

1. Go to [runpod.io](https://runpod.io) → **Pods** → **+ New Pod**.
2. Pick a GPU template (RTX 4090 or A100 recommended).
3. Choose the **RunPod PyTorch 2.x** template (comes with CUDA + PyTorch).
4. Under **Expose Ports**, add port **22** (SSH). Click **Deploy**.
5. Wait for status **Running**.

### Step 2 — Get SSH Credentials

1. Click your pod → **Connect** → **SSH over exposed TCP**.
2. You'll see something like:
   ```
   ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519
   ```
3. If you haven't added an SSH key yet:
   - On your local machine: `ssh-keygen -t ed25519`
   - Copy `~/.ssh/id_ed25519.pub` into RunPod **Settings → SSH Keys**.

### Step 3 — Connect VSCode

1. Install the **Remote - SSH** extension in VSCode.
2. Open Command Palette → **Remote-SSH: Add New SSH Host**.
3. Paste: `ssh root@<IP> -p <PORT>`
4. Select the config file to save to.
5. Command Palette → **Remote-SSH: Connect to Host** → choose the pod.
6. VSCode opens a remote window on the pod.

### Step 4 — Set Up the Project on the Pod

```bash
# Inside the RunPod terminal (via VSCode)
cd /workspace
git clone <your-repo-url> && cd eeg-hybrid-attention
pip install -r requirements.txt
python -m data.download_bciciv2a --output_dir ./data/raw
bash scripts/run_all_ablations.sh
```

### Tips

- **Persistent storage**: Use `/workspace` — it survives pod restarts.
- **Spot instances**: ~60% cheaper; use checkpointing (already built in).
- **Monitoring**: `nvidia-smi -l 1` in a separate terminal.
- **Sync results locally**: `scp -P <PORT> -r root@<IP>:/workspace/eeg-hybrid-attention/results ./results_remote`

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Override via CLI:

```bash
python scripts/run_single.py --config configs/default.yaml \
    model.attention_type=se \
    training.lr=0.001 \
    data.subject_id=1
```

---

## Citation

If you use this code, please cite the accompanying paper (BibTeX TBD).
