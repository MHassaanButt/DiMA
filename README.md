# A Differential Memory Attention Mamba for Spatial-Spectral Representation Learning toward Hyperspectral Image Classification

This repository contains code and data associated with the paper:

> **"A Differential Memory Attention Mamba for Spatial-Spectral Representation Learning toward Hyperspectral Image Classification"**, currently under review in *IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTAR)*. The source code will be publicly released upon acceptance of the paper.

## Repository Structure
```
├── dataset/
│   ├── HH/                  # Houston Hyperspectral dataset
│   │   ├── WHU_Hi_HongHu.mat      # HSI data cube
│   │   └── WHU_Hi_HongHu_gt.mat           # Ground truth labels
│   ├── HC/                  # Houston Urban dataset
│   │   ├── WHU_Hi_HanChuan.mat
│   │   └── WHU_Hi_HanChuan_gt.mat
│   ├── Tangdaowan/          # QUH-Tangdaowan UAV dataset
│   │   ├── QUH-Tangdaowan.mat
│   │   └── QUH-Tangdaowan_GT.mat
│   └── Pingan/              # QUH-Pingan UAV dataset
│       ├── QUH-Pingan.mat
│       └── QUH-Pingan_GT.mat
├── ablations/               # Ablation study results
│   ├── patch_sizes/         # Results for different patch (window) sizes
│   ├── training_ratios/     # Results for different train/val/test splits
│   └── attention_mechanisms/ # Results for different attention modules
├── comparative_results/     # Comparison of DiMAMamba against state-of-the-art methods
├── gt_viz/                  # Ground truth visualizations for all datasets
│   └── (exported images .png/.jpg)
├── sample_viz.py            # Script to generate ground truth visualizations
└── DiMA_notebook.ipynb      # Interactive notebook implementing DiMAMamba
```

## Downloading the Datasets

Please download the hyperspectral datasets and place them in the corresponding subfolders under `dataset/` before running any scripts:

- **Wahuan Hyperspectral Dataset (HH & HC)**
  - RSIDEA research group of Wuhan University: [https://rsidea.whu.edu.cn/resource_WHUHi_sharing.htm]

- **Qingdao UAV-borne HSI (QUH) Dataset**
  - Qingdao UAV-borne HSI (Tangdaowan, Qingyun, Pingan): [https://eod-grss-ieee.com/dataset-detail/WUpwZEd5Ukd0Tk5PZ0pmajRYTWJoUT09]

After downloading, each dataset folder should contain two files:
- `dataset.mat` — the hyperspectral image cube (height × width × bands)
- `gt.mat`      — the corresponding ground truth labels (height × width)

## Ablation Studies

The `ablations/` directory contains precomputed results for:  
1. **Patch Sizes** — evaluation with different window sizes (e.g., 2×2, 4×4, …).  
2. **Training Ratios** — evaluation with varying train/val/test splits (5%, 10%, 15%…).  
3. **Attention Mechanisms** — comparison among baseline, Multi-Head Self-Attention, Cross-Attention, Scaled Dot-Product Attention, Additive (Bahdanau) Attention, Simple Self-Attention and Differential Memory Attention modules.

Each subfolder includes logs, metrics tables, and visualizations of classification maps.

## Comparative Results

The `comparative_results/` directory includes detailed tables and figures comparing DiMAMamba’s performance against state-of-the-art models on each benchmark dataset.

## Ground Truth Visualizations

Use the `sample_viz.py` script to regenerate or customize ground truth maps. Example usage:
```bash
python sample_viz.py --dataset HH 
```

## Interactive Notebook
Open ``DiMA_notebook.ipynb`` to explore the full implementation of the DiMAMamba model, training loops, and evaluation pipelines.

Note: This repository is under active development. Code and data organization may evolve upon paper acceptance.

