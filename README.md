# GSHGL — Reproduction Package

**G**ene-**S**upervised **H**ierarchical **G**raph **L**earning


This repository contains a **minimal, self-contained reproduction** of **GSHGL** for WSI-based genomic mutation prediction.

Pipeline:

1. **WSI patching** and **UNI-2h feature extraction** (1536-dim patch embeddings)
2. **Graph construction** from patch features and coordinates
3. **Hierarchical graph encoder** (patch-level + region-level)
4. **Gene-supervised multi-scale contrastive pretraining**
5. **Downstream mutation classification**
6. **Interpretability** (heatmap and top-patch visualization)

---

## Method Overview

```
WSI (.svs)
        │
        ▼
 Tissue patches (256×256 @ 20×)
        │
        ▼
 UNI-2h encoder (frozen, 1536-dim)
        │
        ▼
  Hybrid graph construction
  (spatial + feature neighbors)
        │
        ▼
 HierarchicalGraphEncoder
  ├─ patch-level GAT + pooling
  └─ region-level GAT + pooling
        │
        ▼
 Multi-scale gene-supervised
 contrastive pretraining
        │
        ▼
 MultiScaleGraphClassifier
 (cross-scale fusion)
        │
        ▼
 Mutation status prediction
```

---

## Project Layout

```text
fWCGM_repro/
├── configs/
│   ├── config.py
│   ├── runtime.py
│   ├── dataset/        # e.g. thym_gtf2i.json
│   ├── experiment/     # default.json
│   └── model/          # default_multiscale.json
├── data/
├── data_prep/          # TCGA WSI / MAF / label utilities
├── models/
├── train/
│   ├── pipeline.py     # full training flow
│   ├── pretrain.py
│   ├── contrastive.py
│   └── augmentation.py
├── explain/
├── scripts/
│   ├── data_prep.py            # unified TCGA data-prep CLI
│   ├── extract_patches.py      # WSI -> patch JPEGs
│   ├── extract_uni2h_features.py
│   └── build_labels.py
├── preprocess.py
├── train.py
├── explain.py
└── requirements.txt
```

---

## Environment Setup

```bash
cd fWCGM_repro
pip install -r requirements.txt
```

Requirements:

- Python 3.10+
- CUDA-capable GPU (recommended)
- PyTorch + [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) (follow the official install guide for your CUDA version)

For WSI patching and top-patch visualization, install [OpenSlide](https://openslide.org/) system libraries in addition to `openslide-python`.

**UNI-2h access:** patch features in this project use the [UNI-2h](https://huggingface.co/MahmoodLab/UNI2-h) pathology foundation model (ViT-Giant, **1536-dim** output). You must request model access on Hugging Face and authenticate before the first feature-extraction run:

```bash
pip install huggingface_hub
huggingface-cli login
```

---


## Data Preparation

### Example: TCGA-THYM / GTF2I

The following walks through preparing data for the **TCGA-THYM_GTF2I** task as a concrete example. The same steps apply to other TCGA cohorts by changing the project and gene name.

#### Step 1 — Download WSI from GDC

TCGA whole-slide images (`.svs`) are hosted on the [NCI GDC Data Portal](https://portal.gdc.cancer.gov/).

**Option A: built-in helper (one diagnostic slide per case)**

```bash
python scripts/data_prep.py caselevel-query \
  --project TCGA-THYM \
  --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt

python scripts/data_prep.py caselevel-download \
  --manifest data/TCGA-THYM/MANIFEST_CASELEVEL.txt \
  --out data/TCGA-THYM/wsi
```

**Option B: GDC Data Transfer Tool**

1. Register a GDC account and install [gdc-client](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
2. Download a manifest from the portal and run:

```bash
gdc-client download -m MANIFEST_TCGA-THYM.txt -d data/TCGA-THYM/wsi
```

Expected layout after download:

```text
data/TCGA-THYM/
└── wsi/
    ├── TCGA-4V-A9QS-01A-01-TSA.svs
    ├── TCGA-4V-A9QT-01A-01-TSA.svs
    └── ...
```

#### Step 2 — Tile WSIs into patches

```bash
python scripts/extract_patches.py --dataset-dir data/TCGA-THYM
```

The script detects tissue on a thumbnail, selects an appropriate pyramid level, filters blank/dark patches, and writes JPEG tiles under:

```text
data/TCGA-THYM/patches/
    TCGA-4V-A9QS-01A-01-TSA/
        TCGA-4V-A9QS-01A-01-TSA_4928_16352.jpg
        TCGA-4V-A9QS-01A-01-TSA_5184_16352.jpg
```

Filename format: `{slide_prefix}_{x}_{y}.jpg`.

#### Step 3 — Build gene tables from MAF

Query/download MAF files and convert them to per-case gene TSV tables:

```bash
python scripts/data_prep.py maf-build --dataset-dir data/TCGA-THYM
```

This creates:

```text
data/TCGA-THYM/
├── maf_files/
└── gene/
    └── TCGA-4V-A9QS/
        └── TCGA-4V-A9QS.tsv
```

Each TSV contains `Hugo_Symbol` and `Variant_Classification`. Slides without a recorded GTF2I variant are treated as wild-type (label 0); slides with a somatic variant are labeled 1.

Optional utilities:

```bash
python scripts/data_prep.py gene-summary --gene-root data/TCGA-THYM/gene --topn 20
python scripts/data_prep.py maf-stats --out data/TCGA-THYM
```

#### Step 4 — Extract UNI-2h features

This repository ships a feature-extraction script that encodes all patches with the frozen **UNI-2h** model and writes `data_features.pkl`:

```bash
python scripts/extract_uni2h_features.py \
  --dataset-dir data/TCGA-THYM \
  --gpu 0 \
  --batch-size 256
```

On first run, UNI-2h weights are downloaded to `assets/ckpts/uni2-h/` (requires Hugging Face login and approved access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h)).

Output structure:

```python
{
    "names_list":    ["TCGA-4V-A9QS", ...],       # slide IDs (TCGA-XX-XXXX, no .svs suffix)
    "coords_list":   [ndarray(N, 2), ...],         # level-0 patch coordinates
    "features_list": [ndarray(N, 1536), ...],      # UNI-2h embeddings
}
```

UNI-2h uses 224×224 center-cropped inputs with ImageNet normalization (vendor-recommended inference settings). The default GSHGL encoder expects **1536-dimensional** node features (`configs/model/default_multiscale.json`); do not change this unless you switch encoders.

#### Step 5 — Build `labels_GTF2I.pkl`

Align UNI-2h features with GTF2I mutation labels:

```bash
python scripts/build_labels.py \
  --dataset-dir data/TCGA-THYM \
  --gene GTF2I
```

The final input file must contain:

| Key | Type | Description |
|-----|------|-------------|
| `names_list` | list[str] | Slide identifiers |
| `coords_list` | list[ndarray] | Patch coordinates, shape `(N, 2)` |
| `features_list` | list[ndarray] | Patch features, shape `(N, D)` |
| `labels_list` | list[int] | Binary labels: 0 = wild-type, 1 = mutated |

Place the file at:

```text
data/TCGA-THYM/labels_GTF2I.pkl
```

#### Final data layout

```text
data/
└── TCGA-THYM/
    ├── wsi/                  # raw .svs files (for interpretability)
    ├── maf_files/            # somatic MAF (optional, for label construction)
    ├── gene/                 # per-slide gene TSV (optional, for label construction)
    ├── patches/                # 256×256 tissue patches (.jpg)
    ├── data_features.pkl     # UNI-2h features (from Step 4)
    └── labels_GTF2I.pkl      # training input required by GSHGL
```

---

## Quick Start

All commands below use the TCGA-THYM / GTF2I example. Override `--data-root` if your data lives elsewhere.

### Step 1 — Build graphs

```bash
python preprocess.py \
  --dataset-config thym_gtf2i \
  --run-name repro_exp \
  --gpu 0
```

Outputs:

```text
output/TCGA-THYM_GTF2I/graphs/
├── slides/
└── cache/
    └── cache_graphs_hybrid_GTF2I.pt
```

### Step 2 — Train

```bash
python train.py \
  --dataset-config thym_gtf2i \
  --experiment-config default \
  --model-config default_multiscale \
  --run-name repro_exp \
  --gpu 0
```

Outputs:

```text
output/TCGA-THYM_GTF2I/repro_exp/
├── config_snapshot.json
├── pretrain/
│   └── encoder_final.pt
└── results/
    ├── fold_1/
    │   ├── best_classifier.pt
    │   ├── fold_summary.json
    │   └── epoch_history.json
    ├── ...
    └── summary.json          # mean ± std over all folds
```

### Step 3 — Override presets from CLI

```bash
python train.py \
  --dataset THYM \
  --gene GTF2I \
  --run-name repro_custom \
  --epochs 100 \
  --batch-size 4 \
  --gpu 0
```

---

## Configuration

Three layers:

1. **Defaults** in `configs/config.py`
2. **JSON presets** in `configs/dataset/`, `configs/experiment/`, `configs/model/`
3. **CLI overrides** via `configs/runtime.py`

| Flag | Description |
|------|-------------|
| `--dataset-config` | Load dataset preset (e.g. `thym_gtf2i`) |
| `--experiment-config` | Load experiment preset (e.g. `default`) |
| `--model-config` | Load model preset |
| `--dataset` / `--gene` | Override cancer type and gene |
| `--run-name` | Experiment run identifier |
| `--gpu` | GPU index |
| `--data-root` | Root directory for input data (default: `./data`) |
| `--output-root` | Root directory for outputs (default: `output`) |
| `--epochs` / `--batch-size` / `--lr` | Training hyperparameters |

---

## Interpretability

### Heatmap

```bash
python explain.py heatmap \
  --pt output/TCGA-THYM_GTF2I/graphs/slides/0000_TCGA-4V-A9QS_0.pt \
  --encoder output/TCGA-THYM_GTF2I/repro_exp/pretrain/encoder_final.pt \
  --classifier output/TCGA-THYM_GTF2I/repro_exp/results/fold_1/best_classifier.pt \
  --out output/heatmap.png \
  --cls 1 \
  --method grad_x_input
```

### Top patches (requires WSI file)

```bash
python explain.py top_patches \
  --pt output/TCGA-THYM_GTF2I/graphs/slides/0000_TCGA-4V-A9QS_0.pt \
  --wsi data/TCGA-THYM/wsi/TCGA-4V-A9QS-01A-01-TSA.svs \
  --encoder output/TCGA-THYM_GTF2I/repro_exp/pretrain/encoder_final.pt \
  --classifier output/TCGA-THYM_GTF2I/repro_exp/results/fold_1/best_classifier.pt \
  --out output/top_patches.png \
  --cls 1 \
  --topk 9
```

---

## Reproducing Paper Results

This repository reproduces **GSHGL** (*Gene-Supervised Hierarchical Graph Learning for Weakly Supervised Mutation Prediction from Whole Slide Images*).

1. Prepare patch features and mutation labels following the data steps above.
2. Run `preprocess.py` with matching graph settings (`--graph-tag hybrid` by default).
3. Run `train.py` and compare `results/summary.json` (mean ± std over folds).

Each run writes `config_snapshot.json` under the run directory so hyperparameters can be traced exactly.
