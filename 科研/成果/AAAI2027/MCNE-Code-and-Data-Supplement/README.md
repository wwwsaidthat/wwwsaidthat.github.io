# MCNE Code and Data Supplement

Anonymous review copy for the paper **Matryoshka Graph Contrastive
Learning**. This package contains the training and evaluation code for MCNE,
its ablations, and the comparison methods used by the unified experiment
driver.

## 1. Package contents

- `train_unified_methods.py`: unified training and linear-evaluation entry point.

- `unified_methods/`: MCNE, CDMD, HPEM, DALS, and baseline implementations.

- `pyagc/`: data loading, graph encoders, augmentations, and utility code.

The package intentionally excludes downloaded datasets, checkpoints, result
logs, virtual environments, IDE metadata, and version-control history.

## 2. Environment

The code requires Python 3.10 or later. A CUDA-enabled GPU is strongly
recommended for the three large OGB datasets.

Install PyTorch and its matching PyTorch Geometric packages for the local CUDA
version first, then install the remaining dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Installation commands for PyTorch Geometric can vary with the PyTorch and CUDA
versions. Follow the official PyTorch Geometric installation instructions when
prebuilt extension wheels are needed.

## 3. Data

The experiments use `ogbn-arxiv`, `ogbn-mag`, and `ogbn-products`. No dataset
is distributed with this supplement, and the training code does **not**
download data automatically.

Before running an experiment, download and preprocess the required OGB dataset
locally using the standard OGB/PyG format. With the default `--root ./data`,
the following non-empty directories must already exist:

```text
data/
├── ogbn_arxiv/
├── ogbn_mag/
└── ogbn_products/
```

Only the directory for the dataset being executed is required. The program
checks that the corresponding directory exists before loading it and exits
with an error if the data have not been prepared. A different prepared data
location can be supplied through `--root`.

## 4. Reproducing MCNE

The following commands pass all relevant experiment parameters explicitly.

For `ogbn-arxiv`:

```bash
python train_unified_methods.py \
  --method mcne \
  --dataset arxiv \
  --root ./data \
  --mode neighbor \
  --gpu-id auto \
  --infer-device auto \
  --hidden-dim 768 \
  --num-layers 3 \
  --dropout 0.0 \
  --tau 0.1 \
  --grace-only-epochs 200 \
  --grace-ml-epochs 200 \
  --pretrain-lr 0.001 \
  --grace-ml-lr 0.00001 \
  --pretrain-weight-decay 0.00001 \
  --batch-size 4096 \
  --num-neighbors 10 10 10 \
  --eval-batch-size 256 \
  --eval-num-neighbors -1 -1 -1 \
  --p-feat-mask-1 0.0 \
  --p-edge-drop-1 0.4 \
  --p-feat-mask-2 0.0 \
  --p-edge-drop-2 0.4 \
  --mrl-dims 32,64,128,256,384,512,768 \
  --mrl-weight 1.0 \
  --ml-weight 1.0 \
  --ml-module ml2 \
  --hpem-beta-init 0.1 \
  --hpem-tau-0 0.1
```

For `ogbn-products`:

```bash
python train_unified_methods.py \
  --method mcne \
  --dataset products \
  --root ./data \
  --mode neighbor \
  --gpu-id auto \
  --infer-device auto \
  --hidden-dim 768 \
  --num-layers 2 \
  --dropout 0.0 \
  --tau 0.1 \
  --grace-only-epochs 20 \
  --grace-ml-epochs 20 \
  --pretrain-lr 0.001 \
  --grace-ml-lr 0.00001 \
  --pretrain-weight-decay 0.00001 \
  --batch-size 4096 \
  --num-neighbors 10 10 \
  --eval-batch-size 4096 \
  --eval-num-neighbors 10 10 \
  --p-feat-mask-1 0.0 \
  --p-edge-drop-1 0.5 \
  --p-feat-mask-2 0.0 \
  --p-edge-drop-2 0.5 \
  --mrl-dims 32,64,128,256,384,512,768 \
  --mrl-weight 1.0 \
  --ml-weight 0.1 \
  --ml-module ml2 \
  --hpem-beta-init 0.1 \
  --hpem-tau-0 0.1
```

For `ogbn-mag`:

```bash
python train_unified_methods.py \
  --method mcne \
  --dataset mag \
  --root ./data \
  --mode neighbor \
  --gpu-id auto \
  --infer-device auto \
  --hidden-dim 768 \
  --num-layers 2 \
  --dropout 0.0 \
  --tau 0.1 \
  --grace-only-epochs 50 \
  --grace-ml-epochs 50 \
  --pretrain-lr 0.001 \
  --grace-ml-lr 0.00001 \
  --pretrain-weight-decay 0.00001 \
  --batch-size 4096 \
  --num-neighbors 10 10 \
  --eval-batch-size 256 \
  --eval-num-neighbors 10 10 \
  --p-feat-mask-1 0.0 \
  --p-edge-drop-1 0.6 \
  --p-feat-mask-2 0.0 \
  --p-edge-drop-2 0.6 \
  --mrl-dims 32,64,128,256,384,512,768 \
  --mrl-weight 1.0 \
  --ml-weight 1.0 \
  --ml-module ml2 \
  --hpem-beta-init 0.1 \
  --hpem-tau-0 0.1
```

Each command assumes that the corresponding local data directory has already
been prepared. All experiments use a 768-dimensional encoder, nested
dimensions `32,64,128,256,384,512,768`, `beta_0 = 0.1`, and
`tau_0 = 0.1`.

Training uses seed 0. Linear evaluation is repeated with seeds 0--4. Each run
writes its resolved configuration, log, checkpoint metadata, and evaluation
summary under `results/` and `checkpoints/`.

## 5. Ablation study

The ablation names in the paper correspond to the following configurations:

- `MRL`: `--method grace_mrl`
- `+ CDMD`: `--method mcne --disable-hpem`
- `+ HPEM`: `--method mcne --disable-cdmd --disable-dals`
- `+ HPEM + DALS`: `--method mcne --disable-cdmd`
- `MCNE`: `--method mcne`

## 6. Output files

The main summary files produced for each run are:

- `summary.json`: metrics and resolved run information.
- `accuracy_summary.txt`: compact accuracy statistics.
- `run.log`: training and evaluation log.

Dataset preprocessing must be completed before training. Storage and memory
requirements are governed by the respective OGB datasets.

## 7. Review anonymity

This archive is self-contained with respect to the submitted source code and
does not require an external repository. Author names, personal contact
details, local machine paths, and repository history have been removed.
