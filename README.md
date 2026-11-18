# Fashion-MNIST Generative Models (D6421 Session 07)

PyTorch implementations and experiments for Fashion-MNIST generative models.
The repository currently contains the MLP classifier and its analysis; the
remaining generative models are not implemented yet.

## Dataset and split

- Fashion-MNIST, 28×28 grayscale, 10 classes, 70,000 images.
- All notebooks call `data_loader.get_dataloaders()` for the same split:
  56,000 train / 7,000 validation / 7,000 test.
- Batch size 128, seed 42, pixel range `[0, 1]`.
- Checkpoints are selected on validation data. Test data is used only for final
  evaluation and the MLP prior-statistics experiment.

## Repository layout

```text
ASG_2/
├── data/FashionMNIST/          # dataset (not committed)
├── models/
│   ├── mlp_classifier.py
│   └── autoencoder.py
├── notebooks/
│   ├── 01_mlp_classifier.ipynb
│   ├── 02_autoencoder.ipynb
│   └── 08_classifier_analysis.ipynb
├── utils/
│   ├── metrics.py
│   ├── paths.py
│   └── reproducibility.py
├── results/
│   ├── checkpoints/
│   ├── features/
│   ├── figures/
│   └── metrics/
├── data_loader.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The project was developed with Python 3.12 and PyTorch 2.14. On an Intel Arc
GPU, install the XPU build of PyTorch; the shared `get_device()` falls back to
CUDA or CPU automatically.

## Run order

1. `notebooks/01_mlp_classifier.ipynb` — trains the classifier and writes the
   checkpoint plus MLP feature artifacts.
2. `notebooks/02_autoencoder.ipynb` — trains the 64-dimensional Vanilla
   Autoencoder and writes `vanilla_ae_best.pth`.
3. `notebooks/08_classifier_analysis.ipynb` — runs the five classifier
   experiments. Experiment 3 loads the autoencoder from step 2.

Notebooks 03–07 (Sparse AE, VAE, GAN, DCGAN, DDPM) and 09 (comparative
evaluation) are not implemented yet. Do not create empty placeholders for them.

## Artifacts

All artifacts live under the repository-root `results/` directory. Notebooks
resolve the project root by looking for `data_loader.py` and
`models/mlp_classifier.py`, so they can be launched from `notebooks/` or from
the repository root.

| Artifact | Producer |
|---|---|
| `results/checkpoints/mlp_classifier_best.pth` | notebook 01 |
| `results/checkpoints/vanilla_ae_best.pth` | notebook 02 |
| `results/checkpoints/mlp_frozen_decoder_best.pth` | notebook 08 |
| `results/metrics/*.json` | each notebook |
| `results/features/*.pt`, `*.pth` | notebooks 01 and 08 |
| `results/figures/*.png` | notebooks 01, 02 and 08 |

Notebook 08 loads an existing frozen-decoder checkpoint by default. Set
`RETRAIN_DECODER = True` in its configuration cell to retrain and overwrite it.

## Current status

Executed and preserved:

- MLP training and test evaluation (test accuracy 89.34%).
- MLP analysis Experiments 1 and 2.

Implemented and smoke-tested, pending the final training run:

- Vanilla Autoencoder training (notebook 02 was executed for one epoch in an
  isolated temporary directory to verify the code path; no final checkpoint was
  written to `results/`).
- MLP analysis Experiments 3 and 4, which depend on the full autoencoder
  checkpoint.
