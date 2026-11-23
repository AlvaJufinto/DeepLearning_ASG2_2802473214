<!-- @format -->

# Fashion-MNIST Generative Models

PyTorch implementations and experiments for Fashion-MNIST generative models (D6421 Session 07).

## Additional information

Name : Stanislaus Alva Jufinto
Class : LA01
NIM : 2802473214

## Dataset

- Fashion-MNIST: 70,000 grayscale images (28×28), 10 classes.
- Split: 56,000 training, 7,000 validation, 7,000 test.
- Batch size: 128.
- Random seed: 42.
- Pixel range: `[0, 1]`.

All notebooks use `data_loader.get_dataloaders()` to ensure consistent data splits. Validation data is used for checkpoint selection, while test data is reserved for final evaluation and the MLP prior-statistics experiment.

## Setup

```bash
pip install -r requirements.txt
```

NB: I used XPU (Intel Arc) for this experiment.

## Run Order

Run the notebooks in this order:

1. `01_mlp_classifier.ipynb`: trains the MLP classifier and extracts features.
2. `02_autoencoder.ipynb`: trains the 64-dimensional Vanilla Autoencoder.
3. Run the 0-7 notebooks to have all of the results first.
4. `08_classifier_analysis.ipynb`: runs five classifier experiments. Requires 0-7 notebooks

## Results

All generated artifacts are stored in the repository's `results/` directory.

| Directory      | Contents                           |
| -------------- | ---------------------------------- |
| `checkpoints/` | Model checkpoints                  |
| `features/`    | Extracted feature tensors          |
| `figures/`     | Generated plots and visualizations |
| `metrics/`     | Evaluation metrics in JSON format  |

Main checkpoints:

- `mlp_classifier_best.pth`
- `vanilla_ae_best.pth`
- `mlp_frozen_decoder_best.pth`

Notebooks automatically resolve the project root, so they can be launched from either the repository root or `notebooks/`.
