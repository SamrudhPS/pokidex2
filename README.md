# pokidex2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SamrudhPS/pokidex2/blob/main/pokidex2.ipynb)

A PyTorch convolutional neural network that classifies images of the original 151 Generation 1 Pokémon, trained on a ~20k-image Kaggle dataset. A follow-up to [pokidex](https://github.com/SamrudhPS/pokidex), this version trains a custom CNN to completion and evaluates it with a full classification report and confusion matrix.

The full workflow — data download, EDA, preprocessing, model definition, training, and evaluation — lives in [`pokidex2.ipynb`](pokidex2.ipynb).

## Dataset

- Source: [`bhawks/pokemon-generation-one-22k`](https://www.kaggle.com/datasets/bhawks/pokemon-generation-one-22k) on Kaggle, downloaded via the Kaggle API.
- **151 classes** (all Generation 1 Pokémon), **20,099 images** total, roughly 89–141 images per species.
- Images are variable-sized RGB photos/artwork, resized to 224×224 for training.
- Split **80% train / 20% validation** via `random_split` (16,079 train / 4,020 validation images).
- Train-time augmentation: random horizontal flip, random rotation (±15°), then normalized with ImageNet mean/std.

## Approach

1. Download and unzip the Kaggle dataset, then load it with `torchvision.datasets.ImageFolder`.
2. Inspect class balance and sample images.
3. Build train/validation `transforms.Compose` pipelines (augmentation on train only) and split into `DataLoader`s (batch size 32).
4. Define two candidate architectures (see below).
5. Train the custom CNN for 12 epochs with `CrossEntropyLoss` + Adam (`lr=1e-3`), tracking train/val loss and accuracy each epoch.
6. Evaluate on the validation set with a confusion matrix, per-class precision/recall/F1, and overall accuracy.

## Model architectures

**`PokemonCNN`** (the model actually trained and evaluated)
- 4 convolutional blocks (Conv2d → BatchNorm2d → ReLU → MaxPool2d), channels growing 32 → 64 → 128 → 256
- Classifier head: Flatten → Dropout(0.4) → Linear(256×8×8 → 256) → ReLU → Dropout(0.3) → Linear(256 → 151)

**ResNet18 (transfer learning)** — set up but not the model used for the final results
- Pretrained `torchvision.models.resnet18` with the backbone frozen and the final fully-connected layer replaced/retrained for 151 classes
- Present in the notebook as an alternative approach; the training/evaluation cells that follow run on `PokemonCNN` instead

## Results

12 epochs, custom `PokemonCNN`, batch size 32:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|---|---|---|---|---|
| 1 | 2.874 | 45.0% | 1.543 | 70.5% |
| 4 | 0.705 | 84.3% | 0.794 | 81.4% |
| 8 | 0.446 | 89.0% | 0.651 | 83.3% |
| 12 | 0.336 | **91.3%** | 0.626 | **84.0%** |

**Final validation accuracy: 84.0%** (weighted F1 ≈ 0.84) across all 151 classes.

Per-class performance is uneven: some species are classified almost perfectly (`Caterpie`, `Squirtle` — precision & recall of 1.00), while visually similar or less-represented ones are weaker (`Alakazam` — 0.48 precision, `Aerodactyl` — 0.45 precision). The notebook prints the full 151-class confusion matrix and classification report for a per-species breakdown.

## Repository structure

```
pokidex2/
├── pokidex2.ipynb   # data download → EDA → preprocessing → model training → evaluation
└── README.md
```

**Note:** this repo does not yet include a `predict.py` script for running inference on new images — training and evaluation currently only happen inside the notebook.

## Getting started

```bash
git clone https://github.com/SamrudhPS/pokidex2.git
cd pokidex2

pip install torch torchvision kaggle scikit-learn matplotlib seaborn pillow numpy
```

You'll need a Kaggle API token (`kaggle.json`) to download the dataset — see the [Kaggle API docs](https://www.kaggle.com/docs/api) for how to generate one. Then:

```bash
kaggle datasets download -d bhawks/pokemon-generation-one-22k
unzip pokemon-generation-one-22k.zip -d ./data
```

```bash
jupyter notebook pokidex2.ipynb
```

Run the cells top to bottom, or open directly in Colab via the badge above. A GPU runtime is recommended.

## Next steps

- Add a standalone `predict.py` for inference on a single image without re-running the notebook.
- Actually train and evaluate the ResNet18 transfer-learning path and compare it against the custom CNN.
- Investigate the weakest classes (e.g. `Alakazam`, `Aerodactyl`) — likely candidates for more training data or stronger augmentation.
- Save the trained model weights (e.g. via `torch.save`) so training doesn't need to be repeated for inference.
