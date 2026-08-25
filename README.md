# InfraSight AI

Deep learning for satellite land-use classification using **PyTorch**, **transfer learning**, and **EuroSAT** imagery.

InfraSight AI fine-tunes a pretrained ResNet18 to classify satellite images into 10 land-use / land-cover categories such as residential areas, highways, industrial zones, forests, rivers, and agricultural land. The project is designed as a clean, reproducible computer-vision pipeline rather than a single notebook.

## Why this project matters

Satellite understanding is useful in smart-city planning, infrastructure monitoring, environmental analysis, disaster response, renewable-energy site assessment, and geospatial AI. This repository demonstrates an end-to-end deep-learning workflow that can be discussed in internship interviews and expanded into a larger research or engineering project.

## What it demonstrates

- PyTorch training and inference
- Transfer learning with ImageNet-pretrained ResNet18
- Data augmentation and normalization
- Reproducible train / validation / test splits
- GPU, Apple Silicon MPS, and CPU support
- AdamW optimization and cosine learning-rate scheduling
- Best-checkpoint saving based on validation accuracy
- Accuracy, macro F1, per-class metrics, and confusion matrix
- Command-line training, evaluation, and single-image inference

## Dataset

The project uses the **EuroSAT RGB** dataset, available directly through `torchvision.datasets.EuroSAT`. It contains 27,000 Sentinel-2 satellite images across 10 classes.

Classes:

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`

The dataset is downloaded automatically the first time training or evaluation runs.

## Project structure

```text
infrasight-ai1/
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data.py
    ├── model.py
    ├── train.py
    ├── evaluate.py
    └── predict.py
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Train

```bash
python -m src.train --epochs 10 --batch-size 64
```

Useful options:

```bash
python -m src.train \
  --epochs 15 \
  --batch-size 64 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --seed 42
```

The best model is saved to:

```text
artifacts/best_model.pt
```

Training history is saved to:

```text
artifacts/history.json
```

## Evaluate

```bash
python -m src.evaluate --checkpoint artifacts/best_model.pt
```

This creates:

- `artifacts/test_metrics.json`
- `artifacts/confusion_matrix.png`

## Predict one image

```bash
python -m src.predict path/to/satellite_image.jpg \
  --checkpoint artifacts/best_model.pt \
  --top-k 3
```

Example output:

```text
1. Residential            0.9124
2. Industrial             0.0617
3. Highway                0.0146
```

## Model

The network is a ResNet18 initialized with ImageNet weights. Its final fully connected layer is replaced with a 10-class classifier. The full network is fine-tuned using cross-entropy loss and AdamW.

## Results

This repository intentionally does **not** publish invented metrics. Train the model on your machine or GPU environment, then run the evaluation command and add the measured test accuracy, macro F1, and confusion matrix here.

Suggested results table after training:

| Metric | Score |
|---|---:|
| Test accuracy | _run evaluation_ |
| Macro F1 | _run evaluation_ |

## CV-ready description

**InfraSight AI — Satellite Image Classification | PyTorch, Computer Vision**  
Built an end-to-end deep-learning pipeline using PyTorch and transfer learning with ResNet18 to classify EuroSAT satellite imagery across 10 land-use categories; implemented reproducible data splits, augmentation, checkpointing, GPU/MPS support, macro-F1 evaluation, and confusion-matrix analysis.

> After training, replace the generic wording with your actual measured accuracy. Never claim a metric you did not reproduce.

## Strong next extensions

1. Compare ResNet18 with EfficientNet-B0 or ConvNeXt-Tiny.
2. Add Grad-CAM explainability to visualize what image regions drive predictions.
3. Track experiments with Weights & Biases or MLflow.
4. Export the best model to ONNX for deployment.
5. Build a Streamlit demo where users upload satellite images.
6. Move from classification to object detection or semantic segmentation for roads, buildings, solar farms, or damaged infrastructure.

## License

MIT
