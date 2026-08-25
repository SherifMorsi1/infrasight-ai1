from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from src.data import create_dataloaders
from src.model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained InfraSight AI checkpoint.")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)
    seed = checkpoint.get("seed", 42)

    loaders = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        seed=seed,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    targets = []

    with torch.inference_mode():
        for images, labels in loaders.test:
            images = images.to(device)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    correct = sum(int(pred == target) for pred, target in zip(predictions, targets))
    accuracy = correct / len(targets)
    macro_f1 = f1_score(targets, predictions, average="macro")
    report = classification_report(
        targets,
        predictions,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "test_accuracy": accuracy,
        "macro_f1": macro_f1,
        "num_test_samples": len(targets),
        "classification_report": report,
    }

    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    matrix = confusion_matrix(targets, predictions)
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=range(len(classes)),
        yticks=range(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title="InfraSight AI — EuroSAT Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="white" if matrix[row, col] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1:      {macro_f1:.4f}")
    print(f"Saved metrics to {output_dir / 'test_metrics.json'}")
    print(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
