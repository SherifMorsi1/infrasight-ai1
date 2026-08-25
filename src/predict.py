from __future__ import annotations

import argparse

import torch
from PIL import Image

from src.data import build_transforms
from src.model import build_model, get_device


def parse_args():
    parser = argparse.ArgumentParser(description="Classify one satellite image with InfraSight AI.")
    parser.add_argument("image", help="Path to an RGB image")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device()

    checkpoint = torch.load(args.checkpoint, map_location=device)
    classes = checkpoint["classes"]
    image_size = checkpoint.get("image_size", 224)

    model = build_model(num_classes=len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, eval_transform = build_transforms(image_size)
    image = Image.open(args.image).convert("RGB")
    tensor = eval_transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1).squeeze(0)

    k = min(args.top_k, len(classes))
    values, indices = torch.topk(probabilities, k=k)

    for rank, (value, index) in enumerate(zip(values.tolist(), indices.tolist()), start=1):
        print(f"{rank}. {classes[index]:<24} {value:.4f}")


if __name__ == "__main__":
    main()
