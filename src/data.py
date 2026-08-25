from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: list[str]


class TransformSubset(Dataset):
    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, index: int):
        image, label = self.subset[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def create_dataloaders(
    data_dir: str | Path = "data",
    batch_size: int = 64,
    image_size: int = 224,
    seed: int = 42,
    num_workers: int = 2,
) -> DataLoaders:
    data_dir = Path(data_dir)
    base_dataset = datasets.EuroSAT(root=data_dir, download=True)
    classes = list(base_dataset.classes)

    n = len(base_dataset)
    train_size = int(0.70 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = torch.utils.data.random_split(
        base_dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )

    train_transform, eval_transform = build_transforms(image_size)
    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, eval_transform)
    test_dataset = TransformSubset(test_subset, eval_transform)

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    return DataLoaders(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        classes=classes,
    )
