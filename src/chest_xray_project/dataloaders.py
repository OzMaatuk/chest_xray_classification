from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chest_xray_project.config import DataConfig
from chest_xray_project.data import DatasetBundle, build_splits, compute_class_weights
from chest_xray_project.transforms import build_eval_transforms, build_train_transforms


class ChestXrayDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.file_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


@dataclass
class DataModule:
    splits: DatasetBundle
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    criterion: nn.Module
    class_weights: list[float]


def build_dataloaders(config: DataConfig, device: str) -> DataModule:
    splits = build_splits(
        dataset_root=config.dataset_root,
        train_splits=config.train_splits,
        test_splits=config.test_splits,
        val_size=config.val_size,
        seed=config.seed,
    )

    train_dataset = ChestXrayDataset(
        splits.train.paths,
        splits.train.labels,
        build_train_transforms(config.image_size),
    )
    eval_transform = build_eval_transforms(config.image_size)
    val_dataset = ChestXrayDataset(splits.val.paths, splits.val.labels, eval_transform)
    test_dataset = ChestXrayDataset(splits.test.paths, splits.test.labels, eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device == "cuda",
    )

    weights = compute_class_weights(splits.train.labels, config.viral_weight_multiplier)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    return DataModule(
        splits=splits,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        class_weights=weights.tolist(),
    )

