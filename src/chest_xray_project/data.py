from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from chest_xray_project.constants import CLASS_NAMES, IMAGE_EXTENSIONS


def get_label(filepath: str | Path) -> int:
    path = Path(filepath)
    parent = path.parent.name
    name = path.name.lower()

    if parent == "NORMAL":
        return 0
    if parent == "PNEUMONIA":
        if "bacteria" in name:
            return 1
        if "virus" in name:
            return 2
    raise ValueError(f"Cannot infer label for {path}")


def collect_paths_labels(base_dir: str | Path, splits: list[str]) -> tuple[list[str], list[int]]:
    base_path = Path(base_dir)
    paths: list[str] = []
    labels: list[int] = []

    for split in splits:
        for class_dir in ("NORMAL", "PNEUMONIA"):
            folder = base_path / split / class_dir
            if not folder.exists():
                continue

            for item in sorted(folder.iterdir()):
                if item.suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(item))
                    labels.append(get_label(item))

    return paths, labels


@dataclass
class DatasetSplit:
    paths: list[str]
    labels: list[int]


@dataclass
class DatasetBundle:
    train: DatasetSplit
    val: DatasetSplit
    test: DatasetSplit


def build_splits(dataset_root: str | Path, train_splits: list[str], test_splits: list[str], val_size: float, seed: int) -> DatasetBundle:
    pool_paths, pool_labels = collect_paths_labels(dataset_root, train_splits)
    test_paths, test_labels = collect_paths_labels(dataset_root, test_splits)

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        pool_paths,
        pool_labels,
        test_size=val_size,
        stratify=pool_labels,
        random_state=seed,
    )

    return DatasetBundle(
        train=DatasetSplit(train_paths, train_labels),
        val=DatasetSplit(val_paths, val_labels),
        test=DatasetSplit(test_paths, test_labels),
    )


def class_distribution(labels: list[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {CLASS_NAMES[index]: counts.get(index, 0) for index in range(len(CLASS_NAMES))}


def compute_class_weights(labels: list[int], viral_weight_multiplier: float) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    weights = len(labels) / (len(CLASS_NAMES) * counts)
    weights[2] *= viral_weight_multiplier
    return weights

