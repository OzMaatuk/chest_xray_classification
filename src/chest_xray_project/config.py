from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    dataset_root: str = "data/raw/chest_xray"
    train_splits: list[str] = field(default_factory=lambda: ["train", "val"])
    test_splits: list[str] = field(default_factory=lambda: ["test"])
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 2
    val_size: float = 0.15
    seed: int = 42
    viral_weight_multiplier: float = 1.3


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 5e-4
    weight_decay: float = 1e-4


@dataclass
class TrainingConfig:
    epochs: int = 30
    patience: int = 7
    device: str = "auto"


@dataclass
class ModelConfig:
    name: str = "cnn"
    num_classes: int = 3
    freeze_backbone: bool = False
    finetune_blocks: list[int] = field(default_factory=list)
    patch_size: int = 16
    embed_dim: int = 256
    hidden_size: int = 512
    num_layers: int = 2
    dropout: float = 0.2


@dataclass
class OutputConfig:
    root_dir: str = "outputs/runs"
    experiment_name: str = "manual_run"
    save_figures: bool = True
    save_checkpoints: bool = True


@dataclass
class ExperimentConfig:
    name: str = "manual_run"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> ExperimentConfig:
    path = Path(config_path)
    raw = yaml.safe_load(path.read_text()) or {}

    if "base_config" in raw:
        base_path = (path.parent / raw["base_config"]).resolve()
        base_raw = yaml.safe_load(base_path.read_text()) or {}
        raw = _merge_dicts(base_raw, {k: v for k, v in raw.items() if k != "base_config"})

    return ExperimentConfig(
        name=raw.get("name", "manual_run"),
        seed=raw.get("seed", 42),
        data=DataConfig(**raw.get("data", {})),
        optimizer=OptimizerConfig(**raw.get("optimizer", {})),
        training=TrainingConfig(**raw.get("training", {})),
        model=ModelConfig(**raw.get("model", {})),
        output=OutputConfig(**raw.get("output", {})),
    )

