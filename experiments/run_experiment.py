from __future__ import annotations

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run one chest X-ray experiment.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    return parser.parse_args()


def main():
    args = parse_args()

    import torch

    from chest_xray_project.config import load_config
    from chest_xray_project.constants import CLASS_NAMES
    from chest_xray_project.data import class_distribution
    from chest_xray_project.dataloaders import build_dataloaders
    from chest_xray_project.models import build_model, total_parameter_count, trainable_parameter_count
    from chest_xray_project.training import run_training
    from chest_xray_project.utils import ensure_dir, resolve_device, set_seed, write_json
    from chest_xray_project.visualization import save_confusion_matrix, save_training_curves

    config = load_config(args.config)
    set_seed(config.seed)

    device = resolve_device(config.training.device)
    config.training.device = device
    config.data.seed = config.seed
    config.output.experiment_name = config.name

    output_dir = ensure_dir(Path(config.output.root_dir) / config.output.experiment_name)
    write_json(output_dir / "resolved_config.json", config)

    datamodule = build_dataloaders(config.data, device)
    model = build_model(config.model, config.data.image_size).to(device)

    results = run_training(config, model, datamodule, CLASS_NAMES, output_dir)
    results["model"] = {
        "name": config.model.name,
        "total_parameters": total_parameter_count(model),
        "trainable_parameters": trainable_parameter_count(model),
    }
    results["data"] = {
        "train_size": len(datamodule.splits.train.paths),
        "val_size": len(datamodule.splits.val.paths),
        "test_size": len(datamodule.splits.test.paths),
        "train_distribution": class_distribution(datamodule.splits.train.labels),
        "val_distribution": class_distribution(datamodule.splits.val.labels),
        "test_distribution": class_distribution(datamodule.splits.test.labels),
        "class_weights": datamodule.class_weights,
    }

    if config.output.save_checkpoints:
        torch.save(
            {
                "config": config,
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
            },
            output_dir / "best_model.pt",
        )

    if config.output.save_figures:
        save_training_curves(results["history"], config.name, output_dir / "training_curves.png")
        save_confusion_matrix(
            results["validation"]["confusion_matrix"],
            CLASS_NAMES,
            f"{config.name} Validation",
            output_dir / "validation_confusion_matrix.png",
        )
        save_confusion_matrix(
            results["test"]["confusion_matrix"],
            CLASS_NAMES,
            f"{config.name} Test",
            output_dir / "test_confusion_matrix.png",
        )

    write_json(output_dir / "metrics.json", results)
    print(f"Finished experiment '{config.name}'. Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
