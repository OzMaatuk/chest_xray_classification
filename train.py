from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = "configs/model_cnn.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Thin compatibility wrapper over the config-driven experiment runner."
    )
    parser.add_argument(
        "--config",
        default=os.environ.get("TRAIN_CONFIG", DEFAULT_CONFIG),
        help="Experiment config to run. Defaults to configs/model_cnn.yaml.",
    )
    return parser.parse_args()


def merge_dicts(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_resolved_config(config_path: Path) -> dict:
    raw = yaml.safe_load(config_path.read_text()) or {}
    if "base_config" not in raw:
        return raw

    base_path = (config_path.parent / raw["base_config"]).resolve()
    base_raw = yaml.safe_load(base_path.read_text()) or {}
    return merge_dicts(base_raw, {k: v for k, v in raw.items() if k != "base_config"})


def print_summary(metrics: dict, output_dir: Path) -> None:
    validation = metrics.get("validation", {})
    test = metrics.get("test", {})
    model = metrics.get("model", {})

    print("---")
    print(f"output_dir:         {output_dir}")
    print(f"val_macro_f1:       {validation.get('macro_f1', 0.0):.6f}")
    print(f"val_accuracy:       {validation.get('accuracy', 0.0):.6f}")
    print(f"test_macro_f1:      {test.get('macro_f1', 0.0):.6f}")
    print(f"test_accuracy:      {test.get('accuracy', 0.0):.6f}")
    print(f"training_seconds:   {metrics.get('train_time_seconds', 0.0):.1f}")
    print(f"num_params:         {model.get('total_parameters', 0)}")
    print(f"trainable_params:   {model.get('trainable_parameters', 0)}")


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    config_path = (project_root / args.config).resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    env = dict(os.environ)
    src_path = str(project_root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    subprocess.run(
        [
            sys.executable,
            str(project_root / "experiments" / "run_experiment.py"),
            "--config",
            str(config_path),
        ],
        check=True,
        cwd=project_root,
        env=env,
    )

    resolved_config = load_resolved_config(config_path)
    experiment_name = resolved_config.get("name", config_path.stem)
    output_dir = project_root / "outputs" / "runs" / experiment_name
    metrics_path = output_dir / "metrics.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics file not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text())
    print_summary(metrics, output_dir)


if __name__ == "__main__":
    main()
