# AutoResearch Migration Notes

This repository has been reshaped from a single notebook-export script into a small config-driven project so an automated research loop can operate on it.

## What changed

- Training now has a single CLI entry point: `experiments/run_experiment.py`
- Experiments are defined by YAML files in `configs/`
- Reusable code lives under `src/chest_xray_project/`
- Each run writes its own config, metrics, checkpoint, and figures to `outputs/runs/<experiment_name>/`

## Why this fits AutoResearch better

AutoResearch-style agents need a stable command they can rerun while changing only configuration. The new workflow is:

1. Choose or modify a config file
2. Run one experiment
3. Read `metrics.json`
4. Compare outputs
5. Propose the next experiment

## Suggested agent loop

```bash
python3 experiments/run_experiment.py --config configs/model_effnet_frozen.yaml
python3 experiments/run_experiment.py --config configs/model_effnet_finetune.yaml
```

## Remaining manual step

The dataset still needs to exist at `data/raw/chest_xray/` with the Kaggle folder structure:

```text
data/raw/chest_xray/
  train/
  val/
  test/
```

