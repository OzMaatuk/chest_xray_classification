# AutoResearch Notes

This repository is compatible with AutoResearch in an adapted form.

## Important clarification

The canonical `prepare.py` + `train.py` AutoResearch template comes from language-model projects that build tokenizers and binary shards. This chest X-ray repository uses a local Kaggle image dataset, so that exact contract does not apply directly.

`prepare.py` is therefore not useless, but it serves a different role here:

- validate that the Kaggle dataset is present locally
- verify the expected folder structure
- inspect label balance
- write a small stable preparation report

The experiment entry point remains:

```bash
PYTHONPATH=src python3 experiments/run_experiment.py --config configs/model_cnn.yaml
```

## Current workflow

1. Run dataset validation:

```bash
python3 prepare.py
```

2. Launch a config-driven experiment:

```bash
PYTHONPATH=src python3 experiments/run_experiment.py --config configs/model_effnet_frozen.yaml
```

3. Inspect run outputs:

```text
outputs/runs/<experiment_name>/
```

## What AutoResearch should treat as fixed vs variable

Fixed:

- `prepare.py`
- reusable package code unless you intentionally decide to research code changes
- dataset location and label mapping contract

Variable:

- `configs/*.yaml`
- model choice
- optimizer settings
- patch size
- fine-tuning depth
- training hyperparameters

## Why this is still a good fit

AutoResearch still gets the same outer-loop benefits:

- stable CLI execution
- saved metrics per run
- reproducible config files
- comparable artifacts across experiments

The main difference is that this repo is centered around a config-driven image pipeline rather than a tiny `train.py`-only loop.
