# autoresearch

This repository uses a custom AutoResearch contract built on top of a config-driven chest X-ray training pipeline.

## Setup

To set up a new run, work with the user to:

1. Agree on a run tag and create a fresh branch `autoresearch/<tag>`.
2. Read the in-scope files for context:
   - `README.md`
   - `program.md`
   - `prepare.py`
   - `train.py`
   - `experiments/run_experiment.py`
   - `configs/base.yaml`
3. Ensure the dataset is processed to separate Pneumonia cases into **Bacterial** and **Viral** classes based on file names.
4. The training and validation sets must be combined and re-split into an **80% training** and **20% validation** set
5. Verify the local Kaggle dataset exists under `data/raw/chest_xray/`.
6. Run `python3 prepare.py` to validate the dataset and generate `data/processed/prepare_report.json`.
7. Initialize `results.tsv` with:

```tsv
commit   val_accuracy   test_accuracy   status   description
```

6. Confirm the setup looks correct, then begin experimentation.

## Repository contract

This is not the original minimal `prepare.py` + `train.py` text-model setup.

- `prepare.py` is fixed and should not be modified.
- `train.py` is a thin compatibility wrapper.
- The real experiment engine is `experiments/run_experiment.py`.
- Experiments are defined by YAML configs in `configs/`.
- Core implementation lives in `src/chest_xray_project/`.

## What you can change

- `configs/*.yaml`
- `src/chest_xray_project/*.py`
- `experiments/run_experiment.py` if you need a better stable experiment surface

Prefer simpler config changes first. Only edit package code when the experiment really requires it.

## What you should not change

- `prepare.py`
- dataset contents
- generated run artifacts under `outputs/runs/`

## Running experiments

Use:

```bash
python3 train.py --config configs/model_cnn.yaml > run.log 2>&1
```

You may also point `train.py` at a temporary config generated during the loop.

## Result reading

Each run writes artifacts to:

```text
outputs/runs/<experiment_name>/
```

The source of truth is:

- `metrics.json`
- `resolved_config.json`

`train.py` should print a short summary at the end of the run. If needed, inspect the run directory directly.

## Optimization target

**Validation Accuracy** is the primary metric to match the reference report.

Secondary considerations:

- validation accuracy
- generalization to test metrics
- model simplicity
- avoiding unnecessary memory growth

**Target Benchmarks (Test Accuracy):**
**Stage A (CNN Baseline):** ~68%[cite: 311, 358].
**Stage B1 (Frozen EfficientNetB0):** ~78%[cite: 367, 405].
**Stage B2 (Fine-Tuning):** ~79.6%[cite: 414, 451].
**Stage C (Vision Transformer):** ~75%[cite: 460, 490].

## Experiment Roadmap

### Stage A: CNN Baseline
Establish a baseline using three convolutional layers with ReLU activation and max pooling[cite: 307].

### Stage B: EfficientNetB0 Transfer Learning
* **Part 1 (Frozen)**: Use pre-trained ImageNet weights; freeze the base and add a custom dense head (256, 128 units)[cite: 363, 364].
* **Part 2 (Fine-Tuning)**: Unfreeze the base model using a very low learning rate ($1 \times 10^{-5}$) to adapt features to medical X-rays[cite: 409, 412].

### Stage C: Vision Transformer (ViT)
Evaluate an attention-based approach using 32x32 patches and a Multi-Layer Perceptron (MLP) head[cite: 456, 458].


## Logging results

Record each run in `results.tsv` as tab-separated values:

```tsv
commit   val_accuracy   test_accuracy   status   description
```

Where:

1. `commit`: short git hash
2. `val_accuracy`: validation accuracy, use `0.000000` on crash
3. `test_accuracy`: validation test, use `0.000000` on crash
4. `status`: `keep`, `discard`, or `crash`
5. `description`: short summary of the idea

Do not commit `results.tsv`.

## Experiment loop

Loop:

1. Check current branch and commit.
2. Choose one idea (For example Identify Stage: Determine if you are improving the baseline, transfer learning, etc...).
3. Make the change (For example Adjust Config: For Fine-Tuning, prioritize learning rate adjustments.).
4. Commit it.
5. Run `python3 train.py --config ... > run.log 2>&1`.
6. Read summary metrics from `run.log` or `outputs/runs/<experiment_name>/metrics.json`.
7. Validate: Ensure results align with or exceed the reference report benchmarks.
8. Log the result in `results.tsv`.
9. Keep the commit if it improves the target metric.
10. Revert to the previous best commit if it does not improve the target metric.
(Revert/Keep: If a fine-tuning run fails to beat 79% accuracy, analyze the loss curves for overfitting before proceeding.)

## Simplicity criterion

All else equal, prefer simpler changes:

- removing code and matching performance is a win
- small gains with large complexity cost are usually not worth keeping
- clean, understandable experiments beat clever hacks