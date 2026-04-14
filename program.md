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
3. Verify the local Kaggle dataset exists under `data/raw/chest_xray/`.
4. Run `python3 prepare.py` to validate the dataset and generate `data/processed/prepare_report.json`.
5. Initialize `results.tsv` with:

```tsv
commit	val_macro_f1	val_accuracy	status	description
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

Use validation Macro-F1 as the main metric unless the user specifies otherwise.

Secondary considerations:

- validation accuracy
- generalization to test metrics
- model simplicity
- avoiding unnecessary memory growth

## Logging results

Record each run in `results.tsv` as tab-separated values:

```tsv
commit	val_macro_f1	val_accuracy	status	description
```

Where:

1. `commit`: short git hash
2. `val_macro_f1`: validation macro F1, use `0.000000` on crash
3. `val_accuracy`: validation accuracy, use `0.000000` on crash
4. `status`: `keep`, `discard`, or `crash`
5. `description`: short summary of the idea

Do not commit `results.tsv`.

## Experiment loop

Loop:

1. Check current branch and commit.
2. Choose one idea.
3. Make the change.
4. Commit it.
5. Run `python3 train.py --config ... > run.log 2>&1`.
6. Read summary metrics from `run.log` or `outputs/runs/<experiment_name>/metrics.json`.
7. Log the result in `results.tsv`.
8. Keep the commit if it improves the target metric.
9. Revert to the previous best commit if it does not improve the target metric.

## Simplicity criterion

All else equal, prefer simpler changes:

- removing code and matching performance is a win
- small gains with large complexity cost are usually not worth keeping
- clean, understandable experiments beat clever hacks

