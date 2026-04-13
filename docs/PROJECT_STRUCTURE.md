# Project Structure

## Main folders

- `src/chest_xray_project/`: reusable package code
- `configs/`: experiment definitions
- `experiments/`: CLI entry points
- `outputs/`: run artifacts
- `docs/`: project and migration notes

## Module map

- `config.py`: YAML loading into dataclasses
- `data.py`: dataset discovery, label inference, class weights, split creation
- `dataloaders.py`: PyTorch datasets and loaders
- `transforms.py`: train and evaluation transforms
- `models.py`: CNN, EfficientNet, and Patch-LSTM models
- `training.py`: fit loop, evaluation, metrics
- `visualization.py`: saved plots
- `utils.py`: seed handling, directories, JSON writing

## Expected outputs per run

Each experiment directory under `outputs/runs/` is designed to contain:

- `resolved_config.json`
- `metrics.json`
- `best_model.pt`
- `training_curves.png`
- `validation_confusion_matrix.png`
- `test_confusion_matrix.png`

