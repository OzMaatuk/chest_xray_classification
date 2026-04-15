# Chest X-Ray AutoResearch Project

This project migrates the original chest X-ray notebook workflow into a simple AutoResearch-friendly structure.

## Quick start

1. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place the Kaggle dataset at `data/raw/chest_xray/`

3. Validate the local dataset:

```bash
python3 prepare.py
```

4. Run an experiment:

```bash
PYTHONPATH=src python3 experiments/run_experiment.py --config configs/model_cnn.yaml
```

Or use the thin compatibility wrapper:

```bash
python3 train.py --config configs/model_cnn.yaml
```

5. Inspect artifacts in `outputs/runs/<experiment_name>/`

## Colab

The preferred Colab workflow is [notebooks/colab_experiment_suite.ipynb](/home/oz/Workspace/aya/notebooks/colab_experiment_suite.ipynb:1).

## Available experiment configs

- `configs/model_cnn.yaml`
- `configs/model_effnet_frozen.yaml`
- `configs/model_effnet_finetune.yaml`
- `configs/model_lstm16.yaml`
- `configs/model_lstm32.yaml`

## Notes

- The original notebook and script are kept as source material, but the new CLI workflow should be the primary path for future work.
- `prepare.py` validates the local Kaggle image dataset. It is not a tokenizer/sharding script like the one used in text-model AutoResearch examples.
- `program.md` defines the custom AutoResearch contract for this repository.
- Full runs require PyTorch, torchvision, scikit-learn, matplotlib, seaborn, pandas, and Pillow.
