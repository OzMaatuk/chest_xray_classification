# AutoResearch Program

The canonical machine-readable instructions live at the repository root in `program.md`.

This document is a short pointer for human readers:

- `program.md` defines the custom AutoResearch contract
- `prepare.py` validates the local Kaggle dataset
- `train.py` is a thin compatibility wrapper over `experiments/run_experiment.py`
- `configs/*.yaml` and `src/chest_xray_project/` are the main experiment surfaces
