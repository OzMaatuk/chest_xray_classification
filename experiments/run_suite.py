from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run a list of experiment configs sequentially.")
    parser.add_argument("configs", nargs="+", help="Config files to run.")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    src_path = str(root / "src")
    env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else f"{src_path}:{env['PYTHONPATH']}"

    for config in args.configs:
        print(f"Running {config}")
        subprocess.run(
            [sys.executable, str(root / "experiments" / "run_experiment.py"), "--config", config],
            check=True,
            cwd=root,
            env=env,
        )


if __name__ == "__main__":
    main()
