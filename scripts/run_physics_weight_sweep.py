"""Run the PDE-loss weighting sweep for the PCA k=8 Burgers model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_pca.physics_weight import run_physics_weight_sweep  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/burgers_physics_weight_sweep.json")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    result = run_physics_weight_sweep(config, ROOT)
    print(f"\nSaved results to: {result['output_dir']}")


if __name__ == "__main__":
    main()
