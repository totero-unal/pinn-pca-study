"""Analyze saved validation trajectories; no training or JAX initialization."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn_pca.time_to_target import run_analysis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/high_dim_time_to_target.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    report = run_analysis(config, ROOT)
    print(json.dumps({"target_rmse": report["target_rmse"],
                      "models": report["models"],
                      "paired": report["paired"]}, ensure_ascii=False, indent=2))
