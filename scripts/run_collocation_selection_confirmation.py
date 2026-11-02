"""Run the preconfigured ten-seed Burgers collocation-selection confirmation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_pca.collocation import run_collocation_pilot  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="pca_collocation/configs/burgers_collocation_selection_confirmation.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    output = run_collocation_pilot(config, ROOT)
    print(f"Confirmation artifacts: {output}")


if __name__ == "__main__":
    main()
