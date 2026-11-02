"""Run the one-seed Burgers collocation-selection pilot."""

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
    parser.add_argument("--config", default="pca_collocation/configs/burgers_collocation_selection_pilot.json")
    args = parser.parse_args()
    root = ROOT
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    output = run_collocation_pilot(config, root)
    print(f"Pilot artifacts: {output}")


if __name__ == "__main__":
    main()
