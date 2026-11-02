"""Compare POD, centered PCA and PCA+mean under one fixed selector protocol."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn_pca.collocation_basis_ablation import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_basis_ablation.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text(encoding="utf-8")), ROOT, args.report_only)
