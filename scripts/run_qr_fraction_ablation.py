"""Run the paired Burgers QR fraction ablation."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pinn_pca.collocation_ablation import run_qr_fraction_ablation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_qr_fraction_ablation.json")
    parser.add_argument("--summarize-only", action="store_true", help="Regenerate summaries from saved seed CSVs with the same configuration")
    args = parser.parse_args()
    output = run_qr_fraction_ablation(json.loads((ROOT / args.config).read_text(encoding="utf-8")), ROOT, summarize_only=args.summarize_only)
    print(f"Ablation artifacts: {output}")
