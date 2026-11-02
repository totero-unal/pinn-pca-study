"""Run or rebuild the separate hard-constraint QR-DEIM reference."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pinn_pca.qr_deim_reference import run
from pinn_pca.qr_deim_reference_report import report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_qr_deim_reference_pilot.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if not args.report_only:
        run(config, ROOT)
    report(config, ROOT)
