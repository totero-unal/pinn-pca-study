"""Diagnose oracle physics and the effect of training-only PCA coefficient scaling."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wave_diagnostic_round.json")
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.report_only:
        from pinn_pca.wave_diagnostic_report import create_report
        output = ROOT / "results/wave/diagnostics" / config["run_id"]
        create_report(ROOT, output)
    elif args.oracle_only:
        from pinn_pca.wave_diagnostics import run_oracle_stage
        result = run_oracle_stage(config, ROOT)
        print(json.dumps(result["gate"], ensure_ascii=False, indent=2))
    else:
        from pinn_pca.wave_diagnostics import run_diagnostic_round
        run_diagnostic_round(config, ROOT)
