"""Run or report the unchanged-objective Burgers selection diagnosis."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_collocation_diagnostics.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.report_only:
        from pinn_pca.collocation_diagnostic_report import build_diagnostic_report
        output = ROOT / "pca_collocation/results/burgers/collocation_diagnostics" / config["run_id"]
        build_diagnostic_report(config, ROOT, output)
    else:
        from pinn_pca.collocation_diagnostics import run_diagnostics
        output = run_diagnostics(config, ROOT)
    print(f"Diagnostic artifacts: {output}")
