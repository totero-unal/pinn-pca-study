"""Run or report the paired Burgers IC-placement control."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_ic_coverage_control.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    from pinn_pca.collocation_ic_control import build_ic_control_report, run_ic_control
    if args.report_only:
        build_ic_control_report(config, ROOT)
        output = ROOT / "pca_collocation/results/burgers/ic_coverage_control" / config["run_id"]
    else:
        output = run_ic_control(config, ROOT)
    print(f"IC coverage control artifacts: {output}")
