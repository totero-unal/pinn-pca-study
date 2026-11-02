"""Run or report a configured budget extension with the distributed-IC protocol."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pca_collocation/configs/burgers_ic_long_budget.json")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.report_only:
        from pinn_pca.collocation_ic_budget_report import build_budget_report
        build_budget_report(config, ROOT)
        from pinn_pca.collocation_ic_budget import experiment_paths
        _, output = experiment_paths(config, ROOT)
    else:
        from pinn_pca.collocation_ic_budget import run_long_budget
        output = run_long_budget(config, ROOT)
    print(f"Extended IC budget artifacts: {output}")
