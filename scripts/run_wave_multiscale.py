"""Run the paired two-/three-frequency standing-wave PINN/PCA pilot."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/wave_multiscale_pilot.json")
    parser.add_argument("--report-only", action="store_true", help="Refresh figures/notebook from saved results without importing JAX.")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.report_only:
        from pinn_pca.wave_report import create_wave_report
        output = ROOT / "results/wave/base_vs_pca" / config["run_id"]
        result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        def read_csv(name):
            with (output / name).open(encoding="utf-8", newline="") as stream:
                return list(csv.DictReader(stream))
        create_wave_report(ROOT, output, result, read_csv("validation_trajectories.csv"), read_csv("mode_metrics.csv"))
        print(f"Updated report: {output}", flush=True)
    else:
        from pinn_pca.wave import run_wave_benchmark
        result = run_wave_benchmark(config, ROOT)
        print(f"Saved results: {result['output_dir']}", flush=True)
