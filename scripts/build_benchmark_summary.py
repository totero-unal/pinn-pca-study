"""Build the global benchmark tables from saved results, without training or JAX."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pinn_pca.benchmark_summary import build_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply-closures", action="store_true", help="Update status/closure cells in explicitly closed notebooks.")
    args = parser.parse_args()
    report = build_summary(ROOT, apply_closures=args.apply_closures)
    print(f'Built {report["comparison_count"]} paired comparisons; {len(report["archive"])} historical result summaries.')
