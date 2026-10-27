"""Compare early saved validation checkpoints, without training or JAX."""
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pinn_pca.early_training import run_analysis


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/early_training_analysis.json')
    args=parser.parse_args()
    report=run_analysis(json.loads((ROOT/args.config).read_text(encoding='utf-8')),ROOT)
    print(json.dumps(report['primary_case_counts'],ensure_ascii=False,indent=2))
