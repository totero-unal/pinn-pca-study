import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pinn_pca.collocation_partial import run
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--report-only",action="store_true")
    args=parser.parse_args()
    run(json.loads((ROOT/"pca_collocation/configs/burgers_partial_update.json").read_text()),ROOT,args.report_only)
