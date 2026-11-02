import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pinn_pca.collocation_intervals import run
run(json.loads((ROOT/"pca_collocation/configs/burgers_interval_history.json").read_text()),ROOT)
