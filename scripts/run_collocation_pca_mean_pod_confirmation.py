"""Run the frozen two-arm confirmation, or rebuild its report without training."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()


if __name__ == '__main__':
    import json
    from contextlib import redirect_stdout, redirect_stderr
    from pinn_pca.collocation_pair_confirmation import run

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='pca_collocation/configs/burgers_pca_mean_pod_confirmation_100k.json')
    parser.add_argument('--report-only', action='store_true', help='Require completed runs; never train.')
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding='utf-8'))
    out = ROOT / 'pca_collocation/results/burgers/pair_confirmation' / config['run_id']
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'execution.log').open('a', encoding='utf-8', buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            run(config, ROOT, report_only=args.report_only)
