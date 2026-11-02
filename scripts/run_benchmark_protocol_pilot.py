"""Execute only notebook 34's pilot, preserving a console log in the repository."""
import os
import sys
import json
import argparse
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,value):
        self.stream.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.stream.flush();self.log.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='pca_collocation/configs/benchmark_protocol_pilot.json')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    from pinn_pca.benchmark_pilot import prepare,run_pilot,report
    c=json.loads((ROOT/args.config).read_text(encoding='utf-8'));out=prepare(ROOT,c)
    with (out/'execution.log').open('a',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            if args.report_only:report(ROOT,c)
            else:run_pilot(ROOT,c)
            print(f'Results saved: {out}',flush=True)


if __name__=='__main__':main()
