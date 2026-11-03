"""Train one full benchmark problem, or report stored results without training."""
import argparse
import json
import os
import sys
import traceback
import platform
import importlib.metadata
from contextlib import redirect_stdout,redirect_stderr
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLBACKEND','Agg')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class Tee:
    def __init__(self,stream,log):self.stream,self.log=stream,log
    def write(self,value):self.stream.write(value);self.log.write(value);self.log.flush();return len(value)
    def flush(self):self.stream.flush();self.log.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--problem',choices=('burgers','wave','allen_cahn'))
    parser.add_argument('--aggregate',action='store_true',help='Report all three completed studies; never train')
    parser.add_argument('--report-only',action='store_true',help='Regenerate one problem report; never train')
    parser.add_argument('--config',default='pca_collocation/configs/benchmark_full.json')
    args=parser.parse_args()
    if args.aggregate and args.problem:parser.error('Choose --aggregate or --problem')
    if not args.aggregate and not args.problem:parser.error('--problem is required unless --aggregate is used')
    from pinn_pca.benchmark_pilot import prepare,run_experiments
    from pinn_pca.benchmark_full import report
    c=json.loads((ROOT/args.config).read_text(encoding='utf-8'))
    if c.get('stage')!='full_benchmark':parser.error('Requires full_benchmark configuration')
    out=prepare(ROOT,c);target=out/('aggregate' if args.aggregate else args.problem);target.mkdir(exist_ok=True)
    with (target/'execution.log').open('a',encoding='utf-8',buffering=1) as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            try:
                if not args.aggregate and not args.report_only:
                    from exchange_benchmark_results import code_digest, compatible_environment
                    from pinn_pca.benchmark_pilot import write_json
                    environment=dict(code_sha256=code_digest(),python=sys.version,platform=platform.platform(),
                        packages={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy','scipy','matplotlib')})
                    provenance=target/'environment.json'
                    if provenance.exists():
                        environment=compatible_environment(json.loads(provenance.read_text()),environment)
                    write_json(provenance,environment)
                    run_experiments(ROOT,c,problem_only=args.problem)
                report(ROOT,c,None if args.aggregate else args.problem)
            except BaseException:
                details=traceback.format_exc()
                try:(target/'last_error.txt').write_text(details,encoding='utf-8')
                except OSError:pass
                print(details,file=sys.stderr,flush=True)
                raise


if __name__=='__main__':main()
