"""Extend frozen physical basis comparison to 50k without Adam resets."""
import json
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pinn_pca.collocation import _run_collocation_seed,_save_csv
from pinn_pca.collocation_hard_conditions import HardRecorder
from pinn_pca.collocation_pde_bases import run as run_bases
from pinn_pca.collocation_basis_ablation import read_rows
from pinn_pca.collocation_ic_confirmation import paired_stats
from pinn_pca.notebook_images import embed_notebook_images

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',default='pca_collocation/configs/burgers_long_confirmation.json')
args=parser.parse_args()
c=json.loads((ROOT/args.config).read_text())
out=ROOT/'pca_collocation/results/burgers/pde_bases'/c['run_id']
old=ROOT/c.get('prefix_source','pca_collocation/results/burgers/pde_bases/matched_pde_bases_confirmation_10fresh_10k_v1')
end=c['warmup_steps']+c['train_steps']
checkpoints=c['test_checkpoints'][1:]

class PrefixRecorder(HardRecorder):
    def __init__(self,config,folder,oldfolder,oldmethod):
        super().__init__(config,folder);self.oldfolder=oldfolder;self.oldmethod=oldmethod
    def checkpoint(self,method,step,params,active,selection_step):
        super().checkpoint(method,step,params,active,selection_step)
        before_rows=read_rows(self.oldfolder/'budget_metrics.csv')
        before=next((r for r in before_rows if int(r['step'])==step),None)
        if before is not None:
            after=self.metrics[-1] if step in self.config['test_checkpoints'] else None
            if after is not None:
                for key in ('test_rmse','independent_residual_rms','initial_condition_rmse','boundary_rms'):
                    np.testing.assert_allclose(float(after[key]),float(before[key]),rtol=0,atol=1e-6)
                print(f'PREFIX VERIFIED {method} seed={self.c["seed"]} step={step}',flush=True)
        if step in self.config['test_checkpoints']:
            print(f'BUDGET RECORDED {method} seed={self.c["seed"]} continuation={step-100}',flush=True)
    def end_seed(self):
        super().end_seed()
        prior=np.load(self.oldfolder/'active_history.npz')
        for key in prior.files:
            step=int(key.rsplit('_',1)[1]);current=next(v for k,v in self.active.items() if int(k.rsplit('_',1)[1])==step)
            np.testing.assert_array_equal(current,prior[key])
        (self.output/'prefix_verified.json').write_text(json.dumps(dict(passed=True,through_step=self.config.get("prefix_step",10100),active_sets=len(prior.files))))

# Both phases regenerate the deterministic prefix; all Adam moments persist to 50100.
for phase,methods in (('references',('pca','sobol')),('matched',('pod_qr','pca_qr'))):
    for seed in c['seeds']:
        reference=ROOT/c['reference_root']/'pca'/f'seed_{seed}'
        for method in methods:
            folder=(ROOT/c['reference_root']/method/f'seed_{seed}' if phase=='references' else out/method/f'seed_{seed}')
            if (folder/'prefix_verified.json').exists():continue
            oldfolder=(old/'reference_runs'/method/f'seed_{seed}' if phase=='references' else old/method/f'seed_{seed}')
            if method=='sobol':spec=dict(base='sobol',label=method)
            elif method=='pca':spec=dict(base='pca_mean_qr',label=method,adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
            else:
                schedule={str(int(r['update_step'])-100):int(r['retained_modes']) for r in read_rows(reference/'selection_diagnostics.csv')}
                spec=dict(base=method,label=method,rank_schedule=schedule)
            print(f'RUN {c["train_steps"]} steps {method} seed={seed}',flush=True)
            _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,pde_only=True,method_specs={method:spec},basis_ablation=phase=='matched',common_selection_seed=True,skip_seed_plots=True),ROOT,folder,
                observer=PrefixRecorder(c,folder,oldfolder,method))
            print(f'DONE {c["train_steps"]} steps {method} seed={seed}',flush=True)
run_bases(c,ROOT)
budgets=[]
for method in ('pca','sobol','pod_qr','pca_qr'):
    for seed in c['seeds']:
        folder=ROOT/c['reference_root']/method/f'seed_{seed}' if method in ('pca','sobol') else out/method/f'seed_{seed}'
        budgets.extend(dict(r,method=method) for r in read_rows(folder/'budget_metrics.csv'))
_save_csv(out/'budget_metrics.csv',budgets)
summary=[]
for step in checkpoints:
    for method in ('pca','sobol','pod_qr','pca_qr'):
        rows=[r for r in budgets if r['method']==method and int(r['step'])==step]
        summary.append(dict(step=step,method=method,rmse=float(np.mean([float(r['test_rmse']) for r in rows])),residual=float(np.mean([float(r['independent_residual_rms']) for r in rows]))))
_save_csv(out/'budget_summary.csv',summary)
p=ROOT/'pca_collocation/notebooks/burgers'/c['notebook_name'];n=json.loads(p.read_text(encoding='utf-8'))
n['cells'].insert(0,dict(id='extended-budget',cell_type='markdown',metadata={},source=f"# Extensión a {c['train_steps']} pasos\n\nSe regenera y audita el prefijo hasta {c.get('prefix_step',10100)-100} pasos de continuación. Adam continúa sin reinicios. Presupuestos prefijados: {[s-100 for s in checkpoints]}. El rango emparejado continúa con PCA + media al 95%.\n\n"+json.dumps(summary,indent=2)))
for cell in n['cells']:
    if cell['cell_type']=='markdown':cell['source']=''.join(cell['source']).replace('diez mil de continuación',str(c['train_steps'])+' de continuación')
p.write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2),flush=True)

import subprocess
subprocess.run([sys.executable,str(ROOT/"scripts/report_collocation_long_confirmation.py"),"--config",args.config],check=True)
