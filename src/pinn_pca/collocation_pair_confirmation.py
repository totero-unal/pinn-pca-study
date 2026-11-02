"""Fresh-seed PCA+mean versus matched-rank POD confirmation at a fixed budget."""
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .collocation import _run_collocation_seed, _save_csv, _qmc_points
from .collocation_basis_ablation import read_rows
from .collocation_hard_conditions import HardRecorder
from .collocation_ic_confirmation import paired_stats
from .collocation_pde_bases import matched_selection
from .notebook_images import embed_notebook_images

LABELS = {'pca': 'PCA + media', 'pod_qr': 'POD'}


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


class ProgressRecorder(HardRecorder):
    def checkpoint(self, method, step, params, active, selection_step):
        super().checkpoint(method, step, params, active, selection_step)
        if step == 100 or (step - 100) % 10000 == 0:
            print(f'PROGRESS {method} seed={self.c["seed"]} continuation={step-100}', flush=True)


def audit_run(c, folder):
    end = c['warmup_steps'] + c['train_steps']
    metrics = read_rows(folder / 'metrics.csv')
    assert len(metrics) == 1 and int(metrics[0]['train_steps']) == c['train_steps']
    assert all(np.isfinite(float(metrics[0][k])) for k in ('test_rmse', 'independent_residual_rms'))
    budgets = read_rows(folder / 'budget_metrics.csv')
    assert [int(r['step']) for r in budgets] == c['test_checkpoints']
    constraints = read_rows(folder / 'constraints.csv')
    assert [int(r['step']) for r in constraints] == c['test_checkpoints']
    assert all(float(r['ic_max']) < 1e-6 and float(r['bc_max']) == 0 for r in constraints)
    trace = read_rows(folder / 'validation_trajectories.csv')
    assert int(trace[-1]['step']) == end
    with np.load(folder / 'active_history.npz') as history:
        assert len(history.files) == c['train_steps'] // c['update_every']
        for key in history.files:
            points = history[key]
            assert points.shape == (c['batch_size'], 2) and np.isfinite(points).all()
            assert len(np.unique(points, axis=0)) == c['batch_size']
    return dict(passed=True, active_sets=c['train_steps']//c['update_every'], terminal_step=end)


def schedule_from(folder):
    return {str(int(r['update_step']) - 100): int(r['retained_modes'])
            for r in read_rows(folder / 'selection_diagnostics.csv')}


def audit_pair(c, out, seed):
    a, b = [out / method / f'seed_{seed}' for method in LABELS]
    schedule = schedule_from(a)
    assert sorted(map(int, schedule)) == list(range(100, 100+c['train_steps'], 100))
    d = read_rows(b / 'selection_diagnostics.csv')
    assert {str(int(r['update_step'])-100): int(r['selected_qr_points']) for r in d} == schedule
    ta, tb = [read_rows(p / 'validation_trajectories.csv') for p in (a,b)]
    np.testing.assert_array_equal(float(ta[0]['validation_rmse']), float(tb[0]['validation_rmse']))
    with np.load(a/'history_audit.npz') as ha, np.load(b/'history_audit.npz') as hb:
        np.testing.assert_array_equal(ha['100_history'], hb['100_history'])
        np.testing.assert_array_equal(ha['100_pool'], hb['100_pool'])
        with np.load(b/'active_history.npz') as active:
            for step in (100,200,1100,5100,10000):
                ids, _ = matched_selection(hb[f'{step}_history'], 'pod_qr', schedule[str(step)])
                np.testing.assert_array_equal(active[f'pod_qr_{step}'][:len(ids)], hb[f'{step}_pool'][ids])
    for method, folder in zip(LABELS, (a,b)):
        with np.load(folder/'active_history.npz') as active:
            for step, k in schedule.items():
                background = _qmc_points(c['batch_size']-k, seed+int(step)+104729,
                                         (-1.,1.), (0.,float(np.float32(.99))))
                np.testing.assert_array_equal(active[f'{method}_{step}'][k:], background)


def report(c, root, out):
    rows, traces, individual = [], [], []
    for seed in c['seeds']:
        for method in LABELS:
            folder = out / method / f'seed_{seed}'
            assert (folder/'completed.json').exists()
            audit_run(c, folder)
            rows.extend(read_rows(folder/'metrics.csv'))
            traces.extend(read_rows(folder/'validation_trajectories.csv'))
        audit_pair(c,out,seed)
    summary = []
    for method in LABELS:
        group = [r for r in rows if r['method'] == method]
        entry = dict(method=method)
        for key in ('test_rmse','independent_residual_rms'):
            v = np.array([float(r[key]) for r in group])
            entry.update({key:float(v.mean()), key+'_std':float(v.std(ddof=1)), key+'_median':float(np.median(v))})
        summary.append(entry)
    for seed in c['seeds']:
        values = {m:float(next(r for r in rows if int(r['seed'])==seed and r['method']==m)['test_rmse']) for m in LABELS}
        individual.append(dict(seed=seed, pca_mean=values['pca'], pod=values['pod_qr'], difference=values['pca']-values['pod_qr']))
    primary = dict(comparison=c['primary_comparison'], criterion=c['primary_criterion'],
                   **paired_stats([r['difference'] for r in individual]))
    primary['criterion_met'] = primary['bootstrap95'][1] < 0
    _save_csv(out/'metrics.csv',rows)
    _save_csv(out/'traces.csv',traces)
    _save_csv(out/'summary.csv',summary)
    _save_csv(out/'individual_pca_mean_vs_pod.csv',individual)
    save_json(out/'primary_confirmation.json',primary)
    save_json(out/'audit_checks.json',dict(passed=True, completed_runs=2*len(c['seeds']),
        active_sets=2*len(c['seeds'])*c['train_steps']//100, matched_ranks=True,
        matched_background=True, common_warmup=True, qr_reconstructions=5*len(c['seeds'])))
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    for method,label in LABELS.items():
        group = [r for r in traces if r['method']==method]
        steps = sorted({int(r['step']) for r in group})
        axes[0].plot([s-100 for s in steps], [np.mean([float(r['validation_rmse']) for r in group if int(r['step'])==s]) for s in steps],label=label)
    axes[0].set(xlabel='Pasos de continuación',ylabel='RMSE validación medio')
    axes[0].legend()
    axes[1].bar([str(r['seed']) for r in individual], [r['difference'] for r in individual])
    axes[1].axhline(0,color='black',linewidth=.8)
    axes[1].tick_params(axis='x',rotation=45)
    axes[1].set(xlabel='Semilla',ylabel='RMSE PCA + media − POD')
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    table='| Método | RMSE medio ± DE | Mediana RMSE | Residuo RMS medio |\n|---|---:|---:|---:|\n'
    for r in summary:
        table+=f"| {LABELS[r['method']]} | {r['test_rmse']:.5f} ± {r['test_rmse_std']:.5f} | {r['test_rmse_median']:.5f} | {r['independent_residual_rms']:.5f} |\n"
    text=('# Confirmación de PCA + media frente a POD a 100000 pasos\n\n'
          f"Diez semillas nuevas {c['seeds']}, separadas de las exploraciones previas. Cien pasos uniformes y 100000 de continuación, sin reiniciar Adam. "
          'Pérdida solo PDE y condiciones IC/BC duras. Se conservan red 3×32, Adam 0.001, 4096 candidatos fijos, veinte snapshots cada cinco pasos, QR directo y reconstrucción de 128 puntos cada cien pasos. '
          'PCA + media determina el rango al 95 % de energía cuadrática; la media cuenta como dirección. POD usa el mismo rango y complemento Sobol. Es una comparación condicional, no de rangos adaptativos autónomos.\n\n'
          'La comparación principal fijada antes de ejecutar es la diferencia pareada de RMSE terminal PCA + media − POD. El criterio de ventaja es que el extremo superior del bootstrap pareado al 95 % sea menor que cero (20000 remuestreos, semilla 6071). '
          'Medianas, validación y residuo son descriptivos; no se usan para elegir el paso terminal ni descartar semillas. No se mezclan cohortes.\n\n'+table+
          f"\nDiferencia media {primary['mean_difference']:+.5f}; victorias {primary['wins']}/10; bootstrap 95 % {primary['bootstrap95']}. Criterio satisfecho: {primary['criterion_met']}. No satisfacerlo no demuestra equivalencia.\n\n"+
          f"![Resultados](../../results/burgers/pair_confirmation/{c['run_id']}/comparison.png)\n\n"
          'Los CSV conservan valores individuales y trazas; primary_confirmation.json contiene el contraste principal y audit_checks.json las verificaciones. Los tiempos son observados e incluyen instrumentación; no prueban aceleración.\n\n'
          'Ejecutar: `python scripts/run_collocation_pca_mean_pod_confirmation.py`. Regenerar este reporte sin entrenar: añadir `--report-only`.')
    notebook=root/'pca_collocation/notebooks/burgers'/c['notebook_name']
    save_json(notebook,dict(cells=[dict(id='pair-confirmation',cell_type='markdown',metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5))
    embed_notebook_images(notebook,root)
    print(json.dumps(primary,indent=2),flush=True)


def run(c, root, report_only=False):
    root=Path(root)
    assert len(c['seeds'])==len(set(c['seeds']))==10
    assert c['warmup_steps']==c['update_every']==100 and c['train_steps']==100000
    assert c['test_checkpoints']==[100,100100] and c['adaptive_basis_threshold']==.95
    # Check the proposed cohort against every other stored experiment configuration.
    for path in (root/'pca_collocation/configs').glob('*.json'):
        other=json.loads(path.read_text(encoding='utf-8'))
        if other.get('run_id')!=c['run_id']:
            used=set(other.get('seeds',[])) | set(other.get('confirmation_seeds',[])) | set(other.get('development_seeds',[]))
            if used & set(c['seeds']):
                raise ValueError(f'Confirmation seeds overlap with {path}')
    out=root/'pca_collocation/results/burgers/pair_confirmation'/c['run_id']
    out.mkdir(parents=True,exist_ok=True)
    frozen=out/'config.json'
    if frozen.exists() and json.loads(frozen.read_text(encoding='utf-8'))!=c:
        raise ValueError('Configuration changed: use a new run_id.')
    save_json(frozen,c)
    def status(state, **extra):
        completed=sum((out/m/f'seed_{s}'/'completed.json').exists() for m in LABELS for s in c['seeds'])
        save_json(out/'status.json',dict(state=state,updated_utc=datetime.now(timezone.utc).isoformat(),completed_runs=completed,total_runs=20,**extra))
    try:
        for seed in c['seeds']:
            for method in LABELS:
                folder=out/method/f'seed_{seed}'
                if (folder/'completed.json').exists():
                    audit_run(c,folder)
                    print(f'SKIP completed {method} seed={seed}',flush=True)
                    continue
                if report_only:
                    raise RuntimeError(f'Incomplete run: {method} seed={seed}; --report-only never trains.')
                status('running',method=method,seed=seed)
                if method=='pca':
                    spec=dict(base='pca_mean_qr',label=LABELS[method],adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                else:
                    spec=dict(base='pod_qr',label=LABELS[method],rank_schedule=schedule_from(out/'pca'/f'seed_{seed}'))
                print(f'RUN {method} seed={seed}',flush=True)
                _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,pde_only=True,
                    method_specs={method:spec},basis_ablation=method=='pod_qr',
                    common_selection_seed=True,skip_seed_plots=True),root,folder,observer=ProgressRecorder(c,folder))
                save_json(folder/'completed.json',audit_run(c,folder))
                status('running',last_completed_method=method,last_completed_seed=seed)
                print(f'DONE {method} seed={seed}',flush=True)
        status('reporting')
        report(c,root,out)
        status('complete')
    except BaseException as exc:
        status('interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc))
        (out/'last_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
