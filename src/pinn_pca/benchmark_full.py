"""Audit and report the three-problem study. Reporting never invokes training."""
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from scipy.stats import rankdata

from .benchmark_pilot import prepare, write_json
from .collocation import _save_csv
from .collocation_basis_ablation import read_rows
from .collocation_rank_study import adaptive_basis
from .collocation_ic_confirmation import paired_stats
from .notebook_images import embed_notebook_images

NAMES={'burgers':('35','Burgers'),'wave':('36','Onda'),'allen_cahn':('37','Allen–Cahn')}


def audit_problem(c,out,problem):
    rows=[];traces=[];ranks=[];sets=0;reconstructions=0
    audit=json.loads((out/'references'/f'{problem}_audit.json').read_text())
    if not audit['passed']:raise ValueError('Reference audit failed')
    for seed in c['seeds']:
        initial=[]
        for method in c['methods']:
            folder=out/problem/method/f'seed_{seed}'
            marker=folder/'completed.json'
            if not marker.exists():raise RuntimeError(f'Incomplete run: {problem}/{method}/seed_{seed}; reporting never trains')
            info=json.loads(marker.read_text());assert info['passed'] and info['steps']==c['train_steps']
            metrics=read_rows(folder/'metrics.csv')
            assert len(metrics)==2 and {r['checkpoint'] for r in metrics}=={'terminal','best_validation'}
            for r in metrics:
                assert r['problem']==problem and r['method']==method and int(r['seed'])==seed
                assert all(np.isfinite(float(r[k])) and float(r[k])>=0 for k in ('relative_l2','rmse','independent_residual_rms','pipeline_seconds'))
            terminal=next(r for r in metrics if r['checkpoint']=='terminal');assert int(terminal['step'])==c['train_steps']
            trace=read_rows(folder/'trace.csv');assert int(trace[-1]['step'])==c['train_steps']
            assert len(trace)==c['train_steps']//c['validation_every']+1
            best=next(r for r in metrics if r['checkpoint']=='best_validation')
            assert int(best['step'])==int(min(trace,key=lambda r:float(r['validation_residual_mse']))['step'])
            constraints=json.loads((folder/'constraints.json').read_text())
            assert all(abs(v)<1e-6 for state in constraints.values() for v in state.values())
            for step in c['state_checkpoints']:
                if not (folder/f'state_{step}.npz').exists():raise RuntimeError(f'Missing diagnostic checkpoint {folder} at {step}')
            with np.load(folder/'active_points.npz') as active:
                expected=[0] if method in ('random','hammersley') else list(range(0,c['train_steps'],c['update_every']))
                assert sorted(int(key.split('_')[-1]) for key in active.files)==expected
                for key in active.files:
                    points=active[key]
                    assert points.shape==(c['active_count'],2) and np.isfinite(points).all()
                    assert len(np.unique(points,axis=0))==c['active_count']
                    assert np.all(np.abs(points[:,0])<=1) and np.all((points[:,1]>=0)&(points[:,1]<=1))
                    sets+=1
                if method!='hammersley':initial.append(active['step_0'].copy())
                if method=='pca_mean':
                    selection=read_rows(folder/'selection.csv')
                    assert [int(r['step']) for r in selection]==list(range(c['update_every'],c['train_steps'],c['update_every']))
                    ranks.extend(selection)
                    with np.load(folder/'residual_histories.npz') as histories:
                        assert sorted(int(k.split('_')[-1]) for k in histories.files if k!='candidates')==list(range(c['update_every'],c['train_steps']+1,c['update_every']))
                        for step in (1000,25000,50000,75000,99000):
                            history=histories[f'step_{step}'];assert history.shape==(c['pca_candidates'],20)
                            basis,_=adaptive_basis(history,'pca_mean_qr',c['pca_threshold'])
                            _,_,ids=qr(basis.T,mode='economic',pivoting=True);k=basis.shape[1]
                            assert k==int(next(r['rank'] for r in selection if int(r['step'])==step))
                            np.testing.assert_array_equal(active[f'step_{step}'][:k],histories['candidates'][ids[:k]])
                            reconstructions+=1
            for filename in ('terminal_state.npz','best_validation_weights.npz'):
                if not (folder/filename).exists():raise RuntimeError(f'Missing model: {folder/filename}')
            rows.extend(metrics);traces.extend(trace)
        for points in initial[1:]:np.testing.assert_array_equal(points,initial[0])
    return rows,traces,ranks,dict(passed=True,completed_runs=len(rows)//2,active_sets=sets,
        qr_reconstructions=reconstructions,common_initial_points=True,best_validation_audited=True,
        ic_bc_audited=True,late_histories_available=True)


def summaries(c,rows):
    summary=[];paired=[];individual=[]
    for problem in sorted({r['problem'] for r in rows}):
        for checkpoint in ('best_validation','terminal'):
            group=[r for r in rows if r['problem']==problem and r['checkpoint']==checkpoint]
            for method in c['methods']:
                subset=[r for r in group if r['method']==method];assert len(subset)==len(c['seeds'])
                item=dict(problem=problem,checkpoint=checkpoint,method=method,n=len(subset))
                for metric in ('relative_l2','rmse','independent_residual_rms','pipeline_seconds'):
                    a=np.array([float(r[metric]) for r in subset])
                    item.update({metric+'_mean':float(a.mean()),metric+'_std':float(a.std(ddof=1)),metric+'_median':float(np.median(a))})
                summary.append(item)
            for baseline in c['methods']:
                if baseline=='pca_mean':continue
                differences=[]
                for seed in c['seeds']:
                    get=lambda method:float(next(r for r in group if r['method']==method and int(r['seed'])==seed)['relative_l2'])
                    pca,reference=get('pca_mean'),get(baseline);differences.append(pca-reference)
                    individual.append(dict(problem=problem,checkpoint=checkpoint,seed=seed,reference=baseline,
                        pca_relative_l2=pca,reference_relative_l2=reference,difference=pca-reference))
                paired.append(dict(problem=problem,checkpoint=checkpoint,reference=baseline,metric='relative_l2',**paired_stats(differences)))
    return summary,paired,individual


def report(root,c,problem=None):
    root=Path(root);out=prepare(root,c)
    problems=[problem] if problem else c['problems']
    rows=[];traces=[];ranks=[];audits={}
    for p in problems:
        print(f'AUDIT {p}',flush=True)
        r,t,k,a=audit_problem(c,out,p);rows.extend(r);traces.extend(t);ranks.extend(k);audits[p]=a
    target=out/problem if problem else out/'aggregate';target.mkdir(parents=True,exist_ok=True)
    summary,paired,individual=summaries(c,rows)
    _save_csv(target/'metrics.csv',rows);_save_csv(target/'summary.csv',summary)
    _save_csv(target/'traces.csv',traces);_save_csv(target/'pca_rank_diagnostics.csv',ranks)
    _save_csv(target/'paired_individual.csv',individual)
    write_json(target/'paired_comparisons.json',paired);write_json(target/'audit.json',audits)
    fig,axes=plt.subplots(1,len(problems),figsize=(6*len(problems),5),squeeze=False)
    for ax,p in zip(axes.flat,problems):
        values=[[float(r['relative_l2']) for r in rows if r['problem']==p and r['method']==m and r['checkpoint']=='best_validation'] for m in c['methods']]
        ax.boxplot(values,tick_labels=c['methods'],showmeans=True)
        for i,v in enumerate(values):ax.scatter(np.full(len(v),i+1)+np.linspace(-.09,.09,len(v)),v,s=12,alpha=.6)
        ax.set(title=NAMES[p][1],ylabel='Error relativo L2 — mejor validación',yscale='log');ax.tick_params(axis='x',rotation=35)
    fig.tight_layout();fig.savefig(target/'error_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,len(problems),figsize=(6*len(problems),4),squeeze=False)
    for ax,p in zip(axes.flat,problems):
        for m in c['methods']:
            group=[r for r in traces if r['problem']==p and r['method']==m]
            steps=sorted({int(r['step']) for r in group})
            ax.plot(steps,[np.mean([float(r['validation_residual_mse']) for r in group if int(r['step'])==s]) for s in steps],label=m)
        ax.set(title=NAMES[p][1],xlabel='Paso total',ylabel='Residuo MSE validación medio',yscale='log');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(target/'validation_curves.png',dpi=150);plt.close(fig)
    table='| Problema | Método | L2 relativo medio ± DE | Mediana | Residuo RMS medio |\n|---|---|---:|---:|---:|\n'
    for r in summary:
        if r['checkpoint']=='best_validation':table+=f"| {r['problem']} | {r['method']} | {r['relative_l2_mean']:.6f} ± {r['relative_l2_std']:.6f} | {r['relative_l2_median']:.6f} | {r['independent_residual_rms_mean']:.6f} |\n"
    contrasts='| Problema | PCA − referencia | Diferencia media | Victorias / 10 | Bootstrap 95 % |\n|---|---|---:|---:|---|\n'
    for r in paired:
        if r['checkpoint']=='best_validation':contrasts+=f"| {r['problem']} | {r['reference']} | {r['mean_difference']:+.6f} | {r['wins']} | {r['bootstrap95']} |\n"
    text=('# '+('Comparación conjunta de los tres problemas' if not problem else NAMES[problem][1]+' — estudio de cinco selectores')+
        '\n\nDiez semillas nuevas emparejadas, red 5×64 tanh, 2000 puntos y 100000 pasos totales. Adam 0.001 con coseno, pérdida solo PDE y condiciones duras. No se añaden pasos del piloto. Se conserva PCA + media al 95 %, veinte snapshots cada cincuenta pasos, QR directo y complemento uniforme. No hay ampliación log-det.\n\n'+table+'\n\n'+contrasts+
        '\nLa comparación principal usa el modelo de menor residuo MSE de validación, entre pasos 0 y 100000 observados cada 250 pasos; prueba solo se evalúa al terminar. El terminal está en los mismos CSV como control secundario. Los intervalos bootstrap pareados usan 20000 remuestreos con semilla 6071, no corrigen las doce comparaciones y no justifican un ganador global.\n\n')
    relative=target.relative_to(root/'pca_collocation').as_posix()
    for name in ('error_comparison','validation_curves'):text+=f'![{name}](../../{relative}/{name}.png)\n\n'
    if not problem:
        ranking=[]
        for p in problems:
            entries=[next(r for r in summary if r['problem']==p and r['method']==m and r['checkpoint']=='best_validation') for m in c['methods']]
            scores=rankdata([r['relative_l2_mean'] for r in entries])
            ranking.extend(dict(problem=p,method=m,rank=float(score)) for m,score in zip(c['methods'],scores))
        _save_csv(target/'ranks_by_problem.csv',ranking)
        text+='No se promedian RMSE ni residuos entre PDE. ranks_by_problem.csv conserva posiciones por problema, de carácter descriptivo, calculadas por L2 relativo medio. No se mezclan estas semillas con las del piloto.\n\n'
    text+='Configuración congelada: benchmark_full.json. Hammersley usa un desplazamiento módulo uno por semilla. Allen–Cahn conserva el signo +5(u³−u) escrito en el artículo y la referencia propia del piloto; no equivale a la formulación de reacción opuesta. Las referencias numéricas pasan una comparación de dos mallas al 2 %, que no es una cota rigurosa de error. Los tiempos incluyen compilación, observación y validación y excluyen preparar referencias y evaluar prueba al final; no prueban aceleración del selector.\n\n'
    text+='Se conservan cien historias completas por corrida PCA, incluidos instantes tardíos, todos los lotes actualizados, modelos en pasos 0/25000/50000/75000/100000 y mejores pesos de validación. Los estados con Adam son diagnósticos: el runner repite desde el principio una corrida incompleta. audit.json comprueba lotes, condiciones, elección de checkpoint, puntos iniciales comunes y reconstrucciones QR tempranas y tardías.\n'
    stem=f'{NAMES[problem][0]}_{problem}_benchmark' if problem else '38_benchmark_aggregate'
    notebook=root/'pca_collocation/notebooks/benchmark'/f'{stem}_results.ipynb'
    notebook.parent.mkdir(parents=True,exist_ok=True)
    notebook.write_text(json.dumps(dict(cells=[dict(id='benchmark-results',cell_type='markdown',metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding='utf-8')
    embed_notebook_images(notebook,root)
    write_json(target/'report_complete.json',dict(passed=True,problems=problems,completed_runs=len(rows)//2))
    print(f'Report saved: {notebook}',flush=True)
    return target
