"""Rebuild budget, timing and terminal-stability report after the 50k study."""
import csv
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pinn_pca.collocation_basis_ablation import read_rows
from pinn_pca.collocation import _save_csv
from pinn_pca.collocation_ic_confirmation import paired_stats
from pinn_pca.notebook_images import embed_notebook_images

parser=argparse.ArgumentParser()
parser.add_argument('--config',default='pca_collocation/configs/burgers_long_confirmation.json')
args=parser.parse_args()
c=json.loads((ROOT/args.config).read_text())
end=c['warmup_steps']+c['train_steps']
checkpoints=c['test_checkpoints'][1:]
primary_method=c.get('primary_method','pca_qr')
out=ROOT/'pca_collocation/results/burgers/pde_bases'/c['run_id']
labels=dict(pca='PCA + media',sobol='Sobol',pod_qr='POD',pca_qr='PCA centrada')
budgets=read_rows(out/'budget_metrics.csv');paired=[];summary=[];stability=[];prefixes=[]
for method in labels:
    for seed in c['seeds']:
        folder=ROOT/c['reference_root']/method/f'seed_{seed}' if method in ('pca','sobol') else out/method/f'seed_{seed}'
        prefixes.append(json.loads((folder/'prefix_verified.json').read_text()))
        trace=read_rows(folder/'validation_trajectories.csv')
        last=[r for r in trace if int(r['step'])>=end-500]
        x=np.array([int(r['step']) for r in last]);y=np.array([float(r['validation_rmse']) for r in last])
        stability.append(dict(method=method,seed=seed,window_start=end-500,window_end=end,
            mean_validation=float(y.mean()),std_validation=float(y.std()),slope_per_1000=float(np.polyfit(x-x[0],y,1)[0]*1000)))
for step in checkpoints:
    for method in labels:
        group=[r for r in budgets if r['method']==method and int(r['step'])==step]
        assert len(group)==10
        timing=[]
        for seed in c['seeds']:
            folder=ROOT/c['reference_root']/method/f'seed_{seed}' if method in ('pca','sobol') else out/method/f'seed_{seed}'
            r=next(r for r in read_rows(folder/'validation_trajectories.csv') if int(r['step'])==step)
            timing.append(float(r['pipeline_seconds']))
        summary.append(dict(step=step,method=method,
            rmse=float(np.mean([float(r['test_rmse']) for r in group])),rmse_std=float(np.std([float(r['test_rmse']) for r in group],ddof=1)),rmse_median=float(np.median([float(r['test_rmse']) for r in group])),
            residual=float(np.mean([float(r['independent_residual_rms']) for r in group])),pipeline_seconds_mean=float(np.mean(timing))))
    for metric in ('test_rmse','independent_residual_rms'):
        get=lambda seed,m:float(next(r for r in budgets if r['method']==m and int(r['seed'])==seed and int(r['step'])==step)[metric])
        paired.append(dict(step=step,metric=metric,**paired_stats([get(s,primary_method)-get(s,'pod_qr') for s in c['seeds']])))
_save_csv(out/'individual_budget_metrics.csv',budgets)
_save_csv(out/'budget_summary.csv',summary);_save_csv(out/'terminal_stability.csv',stability)
(out/'budget_paired.json').write_text(json.dumps(paired,indent=2))
fig,axes=plt.subplots(1,2,figsize=(12,4))
for method,label in labels.items():
    group=[r for r in summary if r['method']==method]
    for ax,key in zip(axes,('rmse','residual')):
        ax.plot([r['step']-100 for r in group],[r[key] for r in group],'o-',label=label)
        ax.set(xlabel='Pasos de continuación',ylabel=key)
axes[0].legend();fig.tight_layout();fig.savefig(out/'budget_comparison.png',dpi=150);plt.close(fig)
table='| Continuación | Método | RMSE prueba ± DE | Residuo RMS | Tiempo observado medio (s) |\n|---:|---|---:|---:|---:|\n'
for r in summary:table+=f"| {r['step']-100} | {labels[r['method']]} | {r['rmse']:.5f} ± {r['rmse_std']:.5f} | {r['residual']:.5f} | {r['pipeline_seconds_mean']:.1f} |\n"
text='# Presupuesto ampliado de entrenamiento\n\nSe mantienen las mismas diez semillas y protocolo, sin reiniciar Adam en los presupuestos. Se regenera el prefijo porque no estaban disponibles todos los estados de Adam: cuarenta auditorías comprueban las métricas y los lotes del tramo compartido por corrida. No son nuevas semillas de confirmación. El rango emparejado continúa con la misma regla PCA + media al 95 %. La arquitectura, pérdida solo PDE, condiciones duras, candidatos e historia permanecen iguales.\n\n'+table+'\n\n'
for r in paired:
    if r['metric']=='test_rmse':text+=f"{labels[primary_method]} − POD a {r['step']-100} pasos: media {r['mean_difference']:+.5f}, gana en {r['wins']}/10; bootstrap 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n\n"
text+='![Presupuestos](../../results/burgers/pde_bases/'+c['run_id']+'/budget_comparison.png)\n\nLos tiempos observados incluyen warmup, observación, selección y monitores; el registro ocurre antes del diagnóstico de prueba de ese checkpoint. El orden fijo y las cargas del equipo impiden atribuir aceleraciones. terminal_stability.csv describe los últimos 500 pasos de validación: pendiente y variación no prueban convergencia asintótica. Las diferencias bootstrap con diez semillas son inciertas y los múltiples presupuestos son descriptivos: no se elige presupuesto usando prueba.\n\nReproducir con `python scripts/run_collocation_long_confirmation.py --config {args.config}`; regenerar con `python scripts/report_collocation_long_confirmation.py --config {args.config}`.'
text+='\n\nEl control Sobol conserva la observación instrumentada de candidatos para la auditoría compartida, aunque su selector no la utiliza. Estos tiempos describen esta ejecución, no el costo de una implementación Sobol optimizada.'
# Detailed paired seed trajectories and robust summaries.
individual=[]
for step in checkpoints:
    for seed in c['seeds']:
        get=lambda method:float(next(r for r in budgets if r['method']==method and int(r['seed'])==seed and int(r['step'])==step)['test_rmse'])
        individual.append(dict(step=step,seed=seed,pca_mean=get('pca'),pod=get('pod_qr'),difference=get('pca')-get('pod_qr')))
_save_csv(out/'individual_pca_mean_vs_pod.csv',individual)
fig,ax=plt.subplots(figsize=(9,4))
for seed in c['seeds']:
    group=[r for r in individual if r['seed']==seed]
    ax.plot([r['step']-100 for r in group],[r['difference'] for r in group],'o-',alpha=.5,label=str(seed))
ax.axhline(0,color='black',linewidth=.8);ax.set(xlabel='Pasos de continuación',ylabel='RMSE PCA + media − POD')
fig.tight_layout();fig.savefig(out/'individual_differences.png',dpi=150);plt.close(fig)
text+='\n\n## Medianas y resultados individuales\n\n| Pasos | Método | Mediana RMSE |\n|---:|---|---:|\n'
for r in summary:text+=f"| {r['step']-100} | {labels[r['method']]} | {r['rmse_median']:.5f} |\n"
text+='\n![Diferencias individuales](../../results/burgers/pde_bases/'+c['run_id']+'/individual_differences.png)\n\nValores negativos favorecen PCA + media. Se conservan todas las semillas, incluidos casos extremos, sin eliminarlos retrospectivamente. Los valores individuales están en individual_budget_metrics.csv y individual_pca_mean_vs_pod.csv.'
text=text.replace('{args.config}',args.config)
text+=f"\n\nLa comparación principal predefinida es {labels[primary_method]} frente a POD al presupuesto final de {c['train_steps']} pasos. Los demás presupuestos son descriptivos. La cohorte es la misma de la extensión previa; no es una confirmación con nuevas semillas. El rango emparejado se determina mediante la regla de PCA + media al 95 %, sin utilizar etiquetas."
primary=next(r for r in paired if r['step']==end and r['metric']=='test_rmse')
(out/'primary_budget_comparison.json').write_text(json.dumps(dict(method=primary_method,reference='pod_qr',budget=c['train_steps'],result=primary),indent=2))
p=ROOT/'pca_collocation/notebooks/burgers'/c['notebook_name'];n=json.loads(p.read_text(encoding='utf-8'))
n['cells']=[cell for cell in n['cells'] if cell.get('id') not in ('extended-budget','budget-analysis')]
for cell in n['cells']:
    if cell['cell_type']=='markdown':
        source=''.join(cell['source'])
        source=source.replace('Confirmación independiente de las bases del residuo','Comparación de bases al presupuesto ampliado')
        source=source.replace('Diez semillas nuevas 2091–2100','Las mismas diez semillas 2091–2100')
        source=source.replace('El protocolo se congela antes de ejecutar las cuarenta corridas.','El protocolo se conserva en esta extensión; regenerar el prefijo no aporta nuevas semillas independientes.')
        source=source.replace('Comparación principal: PCA centrada menos POD en RMSE terminal.',
            f"Comparación principal de esta extensión: {labels[primary_method]} menos POD al presupuesto final de {c['train_steps']} pasos.")
        source=source.replace('`python scripts/run_collocation_pde_bases_confirmation.py`',
            f'`python scripts/run_collocation_long_confirmation.py --config {args.config}`')
        source=source.replace('pca_collocation/configs/burgers_pde_bases_confirmation.json', args.config)
        cell['source']=source
n['cells'].insert(0,dict(id='budget-analysis',cell_type='markdown',metadata={},source=text))
p.write_text(json.dumps(n,ensure_ascii=False,indent=2),encoding='utf-8');embed_notebook_images(p,ROOT)
print(json.dumps(dict(summary=summary,paired=paired,prefixes_verified=len(prefixes)),indent=2))
