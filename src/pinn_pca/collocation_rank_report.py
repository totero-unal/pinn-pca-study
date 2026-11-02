"""Combined fixed/adaptive study report."""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from .collocation_basis_ablation import read_rows, LABELS
from .collocation import _save_csv
from .notebook_images import embed_notebook_images

def report(config,root,output):
    manifest=json.loads((output/'manifest.json').read_text())
    rows=[]
    for item in manifest:
        if item['status']!='complete': continue
        folder=root/item['path']
        metric=next(r for r in read_rows(folder/'metrics.csv') if r['method']==item['method'])
        trace=[r for r in read_rows(folder/'validation_trajectories.csv') if r['method']==item['method']]
        diag=[r for r in read_rows(folder/'selection_diagnostics.csv') if r['method']==item['method']]
        ranks=[int(r['actual_dimension']) for r in diag]
        rows.append(dict(id=item['id'],stage=item['stage'],method=item['method'],parameter=item['parameter'],seed=item['seed'],validation_rmse=float(trace[-1]['validation_rmse']),test_rmse=float(metric['test_rmse']),residual=float(metric['independent_residual_rms']),mean_rank=float(np.mean(ranks)),min_rank=min(ranks),max_rank=max(ranks)))
    _save_csv(output/'results_by_seed.csv',rows)
    summary=[]
    for key in dict.fromkeys(i['id'] for i in manifest):
        group=[r for r in rows if r['id']==key]
        item=next(i for i in manifest if i['id']==key)
        result=dict(id=key,stage=item['stage'],method=item['method'],parameter=item['parameter'],completed=len(group))
        for metric in ('validation_rmse','test_rmse','residual','mean_rank'):
            result[metric]=float(np.mean([r[metric] for r in group])) if group else None
        summary.append(result)
    _save_csv(output/'summary.csv',summary)
    selected={stage:min([r for r in summary if r['stage']==stage and r['completed']==len(config['seeds'])],key=lambda r:r['validation_rmse']) for stage in ('fixed','adaptive')}
    (output/'selection_by_validation.json').write_text(json.dumps(selected,indent=2))
    cells=[]
    def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
    md('# Bases del residuo: dimensión fija y rango adaptativo\n\nDiez semillas emparejadas, 100 pasos comunes y 10000 de continuación. Cada variante usa 32 puntos guiados y 96 Sobol. Red, pérdida histórica datos+PDE+IC, candidatos y ventana permanecen iguales al notebook 15. Sobol y dimensión 4 se reutilizan desde los artefactos auditados; manifest.json registra su procedencia.')
    md('## Definiciones\n\nEtapa 1: dimensiones totales 2, 4, 8 y 12; la media ocupa una dirección en PCA + media. Etapa 2: umbrales 95%, 99% y 99.9% sobre suma de valores singulares al cuadrado. POD conserva energía sin centrar; PCA conserva varianza centrada. PCA + media selecciona primero los componentes por varianza y añade la media si es independiente: su dimensión total puede ser mayor. Se registra la fracción realmente representada. No se rellenan bases sin rango suficiente; esos casos se declaran inviables. El rango adaptativo puede cambiar en cada actualización.')
    for stage,title in [('fixed','Etapa 1: dimensión fija'),('adaptive','Etapa 2: rango adaptativo')]:
        table='| Base | Parámetro | Semillas completas | Validación RMSE | Prueba RMSE | Residuo | Rango medio |\n|---|---:|---:|---:|---:|---:|---:|\n'
        for r in summary:
            if r['stage']!=stage: continue
            vals=[f"{r[k]:.5f}" if r[k] is not None else '—' for k in ('validation_rmse','test_rmse','residual','mean_rank')]
            table+=f"| {LABELS[r['method']]} | {r['parameter']} | {r['completed']}/10 | "+' | '.join(vals)+' |\n'
        md('## '+title+'\n\n'+table+'\nLas configuraciones incompletas no participan en la elección. La selección usa únicamente RMSE terminal de validación medio; las métricas de prueba describen el resultado y no deciden parámetros.')
        fig,axes=plt.subplots(1,3,figsize=(13,4))
        for method in list(LABELS)[1:]:
            group=sorted([r for r in summary if r['stage']==stage and r['method']==method and r['completed']==10],key=lambda r:r['parameter'])
            for ax,metric in zip(axes,('validation_rmse','test_rmse','mean_rank')):
                ax.plot([r['parameter'] for r in group],[r[metric] for r in group],'o-',label=LABELS[method]); ax.set(xlabel='Dimensión' if stage=='fixed' else 'Umbral',ylabel=metric)
        axes[0].legend();fig.tight_layout();path=output/(stage+'.png');fig.savefig(path,dpi=150);plt.close(fig)
        md('!['+stage+']('+os.path.relpath(path,root/'pca_collocation/notebooks/burgers').replace(os.sep,'/')+')')
    md('## Elección por validación\n\n'+json.dumps(selected,indent=2)+'\n\nLa elección es exploratoria sobre las mismas diez semillas. Una confirmación requiere nuevas semillas. Los tiempos de estas corridas incluyen diagnóstico y compilación. Los rangos por actualización están en selection_diagnostics.csv de cada corrida; results_by_seed.csv conserva dispersión y permite comparaciones pareadas.\n\nReproducir: `python scripts/run_collocation_rank_study.py`; regenerar: añadir `--report-only`.')
    failures=[i for i in manifest if i['status']!='complete']
    md('## Corridas inviables\n\n'+(json.dumps(failures,indent=2) if failures else 'Todas las combinaciones completaron sus diez semillas.'))
    notebook=root/'pca_collocation/notebooks/burgers'/config['notebook_name']
    notebook.write_text(json.dumps(dict(cells=cells,metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding='utf-8')
    embed_notebook_images(notebook,root)
    print(json.dumps(selected,indent=2),flush=True)
