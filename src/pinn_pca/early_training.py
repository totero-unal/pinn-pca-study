"""Retrospective early-validation comparisons; no model imports or test-set access."""
from __future__ import annotations
import csv
import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean,stdev


def _read_csv(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def category(ratio,band):
    if ratio<1-band:return 'PCA menor >5%'
    if ratio>1+band:return 'PCA mayor >5%'
    return 'Similar ±5%'


def _paired_summary(records,base,other,band):
    if not records:return {'pair_count':0}
    a=[float(r[base]) for r in records];b=[float(r[other]) for r in records]
    ratio=mean(b)/mean(a)
    return {'pair_count':len(a),'baseline_mean':mean(a),'pca_mean':mean(b),
        'baseline_std':stdev(a) if len(a)>1 else 0.,'pca_std':stdev(b) if len(b)>1 else 0.,
        'paired_delta_mean':mean(y-x for x,y in zip(a,b)),
        'ratio_of_means':ratio,'pca_better_count':sum(y<x for x,y in zip(a,b)),
        'category':category(ratio,band)}


def window_average(trace,start,end):
    selected=[r for r in trace if start<=r['step']<=end]
    if not selected or selected[0]['step']!=start or selected[-1]['step']!=end or end<=start:
        raise ValueError('El intervalo AUC debe estar delimitado por checkpoints observados.')
    return sum((b['step']-a['step'])*(a['validation_rmse']+b['validation_rmse'])/2
               for a,b in zip(selected,selected[1:]))/(end-start)


def last_before(trace,seconds):
    return next((r for r in reversed(trace) if r['elapsed_pipeline_seconds']<=seconds),None)


def analyze(config,root):
    root=Path(root);cache={};source_hashes={}
    checkpoint_pairs,time_pairs,auc_pairs=[],[],[]
    checkpoint_summary,time_summary,auc_summary=[],[],[]
    curves=[];variants=[]
    for case in config['cases']:
        source=case['source']
        if source not in cache:
            raw=(root/source).read_bytes();source_hashes[source]=hashlib.sha256(raw).hexdigest()
            cache[source]=_read_csv(root/source)
        selected=[r for r in cache[source] if all(str(r.get(k))==str(v) for k,v in case['filters'].items())]
        models=list(dict.fromkeys(r['model'] for r in selected))
        if config['baseline'] not in models or (case['primary_variant'] and case['primary_variant'] not in models):
            raise ValueError(f'Falta un modelo en {case["id"]}.')
        groups={}
        for record in selected:
            row={'step':int(record['step']),'validation_rmse':float(record['validation_rmse']),
                 'elapsed_pipeline_seconds':float(record['elapsed_pipeline_seconds'])}
            if not all(math.isfinite(v) for v in row.values()) or row['validation_rmse']<0:
                raise ValueError('Trayectoria no finita o RMSE negativo.')
            groups.setdefault((record['model'],int(record['seed'])),[]).append(row)
        seeds=sorted(s for m,s in groups if m==config['baseline'])
        for model in models:
            if sorted(s for m,s in groups if m==model)!=seeds:raise ValueError('Faltan semillas pareadas.')
        for trace in groups.values():
            if trace[0]['step']<=0 or trace[0]['elapsed_pipeline_seconds']<=0:raise ValueError('Checkpoint inicial inválido.')
            if any(b['step']<=a['step'] or b['elapsed_pipeline_seconds']<=a['elapsed_pipeline_seconds'] for a,b in zip(trace,trace[1:])):
                raise ValueError('Los checkpoints deben tener orden estricto de paso y tiempo.')
        if case['primary_variant']:
            for model in (config['baseline'],case['primary_variant']):
                for seed in seeds:
                    curves.extend({'case_id':case['id'],'label':case['label'],'model':model,'seed':seed,**p}
                                  for p in groups[(model,seed)] if p['step']<=config['auc_window_steps'][1])
        for model in models:
            if model==config['baseline']:continue
            info={'case_id':case['id'],'label':case['label'],'pca_scope':case['pca_scope'],
                  'model':model,'primary':model==case['primary_variant'],'source':source,'total_pairs':len(seeds)}
            variants.append(info)
            for step in config['checkpoints']:
                local=[]
                for seed in seeds:
                    a=next((r for r in groups[(config['baseline'],seed)] if r['step']==step),None)
                    b=next((r for r in groups[(model,seed)] if r['step']==step),None)
                    if a is None or b is None:raise ValueError(f'Falta el paso {step} en {case["id"]}.')
                    local.append({**info,'seed':seed,'step':step,'baseline_rmse':a['validation_rmse'],
                        'pca_rmse':b['validation_rmse'],'baseline_seconds':a['elapsed_pipeline_seconds'],
                        'pca_seconds':b['elapsed_pipeline_seconds']})
                checkpoint_pairs.extend(local)
                checkpoint_summary.append({**info,'step':step,**_paired_summary(local,'baseline_rmse','pca_rmse',config['similarity_band'])})
            start,end=config['auc_window_steps'];local=[]
            for seed in seeds:
                a,b=groups[(config['baseline'],seed)],groups[(model,seed)]
                if [r['step'] for r in a if start<=r['step']<=end]!=[r['step'] for r in b if start<=r['step']<=end]:
                    raise ValueError('Las rejillas AUC de la pareja deben coincidir.')
                local.append({**info,'seed':seed,'start_step':start,'end_step':end,
                    'baseline_window_rmse':window_average(a,start,end),'pca_window_rmse':window_average(b,start,end)})
            auc_pairs.extend(local)
            auc_summary.append({**info,'start_step':start,'end_step':end,
                **_paired_summary(local,'baseline_window_rmse','pca_window_rmse',config['similarity_band'])})
            for seconds in config['time_budgets_seconds']:
                local=[];missing=0
                for seed in seeds:
                    a=last_before(groups[(config['baseline'],seed)],seconds)
                    b=last_before(groups[(model,seed)],seconds)
                    if a is None or b is None:missing+=1;continue
                    local.append({**info,'seed':seed,'budget_seconds':seconds,'baseline_step':a['step'],'pca_step':b['step'],
                        'baseline_evaluation_seconds':a['elapsed_pipeline_seconds'],'pca_evaluation_seconds':b['elapsed_pipeline_seconds'],
                        'baseline_rmse':a['validation_rmse'],'pca_rmse':b['validation_rmse'],
                        'baseline_finished':groups[(config['baseline'],seed)][-1]['elapsed_pipeline_seconds']<=seconds,
                        'pca_finished':groups[(model,seed)][-1]['elapsed_pipeline_seconds']<=seconds})
                time_pairs.extend(local)
                time_summary.append({**info,'budget_seconds':seconds,'missing_pairs':missing,
                    **_paired_summary(local,'baseline_rmse','pca_rmse',config['similarity_band'])})
    counts={}
    for step in config['checkpoints']:
        selected=[r for r in checkpoint_summary if r['primary'] and r['step']==step]
        counts[str(step)]={'case_count':len(selected),'lower_mean_count':sum(r['ratio_of_means']<1 for r in selected),
            **{label:sum(r['category']==label for r in selected) for label in ['PCA menor >5%','Similar ±5%','PCA mayor >5%']}}
    selected=[r for r in auc_summary if r['primary']]
    counts['window_100_500']={'case_count':len(selected),'lower_mean_count':sum(r['ratio_of_means']<1 for r in selected),
        **{label:sum(r['category']==label for r in selected) for label in ['PCA menor >5%','Similar ±5%','PCA mayor >5%']}}
    return {'configuration':config,'source_sha256':source_hashes,'checkpoint_summary':checkpoint_summary,
        'time_summary':time_summary,'auc_summary':auc_summary,'primary_case_counts':counts,
        'variants':variants,'curves':curves}, {
        'paired_checkpoints.csv':checkpoint_pairs,'checkpoint_summary.csv':checkpoint_summary,
        'paired_window_auc.csv':auc_pairs,'window_auc_summary.csv':auc_summary,
        'paired_time_budgets.csv':time_pairs,'time_budget_summary.csv':time_summary}


def _table(headers,rows):
    return ('<div style="overflow-x:auto"><table><thead><tr>'+''.join(f'<th>{html.escape(str(h))}</th>' for h in headers)
            +'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>')


def primary_table(report):
    cases=[c for c in report['configuration']['cases'] if c['primary_variant']]
    steps=report['configuration']['checkpoints'];rows=[]
    for c in cases:
        values=[next(r for r in report['checkpoint_summary'] if r['case_id']==c['id'] and r['primary'] and r['step']==s) for s in steps]
        window=next(r for r in report['auc_summary'] if r['case_id']==c['id'] and r['primary'])
        rows.append([c['label'],*[f'{r["ratio_of_means"]:.3f}× ({r["pca_better_count"]}/{r["pair_count"]})' for r in values],
                     f'{window["ratio_of_means"]:.3f}× ({window["pca_better_count"]}/{window["pair_count"]})'])
    return _table(['Caso',*[f'Paso {s}: razón y semillas PCA menor' for s in steps],'Ventana 100–500: razón y semillas PCA menor'],rows)


def counts_table(report):
    return _table(['Checkpoint/ventana','Casos','PCA menor >5%','Similar ±5%','PCA mayor >5%'],[
        [key,r['case_count'],r['PCA menor >5%'],r['Similar ±5%'],r['PCA mayor >5%']] for key,r in report['primary_case_counts'].items()])


def time_table(report):
    rows=[]
    for c in report['configuration']['cases']:
        if not c['primary_variant']:continue
        for seconds in report['configuration']['time_budgets_seconds']:
            r=next(r for r in report['time_summary'] if r['case_id']==c['id'] and r['primary'] and r['budget_seconds']==seconds)
            rows.append([c['label'],seconds,f'{r["pair_count"]}/{r["total_pairs"]}',
                f'{r["ratio_of_means"]:.3f}×' if r['pair_count'] else 'Sin observación pareada',
                f'{r["pca_better_count"]}/{r["pair_count"]}' if r['pair_count'] else '—'])
    return _table(['Caso','Presupuesto (s)','Pares observados','Razón RMSE PCA/base','Semillas PCA menor'],rows)


def secondary_table(report):
    rows=[[r['label'],r['model'],r['step'],f'{r["ratio_of_means"]:.3f}×',f'{r["pca_better_count"]}/{r["pair_count"]}']
          for r in report['checkpoint_summary'] if not r['primary']]
    return _table(['Caso','Variante adicional','Paso','Razón RMSE PCA/base','Semillas PCA menor'],rows)


def _artifacts(report,root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    cases=[c for c in report['configuration']['cases'] if c['primary_variant']]
    steps=report['configuration']['checkpoints'];names=[]
    def save(fig,name):
        fig.savefig(output/f'{name}.svg',bbox_inches='tight')
        fig.savefig(output/f'{name}.png',bbox_inches='tight',dpi=160)
        plt.close(fig);names.append(name)
    with plt.rc_context({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False}):
        ratios=[]
        for c in cases:
            r=[next(r for r in report['checkpoint_summary'] if r['case_id']==c['id'] and r['primary'] and r['step']==s)['ratio_of_means'] for s in steps]
            r.append(next(r for r in report['auc_summary'] if r['case_id']==c['id'] and r['primary'])['ratio_of_means']);ratios.append(r)
        data=np.log2(ratios);bound=max(1.,float(np.max(abs(data))))
        fig,ax=plt.subplots(figsize=(10,7.7),layout='constrained')
        im=ax.imshow(data,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
        for i,row in enumerate(ratios):
            for j,value in enumerate(row):ax.text(j,i,f'{value:.3f}×',ha='center',va='center',color='white' if abs(np.log2(value))>.65*bound else 'black',fontsize=10)
        ax.set_yticks(range(len(cases)),[c['label'] for c in cases]);ax.set_xticks(range(4),[f'Paso {s}' for s in steps]+['Ventana\n100–500'])
        ax.set_title('RMSE de validación PCA al 99% / PINN base\nMenor que 1 favorece PCA; valores próximos a 1 indican similitud')
        bar=fig.colorbar(im,ax=ax,shrink=.85,ticks=[-1,0,1]);bar.ax.set_yticklabels(['0.5×','1×','2×']);bar.set_label('Razón de RMSE (escala logarítmica)')
        save(fig,'early_rmse_ratios')

        fig,axes=plt.subplots(5,2,figsize=(12,15),layout='constrained')
        for ax,c in zip(axes.flat,cases):
            for model,color,label in [(report['configuration']['baseline'],'#526274','PINN base'),(c['primary_variant'],'#c8791c','PCA 99%')]:
                selected=[r for r in report['curves'] if r['case_id']==c['id'] and r['model']==model]
                xs=sorted({r['step'] for r in selected});ys=[np.asarray([r['validation_rmse'] for r in selected if r['step']==s]) for s in xs]
                means=np.asarray([v.mean() for v in ys]);sd=np.asarray([v.std(ddof=1) for v in ys])
                ax.plot(xs,means,color=color,label=label);ax.fill_between(xs,np.maximum(0,means-sd),means+sd,color=color,alpha=.13)
            ax.set(title=c['label'],xlabel='Pasos registrados',ylabel='RMSE validación');ax.grid(alpha=.2)
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=2)
        save(fig,'first_500_steps')

        keys=list(report['primary_case_counts']);labels=['PCA menor >5%','Similar ±5%','PCA mayor >5%'];colors=['#238b6b','#b5bbc1','#bd5447']
        fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');bottom=np.zeros(len(keys))
        for label,color in zip(labels,colors):
            values=np.asarray([report['primary_case_counts'][k][label] for k in keys]);ax.bar(range(len(keys)),values,bottom=bottom,color=color,label=label)
            for i,v in enumerate(values):
                if v:ax.text(i,bottom[i]+v/2,str(v),ha='center',va='center')
            bottom+=values
        ax.set_xticks(range(len(keys)),[f'Paso {s}' for s in steps]+['Ventana 100–500']);ax.set(ylabel='Número de casos',title='Resumen descriptivo de diez casos; banda de similitud ±5%',ylim=(0,len(cases)))
        ax.legend(loc='upper center',bbox_to_anchor=(.5,-.14),ncol=3,fontsize=9)
        save(fig,'case_counts')

        budgets=report['configuration']['time_budgets_seconds'];data=np.full((len(cases),len(budgets)),np.nan);cells={}
        for i,c in enumerate(cases):
            for j,seconds in enumerate(budgets):
                r=next(r for r in report['time_summary'] if r['case_id']==c['id'] and r['primary'] and r['budget_seconds']==seconds)
                if r['pair_count']==r['total_pairs']:data[i,j]=np.log2(r['ratio_of_means']);cells[i,j]=f'{r["ratio_of_means"]:.3f}×'
                else:cells[i,j]=f'{r["pair_count"]}/{r["total_pairs"]} pares\nSin estimación completa'
        finite=data[np.isfinite(data)];bound=max(1.,float(np.max(abs(finite))))
        fig,ax=plt.subplots(figsize=(10,7.7),layout='constrained');cmap=plt.get_cmap('RdBu_r').copy();cmap.set_bad('#eeeeee')
        im=ax.imshow(data,aspect='auto',cmap=cmap,norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
        for (i,j),text in cells.items():ax.text(j,i,text,ha='center',va='center',fontsize=9,color='white' if np.isfinite(data[i,j]) and abs(data[i,j])>.65*bound else 'black')
        ax.set_yticks(range(len(cases)),[c['label'] for c in cases]);ax.set_xticks(range(len(budgets)),[f'{s:g} s' for s in budgets])
        ax.set_title('Última validación registrada antes del presupuesto\nRazón RMSE PCA/base; pipeline incluye ajuste PCA, excluye compilación')
        bar=fig.colorbar(im,ax=ax,shrink=.85,ticks=[-1,0,1]);bar.ax.set_yticklabels(['0.5×','1×','2×'])
        save(fig,'early_time_budgets')
    _notebook(report,root,output,names)


def _markdown_report(report,root):
    counts=report['primary_case_counts']
    lines=['# PCA frente a PINN base en las primeras iteraciones','',
        'Análisis retrospectivo de validación guardada, sin nuevos entrenamientos. La comparación principal usa PCA al 99% '
        'en diez casos y no cuenta repetidamente prefijos ni controles reproducidos. Los casos comparten familias y semillas; '
        'no son diez réplicas independientes.','',
        '## Resultado descriptivo','',
        '| Checkpoint o ventana | PCA menor >5% | Similar ±5% | PCA mayor >5% |',
        '| --- | --- | --- | --- |']
    for key,r in counts.items():
        lines.append(f'| {key} | {r["PCA menor >5%"]} | {r["Similar ±5%"]} | {r["PCA mayor >5%"]} |')
    lines += ['','En el paso 100, los diez casos tienen menor RMSE medio con PCA; ocho reducen más del 5%. '
        'La ventaja no se mantiene en todos los problemas. En el promedio aproximado de la ventana 100–500, '
        'dos casos reducen más del 5%, tres quedan similares y cinco aumentan más del 5%. La banda de similitud '
        'es descriptiva y no representa significación estadística.','',
        '| Caso | Paso 100 | Paso 200 | Paso 500 | Ventana 100–500 |',
        '| --- | --- | --- | --- | --- |']
    for c in report['configuration']['cases']:
        if not c['primary_variant']:continue
        values=[next(r for r in report['checkpoint_summary'] if r['case_id']==c['id'] and r['primary'] and r['step']==s)['ratio_of_means'] for s in report['configuration']['checkpoints']]
        values.append(next(r for r in report['auc_summary'] if r['case_id']==c['id'] and r['primary'])['ratio_of_means'])
        lines.append('| '+c['label']+' | '+' | '.join(f'{v:.3f}×' for v in values)+' |')
    lines += ['','Cada valor es media RMSE PCA / media RMSE base: menor que 1 favorece PCA. La ventana se resume '
        'con integración trapezoidal de checkpoints observados, dividida por su longitud.','',
        '## Alcance','',
        '- Los primeros checkpoints comunes son 100, 200 y 500; no se conoce la comparación común de pasos 1–99 ni del estado inicial.\n'
        '- Se mide calidad de u con validación; no hay residuos PDE reservados para todos esos checkpoints.\n'
        '- En PCA espacial la base ya incorpora información de snapshots de entrenamiento. Una buena salida del pipeline no demuestra que el optimizador sea intrínsecamente más rápido.\n'
        '- PCA de entradas y PCA espacial tienen alcances distintos. Se conservan todos los k, pesos físicos y escalas adicionales en tablas separadas, sin elegir el mejor con estos resultados.\n'
        '- La comparación temporal usa la última evaluación registrada antes de 0.5, 1 y 2 s, con conteo explícito de pares faltantes; no equivale a reevaluar exactamente en esos tiempos.\n'
        '- La onda mantiene una predicción próxima a cero mientras la base mejora. La superioridad en un checkpoint aislado no garantiza aprendizaje físico ni calidad final.','',
        '## Artefactos','',
        '- [Notebook con tablas y figuras](../notebooks/01_early_training_comparison.ipynb)\n'
        '- [Resumen JSON y huellas de fuentes](../results/early_training/summary.json)\n'
        '- [Comparación por checkpoint](../results/early_training/checkpoint_summary.csv)\n'
        '- [Resumen de ventana](../results/early_training/window_auc_summary.csv)\n'
        '- [Configuración y fuentes canónicas](../configs/early_training_analysis.json)','',
        'Reproducir: `python scripts/analyze_early_training.py`. No se importa JAX ni se modifican entrenamientos guardados.']
    (root/'reports/early_training.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def _notebook(report,root,output,names):
    def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
    def code(text,data=None):return {'cell_type':'code','metadata':{},'execution_count':None,'source':text.splitlines(True),
        'outputs':[{'output_type':'display_data','metadata':{},'data':data}] if data else []}
    relative=output.relative_to(root).as_posix()
    cells=[md('# PCA frente a PINN base al inicio del entrenamiento\n\n'
        'Análisis retrospectivo de validación guardada, sin nuevos entrenamientos ni uso de prueba. '
        'La comparación principal usa PCA al 99% en diez casos: Burgers, tres condiciones de advección, '
        'tres dimensiones alineadas, un control desalineado y dos condiciones de onda. Se conserva una sola '
        'trayectoria por modelo/caso/semilla: se evita contar repetidamente prefijos de 2000 y 5000 pasos o '
        'controles reproducidos. Estos casos comparten familias, semillas e hipótesis: no son diez réplicas independientes.\n\n'
        '## Qué significan los primeros pasos\n\n'
        'Los primeros checkpoints comunes disponibles son 100, 200 y 500. No hay información común para afirmar '
        'qué ocurre en los pasos 1–99 ni antes de entrenar. Se evalúa error de u con RMSE de validación; no se '
        'guardó residuo PDE de validación en todos esos pasos y no se infiere de la pérdida de entrenamiento.\n\n'
        'La razón es media RMSE PCA / media RMSE base dentro del mismo caso. Menor que 1 favorece PCA. '
        'La banda ±5% es descriptiva y se fijó en la configuración de este análisis; no es un umbral de significación. '
        'Las semillas están pareadas y los conteos indican cuántas obtienen menor RMSE con PCA. '
        'Los k fijos, pesos físicos y escalas alternativas se muestran aparte y no se elige el mejor retrospectivamente.\n\n'
        '## Ventana completa y tiempo\n\n'
        'El promedio de RMSE en 100–500 se calcula por regla trapezoidal sobre checkpoints observados, dividido '
        'por la longitud de la ventana. Es un resumen aproximado entre observaciones, no una medición continua. '
        'Evita juzgar todo el inicio a partir del único checkpoint favorable de 100 pasos.\n\n'
        'Para tiempo se usa la última evaluación registrada antes de 0.5, 1 y 2 s. No se interpola al tiempo '
        'exacto ni se inventan resultados cuando no existe observación. La tabla indica pares faltantes y la '
        'figura solo colorea comparaciones con todos los pares. El tiempo incluye ajuste PCA y validaciones; '
        'excluye compilación. Las corridas de fechas distintas conservan sus tiempos medidos y solo se comparan '
        'modelos dentro de su corrida.'),
        code("from pathlib import Path\nimport json,sys\nfrom IPython.display import display,HTML,SVG\nROOT=Path.cwd()\nwhile not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent:\n    ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'src'))\nfrom pinn_pca.early_training import primary_table,counts_table,time_table,secondary_table\n"
             f"RUN_DIR=ROOT/'{relative}'\nREPORT=json.loads((RUN_DIR/'summary.json').read_text(encoding='utf-8'))"),
        md('## Comparación principal: PCA al 99%\n\nCada celda contiene la razón y el número de semillas con RMSE menor con PCA.'),
        code('display(HTML(primary_table(REPORT)))',{'text/html':primary_table(report)}),
        code('display(HTML(counts_table(REPORT)))',{'text/html':counts_table(report)})]
    for name in names[:3]:cells.append(code(f"display(SVG(filename=str(RUN_DIR/'{name}.svg')))",{'image/svg+xml':(output/f'{name}.svg').read_text(encoding='utf-8')}))
    cells += [md('## Presupuesto de tiempo: aproximación con registros disponibles\n\n'
        'Se comparan los últimos estados evaluados antes de cada presupuesto, que pueden corresponder a pasos distintos. '
        'No equivale a reevaluar el modelo exactamente a T segundos. Los pares sin evaluación de ambos modelos se '
        'excluyen y se cuentan explícitamente. No comparar valores absolutos entre PDE.'),
        code('display(HTML(time_table(REPORT)))',{'text/html':time_table(report)}),
        code(f"display(SVG(filename=str(RUN_DIR/'{names[-1]}.svg')))",{'image/svg+xml':(output/f'{names[-1]}.svg').read_text(encoding='utf-8')}),
        md('## Variantes adicionales\n\nSe conservan todos los modelos de cada corrida elegida, incluidos k fijos, '
           'pesos de Burgers y estandarización de la onda. No se suman al conteo principal: hacerlo daría más peso '
           'a los problemas con más variantes. Las comparaciones con distintos pesos de pérdida no aíslan PCA.'),
        code('display(HTML(secondary_table(REPORT)))',{'text/html':secondary_table(report)})]
    observations=[]
    for key,r in report['primary_case_counts'].items():
        observations.append(f'- **{key}**: {r["lower_mean_count"]}/{r["case_count"]} casos con media menor; '
            f'{r["PCA menor >5%"]} con reducción superior al 5%, {r["Similar ±5%"]} similares y {r["PCA mayor >5%"]} con aumento superior al 5%.')
    cells.append(md('## Interpretación\n\n'+'\n'.join(observations)+'\n\n'
        'Un buen primer checkpoint no asegura convergencia más rápida después. En PCA espacial, la base ya incorpora '
        'información de los snapshots de entrenamiento antes de optimizar; esto aporta una representación previa '
        'distinta de la PINN base. Su costo está incluido en el tiempo del pipeline, pero las funciones físicas '
        'iniciales de las dos arquitecturas no son iguales. En el control de escala de onda se igualaron las dos '
        'variantes PCA entre sí, no con la base.\n\n'
        'La onda ilustra una ventaja inicial engañosa: PCA permanece cerca de cero mientras la base mejora y la '
        'supera. Las diferencias de PCA de entradas suelen ser menores que las de PCA espacial. Las curvas y la '
        'ventana completa deben acompañar al valor del primer checkpoint. Estos datos describen calidad temprana '
        'de u y no demuestran una mejora temprana del residuo físico ni un beneficio general de PCA.\n\n'
        'La configuración explicita las fuentes canónicas y las variantes principales. Las huellas de los CSV '
        'se conservan en el resumen. Reproducir: `python scripts/analyze_early_training.py`. Esta operación no '
        'importa JAX, no cambia los entrenamientos y mantiene cerrados los benchmarks ya documentados.'))
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    (root/report['configuration']['notebook']).write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')


def run_analysis(config,root):
    root=Path(root).resolve();report,files=analyze(config,root)
    output=root/config['output_dir'];output.mkdir(parents=True,exist_ok=True)
    for name,records in files.items():
        with (output/name).open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    _artifacts(report,root,output)
    _markdown_report(report,root)
    return report
