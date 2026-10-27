"""JAX-free report generation for the controlled wave diagnostic round."""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .wave_report import table

BASE, RAW, NORMALIZED = "PINN base", "PCA espacial 99%", "PCA estandarizada 99%"
MODELS = (BASE, RAW, NORMALIZED)
REFERENCES = ("Referencia analítica", "PCA oráculo seno", "PCA oráculo spline C2")
COLORS = ("#526274", "#c8791c", "#238b6b")
TERMS = ("data", "pde", "initial_u", "initial_ut", "boundary")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def oracle_table(oracle):
    rows=[]
    for case in oracle["source_configuration"]["cases"]:
        for scope in ("validation","test"):
            for rep in REFERENCES:
                selected=[r for r in oracle["records"] if r['case']==case['name'] and r['evaluation']==scope and r['representation']==rep]
                keys=("relative_l2","pde_residual_rms","normalized_pde_residual","oracle_to_null_loss_ratio")
                rows.append([case['name'],scope,rep,*[f'{np.mean([r[key] for r in selected]):.4g}' for key in keys]])
    return table(["Caso","Evaluación","Representación","L2 de u","Residuo RMS","Residuo normalizado","Pérdida / nula"],rows)


def metric_table(result):
    rows=[]
    keys=("test_relative_l2","test_pde_residual_rms","validation_rmse","initial_u_rmse","pipeline_seconds")
    for case,models in result['summary'].items():
        for model in MODELS:
            rows.append([case,model,*[f'{models[model][key]["mean"]:.4g} ± {models[model][key]["std"]:.3g}' for key in keys]])
    return table(["Caso","Modelo","L2 prueba","Residuo RMS prueba","RMSE validación","RMSE inicial u","Pipeline (s)"],rows)


def pair_table(result):
    rows=[]
    for case,comparisons in result['paired'].items():
        comparison=f'{NORMALIZED} minus {RAW}'
        for key in ("test_relative_l2","test_pde_residual_rms","validation_rmse","pipeline_seconds"):
            r=comparisons[comparison][key]
            rows.append([case,key,f'{r["other_minus_reference_mean"]:.4g}',
                '['+', '.join(f'{v:.4g}' for v in r['bootstrap_ci_95'])+']',
                f'{r["other_better_count"]}/{r["seed_count"]}'])
    return table(["Caso","Métrica","Estandarizada − original","IC bootstrap 95%","Estandarizada menor"],rows)


def gradient_ratios(records):
    groups={}
    for r in records:
        groups.setdefault((r['case'],r['model'],int(r['seed']),int(r['step'])),{})[r['term']]=float(r['parameter_gradient_norm'])
    return [{"case":case,"model":model,"seed":seed,"step":step,
             "ratio":values['pde']/max(sum(v for k,v in values.items() if k!='pde'),1e-30)}
             for (case,model,seed,step),values in groups.items()]


def gradient_table(records):
    ratios=gradient_ratios(records);rows=[]
    for case in dict.fromkeys(r['case'] for r in records):
        for step in sorted({int(r['step']) for r in records}):
            for model in MODELS:
                values=[r['ratio'] for r in ratios if r['case']==case and r['model']==model and r['step']==step]
                rows.append([case,model,step,f'{np.median(values):.4g}',
                    '['+', '.join(f'{v:.4g}' for v in np.percentile(values,[25,75]))+']'])
    return table(["Caso","Modelo","Paso","Mediana norma grad PDE / suma otras","Rango intercuartílico"],rows)


def scaler_table(records):
    rows=[]
    for case in dict.fromkeys(r['case'] for r in records):
        for component in sorted({int(r['component']) for r in records if r['case']==case}):
            selected=[r for r in records if r['case']==case and int(r['component'])==component]
            rows.append([case,component,f'{np.mean([float(r["training_mean"]) for r in selected]):.3g}',
                         f'{np.mean([float(r["training_std"]) for r in selected]):.4g}'])
    return table(["Caso","Componente","Centro medio (solo ajuste)","Escala media (solo ajuste)"],rows)


def reproduction_audit(root,output,result,traces):
    source=root/result['configuration']['source_run']
    original=read_csv(source/'validation_trajectories.csv')
    key=lambda r:(r['case'],r['model'],int(r['seed']),int(r['step']))
    a={key(r):float(r['validation_rmse']) for r in original}
    b={key(r):float(r['validation_rmse']) for r in traces if r['model'] in (BASE,RAW)}
    shared=set(a)&set(b)
    audit={"matching_checkpoint_keys":len(shared),"expected_source_checkpoints":len(a),
           "max_absolute_validation_difference":max((abs(a[k]-b[k]) for k in shared),default=None),
           "physical_initialization_max_abs_difference":max(p['physical_initialization_max_abs_difference']
               for parts in result['partitions'].values() for p in parts.values()),
           "source_hashes_match":all(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
               for name,digest in result['source_sha256'].items())}
    (output/'reproduction_check.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    return audit


def create_report(project_root,output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root,output=Path(project_root),Path(output_dir)
    oracle=json.loads((output/'oracle_summary.json').read_text(encoding='utf-8'))
    result=(json.loads((output/'summary.json').read_text(encoding='utf-8')) if (output/'summary.json').exists() else None)
    if result is not None and result['configuration']!=oracle['configuration']:
        raise ValueError('Las configuraciones del control oráculo y entrenamiento no coinciden.')
    cases=[c['name'] for c in oracle['source_configuration']['cases']]
    names=['Dos frecuencias','Tres frecuencias'] if len(cases)==2 else cases
    figures=[]
    def save(fig,name):
        fig.savefig(output/f'{name}.svg',bbox_inches='tight')
        fig.savefig(output/f'{name}.png',bbox_inches='tight',dpi=160)
        plt.close(fig);figures.append(name)
    with plt.rc_context({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'}):
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        labels=['Exacta','PCA + seno','PCA + spline C2']
        for col,key in enumerate(['pde_residual_rms','normalized_pde_residual']):
            ax=axes[0,col]
            for index,(rep,color) in enumerate(zip(REFERENCES,COLORS)):
                values=[np.mean([r[key] for r in oracle['records'] if r['evaluation']=='validation' and r['case']==case and r['representation']==rep]) for case in cases]
                ax.bar(np.arange(len(cases))+(index-1)*.24,values,width=.23,color=color,label=labels[index])
            ax.set_xticks(range(len(cases)),names)
            ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
            ax.set_title('Residuo absoluto RMS' if col==0 else 'Residuo / escala de derivadas')
            if col==1:ax.axhline(oracle['gate']['thresholds']['max_normalized_pde_residual'],color='black',ls='--',lw=1)
        for col,(key,threshold,title) in enumerate([
            ('relative_l2','max_validation_relative_l2','Error L2 de u: oráculo spline C2'),
            ('oracle_to_null_loss_ratio','max_oracle_to_null_loss_ratio','Pérdida oráculo C2 / solución nula')]):
            ax=axes[1,col]
            for i,case in enumerate(cases):
                vals=[r[key] for r in oracle['records'] if r['evaluation']=='validation' and r['case']==case and r['representation']==REFERENCES[-1]]
                ax.scatter(np.full(len(vals),i),vals,color=COLORS[2],s=22,alpha=.65)
            ax.axhline(oracle['gate']['thresholds'][threshold],color='black',ls='--',lw=1)
            ax.set_xticks(range(len(cases)),names);ax.set_yscale('log');ax.set_title(title);ax.grid(axis='y',alpha=.2)
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='outside upper center',ncol=3)
        save(fig,'oracle_physics')

        if result is not None:
            traces=read_csv(output/'validation_trajectories.csv')
            gradients=read_csv(output/'gradient_diagnostics.csv')
            scalers=read_csv(output/'coefficient_scalers.csv')
            modes=read_csv(output/'mode_metrics.csv')
            audit=reproduction_audit(root,output,result,traces)
            metrics=[('test_relative_l2','Error relativo L2'),('test_pde_residual_rms','Residuo RMS (AD)'),('pipeline_seconds','Pipeline (s)')]
            fig,axes=plt.subplots(len(cases),3,figsize=(13,3.5*len(cases)),squeeze=False,layout='constrained')
            for i,case in enumerate(cases):
                for j,(metric,label) in enumerate(metrics):
                    stats=[result['summary'][case][m][metric] for m in MODELS]
                    axes[i,j].bar(range(3),[v['mean'] for v in stats],yerr=[v['std'] for v in stats],color=COLORS,capsize=3)
                    axes[i,j].set_xticks(range(3),['Base','PCA original','PCA estand.'],fontsize=8)
                    axes[i,j].set(ylabel=label,title=names[i],ylim=(0,None));axes[i,j].grid(axis='y',alpha=.2)
            save(fig,'learning_comparison')

            fig,axes=plt.subplots(1,len(cases),figsize=(6*len(cases),4.5),squeeze=False,layout='constrained')
            for ax,case,title in zip(axes[0],cases,names):
                for model,color in zip(MODELS,COLORS):
                    selected=[r for r in traces if r['case']==case and r['model']==model]
                    steps=sorted({int(r['step']) for r in selected})
                    vals=[np.asarray([float(r['validation_rmse']) for r in selected if int(r['step'])==step]) for step in steps]
                    mean=np.asarray([v.mean() for v in vals]);sd=np.asarray([v.std(ddof=1) for v in vals])
                    ax.plot(steps,mean,color=color,label=model);ax.fill_between(steps,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.12)
                ax.set(title=title,xlabel='Pasos',ylabel='RMSE validación');ax.grid(alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
            save(fig,'validation_curves')

            ratios=gradient_ratios(gradients)
            checkpoints=sorted({r['step'] for r in ratios})
            fig,axes=plt.subplots(1,len(cases),figsize=(6*len(cases),4.5),squeeze=False,layout='constrained')
            for ax,case,title in zip(axes[0],cases,names):
                for i,(model,color) in enumerate(zip(MODELS,COLORS)):
                    vals=[np.asarray([r['ratio'] for r in ratios if r['case']==case and r['model']==model and r['step']==s]) for s in checkpoints]
                    med=np.asarray([np.median(v) for v in vals]);lo=np.asarray([np.percentile(v,25) for v in vals]);hi=np.asarray([np.percentile(v,75) for v in vals])
                    ax.bar(np.arange(len(checkpoints))+(i-1)*.24,med,width=.23,yerr=[med-lo,hi-med],color=color,capsize=3,label=model)
                ax.set_xticks(range(len(checkpoints)),['Inicio' if s==0 else f'Paso {s}' for s in checkpoints]);ax.set_yscale('log')
                ax.set(title=title,ylabel='Norma grad PDE / suma otras normas');ax.grid(axis='y',alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
            save(fig,'gradient_imbalance')

            fig,axes=plt.subplots(1,len(cases),figsize=(6*len(cases),4.5),squeeze=False,layout='constrained')
            for ax,case,title in zip(axes[0],cases,names):
                ns=sorted({int(r['harmonic']) for r in modes if r['case']==case})
                for i,(model,color) in enumerate(zip(MODELS,COLORS)):
                    vals=[[float(r['coefficient_relative_l2']) for r in modes if r['case']==case and r['model']==model and int(r['harmonic'])==n] for n in ns]
                    ax.bar(np.arange(len(ns))+(i-1)*.24,[np.mean(v) for v in vals],width=.23,yerr=[np.std(v,ddof=1) for v in vals],color=color,capsize=3,label=model)
                ax.set_xticks(range(len(ns)),ns);ax.set(title=title,xlabel='Armónico n',ylabel='Error L2 del coeficiente',ylim=(0,None));ax.grid(axis='y',alpha=.2)
            handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside upper center',ncol=3,fontsize=9)
            save(fig,'frequency_errors')

            source=root/result['configuration']['source_run']
            for case,title in zip(cases,names):
                with np.load(output/f'example_{case}.npz') as data:
                    x,t,truth,base,raw,norm=[data[k] for k in ('x','t','truth','baseline','raw_pca','normalized_pca')];seed=int(data['seed'])
                with np.load(source/f'example_{case}.npz') as data:
                    projection=data['oracle']
                    if int(data['seed'])!=seed:raise ValueError('Las figuras fuente usan otra semilla.')
                fig,axes=plt.subplots(2,4,figsize=(14,6.2),layout='constrained')
                extent=[float(t[0]),float(t[-1]),float(x[0]),float(x[-1])];vmax=max(abs(truth.min()),abs(truth.max()))
                for ax,field,label in zip(axes[0],[truth,base,raw,norm],['Exacta','Base','PCA original','PCA estandarizada']):
                    im=ax.imshow(field,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r',vmin=-vmax,vmax=vmax);ax.set(title=label,xlabel='t',ylabel='x')
                fig.colorbar(im,ax=axes[0].tolist(),label='u',shrink=.8)
                errors=[abs(v-truth) for v in [projection,base,raw,norm]];emax=max(v.max() for v in errors)
                for ax,error,label in zip(axes[1],errors,['Error proyección en malla\n(oráculo, no red)','Error base','Error PCA original','Error PCA estand.']):
                    im=ax.imshow(error,origin='lower',aspect='auto',extent=extent,cmap='magma',vmin=0,vmax=emax);ax.set(title=label,xlabel='t',ylabel='x')
                fig.colorbar(im,ax=axes[1].tolist(),label='|error|',shrink=.8);fig.suptitle(f'{title} · semilla preseleccionada {seed}')
                save(fig,f'fields_{case}')

    def md(text):return {'cell_type':'markdown','metadata':{},'source':text.splitlines(True)}
    def code(text,data=None):return {'cell_type':'code','metadata':{},'execution_count':None,'source':text.splitlines(True),
        'outputs':[{'output_type':'display_data','metadata':{},'data':data}] if data else []}
    relative=output.relative_to(root).as_posix()
    cells=[md("# Diagnóstico de la onda: representación, interpolación y escala de coeficientes\n\n"
        "Se conserva el piloto original y se estudian dos preguntas: ¿la base PCA con coeficientes conocidos puede "
        "satisfacer aproximadamente la física?, ¿estandarizar sus coeficientes mejora el entrenamiento sin cambiar "
        "datos, k, arquitectura, pérdida ni pasos? Los outputs provienen de artefactos guardados.\n\n"
        "## Control oráculo previo\n\n"
        "Se proyecta la solución analítica sobre los modos PCA ajustados con entrenamiento. Sus coeficientes son funciones "
        "analíticas del tiempo, por lo que JAX calcula u_tt y u_xx sin diferencias finitas. Este control no se entrena ni "
        "se usa como competidor. Se evalúan la referencia directa, la misma base PCA expresada en senos conocidos y "
        "su interpolación spline C2 usada por la red. Así se distingue compresión espacial de error de interpolación.\n\n"
        "El residuo normalizado es RMS(u_tt − 4u_xx) / [RMS(u_tt) + RMS(4u_xx)]. Se muestran también los valores "
        "absolutos: normalizar el diagnóstico no cambia la pérdida de entrenamiento. La pérdida oráculo suma los "
        "mismos cinco términos con peso uno; la solución nula falla en datos y desplazamiento inicial.\n\n"
        f'**Control de validación: {"PASS" if oracle["gate"]["passed"] else "FAIL"}.** '
        "Antes de evaluar se fijaron tres criterios para todas las semillas: error L2 ≤ 1e-4, residuo normalizado ≤ 1e-3 "
        "y pérdida oráculo/nula ≤ 0.1. Son criterios prácticos de esta ronda, no garantías teóricas. Solo validación "
        "decide si se continúa; prueba queda como diagnóstico final. El spline puede pasar y aún tener residuo absoluto "
        "apreciable, en especial con tres frecuencias."),
        code("from pathlib import Path\nimport json,sys\nfrom IPython.display import display,HTML,SVG\nROOT=Path.cwd()\nwhile not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent:\n    ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'src'))\nfrom pinn_pca.wave_diagnostic_report import oracle_table,metric_table,pair_table,gradient_table,scaler_table,read_csv\n"
            f"RUN_DIR=ROOT/'{relative}'\nORACLE=json.loads((RUN_DIR/'oracle_summary.json').read_text(encoding='utf-8'))"),
        code("display(HTML(oracle_table(ORACLE)))",{'text/html':oracle_table(oracle)})]
    if result is not None:
        cells += [md("## Cambio único y reproducción\n\n"
            "Se ajustan centro y desviación estándar de cada coeficiente únicamente con snapshots de entrenamiento; "
            "la red predice z y se reconstruye a = centro + escala·z. La pérdida conserva unidades físicas y no se añade "
            "supervisión directa de coeficientes. Se transforma inversamente la última capa para igualar el campo inicial "
            "de PCA original y estandarizada; las capas ocultas y la semilla se comparten. Cambian las coordenadas de los "
            "parámetros de salida, no la predicción física inicial.\n\n"
            "Se repiten base, PCA original y PCA estandarizada en el mismo proceso: mismas semillas, datos, particiones, "
            "k, arquitectura, pesos y 2000 pasos. Los hashes verifican los minibatches originales y las huellas de los "
            "archivos fuente. Se rota el orden de los tres modelos. Compilación se reutiliza por arquitectura.\n\n"
            f'Checkpoints de referencia comparados: {audit["matching_checkpoint_keys"]}/{audit["expected_source_checkpoints"]}; '
            f'diferencia máxima de RMSE frente al piloto: {audit["max_absolute_validation_difference"]:.3g}. '
            f'Diferencia máxima del campo inicial PCA original/estandarizada: {audit["physical_initialization_max_abs_difference"]:.3g}. '
            f'Archivos fuente sin cambios: {audit["source_hashes_match"]}.\n\n'
            "El pipeline incluye PCA, splines, escalador, entrenamiento y validación; excluye oráculos, compilación, "
            "diagnósticos de gradiente, preparación de minibatches y prueba final."),
            code("RESULT=json.loads((RUN_DIR/'summary.json').read_text(encoding='utf-8'))\nSCALERS=read_csv(RUN_DIR/'coefficient_scalers.csv')\ndisplay(HTML(scaler_table(SCALERS)))",{'text/html':scaler_table(scalers)}),
            md("## Resultados de entrenamiento\n\nMedia ± desviación estándar entre diez semillas. El error de solución y las condiciones iniciales deben acompañar al residuo de una PDE homogénea, que también admite la solución nula."),
            code("display(HTML(metric_table(RESULT)))",{'text/html':metric_table(result)}),
            md("## Efecto pareado de estandarizar\n\nEstandarizada − original. Un valor negativo favorece estandarizar. El intervalo bootstrap remuestrea semillas completas; no se cambia k ni se ajustan hiperparámetros con prueba."),
            code("display(HTML(pair_table(RESULT)))",{'text/html':pair_table(result)}),
            md("## Gradientes al inicio y al final\n\nSe registra la norma L2 del gradiente de cada término respecto de todos los parámetros. La tabla resume la norma del gradiente PDE dividida por la suma de las otras cuatro normas, en los minibatches compartidos primero y último. Se muestran mediana y rango intercuartílico. No se modifica el optimizador con estas medidas. Las normas dependen de la parametrización: sirven como diagnóstico dentro de cada modelo, no como cantidades invariantes comparables entre arquitecturas."),
            code("GRADIENTS=read_csv(RUN_DIR/'gradient_diagnostics.csv')\ndisplay(HTML(gradient_table(GRADIENTS)))",{'text/html':gradient_table(gradients)})]
    cells.append(md("## Figuras\n\nLíneas discontinuas del control oráculo: umbrales fijados antes de evaluar. Curvas de validación: media ± DE. Gradientes: mediana y rango intercuartílico. Los campos usan la primera semilla preseleccionada, no la mejor corrida."))
    for name in figures:
        cells.append(code(f"display(SVG(filename=str(RUN_DIR/'{name}.svg')))",{'image/svg+xml':(output/f'{name}.svg').read_text(encoding='utf-8')}))
    if result is not None:
        findings=[]
        for case,models in result['summary'].items():
            findings.append(f'- **{case}**: L2 medio base {models[BASE]["test_relative_l2"]["mean"]:.4f}, '
                f'PCA original {models[RAW]["test_relative_l2"]["mean"]:.4f}, '
                f'PCA estandarizada {models[NORMALIZED]["test_relative_l2"]["mean"]:.4f}.')
        small_effect = all(abs(models[NORMALIZED]['test_relative_l2']['mean']-models[RAW]['test_relative_l2']['mean'])<1e-3
                           and models[NORMALIZED]['test_relative_l2']['mean']>.95 for models in result['summary'].values())
        observed = ("\n\nEn esta ejecución, estandarizar por sí solo no corrige el fallo: la diferencia media de L2 es menor "
                    "que 0.001 y la red estandarizada sigue con error cercano a uno en ambos casos. "
                    "Los mapas y errores por frecuencia muestran predicciones cercanas a cero. "
                    "La compresión conserva los datos y el control oráculo puede alcanzar una pérdida mucho menor "
                    "que la solución nula; la configuración de entrenamiento continúa sin encontrarla."
                    if small_effect else "")
        cells.append(md("## Interpretación de la ronda\n\n"+'\n'.join(findings)+observed+
            "\n\nEl oráculo separa capacidad de representar u de precisión de sus derivadas interpoladas. La ablación "
            "aísla la estandarización de coeficientes con inicialización física equivalente. Si persiste el colapso, "
            "cambiar solo esta escala no corrige la configuración; estos controles no determinan por sí solos si la "
            "causa restante es la pérdida, el optimizador, el presupuesto o la red temporal. Un residuo no nulo del "
            "spline sigue siendo una limitación numérica que debe conservarse en la interpretación. Los resultados "
            "negativos se limitan al protocolo probado y no prueban una imposibilidad general de PCA.\n\n"
            "Reproducir: `python scripts/run_wave_diagnostics.py`. Añadir `--oracle-only` para ejecutar solo controles "
            "físicos o `--report-only` para actualizar figuras/notebook desde artefactos, sin importar JAX. "
            "Se conservan la corrida original, los CSV de oráculos, escaladores y gradientes, las diferencias pareadas "
            "y figuras SVG/PNG en la carpeta de esta ronda."))
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    path=root/'notebooks/wave/02_diagnostic_round.ipynb'
    if result is not None:
        from .benchmark_summary import apply_benchmark_closure
        apply_benchmark_closure(notebook,root,'wave_multiscale',result['configuration']['run_id'],path)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
    print(f'Updated diagnostic notebook: {path}',flush=True)
