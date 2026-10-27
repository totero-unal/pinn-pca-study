"""Saved tables, scientific figures and notebook for the standing-wave pilot."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np

MODELS = ("PINN base", "PCA espacial 99%")
COLORS = ("#526274", "#c8791c")
METRICS = (("test_relative_l2", "Error relativo L2"),
           ("test_pde_residual_rms", "Residuo RMS de onda (AD)"),
           ("pipeline_seconds", "Pipeline (s)"))


def table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
        + "</tr></thead><tbody>" + "".join("<tr>" + "".join(
            f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>")


def metrics_table(result):
    rows = []
    for case, models in result["summary"].items():
        for model in MODELS:
            metrics = models[model]
            rows.append([case, model, *[f'{metrics[key]["mean"]:.4g} ± {metrics[key]["std"]:.3g}'
                                        for key, _ in METRICS]])
    return table(["Caso", "Modelo", *[label + " (media ± DE)" for _, label in METRICS]], rows)


def pca_table(result):
    rows = []
    for case in result["summary"]:
        selected = [r for r in result["runs"] if r["case"] == case and r["model"] == MODELS[1]]
        rows.append([case, str(sorted({r["k"] for r in selected})),
                     f'{100*np.mean([r["retained_variance"] for r in selected]):.6f}%',
                     f'{np.mean([r["pca_oracle_test_relative_l2"] for r in selected]):.3g}',
                     str(sorted({r["parameters"] for r in selected}))])
    return table(["Caso", "k", "Varianza media retenida", "Error L2 proyección test (oráculo)", "Parámetros PCA"], rows)


def paired_table(result):
    rows = []
    for case, metrics in result["paired_PCA_minus_base"].items():
        for key, label in METRICS:
            value = metrics[key]
            rows.append([case, label, f'{value["PCA_minus_base_mean"]:.4g}',
                         '[' + ', '.join(f'{v:.4g}' for v in value["bootstrap_mean_ci_95"]) + ']',
                         f'{value["PCA_better_count"]}/{value["n"]}'])
    return table(["Caso", "Métrica", "Δ media PCA − base", "IC bootstrap 95%", "PCA menor"], rows)


def loss_table(result):
    terms = ["data", "pde", "initial_u", "initial_ut", "boundary"]
    rows = []
    for case in result["summary"]:
        for model in MODELS:
            selected = [r for r in result["runs"] if r["case"] == case and r["model"] == model]
            rows.append([case, model, *[f'{np.mean([r["loss_"+key] for r in selected]):.4g}' for key in terms]])
    return table(["Caso", "Modelo", *["MSE " + key for key in terms]], rows)


def create_wave_report(root, output, result, traces, mode_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root, output = Path(root), Path(output)
    cases = list(result["summary"])
    figure_names = []
    def save(fig, name):
        fig.savefig(output / f"{name}.svg", bbox_inches="tight")
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig); figure_names.append(name)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, axes = plt.subplots(len(cases), 3, figsize=(12, 3.3*len(cases)), squeeze=False, layout="constrained")
        for i, case in enumerate(cases):
            for j, (metric, label) in enumerate(METRICS):
                ax = axes[i,j]
                stats = [result["summary"][case][model][metric] for model in MODELS]
                ax.bar([0,1], [s["mean"] for s in stats], yerr=[s["std"] for s in stats],
                       color=COLORS, capsize=4, width=.6)
                ax.set_xticks([0,1], ["Base", "PCA 99%"])
                ax.set(ylabel=label, title=case, ylim=(0,None)); ax.grid(axis="y", alpha=.2)
        save(fig, "comparison")

        fig, axes = plt.subplots(1,len(cases),figsize=(6*len(cases),4.3),squeeze=False,layout="constrained")
        for ax, case in zip(axes[0], cases):
            for model, color in zip(MODELS, COLORS):
                selected = [r for r in traces if r["case"] == case and r["model"] == model]
                steps = sorted({int(r["step"]) for r in selected})
                vals = [np.asarray([float(r["validation_rmse"]) for r in selected if int(r["step"]) == step]) for step in steps]
                means = np.asarray([v.mean() for v in vals]); stds = np.asarray([v.std(ddof=1) if len(v)>1 else 0 for v in vals])
                ax.plot(steps,means,label=model,color=color)
                ax.fill_between(steps,np.maximum(0,means-stds),means+stds,color=color,alpha=.13)
            ax.set(xlabel="Pasos",ylabel="RMSE validación",title=case); ax.grid(alpha=.2); ax.legend(fontsize=8)
        save(fig,"validation_vs_steps")

        fig, axes = plt.subplots(1,len(cases),figsize=(6*len(cases),4.3),squeeze=False,layout="constrained")
        for ax, case in zip(axes[0],cases):
            ns = sorted({int(r["harmonic"]) for r in mode_rows if r["case"] == case})
            for offset,model,color in zip([-.18,.18],MODELS,COLORS):
                values = [[float(r["coefficient_relative_l2"]) for r in mode_rows
                           if r["case"]==case and r["model"]==model and int(r["harmonic"])==n] for n in ns]
                ax.bar(np.arange(len(ns))+offset,[np.mean(v) for v in values],width=.34,
                       yerr=[np.std(v,ddof=1) if len(v)>1 else 0 for v in values],capsize=3,label=model,color=color)
            ax.set_xticks(range(len(ns)),[str(n) for n in ns]);ax.set(xlabel="Armónico espacial n",ylabel="Error relativo L2 del coeficiente",title=case,ylim=(0,None));ax.grid(axis="y",alpha=.2)
        handles, labels = axes[0,0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=2, fontsize=9)
        save(fig,"frequency_errors")

        for case in cases:
            with np.load(output / f"example_{case}.npz") as data:
                x,t,truth,base,pca,oracle = [data[k] for k in ("x","t","truth","baseline","pca","oracle")]
                seed=int(data["seed"])
            fig,axes=plt.subplots(2,3,figsize=(12,6),layout="constrained")
            extent=[float(t[0]),float(t[-1]),float(x[0]),float(x[-1])]
            vmax=max(abs(truth.min()),abs(truth.max()))
            for ax,field,title in zip(axes[0],[truth,base,pca],["Solución exacta","PINN base","PCA espacial 99%"]):
                im=ax.imshow(field,origin="lower",aspect="auto",extent=extent,cmap="RdBu_r",vmin=-vmax,vmax=vmax)
                ax.set(title=title,xlabel="t",ylabel="x")
            fig.colorbar(im,ax=axes[0].tolist(),label="u",shrink=.8)
            errors=[abs(oracle-truth),abs(base-truth),abs(pca-truth)]
            emax=max(float(v.max()) for v in errors)
            for ax,err,title in zip(axes[1],errors,["Error proyección oráculo\n(diagnóstico, no red entrenada)","Error absoluto base","Error absoluto PCA"]):
                im=ax.imshow(err,origin="lower",aspect="auto",extent=extent,cmap="magma",vmin=0,vmax=emax)
                ax.set(title=title,xlabel="t",ylabel="x")
            fig.colorbar(im,ax=axes[1].tolist(),label="|error|",shrink=.8)
            fig.suptitle(f'{case} · semilla preseleccionada {seed}')
            save(fig,f"fields_{case}")

    def md(text):return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
    def code(text,data=None):return {"cell_type":"code","metadata":{},"execution_count":None,
        "source":text.splitlines(True),"outputs":[{"output_type":"display_data","metadata":{},"data":data}] if data else []}
    relative=output.relative_to(root).as_posix()
    cells=[md(r"""# Onda multiescala: PINN base frente a PCA espacial

Pregunta: ¿PCA espacial al 99% reduce tiempo y mantiene precisión cuando la solución contiene frecuencias separadas?

Se usa $u_{tt}-4u_{xx}=0$ en $x,t\in[0,1]$, con $u(0,t)=u(1,t)=0$ y $u_t(x,0)=0$. La solución de dos frecuencias es

$$u(x,t)=\sin(\pi x)\cos(2\pi t)+0.5\sin(4\pi x)\cos(8\pi t).$$

El segundo caso añade $0.25\sin(9\pi x)\cos(18\pi t)$.

Fuente del problema: Datar et al. (2026), *Fast Training of Accurate Physics-Informed Neural Networks without Gradient Descent*, apéndice C.3, ecuaciones 27a–27h, [arXiv:2405.20836v3](https://arxiv.org/abs/2405.20836v3). El protocolo supervisado es el de este repositorio; no se reproduce el entrenamiento de Frozen-PINN ni se comparan sus tiempos publicados.

## Protocolo fijado antes de entrenar

- Diez semillas y 2000 pasos para cada caso y modelo; mismos lotes, particiones, condiciones iniciales y puntos de colocación dentro de cada pareja.
- PCA centrado se ajusta solo a snapshots de tiempos de entrenamiento. La red reducida predice coeficientes temporales sobre esa base espacial fija, en su escala original y sin estandarización adicional: esta variante difiere del PCA de coordenadas usado en dimensión alta.
- Ambas redes tienen tres capas ocultas tanh, parámetros emparejados dentro de 5% y la misma tasa de aprendizaje. Una misma semilla no implica pesos idénticos entre arquitecturas diferentes.
- La pérdida suma MSE de datos, residuo PDE, desplazamiento inicial, velocidad inicial cero y fronteras. Los cinco pesos se fijan en 1 para ambos modelos. Las muestras supervisadas provienen de los mismos snapshots de entrenamiento.
- JAX calcula las segundas derivadas en ambos modelos. La media y los modos PCA se interpolan con splines cúbicos C2, preparados una vez fuera del entrenamiento y evaluados por JAX.
- Pipeline = ajuste PCA + preparación de splines + entrenamiento + validación periódica. Compilación, preparación de lotes y evaluación final se registran fuera de ese tiempo. Se alterna el orden base/PCA entre semillas y se compila una vez por arquitectura durante el proceso.
- Prueba usa los mismos 20 tiempos reservados por pareja. El residuo se autodiferencia sobre los mismos 512 puntos de prueba. PCA no se ajusta con validación o prueba.

Los outputs se generan desde los artefactos guardados. No es necesario entrenar para abrir este notebook."""),
        code("from pathlib import Path\nimport json, sys\nfrom IPython.display import display, HTML, SVG\nROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').exists() and ROOT != ROOT.parent:\n    ROOT = ROOT.parent\nsys.path.insert(0, str(ROOT / 'src'))\nfrom pinn_pca.wave_report import metrics_table, pca_table, paired_table, loss_table\n"
             f"RUN_DIR = ROOT / '{relative}'\nRESULT = json.loads((RUN_DIR / 'summary.json').read_text(encoding='utf-8'))"),
        md("## Métricas finales\n\nMedia ± desviación estándar entre semillas. Un residuo bajo debe interpretarse junto con el error de solución y las condiciones iniciales."),
        code("display(HTML(metrics_table(RESULT)))",{"text/html":metrics_table(result)}),
        md("## Capacidad de representación\n\nLa proyección oráculo usa los coeficientes calculados desde las etiquetas de prueba exclusivamente como diagnóstico. No es una predicción entrenada ni interviene en la selección de k. Un error de proyección pequeño junto a un error grande de red indica que retener los modos no basta para aprender sus coeficientes temporales. Esta cota mide valores de u en la malla de prueba; no garantiza igual precisión de las derivadas de los splines."),
        code("display(HTML(pca_table(RESULT)))",{"text/html":pca_table(result)}),
        md("## Diferencias pareadas\n\nPCA − base; menor es mejor para las tres métricas. Los intervalos bootstrap remuestrean diferencias dentro de las mismas semillas. Este piloto es exploratorio, sin selección posterior de k con prueba."),
        code("display(HTML(paired_table(RESULT)))",{"text/html":paired_table(result)}),
        md("## Componentes de la pérdida\n\nMedias en el último minibatch después de la última actualización. Una solución pequeña puede satisfacer aproximadamente la ecuación homogénea mientras falla en desplazamiento inicial o datos. Se registra por separado la condición de velocidad inicial."),
        code("display(HTML(loss_table(RESULT)))",{"text/html":loss_table(result)})]
    cells.append(md("## Figuras\n\nLas curvas resumen validación entre semillas. El error por armónico compara los coeficientes obtenidos por proyección de las predicciones en los modos seno con sus valores analíticos en prueba. Las imágenes de campo usan la primera semilla fijada antes de entrenar, no la mejor corrida; incluyen todos los tiempos solo para visualizar el resultado final."))
    for name in figure_names:
        cells.append(code(f"display(SVG(filename=str(RUN_DIR / '{name}.svg')))",
                          {"image/svg+xml":(output/f"{name}.svg").read_text(encoding="utf-8")}))
    findings=[]
    for case,paired in result["paired_PCA_minus_base"].items():
        fragments=[f'{label}: Δ media {paired[key]["PCA_minus_base_mean"]:.4g}, PCA menor en '
                   f'{paired[key]["PCA_better_count"]}/{paired[key]["n"]} semillas' for key,label in METRICS]
        findings.append(f'- **{case}**: '+ '; '.join(fragments)+'.')
    cells += [md("## Lectura del piloto\n\n"+'\n'.join(findings)+
        "\n\nLa evidencia queda limitada a Adam, estos pesos de pérdida y 2000 pasos. Un fallo de este piloto no establece "
        "que la arquitectura sea incapaz de resolver la ecuación; una mejora de tiempo con mayor error tampoco demuestra "
        "eficiencia a calidad equivalente. Los errores por frecuencia y la proyección oráculo permiten decidir qué "
        "dificultad conviene estudiar después. No se reajustan componentes ni pérdidas con el conjunto de prueba.\n\n"
        "Regenerar entrenamiento: `python scripts/run_wave_multiscale.py --config configs/wave_multiscale_pilot.json`. "
        "Regenerar solo figuras y notebook: añadir `--report-only`; esta opción no importa JAX. "
        "CSV, configuración, resumen y figuras SVG/PNG se conservan en la carpeta de resultados.")]
    notebook={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
               "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    path=root/"notebooks/wave/01_multiscale_wave.ipynb"
    path.parent.mkdir(parents=True,exist_ok=True)
    from .benchmark_summary import apply_benchmark_closure
    apply_benchmark_closure(notebook,root,"wave_multiscale",result["configuration"]["run_id"],path)
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
