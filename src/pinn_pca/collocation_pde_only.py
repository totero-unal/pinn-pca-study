"""Remove supervised labels while retaining hard IC/BC and the PCA selector."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from .collocation import _run_collocation_seed,_save_csv
from .collocation_hard_conditions import HardRecorder
from .collocation_rank_study import adaptive_basis
from .collocation_basis_ablation import read_rows
from .collocation_ic_confirmation import paired_stats
from .notebook_images import embed_notebook_images

def run(c,root):
    if c.get("loss_weights"):raise ValueError("PDE-only control cannot override its objective")
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/pde_only"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use a new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];manifest=[];qr_checks=0
    for seed in c["seeds"]:
        for selector in ("pca","sobol"):
            folder=out/selector/f"seed_{seed}"
            if not (folder/"constraints.csv").exists():
                print(f"RUN PDE only {selector} seed={seed}",flush=True)
                spec=dict(base="sobol",label="Sobol PDE only") if selector=="sobol" else dict(
                    base="pca_mean_qr",label="PCA PDE only",adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,pde_only=True,method_specs={selector:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,observer=HardRecorder(c,folder))
                print(f"DONE PDE only {selector} seed={seed}",flush=True)
            reference=root/"pca_collocation/results/burgers/hard_conditions/hard_ic_bc_data_10seeds_10k_v1"/selector/f"seed_{seed}"
            for objective,source in (("pde_only",folder),("data_pde",reference)):
                rows.extend(dict(r,objective=objective,selector=selector) for r in read_rows(source/"metrics.csv"))
                traces.extend(dict(r,objective=objective,selector=selector) for r in read_rows(source/"validation_trajectories.csv"))
                manifest.append(dict(objective=objective,selector=selector,seed=seed,path=source.relative_to(root).as_posix(),reused=objective=="data_pde"))
            if selector=="pca":
                h=np.load(folder/"history_audit.npz");active=np.load(folder/"active_history.npz")
                for step in (100,200,1100,5100,10000):
                    basis,_=adaptive_basis(h[f"{step}_history"],"pca_mean_qr",.95)
                    _,_,pivots=qr(basis.T,pivoting=True,mode="economic")
                    np.testing.assert_array_equal(active[f"pca_{step}"][:basis.shape[1]],h[f"{step}_pool"][pivots[:basis.shape[1]]]);qr_checks+=1
            constraints=read_rows(folder/"constraints.csv")
            assert len(constraints)==4 and all(float(r["ic_max"])<1e-6 and float(r["bc_max"])==0 for r in constraints)
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"traces.csv",traces)
    summary=[];paired=[]
    for objective in ("data_pde","pde_only"):
        for selector in ("pca","sobol"):
            group=[r for r in rows if r["objective"]==objective and r["selector"]==selector];assert len(group)==10
            item=dict(objective=objective,selector=selector)
            for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms"):
                values=np.array([float(r[key]) for r in group]);assert np.isfinite(values).all()
                item[key]=float(values.mean());item[key+"_std"]=float(values.std(ddof=1))
            summary.append(item)
        get=lambda seed,selector,key:float(next(r for r in rows if r["objective"]==objective and r["selector"]==selector and int(r["seed"])==seed)[key])
        for metric in ("test_rmse","independent_residual_rms"):
            paired.append(dict(comparison="PCA minus Sobol",objective=objective,metric=metric,
                **paired_stats([get(s,"pca",metric)-get(s,"sobol",metric) for s in c["seeds"]])))
    for selector in ("pca","sobol"):
        get=lambda seed,objective:float(next(r for r in rows if r["objective"]==objective and r["selector"]==selector and int(r["seed"])==seed)["test_rmse"])
        paired.append(dict(comparison="PDE only minus data+PDE",selector=selector,metric="test_rmse",
            **paired_stats([get(s,"pde_only")-get(s,"data_pde") for s in c["seeds"]])))
    for seed in c["seeds"]:
        warm=[float(r["validation_rmse"]) for r in traces if r["objective"]=="pde_only" and int(r["seed"])==seed and int(r["step"])==100]
        np.testing.assert_array_equal(warm,[warm[0]]*2)
    _save_csv(out/"summary.csv",summary)
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,new_runs=20,reused_runs=20,shared_pde_warmup=True,qr_reconstructions=qr_checks,constraint_checkpoints=80),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for objective in ("data_pde","pde_only"):
        for selector in ("pca","sobol"):
            group=[r for r in traces if r["objective"]==objective and r["selector"]==selector];steps=sorted(set(int(r["step"]) for r in group))
            for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
                ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=f"{selector} {objective}")
                ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Objetivo | Selector | RMSE prueba ± DE | Residuo RMS | RMSE IC | RMS BC |\n|---|---|---:|---:|---:|---:|\n"
    for r in summary:table+=f"| {r['objective']} | {r['selector']} | {r['test_rmse']:.5f} ± {r['test_rmse_std']:.5f} | {r['independent_residual_rms']:.5f} | {r['initial_condition_rmse']:.2e} | {r['boundary_rms']:.2e} |\n"
    text="# PCA y Sobol: entrenamiento únicamente con residuo físico\n\nSe conserva la transformación u=−sin(pi x)+t(1−x²)N, con IC/BC impuestas por construcción, arquitectura 3×32, Adam constante y 128 activos. PCA usa media + componentes al 95 %, QR directo y complemento Sobol, candidatos fijos y veinte snapshots cada cinco pasos del último intervalo de cien. No hay poda ni ampliación log-det.\n\nLa única pérdida del brazo nuevo es mean(r²). Solo se leen las coordenadas físicas del batch: los campos supervisados e IC se ignoran por completo, verificado mediante sustitución por valores arbitrarios/NaN y comparación de pérdida y gradientes. El generador histórico conserva esos campos para compatibilidad, pero no entran en el objetivo. La solución de referencia se usa para métricas de evaluación; la condición inicial es la función analítica impuesta por transformación.\n\nVeinte corridas nuevas, semillas 2071–2080, cien pasos uniformes iniciales y diez mil de continuación. PDE-only se aplica desde la inicialización, incluido warmup; dentro de cada objetivo PCA/Sobol comparten el warmup y las muestras físicas. Entre objetivos los estados tras cien pasos difieren. Los veinte controles datos+PDE se reutilizan del notebook 25. Es una extensión exploratoria en semillas ya utilizadas, no una nueva confirmación independiente.\n\nNo se detiene el entrenamiento, elige pesos ni selecciona checkpoints usando error de solución. Las métricas de validación y prueba se registran a presupuesto prefijado. El presupuesto es menor que el del artículo y no se afirma reproducción de sus cifras.\n\n"+table+"\n\n"
    for r in paired:
        if r["metric"]=="test_rmse":text+=f"{r['comparison']} ({r.get('objective',r.get('selector'))}): diferencia media {r['mean_difference']:+.5f}, mejora en {r['wins']}/10 semillas; bootstrap pareado 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n\n"
    text+="![Comparación](../../results/burgers/pde_only/"+c["run_id"]+"/comparison.png)\n\nDE representa dispersión entre semillas. Los intervalos bootstrap con diez diferencias son exploratorios y no corrigen multiplicidad. El residuo se evalúa en puntos independientes de la selección; IC/BC analíticas se comprueban en una malla adicional. Las historias guardadas contienen el residuo transformado con signo. Los tiempos incluyen instrumentación y no prueban aceleración.\n\nReproducir: `python scripts/run_collocation_pde_only.py`. Configuración en pca_collocation/configs/burgers_pde_only.json; manifest.json registra procedencia. Las figuras están incrustadas como PNG. La poda de k bajo este objetivo sería una prueba posterior, manteniendo reconstrucción como control."
    notebook=root/"pca_collocation/notebooks/burgers/26_pde_only.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
