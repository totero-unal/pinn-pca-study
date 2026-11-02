"""Matched Sobol/PCA objectives and a frozen confirmation on unseen seeds."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .collocation import _run_collocation_seed,_save_csv
from .collocation_loss_balance import BalanceRecorder
from .collocation_basis_ablation import read_rows
from .notebook_images import embed_notebook_images

def paired_stats(values):
    values=np.asarray(values,dtype=float)
    if not np.isfinite(values).all():raise ValueError("Nonfinite paired results")
    rng=np.random.default_rng(6071)
    boot=np.mean(rng.choice(values,(20000,len(values)),replace=True),axis=1)
    return dict(mean_difference=float(values.mean()),std=float(values.std(ddof=1)),
        wins=int(np.sum(values<0)),n=len(values),bootstrap95=[float(x) for x in np.quantile(boot,[.025,.975])])

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/ic_confirmation"/c["run_id"]
    dev=c["development_seeds"];new=c["confirmation_seeds"]
    if set(dev)&set(new) or len(set(dev))!=10 or len(set(new))!=10:raise ValueError("Two disjoint groups of ten seeds required")
    if c["ic_weight"]!=10:raise ValueError("Confirmation freezes the previously proposed IC weight")
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];manifest=[]
    old=root/"pca_collocation/results/burgers"
    for stage,seeds in (("development",dev),("confirmation",new)):
        for seed in seeds:
            for objective in ("baseline","ic10"):
                for method in ("pca","sobol"):
                    reused=stage=="development" and not (objective=="ic10" and method=="sobol")
                    if reused:
                        if method=="pca" and objective=="ic10":folder=old/"loss_balance/ic_bc_controls_10_seeds_10k_v1/ic10/rebuild"/f"seed_{seed}"
                        elif method=="pca":folder=old/"interval_history/interval20_every5_10_seeds_10k_v1/interval_fixed"/f"seed_{seed}"
                        else:folder=old/"partial_update/pca95_partial_10_seeds_10k_v1/sobol"/f"seed_{seed}"
                    else:
                        folder=out/stage/objective/method/f"seed_{seed}"
                        if not (folder/"components.csv").exists():
                            print(f"RUN {stage} {objective} {method} seed={seed}",flush=True)
                            spec=dict(base="sobol",label="Sobol") if method=="sobol" else dict(
                                base="pca_mean_qr",label="PCA + media 95%",adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                            saved=dict(c,seed=seed,method_specs={method:spec},common_selection_seed=True,skip_seed_plots=True)
                            weights=[1.,1.,10. if objective=="ic10" else 1.,0.]
                            # Preserve the original baseline arithmetic; only IC10 changes continuation loss.
                            if objective=="ic10":saved["loss_weights"]=weights
                            recorder=BalanceRecorder(dict(saved,loss_weights=weights),folder,"rebuild")
                            _run_collocation_seed(saved,root,folder,observer=recorder)
                            print(f"DONE {stage} {objective} {method} seed={seed}",flush=True)
                    if not (folder/"metrics.csv").exists():raise ValueError("Missing reference or incomplete run")
                    rows.extend(dict(r,stage=stage,objective=objective,selector=method) for r in read_rows(folder/"metrics.csv"))
                    traces.extend(dict(r,stage=stage,objective=objective,selector=method) for r in read_rows(folder/"validation_trajectories.csv"))
                    manifest.append(dict(stage=stage,objective=objective,selector=method,seed=seed,path=folder.relative_to(root).as_posix(),reused=reused))
                    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"traces.csv",traces)
    summary=[];paired=[]
    for stage,seeds in (("development",dev),("confirmation",new)):
        for seed in seeds:
            warm=[float(r["validation_rmse"]) for r in traces if r["stage"]==stage and int(r["seed"])==seed and int(r["step"])==100]
            assert len(warm)==4;np.testing.assert_array_equal(warm,[warm[0]]*4)
        for objective in ("baseline","ic10"):
            for method in ("pca","sobol"):
                group=[r for r in rows if r["stage"]==stage and r["objective"]==objective and r["selector"]==method]
                assert len(group)==10
                result=dict(stage=stage,objective=objective,selector=method)
                for metric in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms"):
                    values=np.array([float(r[metric]) for r in group]);assert np.isfinite(values).all()
                    result[metric]=float(values.mean());result[metric+"_std"]=float(values.std(ddof=1))
                result["validation_rmse"]=float(np.mean([float(r["validation_rmse"]) for r in traces if r["stage"]==stage and r["objective"]==objective and r["selector"]==method and int(r["step"])==10100]))
                summary.append(result)
        def get(seed,objective,selector,metric):
            return float(next(r for r in rows if r["stage"]==stage and int(r["seed"])==seed and r["objective"]==objective and r["selector"]==selector)[metric])
        for objective in ("baseline","ic10"):
            for metric in ("test_rmse","independent_residual_rms"):
                paired.append(dict(stage=stage,comparison="PCA minus Sobol",objective=objective,metric=metric,
                    **paired_stats([get(s,objective,"pca",metric)-get(s,objective,"sobol",metric) for s in seeds])))
        for method in ("pca","sobol"):
            paired.append(dict(stage=stage,comparison="IC10 minus baseline",selector=method,metric="test_rmse",
                **paired_stats([get(s,"ic10",method,"test_rmse")-get(s,"baseline",method,"test_rmse") for s in seeds])))
    _save_csv(out/"summary.csv",summary)
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,disjoint_seed_groups=True,warmup_equal=True,
        new_runs=50,reused_runs=30,confirmation_runs=40),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,stage in zip(axes,("development","confirmation")):
        for method in ("pca","sobol"):
            group=[r for r in summary if r["stage"]==stage and r["selector"]==method]
            ax.errorbar([0,1],[r["test_rmse"] for r in group],yerr=[r["test_rmse_std"] for r in group],fmt="o-",capsize=4,label=method)
        ax.set(xticks=[0,1],xticklabels=["IC ×1","IC ×10"],ylabel="RMSE prueba: media ± DE",title=stage);ax.legend()
    fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    text="# IC ×10: control Sobol y confirmación con nuevas semillas\n\nLa comparación cruza dos selectores (PCA + media al 95 % con reconstrucción y Sobol completo) y dos pérdidas (datos+PDE+IC; datos+PDE+10 IC). No hay término BC explícito. PCA usa candidatos fijos, QR directo y 20 snapshots cada cinco pasos del último intervalo de cien; ambos métodos tienen 128 activos, red 3×32, mismo Adam y actualizan puntos cada cien pasos.\n\nCien pasos uniformes comunes por semilla; IC10 se aplica solo durante los diez mil pasos de continuación, sin reiniciar Adam. Los datos supervisados, particiones de evaluación y 32 nodos IC son comunes. El control baseline conserva exactamente la función de pérdida histórica. La diferencia inicial de puntos activos PCA/Sobol es parte del selector, no se obliga un lote común después del warmup.\n\n**Desarrollo:** semillas 2051–2060; se reutilizan tres combinaciones y se ejecuta Sobol IC10. **Confirmación:** semillas 2071–2080, elegidas antes de ejecutar, cuatro combinaciones nuevas por semilla. Se congela el peso 10 y el protocolo; no se ajustan parámetros usando los resultados de confirmación. Los grupos se presentan separados. El éxito requiere comparar PCA y Sobol bajo el mismo objetivo, no comparar PCA IC10 contra Sobol IC1.\n\n"
    for stage in ("development","confirmation"):
        text+="## "+stage+"\n\n| Selector | Peso IC | Validación RMSE | Prueba RMSE ± DE | Residuo RMS | RMSE IC |\n|---|---:|---:|---:|---:|---:|\n"
        for r in summary:
            if r["stage"]==stage:text+=f"| {r['selector']} | {10 if r['objective']=='ic10' else 1} | {r['validation_rmse']:.5f} | {r['test_rmse']:.5f} ± {r['test_rmse_std']:.5f} | {r['independent_residual_rms']:.5f} | {r['initial_condition_rmse']:.5f} |\n"
        for r in paired:
            if r["stage"]==stage and r["metric"]=="test_rmse":
                text+=f"\n{r['comparison']} ({r.get('objective',r.get('selector'))}): diferencia media {r['mean_difference']:+.5f}; valores negativos en {r['wins']}/10 semillas; intervalo bootstrap 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n"
    text+="\n![Comparación](../../results/burgers/ic_confirmation/"+c["run_id"]+"/comparison.png)\n\nLos resultados son de modelos terminales a presupuesto prefijado, sin escoger el mejor checkpoint por prueba. El bootstrap pareado usa 20000 remuestras de las diez diferencias por semilla; con diez semillas su incertidumbre puede ser considerable y no corrige multiplicidad. DE describe dispersión entre semillas. No se afirma aceleración: PCA requiere observar candidatos y el instrumental agrega costo.\n\nmanifest.json identifica la procedencia; metrics.csv contiene resultados por semilla y summary.json los contrastes pareados. Las referencias reutilizadas y nuevas no deben contarse como ejecuciones independientes adicionales. Los diagnósticos de componentes y gradientes están en cada corrida nueva. Los pesos se aplican después del warmup idéntico, verificado por métricas comunes.\n\nReproducir: `python scripts/run_collocation_ic_confirmation.py`. Las corridas completas se reutilizan; configuración en pca_collocation/configs/burgers_ic_confirmation.json. Las figuras se incrustan como PNG."
    notebook=root/"pca_collocation/notebooks/burgers/24_ic_confirmation.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
