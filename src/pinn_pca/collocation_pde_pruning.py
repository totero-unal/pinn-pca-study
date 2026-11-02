"""Rank-sized convergence pruning under the PDE-only hard-condition objective."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .collocation import _run_collocation_seed,_save_csv
from .collocation_hard_conditions import HardRecorder,residual_field
from .collocation_rank_pruning import RankPruningRecorder
from .collocation_partial import replace_converged
from .collocation_basis_ablation import read_rows
from .collocation_ic_confirmation import paired_stats
from .notebook_images import embed_notebook_images

class PDEPruningRecorder(RankPruningRecorder):
    def __init__(self,c,folder,variant,schedule=None):
        super().__init__(c,folder,variant,schedule);self.constraints=[]
    residual_values=staticmethod(residual_field)
    training_step=HardRecorder.training_step
    checkpoint=HardRecorder.checkpoint
    def end_seed(self):
        super().end_seed();_save_csv(self.output/"constraints.csv",self.constraints)

LABELS=dict(pca_partial="PCA: poda de k",sobol_partial="Sobol: poda de k emparejado",
            pca_rebuild="PCA: reconstrucción",sobol_rebuild="Sobol: remuestreo completo")

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/pde_pruning"/c["run_id"]
    if c.get("loss_weights"):raise ValueError("PDE-only objective required")
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];updates=[];manifest=[];checks=0
    for seed in c["seeds"]:
        schedule=None;initial=[];prefix=[]
        for arm in ("pca_partial","sobol_partial"):
            folder=out/arm/f"seed_{seed}"
            if not (folder/"constraints.csv").exists():
                print(f"RUN PDE pruning {arm} seed={seed}",flush=True)
                spec=dict(base="pca_mean_qr",label=LABELS[arm],adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,pde_only=True,method_specs={arm:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,
                    observer=PDEPruningRecorder(c,folder,arm,schedule))
                print(f"DONE PDE pruning {arm} seed={seed}",flush=True)
            pruning=read_rows(folder/"rank_pruning.csv")
            if arm=="pca_partial":schedule={int(r["step"]):int(r["k"]) for r in pruning}
            else:assert schedule=={int(r["step"]):int(r["k"]) for r in pruning}
            updates.extend(dict(seed=seed,method=arm,**r) for r in pruning)
            rows.extend(read_rows(folder/"metrics.csv"))
            trace=read_rows(folder/"validation_trajectories.csv");traces.extend(trace)
            prefix.append([float(r["validation_rmse"]) for r in trace if int(r["step"])<=200])
            initial.append(np.load(folder/"active_history.npz")[f"{arm}_100"])
            a=np.load(folder/"rank_pruning_audit.npz")
            for step in (200,1100,5100,10000):
                updated,removed,_,_=replace_converged(a[f"{step}_before"],a[f"{step}_proposals"],a[f"{step}_old"],a[f"{step}_current"])
                np.testing.assert_array_equal(updated,a[f"{step}_after"])
                np.testing.assert_array_equal(removed,a[f"{step}_removed"]);checks+=1
            constraints=read_rows(folder/"constraints.csv")
            assert len(constraints)==4 and all(float(r["ic_max"])<1e-6 and float(r["bc_max"])==0 for r in constraints)
            manifest.append(dict(method=arm,seed=seed,path=folder.relative_to(root).as_posix(),reused=False))
        for selector in ("pca","sobol"):
            reference=root/"pca_collocation/results/burgers/pde_only/pde_only_hard_10seeds_10k_v1"/selector/f"seed_{seed}"
            rows.extend(dict(r,method=selector+"_rebuild") for r in read_rows(reference/"metrics.csv"))
            trace=read_rows(reference/"validation_trajectories.csv")
            traces.extend(dict(r,method=selector+"_rebuild") for r in trace)
            if selector=="pca":
                initial.append(np.load(reference/"active_history.npz")["pca_100"])
                prefix.append([float(r["validation_rmse"]) for r in trace if int(r["step"])<=200])
            manifest.append(dict(method=selector+"_rebuild",seed=seed,path=reference.relative_to(root).as_posix(),reused=True))
        for a in initial[1:]:np.testing.assert_array_equal(a,initial[0])
        for a in prefix[1:]:np.testing.assert_array_equal(a,prefix[0])
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"pruning.csv",updates)
    summary={};paired=[]
    for arm in LABELS:
        group=[r for r in rows if r["method"]==arm];assert len(group)==10
        summary[arm]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms")}
        if arm.endswith("partial"):
            group=[r for r in updates if r["method"]==arm and r["initialization"]=="False"]
            summary[arm]["pruning"]=dict(accepted=sum(r["accepted"]=="True" for r in group),attempts=len(group),
                k_min=min(int(r["k"]) for r in group),k_max=max(int(r["k"]) for r in group),k_mean=float(np.mean([int(r["k"]) for r in group])))
    for arm,reference in (("pca_partial","pca_rebuild"),("pca_partial","sobol_partial"),("pca_partial","sobol_rebuild")):
        for metric in ("test_rmse","independent_residual_rms"):
            get=lambda seed,method:float(next(r for r in rows if r["method"]==method and int(r["seed"])==seed)[metric])
            paired.append(dict(method=arm,reference=reference,metric=metric,
                **paired_stats([get(s,arm)-get(s,reference) for s in c["seeds"]])))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,pruning_reconstructions=checks,matched_k=True,shared_initial_points_and_interval=True,hard_constraints=True),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,label in LABELS.items():
        group=[r for r in traces if r["method"]==arm];steps=sorted(set(int(r["step"]) for r in group))
        for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
            ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=label)
            ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Método | RMSE prueba ± DE | Residuo RMS |\n|---|---:|---:|\n"
    for arm,label in LABELS.items():
        r=summary[arm];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} |\n"
    text="# Poda parcial de k con objetivo puramente físico\n\nSe mantienen IC/BC duras con u=−sin(pi x)+t(1−x²)N, pérdida solo mean(r²) desde la inicialización, red 3×32, Adam constante, 128 activos, candidatos fijos y veinte snapshots cada cinco pasos del último intervalo de cien. No se usan valores de solución para ajustar pesos ni elegir checkpoints.\n\nPCA + media al 95 % determina el rango total k y QR propone k candidatos disjuntos de los activos. La poda retira los k con mayor log2(|r anterior|/|r actual|), con piso 1e-12; si uno de esos grados es negativo se cancela toda la sustitución. Se realinea el residuo basal tras insertar. Tanto la historia como los grados de convergencia usan el residuo de la solución transformada, no el de N sin transformar.\n\nEl control Sobol parcial usa exactamente el k de la trayectoria PCA en la misma semilla e intervalo y la misma regla de poda; es un control emparejado dependiente del calendario PCA. Los dos brazos comparten lote inicial y primer intervalo con la reconstrucción PCA, antes de la primera poda en paso 200. Las referencias de reconstrucción PCA y remuestreo Sobol se reutilizan del notebook 26. Diez semillas 2071–2080 y diez mil pasos de continuación: extensión exploratoria sobre semillas ya vistas, no confirmación independiente.\n\n"+table+"\n\n"
    for r in paired:
        if r["metric"]=="test_rmse":text+=f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia media RMSE {r['mean_difference']:+.5f}; mejora en {r['wins']}/10 semillas; bootstrap pareado 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n\n"
    text+="![Comparación](../../results/burgers/pde_pruning/"+c["run_id"]+"/comparison.png)\n\n"
    for arm in ("pca_partial","sobol_partial"):
        r=summary[arm]["pruning"];text+=f"{LABELS[arm]}: {r['accepted']}/{r['attempts']} podas aceptadas, k entre {r['k_min']} y {r['k_max']}, media {r['k_mean']:.2f}.\n\n"
    text+="El presupuesto terminal está prefijado. DE es dispersión entre semillas; el bootstrap con diez diferencias es exploratorio y no corrige multiplicidad. El residuo se evalúa sobre puntos independientes; IC/BC se comprueban en una malla adicional. Las auditorías reconstruyen ochenta podas, comprueban calendario k común y comienzo idéntico. residence.csv distingue permanencias completas y sobrevivientes censurados. No se afirma aceleración.\n\nReproducir: `python scripts/run_collocation_pde_pruning.py`. Las corridas completas se reutilizan; manifest.json identifica procedencia. Las figuras están incrustadas como PNG."
    notebook=root/"pca_collocation/notebooks/burgers/27_pde_pruning.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
