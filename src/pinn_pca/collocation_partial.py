"""Controlled partial replacement with convergence pruning and PCA proposals."""
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from .collocation import _run_collocation_seed, _residual_field, _qmc_points, _save_csv
from .collocation_basis_ablation import BasisRecorder, read_rows, select_from_basis
from .collocation_rank_study import adaptive_basis
from .notebook_images import embed_notebook_images

LABELS = dict(pca_full="PCA: reconstrucción", pca_partial="PCA: poda parcial",
              sobol_partial="Sobol: poda parcial", sobol="Sobol: remuestreo")


def replace_converged(active, proposals, old, current, floor=1e-12):
    """All-or-nothing safeguard; old/current refer to the SAME coordinates."""
    if not np.isfinite(old).all() or not np.isfinite(current).all():
        raise ValueError("Nonfinite residuals")
    if len(set(map(tuple,proposals))) != len(proposals) or set(map(tuple,active)) & set(map(tuple,proposals)):
        raise ValueError("Proposals must be unique and disjoint from active points")
    if not 0 < len(proposals) <= len(active):
        raise ValueError("Invalid replacement count")
    degree=np.log2(np.maximum(np.abs(old),floor)/np.maximum(np.abs(current),floor))
    removed=np.argsort(-degree,kind="stable")[:len(proposals)]
    accepted=bool(np.all(degree[removed]>=0))
    if not accepted:
        return active.copy(),np.array([],dtype=int),degree,False
    keep=np.ones(len(active),dtype=bool);keep[removed]=False
    return np.concatenate((active[keep],proposals)),removed,degree,True


class Recorder(BasisRecorder):
    def __init__(self,c,folder,variant):
        super().__init__(c,folder)
        self.variant=variant;self.previous=None;self.baseline=None;self.updates=[];self.audit={}

    def adjust_active(self,method,step,params,active,ids,modes,pool,history,seed,bounds):
        if self.variant not in ("pca_partial","sobol_partial"):
            return active,ids,modes
        if self.previous is None:
            # Both partial methods start with the exact PCA reference batch.
            self.previous=active.copy();self.baseline=np.abs(_residual_field(params,active))
            return active,ids,modes
        before=self.previous
        if self.variant=="pca_partial":
            basis,_=adaptive_basis(history,"pca_mean_qr",.95)
            occupied=set(map(tuple,before))
            eligible=np.array([tuple(p) not in occupied for p in pool])
            local,_=select_from_basis(history[eligible],basis[eligible],32)
            ids=np.flatnonzero(eligible)[local];proposals=pool[ids]
            modes=basis.shape[1]
        else:
            proposals=_qmc_points(32,seed+104729,*bounds)
            ids=np.array([],dtype=int);modes=0
        current=np.abs(_residual_field(params,before))
        after,removed,degree,accepted=replace_converged(before,proposals,self.baseline,current)
        # Recompute at the updated coordinates, so inserted points get their own baseline.
        baseline=np.abs(_residual_field(params,after))
        assert len(after)==128 and len(set(map(tuple,after)))==128
        assert len(set(map(tuple,before))-set(map(tuple,after)))==len(removed)
        self.updates.append(dict(step=step,accepted=accepted,replaced=len(removed),
                                 min_top_degree=float(np.min(np.sort(degree)[-32:]))))
        if step in (200,1100,5100,10000):
            for key,value in dict(before=before,proposals=proposals,after=after,old=self.baseline,
                                  current=current,removed=removed,degree=degree,next_baseline=baseline).items():
                self.audit[f"{step}_{key}"]=value.copy()
        self.previous=after.copy();self.baseline=baseline
        return after,ids,modes

    def selection(self,method,step,active,ids,modes):
        assert len(active)==128 and len(set(map(tuple,active)))==128
        self.active[f"{method}_{step}"]=active.copy()

    def end_seed(self):
        _save_csv(self.output/"budget_metrics.csv",self.metrics)
        if self.updates: _save_csv(self.output/"pruning.csv",self.updates)
        np.savez_compressed(self.output/"active_history.npz",**self.active)
        if self.audit: np.savez_compressed(self.output/"pruning_audit.npz",**self.audit)
        (self.output/"audit.json").write_text(json.dumps(dict(passed=True,selections=len(self.active),
             partial_updates=len(self.updates),accepted=sum(r["accepted"] for r in self.updates))),encoding="utf-8")


def run(config,root,report_only=False):
    root=Path(root).resolve();output=root/"pca_collocation/results/burgers/partial_update"/config["run_id"]
    output.mkdir(parents=True,exist_ok=True)
    path=output/"config.json"
    if path.exists() and json.loads(path.read_text())!=config: raise ValueError("Use new run_id")
    path.write_text(json.dumps(config,indent=2),encoding="utf-8")
    for seed in config["seeds"]:
        for variant in LABELS:
            folder=output/variant/f"seed_{seed}"
            if not report_only and not (folder/"audit.json").exists():
                print(f"RUN {variant} seed={seed}",flush=True)
                spec=dict(base="sobol",label=LABELS[variant]) if variant=="sobol" else dict(
                    base="pca_mean_qr",label=LABELS[variant],adaptive_basis_threshold=.95,oversample_count=32)
                saved=dict(config,seed=seed,method_specs={variant:spec},common_selection_seed=True,skip_seed_plots=True)
                _run_collocation_seed(saved,root,folder,observer=Recorder(config,folder,variant))
                print(f"DONE {variant} seed={seed}",flush=True)
            if not (folder/"audit.json").exists(): raise ValueError("Incomplete experiment")
    report(config,root,output)


def report(c,root,output):
    rows=[];traces=[];pruning=[]
    for variant in LABELS:
        for seed in c["seeds"]:
            folder=output/variant/f"seed_{seed}"
            row=read_rows(folder/"metrics.csv")[0];rows.append(row)
            traces.extend(read_rows(folder/"validation_trajectories.csv"))
            if (folder/"pruning.csv").exists():
                pruning.extend(dict(seed=seed,method=variant,**r) for r in read_rows(folder/"pruning.csv"))
            if (folder/"pruning_audit.npz").exists():
                a=np.load(folder/"pruning_audit.npz")
                for step in (200,1100,5100,10000):
                    updated,removed,_,_=replace_converged(a[f"{step}_before"],a[f"{step}_proposals"],a[f"{step}_old"],a[f"{step}_current"])
                    np.testing.assert_array_equal(updated,a[f"{step}_after"])
                    np.testing.assert_array_equal(removed,a[f"{step}_removed"])
    _save_csv(output/"metrics.csv",rows);_save_csv(output/"pruning.csv",pruning)
    replay=[]
    prior=root/"pca_collocation/results/burgers/rank_study/two_stage_10_seeds_10k_v1"
    for seed in c["seeds"]:
        initial=[];prefix=[]
        for variant in ("pca_full","pca_partial","sobol_partial"):
            folder=output/variant/f"seed_{seed}"
            initial.append(np.load(folder/"active_history.npz")[f"{variant}_100"])
            prefix.append([float(r["validation_rmse"]) for r in traces if r["method"]==variant and int(r["seed"])==seed and int(r["step"])<=200])
        for a in initial[1:]:np.testing.assert_array_equal(a,initial[0])
        for a in prefix[1:]:np.testing.assert_array_equal(a,prefix[0])
        old=prior/"adaptive_0p95_pca_mean_qr"/f"seed_{seed}"/"metrics.csv"
        if old.exists():
            a=float(read_rows(old)[0]["test_rmse"])
            b=float(next(r for r in rows if r["method"]=="pca_full" and int(r["seed"])==seed)["test_rmse"])
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
            replay.append(dict(seed=seed,pca_reference_difference=b-a))
    (output/"audit_checks.json").write_text(json.dumps(dict(passed=True,shared_partial_initialization=True,
        reconstructed_pruning_updates=8*len(c["seeds"]),reference_replay=replay),indent=2),encoding="utf-8")
    summary={}
    for v in LABELS:
        selected=[r for r in rows if r["method"]==v]
        summary[v]={key:dict(mean=float(np.mean([float(r[key]) for r in selected])),
            std=float(np.std([float(r[key]) for r in selected],ddof=1))) for key in
            ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms")}
    paired=[]
    for method,reference in (("pca_partial","pca_full"),("pca_partial","sobol_partial"),("sobol_partial","sobol")):
        deltas=[]
        for seed in c["seeds"]:
            get=lambda v:float(next(r for r in rows if r["method"]==v and int(r["seed"])==seed)["test_rmse"])
            deltas.append(get(method)-get(reference))
        paired.append(dict(method=method,reference=reference,mean_difference=float(np.mean(deltas)),wins=int(np.sum(np.array(deltas)<0))))
    (output/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2),encoding="utf-8")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for v,label in LABELS.items():
        selected=[r for r in traces if r["method"]==v];steps=sorted(set(int(r["step"]) for r in selected))
        for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
            ax.plot(steps,[np.mean([float(r[key]) for r in selected if int(r["step"])==s]) for s in steps],label=label)
            ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend();fig.tight_layout();fig.savefig(output/"comparison.png",dpi=150);plt.close(fig)
    cells=[]
    def md(s):cells.append(dict(cell_type="markdown",metadata={},source=s))
    md("# PCA: reconstrucción frente a poda parcial\n\nDiez semillas, 100 pasos uniformes comunes y 10000 de continuación. Se conserva la pérdida histórica datos+PDE+IC, red 3×32, candidatos fijos (4096), ventana de 20 snapshots y selección cada 100 pasos. PCA + media al 95%, QR + log-det y 32 propuestas. El experimento cambia la actualización del lote, manteniendo 128 puntos activos.")
    md("## Protocolo\n\nLa reconstrucción PCA renueva 32 guiados + 96 Sobol. Ambas variantes parciales comparten el lote PCA de la primera selección y entrenan 100 pasos antes de la primera poda. Después proponen 32 puntos disjuntos del lote activo, mediante PCA o Sobol. Para PCA se ajusta la base sobre la historia completa y se restringe el diseño QR/log-det a candidatos elegibles: no se vuelve a ajustar PCA tras excluir activos.\n\nSe retiran los 32 puntos con mayor log2(|r anterior|/|r actual|), con piso 1e-12. Si alguno de esos 32 grados es negativo, se cancela la sustitución completa. Si se acepta, sobreviven 96 puntos. Se recalcula el residuo basal sobre las coordenadas actualizadas, incluidos los insertados. Los controles Sobol parcial y completo tienen historias iniciales de puntos distintas; la comparación directa de propuestas PCA/Sobol parciales sí comparte el lote inicial.\n\nEsta prueba conserva la ampliación a 32 puntos: todavía no estudia seleccionar directamente k pivotes ni renovar candidatos. La exclusión de candidatos activos es necesaria para garantizar 32 puntos realmente nuevos. Se registran las sustituciones canceladas. No se afirma que esto sea una reproducción completa de QR-DEIM.")
    table="| Método | RMSE prueba ± DE | Residuo RMS | RMSE IC |\n|---|---:|---:|---:|\n"
    for v,label in LABELS.items():
        r=summary[v];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} | {r['initial_condition_rmse']['mean']:.5f} |\n"
    md("## Resultados terminales\n\n"+table+"\nEl modelo se evalúa al presupuesto prefijado; no se elige por error de prueba. DE representa dispersión entre semillas.")
    md("![Comparación]("+os.path.relpath(output/"comparison.png",root/"pca_collocation/notebooks/burgers").replace(os.sep,"/")+")")
    md("## Comparaciones pareadas\n\n"+"\n\n".join(f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia media RMSE {r['mean_difference']:+.5f}, mejora en {r['wins']}/10 semillas." for r in paired))
    for v in ("pca_partial","sobol_partial"):
        group=[r for r in pruning if r["method"]==v]
        md(f"{LABELS[v]}: {sum(r['accepted']=='True' for r in group)}/{len(group)} sustituciones aceptadas. Se conservan cuatro auditorías por semilla para reconstruir la poda.")
    md("## Reproducción\n\n`python scripts/run_collocation_partial.py`; añadir `--report-only` para regenerar el reporte. Configuración: pca_collocation/configs/burgers_partial_update.json. Artefactos por método y semilla: métricas, trazas, puntos activos y auditoría de poda. Los tiempos incluyen compilación y diagnóstico. El siguiente control será comparar 32 propuestas con selección directa por rango, si estos resultados justifican continuar.")
    notebook=root/"pca_collocation/notebooks/burgers/17_partial_update.ipynb"
    notebook.write_text(json.dumps(dict(cells=cells,metadata={},nbformat=4,nbformat_minor=5),indent=2,ensure_ascii=False),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
