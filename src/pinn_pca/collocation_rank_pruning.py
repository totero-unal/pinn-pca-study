"""Rank-sized partial pruning, with matched proposal counts for Sobol control."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from .collocation import _run_collocation_seed,_residual_field,_qmc_points,_save_csv
from .collocation_intervals import IntervalRecorder
from .collocation_partial import replace_converged
from .collocation_rank_study import adaptive_basis
from .collocation_basis_ablation import read_rows
from .notebook_images import embed_notebook_images

LABELS=dict(pca_partial="PCA: poda de k",sobol_partial="Sobol: poda de k emparejado",reference="PCA: reconstrucción")

class RankPruningRecorder(IntervalRecorder):
    def __init__(self,c,folder,variant,schedule=None,refresh=False):
        super().__init__(c,folder,refresh)
        self.arm=variant;self.schedule=schedule;self.previous=None;self.baseline=None
        self.pruning_rows=[];self.pruning_saved={};self.residence={};self.exits=[]
    def residual_values(self,params,points):
        return _residual_field(params,points)
    def adjust_active(self,method,step,params,active,ids,modes,pool,history,seed,bounds):
        if self.previous is None:
            self.previous=active.copy();self.baseline=np.abs(self.residual_values(params,active))
            self.residence={tuple(p):step for p in active}
            self.pruning_rows.append(dict(step=step,k=len(ids),accepted=False,replaced=0,initialization=True,min_top_degree=0.))
            return active,ids,modes
        before=self.previous
        if self.arm=="pca_partial":
            basis,_=adaptive_basis(history,"pca_mean_qr",.95)
            k=basis.shape[1]
            occupied=set(map(tuple,before));eligible=np.array([tuple(p) not in occupied for p in pool])
            _,_,pivots=qr(basis[eligible].T,pivoting=True,mode="economic")
            ids=np.flatnonzero(eligible)[pivots[:k]];proposals=pool[ids]
        else:
            k=int(self.schedule[step])
            occupied=set(map(tuple,before))
            proposals=_qmc_points(128,seed+104729,*bounds)
            proposals=np.array([p for p in proposals if tuple(p) not in occupied])[:k]
            ids=np.array([],dtype=int)
        if len(proposals)!=k:raise ValueError("Insufficient unique proposals")
        current=np.abs(self.residual_values(params,before))
        after,removed,degree,accepted=replace_converged(before,proposals,self.baseline,current)
        assert len(after)==len(set(map(tuple,after)))==128
        assert len(set(map(tuple,before))-set(map(tuple,after)))==(k if accepted else 0)
        self.pruning_rows.append(dict(step=step,k=k,accepted=accepted,replaced=len(removed),initialization=False,
            min_top_degree=float(np.min(np.sort(degree)[-k:]))))
        for index in removed:
            key=tuple(before[index]);self.exits.append(dict(exit_step=step,residence_steps=step-self.residence.pop(key)))
        if accepted:
            for point in proposals:self.residence[tuple(point)]=step
        baseline=np.abs(self.residual_values(params,after))
        if step in (200,1100,5100,10000):
            for key,value in dict(before=before,after=after,proposals=proposals,old=self.baseline,current=current,
                                  next_baseline=baseline,removed=removed,degree=degree).items():
                self.pruning_saved[f"{step}_{key}"]=value.copy()
        self.previous=after.copy();self.baseline=baseline
        return after,ids,k
    def end_seed(self):
        super().end_seed()
        _save_csv(self.output/"rank_pruning.csv",self.pruning_rows)
        # Distinguish completed residence durations from right-censored survivors.
        residence=[dict(**r,censored=False) for r in self.exits]
        end=self.config["warmup_steps"]+self.config["train_steps"]
        residence.extend(dict(exit_step=end,residence_steps=end-start,censored=True) for start in self.residence.values())
        _save_csv(self.output/"residence.csv",residence)
        np.savez_compressed(self.output/"rank_pruning_audit.npz",**self.pruning_saved)

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/rank_pruning"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];updates=[];audit_count=0
    for seed in c["seeds"]:
        schedule=None
        for arm in ("pca_partial","sobol_partial"):
            folder=out/arm/f"seed_{seed}"
            if not (folder/"rank_pruning_audit.npz").exists():
                print(f"RUN {arm} seed={seed}",flush=True)
                spec=dict(base="pca_mean_qr",label=LABELS[arm],adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,method_specs={arm:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,
                    observer=RankPruningRecorder(c,folder,arm,schedule))
                print(f"DONE {arm} seed={seed}",flush=True)
            pruning=read_rows(folder/"rank_pruning.csv")
            if arm=="pca_partial":schedule={int(r["step"]):int(r["k"]) for r in pruning}
            else:
                assert schedule=={int(r["step"]):int(r["k"]) for r in pruning}
            updates.extend(dict(seed=seed,method=arm,**r) for r in pruning)
            rows.extend(read_rows(folder/"metrics.csv"));traces.extend(read_rows(folder/"validation_trajectories.csv"))
            a=np.load(folder/"rank_pruning_audit.npz")
            for step in (200,1100,5100,10000):
                updated,removed,_,_=replace_converged(a[f"{step}_before"],a[f"{step}_proposals"],a[f"{step}_old"],a[f"{step}_current"])
                np.testing.assert_array_equal(updated,a[f"{step}_after"])
                np.testing.assert_array_equal(removed,a[f"{step}_removed"]);audit_count+=1
        reference=root/"pca_collocation/results/burgers/interval_history/interval20_every5_10_seeds_10k_v1/interval_fixed"/f"seed_{seed}"
        rows.extend(dict(r,method="reference") for r in read_rows(reference/"metrics.csv"))
        traces.extend(dict(r,method="reference") for r in read_rows(reference/"validation_trajectories.csv"))
        # Identical initial points and prefix through the first trained interval.
        first=[]
        for arm in ("pca_partial","sobol_partial"):
            first.append(np.load(out/arm/f"seed_{seed}/active_history.npz")[f"{arm}_100"])
        first.append(np.load(reference/"active_history.npz")["interval_fixed_100"])
        for points in first[1:]:np.testing.assert_array_equal(points,first[0])
        prefixes=[[float(r["validation_rmse"]) for r in traces if r["method"]==arm and int(r["seed"])==seed and int(r["step"])<=200] for arm in LABELS]
        for prefix in prefixes[1:]:np.testing.assert_array_equal(prefix,prefixes[0])
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"pruning.csv",updates)
    summary={}
    for arm in LABELS:
        group=[r for r in rows if r["method"]==arm]
        assert len(group)==10
        summary[arm]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms")}
        if arm!="reference":
            group=[r for r in updates if r["method"]==arm and r["initialization"]=="False"]
            summary[arm]["pruning"]=dict(accepted=sum(r["accepted"]=="True" for r in group),attempts=len(group),
                k_min=min(int(r["k"]) for r in group),k_max=max(int(r["k"]) for r in group),k_mean=float(np.mean([int(r["k"]) for r in group])))
    paired=[]
    for arm,reference in (("pca_partial","reference"),("pca_partial","sobol_partial")):
        d=[]
        for seed in c["seeds"]:
            get=lambda method:float(next(r for r in rows if r["method"]==method and int(r["seed"])==seed)["test_rmse"])
            d.append(get(arm)-get(reference))
        paired.append(dict(method=arm,reference=reference,difference=float(np.mean(d)),wins=int(np.sum(np.array(d)<0))))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,pruning_reconstructions=audit_count,matched_k_schedules=True,common_initial_state=True),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for arm,label in LABELS.items():
        group=[r for r in traces if r["method"]==arm];steps=sorted(set(int(r["step"]) for r in group))
        for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
            ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=label)
            ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Método | RMSE prueba ± DE | Residuo RMS | RMSE IC |\n|---|---:|---:|---:|\n"
    for arm,label in LABELS.items():
        r=summary[arm];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} | {r['initial_condition_rmse']['mean']:.5f} |\n"
    text="# PCA: sustitución parcial de k puntos por QR directo\n\nDiez semillas, 100 pasos uniformes y 10000 de continuación. Se mantiene PCA centrada + media al 95%, candidatos fijos, 20 snapshots cada 5 pasos del último intervalo de 100 y 128 activos. Red, Adam y pérdida histórica datos+PDE+IC coinciden con el notebook 19.\n\n**Referencia:** reconstruir con k pivotes QR y 128−k Sobol, reutilizada del notebook 19. **PCA parcial:** conservar 128−k activos y proponer k puntos por QR sobre la base restringida a candidatos que no están activos. PCA se ajusta sobre la historia completa. **Sobol parcial:** misma regla de poda y el mismo k de la trayectoria PCA parcial en cada semilla e intervalo; no usa un rango propio para decidir la cantidad. Este control está emparejado y depende de la corrida PCA, no es un método autónomo de rango.\n\nLos tres brazos comparten la primera selección y el primer intervalo. La primera poda ocurre en paso 200. Se retiran los k puntos con mayor log2(|r anterior|/|r actual|), con piso 1e-12. Si alguno tiene grado negativo se cancela toda la sustitución. Se recompone la referencia de residuo sobre las coordenadas actualizadas. Los candidatos propuestos son disjuntos de los activos; esto requiere restringir QR a los elegibles. Después de la primera actualización las historias divergen con cada entrenamiento. No se renuevan candidatos ni se amplía con log-det.\n\n"+table+"\n\n"+"\n\n".join(f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia media RMSE {r['difference']:+.5f}; mejora en {r['wins']}/10 semillas." for r in paired)+"\n\n![Comparación](../../results/burgers/rank_pruning/"+c["run_id"]+"/comparison.png)\n\n"
    for arm in ("pca_partial","sobol_partial"):
        r=summary[arm]["pruning"];text+=f"{LABELS[arm]}: {r['accepted']}/{r['attempts']} sustituciones aceptadas. k entre {r['k_min']} y {r['k_max']}, media {r['k_mean']:.2f}.\n\n"
    text+="Se evalúa el modelo terminal en tiempos de prueba reservados, sin elegir por prueba. Las dispersiones son entre semillas. residence.csv registra permanencias: los sobrevivientes finales son observaciones censuradas y no deben mezclarse sin distinción con permanencias completas. rank_pruning.csv guarda propuestas, aceptación y cantidad efectiva. Se reconstruyeron 80 podas guardadas y se comprobó inicio común y calendario k emparejado. Los tiempos incluyen cálculo y diagnóstico y no prueban aceleración.\n\nReproducir: `python scripts/run_collocation_rank_pruning.py`. Las corridas completas se reutilizan. Las figuras están incrustadas como PNG. El siguiente cambio a considerar es renovar candidatos, condicionado a los resultados de este control."
    notebook=root/"pca_collocation/notebooks/burgers/21_rank_pruning.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
