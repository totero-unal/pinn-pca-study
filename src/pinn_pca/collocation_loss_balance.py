"""Controlled IC weight / explicit boundary penalty, after a shared warmup."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .models import burgers_solution,burgers_residual
from .collocation import _run_collocation_seed,_save_csv
from .collocation_rank_pruning import RankPruningRecorder
from .collocation_basis_ablation import read_rows
from .notebook_images import embed_notebook_images

COMPONENTS=("data","physics","initial","boundary")

def components(params,batch):
    dx,dt,du,rx,rt,ix,iu=batch
    pred=jax.vmap(lambda x,t:burgers_solution(params,x,t))(dx,dt)
    residual=jax.vmap(lambda x,t:burgers_residual(params,x,t))(rx,rt)
    ic=jax.vmap(lambda x:burgers_solution(params,x,jnp.float32(0)))(ix)
    # Common supervised-training times, independent of the adaptive point selector.
    left=jax.vmap(lambda t:burgers_solution(params,jnp.float32(-1),t))(dt)
    right=jax.vmap(lambda t:burgers_solution(params,jnp.float32(1),t))(dt)
    return jnp.stack((jnp.mean((pred-du)**2),jnp.mean(residual**2),jnp.mean((ic-iu)**2),
                      .5*(jnp.mean(left**2)+jnp.mean(right**2))))

def weighted_loss(weights):
    w=jnp.asarray(weights,dtype=jnp.float32)
    if w.shape!=(4,) or np.any(np.asarray(w)<0):raise ValueError("Four nonnegative weights required")
    return lambda params,batch:jnp.dot(w,components(params,batch))

diagnostic=jax.jit(lambda params,batch:(components(params,batch),jax.jacrev(components)(params,batch)))

class BalanceRecorder(RankPruningRecorder):
    def __init__(self,c,folder,selector):
        super().__init__(c,folder,"pca_partial" if selector=="prune" else "rebuild")
        self.selector=selector;self.component_rows=[]
    def adjust_active(self,*args):
        if self.selector=="prune":return super().adjust_active(*args)
        return args[3],args[4],args[5]
    def checkpoint(self,method,step,params,active,selection_step):
        super().checkpoint(method,step,params,active,selection_step)
        if step not in (100,1100,5100,10100):return
        batch=self.c["probe_batch"]
        points=self.c["independent"][:128]
        probe=(*batch[:3],jnp.asarray(points[:,0]),jnp.asarray(points[:,1]),*batch[5:])
        values,grads=diagnostic(params,probe)
        flat=np.concatenate([np.asarray(a).reshape(4,-1) for a in jax.tree.leaves(grads)],axis=1)
        norms=np.linalg.norm(flat,axis=1)
        w=np.asarray(self.config["loss_weights"])
        row=dict(step=step)
        for i,key in enumerate(COMPONENTS):
            row[key+"_loss"]=float(values[i]);row[key+"_grad_norm"]=float(norms[i]);row[key+"_weighted_grad_norm"]=float(w[i]*norms[i])
        for i,name in ((2,"physics_initial_cosine"),(3,"physics_boundary_cosine")):
            denom=norms[1]*norms[i];row[name]=float(np.dot(flat[1],flat[i])/denom) if denom else 0.
        self.component_rows.append(row)
    def end_seed(self):
        super().end_seed()
        _save_csv(self.output/"components.csv",self.component_rows)

OBJECTIVES=dict(baseline=[1.,1.,1.,0.],ic10=[1.,1.,10.,0.],bc1=[1.,1.,1.,1.])

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/loss_balance"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];diags=[];manifest=[]
    old=root/"pca_collocation/results/burgers"
    for objective,weights in OBJECTIVES.items():
        for selector in ("prune","rebuild"):
            for seed in c["seeds"]:
                reused=objective=="baseline"
                folder=(old/"rank_pruning/pca95_rank_pruning_10_seeds_10k_v1/pca_partial"/f"seed_{seed}" if selector=="prune" else
                        old/"interval_history/interval20_every5_10_seeds_10k_v1/interval_fixed"/f"seed_{seed}") if reused else out/objective/selector/f"seed_{seed}"
                if not reused and not (folder/"components.csv").exists():
                    print(f"RUN {objective} {selector} seed={seed}",flush=True)
                    name=f"{objective}_{selector}"
                    spec=dict(base="pca_mean_qr",label=name,adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                    saved=dict(c,seed=seed,loss_weights=weights,method_specs={name:spec},common_selection_seed=True,skip_seed_plots=True)
                    _run_collocation_seed(saved,root,folder,observer=BalanceRecorder(saved,folder,selector))
                    print(f"DONE {objective} {selector} seed={seed}",flush=True)
                if not (folder/"metrics.csv").exists():raise ValueError("Incomplete result")
                rows.extend(dict(r,objective=objective,selector=selector) for r in read_rows(folder/"metrics.csv"))
                traces.extend(dict(r,objective=objective,selector=selector) for r in read_rows(folder/"validation_trajectories.csv"))
                if not reused:diags.extend(dict(r,objective=objective,selector=selector,seed=seed) for r in read_rows(folder/"components.csv"))
                manifest.append(dict(objective=objective,selector=selector,seed=seed,path=folder.relative_to(root).as_posix(),reused=reused))
                (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"components.csv",diags)
    summary=[];paired=[]
    for objective in OBJECTIVES:
        for selector in ("prune","rebuild"):
            group=[r for r in rows if r["objective"]==objective and r["selector"]==selector];assert len(group)==10
            row=dict(objective=objective,selector=selector)
            for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms"):
                row[key]=float(np.mean([float(r[key]) for r in group]));row[key+"_std"]=float(np.std([float(r[key]) for r in group],ddof=1))
            summary.append(row)
            if objective!="baseline":
                delta=[]
                for seed in c["seeds"]:
                    get=lambda obj:float(next(r for r in rows if r["objective"]==obj and r["selector"]==selector and int(r["seed"])==seed)["test_rmse"])
                    delta.append(get(objective)-get("baseline"))
                paired.append(dict(objective=objective,selector=selector,difference=float(np.mean(delta)),wins=int(np.sum(np.array(delta)<0))))
    _save_csv(out/"summary.csv",summary)
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    for selector in ("prune","rebuild"):
        for seed in c["seeds"]:
            values=[]
            for objective in OBJECTIVES:
                values.append(float(next(r for r in traces if r["objective"]==objective and r["selector"]==selector and int(r["seed"])==seed and int(r["step"])==100)["validation_rmse"]))
            np.testing.assert_array_equal(values,[values[0]]*3)
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,shared_warmup=True,new_runs=40,reused_runs=20,diagnostic_checkpoints=len(diags)),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,selector in zip(axes,("prune","rebuild")):
        for objective in OBJECTIVES:
            group=[r for r in traces if r["objective"]==objective and r["selector"]==selector];steps=sorted(set(int(r["step"]) for r in group))
            ax.plot(steps,[np.mean([float(r["validation_rmse"]) for r in group if int(r["step"])==s]) for s in steps],label=objective)
        ax.set(xlabel="Paso total",ylabel="RMSE validación",title=selector);ax.legend()
    fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Pérdida | Selector | RMSE prueba ± DE | Residuo RMS | RMSE IC | RMS BC |\n|---|---|---:|---:|---:|---:|\n"
    for r in summary:table+=f"| {r['objective']} | {r['selector']} | {r['test_rmse']:.5f} ± {r['test_rmse_std']:.5f} | {r['independent_residual_rms']:.5f} | {r['initial_condition_rmse']:.5f} | {r['boundary_rms']:.5f} |\n"
    text="# Balance de pérdida: condición inicial y frontera\n\nDos selectores: poda parcial de k y reconstrucción, con PCA + media al 95 %, QR directo, candidatos fijos, 20 snapshots cada cinco pasos del último intervalo de cien y 128 activos. Diez semillas, 100 pasos uniformes comunes y 10000 de continuación.\n\nSe cambia únicamente la pérdida después del warmup, conservando los momentos de Adam. baseline = datos + PDE + IC; ic10 = datos + PDE + 10 IC; bc1 = datos + PDE + IC + BC. No se combinan todavía IC10 y BC: estos controles separan los efectos. El peso 10 se fija como contraste exploratorio, no como óptimo. Las referencias baseline se reutilizan de notebooks 21 y 19. Las historias divergen después del cambio de pérdida, como efecto del entrenamiento adaptativo.\n\nBC es la media del error cuadrado en x=−1 y x=1 con objetivo cero; se usan los tiempos supervisados del batch de entrenamiento, compartidos entre selectores. No se usan tiempos de prueba para imponer frontera. La condición inicial mantiene 32 nodos distribuidos. El término BC añade restricciones al objetivo y costo de derivación; no se afirma igualdad de costo computacional.\n\n"+table+"\n\n"+"\n\n".join(f"{r['objective']} / {r['selector']} frente a su baseline: diferencia media RMSE {r['difference']:+.5f}; mejora en {r['wins']}/10 semillas." for r in paired)+"\n\n![Validación](../../results/burgers/loss_balance/"+c["run_id"]+"/comparison.png)\n\nLos modelos se evalúan al presupuesto terminal, sin seleccionar por prueba. La desviación representa dispersión entre semillas. components.csv registra valores sin ponderar, normas de gradiente y normas ponderadas, y cosenos PDE–IC y PDE–BC en los pasos 100, 1100, 5100 y 10100. Los monitores usan el mismo batch supervisado/IC de referencia y 128 puntos físicos independientes fijos por semilla, para comparar redes sin confundir el lote adaptativo. Son diagnósticos locales de gradiente: no prueban por sí solos causalidad ni describen todo el entrenamiento. Las referencias reutilizadas no tienen este diagnóstico adicional, lo que se mantiene explícito en manifest.json.\n\nReproducir: `python scripts/run_collocation_loss_balance.py`. Se reutilizan corridas completas. Esta prueba cambia el objetivo tras el warmup; no reproduce las condiciones duras del artículo. Si un control mejora, conviene confirmarlo con nuevas semillas y estudiar sensibilidad de pesos antes de adoptarlo."
    gradient_table="\n\n## Diagnóstico final de gradientes\n\n| Pérdida | Selector | Norma PDE | Norma IC ponderada | Norma BC ponderada | Coseno PDE–IC | Coseno PDE–BC |\n|---|---|---:|---:|---:|---:|---:|\n"
    for objective in ("ic10","bc1"):
        for selector in ("prune","rebuild"):
            group=[r for r in diags if r["objective"]==objective and r["selector"]==selector and int(r["step"])==10100]
            keys=("physics_weighted_grad_norm","initial_weighted_grad_norm","boundary_weighted_grad_norm","physics_initial_cosine","physics_boundary_cosine")
            gradient_table+=f"| {objective} | {selector} | "+" | ".join(f"{np.mean([float(r[key]) for r in group]):.4f}" for key in keys)+" |\n"
    text+=gradient_table+"\nPromedios entre semillas sobre el probe común al paso final. Un coseno negativo indica oposición local de gradientes, no demuestra por sí solo la causa del error de solución. La magnitud de una componente de pérdida no equivale a la magnitud de su gradiente."
    notebook=root/"pca_collocation/notebooks/burgers/23_loss_balance.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
