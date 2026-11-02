"""Hard IC/BC transform with the existing supervised-data plus PDE objective."""
import json
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from .models import mlp,NU
from .collocation import _run_collocation_seed,_save_csv
from .collocation_intervals import IntervalRecorder
from .collocation_basis_ablation import read_rows
from .collocation_ic_confirmation import paired_stats
from .notebook_images import embed_notebook_images

def solution(params,x,t):
    initial=jnp.where(jnp.abs(x)==1,0.,-jnp.sin(jnp.pi*x))
    return initial+t*(1-x*x)*mlp(params,jnp.stack((x,t)))[0]

def residual(params,x,t):
    f=lambda xx,tt:solution(params,xx,tt)
    ux=jax.grad(f,0)(x,t);ut=jax.grad(f,1)(x,t)
    uxx=jax.grad(lambda xx:jax.grad(f,0)(xx,t))(x)
    return ut+f(x,t)*ux-NU*uxx

residual_batch=jax.jit(jax.vmap(residual,in_axes=(None,0,0)))
predict=jax.jit(jax.vmap(solution,in_axes=(None,0,0)))

def residual_field(params,points,chunk=256):
    return np.concatenate([np.asarray(residual_batch(params,p[:,0],p[:,1])) for p in np.array_split(points,max(1,int(np.ceil(len(points)/chunk))))])

def loss(params,batch):
    x,t,u,rx,rt,_,_=batch
    return jnp.mean((predict(params,x,t)-u)**2)+jnp.mean(residual_batch(params,rx,rt)**2)

def pde_only_loss(params,batch):
    """Only physical collocation coordinates are read; all label fields are ignored."""
    return jnp.mean(residual_batch(params,batch[3],batch[4])**2)

def evaluate(params,x,t,u,ids,points):
    xx,tt=np.meshgrid(x,t[ids],indexing="ij")
    values=np.asarray(predict(params,xx.ravel(),tt.ravel())).reshape(xx.shape)
    truth=u[:,ids];r=residual_field(params,points)
    ic=np.asarray(predict(params,x,np.zeros_like(x)))
    boundary=np.asarray(predict(params,np.tile(np.array([-1,1],np.float32),len(ids)),np.repeat(t[ids],2)))
    return dict(test_rmse=float(np.sqrt(np.mean((values-truth)**2))),test_relative_l2=float(np.linalg.norm(values-truth)/np.linalg.norm(truth)),
        independent_residual_rms=float(np.sqrt(np.mean(r**2))),initial_condition_rmse=float(np.sqrt(np.mean((ic-u[:,0])**2))),boundary_rms=float(np.sqrt(np.mean(boundary**2))))

class HardRecorder(IntervalRecorder):
    def __init__(self,c,folder):
        super().__init__(c,folder,False);self.constraints=[]
    def training_step(self,step,params):
        if step%5==0:
            self.columns.append(residual_field(params,self.next_pool));self.times.append(step)
    def checkpoint(self,method,step,params,active,selection_step):
        c=self.c
        if step in self.config["test_checkpoints"]:
            self.metrics.append(dict(seed=c["seed"],method=method,step=step,
                **evaluate(params,c["x"],c["t"],c["u"],c["test_ids"],c["independent"])))
            x=np.linspace(-1,1,257,dtype=np.float32);times=np.linspace(0,1,257,dtype=np.float32)
            exact=np.where(np.abs(x)==1,0.,-np.sin(np.pi*x))
            ic=np.asarray(predict(params,x,np.zeros_like(x)))
            bc=np.asarray(predict(params,np.tile(np.array([-1,1],np.float32),len(times)),np.repeat(times,2)))
            self.constraints.append(dict(step=step,ic_max=float(np.max(np.abs(ic-exact))),bc_max=float(np.max(np.abs(bc)))))
            assert self.constraints[-1]["ic_max"]<1e-6 and self.constraints[-1]["bc_max"]==0
    def end_seed(self):
        super().end_seed();_save_csv(self.output/"constraints.csv",self.constraints)

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/hard_conditions"/c["run_id"]
    if c.get("loss_weights"):raise ValueError("Hard transform uses data+PDE only")
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];manifest=[]
    for seed in c["seeds"]:
        for selector in ("pca","sobol"):
            folder=out/selector/f"seed_{seed}"
            if not (folder/"constraints.csv").exists():
                print(f"RUN hard conditions {selector} seed={seed}",flush=True)
                spec=dict(base="sobol",label="Sobol hard IC/BC") if selector=="sobol" else dict(
                    base="pca_mean_qr",label="PCA hard IC/BC",adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,method_specs={selector:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,observer=HardRecorder(c,folder))
                print(f"DONE hard conditions {selector} seed={seed}",flush=True)
            rows.extend(dict(r,conditions="hard",selector=selector) for r in read_rows(folder/"metrics.csv"))
            traces.extend(dict(r,conditions="hard",selector=selector) for r in read_rows(folder/"validation_trajectories.csv"))
            reference=root/"pca_collocation/results/burgers/ic_confirmation/ic10_sobol_confirmation_10new_v1/confirmation/ic10"/selector/f"seed_{seed}"
            rows.extend(dict(r,conditions="soft_ic10",selector=selector) for r in read_rows(reference/"metrics.csv"))
            traces.extend(dict(r,conditions="soft_ic10",selector=selector) for r in read_rows(reference/"validation_trajectories.csv"))
            manifest.extend([dict(selector=selector,seed=seed,conditions="hard",path=folder.relative_to(root).as_posix(),reused=False),
                dict(selector=selector,seed=seed,conditions="soft_ic10",path=reference.relative_to(root).as_posix(),reused=True)])
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"traces.csv",traces)
    summary=[];paired=[]
    for condition in ("soft_ic10","hard"):
        for selector in ("pca","sobol"):
            group=[r for r in rows if r["conditions"]==condition and r["selector"]==selector];assert len(group)==10
            item=dict(conditions=condition,selector=selector)
            for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms"):
                a=np.array([float(r[key]) for r in group]);assert np.isfinite(a).all()
                item[key]=float(a.mean());item[key+"_std"]=float(a.std(ddof=1))
            summary.append(item)
        get=lambda seed,selector,key:float(next(r for r in rows if r["conditions"]==condition and r["selector"]==selector and int(r["seed"])==seed)[key])
        for key in ("test_rmse","independent_residual_rms"):
            paired.append(dict(comparison="PCA minus Sobol",conditions=condition,metric=key,
                **paired_stats([get(s,"pca",key)-get(s,"sobol",key) for s in c["seeds"]])))
    for selector in ("pca","sobol"):
        get=lambda seed,condition:float(next(r for r in rows if r["conditions"]==condition and r["selector"]==selector and int(r["seed"])==seed)["test_rmse"])
        paired.append(dict(comparison="hard minus soft",selector=selector,metric="test_rmse",
            **paired_stats([get(s,"hard")-get(s,"soft_ic10") for s in c["seeds"]])))
    # Hard methods share initial raw parameters and transformed warmup, not the soft weights.
    for seed in c["seeds"]:
        warm=[float(r["validation_rmse"]) for r in traces if r["conditions"]=="hard" and int(r["seed"])==seed and int(r["step"])==100]
        np.testing.assert_array_equal(warm,[warm[0]]*2)
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    _save_csv(out/"summary.csv",summary)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
        for condition in ("soft_ic10","hard"):
            for selector in ("pca","sobol"):
                group=[r for r in traces if r["conditions"]==condition and r["selector"]==selector];steps=sorted(set(int(r["step"]) for r in group))
                ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=f"{selector} {condition}")
        ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Condiciones | Selector | RMSE prueba ± DE | Residuo RMS | RMSE IC | RMS BC |\n|---|---|---:|---:|---:|---:|\n"
    for r in summary:table+=f"| {r['conditions']} | {r['selector']} | {r['test_rmse']:.5f} ± {r['test_rmse_std']:.5f} | {r['independent_residual_rms']:.5f} | {r['initial_condition_rmse']:.2e} | {r['boundary_rms']:.2e} |\n"
    text="# PCA y Sobol con condiciones iniciales y de frontera duras\n\nTransformación: $u_\\theta(x,t)=-\\sin(\\pi x)+t(1-x^2)N_\\theta(x,t)$. Impone IC y BC homogéneas; una corrección del seno en x=±1 elimina redondeo float32. El residuo y sus derivadas automáticas se calculan sobre toda la solución transformada. La pérdida hard es **datos supervisados + PDE**: se conserva la información supervisada; IC y BC dejan de ser penalizaciones.\n\nLa referencia soft usa datos+PDE+10 IC, sin BC explícita, del notebook 24. Cambiar imposición modifica también el espacio de soluciones y la optimización: no es solo cambiar un coeficiente de pérdida. La transformación se aplica desde la inicialización, incluido el warmup; la referencia aplica IC10 después de su warmup original. Por tanto, los brazos hard y soft comparten semilla y parámetros crudos iniciales, pero sus estados tras cien pasos son distintos. Dentro de hard, PCA y Sobol comparten exactamente el warmup.\n\nDiez semillas 2071–2080, 100 pasos uniformes y 10000 de continuación, red 3×32 y Adam constante, mismos datos supervisados y particiones. PCA usa reconstrucción, candidatos fijos, media + componentes al 95 %, QR directo, veinte snapshots cada cinco pasos del último intervalo de cien. Se completa a 128 con Sobol; el control usa 128 Sobol. No se utiliza poda. Las referencias soft se reutilizan; manifest.json identifica las rutas. Es una extensión exploratoria sobre las semillas de confirmación anteriores, no otra confirmación independiente.\n\n"+table+"\n\n"
    for r in paired:
        if r["metric"]=="test_rmse":text+=f"{r['comparison']} ({r.get('conditions',r.get('selector'))}): diferencia media {r['mean_difference']:+.5f}; mejora en {r['wins']}/10 semillas; intervalo bootstrap 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n\n"
    text+="![Comparación](../../results/burgers/hard_conditions/"+c["run_id"]+"/comparison.png)\n\nLas métricas terminales se evalúan en prueba reservada sin elegir por prueba. El bootstrap pareado con diez semillas tiene incertidumbre y no corrige multiplicidad. DE es dispersión entre semillas. IC global se compara con el archivo de referencia y puede tener redondeo distinto de la identidad analítica. constraints.csv verifica máximos analíticos IC y BC en una malla independiente, incluidos extremos y t=1. history_audit.npz conserva historias del residuo transformado con sus coordenadas. No se afirma aceleración: la transformación agrega derivadas y PCA requiere observar candidatos.\n\nReproducir: `python scripts/run_collocation_hard_conditions.py`. Las corridas completas se reutilizan. El objetivo conserva datos supervisados; retirarlos para una comparación puramente física sería otro control. Las figuras están incrustadas como PNG."
    notebook=root/"pca_collocation/notebooks/burgers/25_hard_conditions.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,shared_hard_warmup=True,new_runs=20,reused_runs=20,analytic_constraints_checked=True),indent=2))
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
