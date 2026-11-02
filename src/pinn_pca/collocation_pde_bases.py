"""Matched residual subspaces under hard conditions and PDE-only training."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from .collocation import _run_collocation_seed,_save_csv,_qmc_points
from .collocation_basis_ablation import fixed_basis,read_rows
from .collocation_hard_conditions import HardRecorder
from .collocation_ic_confirmation import paired_stats
from .notebook_images import embed_notebook_images

LABELS=dict(pod_qr="POD sin centrar",pca_qr="PCA centrada",pca_mean_qr="PCA + media",sobol="Sobol")

def matched_selection(history,method,dimension):
    basis=fixed_basis(history,method,dimension)
    _,_,pivots=qr(basis.T,pivoting=True,mode="economic")
    h=np.asarray(history,dtype=float);mean=h.mean(axis=1);centered=h-mean[:,None]
    def fraction(matrix):
        norm=np.linalg.norm(matrix)**2
        return float(np.linalg.norm(basis.T@matrix)**2/norm) if norm else 1.
    return np.asarray(pivots[:dimension],dtype=int),dict(actual_dimension=dimension,
        centered_variance_captured=fraction(centered),raw_energy_captured=fraction(h),
        mean_projection_fraction=fraction(mean),qr_initial_count=dimension)

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/pde_bases"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    old=root/c.get("reference_root","pca_collocation/results/burgers/pde_only/pde_only_hard_10seeds_10k_v1")
    rows=[];traces=[];diagnostics=[];manifest=[];checked=0
    for seed in c["seeds"]:
        reference=old/"pca"/f"seed_{seed}"
        refdiag=read_rows(reference/"selection_diagnostics.csv")
        schedule={str(int(r["update_step"])-100):int(r["retained_modes"]) for r in refdiag}
        assert sorted(map(int,schedule))==list(range(c["warmup_steps"],c["warmup_steps"]+c["train_steps"],c["update_every"]))
        for method in LABELS:
            reused=method in ("pca_mean_qr","sobol")
            folder=old/("pca" if method=="pca_mean_qr" else "sobol")/f"seed_{seed}" if reused else out/method/f"seed_{seed}"
            if not reused and not (folder/"constraints.csv").exists():
                print(f"RUN matched PDE basis {method} seed={seed}",flush=True)
                spec=dict(base=method,label=LABELS[method],rank_schedule=schedule)
                _run_collocation_seed(dict(c,seed=seed,hard_conditions=True,pde_only=True,method_specs={method:spec},
                    basis_ablation=True,common_selection_seed=True,skip_seed_plots=True),root,folder,observer=HardRecorder(c,folder))
                print(f"DONE matched PDE basis {method} seed={seed}",flush=True)
            metric=read_rows(folder/"metrics.csv");original=metric[0]["method"]
            rows.extend(dict(r,method=method) for r in metric)
            traces.extend(dict(r,method=method) for r in read_rows(folder/"validation_trajectories.csv"))
            d=read_rows(folder/"selection_diagnostics.csv")
            if method!="sobol":
                assert {str(int(r["update_step"])-100):int(r["selected_qr_points"]) for r in d}==schedule
            diagnostics.extend(dict(r,method=method) for r in d)
            if not reused:
                archive=np.load(folder/"history_audit.npz");active=np.load(folder/"active_history.npz")
                for step in (100,200,1100,5100,10000):
                    ids,info=matched_selection(archive[f"{step}_history"],method,schedule[str(step)])
                    np.testing.assert_array_equal(active[f"{original}_{step}"][:len(ids)],archive[f"{step}_pool"][ids]);checked+=1
                for step in range(c["warmup_steps"],c["warmup_steps"]+c["train_steps"],c["update_every"]):
                    points=active[f"{original}_{step}"];k=schedule[str(step)]
                    assert len(points)==len(set(map(tuple,points)))==128
                    background=_qmc_points(128-k,seed+step+104729,(-1.,1.),(0.,float(np.float32(.99))))
                    np.testing.assert_array_equal(points[k:],background)
            manifest.append(dict(method=method,seed=seed,path=folder.relative_to(root).as_posix(),
                reused=reused and not c.get("confirmation",False),
                reused_within_study=reused and c.get("confirmation",False)))
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
        warm=[float(r["validation_rmse"]) for r in traces if int(r["seed"])==seed and int(r["step"])==100]
        assert len(warm)==4;np.testing.assert_array_equal(warm,[warm[0]]*4)
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"traces.csv",traces)
    # References predate the projection diagnostic; retain missing values explicitly.
    keys=list(dict.fromkeys(key for r in diagnostics for key in r))
    _save_csv(out/"selection_diagnostics.csv",[{key:r.get(key) for key in keys} for r in diagnostics])
    summary={};paired=[]
    for method in LABELS:
        group=[r for r in rows if r["method"]==method];assert len(group)==10
        summary[method]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms")}
    for method,reference in (("pca_mean_qr","pod_qr"),("pca_qr","pod_qr"),("pca_mean_qr","pca_qr"),("pod_qr","sobol"),("pca_qr","sobol"),("pca_mean_qr","sobol")):
        for metric in ("test_rmse","independent_residual_rms"):
            get=lambda seed,m:float(next(r for r in rows if r["method"]==m and int(r["seed"])==seed)[metric])
            paired.append(dict(method=method,reference=reference,metric=metric,
                **paired_stats([get(s,method)-get(s,reference) for s in c["seeds"]])))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,new_runs=40 if c.get("confirmation") else 20,reused_runs=0 if c.get("confirmation") else 20,matched_ranks=True,
        matched_background=True,qr_reconstructions=checked,common_warmup=True),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for method,label in LABELS.items():
        group=[r for r in traces if r["method"]==method];steps=sorted(set(int(r["step"]) for r in group))
        for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
            ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=label)
            ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Base/control | RMSE prueba ± DE | Residuo RMS |\n|---|---:|---:|\n"
    for method,label in LABELS.items():
        r=summary[method];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} |\n"
    text="# Bases del residuo bajo entrenamiento puramente físico\n\nSe comparan POD sin centrar, PCA centrada y PCA + media con igual dimensión total por semilla e intervalo, QR directo y reconstrucción del lote hasta 128 con el mismo prefijo Sobol. La media ocupa una dirección dentro del presupuesto. Se conserva la pérdida solo PDE, transformación IC/BC, red 3×32, Adam, candidatos fijos y veinte snapshots cada cinco pasos del último intervalo de cien.\n\nEl calendario de dimensiones se extrae de la trayectoria PCA + media al 95 % del notebook 26, antes de ejecutar los nuevos brazos. POD y PCA centrada usan ese calendario, no eligen su propio rango ni umbral. Es un control condicional emparejado para comparar subespacios a igual presupuesto; no evalúa sus versiones adaptativas autónomas. PCA + media y Sobol se reutilizan. La dimensión total incluye la media independiente en la referencia.\n\nPOD toma los primeros d modos de R; PCA toma los primeros d componentes de R centrada por punto. Se construyen bases ortonormales antes del mismo QR. El selector falla explícitamente si no alcanza el rango numérico solicitado, sin añadir direcciones artificiales. Las historias iniciales son comunes; después evolucionan con cada red. Las coordenadas y el complemento Sobol permanecen emparejados.\n\nDiez semillas 2071–2080, cien pasos uniformes y diez mil de continuación. Son semillas ya estudiadas: control exploratorio, no confirmación independiente. No hay datos supervisados en el objetivo, poda ni ampliación log-det.\n\n"+table+"\n\n"
    for r in paired:
        if r["metric"]=="test_rmse":text+=f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia media RMSE {r['mean_difference']:+.5f}; mejora en {r['wins']}/10 semillas; bootstrap pareado 95 % [{r['bootstrap95'][0]:+.5f}, {r['bootstrap95'][1]:+.5f}].\n\n"
    text+="![Comparación](../../results/burgers/pde_bases/"+c["run_id"]+"/comparison.png)\n\nSe evalúa el modelo terminal en prueba reservada sin ajustar parámetros a prueba. DE describe dispersión entre semillas. El bootstrap con diez diferencias es exploratorio y no corrige multiplicidad. El menor residuo no garantiza menor error de solución. Las fracciones de energía, fluctuaciones y media capturadas se guardan para los brazos nuevos; las referencias reutilizadas no incluyen esas columnas, que se dejan vacías.\n\nLas auditorías reconstruyen cien selecciones QR y comprueban calendario de rangos, tamaño activo, complemento Sobol e inicio común. Las restricciones IC/BC se verifican por el mismo mecanismo del notebook 26. manifest.json identifica cada procedencia.\n\nReproducir: `python scripts/run_collocation_pde_bases.py`. Configuración en pca_collocation/configs/burgers_pde_bases.json. Corridas completas se reutilizan. Las figuras están incrustadas como PNG."
    if c.get("confirmation"):
        text=text.replace("# Bases del residuo bajo entrenamiento puramente físico", "# Confirmación independiente de las bases del residuo")
        text=text.replace("del notebook 26, antes de ejecutar los nuevos brazos", "generada en las semillas nuevas con la regla congelada al 95 %, antes de ejecutar POD y PCA centrada")
        text=text.replace("PCA + media y Sobol se reutilizan.", "PCA + media y Sobol se ejecutan nuevamente para esta cohorte; sus archivos se reutilizan únicamente dentro de este estudio.")
        text=text.replace("Diez semillas 2071–2080, cien pasos uniformes y diez mil de continuación. Son semillas ya estudiadas: control exploratorio, no confirmación independiente.", "Diez semillas nuevas 2091–2100, cien pasos uniformes y diez mil de continuación. El protocolo se congela antes de ejecutar las cuarenta corridas. Comparación principal: PCA centrada menos POD en RMSE terminal. Los demás contrastes son secundarios y descriptivos.")
        text=text.replace("Las fracciones de energía, fluctuaciones y media capturadas se guardan para los brazos nuevos; las referencias reutilizadas no incluyen esas columnas, que se dejan vacías.", "Las fracciones de energía, fluctuaciones y media se guardan para POD/PCA centrada; los brazos que generan la referencia de rangos no incluyen esas columnas y se dejan vacías.")
        text=text.replace("python scripts/run_collocation_pde_bases.py", "python scripts/run_collocation_pde_bases_confirmation.py")
        text=text.replace("configs/burgers_pde_bases.json", "configs/burgers_pde_bases_confirmation.json")
    notebook=root/"pca_collocation/notebooks/burgers"/c.get("notebook_name","28_pde_bases.ipynb")
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
