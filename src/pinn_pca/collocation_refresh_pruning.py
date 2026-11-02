"""Refresh snapshot candidates while preserving rank-sized convergence pruning."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr
from .collocation import _run_collocation_seed,_save_csv
from .collocation_rank_pruning import RankPruningRecorder
from .collocation_rank_study import adaptive_basis
from .collocation_partial import replace_converged
from .collocation_basis_ablation import read_rows
from .notebook_images import embed_notebook_images

LABELS=dict(prune_refresh="Poda: candidatos renovados",prune_fixed="Poda: candidatos fijos",
            rebuild_fixed="Reconstrucción: candidatos fijos",rebuild_refresh="Reconstrucción: candidatos renovados")

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/refresh_pruning"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use a new run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];pruning=[];manifest=[];checks=0;sets=0
    previous=root/"pca_collocation/results/burgers"
    for seed in c["seeds"]:
        folder=out/f"seed_{seed}"
        if not (folder/"rank_pruning_audit.npz").exists():
            print(f"RUN refreshed rank pruning seed={seed}",flush=True)
            spec=dict(base="pca_mean_qr",label=LABELS["prune_refresh"],adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
            _run_collocation_seed(dict(c,seed=seed,method_specs={"prune_refresh":spec},common_selection_seed=True,skip_seed_plots=True),root,folder,
                observer=RankPruningRecorder(c,folder,"pca_partial",refresh=True))
            print(f"DONE refreshed rank pruning seed={seed}",flush=True)
        sources={"prune_refresh":folder,
            "prune_fixed":previous/"rank_pruning/pca95_rank_pruning_10_seeds_10k_v1/pca_partial"/f"seed_{seed}",
            "rebuild_fixed":previous/"interval_history/interval20_every5_10_seeds_10k_v1/interval_fixed"/f"seed_{seed}",
            "rebuild_refresh":previous/"interval_history/interval20_every5_10_seeds_10k_v1/interval_refresh"/f"seed_{seed}"}
        initial=[];prefix=[]
        for arm,source in sources.items():
            metric=read_rows(source/"metrics.csv");old_method=metric[0]["method"]
            rows.extend(dict(r,method=arm) for r in metric)
            trace=read_rows(source/"validation_trajectories.csv")
            traces.extend(dict(r,method=arm) for r in trace)
            prefix.append([float(r["validation_rmse"]) for r in trace if int(r["step"])<=200])
            initial.append(np.load(source/"active_history.npz")[f"{old_method}_100"])
            manifest.append(dict(method=arm,seed=seed,path=source.relative_to(root).as_posix(),reused=arm!="prune_refresh"))
        for a in initial[1:]:np.testing.assert_array_equal(a,initial[0])
        for a in prefix[1:]:np.testing.assert_array_equal(a,prefix[0])
        a=np.load(folder/"rank_pruning_audit.npz")
        for step in (200,1100,5100,10000):
            updated,removed,_,_=replace_converged(a[f"{step}_before"],a[f"{step}_proposals"],a[f"{step}_old"],a[f"{step}_current"])
            np.testing.assert_array_equal(updated,a[f"{step}_after"])
            np.testing.assert_array_equal(removed,a[f"{step}_removed"]);checks+=1
        h=np.load(folder/"history_audit.npz");active=np.load(folder/"active_history.npz")
        for step in (100,200,1100,5100,10000):
            basis,_=adaptive_basis(h[f"{step}_history"],"pca_mean_qr",.95)
            pool=h[f"{step}_pool"]
            if step>100:
                occupied=set(map(tuple,a[f"{step}_before"]))
                mask=np.array([tuple(p) not in occupied for p in pool])
                _,_,pivots=qr(basis[mask].T,pivoting=True,mode="economic")
                np.testing.assert_array_equal(a[f"{step}_proposals"],pool[mask][pivots[:basis.shape[1]]])
            assert pool.shape==(4096,2)
        assert not np.array_equal(h["100_pool"],h["200_pool"])
        for points in active.values():assert len(points)==len(set(map(tuple,points)))==128;sets+=1
        for arm in ("prune_refresh","prune_fixed"):
            pruning.extend(dict(seed=seed,method=arm,**r) for r in read_rows(sources[arm]/"rank_pruning.csv"))
        (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"pruning.csv",pruning)
    summary={}
    for arm in LABELS:
        group=[r for r in rows if r["method"]==arm];assert len(group)==10
        summary[arm]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse","boundary_rms")}
        if arm.startswith("prune"):
            group=[r for r in pruning if r["method"]==arm and r["initialization"]=="False"]
            summary[arm]["pruning"]=dict(accepted=sum(r["accepted"]=="True" for r in group),attempts=len(group),
                k_min=min(int(r["k"]) for r in group),k_max=max(int(r["k"]) for r in group),k_mean=float(np.mean([int(r["k"]) for r in group])))
    paired=[]
    for arm,reference in (("prune_refresh","prune_fixed"),("prune_refresh","rebuild_fixed"),("prune_refresh","rebuild_refresh")):
        d=[]
        for seed in c["seeds"]:
            get=lambda method:float(next(r for r in rows if r["method"]==method and int(r["seed"])==seed)["test_rmse"])
            d.append(get(arm)-get(reference))
        paired.append(dict(method=arm,reference=reference,difference=float(np.mean(d)),wins=int(np.sum(np.array(d)<0))))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,pruning_reconstructions=checks,active_sets=sets,
        shared_initial_points_and_first_interval=True,renewed_pools=True,eligible_qr_reconstructed=True),indent=2))
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
    text="# PCA: renovar candidatos dentro de la poda de k\n\nDiez semillas, 100 pasos uniformes y 10000 de continuación. PCA centrada + media al 95 %, QR directo, 128 activos, pérdida histórica datos+PDE+IC, veinte snapshots cada cinco pasos del último intervalo de cien. Se conserva la regla de poda por log2(|r anterior|/|r actual|), piso 1e-12 y cancelación si uno de los k grados elegidos es negativo.\n\nEl brazo nuevo renueva 4096 candidatos Sobol antes de recoger cada intervalo de snapshots. El conjunto permanece fijo durante los veinte momentos, por lo que cada fila mantiene sus coordenadas. La primera selección y primer intervalo comparten pesos, Adam y puntos con los tres controles. La primera actualización diferenciada ocurre en paso 200. QR se restringe a candidatos que no están activos; el rango se obtiene de la historia completa y determina k. Tras insertar puntos se realinea el residuo basal en las nuevas coordenadas.\n\nLos controles de poda fija, reconstrucción fija y reconstrucción con candidatos renovados se reutilizan de los notebooks 21 y 19; manifest.json identifica sus rutas. Comparar las dos podas aísla la renovación y sus consecuencias sobre el rango. k puede diferir entre trayectorias; no se impone un calendario compartido en esta prueba. Se conserva también la reconstrucción renovada para comparar actualización bajo la misma política de candidatos.\n\n"+table+"\n\n"+"\n\n".join(f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia media RMSE {r['difference']:+.5f}; mejora en {r['wins']}/10 semillas." for r in paired)+"\n\n![Comparación](../../results/burgers/refresh_pruning/"+c["run_id"]+"/comparison.png)\n\n"
    for arm in ("prune_refresh","prune_fixed"):
        r=summary[arm]["pruning"];text+=f"{LABELS[arm]}: {r['accepted']}/{r['attempts']} podas aceptadas; k entre {r['k_min']} y {r['k_max']}, media {r['k_mean']:.2f}.\n\n"
    text+="Se evalúa el modelo terminal en prueba reservada; ninguna configuración se elige usando prueba. La desviación es dispersión entre semillas. Los tiempos incluyen compilación y observación y no prueban aceleración. Las auditorías verifican la poda, las propuestas QR elegibles, los conjuntos renovados, el inicio común y tamaño constante. Permanencias y censura final están en residence.csv de cada corrida nueva.\n\nReproducir: `python scripts/run_collocation_refresh_pruning.py`. Se reutilizan las corridas completas. Las figuras están incrustadas como PNG. Si este control no mejora el error de solución, conviene estudiar el balance de pérdidas antes de seguir modificando el selector."
    notebook=root/"pca_collocation/notebooks/burgers/22_refresh_pruning.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
