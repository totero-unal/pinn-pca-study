"""Last-interval residual histories with fixed or refreshed candidate locations."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .collocation import _run_collocation_seed,_residual_field,_qmc_points,_save_csv
from .collocation_partial import Recorder
from .collocation_basis_ablation import read_rows
from .collocation_rank_study import adaptive_basis
from .notebook_images import embed_notebook_images


class IntervalRecorder(Recorder):
    def __init__(self,c,folder,refresh):
        super().__init__(c,folder,"interval")
        self.refresh=refresh;self.next_pool=None;self.columns=[];self.times=[];self.history_rows=[];self.saved={}

    def prepare_history(self,step,pool,history,bounds):
        if self.next_pool is not None:
            pool=self.next_pool
            history=np.column_stack(self.columns)
            assert history.shape==(self.config["candidate_count"],20)
            assert self.times==list(range(step-95,step+1,5))
        self.history_rows.append(dict(selection_step=step,first_snapshot_step=self.times[0] if self.times else 1,
            last_snapshot_step=self.times[-1] if self.times else 100,snapshot_count=history.shape[1],
            renewed=self.refresh and step>100))
        if step in (100,200,1100,5100,10000):
            self.saved[f"{step}_pool"]=pool.copy();self.saved[f"{step}_history"]=history.copy()
        self.next_pool=(_qmc_points(self.config["candidate_count"],self.c["seed"]+100000+step,*bounds)
                        if self.refresh else pool)
        self.columns=[];self.times=[]
        return pool,history

    def training_step(self,step,params):
        if step%5==0:
            self.columns.append(_residual_field(params,self.next_pool));self.times.append(step)

    def end_seed(self):
        super().end_seed()
        _save_csv(self.output/"history_intervals.csv",self.history_rows)
        np.savez_compressed(self.output/"history_audit.npz",**self.saved)


LABELS=dict(baseline="Ventana previa, candidatos fijos",interval_fixed="Último intervalo, candidatos fijos",
            interval_refresh="Último intervalo, candidatos renovados",sobol="Sobol completo")


def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/interval_history"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True)
    p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use another run_id")
    p.write_text(json.dumps(c,indent=2))
    rows=[];traces=[];counts=[]
    for seed in c["seeds"]:
        for variant in ("interval_fixed","interval_refresh"):
            folder=out/variant/f"seed_{seed}"
            if not (folder/"history_audit.npz").exists():
                print(f"RUN {variant} seed={seed}",flush=True)
                spec=dict(base="pca_mean_qr",label=LABELS[variant],adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,method_specs={variant:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,
                    observer=IntervalRecorder(c,folder,variant=="interval_refresh"))
                print(f"DONE {variant} seed={seed}",flush=True)
            rows.extend(read_rows(folder/"metrics.csv"));traces.extend(read_rows(folder/"validation_trajectories.csv"))
            counts.extend(read_rows(folder/"selection_diagnostics.csv"))
        direct=root/"pca_collocation/results/burgers/direct_qr/pca95_direct_qr_10_seeds_10k_v1"/f"seed_{seed}"
        for filename,target in (("metrics.csv",rows),("validation_trajectories.csv",traces)):
            target.extend(dict(r,method="baseline") for r in read_rows(direct/filename))
        prior=root/"pca_collocation/results/burgers/partial_update/pca95_partial_10_seeds_10k_v1/sobol"/f"seed_{seed}"
        rows.extend(read_rows(prior/"metrics.csv"));traces.extend(read_rows(prior/"validation_trajectories.csv"))
        # Same signed history and same active set at the first selection.
        a=np.load(out/"interval_fixed"/f"seed_{seed}/history_audit.npz")
        b=np.load(out/"interval_refresh"/f"seed_{seed}/history_audit.npz")
        np.testing.assert_array_equal(a["100_history"],b["100_history"])
        np.testing.assert_array_equal(a["100_pool"],b["100_pool"])
        for archive in (a,b):
            for step in (100,200,1100,5100,10000):
                basis,_=adaptive_basis(archive[f"{step}_history"],"pca_mean_qr",.95)
                assert np.isfinite(basis).all()
        np.testing.assert_array_equal(a["100_pool"],a["200_pool"])
        assert not np.array_equal(b["100_pool"],b["200_pool"])
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"traces.csv",traces);_save_csv(out/"selection_counts.csv",counts)
    summary={}
    for variant in LABELS:
        group=[r for r in rows if r["method"]==variant]
        assert len(group)==10
        summary[variant]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse")}
    paired=[]
    for variant,reference in (("interval_fixed","baseline"),("interval_refresh","interval_fixed")):
        d=[]
        for seed in c["seeds"]:
            get=lambda m:float(next(r for r in rows if r["method"]==m and int(r["seed"])==seed)["test_rmse"])
            d.append(get(variant)-get(reference))
        paired.append(dict(method=variant,reference=reference,difference=float(np.mean(d)),wins=int(np.sum(np.array(d)<0))))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for variant,label in LABELS.items():
        group=[r for r in traces if r["method"]==variant];steps=sorted(set(int(r["step"]) for r in group))
        for ax,key in zip(axes,("validation_rmse","independent_residual_rms")):
            ax.plot(steps,[np.mean([float(r[key]) for r in group if int(r["step"])==s]) for s in steps],label=label)
            ax.set(xlabel="Paso total",ylabel=key)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Método | RMSE prueba ± DE | Residuo RMS |\n|---|---:|---:|\n"
    for variant,label in LABELS.items():
        r=summary[variant];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} |\n"
    text="# PCA: historia del último intervalo y renovación de candidatos\n\nSe conserva PCA centrada + media al 95%, k puntos por QR directo y 128−k Sobol; reconstrucción cada 100 pasos, pérdida histórica, red y Adam, diez semillas y 10000 pasos de continuación.\n\n**Referencia previa:** ventana de 20 snapshots, uno nuevo cada 100 pasos después del warmup, candidatos fijos. **Intervalo fijo:** 20 snapshots posteriores a cada quinto paso del último intervalo de 100, candidatos fijos. **Intervalo renovado:** la misma cadencia e historia, con 4096 candidatos Sobol nuevos por intervalo.\n\nLa primera selección es idéntica y usa la historia del warmup común. Después se recolecta la siguiente historia mientras se entrena sobre el lote ya elegido. En la variante renovada se genera un nuevo conjunto al comenzar la recolección y se mantiene fijo durante los 20 snapshots: cada fila siempre corresponde a una coordenada. No se concatenan residuos de coordenadas distintas. Los snapshots se toman después de actualizar Adam. Los últimos candidatos observados al terminar no se utilizan para otra selección.\n\n**Alcance causal:** referencia frente a intervalo fijo cambia conjuntamente cadencia y memoria temporal; no atribuye efectos únicamente a borrar la historia. Intervalo fijo frente a renovado sí aísla el cambio de candidatos. Los dos brazos nuevos comparten frecuencia, duración y presupuesto de observación. El costo de observar es mayor que en la referencia; no se afirma aceleración. El número k puede cambiar por el nuevo espectro. Sobol es el control general reutilizado.\n\n"+table+"\n\n"+"\n\n".join(f"{LABELS[r['method']]} frente a {LABELS[r['reference']]}: diferencia RMSE {r['difference']:+.5f}, mejora en {r['wins']}/10 semillas." for r in paired)+"\n\n![Comparación](../../results/burgers/interval_history/"+c["run_id"]+"/comparison.png)\n\nSe evalúa el modelo terminal en tiempos de prueba reservados, sin elegir por prueba. Las referencias anteriores se reutilizan; métricas y trazas identifican su método. history_intervals.csv guarda los pasos de cada intervalo y history_audit.npz guarda coordenadas y matrices con signo. Se verifica historia inicial común, candidatos fijos frente a renovados y forma/cadencia de las historias. Las figuras están incrustadas como PNG.\n\nEjecutar `python scripts/run_collocation_intervals.py`; las corridas completas se reutilizan. El control posterior podrá separar frecuencia de observación y duración de la memoria."
    notebook=root/"pca_collocation/notebooks/burgers/19_interval_history.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),indent=2,ensure_ascii=False),encoding="utf-8")
    embed_notebook_images(notebook,root)
    (out/"audit_checks.json").write_text(json.dumps(dict(passed=True,seeds=10,intervals_per_new_method=100,common_initial_history=True,coordinates_consistent=True),indent=2))
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
