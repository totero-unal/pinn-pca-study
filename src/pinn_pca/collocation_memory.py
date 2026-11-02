"""Separate snapshot cadence from temporal memory using fixed candidates."""
import json
import os
from collections import deque
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .collocation import _run_collocation_seed,_residual_field,_save_csv
from .collocation_partial import Recorder
from .collocation_basis_ablation import read_rows
from .notebook_images import embed_notebook_images

VARIANTS={"f5_m100":(5,100),"f10_m100":(10,100),"f20_m100":(20,100),"f5_m200":(5,200),"f5_m400":(5,400)}

class MemoryRecorder(Recorder):
    def __init__(self,c,folder,frequency,memory):
        super().__init__(c,folder,"memory")
        if memory%frequency:raise ValueError("Memory must be divisible by cadence")
        self.frequency=frequency;self.memory=memory
        self.columns=deque(maxlen=memory//frequency);self.times=deque(maxlen=memory//frequency)
        self.pool=None;self.log=[];self.saved={}
    def prepare_history(self,step,pool,history,bounds):
        if self.pool is None:self.pool=pool.copy()
        else:
            np.testing.assert_array_equal(pool,self.pool)
            history=np.column_stack(self.columns)
            expected=list(range(max(100,step-self.memory)+self.frequency,step+1,self.frequency))
            assert list(self.times)==expected
        self.log.append(dict(step=step,snapshots=history.shape[1],
            first_step=self.times[0] if self.times else 1,last_step=self.times[-1] if self.times else 100,
            full_window=step>=100+self.memory,frequency=self.frequency,memory=self.memory))
        if step in (100,200,300,500,1100,5100,10000):
            self.saved[f"{step}_history"]=history.copy()
            self.saved[f"{step}_times"]=np.array(self.times)
        return pool,history
    def training_step(self,step,params):
        if step%self.frequency==0:
            self.columns.append(_residual_field(params,self.pool));self.times.append(step)
    def end_seed(self):
        super().end_seed()
        _save_csv(self.output/"memory_history.csv",self.log)
        np.savez_compressed(self.output/"memory_audit.npz",pool=self.pool,**self.saved)

def run(c,root):
    root=Path(root).resolve();out=root/"pca_collocation/results/burgers/memory_cadence"/c["run_id"]
    out.mkdir(parents=True,exist_ok=True);p=out/"config.json"
    if p.exists() and json.loads(p.read_text())!=c:raise ValueError("Use new run_id")
    p.write_text(json.dumps(c,indent=2))
    manifest=[];rows=[];counts=[]
    for name,(frequency,memory) in VARIANTS.items():
        for seed in c["seeds"]:
            reuse=name=="f5_m100"
            folder=(root/"pca_collocation/results/burgers/interval_history/interval20_every5_10_seeds_10k_v1/interval_fixed"/f"seed_{seed}"
                if reuse else out/name/f"seed_{seed}")
            if not reuse and not (folder/"memory_audit.npz").exists():
                print(f"RUN {name} seed={seed}",flush=True)
                spec=dict(base="pca_mean_qr",label=name,adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(c,seed=seed,method_specs={name:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,
                    observer=MemoryRecorder(c,folder,frequency,memory))
                print(f"DONE {name} seed={seed}",flush=True)
            if not (folder/"audit.json").exists():raise ValueError("Incomplete run")
            rows.extend(dict(r,method=name) for r in read_rows(folder/"metrics.csv"))
            counts.extend(dict(r,method=name) for r in read_rows(folder/"selection_diagnostics.csv"))
            manifest.append(dict(method=name,seed=seed,path=folder.relative_to(root).as_posix(),reused=reuse))
            (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    _save_csv(out/"metrics.csv",rows);_save_csv(out/"ranks.csv",counts)
    summary={}
    for name in VARIANTS:
        group=[r for r in rows if r["method"]==name]
        ranks=[int(r["retained_modes"]) for r in counts if r["method"]==name]
        summary[name]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse")}
        summary[name]["rank"]=dict(min=min(ranks),max=max(ranks),mean=float(np.mean(ranks)))
    paired=[]
    for name in list(VARIANTS)[1:]:
        delta=[]
        for seed in c["seeds"]:
            get=lambda method:float(next(r for r in rows if r["method"]==method and int(r["seed"])==seed)["test_rmse"])
            delta.append(get(name)-get("f5_m100"))
        paired.append(dict(method=name,reference="f5_m100",mean_difference=float(np.mean(delta)),wins=int(np.sum(np.array(delta)<0))))
    (out/"summary.json").write_text(json.dumps(dict(methods=summary,paired=paired),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,names,x,label in ((axes[0],["f5_m100","f10_m100","f20_m100"],[5,10,20],"Pasos entre snapshots; memoria 100"),
                            (axes[1],["f5_m100","f5_m200","f5_m400"],[100,200,400],"Memoria en pasos; cadencia 5")):
        mean=[summary[n]["test_rmse"]["mean"] for n in names];std=[summary[n]["test_rmse"]["std"] for n in names]
        ax.errorbar(x,mean,yerr=std,fmt="o-",capsize=4);ax.set(xlabel=label,ylabel="RMSE prueba: media ± DE")
    fig.tight_layout();fig.savefig(out/"comparison.png",dpi=150);plt.close(fig)
    table="| Cadencia | Memoria | RMSE prueba ± DE | Residuo RMS | Rango medio |\n|---:|---:|---:|---:|---:|\n"
    for name,(f,m) in VARIANTS.items():
        r=summary[name];table+=f"| {f} | {m} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} | {r['rank']['mean']:.2f} |\n"
    text="# PCA: frecuencia de observación y duración de memoria\n\nSe comparan dos controles con candidatos fijos, PCA + media al 95%, QR directo y 128 puntos activos completados con Sobol. Se mantiene la pérdida histórica datos+PDE+IC, red, semillas, Adam, 100 pasos uniformes y 10000 de continuación, con reconstrucción cada 100 pasos.\n\n**Frecuencia:** observar cada 5, 10 o 20 pasos sobre los últimos 100 pasos. Cambia la densidad de observaciones, manteniendo el horizonte. **Memoria:** conservar los últimos 100, 200 o 400 pasos observando cada 5. Cambia el horizonte, manteniendo la cadencia. Los cambios del rango efectivo son parte de la respuesta del método adaptativo. No se conserva artificialmente la dimensión de la base.\n\nLa primera selección usa la misma historia común del warmup. Después las historias contienen únicamente snapshots de continuación; las ventanas largas se llenan gradualmente y están completas desde los pasos totales 300 y 500. Esto evita modificar el estado inicial para introducir una historia larga. Durante ese arranque no se puede atribuir la diferencia exclusivamente a un horizonte completo. El caso memoria 100/cadencia 5 se reutiliza del notebook 19.\n\nEn todos los casos cada fila conserva sus coordenadas, los snapshots tienen signo y se evalúan después de Adam. Una ventana de 100 con cadencia 5 contiene 20 observaciones desde paso t−95 hasta t; el horizonte nominal se define por el intervalo (t−100,t]. El rango total incluye la media y puede superar 32 en las ventanas largas; QR directo selecciona k puntos y completa con 128−k Sobol. No hay ampliación log-det ni poda. No se afirma ahorro temporal; observar más frecuentemente añade costo.\n\n"+table+"\n\n"+"\n\n".join(f"{r['method']} frente a cadencia 5/memoria 100: diferencia media RMSE {r['mean_difference']:+.5f}, mejora en {r['wins']}/10 semillas." for r in paired)+"\n\n![Comparación](../../results/burgers/memory_cadence/"+c["run_id"]+"/comparison.png)\n\nSe evalúan los modelos terminales en prueba reservada; no se eligen configuraciones por prueba. Las diferencias son exploratorias y deben confirmarse con nuevas semillas. Los monitores de validación, residuo, IC y BC están en las trazas por semilla. La desviación representa dispersión entre semillas.\n\nReproducir: `python scripts/run_collocation_memory.py`. memory_history.csv conserva tiempos reales y estado de llenado de cada ventana, memory_audit.npz sus matrices con signo y coordenadas. manifest.json identifica las referencias reutilizadas. Las figuras están incrustadas como PNG."
    notebook=root/"pca_collocation/notebooks/burgers/20_memory_cadence.ipynb"
    notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
