"""Compare direct rank-sized QR with 32-point QR/log-det reconstruction."""
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from pinn_pca.collocation import _run_collocation_seed,_save_csv
from pinn_pca.collocation_partial import Recorder
from pinn_pca.collocation_basis_ablation import read_rows
from pinn_pca.notebook_images import embed_notebook_images

c=json.loads((ROOT/"pca_collocation/configs/burgers_partial_update.json").read_text())
c.update(run_id="pca95_direct_qr_10_seeds_10k_v1")
out=ROOT/"pca_collocation/results/burgers/direct_qr"/c["run_id"]
out.mkdir(parents=True,exist_ok=True)
(out/"config.json").write_text(json.dumps(c,indent=2))
rows=[];counts=[]
for seed in c["seeds"]:
    folder=out/f"seed_{seed}"
    spec=dict(base="pca_mean_qr",label="PCA + media: QR directo",adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
    if not (folder/"audit.json").exists():
        print(f"RUN direct QR seed={seed}",flush=True)
        _run_collocation_seed(dict(c,seed=seed,method_specs={"direct_qr":spec},common_selection_seed=True,skip_seed_plots=True),ROOT,folder,observer=Recorder(c,folder,"direct_qr"))
        print(f"DONE direct QR seed={seed}",flush=True)
    rows.extend(read_rows(folder/"metrics.csv"))
    counts.extend(read_rows(folder/"selection_diagnostics.csv"))
    prior=ROOT/"pca_collocation/results/burgers/partial_update/pca95_partial_10_seeds_10k_v1"
    for method in ("pca_full","sobol"):
        rows.extend(read_rows(prior/method/f"seed_{seed}"/"metrics.csv"))
_save_csv(out/"metrics.csv",rows);_save_csv(out/"selection_counts.csv",counts)
summary={}
for method in ("direct_qr","pca_full","sobol"):
    group=[r for r in rows if r["method"]==method]
    summary[method]={key:dict(mean=float(np.mean([float(r[key]) for r in group])),std=float(np.std([float(r[key]) for r in group],ddof=1))) for key in ("test_rmse","independent_residual_rms","initial_condition_rmse")}
deltas=[]
for seed in c["seeds"]:
    get=lambda m:float(next(r for r in rows if r["method"]==m and int(r["seed"])==seed)["test_rmse"])
    deltas.append(get("direct_qr")-get("pca_full"))
summary["paired"]=dict(mean_difference=float(np.mean(deltas)),wins=int(np.sum(np.array(deltas)<0)))
k=[int(r["selected_qr_points"]) for r in counts]
assert all(int(r["retained_modes"])==int(r["selected_qr_points"]) for r in counts)
summary["k"]=dict(min=min(k),max=max(k),mean=float(np.mean(k)))
(out/"summary.json").write_text(json.dumps(summary,indent=2))
fig,ax=plt.subplots(figsize=(8,4))
for seed in c["seeds"]:
    group=[r for r in counts if int(r["seed"])==seed]
    ax.plot([int(r["update_step"])-100 for r in group],[int(r["selected_qr_points"]) for r in group],alpha=.5)
ax.set(xlabel="Paso de selección",ylabel="Puntos guiados = dimensión total",title="QR directo: rango adaptativo por semilla")
fig.tight_layout();fig.savefig(out/"rank.png",dpi=150);plt.close(fig)
table="| Método | RMSE medio ± DE | Residuo RMS |\n|---|---:|---:|\n"
for m,label in (("direct_qr","QR directo"),("pca_full","32 por QR + log-det"),("sobol","Sobol")):
    r=summary[m];table+=f"| {label} | {r['test_rmse']['mean']:.5f} ± {r['test_rmse']['std']:.5f} | {r['independent_residual_rms']['mean']:.5f} |\n"
text="# PCA: QR directo frente a ampliación a 32 puntos\n\nDiez semillas, 100 pasos uniformes + 10000 de continuación, reconstrucción cada 100 pasos, PCA centrada con media preservada al 95%. La dimensión total incluye la media independiente. Se conserva la pérdida histórica datos+PDE+IC, candidatos fijos y ventana de 20 snapshots.\n\nLa variante directa toma los k pivotes de QR sobre la base ortonormal transpuesta y completa con 128−k puntos Sobol. La referencia toma 32 puntos mediante QR + log-det y completa con 96 Sobol. Ambos usan el mismo prefijo Sobol por semilla y paso. Las historias divergen después de la primera selección por la retroalimentación del entrenamiento. No hay poda en este control. Cambian juntos la ampliación y el número de puntos guiados: no puede atribuirse la diferencia únicamente a log-det.\n\nLas referencias PCA de 32 puntos y Sobol se reutilizan del notebook 17; los parámetros de entrenamiento son comunes. Se evalúa el modelo terminal en tiempos de prueba reservados, sin elegir por prueba.\n\n"+table+f"\nQR directo mejora frente a 32 puntos en {summary['paired']['wins']}/10 semillas; diferencia media RMSE {summary['paired']['mean_difference']:+.5f}. Rango total observado: {min(k)}–{max(k)}, media {np.mean(k):.2f}.\n\n![Rango](../../../results/burgers/direct_qr/{c['run_id']}/rank.png)\n\nReproducir: `python scripts/run_collocation_direct_qr.py`. Corridas completas se reutilizan. Los tiempos no se interpretan como aceleración; esta implementación aún evalúa el diseño ampliado antes de conservar los pivotes directos. El control estudia calidad y presupuesto de puntos."
# Image path is relative to notebooks/burgers (two levels up to pca_collocation).
text=text.replace("../../../results/","../../results/")
notebook=ROOT/"pca_collocation/notebooks/burgers/18_direct_qr.ipynb"
notebook.write_text(json.dumps(dict(cells=[dict(cell_type="markdown",metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding="utf-8")
embed_notebook_images(notebook,ROOT)
print(json.dumps(summary,indent=2),flush=True)
