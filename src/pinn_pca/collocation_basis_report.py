"""Report the fixed-dimensional basis experiment without selecting on test data."""
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import subspace_angles

from .collocation import _save_csv
from .collocation_basis_ablation import LABELS, fixed_basis, read_rows
from .notebook_images import embed_notebook_images


def report(config, root, output):
    tables = {name: [] for name in ("budget_metrics", "metrics", "validation_trajectories", "selection_diagnostics")}
    for seed in config["seeds"]:
        folder = output / f"seed_{seed}"
        for name in tables:
            tables[name].extend(read_rows(folder / f"{name}.csv"))
    for name, rows in tables.items():
        _save_csv(output / f"{name}_by_seed.csv", rows)
    metrics = tables["budget_metrics"]
    methods = list(LABELS)
    guided = methods[1:]
    final_step = config["warmup_steps"] + config["train_steps"]
    def values(method, metric, step=final_step):
        rows = sorted([r for r in metrics if r["method"] == method and int(r["step"]) == step], key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in rows] != sorted(config["seeds"]):
            raise ValueError("Incomplete/duplicated paired metrics")
        result = np.array([float(r[metric]) for r in rows])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite results")
        return result
    def stats(v):
        return dict(mean=float(np.mean(v)), std=float(np.std(v, ddof=1)) if len(v)>1 else 0.)
    summary = {m: {key: stats(values(m, key)) for key in
        ("test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms")} for m in methods}
    paired = []
    for a, b in itertools.combinations(methods, 2):
        for metric in ("test_rmse", "independent_residual_rms"):
            delta = values(b, metric) - values(a, metric)
            paired.append(dict(method=b, reference=a, metric=metric, **stats(delta),
                               better_count=int(np.sum(delta < 0)), seeds=len(delta)))
    _save_csv(output / "paired_differences.csv", paired)
    (output / "summary.json").write_text(json.dumps(dict(methods=summary, paired=paired,
        config=config, selection="orthonormal fixed dimension + QR + logdet + common Sobol", criterion="terminal model at preset budgets"), indent=2), encoding="utf-8")
    # The shared warmup is identical across all methods, including held-out errors.
    for method in guided:
        np.testing.assert_array_equal(values(method, "test_rmse", config["warmup_steps"]),
                                      values("sobol", "test_rmse", config["warmup_steps"]))
    diagnostics = tables["selection_diagnostics"]
    for row in diagnostics:
        n = 0 if row["method"] == "sobol" else config["guided_points"]
        d = 0 if row["method"] == "sobol" else config["basis_dimension"]
        assert int(row["selected_qr_points"]) == n and int(row["actual_dimension"]) == d
        if row["method"] == "pca_mean_qr":
            np.testing.assert_allclose(float(row["mean_projection_fraction"]), 1., atol=1e-10)
    # Audit Sobol against the previous same-budget control where available.
    previous = root / "pca_collocation/results/burgers/ic_long_budget/ic_uniform_domain_10_seeds_10k_v1"
    replay = []
    for seed in config["seeds"]:
        old_path = previous / f"seed_{seed}/validation_trajectories.csv"
        if old_path.exists():
            old = [r for r in read_rows(old_path) if r["method"] == "sobol"]
            new = [r for r in tables["validation_trajectories"] if r["method"] == "sobol" and int(r["seed"]) == seed]
            if [r["step"] for r in old] == [r["step"] for r in new]:
                delta = max(abs(float(a["validation_rmse"])-float(b["validation_rmse"])) for a,b in zip(old,new))
                if delta > 1e-6:
                    raise ValueError(f"Sobol control changed: seed {seed}, delta={delta}")
                replay.append(dict(seed=seed, max_validation_difference=delta))
    (output / "audit_checks.json").write_text(json.dumps(dict(passed=True, same_warmup=True,
        checked_selections=len(diagnostics), equal_basis_dimension=True, fixed_guided_count=True,
        mean_preserved=True, previous_sobol_replay=replay), indent=2), encoding="utf-8")

    plt.rcParams.update({"font.size": 10})
    colors = dict(zip(methods, ["#777777", "#0072B2", "#D55E00", "#009E73"]))
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / (name+".png"), dpi=150)
        fig.savefig(output / (name+".svg"))
        plt.close(fig)
    budgets = config["test_checkpoints"][1:]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for m in methods:
        for ax, metric, title in zip(axes, ["test_rmse", "independent_residual_rms"], ["RMSE de solución (prueba)", "Residuo RMS independiente"]):
            a = np.array([values(m, metric, step) for step in budgets])
            avg, sd = a.mean(axis=1), a.std(axis=1, ddof=1)
            ax.plot(np.array(budgets)-config["warmup_steps"], avg, "o-", label=LABELS[m], color=colors[m])
            ax.fill_between(np.array(budgets)-config["warmup_steps"], np.maximum(avg-sd,0), avg+sd, alpha=.1, color=colors[m])
            ax.set(xlabel="Pasos de continuación", ylabel=title)
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Misma dimensión y presupuesto; media ± desviación entre semillas")
    save(fig, "budget_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in guided:
        for ax, metric in zip(axes, ["test_rmse", "independent_residual_rms"]):
            delta = values(m, metric) - values("sobol", metric)
            ax.plot(config["seeds"], delta, "o-", label=LABELS[m], color=colors[m])
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(xlabel="Semilla", ylabel=f"{metric}: método − Sobol")
    axes[0].legend()
    fig.suptitle("Comparación pareada final: valores negativos favorecen la base")
    save(fig, "paired_differences")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for m in guided:
        rows = [r for r in diagnostics if r["method"] == m]
        steps = sorted(set(int(r["update_step"])-config["update_every"] for r in rows))
        for ax, key, title in zip(axes, ["centered_variance_captured", "mean_projection_fraction"],
                                 ["Fracción de fluctuaciones representada", "Fracción de la media representada"]):
            means = [np.mean([float(r[key]) for r in rows if int(r["update_step"])-config["update_every"] == s]) for s in steps]
            ax.plot(steps, means, label=LABELS[m], color=colors[m])
            ax.set(xlabel="Paso total de selección", ylabel=title, ylim=(0,1.02))
    axes[0].legend()
    save(fig, "basis_representation")

    first = output / f"seed_{config['seeds'][0]}"
    initial = np.load(first / "selection_snapshot.npz")
    h, points = initial["residual_history"], initial["candidates"]
    initial_bases = {m: fixed_basis(h, m, config["basis_dimension"]) for m in guided}
    angles = [{"first": a, "second": b, "principal_angles_degrees": np.rad2deg(subspace_angles(initial_bases[a],initial_bases[b])).tolist()}
              for a,b in itertools.combinations(guided,2)]
    (output / "initial_subspace_angles.json").write_text(json.dumps(angles,indent=2),encoding="utf-8")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    for ax, method in zip(axes, methods):
        selected = initial["selected_"+method]
        n = 0 if method == "sobol" else config["guided_points"]
        ax.scatter(selected[n:,0], selected[n:,1], s=12, color="gray", alpha=.5, label="Sobol")
        if n:
            ax.scatter(selected[:n,0], selected[:n,1], s=18, color=colors[method], label="Base")
        ax.set(xlabel="x", ylabel="t", title=LABELS[method], xlim=(-1,1), ylim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle("Primera selección: todas las bases observan exactamente la misma historia")
    save(fig, "initial_selection")

    fig, axes = plt.subplots(3, config["basis_dimension"], figsize=(13, 8), squeeze=False)
    for i, method in enumerate(guided):
        for j in range(config["basis_dimension"]):
            mode = initial_bases[method][:,j]
            scale = np.max(np.abs(mode))
            ax = axes[i,j]
            ax.scatter(points[:,0], points[:,1], c=mode, s=3, cmap="coolwarm", vmin=-scale, vmax=scale, rasterized=True)
            ax.set(title=f"{LABELS[method]}: {'media' if method=='pca_mean_qr' and j==0 else 'dirección '+str(j+1)}", xlabel="x", ylabel="t")
    fig.suptitle("Bases iniciales ortonormales (signo arbitrario; escala de color por dirección)")
    save(fig, "initial_modes")

    field = np.load(first / "fields.npz")
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.7))
    errors = [np.abs(field[m]-field["reference"]) for m in methods]
    vmax = max(float(a.max()) for a in errors)
    for ax, m, error in zip(axes, methods, errors):
        im = ax.imshow(error.T, origin="lower", extent=[field["x"].min(),field["x"].max(),field["t"].min(),field["t"].max()],
                       aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        ax.set(xlabel="x", ylabel="t", title=LABELS[m])
        fig.colorbar(im, ax=ax)
    fig.suptitle(f"Error absoluto final, semilla {config['seeds'][0]}; malla completa ilustrativa")
    save(fig, "final_errors")

    notebook = root / "pca_collocation/notebooks/burgers" / config["notebook_name"]
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text))
    def figcell(name):
        md(f"![{name}]({os.path.relpath(output / (name+'.png'),notebook.parent).replace(os.sep,'/')})")
    md("# Primera ablación: ¿qué aporta centrar la historia residual?\n\n"
       "Comparamos POD, PCA centrada y PCA + media. Solo cambia la construcción de la base dentro del selector. "
       "Sobol sirve como control. Esta es la línea propia de PCA; no usa la transformación de salida ni la pérdida de la reproducción QR-DEIM del notebook 14.")
    md("## Hipótesis y dimensión comparable\n\n"
       f"Fijamos **{config['basis_dimension']} direcciones totales**, antes de ejecutar: POD toma los modos de R; PCA toma los modos de R centrada "
       "por la media temporal de cada punto; PCA + media reserva una dirección para la media y añade direcciones PCA independientes hasta completar el mismo total. "
       "La media cuenta dentro del presupuesto, nunca se añade como una quinta dirección.\n\n"
       "Todas las bases se ortonormalizan antes del mismo QR con pivoteo y la misma ampliación log-det (ridge 1e-10). "
       "PCA + media conserva la media primero y ortogonaliza sucesivamente los modos PCA; si un modo es dependiente, continúa con el siguiente. "
       "Los componentes resultantes no son todos componentes principales puros. Si no hay suficiente rango numérico, la corrida falla explícitamente; "
       "no inventa direcciones ni rellena silenciosamente. El umbral numérico usa precisión float32 de los snapshots.\n\n"
       "La dimensión 4 es un punto de partida, no un óptimo demostrado. No se selecciona por varianza explicada en esta ablación. "
       "Los campos energy_threshold/max_modes se conservan por compatibilidad del runner pero no intervienen en estas bases. "
       "La ortonormalización común evita confundir orientación/escalamiento de columnas con construcción del subespacio; "
       "por eso tampoco es una reproducción exacta de los antiguos híbridos de rango variable.")
    md("## Todo lo demás permanece fijo\n\n"
       f"Diez semillas emparejadas, red tanh 3×{config['width']}, Adam constante {config['learning_rate']}, "
       f"{config['warmup_steps']} pasos uniformes comunes y {config['train_steps']} de continuación sin reiniciar Adam. "
       f"Cada método guiado usa {config['guided_points']} puntos de la base + {config['batch_size']-config['guided_points']} del mismo prefijo Sobol; "
       f"el control usa {config['batch_size']} Sobol. Se reconstruye el lote cada {config['update_every']} pasos. "
       f"Candidatos fijos: {config['candidate_count']}; historia de {config['history_snapshots']} snapshots. "
       "La historia inicial se recoge durante el warmup; luego entra un snapshot cada 100 pasos y sale el más antiguo.\n\n"
       "Pesos, momentos de Adam, datos supervisados y particiones temporales son comunes por semilla. "
       "Pérdida histórica: datos + residuo + IC; 32 nodos IC distribuidos en el dominio, sin término BC separado. "
       "IC y BC se evalúan globalmente. Las historias son iguales en la primera selección, pero después evolucionan con cada red: "
       "esa retroalimentación es parte del efecto del método, no un cambio de protocolo.\n\n"
       "Error de solución en tiempos de prueba reservados; residuo en 2048 puntos independientes. "
       "Los presupuestos 2100, 5100 y 10100 pasos totales se fijan antes de ejecutar y se evalúa el modelo de ese paso, "
       "sin seleccionar por prueba ni elegir un mejor checkpoint. Tiempos incluyen diagnóstico y tienen orden de ejecución fijo: no prueban aceleración.")
    table = "| Base/control | RMSE prueba: media ± DE | Residuo RMS: media |\n|---|---:|---:|\n"
    for method in methods:
        r = summary[method]
        table += f"| {LABELS[method]} | {r['test_rmse']['mean']:.6f} ± {r['test_rmse']['std']:.6f} | {r['independent_residual_rms']['mean']:.6f} |\n"
    md("## Resultados al presupuesto final\n\n"+table+"\nDE es dispersión entre semillas, no un intervalo de confianza. "
       "Un residuo menor no demuestra por sí solo una solución más precisa. Las diferencias pareadas de todos los pares se guardan en paired_differences.csv.")
    wins = [p for p in paired if p["reference"]=="sobol" and p["metric"]=="test_rmse"]
    md("Frente a Sobol, número de semillas con menor RMSE: " + "; ".join(f"**{LABELS[p['method']]}: {p['better_count']}/{p['seeds']}**" for p in wins)+". "
       "Esta comparación permite valorar las bases a dimensión 4; no permite generalizar a cualquier rango, historia o presupuesto.")
    pair_table = "| Variante | Referencia | Diferencia media RMSE (variante − referencia) | Semillas favorables |\n|---|---|---:|---:|\n"
    for p in paired:
        if p["reference"] != "sobol" and p["metric"] == "test_rmse":
            pair_table += f"| {LABELS[p['method']]} | {LABELS[p['reference']]} | {p['mean']:+.6f} | {p['better_count']}/{p['seeds']} |\n"
    md("### Comparación directa entre bases\n\n"+pair_table+"\nValores negativos favorecen la variante. "
       "Una mejor media con pocas victorias puede estar dominada por unas pocas semillas; no basta para afirmar superioridad consistente.")
    constraints = "| Método | RMSE IC global medio | RMS frontera medio |\n|---|---:|---:|\n"
    for m in methods:
        constraints += f"| {LABELS[m]} | {summary[m]['initial_condition_rmse']['mean']:.6f} | {summary[m]['boundary_rms']['mean']:.6f} |\n"
    md("### Condiciones iniciales y de frontera\n\n"+constraints+"\nEn esta pérdida las condiciones no se imponen por transformación de salida. "
       "Estos monitores ayudan a interpretar por qué reducir el residuo físico puede no reducir el error de solución.")
    figcell("budget_comparison")
    figcell("paired_differences")
    md("## Qué está representando cada base\n\n"
       "Se mide la fracción de norma cuadrada de la historia centrada y de su media recuperada al proyectarlas en la base. "
       "PCA maximiza la primera a dimensión fija; PCA + media garantiza conservar la segunda. "
       "Las curvas posteriores usan historias diferentes porque las redes ya evolucionaron de manera distinta. "
       "Para una comparación puramente geométrica, las siguientes selecciones y modos iniciales sí usan la misma matriz. "
       "Los ángulos entre esos subespacios están en initial_subspace_angles.json.")
    for name in ("basis_representation", "initial_selection", "initial_modes", "final_errors"):
        figcell(name)
    md("## Auditoría y reproducción\n\n"
       f"Se verificaron {len(diagnostics)} selecciones: dimensión igual, 32 puntos guiados, 128 activos únicos, "
       "complemento Sobol común y media conservada en PCA + media. Se verificó igualdad de métricas después del warmup. "
       f"El control Sobol se contrastó además contra el experimento previo de 10000 pasos en {len(replay)} semillas.\n\n"
       "```bash\npython scripts/run_collocation_basis_ablation.py\npython scripts/run_collocation_basis_ablation.py --report-only\n```\n\n"
       "Configuración: pca_collocation/configs/burgers_basis_ablation.json. "
       f"Resultados: {output.relative_to(root).as_posix()}. "
       "Se guardan métricas, trazas, dimensión real, proyecciones, historial de puntos activos y campos ilustrativos. "
       "Las semillas completas se reutilizan; cambiar configuración requiere otro run_id. Las figuras están incrustadas como PNG.\n\n"
       "El siguiente control será estudiar dimensión/rango manteniendo este mismo selector, sin cambiar simultáneamente historia, candidatos o poda.")
    notebook.parent.mkdir(parents=True,exist_ok=True)
    notebook.write_text(json.dumps(dict(cells=cells,metadata={"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"}},nbformat=4,nbformat_minor=5),indent=2,ensure_ascii=False),encoding="utf-8")
    embed_notebook_images(notebook,root)
    print(json.dumps(dict(methods=summary,paired=paired),indent=2),flush=True)
