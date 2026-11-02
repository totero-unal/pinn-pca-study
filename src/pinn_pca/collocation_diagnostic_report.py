"""Plots and a self-contained explanatory notebook for the Burgers diagnosis."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
import numpy as np

from .notebook_images import embed_notebook_images

METHODS = ("sobol", "pod_qr_32", "pca_mean_qr_64")
LABELS = {"sobol": "Sobol", "pod_qr_32": "POD: 32 puntos", "pca_mean_qr_64": "PCA + media: 64 puntos"}
COLORS = dict(zip(METHODS, ("#2678b2", "#d34442", "#8055a0")))


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(value):
    return float(value) if value not in (None, "") else float("nan")


def stats(values):
    values = np.asarray([number(v) for v in values], dtype=float)
    values = values[np.isfinite(values)]
    return {"mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": int(len(values))}


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_diagnostic_report(config, root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    names = ("metrics", "validation_trajectories", "loss_gradient_probes", "coverage", "regional_test_errors")
    tables = {name: [] for name in names}
    for seed in config["seeds"]:
        for name in names:
            tables[name].extend(read_csv(output / f"seed_{seed}/{name}.csv"))
    for name, rows in tables.items():
        write_csv(output / f"{name}_by_seed.csv", rows)
    reference = root / "pca_collocation/results/burgers/collocation_selection" / config["reference_run_id"]
    old_traces = {(row["seed"], row["method"], row["step"]): row for row in read_csv(reference / "validation_trajectories_by_seed.csv")}
    trajectory_replay = []
    for row in tables["validation_trajectories"]:
        old = old_traces[(row["seed"], row["method"], row["step"])]
        for key in ("validation_rmse", "independent_residual_rms", "initial_condition_rmse", "boundary_rms"):
            difference = float(row[key]) - float(old[key])
            if abs(difference) > config["replay_tolerance"]:
                raise ValueError(f"Diagnostic replay changed checkpoint {row['step']} for seed {row['seed']}, method {row['method']}: {key}")
            trajectory_replay.append({"seed": row["seed"], "method": row["method"], "step": row["step"], "metric": key, "difference": difference})
    write_csv(output / "trajectory_replay_comparison.csv", trajectory_replay)
    last = config["warmup_steps"] + config["train_steps"]
    summary_rows = []
    for method in METHODS:
        metrics = [r for r in tables["metrics"] if r["method"] == method]
        probes = [r for r in tables["loss_gradient_probes"] if r["method"] == method and r["probe"] == "common"]
        final = [r for r in probes if int(r["step"]) == last]
        final_coverage = [r for r in tables["coverage"] if r["method"] == method and int(r["selection_step"]) == last - config["update_every"]]
        fractions, gaps = [], []
        for seed in config["seeds"]:
            trajectory = [r for r in probes if int(r["seed"]) == seed and int(r["step"]) > config["warmup_steps"] and np.isfinite(number(r["cos_physics_data_initial"]))]
            fractions.append(np.mean([number(r["cos_physics_data_initial"]) < 0 for r in trajectory]))
            paired = [r for r in tables["loss_gradient_probes"] if r["method"] == method and int(r["seed"]) == seed and int(r["step"]) == last]
            common = next(number(r["loss_physics"]) for r in paired if r["probe"] == "common")
            active = next(number(r["loss_physics"]) for r in paired if r["probe"] == "active")
            gaps.append(common / active if active > 0 else np.nan)
        summary_rows.append({
            "method": method, "label": LABELS[method],
            "metrics": {key: stats([r[key] for r in metrics]) for key in ("test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms")},
            "final_common_probe": {key: stats([r[key] for r in final]) for key in ("loss_data", "loss_physics", "loss_initial", "grad_data", "grad_physics", "grad_initial", "grad_data_initial", "cos_physics_data_initial", "physics_to_data_initial_grad_ratio", "gradient_cancellation_ratio")},
            "fraction_checkpoints_with_common_gradient_conflict": stats(fractions),
            "final_common_to_active_physics_loss_ratio": stats(gaps),
            "final_coverage": {key: stats([r[key] for r in final_coverage]) for key in ("front_fraction", "occupied_cell_fraction", "independent_nearest_distance_q95")},
            "regional_test_rmse": {region: stats([r["rmse"] for r in tables["regional_test_errors"] if r["method"] == method and int(r["step"]) == last and r["region"] == region]) for region in ("all_test", "central_late_test", "outside_window_test")},
        })
    protocol = json.loads((output / f"seed_{config['representative_seed']}/diagnostic_protocol.json").read_text(encoding="utf-8"))
    replay = read_csv(output / "replay_comparison.csv")
    summary = {"configuration": config, "protocol": protocol, "methods": summary_rows,
               "max_replay_difference": max(abs(float(r["difference"])) for r in replay),
               "max_trajectory_replay_difference": max(abs(r["difference"]) for r in trajectory_replay),
               "interpretation_limit": "Observational replay: Euclidean gradients on frozen probes do not identify a causal effect or the effective Adam step. No boundary penalty is present; initial-condition supervision covers only the first spatial nodes. Full-field maps include fit and held-out times; regional error tables use held-out test times only."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    figures = _plots(config, output, tables)
    _notebook(root, output, summary, figures)
    return summary


def _save(fig, output, name):
    for extension in ("png", "svg"):
        fig.savefig(output / f"{name}.{extension}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return name + ".png"


def _lines(ax, rows, key, step_key="step", probe=None, logarithmic=False):
    for method in METHODS:
        selected = [r for r in rows if r["method"] == method and (probe is None or r.get("probe") == probe)]
        steps = sorted({int(r[step_key]) for r in selected})
        values = [stats([r[key] for r in selected if int(r[step_key]) == step]) for step in steps]
        mean = np.asarray([v["mean"] if v["mean"] is not None else np.nan for v in values])
        deviation = np.asarray([v["std"] for v in values])
        ax.plot(steps, mean, color=COLORS[method], label=LABELS[method])
        if logarithmic:
            # Positive empirical quantiles avoid arbitrary clipping when mean-DE < 0.
            bands = []
            for step in steps:
                sample = np.asarray([number(r[key]) for r in selected if int(r[step_key]) == step])
                sample = sample[np.isfinite(sample) & (sample > 0)]
                bands.append(np.quantile(sample, [.16, .84]) if len(sample) else [np.nan, np.nan])
            lower, upper = np.asarray(bands).T
        else:
            lower, upper = mean - deviation, mean + deviation
        ax.fill_between(steps, lower, upper, color=COLORS[method], alpha=.12)
    if logarithmic:
        ax.set_yscale("log")
    ax.set_xlabel("Pasos totales")
    ax.grid(alpha=.2)


def _plots(config, output, tables):
    figures = {}
    seed = config["representative_seed"]
    seed_dir = output / f"seed_{seed}"
    with np.load(seed_dir / "fields.npz", allow_pickle=False) as stored:
        fields = {key: stored[key] for key in stored.files}
    x, t, truth = fields["x"], fields["t"], fields["reference"]
    extent = (x[0], x[-1], t[0], t[-1])
    fig, ax = plt.subplots(figsize=(7, 4))
    image = ax.imshow(truth.T, origin="lower", extent=extent, aspect="auto", cmap="RdBu_r", vmin=-np.max(np.abs(truth)), vmax=np.max(np.abs(truth)))
    ax.set(xlabel="x", ylabel="t", title="Referencia de Burgers (datos del repositorio)")
    fig.colorbar(image, ax=ax, label="u")
    figures["reference"] = _save(fig, output, "reference_solution")
    steps = config["field_checkpoints"]
    u_limit = max(np.max(np.abs(fields[f"{method}__{step}__u"])) for method in METHODS for step in steps)
    u_limit = max(u_limit, np.max(np.abs(truth)))
    error_limit = max(np.max(np.abs(fields[f"{method}__{step}__error"])) for method in METHODS for step in steps)
    residual_limit = max(np.max(np.abs(fields[f"{method}__{step}__r"])) for method in METHODS for step in steps)
    for step in steps:
        fig, axes = plt.subplots(3, 3, figsize=(11.5, 10), constrained_layout=True, sharex=True, sharey=True)
        for column, method in enumerate(METHODS):
            for row, (field, label) in enumerate((("u", "u predicha"), ("error", "Error absoluto"), ("r", "|Residuo AD|"))):
                values = fields[f"{method}__{step}__{field}"]
                if field == "u":
                    options = dict(cmap="RdBu_r", vmin=-u_limit, vmax=u_limit)
                elif field == "error":
                    values = np.abs(values)
                    options = dict(cmap="magma", vmin=0, vmax=error_limit)
                else:
                    values = np.maximum(np.abs(values), 1e-5)
                    options = dict(cmap="viridis", norm=LogNorm(vmin=1e-5, vmax=max(residual_limit, 1e-4)))
                image = axes[row, column].imshow(values.T, origin="lower", extent=extent, aspect="auto", **options)
                if column == 0:
                    axes[row, column].set_ylabel(label + "\nt")
                if row == 0:
                    axes[row, column].set_title(LABELS[method])
                if row == 2:
                    axes[row, column].set_xlabel("x")
                width = config["front_abs_x_max"]
                axes[row, column].add_patch(Rectangle((-width, config["front_time_min"]), 2 * width, t[-1] - config["front_time_min"], fill=False, linestyle="--", linewidth=.8, edgecolor="black"))
                if column == 2:
                    fig.colorbar(image, ax=axes[row, :].tolist(), shrink=.8, label=label)
        fig.suptitle(f"Semilla predefinida {seed}, paso {step}; escalas compartidas entre métodos y pasos")
        figures[f"fields_{step}"] = _save(fig, output, f"fields_step_{step}")
    final = config["warmup_steps"] + config["train_steps"]
    fig, axes = plt.subplots(2, len(config["profile_times"]), figsize=(12, 6.5), squeeze=False)
    for column, requested in enumerate(config["profile_times"]):
        ti = int(np.argmin(np.abs(t - requested)))
        for row in range(2):
            ax = axes[row, column]
            ax.plot(x, truth[:, ti], color="black", linewidth=2, label="Referencia")
            for method in METHODS:
                ax.plot(x, fields[f"{method}__{final}__u"][:, ti], color=COLORS[method], label=LABELS[method])
            if row:
                ax.set_xlim(-config["profile_zoom_abs_x"], config["profile_zoom_abs_x"])
            elif ti == 0:
                ax.axvspan(fields["initial_x"].min(), fields["initial_x"].max(), color="gold", alpha=.3, label="Zona IC entrenada")
            ax.set(title=f"t={t[ti]:.2f}" + ("; zoom central" if row else ""), xlabel="x", ylabel="u")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"Perfiles finales, semilla {seed}; la franja amarilla muestra la supervisión IC")
    fig.tight_layout(rect=(0, .05, 1, .95))
    figures["profiles"] = _save(fig, output, "solution_profiles")

    points = read_csv(seed_dir / "active_points.csv")
    selection_steps = config["selection_plot_steps"]
    fig, axes = plt.subplots(3, len(selection_steps), figsize=(13, 9), sharex=True, sharey=True, squeeze=False)
    for row, method in enumerate(METHODS):
        for column, step in enumerate(selection_steps):
            ax = axes[row, column]
            selected = [r for r in points if r["method"] == method and int(r["selection_step"]) == step]
            for source, color, marker, size in (("sobol", "#888888", ".", 10), ("basis", COLORS[method], "*", 22)):
                group = [r for r in selected if r["source"] == source]
                if group:
                    ax.scatter([float(r["x"]) for r in group], [float(r["t"]) for r in group], color=color, marker=marker, s=size, label="Modos" if source == "basis" else "Sobol")
            ax.add_patch(Rectangle((-config["front_abs_x_max"], config["front_time_min"]), 2 * config["front_abs_x_max"], t[-1] - config["front_time_min"], fill=False, linestyle="--", edgecolor="black", linewidth=.8))
            ax.set(xlim=(x.min(), x.max()), ylim=(t.min(), t.max()))
            if row == 0:
                ax.set_title(f"Selección en paso {step}")
            if column == 0:
                ax.set_ylabel(LABELS[method] + "\nt")
            if row == 2:
                ax.set_xlabel("x")
    axes[-1, -1].legend(fontsize=8)
    fig.suptitle(f"128 puntos activos por actualización, semilla {seed}; estrellas: modos, gris: Sobol")
    fig.tight_layout()
    figures["selection"] = _save(fig, output, "selected_points_evolution")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    coverage_keys = (("front_fraction", "Fracción del lote en ventana central tardía"),
                     ("occupied_cell_fraction", "Fracción de celdas ocupadas (8 × 8)"),
                     ("independent_nearest_distance_q95", "Distancia al punto activo más próximo: percentil 95"),
                     ("basis_front_fraction", "Fracción de puntos de modos en ventana central tardía"))
    for ax, (key, label) in zip(axes.flat, coverage_keys):
        _lines(ax, tables["coverage"], key, step_key="selection_step")
        ax.set(ylabel=label, xlabel="Paso de selección")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Cobertura del dominio: media ± DE entre semillas; distancias en coordenadas [0,1]²")
    fig.tight_layout()
    figures["coverage"] = _save(fig, output, "coverage_evolution")

    for prefix, title in (("loss", "Pérdidas en sondas fijas"), ("grad", "Normas euclídeas de gradiente por término")):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for row, probe in enumerate(("common", "active")):
            for column, (component, label) in enumerate((("data", "Datos"), ("physics", "Física"), ("initial", "Condición inicial"))):
                ax = axes[row, column]
                _lines(ax, tables["loss_gradient_probes"], f"{prefix}_{component}", probe=probe, logarithmic=True)
                ax.set_title(label)
                if column == 0:
                    ax.set_ylabel(("Sonda común" if probe == "common" else "Puntos activos") + "\n" + ("MSE" if prefix == "loss" else "Norma L2"))
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(title + ": media y banda percentil 16–84 entre semillas; ejes logarítmicos")
        fig.tight_layout()
        figures[prefix] = _save(fig, output, prefix + "_components")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, probe in zip(axes, ("common", "active")):
        _lines(ax, tables["loss_gradient_probes"], "cos_physics_data_initial", probe=probe)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title="Sonda común" if probe == "common" else "Puntos activos", ylabel="cos(∇L física, ∇(L datos + L IC))", ylim=(-1.05, 1.05))
    axes[0].legend(fontsize=8)
    fig.suptitle("Conflicto local de gradientes: coseno negativo indica oposición")
    fig.tight_layout()
    figures["cosines"] = _save(fig, output, "gradient_conflict")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, key, label in zip(axes.flat,
                             ("validation_rmse", "independent_residual_rms", "initial_condition_rmse", "boundary_rms"),
                             ("RMSE solución: validación", "Residuo RMS: 2048 puntos independientes", "RMSE IC: todos los nodos espaciales", "RMS frontera: tiempos de validación")):
        _lines(ax, tables["validation_trajectories"], key)
        ax.set_ylabel(label)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Monitores fuera de la selección: media ± DE entre semillas")
    fig.tight_layout()
    figures["monitors"] = _save(fig, output, "independent_monitors")
    return figures


def _notebook(root, output, summary, figures):
    config, protocol = summary["configuration"], summary["protocol"]
    seed, last = config["representative_seed"], config["warmup_steps"] + config["train_steps"]
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text}
    asset = "../../" + output.relative_to(root / "pca_collocation").as_posix() + "/"
    def image(key, caption):
        return md(f"![{caption}]({asset}{figures[key]})")
    def fmt(stat):
        return f"{stat['mean']:.4f} ± {stat['std']:.4f}" if stat["mean"] is not None else "No definido"
    table = "| Método | RMSE prueba | Residuo RMS independiente | RMSE IC global | RMS frontera |\n|---|---:|---:|---:|---:|\n"
    regional = "| Método | RMSE en ventana central tardía | RMSE fuera de la ventana |\n|---|---:|---:|\n"
    gradients = "| Método | Norma ∇L física / norma ∇(L datos + L IC) | Coseno física vs datos + IC | Fracción de controles con conflicto |\n|---|---:|---:|---:|\n"
    probe_table = "| Método | L datos (común) | L física (común) | L IC en nodos entrenados | L física común / activa |\n|---|---:|---:|---:|\n"
    for row in summary["methods"]:
        table += f"| {row['label']} | " + " | ".join(fmt(row["metrics"][key]) for key in ("test_rmse", "independent_residual_rms", "initial_condition_rmse", "boundary_rms")) + " |\n"
        regional += f"| {row['label']} | {fmt(row['regional_test_rmse']['central_late_test'])} | {fmt(row['regional_test_rmse']['outside_window_test'])} |\n"
        gradients += f"| {row['label']} | {fmt(row['final_common_probe']['physics_to_data_initial_grad_ratio'])} | {fmt(row['final_common_probe']['cos_physics_data_initial'])} | {fmt(row['fraction_checkpoints_with_common_gradient_conflict'])} |\n"
        probe_table += f"| {row['label']} | " + " | ".join(fmt(row["final_common_probe"][key]) for key in ("loss_data", "loss_physics", "loss_initial")) + f" | {fmt(row['final_common_to_active_physics_loss_ratio'])} |\n"
    observations = "| Método | Lote final en ventana central tardía | RMSE de prueba en esa ventana | Controles con oposición de gradientes |\n|---|---:|---:|---:|\n"
    for row in summary["methods"]:
        observations += (f"| {row['label']} | {100*row['final_coverage']['front_fraction']['mean']:.1f}% "
                         f"| {row['regional_test_rmse']['central_late_test']['mean']:.4f} "
                         f"| {100*row['fraction_checkpoints_with_common_gradient_conflict']['mean']:.1f}% |\n")
    gradient_ratios = "; ".join(f"{row['label']}: {row['final_common_probe']['physics_to_data_initial_grad_ratio']['mean']:.2f}" for row in summary["methods"])
    distance_values = "; ".join(f"{row['label']}: {row['final_coverage']['independent_nearest_distance_q95']['mean']:.4f}" for row in summary["methods"])
    cells = [md("# Diagnóstico de Burgers: precisión, residuo y selección de puntos\n\n"
                "**Pregunta:** ¿por qué POD/PCA reduce el residuo, pero empeora la solución frente a Sobol? Este notebook observa el entrenamiento y distingue mediciones de hipótesis. No cambia la pérdida ni el selector.\n\n"
                f"Se repiten Sobol, POD con 32 puntos y PCA centrado + media con 64 puntos, en {len(config['seeds'])} semillas, con {config['warmup_steps']} pasos de calentamiento y {config['train_steps']} de continuación. El lote físico tiene 128 puntos y se actualiza cada 100 pasos. Los puntos guiados usan QR inicial y ampliación log-det; el complemento es Sobol. La mayor diferencia absoluta frente a las cinco métricas finales guardadas anteriormente es **{summary['max_replay_difference']:.2g}**. La instrumentación reproduce el entrenamiento de referencia.\n\n"
                f"Los mapas corresponden a la semilla **{seed}**, fijada en la configuración antes de correr. Las curvas y tablas agregan todas las semillas; las bandas muestran desviación estándar entre semillas, salvo las curvas logarítmicas de pérdidas y normas, que usan percentiles 16–84 empíricos. No son intervalos de confianza. La mayor diferencia entre los monitores de las trayectorias repetidas y las originales es **{summary['max_trajectory_replay_difference']:.2g}**."),
             md("## 1. La pérdida que realmente se está entrenando\n\n"
                "\\[L=L_{datos}+L_{física}+L_{IC},\\qquad L_{física}=\\frac1{128}\\sum r_\\theta(x_i,t_i)^2.\\]\n\n"
                "Los tres pesos son 1. Datos usa minibatches de los tiempos de ajuste; IC compara u(x,0) en el subconjunto espacial configurado. **No hay una penalización separada de frontera en esta pérdida.** Los datos supervisados pueden incluir nodos de frontera; además, esta se monitoriza para evaluar el modelo.\n\n"
                f"**Hallazgo del protocolo:** `initial_ids = arange(initial_condition_points)` impone IC en los primeros **{protocol['initial_points']} de {protocol['spatial_grid_points']} nodos**, entre x={protocol['initial_x_min']:.4f} y x={protocol['initial_x_max']:.4f}. Su intervalo cubre aproximadamente **{100*protocol['initial_span_fraction']:.1f}%** del ancho espacial. Una L_IC pequeña en esa zona no demuestra que IC sea correcta en todo el dominio. Los datos supervisados también pueden contener t=0, pero su muestreo cambia entre minibatches.\n\n"
                "Este diagnóstico conserva ese protocolo para reproducir las corridas previas. La franja amarilla del perfil t=0 muestra los nodos de IC impuestos explícitamente."),
             md("## 2. Qué datos intervienen en cada medición\n\n"
                "| Medición | Puntos y uso |\n|---|---|\n"
                "| Sonda común | 128 puntos físicos Sobol fijos, distintos de candidatos y evaluación; mismos puntos para todos los métodos de una semilla. |\n"
                "| Sonda activa | Los mismos datos e IC de la sonda común; solo se sustituyen las coordenadas físicas por el lote activo del método. |\n"
                "| Datos de las sondas | Un minibatch supervisado congelado de tiempos de ajuste, igual en todos los métodos y controles; no se añaden datos de prueba al entrenamiento. |\n"
                "| Error de solución y regiones | Tiempos de prueba reservados por semilla; el mismo conjunto para los tres métodos. |\n"
                "| Residuo independiente | 2048 puntos Sobol fuera del candidato y de la selección. |\n"
                "| Mapas y perfiles | Malla de referencia con tiempos de ajuste, validación y prueba; visualización exploratoria, no métrica de generalización. |\n\n"
                f"Las sondas se calculan después de cada control de entrenamiento, incluido el calentamiento en paso {config['warmup_steps']}. La sonda activa usa el lote del intervalo recién terminado; se guarda `active_selection_step`. Las pérdidas de las sondas no son el último minibatch aleatorio del optimizador."),
             md("## 3. Resultado global y localización del error\n\n" + table + "\nTodos los valores son media ± DE entre semillas. El residuo es RMS, las pérdidas físicas posteriores son MSE; no deben confundirse.\n\n"
                f"Se fija una ventana de inspección **|x| ≤ {config['front_abs_x_max']} y t ≥ {config['front_time_min']}** para examinar el frente central tardío. No es un detector aprendido de shock y no interviene en el selector. La región exterior es su complemento, incluidos los tiempos anteriores.\n\n" + regional),
             image("reference", "Solución de referencia"),
             image(f"fields_{last}", "Solución predicha, error absoluto y residuo final"),
             md("Las escalas de color son comunes entre métodos y checkpoints; la escala de |r| es logarítmica y solo su dibujo se recorta por debajo de 1e−5. El residuo de las redes se calcula por autodiferenciación; no se obtiene diferenciando numéricamente los datos. La malla finita puede subrepresentar un frente muy estrecho; las métricas independientes permanecen en los CSV."),
             image("profiles", "Perfiles finales y zoom alrededor del frente"),
             md("## 4. Qué términos de la pérdida bajan y dónde\n\n" + probe_table + "\nValores al último checkpoint. Una razón física común/activa mayor que 1 significa que el residuo MSE es mayor en la sonda común que en los puntos activos. Es un contraste entre dos muestras de 128 puntos, no una prueba de sobreajuste ni un reemplazo del residuo independiente de 2048 puntos.\n\nLas filas de datos e IC coinciden entre sondas por construcción: sus puntos son idénticos. La física es la única parte que cambia de ubicaciones. En las curvas logarítmicas, la línea es la media aritmética y la banda muestra percentiles 16–84 de los valores positivos observados; se evita recortar una banda media−DE negativa a un número arbitrario cercano a cero."),
             image("loss", "Pérdidas separadas en las sondas común y activa"),
             md("## 5. Tamaño y oposición de los gradientes\n\n"
                "Se calcula el gradiente de cada término respecto a todos los parámetros de la misma red, con los pesos de entrenamiento 1. La norma L2 mide su tamaño en coordenadas de parámetros; una pérdida pequeña puede tener un gradiente grande.\n\n"
                "\\[c=\\frac{\\nabla L_{física}^{T}\\nabla(L_{datos}+L_{IC})}{\\|\\nabla L_{física}\\|\\,\\|\\nabla(L_{datos}+L_{IC})\\|}.\\]\n\n"
                "c < 0 indica oposición local entre esos gradientes; c > 0 indica alineación local. El cociente de normas compara la física con el gradiente conjunto de datos e IC. La frecuencia de conflicto se calcula por semilla sobre controles posteriores al calentamiento y luego se resume entre semillas. No se tratan los controles temporales como réplicas independientes.\n\n" + gradients + "\nLa tabla usa la **sonda común**, para evitar atribuir al modelo un cambio causado solamente por cambiar los puntos. Son gradientes euclídeos sin el precondicionamiento ni los momentos de Adam: no equivalen a la contribución efectiva de cada término al paso del optimizador."),
             image("grad", "Normas de gradiente por término"), image("cosines", "Conflicto entre física y supervisión"),
             md("## 6. Dónde se concentran los puntos\n\n"
                "Las estrellas son puntos del conjunto candidato elegidos por los modos; los puntos grises son el complemento Sobol. Todos los lotes contienen 128 puntos. El paso del título es el momento de selección: ese lote se usa en las siguientes 100 actualizaciones. El rectángulo discontinuo marca la ventana central tardía definida anteriormente."),
             image("selection", "Evolución de la selección de puntos"),
             md("La cobertura se resume con fracción del lote en la ventana, fracción de celdas de una partición 8×8 que contienen puntos y percentil 95 de la distancia desde los puntos independientes al punto activo más cercano. Las distancias usan coordenadas x,t normalizadas por los anchos del dominio; menor distancia indica mejor cobertura según esa métrica, no mejor precisión garantizada. La fracción de puntos de modos no está definida para Sobol puro y su curva se omite."),
             image("coverage", "Cobertura durante el entrenamiento"),
             md("## 7. Monitores independientes y evolución de los campos\n\n"
                "El error IC de esta figura usa **todos** los nodos espaciales; la L_IC de la sección 4 usa solo los primeros nodos entrenados. La frontera se evalúa en tiempos de validación y permanece fuera de la pérdida explícita. Estas diferencias permiten identificar mejoras locales que no se trasladan al resto del dominio."),
             image("monitors", "Monitores independientes del selector")]
    for step in config["field_checkpoints"]:
        if step != last:
            cells += [md(f"### Campos en paso {step}"), image(f"fields_{step}", f"Campos y errores en paso {step}")]
    cells += [md("## 8. Qué permite concluir este diagnóstico\n\n"
                 "**Observaciones de esta corrida:**\n\n" + observations + "\n"
                 "POD/PCA concentra una fracción mayor de sus puntos en la ventana central tardía; eso no se traduce en menor error medio de solución en esa ventana. La oposición de gradientes aparece también con Sobol: su sola presencia no explica las diferencias entre métodos.\n\n"
                 f"La media del cociente final de normas física/(datos+IC) en la sonda común es {gradient_ratios}. Su dispersión está en la tabla de gradientes. Estos datos no respaldan reducir la explicación a que el gradiente físico sea siempre mayor con POD/PCA.\n\n"
                 f"El percentil 95 de la distancia de cobertura, promediado entre semillas, es {distance_values}. La ocupación 8×8 alcanza el 100% en el último lote de los tres métodos; esa partición gruesa no distingue su cobertura.\n\n"
                 "**Hipótesis por contrastar:** cambiar la distribución de puntos físicos puede cambiar el equilibrio entre el ajuste supervisado y el residuo. Un coseno negativo o una norma grande apoyan un posible mecanismo local, pero no demuestran que ese mecanismo cause la diferencia final. La cobertura parcial de IC es además una limitación concreta del protocolo.\n\n"
                 "Para atribuir causalidad hace falta una ablación que cambie una sola cosa. El control siguiente más directo es distribuir los mismos 32 nodos de IC por todo el dominio, manteniendo métodos, pesos, semillas y presupuesto. Después podría estudiarse el peso físico, con una selección definida por validación. El subconjunto de prueba y los mapas no deben usarse para escoger esos pesos."),
              md("## 9. Archivos y reproducción\n\n"
                 "Cada `seed_<semilla>/` conserva `fields.npz`, `parameters.npz`, `active_points.csv`, `coverage.csv`, `loss_gradient_probes.csv`, `regional_test_errors.csv` y `diagnostic_protocol.json`, además de las métricas y trazas del entrenador. Los parámetros se guardan por capa en los checkpoints de campos, sin pickle. Los CSV globales agregan las semillas conservando sus identificadores; `replay_comparison.csv` y `trajectory_replay_comparison.csv` registran las diferencias finales y de las trayectorias contra la referencia. Los tiempos incluyen la instrumentación y no sirven para comparar velocidad con corridas anteriores.\n\n"
                 "```powershell\npython scripts/run_collocation_diagnostics.py\npython scripts/run_collocation_diagnostics.py --report-only\n```\n\n"
                 "La segunda orden regenera el reporte desde los artefactos guardados. Las figuras están incrustadas como PNG en este notebook y también se exportan en PNG/SVG."),
              {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
               "source": "import json\nfrom pathlib import Path\nROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').is_file())\nOUT = ROOT / " + repr(output.relative_to(root).as_posix()) + "\nsummary = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))\nsummary['methods']"}]
    path = root / "pca_collocation/notebooks/burgers" / config["notebook_name"]
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1, ensure_ascii=False), encoding="utf-8")
    embed_notebook_images(path, root)
    return path
