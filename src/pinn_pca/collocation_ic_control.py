"""Paired control changing only the spatial placement of the fixed IC budget."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .collocation import _run_collocation_seed
from .collocation_diagnostics import DiagnosticRecorder, METHOD_SPECS
from .collocation_diagnostic_report import COLORS, LABELS, METHODS, _plots, _save, read_csv, stats, write_csv
from .notebook_images import embed_notebook_images

INVARIANTS = ("data_path", "seeds", "width", "learning_rate", "warmup_steps", "train_steps", "batch_size", "update_every",
              "candidate_count", "history_snapshots", "independent_count", "initial_condition_points", "test_time_count",
              "validation_time_count", "energy_threshold", "max_modes", "qr_point_count", "validation_rmse_target",
              "representative_seed", "field_checkpoints", "selection_plot_steps", "grid_x_count", "grid_t_count",
              "front_abs_x_max", "front_time_min", "coverage_bins", "profile_times", "profile_zoom_abs_x")
METRICS = ("test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms")


def validate_control(config, baseline):
    for key in INVARIANTS:
        if config[key] != baseline[key]:
            raise ValueError(f"IC placement control requires unchanged {key}.")
    if baseline.get("initial_condition_strategy", "left_prefix") != "left_prefix":
        raise ValueError("Baseline must use the original left-prefix IC placement.")
    if config.get("initial_condition_strategy") != "uniform_domain":
        raise ValueError("Treatment must use uniform-domain IC placement.")
    if baseline["method_specs"] != METHOD_SPECS:
        raise ValueError("Baseline method definitions do not match this control.")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be distinct.")
    return {key: config[key] for key in INVARIANTS}


def _paths(config, root):
    baseline = root / "pca_collocation/results/burgers/collocation_diagnostics" / config["baseline_diagnostic_run_id"]
    output = root / "pca_collocation/results/burgers/ic_coverage_control" / config["run_id"]
    return baseline, output


def run_ic_control(config, project_root):
    root = Path(project_root).resolve()
    baseline, output = _paths(config, root)
    previous = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    validate_control(config, previous)
    if output.resolve() == baseline.resolve():
        raise ValueError("Control output must be distinct from the baseline.")
    output.mkdir(parents=True, exist_ok=True)
    saved = dict(config, method_specs=METHOD_SPECS, common_selection_seed=True, skip_seed_plots=True)
    (output / "config.json").write_text(json.dumps(saved, indent=2), encoding="utf-8")
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        _run_collocation_seed(dict(saved, seed=seed), root, seed_dir,
                              observer=DiagnosticRecorder(config, seed_dir))
        _audit_pair(baseline / f"seed_{seed}", seed_dir)
        print(f"IC coverage control: seed {seed} completed; pairing verified.", flush=True)
    build_ic_control_report(config, root)
    return output


def _audit_pair(before, after):
    checked = []
    with np.load(before / "fields.npz", allow_pickle=False) as old, np.load(after / "fields.npz", allow_pickle=False) as new:
        for key in ("x", "t", "reference", "fit_time_indices", "validation_time_indices", "test_time_indices",
                    "common_probe_points", "probe_data_x", "probe_data_t", "probe_data_u"):
            if not np.array_equal(old[key], new[key]):
                raise ValueError(f"Paired IC control changed {key} in {after}.")
            checked.append(key)
        if len(old["initial_x"]) != len(new["initial_x"]):
            raise ValueError("IC point count changed.")
        if np.array_equal(old["initial_x"], new["initial_x"]):
            raise ValueError("IC placement did not change.")
        if not (new["initial_x"][0] == new["x"][0] and new["initial_x"][-1] == new["x"][-1]):
            raise ValueError("Distributed IC nodes must include both domain endpoints.")
        for method in METHODS[1:]:
            if not np.array_equal(new[f"{method}__100__u"], new["sobol__100__u"]):
                raise ValueError("Methods within the control must share the warmed-up state.")
    with np.load(before / "selection_snapshot.npz", allow_pickle=False) as old, np.load(after / "selection_snapshot.npz", allow_pickle=False) as new:
        if not np.array_equal(old["candidates"], new["candidates"]):
            raise ValueError("The candidate pool changed.")
    return checked + ["candidate_pool", "same_ic_count", "changed_ic_placement", "ic_domain_endpoints", "shared_warmup_within_treatment"]


def build_ic_control_report(config, project_root):
    root = Path(project_root).resolve()
    baseline, output = _paths(config, root)
    previous = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    invariants = validate_control(config, previous)
    tables = {name: [] for name in ("metrics", "validation_trajectories", "loss_gradient_probes", "coverage", "regional_test_errors")}
    audits = []
    for seed in config["seeds"]:
        checked = _audit_pair(baseline / f"seed_{seed}", output / f"seed_{seed}")
        audits.append({"seed": seed, "verified": checked,
                       "baseline_metrics_sha256": hashlib.sha256((baseline / f"seed_{seed}/metrics.csv").read_bytes()).hexdigest()})
        for name in tables:
            tables[name].extend(read_csv(output / f"seed_{seed}/{name}.csv"))
    for name, rows in tables.items():
        write_csv(output / f"{name}_by_seed.csv", rows)
    old_metrics = read_csv(baseline / "metrics_by_seed.csv")
    old_regions = {(r["seed"], r["method"], r["region"]): r for r in read_csv(baseline / "regional_test_errors_by_seed.csv")
                   if int(r["step"]) == config["warmup_steps"] + config["train_steps"]}
    old_index = {(r["seed"], r["method"]): r for r in old_metrics}
    new_index = {(r["seed"], r["method"]): r for r in tables["metrics"]}
    differences, results = [], []
    for row in tables["metrics"]:
        for metric in METRICS:
            before = float(old_index[(row["seed"], row["method"])][metric])
            after = float(row[metric])
            differences.append({"seed": row["seed"], "method": row["method"], "metric": metric,
                                "left_prefix": before, "uniform_domain": after, "after_minus_before": after - before})
    for method in METHODS:
        result = {"method": method, "label": LABELS[method], "metrics": {}}
        for metric in METRICS:
            rows = [r for r in differences if r["method"] == method and r["metric"] == metric]
            result["metrics"][metric] = {"before": stats([r["left_prefix"] for r in rows]),
                                          "after": stats([r["uniform_domain"] for r in rows]),
                                          "paired_change": stats([r["after_minus_before"] for r in rows]),
                                          "improved_seed_count": sum(r["after_minus_before"] < 0 for r in rows)}
        results.append(result)
    regional_changes = []
    for row in tables["regional_test_errors"]:
        if int(row["step"]) != config["warmup_steps"] + config["train_steps"]:
            continue
        before = float(old_regions[(row["seed"], row["method"], row["region"])]["rmse"])
        regional_changes.append({"seed": row["seed"], "method": row["method"], "region": row["region"],
                                 "before": before, "after": float(row["rmse"]), "after_minus_before": float(row["rmse"]) - before})
    write_csv(output / "paired_regional_changes.csv", regional_changes)
    for result in results:
        result["regional_test_rmse"] = {}
        for region in ("central_late_test", "outside_window_test"):
            rows = [r for r in regional_changes if r["method"] == result["method"] and r["region"] == region]
            result["regional_test_rmse"][region] = {key: stats([r[key] for r in rows]) for key in ("before", "after", "after_minus_before")}
    interactions = []
    for method in METHODS[1:]:
        for metric in METRICS:
            before_gaps, after_gaps, changes = [], [], []
            for seed in config["seeds"]:
                key = str(seed)
                old_gap = float(old_index[(key, method)][metric]) - float(old_index[(key, "sobol")][metric])
                new_gap = float(new_index[(key, method)][metric]) - float(new_index[(key, "sobol")][metric])
                before_gaps.append(old_gap); after_gaps.append(new_gap); changes.append(new_gap - old_gap)
            interactions.append({"method": method, "metric": metric, "before_gap": stats(before_gaps),
                                 "after_gap": stats(after_gaps), "change_in_gap": stats(changes),
                                 "after_better_than_sobol_count": sum(v < 0 for v in after_gaps)})
    write_csv(output / "paired_changes_by_seed.csv", differences)
    write_csv(output / "method_gap_changes.csv", [
        {"method": r["method"], "metric": r["metric"], "before_gap_mean": r["before_gap"]["mean"],
         "after_gap_mean": r["after_gap"]["mean"], "gap_change_mean": r["change_in_gap"]["mean"],
         "gap_change_std": r["change_in_gap"]["std"], "after_better_count": r["after_better_than_sobol_count"]} for r in interactions])
    summary = {"configuration": config, "baseline_run_id": config["baseline_diagnostic_run_id"],
               "intervention": "Same 32 IC nodes: original first spatial indices versus rounded equally spaced indices including both endpoints; applied throughout warmup and continuation.",
               "unchanged_configuration": invariants, "pairing_audit": audits, "methods": results,
               "method_gaps_vs_sobol": interactions,
               "protocol_before": json.loads((baseline / f"seed_{config['representative_seed']}/diagnostic_protocol.json").read_text(encoding="utf-8")),
               "protocol_after": json.loads((output / f"seed_{config['representative_seed']}/diagnostic_protocol.json").read_text(encoding="utf-8")),
               "limitations": "Ten reused seeds on Burgers; paired descriptive statistics. Training can change selected collocation points through the changed residual history. No separate boundary penalty is introduced. IC is imposed at 32 discrete nodes, not at every spatial point. Instrumented timings are not used to claim acceleration."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    figures = _plots(config, output, tables)
    figures.update(_comparison_plots(config, baseline, output, old_metrics, tables, summary))
    _notebook(root, output, baseline, summary, figures)
    return summary


def _comparison_plots(config, baseline, output, old_metrics, tables, summary):
    seed = config["representative_seed"]
    final = config["warmup_steps"] + config["train_steps"]
    with np.load(baseline / f"seed_{seed}/fields.npz", allow_pickle=False) as old_file, np.load(output / f"seed_{seed}/fields.npz", allow_pickle=False) as new_file:
        old, new = dict(old_file), dict(new_file)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(old["x"], old["reference"][:, 0], color="black", label="Referencia u(x,0)")
    for fields, label, color, marker in ((old, "32 nodos: primeros índices", "#d38816", "o"), (new, "32 nodos: dominio completo", "#207c61", "x")):
        ax.scatter(fields["initial_x"], fields["initial_u"], color=color, marker=marker, label=label, s=45, zorder=3)
    ax.set(xlabel="x", ylabel="u(x,0)", title="Única intervención: cambiar la ubicación de los mismos 32 nodos IC")
    ax.grid(alpha=.2); ax.legend(fontsize=9)
    fig.tight_layout()
    figures = {"placement": _save(fig, output, "ic_nodes_before_after")}
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric, label in zip(axes, ("test_rmse", "initial_condition_rmse", "independent_residual_rms"),
                                 ("RMSE de solución en prueba", "RMSE IC global", "RMS residuo independiente")):
        for arm, offset, color in (("before", -.12, "#888888"), ("after", .12, "#207c61")):
            values = [row["metrics"][metric][arm] for row in summary["methods"]]
            ax.errorbar(np.arange(3) + offset, [v["mean"] for v in values], yerr=[v["std"] for v in values],
                        fmt="o", capsize=4, color=color, label="Primeros 32" if arm == "before" else "32 distribuidos")
        ax.set_xticks(range(3), ["Sobol", "POD 32", "PCA 64"])
        ax.set_ylabel(label); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Efecto del reparto de IC: media ± DE, diez semillas pareadas")
    fig.tight_layout()
    figures["comparison"] = _save(fig, output, "ic_control_metrics")
    before_traces = read_csv(baseline / "validation_trajectories_by_seed.csv")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for column, method in enumerate(METHODS):
        for row, (metric, label) in enumerate((("validation_rmse", "RMSE validación"), ("initial_condition_rmse", "RMSE IC global"))):
            ax = axes[row, column]
            for data, style, arm in ((before_traces, "--", "Primeros 32"), (tables["validation_trajectories"], "-", "32 distribuidos")):
                selected = [r for r in data if r["method"] == method]
                steps = sorted({int(r["step"]) for r in selected})
                values = [stats([r[metric] for r in selected if int(r["step"]) == step]) for step in steps]
                means, deviations = np.array([v["mean"] for v in values]), np.array([v["std"] for v in values])
                ax.plot(steps, means, linestyle=style, color=COLORS[method], label=arm)
                ax.fill_between(steps, np.maximum(means - deviations, 0), means + deviations, color=COLORS[method], alpha=.1)
            ax.set(xlabel="Pasos totales", ylabel=label); ax.grid(alpha=.2)
            if row == 0: ax.set_title(LABELS[method])
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Comparación durante el entrenamiento: línea continua = IC distribuida")
    fig.tight_layout()
    figures["trajectories"] = _save(fig, output, "ic_control_trajectories")
    limit = max(np.max(np.abs(fields[f"{method}__{final}__error"])) for fields in (old, new) for method in METHODS)
    fig, axes = plt.subplots(3, 2, figsize=(9, 10), constrained_layout=True, sharex=True, sharey=True)
    for row, method in enumerate(METHODS):
        for column, (fields, label) in enumerate(((old, "Primeros 32"), (new, "32 distribuidos"))):
            image = axes[row, column].imshow(np.abs(fields[f"{method}__{final}__error"]).T, origin="lower", aspect="auto",
                                            extent=(fields["x"][0], fields["x"][-1], fields["t"][0], fields["t"][-1]), cmap="magma", vmin=0, vmax=limit)
            axes[row, column].set_title(LABELS[method] + ": " + label)
            axes[row, column].set(xlabel="x", ylabel="t")
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Error absoluto, escala compartida", shrink=.8)
    fig.suptitle(f"Error final antes/después, semilla predefinida {seed}")
    figures["paired_fields"] = _save(fig, output, "ic_control_error_maps")
    return figures


def _notebook(root, output, baseline, summary, figures):
    config = summary["configuration"]
    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    asset = "../../" + output.relative_to(root / "pca_collocation").as_posix() + "/"
    def picture(key, caption):
        return md(f"![{caption}]({asset}{figures[key]})")
    def fmt(value):
        return f"{value['mean']:.4f} ± {value['std']:.4f}"
    def table(metric):
        text = "| Método | Primeros 32 nodos | 32 distribuidos | Δ distribuida − anterior | Semillas con menor valor |\n|---|---:|---:|---:|---:|\n"
        for row in summary["methods"]:
            values = row["metrics"][metric]
            text += f"| {row['label']} | {fmt(values['before'])} | {fmt(values['after'])} | {fmt(values['paired_change'])} | {values['improved_seed_count']}/{len(config['seeds'])} |\n"
        return text
    gap_table = "| Método | Brecha frente a Sobol: antes | Brecha: después | Cambio de brecha | Semillas que superan a Sobol después |\n|---|---:|---:|---:|---:|\n"
    region_table = "| Método | Ventana central: antes | Ventana central: después | Fuera de ventana: antes | Fuera de ventana: después |\n|---|---:|---:|---:|---:|\n"
    for row in summary["methods"]:
        region_table += f"| {row['label']} | " + " | ".join(fmt(row["regional_test_rmse"][region][arm]) for region in ("central_late_test", "outside_window_test") for arm in ("before", "after")) + " |\n"
    for row in summary["method_gaps_vs_sobol"]:
        if row["metric"] == "test_rmse":
            gap_table += f"| {LABELS[row['method']]} | {fmt(row['before_gap'])} | {fmt(row['after_gap'])} | {fmt(row['change_in_gap'])} | {row['after_better_than_sobol_count']}/{len(config['seeds'])} |\n"
    best = min(summary["methods"], key=lambda r: r["metrics"]["test_rmse"]["after"]["mean"])
    findings = []
    for row in summary["methods"]:
        values = row["metrics"]
        findings.append(f"- **{row['label']}**: IC global mejora en {values['initial_condition_rmse']['improved_seed_count']}/10 semillas; "
                        f"el RMSE de solución mejora en {values['test_rmse']['improved_seed_count']}/10 y el residuo físico en {values['independent_residual_rms']['improved_seed_count']}/10.")
    if all(row["metrics"]["independent_residual_rms"]["paired_change"]["mean"] > 0 for row in summary["methods"]):
        findings.append("\nEl residuo físico independiente aumenta en media en los tres métodos. La mejora de IC global no equivale a una mejora de todas las métricas. El cambio de distribución modifica el compromiso que alcanza el optimizador con los mismos pesos de pérdida.")
    findings.append("\nLas brechas medias frente a Sobol se interpretan junto con su variabilidad pareada; reducir una brecha media no demuestra que el método guiado supere a Sobol de forma consistente.")
    old, new = summary["protocol_before"], summary["protocol_after"]
    cells = [md("# Control de cobertura de la condición inicial\n\n"
                "**Pregunta:** ¿cambia el resultado de la selección de colocación al repartir los mismos nodos de condición inicial por todo el dominio?\n\n"
                "La única intervención configurada es la ubicación de los 32 nodos IC. Se comparan las mismas diez semillas y tres métodos: Sobol, POD con 32 puntos guiados y PCA centrado + media con 64. Las estadísticas son descriptivas y pareadas por semilla. La referencia es la corrida instrumentada anterior, cuya reproducción ya había coincidido exactamente con el entrenamiento original.\n\n"
                f"Con IC distribuida, el menor RMSE medio de prueba corresponde a **{best['label']}**, con **{best['metrics']['test_rmse']['after']['mean']:.4f}**. Las tablas siguientes muestran tanto el efecto del cambio de IC como la comparación entre selectores."),
             md("## 1. Qué cambia y qué permanece fijo\n\n"
                "| Elemento | Antes | Control |\n|---|---|---|\n"
                "| Nodos IC | Primeros 32 índices espaciales | 32 índices aproximadamente equiespaciados, incluidos ambos extremos |\n"
                "| Pesos de datos, física e IC | 1, 1, 1 | 1, 1, 1 |\n"
                "| Penalización separada de frontera | Ausente | Ausente |\n"
                "| Puntos físicos activos | 128, renovados cada 100 pasos | Igual |\n"
                "| Arquitectura / Adam / datos | Tres capas ocultas de ancho 32 / misma tasa y secuencia supervisada | Igual |\n"
                "| Presupuesto | 100 pasos de calentamiento + 2000 de continuación | Igual |\n"
                "| Semillas, candidatos y evaluación | Diez semillas; 4096 candidatos; evaluación independiente y tiempos reservados | Iguales por pareja |\n\n"
                "Los 32 nodos IC se usan en t=0 y son un conjunto separado de los 128 puntos físicos activos. Las composiciones de colocación siguen siendo Sobol: 128 Sobol; POD: 32 guiados + 96 Sobol; PCA: 64 guiados + 64 Sobol. Los 32/64 puntos guiados no representan cantidades de componentes principales.\n\n"
                "La distribución IC se aplica desde el calentamiento. Los pesos iniciales aleatorios son los mismos por semilla; el estado después del calentamiento puede cambiar como efecto de la intervención. Los tres selectores comparten ese nuevo estado dentro del control. Sus puntos seleccionados pueden cambiar después porque la historia del residuo también cambia: es una consecuencia del cambio de IC.\n\n"
                f"El intervalo entre nodos extremos pasa de [{old['initial_x_min']:.4f}, {old['initial_x_max']:.4f}] a [{new['initial_x_min']:.4f}, {new['initial_x_max']:.4f}]. La nueva distribución sigue usando solo 32 nodos discretos de {new['spatial_grid_points']}; no impone IC en cada posición del continuo. La auditoría verifica igualdad de las divisiones de datos, las sondas, los datos de referencia y los candidatos para cada pareja."),
             picture("placement", "Ubicación de los 32 nodos de condición inicial"),
             md("## 2. Efecto en la precisión de solución\n\n" + table("test_rmse") + "\nRMSE en los mismos tiempos de prueba reservados por semilla. Cada celda es media ± DE entre semillas. La diferencia se calcula primero dentro de cada pareja, no restando desviaciones estándar. Un Δ negativo indica mejora. El conjunto de prueba no se utiliza para seleccionar nodos ni ajustar pesos."),
             md("## 3. Condición inicial global y residuo físico\n\n"
                "### RMSE de IC en todos los nodos espaciales\n\n" + table("initial_condition_rmse") + "\nLa evaluación global permite comparar ambas ubicaciones. La pérdida IC de entrenamiento cambia los nodos donde se mide y no sustituye esta métrica.\n\n"
                "### RMS del residuo en los 2048 puntos independientes\n\n" + table("independent_residual_rms") + "\nUna reducción del error de solución puede acompañarse de otro cambio en el residuo; se evalúan ambos criterios."),
             picture("comparison", "Comparación de métricas antes y después"),
             md("## 4. ¿Cambia la ventaja relativa frente a Sobol?\n\n" + gap_table + "\nLa brecha es RMSE del método guiado menos RMSE de Sobol en la misma semilla. Una brecha negativa favorece al método guiado. Su cambio es (guiado−Sobol) con IC distribuida menos (guiado−Sobol) con IC anterior; un cambio negativo reduce la desventaja o amplía la ventaja del guiado. Todas las medias y DE se calculan sobre esas diferencias pareadas. Diez semillas y un único problema no prueban superioridad general."),
             picture("trajectories", "Precisión e IC global durante el entrenamiento"),
             md("## 5. Dónde cambia la solución\n\n"
                f"La semilla de las figuras detalladas es {config['representative_seed']}, fijada antes de entrenar. El par de mapas usa la misma escala de error. Las imágenes abarcan tiempos de ajuste, validación y prueba para exploración; las métricas de generalización anteriores usan exclusivamente tiempos de prueba.\n\n"
                f"La ventana central tardía es |x| ≤ {config['front_abs_x_max']} y t ≥ {config['front_time_min']}. Se mantiene idéntica a la del diagnóstico, sin intervenir en el entrenamiento. Los RMSE siguientes se miden en tiempos de prueba; la región exterior es el complemento completo de esa ventana.\n\n" + region_table),
             picture("paired_fields", "Mapas de error antes y después del cambio de IC"),
             picture("profiles", "Perfiles del control con IC distribuida"),
             md("## 6. Mecanismo observado dentro del control\n\n"
                "Las siguientes figuras corresponden al brazo con IC distribuida. Conservan las mediciones del diagnóstico: sonda física común fija y sonda activa de cada método, con los mismos datos supervisados y los nuevos nodos IC. Las normas son gradientes euclídeos de cada término; no incorporan los momentos ni el precondicionamiento de Adam. Un coseno negativo indica oposición local, sin demostrar por sí solo causalidad.\n\n"
                "En pérdidas y normas, las líneas son medias y las bandas percentiles 16–84 de los valores positivos. En conflicto y cobertura, las bandas representan media ± DE. La comparación entre brazos de la norma de IC corresponde deliberadamente a dos conjuntos de nodos distintos."),
             picture("loss", "Pérdidas del control"), picture("grad", "Normas de gradiente del control"),
             picture("cosines", "Conflicto de gradientes del control"), picture("selection", "Puntos seleccionados con IC distribuida"),
             picture("coverage", "Cobertura de colocación con IC distribuida"),
             md("## 7. Lectura del control\n\n" + "\n".join(findings)),
             md("## 8. Alcance y reproducción\n\n"
                "Este control permite atribuir el cambio observado del pipeline a la intervención de ubicación IC, dentro de las parejas y del protocolo especificado. No identifica todos los mecanismos internos ni garantiza generalización a otras ecuaciones. La distribución física y los modos pueden evolucionar como consecuencia de esa intervención. No se añade un término de frontera ni se seleccionan nuevos pesos a partir de la prueba.\n\n"
                "La referencia anterior se reutiliza; `summary.json` guarda la configuración invariable, las estrategias de IC, las auditorías por semilla y las huellas SHA-256 de las métricas de referencia. `paired_changes_by_seed.csv` conserva cada diferencia y `method_gap_changes.csv` las comparaciones frente a Sobol. Cada carpeta de semilla contiene los puntos activos, gradientes, campos, parámetros y métricas. Las mediciones de tiempo incluyen la instrumentación y no sustentan una afirmación de aceleración.\n\n"
                "```powershell\npython scripts/run_ic_coverage_control.py\npython scripts/run_ic_coverage_control.py --report-only\n```\n\n"
                "Las figuras se conservan como PNG/SVG en los resultados y como PNG incrustado en este notebook. El [diagnóstico anterior](10_residual_balance_diagnostics.ipynb) documenta el brazo de referencia.")]
    path = root / "pca_collocation/notebooks/burgers" / config["notebook_name"]
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, ensure_ascii=False, indent=1), encoding="utf-8")
    embed_notebook_images(path, root)
    return path
