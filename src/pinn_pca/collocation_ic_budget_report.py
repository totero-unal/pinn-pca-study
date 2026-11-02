"""Fixed-budget comparisons and a portable report for distributed-IC training."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .collocation_ic_budget import audit_seed, experiment_paths, validate_budget
from .collocation_ic_control import METRICS
from .collocation_diagnostic_report import COLORS, LABELS, METHODS, _plots, _save, read_csv, stats, write_csv
from .notebook_images import embed_notebook_images


def build_budget_report(config, project_root):
    root = Path(project_root).resolve()
    baseline, output = experiment_paths(config, root)
    old_config = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    invariants = validate_budget(config, old_config)
    tables = {key: [] for key in ("metrics", "validation_trajectories", "loss_gradient_probes", "coverage", "regional_test_errors", "budget_metrics")}
    prefix, audits = [], []
    for seed in config["seeds"]:
        before, after = baseline / f"seed_{seed}", output / f"seed_{seed}"
        checked, differences = audit_seed(config, before, after)
        prefix.extend({"seed": seed, **row} for row in differences)
        audits.append({"seed": seed, "checks": checked,
                       "reference_metrics_sha256": hashlib.sha256((before / "metrics.csv").read_bytes()).hexdigest()})
        for key in tables:
            tables[key].extend(read_csv(after / f"{key}.csv"))
    for key, rows in tables.items():
        write_csv(output / f"{key}_by_seed.csv", rows)
    write_csv(output / "prefix_audit.csv", prefix)
    rows = tables["budget_metrics"]
    expected = {(str(seed), method, str(step)) for seed in config["seeds"] for method in METHODS for step in config["test_checkpoint_steps"]}
    if len(rows) != len(expected) or {(r["seed"], r["method"], r["step"]) for r in rows} != expected:
        raise ValueError("Missing or duplicated fixed-budget test results.")
    for row in rows:
        if any(not np.isfinite(float(row[key])) for key in METRICS):
            raise ValueError(f"Non-finite fixed-budget result: {row}")
    first = config["test_checkpoint_steps"][0]
    index = {(row["seed"], row["method"], int(row["step"])): row for row in rows}
    aggregate, differences, gaps = [], [], []
    for step in config["test_checkpoint_steps"]:
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method and int(r["step"]) == step]
            entry = {"method": method, "step": step, "continuation_steps": step - config["warmup_steps"],
                     "metrics": {key: stats([r[key] for r in selected]) for key in METRICS}, "changes_from_reference_budget": {}}
            for metric in METRICS:
                change_values, gap_values = [], []
                for row in selected:
                    reference = float(index[(row["seed"], method, first)][metric])
                    current = float(row[metric])
                    delta = current - reference
                    change_values.append(delta)
                    differences.append({"seed": row["seed"], "method": method, "step": step, "metric": metric,
                                        "reference_budget_value": reference, "current_value": current, "difference": delta})
                    if method != "sobol":
                        gap_values.append(current - float(index[(row["seed"], "sobol", step)][metric]))
                entry["changes_from_reference_budget"][metric] = {**stats(change_values), "improved_seed_count": sum(v < 0 for v in change_values)}
                if method != "sobol":
                    sobol_mean = np.mean([float(index[(r["seed"], "sobol", step)][metric]) for r in selected])
                    gaps.append({"method": method, "step": step, "metric": metric,
                                 **stats(gap_values), "better_than_sobol_count": sum(v < 0 for v in gap_values),
                                 "relative_mean_gap_percent": 100 * np.mean(gap_values) / sobol_mean})
            aggregate.append(entry)
    write_csv(output / "budget_changes_by_seed.csv", differences)
    write_csv(output / "paired_gaps_vs_sobol.csv", gaps)
    final, window = config["warmup_steps"] + config["train_steps"], config["trend_window_steps"]
    trends = []
    for method in METHODS:
        trace_index = {(r["seed"], int(r["step"])): r for r in tables["validation_trajectories"] if r["method"] == method}
        changes = [float(trace_index[(str(seed), final)]["validation_rmse"]) - float(trace_index[(str(seed), final-window)]["validation_rmse"]) for seed in config["seeds"]]
        trends.append({"method": method, "start_step": final-window, "end_step": final,
                       "validation_rmse_change": stats(changes), "improved_seed_count": sum(v < 0 for v in changes)})
    summary = {"configuration": config, "reference_ic_run_id": config["reference_ic_run_id"],
               "unchanged_configuration": invariants, "audits": audits,
               "max_prefix_difference": max(abs(float(r["difference"])) for r in prefix),
               "fixed_budget_results": aggregate, "paired_gaps_vs_sobol": gaps, "terminal_validation_trends": trends,
               "protocol": f"Same distributed 32 IC nodes, three selectors, unit loss weights, Adam and paired seeds. Fresh deterministic replay from initialization preserves optimizer state continuously through {final} total steps. Test metrics only at the configured fixed budgets; no stopping or weight tuning on test.",
               "limitations": "Ten reused seeds, one Burgers problem, no explicit boundary penalty. Instrumented time includes diagnostic costs and does not establish acceleration."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    figures = _plots(config, output, tables)
    figures.update(_budget_plots(config, output, rows, aggregate))
    _notebook(root, output, summary, figures)
    return summary


def _budget_plots(config, output, rows, aggregate):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.3))
    for ax, metric, label in zip(axes, ("test_rmse", "independent_residual_rms", "initial_condition_rmse"),
                                 ("RMSE de solución en prueba", "RMS residuo independiente", "RMSE IC global")):
        for method in METHODS:
            selected = [r for r in aggregate if r["method"] == method]
            ax.errorbar([r["step"] for r in selected], [r["metrics"][metric]["mean"] for r in selected],
                        yerr=[r["metrics"][metric]["std"] for r in selected], marker="o", capsize=3,
                        label=LABELS[method], color=COLORS[method])
        ax.set(xlabel="Pasos totales", ylabel=label, xticks=config["test_checkpoint_steps"])
        ax.tick_params(axis="x", labelrotation=25); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Comparación a presupuestos iguales: media ± DE de diez semillas")
    fig.tight_layout()
    figures = {"budgets": _save(fig, output, "fixed_budget_comparison")}
    index = {(r["seed"], r["method"], int(r["step"])): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, method in zip(axes, METHODS[1:]):
        for position, step in enumerate(config["test_checkpoint_steps"]):
            values = [float(index[(str(seed), method, step)]["test_rmse"]) - float(index[(str(seed), "sobol", step)]["test_rmse"]) for seed in config["seeds"]]
            ax.scatter(position + np.linspace(-.12, .12, len(values)), values, color=COLORS[method], alpha=.65)
            ax.plot([position-.2, position+.2], [np.mean(values)]*2, color="black", linewidth=2)
        ax.axhline(0, color="black", linestyle="--", linewidth=1)
        ax.set(title=LABELS[method], xlabel="Pasos totales", ylabel="RMSE guiado − RMSE Sobol",
               xticks=range(len(config["test_checkpoint_steps"])), xticklabels=config["test_checkpoint_steps"])
        ax.grid(alpha=.2)
    fig.suptitle("Diferencias pareadas de prueba: cada punto es una semilla; barra negra = media")
    fig.tight_layout()
    figures["gaps"] = _save(fig, output, "paired_test_gaps")
    seed, first, final = config["representative_seed"], config["test_checkpoint_steps"][0], config["test_checkpoint_steps"][-1]
    with np.load(output / f"seed_{seed}/fields.npz", allow_pickle=False) as data:
        limit = max(np.max(np.abs(data[f"{method}__{step}__error"])) for method in METHODS for step in (first, final))
        fig, axes = plt.subplots(3, 2, figsize=(9, 10), constrained_layout=True, sharex=True, sharey=True)
        for row, method in enumerate(METHODS):
            for column, step in enumerate((first, final)):
                image = axes[row, column].imshow(np.abs(data[f"{method}__{step}__error"]).T,
                    origin="lower", aspect="auto", extent=(data["x"][0], data["x"][-1], data["t"][0], data["t"][-1]), cmap="magma", vmin=0, vmax=limit)
                axes[row, column].set(title=f"{LABELS[method]}: paso {step}", xlabel="x", ylabel="t")
        fig.colorbar(image, ax=axes.ravel().tolist(), label="Error absoluto, escala compartida", shrink=.8)
        fig.suptitle(f"Efecto de ampliar el presupuesto; semilla predefinida {seed}")
        figures["field_change"] = _save(fig, output, f"error_{first}_vs_{final}")
    return figures


def _notebook(root, output, summary, figures):
    config = summary["configuration"]
    first = config["test_checkpoint_steps"][0]
    final = config["test_checkpoint_steps"][-1]
    warm = config["warmup_steps"]
    def md(source): return {"cell_type": "markdown", "metadata": {}, "source": source}
    def fmt(v): return f"{v['mean']:.4f} ± {v['std']:.4f}"
    asset = "../../" + output.relative_to(root / "pca_collocation").as_posix() + "/"
    def picture(key, caption): return md(f"![{caption}]({asset}{figures[key]})")
    def metric_table(metric):
        table = "| Pasos continuación / totales | Sobol | POD 32 | PCA 64 |\n|---|---:|---:|---:|\n"
        for step in config["test_checkpoint_steps"]:
            entries = [next(r for r in summary["fixed_budget_results"] if r["method"] == m and r["step"] == step) for m in METHODS]
            table += f"| {step-config['warmup_steps']} / {step} | " + " | ".join(fmt(r["metrics"][metric]) for r in entries) + " |\n"
        return table
    changes = "| Método | Δ RMSE final − referencia | Semillas que mejoran | Δ residuo RMS | Δ IC global |\n|---|---:|---:|---:|---:|\n"
    terminal = [r for r in summary["fixed_budget_results"] if r["step"] == final]
    for row in terminal:
        values = row["changes_from_reference_budget"]
        changes += f"| {LABELS[row['method']]} | {fmt(values['test_rmse'])} | {values['test_rmse']['improved_seed_count']}/{len(config['seeds'])} | {fmt(values['independent_residual_rms'])} | {fmt(values['initial_condition_rmse'])} |\n"
    gap_table = "| Pasos totales | Método | Δ RMSE frente a Sobol | Brecha relativa de medias | Semillas con menor RMSE que Sobol |\n|---|---|---:|---:|---:|\n"
    for row in summary["paired_gaps_vs_sobol"]:
        if row["metric"] == "test_rmse":
            gap_table += f"| {row['step']} | {LABELS[row['method']]} | {fmt(row)} | {row['relative_mean_gap_percent']:.1f}% | {row['better_than_sobol_count']}/{len(config['seeds'])} |\n"
    trend_table = f"| Método | Δ validación en los últimos {config['trend_window_steps']} pasos | Semillas que mejoran |\n|---|---:|---:|\n"
    for row in summary["terminal_validation_trends"]:
        trend_table += f"| {LABELS[row['method']]} | {fmt(row['validation_rmse_change'])} | {row['improved_seed_count']}/{len(config['seeds'])} |\n"
    best = min(terminal, key=lambda row: row["metrics"]["test_rmse"]["mean"])
    reading = []
    for row in terminal:
        reading.append(f"- **{LABELS[row['method']]}**: RMSE final {row['metrics']['test_rmse']['mean']:.4f}; "
                       f"mejora respecto a {first} pasos en {row['changes_from_reference_budget']['test_rmse']['improved_seed_count']}/{len(config['seeds'])} semillas; "
                       f"residuo RMS independiente {row['metrics']['independent_residual_rms']['mean']:.4f}.")
    for row in summary["paired_gaps_vs_sobol"]:
        if row["step"] == final and row["metric"] == "test_rmse":
            reading.append(f"- Al presupuesto final, **{LABELS[row['method']]}** tiene menor RMSE que Sobol en {row['better_than_sobol_count']}/{len(config['seeds'])} semillas.")
            initial_gap = next(r for r in summary["paired_gaps_vs_sobol"] if r["method"] == row["method"] and r["metric"] == "test_rmse" and r["step"] == first)
            reading.append(f"- La brecha absoluta de **{LABELS[row['method']]}** pasa de {initial_gap['mean']:.4f} a {row['mean']:.4f}; "
                           f"la brecha relativa de medias pasa de {initial_gap['relative_mean_gap_percent']:.1f}% a {row['relative_mean_gap_percent']:.1f}%.")
    cells = [md(f"# IC distribuida: ampliar de {first-warm} a {config['train_steps']} pasos de continuación\n\n"
                "**Pregunta:** ¿la diferencia entre Sobol y POD/PCA depende de entrenar durante más tiempo?\n\n"
                f"Se mantiene el protocolo corregido: 32 nodos IC distribuidos por todo el dominio, {len(config['seeds'])} semillas y tres selectores, con 128 puntos físicos activos y actualización cada 100 pasos. Los {warm} pasos de calentamiento se suman a los {config['train_steps']} de continuación: **{final} pasos totales**.\n\n"
                f"Al presupuesto final, el menor RMSE medio de prueba corresponde a **{LABELS[best['method']]}** ({best['metrics']['test_rmse']['mean']:.4f}). La comparación completa incluye residuo, condición inicial y frontera. La mayor diferencia numérica en el prefijo auditado es **{summary['max_prefix_difference']:.2g}**."),
             md("## 1. Protocolo fijado antes de ejecutar\n\n"
                "| Elemento | Configuración |\n|---|---|\n"
                "| IC | Los mismos 32 índices equiespaciados, incluidos ambos extremos |\n"
                "| Sobol | 128 puntos Sobol activos |\n| POD 32 | 32 puntos guiados por modos + 96 Sobol |\n| PCA 64 | 64 puntos guiados por modos + 64 Sobol |\n"
                "| Objetivo | Datos + física + IC, pesos 1/1/1; sin penalización separada de frontera |\n"
                "| Optimización | Misma arquitectura, Adam y tasa de aprendizaje |\n"
                "| Validación | Cada 100 pasos, sin detener ni cambiar pesos a partir de la prueba |\n"
                f"| Prueba numérica | Pasos totales {', '.join(map(str, config['test_checkpoint_steps']))}, iguales para los tres métodos |\n\n"
                f"La corrida previa guardó parámetros, pero no todos los momentos de Adam. Se repite desde la inicialización determinista y se conserva continuamente el estado del optimizador durante esta ejecución; no se reinicia Adam en el paso {first}. Los mismos lotes aleatorios, historia residual y puntos activos del prefijo se auditan frente a la referencia. Las sondas, candidatos, nodos IC y particiones de ajuste/validación/prueba se conservan por semilla.\n\n"
                "Los mapas pueden mostrar todos los tiempos para exploración. Las tablas de prueba se calculan exclusivamente en los presupuestos fijados; no se buscan retrospectivamente pesos o checkpoints con la prueba."),
             md("## 2. Precisión a presupuestos iguales\n\n" + metric_table("test_rmse") + "\nRMSE de solución en tiempos de prueba reservados. Los valores son media ± DE entre semillas, no intervalos de confianza.\n\n" + changes + f"\nCada cambio se calcula primero dentro de la misma semilla, tomando {first} pasos como referencia. Un valor negativo indica mejora. Entrenar más puede mejorar una métrica y empeorar otra."),
             picture("budgets", "Comparación de precisión, residuo e IC a presupuestos iguales"),
             md("## 3. ¿POD/PCA alcanza a Sobol?\n\n" + gap_table + "\nΔ = RMSE del método guiado − RMSE de Sobol, en la misma semilla y presupuesto. Un valor negativo favorece al método guiado. La brecha relativa es 100 × Δ medio / RMSE medio de Sobol; es un cociente de medias, no la media de cocientes por semilla. Las barras de la figura son medias; los puntos conservan la variabilidad entre semillas. Que ambas curvas bajen no demuestra que Δ tienda a cero. La mejor media de una corrida no establece superioridad general."),
             picture("gaps", "Diferencias pareadas frente a Sobol"),
             md("## 4. Cumplimiento físico y de las condiciones\n\n### Residuo RMS independiente\n\n" + metric_table("independent_residual_rms") + "\n### Error IC global\n\n" + metric_table("initial_condition_rmse") + "\n### RMS de frontera\n\n" + metric_table("boundary_rms") + "\nLa IC global usa todos los nodos espaciales; el residuo físico usa 2048 puntos independientes. La frontera se evalúa en tiempos de prueba y no se penaliza con un término separado en el objetivo. Los nodos IC siguen siendo 32 puntos discretos, no una imposición exacta en todo el continuo."),
             picture("monitors", "Evolución de los monitores independientes"),
             md("## 5. ¿Se observa una meseta al final?\n\n" + trend_table + f"\nEste resumen compara validación entre los pasos {final-config['trend_window_steps']} y {final}. Es descriptivo; el presupuesto final se fijó antes de observarlo. Una mejora media pequeña puede ocultar diferencias entre semillas."),
             md("## 6. Campos, gradientes y selección\n\n"
                f"Las figuras detalladas usan la semilla {config['representative_seed']}, fijada previamente. Los mapas antes/después comparten escala. Las curvas de pérdidas y gradientes usan las mismas sondas del diagnóstico; sus normas euclídeas no incorporan el precondicionamiento de Adam y los cosenos no prueban causalidad."),
             picture("field_change", f"Error con {first} y {final} pasos"), picture("profiles", "Perfiles de la solución final"),
             picture("loss", "Pérdidas separadas durante la extensión"), picture("cosines", "Oposición local de gradientes"),
             picture("selection", "Evolución de los puntos de colocación"), picture("coverage", "Cobertura del dominio"),
             md("## 7. Lectura del resultado\n\n" + "\n".join(reading) + "\n\n"
                "Las medias de error de solución se evalúan junto a sus cambios por semilla. El método con mejor media de solución se distingue del que obtiene menor residuo. Una reducción del error propio, e incluso de una brecha absoluta, no garantiza que la brecha relativa también baje. Estas observaciones se limitan al intervalo de presupuestos estudiado; no identifican el límite asintótico ni predicen que las curvas mantengan el mismo orden con presupuestos arbitrariamente mayores."),
             md("## 8. Reproducción y alcance\n\n"
                f"```powershell\npython scripts/run_ic_long_budget.py --config {config.get('config_path', 'pca_collocation/configs/burgers_ic_long_budget.json')}\npython scripts/run_ic_long_budget.py --config {config.get('config_path', 'pca_collocation/configs/burgers_ic_long_budget.json')} --report-only\n```\n\n"
                f"`budget_metrics_by_seed.csv` guarda las métricas en los {len(config['test_checkpoint_steps'])} presupuestos; `budget_changes_by_seed.csv` conserva cambios pareados; `paired_gaps_vs_sobol.csv` resume las comparaciones entre métodos; `prefix_audit.csv` contrasta la trayectoria compartida con la corrida anterior. `summary.json` conserva las auditorías y huellas de las métricas de referencia. Cada semilla mantiene campos, parámetros, sondas y puntos activos.\n\n"
                f"Son diez semillas reutilizadas de un único problema Burgers. El objetivo de esta ronda es el efecto del presupuesto, con la distribución IC corregida. Los tiempos medidos incluyen instrumentación y no se usan para afirmar aceleración. Las figuras están incrustadas como PNG y exportadas como PNG/SVG. El [control anterior]({config.get('reference_notebook', '11_ic_coverage_control.ipynb')}) documenta el punto de partida.")]
    path = root / "pca_collocation/notebooks/burgers" / config["notebook_name"]
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}, indent=1, ensure_ascii=False), encoding="utf-8")
    embed_notebook_images(path, root)
    return path
