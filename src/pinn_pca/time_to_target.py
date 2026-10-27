"""Retrospective time-to-validation-target analysis with explicit censoring."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import platform
from pathlib import Path

import numpy as np


def first_hit(trace, target):
    """Use the first observed crossing; never interpolate or impute a failure."""
    if not trace or not math.isfinite(target) or target < 0:
        raise ValueError("Se requiere una trayectoria y un umbral finito no negativo.")
    previous_step, previous_time = 0, 0.0
    hit = None
    for point in trace:
        step = int(point["step"])
        seconds = float(point["elapsed_pipeline_seconds"])
        rmse = float(point["validation_rmse"])
        if (step <= previous_step or seconds <= previous_time
                or not all(map(math.isfinite, (seconds, rmse))) or rmse < 0):
            raise ValueError("Checkpoints inválidos: orden, tiempo o RMSE.")
        if hit is None and rmse <= target:
            hit = (step, seconds, previous_step, previous_time)
        previous_step, previous_time = step, seconds
    last = trace[-1]
    return {
        "hit": hit is not None,
        "hit_step": hit[0] if hit else None,
        "hit_pipeline_seconds": hit[1] if hit else None,
        "previous_checkpoint_step": hit[2] if hit else None,
        "previous_checkpoint_seconds": hit[3] if hit else None,
        "terminal_step": int(last["step"]),
        "terminal_pipeline_seconds": float(last["elapsed_pipeline_seconds"]),
        "terminal_validation_rmse": float(last["validation_rmse"]),
        "right_censored": hit is None,
    }


def common_horizon_pair(base, other):
    """Compare capped waiting times on a horizon fully observed by both runs."""
    horizon = min(base["terminal_pipeline_seconds"], other["terminal_pipeline_seconds"])
    def capped(row):
        hit = row["hit_pipeline_seconds"]
        return min(hit, horizon) if hit is not None else horizon
    base_hit = base["hit"] and base["hit_pipeline_seconds"] <= horizon
    other_hit = other["hit"] and other["hit_pipeline_seconds"] <= horizon
    return {
        "common_horizon_seconds": horizon,
        "base_hit_within_horizon": bool(base_hit),
        "other_hit_within_horizon": bool(other_hit),
        "base_capped_seconds": capped(base),
        "other_capped_seconds": capped(other),
        "other_minus_base_capped_seconds": capped(other) - capped(base),
    }


def _bootstrap(delta, draws, seed):
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def analyze_trajectories(trajectories, baseline, draws=10000, seed=1730):
    groups = {}
    for row in trajectories:
        groups.setdefault((row["model"], int(row["seed"])), []).append(row)
    models = list(dict.fromkeys(row["model"] for row in trajectories))
    if baseline not in models or int(draws) < 1:
        raise ValueError("Falta el modelo base o el número de remuestreos es inválido.")
    seeds = sorted(s for m, s in groups if m == baseline)
    expected_grid = [int(row["step"]) for row in groups[(baseline, seeds[0])]]
    for model in models:
        if sorted(s for m, s in groups if m == model) != seeds:
            raise ValueError("Los modelos deben compartir todas las semillas.")
        for s in seeds:
            if [int(row["step"]) for row in groups[(model, s)]] != expected_grid:
                raise ValueError("Los modelos deben compartir la rejilla de checkpoints.")
    baseline_final = [float(groups[(baseline, s)][-1]["validation_rmse"]) for s in seeds]
    target = float(np.mean(baseline_final))
    records = [{"model": model, "seed": s, "target_rmse": target,
                **first_hit(groups[(model, s)], target)} for model in models for s in seeds]
    lookup = {(r["model"], r["seed"]): r for r in records}
    global_horizon = min(r["terminal_pipeline_seconds"] for r in records)
    summaries = {}
    for model in models:
        selected = [r for r in records if r["model"] == model]
        hit_times = [r["hit_pipeline_seconds"] for r in selected if r["hit"]]
        hit_steps = [r["hit_step"] for r in selected if r["hit"]]
        summaries[model] = {
            "seeds": len(seeds), "hit_count": len(hit_times),
            "censored_count": len(seeds) - len(hit_times),
            "median_seconds_if_hit": float(np.median(hit_times)) if hit_times else None,
            "median_steps_if_hit": float(np.median(hit_steps)) if hit_steps else None,
            "hit_count_within_global_horizon": sum(
                t <= global_horizon for t in hit_times),
        }
    paired_rows, paired = [], {}
    for model in models:
        if model == baseline:
            continue
        local = []
        for s in seeds:
            b, o = lookup[(baseline, s)], lookup[(model, s)]
            both = b["hit"] and o["hit"]
            row = {"model": model, "seed": s,
                   "base_hit": b["hit"], "other_hit": o["hit"],
                   "both_hit": both,
                   "other_minus_base_seconds_if_both_hit": (
                       o["hit_pipeline_seconds"] - b["hit_pipeline_seconds"] if both else None),
                   **common_horizon_pair(b, o)}
            local.append(row)
        delta = [r["other_minus_base_capped_seconds"] for r in local]
        paired[model] = {
            "both_hit_count": sum(r["both_hit"] for r in local),
            "only_other_hit_count": sum(r["other_hit"] and not r["base_hit"] for r in local),
            "only_base_hit_count": sum(r["base_hit"] and not r["other_hit"] for r in local),
            "neither_hit_count": sum(not r["base_hit"] and not r["other_hit"] for r in local),
            "common_horizon_other_hit_count": sum(r["other_hit_within_horizon"] for r in local),
            "common_horizon_base_hit_count": sum(r["base_hit_within_horizon"] for r in local),
            "mean_capped_difference_seconds": float(np.mean(delta)),
            "bootstrap_mean_capped_difference_ci_95": _bootstrap(delta, int(draws), seed),
            "horizon_seconds_range": [min(r["common_horizon_seconds"] for r in local),
                                      max(r["common_horizon_seconds"] for r in local)],
        }
        paired_rows.extend(local)
    return {"target_rmse": target, "baseline_model": baseline, "seed_count": len(seeds),
            "baseline_final_validation_distribution": {
                "mean": target, "median": float(np.median(baseline_final)),
                "min": min(baseline_final), "max": max(baseline_final),
                "values_by_seed": dict(zip(map(str, seeds), baseline_final))},
            "final_step": expected_grid[-1], "global_common_horizon_seconds": global_horizon,
            "models": summaries, "paired": paired, "records": records,
            "paired_records": paired_rows}


def _figures(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    colors = ["#526274", "#2782a3", "#c8791c"]
    models = list(report["models"])
    def save(fig, name):
        fig.savefig(output / (name + ".svg"), bbox_inches="tight")
        fig.savefig(output / (name + ".png"), dpi=170, bbox_inches="tight")
        plt.close(fig)
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(8.2, 4.8), layout="constrained")
        for model, color in zip(models, colors):
            rows = [r for r in report["records"] if r["model"] == model]
            hits = sorted(r["hit_step"] for r in rows if r["hit"])
            x = [0, *hits, report["final_step"]]
            y = [0, *[(i + 1) / len(rows) for i in range(len(hits))], len(hits) / len(rows)]
            ax.step(x, y, where="post", label=f'{model}: {len(hits)}/{len(rows)}', color=color)
        ax.set(xlabel="Pasos de entrenamiento", ylabel="Fracción que alcanzó el umbral",
               ylim=(-.02, 1.02), xlim=(0, report["final_step"]),
               title=f'Primer RMSE de validación ≤ {report["target_rmse"]:.4f}')
        ax.grid(alpha=.2); ax.legend(loc="upper left")
        save(fig, "success_vs_steps")

        fig, ax = plt.subplots(figsize=(9, 8.4), layout="constrained")
        labels = []
        for index, row in enumerate(report["records"]):
            color = colors[models.index(row["model"])]
            time = row["hit_pipeline_seconds"] if row["hit"] else row["terminal_pipeline_seconds"]
            ax.hlines(index, 0, time, color=color, alpha=.25)
            ax.scatter(time, index, color=color, marker="o" if row["hit"] else "x", s=34)
            labels.append(f'{row["model"]} · {row["seed"]}')
        ax.axvline(report["global_common_horizon_seconds"], ls="--", color="#888", lw=1)
        ax.set_yticks(range(len(labels)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set(xlabel="Segundos de pipeline registrados", title="Tiempo al primer cruce o al fin de observación")
        ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#444", label="Umbral alcanzado"),
                           Line2D([], [], marker="x", ls="", color="#444", label="No alcanzado: censura"),
                           Line2D([], [], ls="--", color="#888", label="Horizonte común a todas las corridas")],
                  loc="upper right", fontsize=8)
        ax.grid(axis="x", alpha=.2)
        save(fig, "time_to_target_by_seed")

        fig, ax = plt.subplots(figsize=(8.5, 4.6), layout="constrained")
        other_models = list(report["paired"])
        for index, model in enumerate(other_models):
            delta = [r["other_minus_base_capped_seconds"] for r in report["paired_records"]
                     if r["model"] == model]
            ax.scatter(delta, np.linspace(index - .12, index + .12, len(delta)),
                       alpha=.6, color=colors[models.index(model)], s=24)
            summary = report["paired"][model]
            mean = summary["mean_capped_difference_seconds"]
            low, high = summary["bootstrap_mean_capped_difference_ci_95"]
            ax.plot([low, high], [index, index], color="black", lw=3)
            ax.scatter([mean], [index], marker="D", color="black", zorder=5)
        ax.axvline(0, color="#888", ls="--")
        ax.set_yticks(range(len(other_models)), other_models)
        ax.set(xlabel="Δ tiempo de espera limitado: PCA − base (s)",
               title="Horizonte común por pareja de semillas\nPuntos: semillas; rombo: media; barra: IC bootstrap 95%")
        ax.grid(axis="x", alpha=.2)
        save(fig, "paired_common_horizon")


def _table(headers, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers)
            + "</tr></thead><tbody>" + "".join("<tr>" + "".join(
                f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows)
            + "</tbody></table>")


def _notebook(report, output, root, path):
    def md(text):
        return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}
    def code(text, data=None):
        return {"cell_type": "code", "metadata": {}, "execution_count": None,
                "source": text.splitlines(True), "outputs": [
                    {"output_type": "display_data", "metadata": {}, "data": data}] if data else []}
    relative = output.relative_to(root).as_posix()
    cells = [md("# Tiempo hasta alcanzar una calidad de validación equivalente\n\n"
        "Análisis retrospectivo de las diez semillas a 5000 pasos. Los outputs se generan desde los CSV guardados, "
        "sin volver a entrenar. El umbral usa únicamente validación; no se lee el error de prueba.\n\n"
        f'**RMSE objetivo: {report["target_rmse"]:.8f}**, media del RMSE final de la base. '
        f'La mediana final de la base es {report["baseline_final_validation_distribution"]["median"]:.4f}; '
        "la media elegida es más exigente por la asimetría entre semillas. "
        "La regla se eligió después de observar este experimento: los resultados son exploratorios. "
        "Se puede fijar este valor antes de una réplica con semillas nuevas.\n\n"
        "Se registra el primer checkpoint con RMSE ≤ objetivo. El cruce ocurrió como máximo en ese checkpoint; "
        "no se interpola entre evaluaciones ni se exige permanencia por debajo del umbral. "
        "Los tiempos incluyen entrenamiento, validación periódica y ajuste PCA; excluyen compilación JAX, "
        "generación de minibatches y evaluación final de prueba. Los tiempos de PCA ya incluyen el ajuste una sola vez.\n\n"
        "Las corridas que no alcanzan el objetivo quedan censuradas al final. No se les asigna su duración final "
        "como si hubieran alcanzado el umbral. Igual RMSE no garantiza igual residuo PDE."),
        code("from pathlib import Path\nimport json\nfrom IPython.display import display, HTML, SVG\n\n"
             "ROOT = Path.cwd()\nwhile not (ROOT / 'pyproject.toml').exists() and ROOT != ROOT.parent:\n    ROOT = ROOT.parent\n"
             f"RUN_DIR = ROOT / '{relative}'\n"
             "REPORT = json.loads((RUN_DIR / 'summary.json').read_text(encoding='utf-8'))\n"
             "def table(headers, rows):\n    import html\n    return '<table><tr>' + ''.join(f'<th>{html.escape(str(v))}</th>' for v in headers) + '</tr>' + ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in row) + '</tr>' for row in rows) + '</table>'")]
    headers = ["Modelo", "Alcanza / total", "Censuradas", "Mediana s solo aciertos", "Mediana pasos solo aciertos"]
    rows = [[m, f'{s["hit_count"]}/{s["seeds"]}', s["censored_count"],
             f'{s["median_seconds_if_hit"]:.3f}' if s["median_seconds_if_hit"] is not None else "—",
             s["median_steps_if_hit"] or "—"] for m, s in report["models"].items()]
    cells += [md("## Alcance del objetivo\n\nLas medianas condicionales describen solo los aciertos y no permiten comparar velocidad general entre modelos con tasas de éxito distintas."),
              code("rows = [[m, f'{s[\"hit_count\"]}/{s[\"seeds\"]}', s['censored_count'], s['median_seconds_if_hit'], s['median_steps_if_hit']] for m,s in REPORT['models'].items()]\ndisplay(HTML(table(['Modelo','Alcanza / total','Censuradas','Mediana s solo aciertos','Mediana pasos solo aciertos'], rows)))",
                   {"text/html": _table(headers, rows)})]
    paired_headers = ["PCA", "Ambos alcanzan", "Solo PCA", "Solo base", "Ninguno", "Δ limitado medio (s)", "IC 95%"]
    paired_rows = [[m, s["both_hit_count"], s["only_other_hit_count"], s["only_base_hit_count"], s["neither_hit_count"],
                    f'{s["mean_capped_difference_seconds"]:.3f}',
                    ', '.join(f'{v:.3f}' for v in s["bootstrap_mean_capped_difference_ci_95"])]
                   for m, s in report["paired"].items()]
    global_hits = ', '.join(f'{m}: {s["hit_count_within_global_horizon"]}/{s["seeds"]}'
                            for m, s in report["models"].items())
    cells += [md("## Comparación pareada con censura\n\n"
        f'El horizonte de tiempo observado por todas las corridas es {report["global_common_horizon_seconds"]:.3f} s. '
        f'Aciertos dentro de ese horizonte: {global_hits}.\n\n'
        "Como diagnóstico adicional, para cada pareja se usa τ = mínimo de sus duraciones finales y se calcula "
        "min(tiempo al objetivo, τ). Para una corrida sin cruce durante ese intervalo, el tiempo limitado vale τ; "
        "el tiempo real al objetivo sigue siendo desconocido. Se compara PCA − base dentro de cada semilla y se "
        "remuestrean parejas completas para el IC bootstrap. Los horizontes varían entre semillas y fueron definidos "
        "por los tiempos observados: esta comparación es descriptiva, no una estimación con un presupuesto fijo preestablecido."),
        code("rows = [[m,s['both_hit_count'],s['only_other_hit_count'],s['only_base_hit_count'],s['neither_hit_count'],s['mean_capped_difference_seconds'],s['bootstrap_mean_capped_difference_ci_95']] for m,s in REPORT['paired'].items()]\ndisplay(HTML(table(['PCA','Ambos alcanzan','Solo PCA','Solo base','Ninguno','Δ limitado medio (s)','IC 95%'], rows)))",
             {"text/html": _table(paired_headers, paired_rows)})]
    both = []
    lookup = {(r["model"], r["seed"]): r for r in report["records"]}
    for row in report["paired_records"]:
        if row["both_hit"]:
            b = lookup[(report["baseline_model"], row["seed"])]
            o = lookup[(row["model"], row["seed"])]
            both.append([row["model"], row["seed"], f'{b["hit_pipeline_seconds"]:.3f}',
                         f'{o["hit_pipeline_seconds"]:.3f}',
                         f'{row["other_minus_base_seconds_if_both_hit"]:.3f}'])
    cells += [md("## Parejas que alcanzaron el objetivo en ambos modelos\n\n"
                 "Esta tabla es condicional al éxito de ambos y no representa a las semillas que fallan."),
              code("lookup = {(r['model'],r['seed']):r for r in REPORT['records']}\nrows = [[r['model'],r['seed'],lookup[(REPORT['baseline_model'],r['seed'])]['hit_pipeline_seconds'],lookup[(r['model'],r['seed'])]['hit_pipeline_seconds'],r['other_minus_base_seconds_if_both_hit']] for r in REPORT['paired_records'] if r['both_hit']]\ndisplay(HTML(table(['Modelo','Semilla','Base (s)','PCA (s)','PCA − base (s)'], rows)))",
                   {"text/html": _table(["Modelo", "Semilla", "Base (s)", "PCA (s)", "PCA − base (s)"], both)})]
    for name in ("success_vs_steps", "time_to_target_by_seed", "paired_common_horizon"):
        cells.append(code(f"display(SVG(filename=str(RUN_DIR / '{name}.svg')))",
                          {"image/svg+xml": (output / f"{name}.svg").read_text(encoding="utf-8")}))
    hit_description = '; '.join(f'{m}: {s["hit_count"]}/{s["seeds"]}'
                                for m, s in report["models"].items())
    both_description = (' '.join(f'{r[0]}, semilla {r[1]}: base {r[2]} s, PCA {r[3]} s.' for r in both)
                        or 'No hay parejas con cruce en ambos modelos.')
    cells += [md("## Interpretación\n\n"
        f'Aciertos en {report["final_step"]} pasos: {hit_description}. '
        f'Parejas que alcanzan el umbral en ambos modelos: {both_description}\n\n'
        "Una mayor tasa de éxito dentro del presupuesto de pasos es evidencia de capacidad para alcanzar esta "
        "calidad, pero por sí sola no basta para concluir una aceleración general. El diagnóstico con horizontes por pareja "
        "aprovecha información de las corridas censuradas y debe leerse junto a su variación de presupuestos.\n\n"
        "Una confirmación debe mantener este umbral fijo, usar semillas nuevas y medir un presupuesto de tiempo "
        "común definido antes de correr. Estos datos no contienen el residuo PDE en el instante del cruce; "
        "la conclusión de calidad equivalente se limita al RMSE de validación.\n\n"
        "Regenerar: `python scripts/analyze_high_dim_time_to_target.py --config configs/high_dim_time_to_target.json`. "
        "Requiere Matplotlib, incluido en las dependencias opcionales de notebooks. Los CSV, el resumen y las "
        "figuras SVG/PNG se conservan en la carpeta de resultados del análisis.")]
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")


def run_analysis(config, root):
    root = Path(root).resolve()
    if config["target_rule"] != "baseline_final_validation_mean":
        raise ValueError("Regla de selección del umbral no soportada.")
    source = root / config["source_run"]
    trajectory_path = source / "validation_trajectories.csv"
    with trajectory_path.open(encoding="utf-8", newline="") as stream:
        trajectories = list(csv.DictReader(stream))
    report = analyze_trajectories(trajectories, config["baseline_model"],
                                  config["bootstrap_draws"], config["bootstrap_seed"])
    report["configuration"] = config
    report["source_sha256"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                for name in ("validation_trajectories.csv", "config.json")}
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__}
    report["timing_scope"] = "Training + periodic validation + PCA fitting; excludes JAX compilation, batch generation and test scoring."
    report["analysis_scope"] = "Retrospective, validation-only target; censored runs kept; pair-specific capped-time diagnostic."
    output = root / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("time_to_target.csv", report["records"]),
                          ("paired_time_to_target.csv", report["paired_records"])):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    _figures(report, output)
    _notebook(report, output, root, root / config["notebook"])
    return report
