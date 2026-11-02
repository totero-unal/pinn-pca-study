"""Paired ablation of the actual fraction of residual-basis collocation points."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .collocation import _mean_std, _run_collocation_seed, _save_csv


def run_qr_fraction_ablation(config, project_root, summarize_only=False):
    root = Path(project_root).resolve()
    output = root / "pca_collocation/results/burgers/collocation_selection" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    counts = config["basis_point_counts"]
    seeds = config["seeds"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("Seeds must be nonempty and distinct.")
    if len(set(counts)) != len(counts) or any(not 0 < n <= config["batch_size"] for n in counts):
        raise ValueError("Basis counts must be distinct and fit the active batch.")
    specs = {"sobol": {"base": "sobol", "label": "Sobol (0 puntos de modos)"}}
    for base, label in (("pod_qr", "POD"), ("pca_mean_qr", "PCA + media")):
        specs[base + "_rank"] = {"base": base, "label": label + " + QR por rango"}
        for count in counts:
            specs[f"{base}_{count}"] = {"base": base, "label": f"{label}: {count} puntos", "oversample_count": count}
    run_config = dict(config, method_specs=specs, common_selection_seed=True, skip_seed_plots=True)
    if summarize_only:
        saved = json.loads((output / "config.json").read_text(encoding="utf-8"))
        if saved != run_config:
            raise ValueError("Saved configuration does not match; summary regeneration refused.")
    else:
        (output / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    tables = {name: [] for name in ("metrics", "validation_trajectories", "selection_diagnostics")}
    for seed in seeds:
        seed_dir = output / f"seed_{seed}"
        if not summarize_only:
            _run_collocation_seed(dict(run_config, seed=seed), root, seed_dir)
        for name, rows in tables.items():
            with (seed_dir / f"{name}.csv").open(encoding="utf-8", newline="") as file:
                rows.extend(csv.DictReader(file))
        print(f"{'Loaded' if summarize_only else 'Completed'} seed {seed} ({len(specs)} variants)", flush=True)
    for name, rows in tables.items():
        _save_csv(output / f"{name}_by_seed.csv", rows)
    metrics = tables["metrics"]
    traces = tables["validation_trajectories"]
    diagnostics = tables["selection_diagnostics"]
    fields = ("test_rmse", "test_relative_l2", "independent_residual_rms", "initial_condition_rmse", "boundary_rms", "pipeline_seconds_including_shared_setup")
    target = float(config["validation_rmse_target"])
    stability_count = int(config.get("stability_checkpoints", 5))
    aggregates, paired, target_rows = [], [], []
    for method, spec in specs.items():
        rows = [r for r in metrics if r["method"] == method]
        aggregate = {"method": method, "label": spec["label"], "seed_count": len(rows)}
        aggregate.update({key: _mean_std([float(r[key]) for r in rows]) for key in fields})
        selection_counts = [int(r["selected_qr_points"]) for r in diagnostics if r["method"] == method]
        aggregate["actual_basis_points"] = {"min": min(selection_counts), "max": max(selection_counts), "mean": float(np.mean(selection_counts))}
        hit_count, final_count, stable_count = 0, 0, 0
        for seed in seeds:
            trajectory = sorted([r for r in traces if r["method"] == method and int(r["seed"]) == seed], key=lambda r: int(r["step"]))
            hits = [r for r in trajectory if float(r["validation_rmse"]) <= target]
            hit_count += bool(hits)
            final_rmse = float(trajectory[-1]["validation_rmse"])
            final_count += final_rmse <= target
            stable = len(trajectory) >= stability_count and all(float(r["validation_rmse"]) <= target for r in trajectory[-stability_count:])
            stable_count += stable
            target_rows.append({"seed": seed, "method": method, "target": target, "reached": bool(hits),
                                "step_at_target": int(hits[0]["step"]) if hits else None,
                                "pipeline_seconds_at_target": float(hits[0]["pipeline_seconds"]) if hits else None,
                                "censor_step": None if hits else int(trajectory[-1]["step"]),
                                "final_validation_rmse": final_rmse,
                                "final_below_target": final_rmse <= target,
                                "last_checkpoints_below_target": stable,
                                "stability_checkpoint_count": stability_count})
        aggregate["target_hits"] = hit_count
        aggregate["final_below_target_count"] = final_count
        aggregate["stable_target_count"] = stable_count
        aggregates.append(aggregate)
        if method == "sobol":
            continue
        references = ["sobol"]
        if "oversample_count" in spec:
            references.append(spec["base"] + "_rank")
        for reference in references:
            for key in fields:
                differences = []
                for row in rows:
                    baseline = next(r for r in metrics if r["method"] == reference and r["seed"] == row["seed"])
                    differences.append(float(row[key]) - float(baseline[key]))
                paired.append({"method": method, "reference": reference, "metric": key,
                               **_mean_std(differences), "better_count": int(np.sum(np.asarray(differences) < 0))})
    _save_csv(output / "paired_differences.csv", paired)
    _save_csv(output / "time_to_target.csv", target_rows)
    summary = {"run_id": config["run_id"], "seeds": seeds, "methods": aggregates, "validation_target": target,
               "selection_rule": "Rank-limited pivoted QR initializes a greedy log-det design on the retained residual basis; the remaining slots use a nested common scrambled Sobol prefix.",
               "timing_note": "Measured pipeline time includes shared warmup, candidate histories, selection and evaluations. Fixed execution order and first-use compilation limit timing comparisons; quality is the primary endpoint.",
               "protocol": {key: config[key] for key in ("warmup_steps", "train_steps", "batch_size", "update_every", "candidate_count", "history_snapshots", "energy_threshold", "max_modes")},
               "stability_checkpoints": stability_count,
               "scope": f"{config['warmup_steps']} warmup + {config['train_steps']} continuation steps; basis_point_counts are actual points, not mode counts. Zero-information histories fall back to Sobol and actual counts expose this.",
               "paired_differences": paired}
    if config.get("reference_run_id"):
        summary["budget_comparison"] = _compare_budgets(output, config, metrics, traces, fields)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _plot(output, specs, metrics, traces, seeds, target)
    _notebook(root, output, summary, config)
    return output


def _plot(output, specs, metrics, traces, seeds, target):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for base, label, color in (("pod_qr", "POD", "tab:red"), ("pca_mean_qr", "PCA + media", "tab:purple")):
        variants = ["sobol", base + "_rank"] + [key for key in specs if key.startswith(base + "_") and "oversample_count" in specs[key]]
        for ax, metric in zip(axes, ("test_rmse", "independent_residual_rms")):
            means, deviations = [], []
            for method in variants:
                values = [float(r[metric]) for r in metrics if r["method"] == method]
                means.append(np.mean(values))
                deviations.append(np.std(values, ddof=1))
            ax.errorbar(range(len(variants)), means, yerr=deviations, marker="o", capsize=3, label=label, color=color)
            ax.set_xticks(range(len(variants)), ["Sobol", "QR por rango"] + [str(specs[k]["oversample_count"]) for k in variants[2:]])
            ax.set_xlabel("Puntos elegidos por los modos (lote total: 128)")
            ax.grid(alpha=.2)
    axes[0].set_ylabel("RMSE de solución en prueba")
    axes[1].set_ylabel("RMS del residuo independiente")
    axes[0].legend()
    fig.suptitle(f"Fracción de puntos: media ± DE, {len(seeds)} semillas")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"fraction_ablation.{extension}", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, base, label in zip(axes, ("pod_qr", "pca_mean_qr"), ("POD", "PCA + media")):
        variants = ["sobol", base + "_rank"] + [key for key in specs if key.startswith(base + "_") and "oversample_count" in specs[key]]
        for method in variants:
            steps = sorted({int(row["step"]) for row in traces if row["method"] == method})
            values = [[float(row["validation_rmse"]) for row in traces if row["method"] == method and int(row["step"]) == step] for step in steps]
            ax.plot(steps, np.mean(values, axis=1), marker="o", label=specs[method]["label"])
        ax.axhline(target, color="black", linestyle="--", linewidth=1, label="Umbral 0.35")
        ax.set(title=label, xlabel="Pasos totales", ylabel="RMSE medio de validación")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("El primer cruce del umbral puede ser transitorio")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(output / f"fraction_validation.{extension}", dpi=180)
    plt.close(fig)


def _compare_budgets(output, config, metrics, traces, fields):
    reference = output.parent / config["reference_run_id"]
    prior_config = json.loads((reference / "config.json").read_text(encoding="utf-8"))
    for key in ("seeds", "width", "learning_rate", "warmup_steps", "batch_size", "update_every", "candidate_count", "history_snapshots", "energy_threshold", "max_modes", "basis_point_counts"):
        if prior_config[key] != config[key]:
            raise ValueError(f"Budget comparison is not paired: {key} differs.")
    with (reference / "metrics_by_seed.csv").open(encoding="utf-8", newline="") as file:
        previous = {(r["method"], r["seed"]): r for r in csv.DictReader(file)}
    with (reference / "validation_trajectories_by_seed.csv").open(encoding="utf-8", newline="") as file:
        prior_traces = {(r["method"], r["seed"], r["step"]): r for r in csv.DictReader(file)}
    shared_differences = [abs(float(row["validation_rmse"]) - float(prior_traces[(row["method"], row["seed"], row["step"])]["validation_rmse"]))
                          for row in traces if (row["method"], row["seed"], row["step"]) in prior_traces]
    if max(shared_differences) > 1e-6:
        raise ValueError("Shared early validation trajectory changed; budget comparison is not a continuation of the same protocol.")
    differences, aggregate = [], []
    for row in metrics:
        for key in fields:
            if "seconds" in key:
                continue
            old = float(previous[(row["method"], row["seed"])][key])
            differences.append({"seed": row["seed"], "method": row["method"], "metric": key,
                                "short_budget_value": old, "long_budget_value": float(row[key]),
                                "long_minus_short": float(row[key]) - old})
    for method in dict.fromkeys(r["method"] for r in metrics):
        for key in (k for k in fields if "seconds" not in k):
            values = [r["long_minus_short"] for r in differences if r["method"] == method and r["metric"] == key]
            aggregate.append({"method": method, "metric": key, **_mean_std(values), "improved_count": sum(v < 0 for v in values)})
    _save_csv(output / "budget_changes_by_seed.csv", differences)
    _save_csv(output / "budget_changes_summary.csv", aggregate)
    return {"reference_run_id": config["reference_run_id"], "short_continuation_steps": prior_config["train_steps"],
            "long_continuation_steps": config["train_steps"], "max_shared_validation_difference": max(shared_differences),
            "paired_changes": aggregate}


def _notebook(root, output, summary, config):
    table = "| Variante | Puntos de modos reales (mín–máx) | RMSE (media ± DE) | Residuo RMS | Primer cruce | Bajo umbral al final | Últimos checkpoints bajo umbral |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for row in summary["methods"]:
        counts, error = row["actual_basis_points"], row["test_rmse"]
        table += f"| {row['label']} | {counts['min']}–{counts['max']} | {error['mean']:.4f} ± {error['std']:.4f} | {row['independent_residual_rms']['mean']:.4f} | {row['target_hits']}/{row['seed_count']} | {row['final_below_target_count']}/{row['seed_count']} | {row['stable_target_count']}/{row['seed_count']} |\n"
    relative = output.relative_to(root).as_posix()
    def md(source):
        return {"cell_type": "markdown", "metadata": {}, "source": source}
    cells = [md(f"# Proporción de puntos seleccionados por la historia del residuo\n\n{len(summary['seeds'])} semillas pareadas, {config['warmup_steps']} pasos de calentamiento y {config['train_steps']} de continuación. Se mantiene un lote de {config['batch_size']} puntos y actualización cada {config['update_every']} pasos. Sobol puro representa cero puntos elegidos por los modos.\n\nQR básico está limitado por el rango retenido: aumentar su límite no garantiza más puntos. Aquí se conservan esas variantes y se añaden 32, 64 y 128 puntos mediante QR inicial seguido por selección voraz de ganancia log-det en los mismos modos. Esto es una extensión de la selección anterior, no una reproducción adicional del artículo QR-DEIM."),
             md("## Resultados al mismo presupuesto\n\n" + table + f"\nEl umbral es RMSE de validación ≤ {summary['validation_target']}; se distingue el primer cruce, el estado final y permanecer bajo él en los últimos {summary['stability_checkpoints']} checkpoints. Si no hay suficientes observaciones, no se acredita estabilidad. Las semillas sin cruce son censuradas. El test no decide el umbral ni la selección."),
             md("## Comparación visual\n\n![Fracción de puntos](../../" + output.relative_to(root / "pca_collocation").as_posix() + "/fraction_ablation.png)"),
             md("## Evolución de validación\n\n![Validación](../../" + output.relative_to(root / "pca_collocation").as_posix() + "/fraction_validation.png)\n\nUn primer cruce cuenta incluso si el entrenamiento posterior empeora. La tabla distingue ese evento de terminar bajo el umbral."),
             md("## Interpretación y límites\n\nEl número de puntos seleccionados se registra en cada actualización; las 32/64/128 filas elegidas son índices distintos del conjunto candidato. El número de modos sigue fijado por energía de 0.99 y máximo 20. El complemento usa el mismo prefijo Sobol por semilla y actualización, también compartido entre POD y PCA.\n\nLas diferencias pareadas se guardan frente a Sobol y frente al QR limitado por rango de cada enfoque. Un menor residuo independiente debe acompañarse de mejora en la solución y las condiciones iniciales y de frontera. El presupuesto corto limita las conclusiones. Los tiempos incluyen evaluaciones y compilación inicial; no se usan para afirmar aceleración."),
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": "import json\nfrom pathlib import Path\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\noutput = root / " + repr(relative) + "\nsummary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))\nsummary['methods']"}]
    if summary.get("budget_comparison"):
        comparison = summary["budget_comparison"]
        text = f"## Cambio al ampliar el entrenamiento\n\nComparación pareada de {comparison['short_continuation_steps']} a {comparison['long_continuation_steps']} pasos de continuación, con las mismas semillas, lotes y protocolo. Máxima diferencia en la validación de los checkpoints compartidos: {comparison['max_shared_validation_difference']:.2g}.\n\n| Variante | Δ RMSE prueba (largo − corto) | Semillas que mejoran |\n|---|---:|---:|\n"
        for row in comparison["paired_changes"]:
            if row["metric"] == "test_rmse":
                text += f"| {row['method']} | {row['mean']:.4f} ± {row['std']:.4f} | {row['improved_count']}/{row['n']} |\n"
        cells.insert(-1, md(text))
        sobol = next(row for row in summary["methods"] if row["method"] == "sobol")
        guided = [row for row in summary["methods"] if row["method"] != "sobol"]
        best_solution = min(guided, key=lambda row: row["test_rmse"]["mean"])
        best_residual = min(guided, key=lambda row: row["independent_residual_rms"]["mean"])
        cells.insert(-1, md(
            "## Lectura del resultado\n\n"
            f"Sobol obtiene RMSE medio de solución {sobol['test_rmse']['mean']:.4f}; la variante guiada con menor media es {best_solution['label']} ({best_solution['test_rmse']['mean']:.4f}). "
            f"El menor residuo independiente entre las variantes guiadas corresponde a {best_residual['label']} ({best_residual['independent_residual_rms']['mean']:.4f}), frente a {sobol['independent_residual_rms']['mean']:.4f} con Sobol. "
            "Las condiciones iniciales y de frontera se conservan en los CSV para analizar este compromiso entre ajuste de la solución y residuo.\n\n"
            f"{sum(row['stable_target_count'] for row in summary['methods'])} de {sum(row['seed_count'] for row in summary['methods'])} combinaciones permanecen bajo el umbral {summary['validation_target']} en los últimos {summary['stability_checkpoints']} controles. "
            "Alcanzar ese umbral no implica igual precisión final. Este experimento no selecciona un umbral nuevo después de observar las pruebas ni demuestra aceleración general."))
    path = root / "pca_collocation/notebooks/burgers" / config.get("notebook_name", "08_qr_fraction_ablation.ipynb")
    path.write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 4}, ensure_ascii=False, indent=1), encoding="utf-8")
    from .notebook_images import embed_notebook_images
    embed_notebook_images(path, root)
