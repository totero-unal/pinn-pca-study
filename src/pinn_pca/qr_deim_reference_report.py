"""Auditable plots and notebook for the separate QR-DEIM reference."""
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .notebook_images import embed_notebook_images
from .qr_deim_reference import select_qr, prune, validate_config


def audit_results(config, output):
    """Check actual saved runs, not just isolated selector examples."""
    validate_config(config)
    checked_histories, checked_snapshots = 0, 0
    for seed in config["seeds"]:
        sets = []
        traces = []
        for method in ("qr_deim", "sobol_resample"):
            folder = output / f"seed_{seed}" / method
            data = np.load(folder / "points.npz")
            active = data["active"]
            assert active.shape == (config["steps"] // config["period"], config["active_points"], 2)
            for points in active:
                assert len(set(map(tuple, points))) == config["active_points"]
            sets.append(data)
            trace = list(csv.DictReader((folder / "trace.csv").open()))
            traces.append(trace)
            for row in csv.DictReader((folder / "metrics.csv").open()):
                if row["checkpoint"] == "best_validation":
                    prefix = [r for r in trace if int(r["step"]) <= int(row["budget"])]
                    best = min(prefix, key=lambda r: float(r["validation_mse"]))
                    assert int(row["selected_step"]) == int(best["step"])
                assert float(row["ic_max"]) < 1e-6 and float(row["bc_max"]) < 1e-6
            checked_histories += 1
        for key in ("validation", "independent_test"):
            np.testing.assert_array_equal(sets[0][key], sets[1][key])
        np.testing.assert_array_equal(sets[0]["active"][0], sets[1]["active"][0])
        np.testing.assert_allclose(float(traces[0][0]["validation_mse"]), float(traces[1][0]["validation_mse"]), rtol=1e-6)
        updates = list(csv.DictReader((output / f"seed_{seed}/qr_deim/updates.csv").open()))
        history = np.concatenate((sets[0]["active"], sets[0]["final"][None]))
        for i, row in enumerate(updates):
            before, after = set(map(tuple, history[i])), set(map(tuple, history[i+1]))
            assert len(before - after) == len(after - before) == int(row["replaced"])
        if seed == config["seeds"][0]:
            for step in config["audit_steps"]:
                a = np.load(output / f"seed_{seed}/qr_deim/audit_{step}.npz")
                assert a["snapshots"].shape == (config["snapshot_points"], config["period"])
                indices, singular, _ = select_qr(a["snapshots"], config["epsilon"])
                np.testing.assert_array_equal(indices, a["selected"])
                np.testing.assert_allclose(singular, a["singular_values"])
                updated, baseline, removed, _, accepted = prune(a["before"], a["candidates"], indices,
                    a["old"], a["current"], np.abs(a["snapshots"][:, -1]))
                np.testing.assert_array_equal(updated, a["after"])
                np.testing.assert_array_equal(baseline, a["next_baseline"])
                np.testing.assert_array_equal(removed, a["removed"])
                assert accepted == bool(a["accepted"])
                checked_snapshots += 1
    result = dict(passed=True, checked_histories=checked_histories, reconstructed_snapshot_updates=checked_snapshots,
                  checks=["fixed_unique_active_count", "paired_initial_points_and_validation", "paired_first_period",
                          "replacement_count", "best_checkpoint_selected_without_test", "hard_ic_bc", "SVD_QR_pruning_reconstruction"])
    (output / "audit_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def report(config, root):
    root = Path(root).resolve()
    output = root / config["output"]
    audit_results(config, output)
    rows = list(csv.DictReader((output / "metrics.csv").open(encoding="utf-8")))
    expected = len(config["seeds"]) * 2 * len(config["budgets"]) * 2
    if len(rows) != expected:
        raise ValueError(f"Incomplete metrics: {len(rows)} of {expected}")
    methods = ["qr_deim", "sobol_resample"]
    labels = ["QR-DEIM", "Sobol con remuestreo"]
    def values(method, budget, metric, checkpoint="best_validation"):
        selected = sorted([r for r in rows if r["method"] == method and int(r["budget"]) == budget
                           and r["checkpoint"] == checkpoint], key=lambda r: int(r["seed"]))
        if [int(r["seed"]) for r in selected] != sorted(config["seeds"]):
            raise ValueError("Missing or duplicated seeds")
        result = np.array([float(r[metric]) for r in selected])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite report metric")
        return result
    summary = {}
    for method in methods:
        summary[method] = {}
        for checkpoint in ("last", "best_validation"):
            summary[method][checkpoint] = {m: {
                "mean": float(values(method, config["steps"], m, checkpoint).mean()),
                "std": float(values(method, config["steps"], m, checkpoint).std(ddof=1)) if len(config["seeds"]) > 1 else 0.
            } for m in ("rmse", "relative_l2", "residual_rms", "ic_max", "bc_max")}
    difference = values(methods[0], config["steps"], "rmse") - values(methods[1], config["steps"], "rmse")
    summary["paired"] = dict(qr_rmse_wins=int(np.sum(difference < 0)), n=len(difference),
                             mean_rmse_qr_minus_sobol=float(difference.mean()),
                             differences=difference.tolist())
    updates = []
    for seed in config["seeds"]:
        updates.extend(csv.DictReader((output / f"seed_{seed}/qr_deim/updates.csv").open(encoding="utf-8")))
    summary["selection"] = dict(min_k=min(int(r["k"]) for r in updates), max_k=max(int(r["k"]) for r in updates),
                                mean_k=float(np.mean([int(r["k"]) for r in updates])),
                                skipped=sum(r["accepted"] == "False" for r in updates), total=len(updates))
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    def save(fig, name):
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=150)
        fig.savefig(output / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, label in zip(methods, labels):
        for ax, metric, title in zip(axes, ["rmse", "residual_rms"], ["RMSE de solución", "RMS del residuo independiente"]):
            a = np.array([values(method, b, metric) for b in config["budgets"]])
            mean = a.mean(axis=1)
            std = a.std(axis=1, ddof=1) if a.shape[1] > 1 else np.zeros(len(a))
            ax.plot(config["budgets"], mean, "o-", label=label)
            ax.fill_between(config["budgets"], np.maximum(mean-std, 0), mean+std, alpha=.15)
            ax.set(xlabel="Pasos totales (sin warmup)", ylabel=title)
            ax.grid(alpha=.2)
    axes[0].legend()
    fig.suptitle("Mejor checkpoint por validación física dentro de cada presupuesto; media ± DE")
    save(fig, "budget_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, checkpoint in zip(axes, ("best_validation", "last")):
        a, b = [values(m, config["steps"], "rmse", checkpoint) for m in methods]
        for ai, bi in zip(a, b):
            ax.plot([0, 1], [ai, bi], "o-", alpha=.65)
        ax.set(xticks=[0, 1], xticklabels=labels, ylabel="RMSE", title=checkpoint)
        ax.grid(alpha=.2)
    save(fig, "paired_rmse")

    first = config["seeds"][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method, label in zip(methods, labels):
        traces = [list(csv.DictReader((output / f"seed_{s}/{method}/trace.csv").open())) for s in config["seeds"]]
        for trace in traces:
            axes[0].semilogy([int(r["step"]) for r in trace], [float(r["validation_mse"]) for r in trace], alpha=.18)
        mean = np.mean([[float(r["validation_mse"]) for r in tr] for tr in traces], axis=0)
        axes[0].semilogy([int(r["step"]) for r in traces[0]], mean, label=label, linewidth=2)
    seed_updates = list(csv.DictReader((output / f"seed_{first}/qr_deim/updates.csv").open()))
    axes[1].plot([int(r["step"]) for r in seed_updates], [int(r["k"]) for r in seed_updates], label="k propuesto")
    axes[1].plot([int(r["step"]) for r in seed_updates], [int(r["replaced"]) for r in seed_updates], label="Reemplazados", linestyle="--")
    axes[0].set(xlabel="Paso", ylabel="MSE de validación física")
    axes[1].set(xlabel="Paso", ylabel="Puntos", title=f"Selección dinámica, semilla {first}")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    save(fig, "validation_and_rank")

    audit_step = config["audit_steps"][0]
    audit = np.load(output / f"seed_{first}/qr_deim/audit_{audit_step}.npz")
    s, candidates = audit["singular_values"], audit["candidates"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    image = axes[0].scatter(candidates[:, 0], candidates[:, 1], c=np.abs(audit["snapshots"][:, -1]), s=15)
    fig.colorbar(image, ax=axes[0], label="|r| al final del periodo")
    axes[0].set(xlabel="x", ylabel="t", title=f"Candidatos, paso {audit_step}")
    axes[1].semilogy(np.arange(1, len(s)+1), np.maximum(s, 1e-16), "o-", markersize=2)
    axes[1].set(xlabel="Índice", ylabel="Valor singular")
    right = axes[1].twinx()
    right.plot(np.arange(1, len(s)+1), np.cumsum(s)/s.sum() if s.sum() else np.zeros_like(s), color="orange")
    right.axhline(1-config["epsilon"], color="orange", linestyle=":")
    right.set_ylabel("Suma acumulada σ / suma σ")
    selected = candidates[audit["selected"]]
    axes[2].scatter(audit["before"][:, 0], audit["before"][:, 1], s=10, label="Activos antes", alpha=.6)
    axes[2].scatter(selected[:, 0], selected[:, 1], marker="x", color="red", label="Propuestos QR")
    removed = audit["before"][audit["removed"]]
    axes[2].scatter(removed[:, 0], removed[:, 1], facecolors="none", edgecolors="black", s=50, label="Podados")
    axes[2].set(xlabel="x", ylabel="t", title=f"k={len(selected)}, aceptado={bool(audit['accepted'])}")
    axes[2].legend(fontsize=8)
    save(fig, "selection_audit")

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7))
    for i, ax in enumerate(axes):
        if i < audit["modes"].shape[1]:
            im = ax.scatter(candidates[:, 0], candidates[:, 1], c=audit["modes"][:, i], cmap="coolwarm", s=15)
            fig.colorbar(im, ax=ax)
        ax.set(xlabel="x", ylabel="t", title=f"Modo espacial {i+1} sin centrar")
    save(fig, "residual_modes")

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for i, method in enumerate(methods):
        field = np.load(output / f"seed_{first}/{method}/field_{config['steps']}_best_validation.npz")
        extent = [float(field["x"].min()), float(field["x"].max()), float(field["t"].min()), float(field["t"].max())]
        for j, z in enumerate((field["prediction"], np.abs(field["prediction"]-field["reference"]))):
            im = axes[i,j].imshow(z.T, origin="lower", extent=extent, aspect="auto", cmap="coolwarm" if j == 0 else "magma")
            fig.colorbar(im, ax=axes[i,j])
            axes[i,j].set(xlabel="x", ylabel="t", title=f"{labels[i]}: {'u predicha' if j == 0 else '|error|'}")
    save(fig, "solution_fields")

    table = "| Método | RMSE medio ± DE | Residuo RMS medio |\n|---|---:|---:|\n"
    for method, label in zip(methods, labels):
        item = summary[method]["best_validation"]
        table += f"| {label} | {item['rmse']['mean']:.6f} ± {item['rmse']['std']:.6f} | {item['residual_rms']['mean']:.6f} |\n"
    notebook = root / config["notebook"]
    cells = []
    def md(text):
        cells.append(dict(cell_type="markdown", metadata={}, source=text))
    def figure(name):
        md(f"![{name}]({Path(os.path.relpath(output / (name+'.png'), notebook.parent)).as_posix()})")
    md("# QR-DEIM de referencia: Burgers con condiciones duras\n\n"
       "Esta implementación sigue el algoritmo 1 de [Celaya et al., versión 4](https://arxiv.org/html/2501.07700v4). "
       "Es una comparación nueva y separada de POD/PCA híbridos: no reutiliza sus pesos ni sus pérdidas. "
       f"Configuración: `{config['description']}`. Resultados reales de {len(config['seeds'])} semillas; no son las cifras publicadas del artículo.")
    md("## Qué cambia\n\nLa solución es $u_\\theta(x,t)=t(x+1)(x-1)N_\\theta(x,t)-\\sin(\\pi x)$. "
       "Esto impone la condición inicial y fronteras homogéneas (con corrección numérica del seno en los extremos). "
       "La pérdida es únicamente $\\operatorname{mean}(r_\\theta^2)$, con $\\nu=0.01/\\pi$. "
       "No se usan etiquetas de solución durante entrenamiento ni para elegir checkpoints.\n\n"
       "1. Cada actualización Adam produce una columna de residuos **con signo**, sobre candidatos fijos durante ese periodo.\n"
       "2. Se aplica SVD completa sin centrar a $R$; se escoge el menor $k$ con $\\sum_{j\\le k}\\sigma_j/\\sum_j\\sigma_j\\ge1-\\varepsilon$. No se elevan al cuadrado los valores singulares.\n"
       "3. QR con pivoteo de columnas de $V_k^T$ propone exactamente $k$ candidatos. No hay relleno Sobol ni selección log-det.\n"
       "4. Se podan los $k$ activos con mayor $d=\\log_2(|r_{old}|/|r_{new}|)$. Si uno de esos grados es negativo, se cancela toda la sustitución.\n"
       "5. Se renuevan los candidatos y se reinicia la historia. Los residuos de referencia quedan alineados con sobrevivientes y puntos insertados.\n\n"
       "Se usa piso 1e-12 en la razón; un espectro totalmente nulo produce k=0. El conjunto candidato y activo son disjuntos. "
       "La actualización del último periodo se registra, pero sus puntos nuevos no se usan para entrenar.")
    md("## Comparación y alcance\n\n"
       f"Ambos métodos: {config['active_points']} puntos activos, {config['hidden_layers']} capas ocultas de {config['width']} neuronas, "
       f"{config['steps']} pasos totales sin warmup, Adam desde {config['learning_rate']}, coseno hasta cero, mismo estado inicial por semilla. "
       f"Periodo {config['period']}; QR-DEIM observa {config['snapshot_points']} candidatos en **cada paso**; epsilon={config['epsilon']}. "
       "Sobol comparte el conjunto uniforme inicial durante el primer periodo y después reemplaza todos sus puntos en cada periodo. "
       "QR-DEIM solo reemplaza k cuando la poda se acepta.\n\n"
       f"La selección del mejor checkpoint usa {config['validation_points']} puntos uniformes fijos, cada periodo; "
       f"el residuo de prueba usa otros {config['test_points']} puntos. El error de solución usa la malla 256×100 del archivo de referencia "
       "(t llega a 0.99), que nunca entra en el entrenamiento. Los presupuestos se fijaron antes de ejecutar. "
       "También se informa el último modelo: seleccionar por residuo no garantiza minimizar error de solución.\n\n"
       "La configuración paper conserva 2000 activos, 1000 candidatos, periodos de 1000, cinco capas de 64, "
       "100000 pasos y 10000 puntos de validación; está disponible pero este piloto no ejecuta ese benchmark. "
       "JAX/CPU, inicialización, validación por periodo y detalles del coseno impiden prometer reproducción numérica exacta de PyTorch/GPU. "
       "Los tiempos registrados incluyen compilación y diagnóstico: no prueban aceleración.")
    md("## Resultados\n\n" + table + f"\nQR-DEIM obtiene menor RMSE que Sobol en {summary['paired']['qr_rmse_wins']}/{len(difference)} semillas "
       "al elegir por validación física. La desviación estándar representa dispersión entre semillas, no un intervalo de confianza. "
       "Este piloto solo permite conclusiones bajo su tamaño y presupuesto; no demuestra superioridad universal ni reproduce la tabla del artículo.")
    figure("budget_comparison")
    figure("paired_rmse")
    md(f"## Auditoría del selector\n\nRango observado: {summary['selection']['min_k']}–{summary['selection']['max_k']}; "
       f"media {summary['selection']['mean_k']:.2f}. Actualizaciones canceladas o sin rango: "
       f"{summary['selection']['skipped']}/{summary['selection']['total']}. "
       "Los archivos audit guardan la matriz completa de residuos, sus modos, índices QR y puntos podados en periodos seleccionados de la primera semilla.")
    for name in ("validation_and_rank", "selection_audit", "residual_modes", "solution_fields"):
        figure(name)
    md("## Reproducir y navegar\n\nDesde la raíz del repositorio:\n\n"
       "```bash\npython scripts/run_qr_deim_reference.py --config pca_collocation/configs/burgers_qr_deim_reference_pilot.json\n"
       "python scripts/run_qr_deim_reference.py --report-only\n```\n\n"
       "Las corridas completas se reutilizan; las incompletas se reinician desde su semilla. El estado final de Adam se guarda, "
       "pero no se implementa reanudación desde un periodo intermedio. Para cambiar parámetros, usar un nuevo directorio output. "
       f"Artefactos: `{config['output']}`. `metrics.csv` conserva ambos criterios; `summary.json` contiene diferencias pareadas; "
       "cada semilla tiene trazas, historial de puntos, parámetros y auditoría de selección.\n\n"
       "Siguiente control: ampliar los tamaños hacia la configuración paper manteniendo el protocolo, antes de atribuir las diferencias al método publicado.")
    notebook.parent.mkdir(parents=True, exist_ok=True)
    notebook.write_text(json.dumps(dict(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, nbformat=4, nbformat_minor=5), ensure_ascii=False, indent=2), encoding="utf-8")
    embed_notebook_images(notebook, root)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
