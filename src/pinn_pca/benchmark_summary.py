"""Read-only aggregation of frozen benchmark results and persistent notebook closures."""
from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
from statistics import mean, stdev

REGISTRY = "reports/benchmark_registry.json"
METRICS = (("l2", "test_relative_l2"), ("pde", "test_pde_residual_rms"), ("time", "pipeline_seconds"))


def load_registry(root):
    return json.loads((Path(root) / REGISTRY).read_text(encoding="utf-8"))


def apply_benchmark_closure(notebook, root, benchmark_id, run_id, notebook_path):
    """Add a dated banner/closure only to the explicitly closed runs; idempotent."""
    root, notebook_path = Path(root), Path(notebook_path)
    entry = next(b for b in load_registry(root)["benchmarks"] if b["id"] == benchmark_id)
    relative = notebook_path.relative_to(root).as_posix()
    eligible = any(n["run_id"] == run_id and n["path"] == relative for n in entry.get("closed_notebooks", []))
    if not eligible:
        return False
    closure = (root / entry["closure_file"]).read_text(encoding="utf-8")
    notebook["cells"] = [cell for cell in notebook["cells"] if not (
        cell.get("metadata", {}).get("benchmark_id") == benchmark_id
        and cell.get("metadata", {}).get("benchmark_documentation_kind") in ("status", "closure"))]
    def cell(text, kind):
        return {"cell_type": "markdown", "metadata": {
            "benchmark_id": benchmark_id, "benchmark_documentation_kind": kind,
            "tags": ["benchmark_" + kind]}, "source": text.splitlines(True)}
    link = os.path.relpath(root / "BENCHMARKS.md", notebook_path.parent).replace("\\", "/")
    banner = (f'**Estado: {entry["status_label"]}.** Cierre: {entry["closed_on"]}. '
              f'Conclusiones y límites al final del notebook. [Tabla global de benchmarks]({link}).\n')
    notebook["cells"].insert(0, cell(banner, "status"))
    notebook["cells"].append(cell(closure, "closure"))
    notebook.setdefault("metadata", {})["benchmark"] = {
        "id": benchmark_id, "status": entry["status"], "closed_on": entry["closed_on"],
        "source_run_id": run_id, "closure_file": entry["closure_file"]}
    return True


def _stats(values):
    values = [float(v) for v in values]
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("La tabla requiere métricas finitas y al menos una semilla.")
    return mean(values), stdev(values) if len(values) > 1 else 0.


def comparison_records(registry, root):
    root = Path(root)
    output, cache = [], {}
    for entry in registry["benchmarks"]:
        for spec in entry["comparisons"]:
            source = spec["source"]
            if source not in cache:
                cache[source] = json.loads((root / source).read_text(encoding="utf-8"))
            document = cache[source]
            settings = document.get("source_configuration", document.get("configuration", {}))
            records = [row for row in document[spec["records_key"]]
                       if all(row.get(key) == value for key, value in spec["filters"].items())]
            available = {row["model"] for row in records}
            if available != {spec["baseline"], *spec["variants"]}:
                raise ValueError(f"La tabla debe conservar todas las variantes de {spec['case']}: {available}")
            def by_seed(model):
                selected = [r for r in records if r["model"] == model]
                lookup = {int(r["seed"]): r for r in selected}
                if len(selected) != len(lookup):
                    raise ValueError(f"Semillas duplicadas en {source}: {model}")
                return lookup
            baseline = by_seed(spec["baseline"])
            for variant in spec["variants"]:
                other = by_seed(variant)
                if set(other) != set(baseline) or not baseline:
                    raise ValueError(f"Faltan semillas pareadas en {source}: {variant}")
                seeds = sorted(baseline)
                row = {"benchmark_id": entry["id"], "benchmark": entry["name"],
                       "status": entry["status"], "case": spec["case"], "evidence_role": spec["evidence_role"],
                       "pca_scope": entry["pca_scope"], "pde_metric": entry["pde_metric"],
                       "baseline": spec["baseline"], "variant": variant,
                       "k_values": sorted({int(r["k"]) for r in other.values() if r.get("k") is not None}),
                       "seed_count": len(seeds), "seeds": seeds, "steps": int(settings["steps"]),
                       "source": source, "records_key": spec["records_key"], "filters": spec["filters"],
                       "notebook": spec.get("notebook", entry["notebook"])}
                for alias, metric in METRICS:
                    a = [baseline[s][metric] for s in seeds]
                    b = [other[s][metric] for s in seeds]
                    delta = [float(y) - float(x) for x, y in zip(a, b)]
                    for name, values in (("baseline", a), ("pca", b), ("delta", delta)):
                        m, sd = _stats(values)
                        row[f"{name}_{alias}_mean"], row[f"{name}_{alias}_std"] = m, sd
                    row[f"pca_better_{alias}_count"] = sum(d < 0 for d in delta)
                output.append(row)
    return output


def _html_table(headers, rows, raw_columns=()):
    def encode(value, column):
        return str(value) if column in raw_columns else html.escape(str(value))
    return ('<div style="overflow-x:auto"><table><thead><tr>'
            + ''.join(f'<th>{html.escape(h)}</th>' for h in headers) + '</tr></thead><tbody>'
            + ''.join('<tr>' + ''.join(f'<td>{encode(v,i)}</td>' for i,v in enumerate(row)) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def _anchor(path, label, prefix):
    return f'<a href="{html.escape(prefix+path, quote=True)}">{html.escape(label)}</a>'


def overview_html(report, prefix="../"):
    rows = []
    for b in report["registry"]["benchmarks"]:
        evidence = _anchor(b["notebook"], "Notebook", prefix) if b["notebook"] else "Sin corrida"
        rows.append([b["name"], b["status_label"], b["pca_scope"], b["finding"], evidence])
    return _html_table(["Benchmark", "Estado", "Aplicación de PCA", "Resultado y alcance", "Evidencia"], rows, (4,))


def _format_stat(row, prefix, alias):
    return f'{row[f"{prefix}_{alias}_mean"]:.4g} ± {row[f"{prefix}_{alias}_std"]:.2g}'


def _display_rows(report, benchmark_id, markup, prefix):
    rows = []
    for r in report["comparisons"]:
        if r["benchmark_id"] != benchmark_id:
            continue
        source_label = (f'{_anchor(r["source"],r["case"],prefix)}<br>{html.escape(r["evidence_role"])}'
                        if markup == "html" else f'[{r["case"]}]({prefix+r["source"]}); {r["evidence_role"]}')
        rows.append([source_label, f'{r["variant"]}; k={",".join(map(str,r["k_values"]))}',
                     f'{r["seed_count"]} / {r["steps"]}',
                     *[f'{_format_stat(r,"baseline",alias)} → {_format_stat(r,"pca",alias)}' for alias,_ in METRICS],
                     ' / '.join(f'{r[f"pca_better_{alias}_count"]}/{r["seed_count"]}' for alias,_ in METRICS)])
    return rows


QUANT_HEADERS = ["Caso y evidencia", "Variante", "Semillas / pasos", "L2 base → PCA", "Residuo base → PCA", "Pipeline base → PCA (s)", "PCA menor: L2 / PDE / tiempo"]


def comparison_html(report, benchmark_id, prefix="../"):
    return _html_table(QUANT_HEADERS, _display_rows(report, benchmark_id, "html", prefix), (0,))


def _markdown_table(headers, rows):
    clean = lambda v: str(v).replace("|", "\\|").replace("\n", " ")
    return ("| " + " | ".join(headers) + " |\n| " + " | ".join("---" for _ in headers) + " |\n"
            + "\n".join("| " + " | ".join(clean(v) for v in row) + " |" for row in rows))


INTERPRETATION = ("La evidencia actual no sostiene una aceleración general que conserve simultáneamente error de solución y residuo físico. "
    "Las mejoras puntuales dependen de la selección de componentes, el problema y el entrenamiento; una menor duración a pasos fijos "
    "con mayor error no establece eficiencia a calidad equivalente. El umbral del 99% no garantizó conservar la dirección física activa "
    "en el control desalineado. Una base capaz de reconstruir datos tampoco garantizó aprender la dinámica en la onda. "
    "Estos resultados corresponden a los pipelines evaluados y no prueban una imposibilidad general de PCA.")


def build_summary(root, apply_closures=False):
    root = Path(root).resolve()
    registry = load_registry(root)
    comparisons = comparison_records(registry, root)
    main_sources = {r["source"] for r in comparisons}
    paths = sorted(path for family in ("burgers", "advection", "high_dimensional_advection", "wave")
                   for path in (root / "results" / family).rglob("summary.json"))
    auxiliary = {root / p for b in registry["benchmarks"] for p in b.get("auxiliary_sources", [])}
    manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(set(paths) | auxiliary)}
    archive = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        settings = data.get("source_configuration", data.get("configuration", data.get("config", {})))
        relative = path.relative_to(root).as_posix()
        archive.append({"run_id": path.parent.name, "source": relative,
                        "seed_count": len(settings["seeds"]) if "seeds" in settings else data.get("seed_count"),
                        "steps": settings.get("steps", data.get("final_step")),
                        "role": "Tabla cuantitativa" if relative in main_sources else "Antecedente o control complementario"})
    report = {"as_of": registry["as_of"], "registry": registry, "comparisons": comparisons,
              "archive": archive, "source_sha256": manifest,
              "comparison_count": len(comparisons), "independent_replication_warning": "Rows share seeds, baselines and historical runs; do not pool as independent trials."}
    output = root / "results/benchmark_summary"
    output.mkdir(parents=True, exist_ok=True)
    for name, records in (("comparisons.csv", comparisons), ("archive_index.csv", archive),
                          ("benchmark_overview.csv", [{key:b[key] for key in ("id","name","status","status_label","pca_scope","finding","limitation")}
                                                      for b in registry["benchmarks"]])):
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in records)
    (output / "summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    overview = [[b["name"],b["status_label"],b["pca_scope"],b["finding"],
                 f'[Notebook]({b["notebook"]})' if b["notebook"] else "Sin implementación o corridas"] for b in registry["benchmarks"]]
    intro = (f'# Tabla global de benchmarks\n\nFecha de síntesis: **{registry["as_of"]}**. '
        'Se agregan artefactos existentes; esta operación no entrena modelos.\n\n'
        '## Estados y conclusiones\n\n' + _markdown_table(["Benchmark","Estado","Aplicación de PCA","Conclusión limitada","Evidencia"],overview)
        + '\n\n## Cómo leer las métricas\n\n'
        '- Media ± desviación estándar entre semillas; los conteos comparan PCA con su base dentro de cada semilla.\n'
        '- Las magnitudes del residuo y los tiempos absolutos no deben ordenarse entre PDE: cambian unidades, operadores y condiciones de ejecución. Comparar dentro de cada fila.\n'
        '- Burgers/advección 1D puntúan el residuo de prueba por diferencias finitas; dimensión alta/onda usan AD. La física de entrenamiento usa AD en estas fases.\n'
        '- Los tiempos son de pipelines individuales según cada protocolo, con compilación separada. No incluyen el costo acumulado de explorar k o pesos. En la onda también se excluyen oráculos y mediciones de gradientes.\n'
        '- Todas las variantes de los barridos resumidos están incluidas. El archivo cuantitativo conserva fases representativas y finales; el índice histórico conserva los estudios anteriores y controles.\n'
        '- Las filas reutilizan semillas y comparadores; no son experimentos independientes para sumar tamaños de muestra o combinar significación. No se selecciona un ganador global usando prueba.\n')
    sections = [intro]
    for b in registry["benchmarks"]:
        if b["comparisons"]:
            sections.append(f'\n## {b["name"]}\n\n{b["pca_scope"]}. {b["pde_metric"]}.\n\n'
                + _markdown_table(QUANT_HEADERS,_display_rows(report,b["id"],"markdown",""))
                + f'\n\nLímite de interpretación: {b["limitation"]}\n')
    sections.append('\n## Interpretación transversal provisional\n\n'+INTERPRETATION+'\n\n'
        'El benchmark de onda está cerrado con resultado negativo del protocolo.\n\n'
        '## Artefactos y reproducción\n\n'
        '- [Notebook global](notebooks/00_benchmark_summary.ipynb)\n'
        '- [Análisis de las primeras iteraciones](notebooks/01_early_training_comparison.ipynb)\n'
        '- [Comparaciones numéricas CSV](results/benchmark_summary/comparisons.csv)\n'
        '- [Índice de todas las corridas históricas](results/benchmark_summary/archive_index.csv)\n'
        '- [Estados y reglas de inclusión](reports/benchmark_registry.json)\n'
        '- [Cierre de la onda](reports/wave_multiscale_closure.md)\n\n'
        'Regenerar: `python scripts/build_benchmark_summary.py`. Para aplicar los cierres registrados a los notebooks existentes, añadir `--apply-closures`. '
        'Este script usa solo la biblioteca estándar y no importa JAX. Los generadores de reportes de onda conservan el cierre al regenerar las corridas registradas.\n')
    (root / "BENCHMARKS.md").write_text('\n'.join(sections),encoding="utf-8")
    def md(text): return {"cell_type":"markdown","metadata":{},"source":text.splitlines(True)}
    def code(text, data=None): return {"cell_type":"code","metadata":{},"execution_count":None,"source":text.splitlines(True),
        "outputs":[{"output_type":"display_data","metadata":{},"data":data}] if data else []}
    cells = [md("# Síntesis global de benchmarks\n\n"
        f'Estado documental: {registry["as_of"]}. Se muestran {len(registry["benchmarks"])} familias evaluadas. '
        f'Las {len(comparisons)} comparaciones numéricas comparten semillas y bases; no constituyen réplicas independientes. '
        "El índice histórico conserva las fases previas. Los outputs son tablas generadas desde resultados guardados.\n\n"
        "La onda se cierra como exploración negativa del protocolo probado."),
        code("from pathlib import Path\nimport json,sys\nfrom IPython.display import display,HTML\nROOT=Path.cwd()\nwhile not (ROOT/'pyproject.toml').exists() and ROOT!=ROOT.parent:\n    ROOT=ROOT.parent\nsys.path.insert(0,str(ROOT/'src'))\nfrom pinn_pca.benchmark_summary import overview_html,comparison_html\nREPORT=json.loads((ROOT/'results/benchmark_summary/summary.json').read_text(encoding='utf-8'))"),
        code("display(HTML(overview_html(REPORT)))",{"text/html":overview_html(report)}),
        md("## Comparaciones dentro de cada problema\n\nMedia ± DE; flecha base → PCA. Los conteos indican cuántas semillas tienen menor error, residuo o tiempo. Los tiempos corresponden al pipeline individual, sin incluir búsqueda de hiperparámetros. Los residuos de prueba usan diferencias finitas en Burgers/advección 1D y AD en dimensión alta/onda. Las unidades y protocolos difieren entre PDE: no usar la tabla como clasificación global de tiempos o residuos.")]
    for b in registry["benchmarks"]:
        if b["comparisons"]:
            cells += [md(f'## {b["name"]}\n\n{b["pca_scope"]}.\n\n{b["limitation"]}'),
                      code(f'display(HTML(comparison_html(REPORT,"{b["id"]}")))',{"text/html":comparison_html(report,b["id"])})]
    cells.append(md("## Conclusiones y alcance\n\n"+INTERPRETATION+"\n\n"
        "El desequilibrio de gradientes en la onda es una hipótesis de explicación que no demuestra una causa única. "
        "El cierre conserva esa incertidumbre y el residuo no nulo de los splines. Los resultados de selección por validación, "
        "exploración de componentes y confirmación con otra condición inicial se distinguen en cada fila.\n\n"
        "Reproducir sin entrenar: `python scripts/build_benchmark_summary.py`. Las tablas CSV/JSON y las huellas de las fuentes "
        "se guardan en `results/benchmark_summary/`. El registro de estados vive en `reports/benchmark_registry.json`. "
        "El [análisis temprano](01_early_training_comparison.ipynb) estudia los primeros checkpoints usando validación guardada."))
    notebook = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                "language_info":{"name":"python"}},"nbformat":4,"nbformat_minor":5}
    (root / "notebooks/00_benchmark_summary.ipynb").write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding="utf-8")
    if apply_closures:
        for b in registry["benchmarks"]:
            for closed in b.get("closed_notebooks", []):
                path = root / closed["path"]
                nb = json.loads(path.read_text(encoding="utf-8"))
                apply_benchmark_closure(nb,root,b["id"],closed["run_id"],path)
                path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding="utf-8")
    return report
