"""Small dependency-free SVG figures for saved experiment results."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def grouped_summary_svg(path: str | Path, summary: dict, rows: list[dict],
                        pde_name: str = "Burgers"):
    methods = list(summary)
    palette = ["#727b86", "#3584a8", "#d08a27", "#7c65a3", "#2c8c62", "#c45858"]
    colors = [palette[index % len(palette)] for index in range(len(methods))]
    metrics = [
        ("test_relative_l2", "Error relativo L2 en prueba"),
        ("test_pde_residual_rms", f"RMS del residuo de {pde_name} en prueba"),
        ("pipeline_seconds", "Tiempo por corrida (PCA incluye ajuste)"),
    ]
    width, panel_height, left, right = 980, 235, 125, 45
    height = 55 + panel_height * len(metrics)
    output = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
              '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:18px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}</style>',
              f'<text x="{width/2}" y="28" class="title" text-anchor="middle">PINN base vs. PCA como preprocesamiento — {len({r["seed"] for r in rows})} semillas</text>']
    for panel, (metric, label) in enumerate(metrics):
        top = 45 + panel * panel_height
        y0, y1 = top + 22, top + panel_height - 48
        plot_height, plot_width = y1-y0, width-left-right
        means = [summary[name][metric]["mean"] for name in methods]
        stds = [summary[name][metric]["std"] for name in methods]
        maximum = max(m+s for m,s in zip(means,stds)) * 1.2
        maximum = maximum if maximum > 0 else 1.0
        group_width = plot_width / len(methods)
        bar_width = min(88, group_width * 0.5)
        output.append(f'<text x="{left}" y="{top+13}" class="label">{label}</text>')
        for tick in range(5):
            value = maximum * tick / 4
            yy = y1 - plot_height * tick / 4
            output += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>',
                       f'<text x="{left-8}" y="{yy+4:.1f}" class="small" text-anchor="end">{value:.3g}</text>']
        output += [f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y1}" class="axis"/>',
                   f'<line x1="{left}" y1="{y1}" x2="{width-right}" y2="{y1}" class="axis"/>']
        for method_index, name in enumerate(methods):
            cx = left + group_width * (method_index + .5)
            mean, std = means[method_index], stds[method_index]
            bar_h = plot_height * mean / maximum
            yy = y1 - bar_h
            err = plot_height * std / maximum
            output += [f'<rect x="{cx-bar_width/2:.1f}" y="{yy:.1f}" width="{bar_width}" height="{bar_h:.1f}" fill="{colors[method_index]}"/>',
                       f'<line x1="{cx}" y1="{yy-err:.1f}" x2="{cx}" y2="{yy+err:.1f}" stroke="#222"/>',
                       f'<line x1="{cx-7}" y1="{yy-err:.1f}" x2="{cx+7}" y2="{yy-err:.1f}" stroke="#222"/>',
                       f'<line x1="{cx-7}" y1="{yy+err:.1f}" x2="{cx+7}" y2="{yy+err:.1f}" stroke="#222"/>',
                       f'<text x="{cx}" y="{y1+19}" class="small" text-anchor="middle">{name}</text>',
                       f'<text x="{cx}" y="{yy-err-7:.1f}" class="small" text-anchor="middle">{mean:.4g} ± {std:.2g}</text>']
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")


def budget_grouped_metric_svg(path: str | Path, summaries: dict[str, dict],
                              metric: str, title: str, y_label: str,
                              model_labels: dict[str, str] | None = None):
    """Compare one final metric across budgets with seed-level error bars."""
    budget_names = list(summaries)
    model_names = list(next(iter(summaries.values())))
    labels = model_labels or {name: name for name in model_names}
    colors = ["#a8b1bb", "#167caa", "#d08a27", "#7c65a3"]
    means = [summaries[budget][model][metric]["mean"]
             for budget in budget_names for model in model_names]
    stds = [summaries[budget][model][metric]["std"]
            for budget in budget_names for model in model_names]
    maximum = max((mean + std for mean, std in zip(means, stds)), default=1.0) * 1.16
    maximum = maximum if maximum > 0 else 1.0
    width, height = 1120, 560
    left, right, top, bottom = 110, 45, 80, 118
    plot_width, plot_height = width-left-right, height-top-bottom
    group_width = plot_width / len(model_names)
    bar_width = min(42, group_width/(len(budget_names)+1))
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:19px;font-weight:bold}.label{font-size:14px}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="32" class="title" text-anchor="middle">{title}</text>',
    ]
    for tick in range(6):
        fraction = tick/5
        value = maximum*fraction
        y = top+plot_height-plot_height*fraction
        output += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" class="grid"/>',
                   f'<text x="{left-9}" y="{y+4:.1f}" class="small" text-anchor="end">{value:.3g}</text>']
    output += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" class="axis"/>',
               f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" class="axis"/>',
               f'<text x="24" y="{top+plot_height/2}" class="label" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2})">{y_label}</text>']
    for model_index, model in enumerate(model_names):
        center = left+group_width*(model_index+0.5)
        for budget_index, budget in enumerate(budget_names):
            stats = summaries[budget][model][metric]
            mean, std = stats["mean"], stats["std"]
            x = center+(budget_index-(len(budget_names)-1)/2)*(bar_width+8)
            bar_height = plot_height*mean/maximum
            y = top+plot_height-bar_height
            err = plot_height*std/maximum
            color = colors[budget_index % len(colors)]
            output += [f'<rect x="{x-bar_width/2:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{color}"/>',
                       f'<line x1="{x:.1f}" y1="{y-err:.1f}" x2="{x:.1f}" y2="{y+err:.1f}" stroke="#222"/>',
                       f'<line x1="{x-5:.1f}" y1="{y-err:.1f}" x2="{x+5:.1f}" y2="{y-err:.1f}" stroke="#222"/>',
                       f'<line x1="{x-5:.1f}" y1="{y+err:.1f}" x2="{x+5:.1f}" y2="{y+err:.1f}" stroke="#222"/>']
        output.append(f'<text x="{center:.1f}" y="{top+plot_height+24}" class="small" text-anchor="middle">{labels.get(model, model)}</text>')
    legend_x = max(left, width-right-180)
    for index, budget in enumerate(budget_names):
        y = height-44+index*19
        output += [f'<rect x="{legend_x}" y="{y-11}" width="14" height="14" fill="{colors[index % len(colors)]}"/>',
                   f'<text x="{legend_x+21}" y="{y}" class="small">{budget}</text>']
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")


def dimension_scaling_svg(path: str | Path, summaries_by_dimension: dict[str, dict]):
    """Plot paired model metrics as the ambient input dimension increases."""
    dimensions = sorted(int(value) for value in summaries_by_dimension)
    models = list(summaries_by_dimension[str(dimensions[0])])
    colors = ["#596775", "#167caa", "#d08a27", "#7c65a3"]
    model_colors = {name: colors[index % len(colors)] for index, name in enumerate(models)}
    metrics = [
        ("test_relative_l2", "Error relativo L2 de prueba"),
        ("test_pde_residual_rms", "Residuo RMS PDE de prueba"),
        ("pipeline_seconds", "Tiempo de pipeline (s)"),
    ]
    width, panel_height, left, right = 1000, 230, 105, 45
    height = 55 + panel_height * len(metrics)
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:19px;font-weight:bold}.label{font-size:13px}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="30" class="title" text-anchor="middle">Escalamiento al aumentar la dimensión de entrada</text>',
    ]
    x_positions = [left + (width-left-right)*i/max(1, len(dimensions)-1)
                   for i in range(len(dimensions))]
    for panel, (metric, title) in enumerate(metrics):
        top = 48 + panel*panel_height
        y0, y1 = top+24, top+panel_height-48
        plot_h = y1-y0
        values = [summaries_by_dimension[str(d)][model][metric]
                  for d in dimensions for model in models]
        maximum = max((item["mean"]+item["std"] for item in values), default=1.0)*1.18
        maximum = maximum if maximum > 0 else 1.0
        output.append(f'<text x="{left}" y="{top+13}" class="label">{title}</text>')
        for tick in range(5):
            fraction = tick/4
            y = y1-plot_h*fraction
            value = maximum*fraction
            output += [f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" class="grid"/>',
                       f'<text x="{left-8}" y="{y+4:.1f}" class="small" text-anchor="end">{value:.3g}</text>']
        output += [f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y1}" class="axis"/>',
                   f'<line x1="{left}" y1="{y1}" x2="{width-right}" y2="{y1}" class="axis"/>']
        for x, dimension in zip(x_positions, dimensions):
            output.append(f'<text x="{x:.1f}" y="{y1+19}" class="small" text-anchor="middle">{dimension}</text>')
        for model in models:
            points = []
            color = model_colors[model]
            for x, dimension in zip(x_positions, dimensions):
                stat = summaries_by_dimension[str(dimension)][model][metric]
                y = y1-plot_h*stat["mean"]/maximum
                error = plot_h*stat["std"]/maximum
                points.append(f'{x:.1f},{y:.1f}')
                output += [f'<line x1="{x:.1f}" y1="{y-error:.1f}" x2="{x:.1f}" y2="{y+error:.1f}" stroke="{color}"/>',
                           f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{model}, d={dimension}: {stat["mean"]:.4g} ± {stat["std"]:.3g}</title></circle>']
            output.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        if panel == 0:
            for index, model in enumerate(models):
                x = width-right-210+index*105
                output += [f'<line x1="{x}" y1="53" x2="{x+22}" y2="53" stroke="{model_colors[model]}" stroke-width="3"/>',
                           f'<text x="{x+28}" y="57" class="small">{model}</text>']
    output.append(f'<text x="{width/2}" y="{height-17}" class="label" text-anchor="middle">Dimensión de entrada d</text>')
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")


def controlled_trajectory_svg(path: str | Path, trajectories: list[dict], targets: list[float]):
    """Plot validation error against end-to-end elapsed time for every seed."""
    width, height = 1040, 660
    left, right, top, bottom = 90, 45, 70, 70
    plot_width, plot_height = width-left-right, height-top-bottom
    max_time = max(row["elapsed_pipeline_seconds"] for row in trajectories) * 1.05
    max_error = max(row["validation_rmse"] for row in trajectories) * 1.12
    palette = ["#66717e", "#2582a8", "#d08a27", "#7c65a3", "#2c8c62", "#c45858"]
    model_names = list(dict.fromkeys(row["model"] for row in trajectories))
    colors = {name: palette[index % len(palette)] for index, name in enumerate(model_names)}
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:19px;font-weight:bold}.label{font-size:14px}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="30" class="title" text-anchor="middle">Error de validación frente a tiempo del pipeline</text>',
    ]
    for tick in range(6):
        xval = max_time * tick / 5
        xpos = left + plot_width * tick / 5
        output += [f'<line x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{top+plot_height}" class="grid"/>',
                   f'<text x="{xpos:.1f}" y="{top+plot_height+22}" class="small" text-anchor="middle">{xval:.2f}</text>']
        yval = max_error * tick / 5
        ypos = top + plot_height - plot_height * tick / 5
        output += [f'<line x1="{left}" y1="{ypos:.1f}" x2="{left+plot_width}" y2="{ypos:.1f}" class="grid"/>',
                   f'<text x="{left-9}" y="{ypos+4:.1f}" class="small" text-anchor="end">{yval:.3f}</text>']
    output += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" class="axis"/>',
               f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" class="axis"/>',
               f'<text x="{left+plot_width/2}" y="{height-20}" class="label" text-anchor="middle">Tiempo de pipeline (s; incluye PCA si aplica)</text>',
               f'<text x="24" y="{top+plot_height/2}" class="label" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2})">RMSE de validación</text>']
    for target in targets:
        ypos = top + plot_height - plot_height * target / max_error
        if top <= ypos <= top + plot_height:
            output += [f'<line x1="{left}" y1="{ypos:.1f}" x2="{left+plot_width}" y2="{ypos:.1f}" stroke="#ad6b18" stroke-dasharray="6,5"/>',
                       f'<text x="{left+plot_width-4}" y="{ypos-5:.1f}" class="small" text-anchor="end">objetivo {target:g}</text>']
    groups = {}
    for row in trajectories:
        groups.setdefault((row["model"], row["seed"]), []).append(row)
    for (model, seed), values in groups.items():
        values.sort(key=lambda row: row["step"])
        points = []
        for row in values:
            xpos = left + plot_width * row["elapsed_pipeline_seconds"] / max_time
            ypos = top + plot_height - plot_height * row["validation_rmse"] / max_error
            points.append(f"{xpos:.1f},{ypos:.1f}")
        color = colors.get(model, "#555")
        output.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.4" opacity="0.42"><title>{model}, semilla {seed}</title></polyline>')
    for index, (model, color) in enumerate(colors.items()):
        xpos = left + 24 + index * 180
        output += [f'<line x1="{xpos}" y1="50" x2="{xpos+28}" y2="50" stroke="{color}" stroke-width="3"/>',
                   f'<text x="{xpos+35}" y="54" class="small">{model}</text>']
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")


def checkpoint_curve_svg(path: str | Path, records: list[dict], metric: str,
                         title: str, y_label: str):
    """Plot seed-averaged validation metrics across fixed training checkpoints."""
    model_names = list(dict.fromkeys(row["model"] for row in records))
    steps = sorted({row["step"] for row in records})
    palette = ["#596775", "#167caa", "#d08a27", "#7c65a3", "#2c8c62", "#c45858"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(model_names)}
    grouped = {}
    for row in records:
        if row.get(metric) is not None:
            grouped.setdefault((row["model"], row["step"]), []).append(row[metric])
    means = [float(np.mean(values)) for values in grouped.values()]
    deviations = [float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 for values in grouped.values()]
    ymin = min(0.0, min((mean-std for mean, std in zip(means, deviations)), default=0.0))
    ymax = max((mean+std for mean, std in zip(means, deviations)), default=1.0) * 1.12
    if ymax <= ymin:
        ymax = ymin + 1.0
    width, height = 1040, 590
    left, right, top, bottom = 100, 40, 74, 78
    plot_w, plot_h = width-left-right, height-top-bottom
    x_min, x_max = min(steps), max(steps)
    if x_min == x_max:
        x_min, x_max = 0, max(1, x_max)
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:19px;font-weight:bold}.label{font-size:14px}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="30" class="title" text-anchor="middle">{title}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        value = ymin + (ymax-ymin) * fraction
        y = top + plot_h - plot_h * fraction
        output += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>',
                   f'<text x="{left-9}" y="{y+4:.1f}" class="small" text-anchor="end">{value:.4g}</text>']
        x_value = x_min + (x_max-x_min) * fraction
        x = left + plot_w * fraction
        output += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>',
                   f'<text x="{x:.1f}" y="{top+plot_h+21}" class="small" text-anchor="middle">{x_value:.0f}</text>']
    output += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
               f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
               f'<text x="{left+plot_w/2}" y="{height-24}" class="label" text-anchor="middle">Pasos de optimización</text>',
               f'<text x="24" y="{top+plot_h/2}" class="label" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})">{y_label}</text>']
    for model in model_names:
        points = []
        for step in steps:
            values = grouped.get((model, step))
            if not values:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            x = left + plot_w * (step-x_min) / (x_max-x_min)
            y = top + plot_h - plot_h * (mean-ymin) / (ymax-ymin)
            y0 = top + plot_h - plot_h * (mean-std-ymin) / (ymax-ymin)
            y1 = top + plot_h - plot_h * (mean+std-ymin) / (ymax-ymin)
            points.append(f'{x:.1f},{y:.1f}')
            output.append(f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" y2="{y1:.1f}" stroke="{colors[model]}"/>')
            output.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{colors[model]}"/>')
        if points:
            output.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[model]}" stroke-width="2.4"/>')
    for index, model in enumerate(model_names):
        x = left + index * 190
        output += [f'<line x1="{x}" y1="52" x2="{x+24}" y2="52" stroke="{colors[model]}" stroke-width="3"/>',
                   f'<text x="{x+30}" y="56" class="small">{model}</text>']
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")


def metric_tradeoff_svg(path: str | Path, summary: dict, x_metric: str, y_metric: str,
                        x_label: str, y_label: str, title: str):
    """Scatter mean model metrics with across-seed standard-deviation bars."""
    methods = list(summary)
    palette = ["#596775", "#167caa", "#d08a27", "#7c65a3", "#2c8c62", "#c45858"]
    colors = {name: palette[i % len(palette)] for i, name in enumerate(methods)}
    x_stats = [summary[name][x_metric] for name in methods]
    y_stats = [summary[name][y_metric] for name in methods]
    x_low = min(max(0.0, item["mean"]-item["std"]) for item in x_stats)
    x_high = max(item["mean"]+item["std"] for item in x_stats)
    y_low = min(max(0.0, item["mean"]-item["std"]) for item in y_stats)
    y_high = max(item["mean"]+item["std"] for item in y_stats)
    x_span = max(1e-9, x_high-x_low)
    y_span = max(1e-9, y_high-y_low)
    x_low = max(0.0, x_low-0.08*x_span)
    x_high += 0.12*x_span
    y_low = max(0.0, y_low-0.08*y_span)
    y_high += 0.12*y_span
    width, height = 1000, 640
    left, right, top, bottom = 110, 230, 80, 85
    plot_w, plot_h = width-left-right, height-top-bottom
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#555}.title{font-size:19px;font-weight:bold}.label{font-size:14px}.small{font-size:11px}</style>',
        f'<text x="{width/2}" y="32" class="title" text-anchor="middle">{title}</text>',
    ]
    for tick in range(6):
        fraction = tick/5
        xv = x_low + (x_high-x_low)*fraction
        xp = left + plot_w*fraction
        yv = y_low + (y_high-y_low)*fraction
        yp = top + plot_h - plot_h*fraction
        output += [f'<line x1="{xp:.1f}" y1="{top}" x2="{xp:.1f}" y2="{top+plot_h}" class="grid"/>',
                   f'<text x="{xp:.1f}" y="{top+plot_h+21}" class="small" text-anchor="middle">{xv:.3g}</text>',
                   f'<line x1="{left}" y1="{yp:.1f}" x2="{left+plot_w}" y2="{yp:.1f}" class="grid"/>',
                   f'<text x="{left-9}" y="{yp+4:.1f}" class="small" text-anchor="end">{yv:.3g}</text>']
    output += [f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
               f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
               f'<text x="{left+plot_w/2}" y="{height-24}" class="label" text-anchor="middle">{x_label}</text>',
               f'<text x="24" y="{top+plot_h/2}" class="label" text-anchor="middle" transform="rotate(-90 24 {top+plot_h/2})">{y_label}</text>']
    for index, name in enumerate(methods):
        xs, ys = summary[name][x_metric], summary[name][y_metric]
        xp = left + plot_w*(xs["mean"]-x_low)/(x_high-x_low)
        yp = top + plot_h - plot_h*(ys["mean"]-y_low)/(y_high-y_low)
        xerr = plot_w*xs["std"]/(x_high-x_low)
        yerr = plot_h*ys["std"]/(y_high-y_low)
        color = colors[name]
        output += [f'<line x1="{xp-xerr:.1f}" y1="{yp:.1f}" x2="{xp+xerr:.1f}" y2="{yp:.1f}" stroke="{color}"/>',
                   f'<line x1="{xp:.1f}" y1="{yp-yerr:.1f}" x2="{xp:.1f}" y2="{yp+yerr:.1f}" stroke="{color}"/>',
                   f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="6" fill="{color}"><title>{name}</title></circle>',
                   f'<line x1="{left+plot_w+15}" y1="{top+22+index*28}" x2="{left+plot_w+35}" y2="{top+22+index*28}" stroke="{color}" stroke-width="3"/>',
                   f'<text x="{left+plot_w+42}" y="{top+26+index*28}" class="small">{name}</text>']
    output.append('</svg>')
    Path(path).write_text("\n".join(output), encoding="utf-8")
