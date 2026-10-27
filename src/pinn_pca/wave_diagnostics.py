"""Oracle PDE controls and one-factor ablation of PCA coefficient scaling."""
from __future__ import annotations

import hashlib
import itertools
import json
import platform
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import wave
from .controlled import _train_trace, matched_pca_width
from .data import split_times
from .models import burgers_solution, init_mlp, mlp, parameter_count
from .training import make_update, warmup_update

NORMALIZED = "PCA estandarizada 99%"
MODELS = (wave.BASE, wave.PCA, NORMALIZED)
REFERENCES = ("Referencia analítica", "PCA oráculo seno", "PCA oráculo spline C2")


def load_sources(config, root):
    source = root / config["source_run"]
    previous = json.loads((source / "summary.json").read_text(encoding="utf-8"))
    settings = json.loads((root / config["source_config"]).read_text(encoding="utf-8"))
    if settings != previous["configuration"]:
        raise ValueError("La configuración fuente ya no coincide con la corrida original.")
    return settings, previous, source


def source_hashes(root, source, settings):
    paths = [source / name for name in ("config.json", "summary.json", "metrics.csv", "validation_trajectories.csv")]
    paths += [root / "data/generated" / f"wave_{case['name']}.npz" for case in settings["cases"]]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def oracle_parameters(x, info, harmonics, amplitudes):
    """Exact time coefficients from the analytic solution, exclusively for diagnostics."""
    sine = np.sin(np.pi * np.asarray(x, dtype=float)[:, None] * np.asarray(harmonics)[None, :])
    weights = info["modes"].T @ (sine * np.asarray(amplitudes)[None, :])
    offset = -info["modes"].T @ info["mean"]
    # Analytic sine evaluation isolates interpolation from the PCA subspace itself.
    fields = np.column_stack((info["mean"], info["modes"]))
    harmonic_fields = np.linalg.lstsq(sine, fields, rcond=None)[0]
    return tuple(jnp.asarray(v, dtype=jnp.float32) for v in (weights, offset, harmonic_fields))


def oracle_solution(basis, oracle, harmonics, amplitudes, speed, x, t, representation):
    weights, offset, harmonic_fields = oracle
    if representation == "Referencia analítica":
        return jnp.sum(amplitudes * jnp.sin(jnp.pi * harmonics * x)
                       * jnp.cos(speed * jnp.pi * harmonics * t))
    coefficients = offset + weights @ jnp.cos(speed * jnp.pi * harmonics * t)
    fields = (jnp.sin(jnp.pi * harmonics * x) @ harmonic_fields
              if representation == "PCA oráculo seno" else wave.spline_fields(basis, x))
    return fields[0] + jnp.dot(fields[1:], coefficients)


@partial(jax.jit, static_argnames=("representation",))
def oracle_point_values(basis, oracle, harmonics, amplitudes, speed, x, t, representation):
    fn = lambda xx, tt: oracle_solution(basis, oracle, harmonics, amplitudes, speed, xx, tt, representation)
    def at(xx, tt):
        ut = jax.grad(fn, argnums=1)(xx, tt)
        utt = jax.grad(lambda ts: jax.grad(fn, argnums=1)(xx, ts))(tt)
        uxx = jax.grad(lambda xs: jax.grad(fn, argnums=0)(xs, tt))(xx)
        return fn(xx, tt), ut, utt, speed**2 * uxx
    return jax.vmap(at)(x, t)


@partial(jax.jit, static_argnames=("representation",))
def oracle_grid(basis, oracle, harmonics, amplitudes, speed, x, t, representation):
    return jax.vmap(lambda xx: jax.vmap(lambda tt: oracle_solution(
        basis, oracle, harmonics, amplitudes, speed, xx, tt, representation))(t))(x)


def evaluate_oracle(basis, info, case, x, t, u, ids, seed, count, speed, evaluation):
    oracle = oracle_parameters(x, info, case["harmonics"], case["amplitudes"])
    ns = jnp.asarray(case["harmonics"], dtype=jnp.float32)
    amplitudes = jnp.asarray(case["amplitudes"], dtype=jnp.float32)
    # Test points exactly reproduce the pilot; the gate uses validation points only.
    rng = np.random.default_rng(seed + (1901 if evaluation == "test" else 2901))
    rx = jnp.asarray(rng.uniform(0, 1, count), dtype=jnp.float32)
    rt = jnp.asarray(rng.choice(t[ids], count), dtype=jnp.float32)
    true_points = np.asarray(jax.vmap(lambda xx, tt: jnp.sum(amplitudes * jnp.sin(ns*jnp.pi*xx)
        * jnp.cos(speed*ns*jnp.pi*tt)))(rx, rt))
    truth = u[:, ids]
    null_loss = float(np.mean(true_points**2) + np.mean(u[:, 0]**2))
    rows = []
    for representation in REFERENCES:
        point_args = (basis, oracle, ns, amplitudes, speed)
        prediction = np.asarray(oracle_grid(*point_args, jnp.asarray(x), jnp.asarray(t[ids]), representation))
        values, ut, utt, cuxx = [np.asarray(v) for v in oracle_point_values(
            *point_args, rx, rt, representation)]
        initial, initial_ut, _, _ = [np.asarray(v) for v in oracle_point_values(
            *point_args, jnp.asarray(x), jnp.zeros(x.size, dtype=jnp.float32), representation)]
        boundary = np.asarray(oracle_grid(*point_args, jnp.asarray([0., 1.]), jnp.asarray(t[ids]), representation))
        rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, dtype=float)**2)))
        residual = rms(utt-cuxx)
        scale = rms(utt) + rms(cuxx)
        terms = [float(np.mean((values-true_points)**2)), residual**2,
                 float(np.mean((initial-u[:, 0])**2)), rms(initial_ut)**2,
                 float(np.mean(boundary**2))]
        objective = sum(terms)
        rows.append({"case": case["name"], "seed": seed, "evaluation": evaluation,
            "representation": representation, "k": info["k"],
            "relative_l2": float(np.linalg.norm(prediction-truth)/np.linalg.norm(truth)),
            "pde_residual_rms": residual, "derivative_scale_rms": scale,
            "normalized_pde_residual": residual/max(scale, 1e-30),
            "initial_u_rmse": rms(initial-u[:, 0]), "initial_ut_rms": rms(initial_ut),
            "boundary_rms": rms(boundary), "oracle_loss": objective, "null_loss": null_loss,
            "oracle_to_null_loss_ratio": objective/null_loss})
    return rows


def oracle_gate(rows, thresholds):
    selected = [r for r in rows if r["evaluation"] == "validation"
                and r["representation"] == "PCA oráculo spline C2"]
    checks = [{"case": r["case"], "seed": r["seed"], "passed": bool(
        r["relative_l2"] <= thresholds["max_validation_relative_l2"] and
        r["normalized_pde_residual"] <= thresholds["max_normalized_pde_residual"] and
        r["oracle_to_null_loss_ratio"] <= thresholds["max_oracle_to_null_loss_ratio"])} for r in selected]
    return {"passed": bool(checks) and all(c["passed"] for c in checks), "checks": checks,
            "thresholds": thresholds, "selection_scope": "Validation only; thresholds fixed before evaluation."}


def run_oracle_stage(config, project_root):
    root = Path(project_root).resolve()
    settings, previous, source = load_sources(config, root)
    if settings["loss_weights"] != [1., 1., 1., 1., 1.]:
        raise ValueError("El control oráculo requiere los cinco pesos originales iguales a uno.")
    output = root / "results/wave/diagnostics" / config["run_id"]
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in settings["cases"]:
        with np.load(root / "data/generated" / f"wave_{case['name']}.npz") as data:
            x, t, u = [data[name] for name in ("x", "t", "u")]
        for seed in settings["seeds"]:
            partition = previous["partitions"][case["name"]][str(seed)]
            basis, info = wave.fit_wave_pca(x, u, partition["fit_time_indices"], settings["pca_variance_threshold"])
            if info["k"] != partition["k"]:
                raise ValueError("Cambió la selección de componentes respecto del piloto.")
            for evaluation, key in (("validation", "validation_time_indices"), ("test", "test_time_indices")):
                rows.extend(evaluate_oracle(basis, info, case, x, t, u, partition[key], seed,
                    config["oracle_points"], settings["wave_speed"], evaluation))
    gate = oracle_gate(rows, config["oracle_gate"])
    report = {"configuration": config, "source_configuration": settings, "gate": gate,
              "source_sha256": source_hashes(root, source, settings), "records": rows}
    wave._write_csv(output / "oracle_diagnostics.csv", rows)
    (output / "oracle_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Oracle validation gate: {'PASS' if gate['passed'] else 'FAIL'} ({len(gate['checks'])} case/seed pairs)", flush=True)
    return report


def fit_coefficient_scaler(u, info, fit_ids):
    """No validation/test snapshots can enter the latent center or scale."""
    coefficients = (np.asarray(u[:, fit_ids].T, dtype=float) - info["mean"]) @ info["modes"]
    center = coefficients.mean(axis=0)
    scale = coefficients.std(axis=0, ddof=0)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 1e-12):
        raise ValueError("No se puede estandarizar un coeficiente degenerado.")
    return jnp.asarray(center, dtype=jnp.float32), jnp.asarray(scale, dtype=jnp.float32)


def normalized_solution(params, state, x, t):
    basis, center, scale = state[:2], state[2], state[3]
    fields = wave.spline_fields(basis, x)
    coefficients = center + scale * mlp(params, jnp.reshape(t, (1,)))
    return fields[0] + jnp.dot(fields[1:], coefficients)


@jax.jit
def normalized_prediction(params, state, x, t):
    fields = jax.vmap(lambda xx: wave.spline_fields(state[:2], xx))(x)
    z = jax.vmap(lambda tt: mlp(params, jnp.reshape(tt, (1,))))(t)
    coefficients = state[2] + state[3] * z
    return fields[:, :1] + fields[:, 1:] @ coefficients.T


@jax.jit
def normalized_validation(params, state, x, t, truth):
    return jnp.sqrt(jnp.mean((normalized_prediction(params, state, x, t)-truth)**2))


def match_physical_initialization(raw_params, center, scale):
    """Inverse affine output-head transform: decoded a(t) initially matches raw PCA."""
    weight, bias = raw_params[-1]
    return [*raw_params[:-1], (weight/scale[None, :], (bias-center)/scale)]


def model_solution(model, params, state, x, t):
    if model == wave.BASE:
        return burgers_solution(params, x, t)
    if model == wave.PCA:
        return wave.pca_wave_solution(params, state, x, t)
    return normalized_solution(params, state, x, t)


@partial(jax.jit, static_argnames=("model",))
def parameter_diagnostics(params, state, batch, speed, model):
    def terms(p):
        return wave.loss_terms(lambda xx, tt: model_solution(model, p, state, xx, tt), batch, speed)
    jacobian = jax.jacrev(terms)(params)
    squared = jnp.zeros(5)
    for leaf in jax.tree.leaves(jacobian):
        squared = squared + jnp.sum(leaf**2, axis=tuple(range(1, leaf.ndim)))
    return terms(params), jnp.sqrt(squared)


@partial(jax.jit, static_argnames=("model",))
def physical_diagnostics(params, state, rx, rt, ix, iu, bt, speed, model):
    solution = lambda xx, tt: model_solution(model, params, state, xx, tt)
    residual = jax.vmap(lambda xx, tt: wave.wave_residual(solution, xx, tt, speed))(rx, rt)
    initial = jax.vmap(lambda xx: solution(xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    initial_ut = jax.vmap(lambda xx: jax.grad(solution, argnums=1)(xx, jnp.asarray(0., dtype=xx.dtype)))(ix)
    left = jax.vmap(lambda tt: solution(jnp.asarray(0., dtype=tt.dtype), tt))(bt)
    right = jax.vmap(lambda tt: solution(jnp.asarray(1., dtype=tt.dtype), tt))(bt)
    return {"test_pde_residual_rms": jnp.sqrt(jnp.mean(residual**2)),
            "initial_u_rmse": jnp.sqrt(jnp.mean((initial-iu)**2)),
            "initial_ut_rms": jnp.sqrt(jnp.mean(initial_ut**2)),
            "test_boundary_rms": jnp.sqrt(.5*(jnp.mean(left**2)+jnp.mean(right**2)))}


def _mean_std(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values)>1 else 0.}


def paired_report(base_rows, other_rows, metric, config):
    seeds = sorted(base_rows)
    delta = np.asarray([other_rows[s][metric]-base_rows[s][metric] for s in seeds])
    rng = np.random.default_rng(config["bootstrap_seed"])
    means = delta[rng.integers(0, len(seeds), (config["bootstrap_draws"], len(seeds)))].mean(axis=1)
    return {"other_minus_reference_mean": float(delta.mean()),
            "other_better_count": int(np.sum(delta<0)), "seed_count": len(seeds),
            "bootstrap_ci_95": [float(v) for v in np.percentile(means, [2.5, 97.5])]}


def run_diagnostic_round(config, project_root):
    # Check report availability before starting a long run.
    from .wave_diagnostic_report import create_report
    root = Path(project_root).resolve()
    oracle_report = run_oracle_stage(config, root)
    output = root / "results/wave/diagnostics" / config["run_id"]
    if not oracle_report["gate"]["passed"]:
        create_report(root, output)
        print("Oracle gate failed; coefficient-scaling ablation was not run.", flush=True)
        return oracle_report
    if config["normalization_initialization"] != "match_raw_physical_function":
        raise ValueError("La inicialización debe conservar la función física original.")
    settings, previous, source = load_sources(config, root)
    original_hashes = oracle_report["source_sha256"]
    speed = settings["wave_speed"]
    base_loss, raw_loss = wave.make_losses(speed, settings["loss_weights"])
    weights = jnp.asarray(settings["loss_weights"])
    def norm_loss(params, batch, state):
        return jnp.dot(weights, wave.loss_terms(lambda xx, tt: normalized_solution(params, state, xx, tt), batch, speed))
    updates = {wave.BASE: make_update(base_loss, settings["learning_rate"]),
               wave.PCA: make_update(raw_loss, settings["learning_rate"], with_basis=True),
               NORMALIZED: make_update(norm_loss, settings["learning_rate"], with_basis=True)}
    width = settings["baseline_width"]
    base_layers = [2, width, width, width, 1]
    base_parameters = parameter_count(base_layers)
    rows, traces, modes, scaler_rows, gradient_rows = [], [], [], [], []
    partitions, compilation = {}, {}
    orders = list(itertools.permutations(MODELS))
    for case in settings["cases"]:
        name = case["name"]
        with np.load(root / "data/generated" / f"wave_{name}.npz") as data:
            x,t,u = [data[key] for key in ("x","t","u")]
        partitions[name] = {}
        for index, seed in enumerate(settings["seeds"]):
            expected = previous["partitions"][name][str(seed)]
            fit, val, test = split_times(t.size, seed, settings["test_time_count"], settings["validation_time_count"])
            initial_ids = np.sort(np.random.default_rng(seed+17).choice(
                x.size, settings["initial_condition_points"], replace=False))
            for actual, key in [(fit,"fit_time_indices"),(val,"validation_time_indices"),
                                (test,"test_time_indices"),(initial_ids,"initial_x_indices")]:
                if actual.tolist() != expected[key]:
                    raise ValueError(f"Las particiones cambiaron: {name}, {seed}, {key}.")
            start = time.perf_counter()
            basis, info = wave.fit_wave_pca(x, u, fit, settings["pca_variance_threshold"])
            pca_seconds = time.perf_counter()-start
            if info["k"] != expected["k"]:
                raise ValueError("Cambió k respecto del piloto.")
            start = time.perf_counter()
            center, scale = fit_coefficient_scaler(u, info, fit)
            center.block_until_ready(); scale.block_until_ready()
            scale_seconds = time.perf_counter()-start
            normalized_state = (*basis, center, scale)
            k = info["k"]
            pw, pc = matched_pca_width(k, base_parameters)
            if abs(pc-base_parameters)/base_parameters>.05:
                raise ValueError("La diferencia de parámetros supera el 5%.")
            batches, digest = wave.make_wave_batches(x,t,u,fit,initial_ids,settings["steps"],settings["batch_size"],seed)
            if digest != expected["shared_batch_sha256"]:
                raise ValueError("Los minibatches no coinciden con el piloto.")
            raw_params = init_mlp(jax.random.PRNGKey(seed),[1,pw,pw,pw,k])
            params = {wave.BASE:init_mlp(jax.random.PRNGKey(seed),base_layers), wave.PCA:raw_params,
                      NORMALIZED:match_physical_initialization(raw_params,center,scale)}
            states = {wave.BASE:None, wave.PCA:basis, NORMALIZED:normalized_state}
            xv,tv,uv = jnp.asarray(x),jnp.asarray(t[val]),jnp.asarray(u[:,val])
            validation_fns = {
                wave.BASE:lambda p:float(wave.validation_base(p,xv,tv,uv).block_until_ready()),
                wave.PCA:lambda p:float(wave.validation_pca(p,basis,xv,tv,uv).block_until_ready()),
                NORMALIZED:lambda p:float(normalized_validation(p,normalized_state,xv,tv,uv).block_until_ready())}
            init_gap = float(jnp.max(jnp.abs(wave.predict_pca(raw_params,basis,xv,tv)-normalized_prediction(
                params[NORMALIZED],normalized_state,xv,tv))).block_until_ready())
            if init_gap>1e-5:
                raise ValueError("La estandarización no preservó la predicción física inicial.")
            for component,(mean,std) in enumerate(zip(np.asarray(center),np.asarray(scale)),start=1):
                scaler_rows.append({"case":name,"seed":seed,"component":component,
                                    "training_mean":float(mean),"training_std":float(std)})
            order = list(orders[index % len(orders)])
            partitions[name][str(seed)] = {**expected,"model_execution_order":order,
                "physical_initialization_max_abs_difference":init_gap,
                "coefficient_center":np.asarray(center).tolist(),"coefficient_scale":np.asarray(scale).tolist()}
            initial_stats = {}
            for model in MODELS:
                key = "base" if model==wave.BASE else f'{model}_k{k}_w{pw}'
                if key not in compilation:
                    start = time.perf_counter()
                    warmup_update(updates[model],params[model],batches[0],states[model])
                    validation_fns[model](params[model])
                    compilation[key] = time.perf_counter()-start
                terms,norms = parameter_diagnostics(params[model],states[model],batches[0],speed,model)
                initial_stats[model] = np.asarray(terms),np.asarray(norms)
            rng = np.random.default_rng(seed+1901)
            rx=jnp.asarray(rng.uniform(0,1,settings["test_residual_points"]),dtype=jnp.float32)
            rt=jnp.asarray(rng.choice(t[test],settings["test_residual_points"]))
            predictions = {}
            print(f'{name}, seed={seed}, k={k}, initial field gap={init_gap:.2g}',flush=True)
            for model in order:
                preprocess = 0. if model==wave.BASE else pca_seconds+(scale_seconds if model==NORMALIZED else 0.)
                trained,train_seconds,final_loss,trace,targets,_ = _train_trace(
                    updates[model],params[model],batches,validation_fns[model],settings["eval_every_steps"],
                    settings["validation_rmse_targets"],basis_state=states[model],preprocessing_seconds=preprocess)
                if model==wave.BASE:
                    prediction=wave.predict_base(trained,xv,jnp.asarray(t))
                elif model==wave.PCA:
                    prediction=wave.predict_pca(trained,basis,xv,jnp.asarray(t))
                else:
                    prediction=normalized_prediction(trained,normalized_state,xv,jnp.asarray(t))
                prediction=np.asarray(prediction.block_until_ready());predictions[model]=prediction
                difference=prediction[:,test]-u[:,test]
                physical = physical_diagnostics(trained,states[model],rx,rt,xv,jnp.asarray(u[:,0]),
                    jnp.asarray(t[test]),speed,model)
                physical={key:float(value.block_until_ready()) for key,value in physical.items()}
                final_terms,final_norms = parameter_diagnostics(trained,states[model],batches[-1],speed,model)
                final_terms,final_norms = np.asarray(final_terms),np.asarray(final_norms)
                initial_terms,initial_norms=initial_stats[model]
                row={"case":name,"model":model,"seed":seed,"k":k if model!=wave.BASE else None,
                    "parameters":base_parameters if model==wave.BASE else pc,"width":width if model==wave.BASE else pw,
                    "pca_fit_seconds":pca_seconds if model!=wave.BASE else 0.,
                    "scaler_fit_seconds":scale_seconds if model==NORMALIZED else 0.,
                    "training_seconds":train_seconds,"pipeline_seconds":train_seconds+preprocess,
                    "validation_rmse":trace[-1]["validation_rmse"],"final_training_loss":final_loss,
                    "test_rmse":float(np.sqrt(np.mean(difference**2))),
                    "test_relative_l2":float(np.linalg.norm(difference)/np.linalg.norm(u[:,test])),**physical,
                    **{f'loss_{n}':float(v) for n,v in zip(wave.TERM_NAMES,final_terms)}}
                row.update({f"time_to_val_rmse_{target.replace('.', '_')}":seconds for target,seconds in targets.items()})
                rows.append(row)
                for checkpoint,terms,norms in [(0,initial_terms,initial_norms),(settings["steps"],final_terms,final_norms)]:
                    for term,value,norm in zip(wave.TERM_NAMES,terms,norms):
                        gradient_rows.append({"case":name,"model":model,"seed":seed,"step":checkpoint,
                                              "term":term,"mse":float(value),"parameter_gradient_norm":float(norm)})
                traces.extend({"case":name,"model":model,"seed":seed,**point} for point in trace)
                for harmonic,amplitude in zip(case["harmonics"],case["amplitudes"]):
                    spatial=np.sin(harmonic*np.pi*np.asarray(x,dtype=float))
                    coefficient=2*np.trapezoid(prediction[:,test]*spatial[:,None],x,axis=0)
                    truth_coefficient=amplitude*np.cos(speed*harmonic*np.pi*t[test])
                    modes.append({"case":name,"model":model,"seed":seed,"harmonic":harmonic,
                        "coefficient_relative_l2":float(np.linalg.norm(coefficient-truth_coefficient)/np.linalg.norm(truth_coefficient))})
                print(f'  {model}: L2={row["test_relative_l2"]:.5f}, PDE={row["test_pde_residual_rms"]:.5f}, pipeline={row["pipeline_seconds"]:.2f}s',flush=True)
            if index==0:
                np.savez_compressed(output/f'example_{name}.npz',x=x,t=t,truth=u,baseline=predictions[wave.BASE],
                    raw_pca=predictions[wave.PCA],normalized_pca=predictions[NORMALIZED],seed=seed,test_ids=test)
            for filename,records in [("metrics.csv",rows),("validation_trajectories.csv",traces),
                ("mode_metrics.csv",modes),("gradient_diagnostics.csv",gradient_rows),("coefficient_scalers.csv",scaler_rows)]:
                wave._write_csv(output/filename,records)
            (output/'progress.json').write_text(json.dumps({"completed_fits":len(rows),
                "planned_fits":len(MODELS)*len(settings['cases'])*len(settings['seeds'])}),encoding='utf-8')
    summaries,paired = {},{}
    metrics=("test_relative_l2","test_pde_residual_rms","pipeline_seconds","validation_rmse","initial_u_rmse","initial_ut_rms")
    for case in settings['cases']:
        name=case['name'];summaries[name]={};paired[name]={}
        for model in MODELS:
            selected=[r for r in rows if r['case']==name and r['model']==model]
            summaries[name][model]={key:_mean_std([r[key] for r in selected]) for key in metrics}
        for other,reference in [(NORMALIZED,wave.PCA),(NORMALIZED,wave.BASE),(wave.PCA,wave.BASE)]:
            a={r['seed']:r for r in rows if r['case']==name and r['model']==reference}
            b={r['seed']:r for r in rows if r['case']==name and r['model']==other}
            paired[name][f'{other} minus {reference}']={key:paired_report(a,b,key,config) for key in metrics[:4]}
    if source_hashes(root,source,settings)!=original_hashes:
        raise ValueError("Cambió un artefacto de la corrida fuente durante el diagnóstico.")
    metadata={"configuration":config,"source_configuration":settings,"summary":summaries,"paired":paired,
        "partitions":partitions,"oracle_gate":oracle_report['gate'],"source_sha256":original_hashes,
        "compilation_seconds_by_architecture":compilation,"runs":rows,
        "runtime":{"python":platform.python_version(),"jax":jax.__version__,"numpy":np.__version__,"device":str(jax.devices()[0])},
        "single_factor":"Only an affine PCA coefficient decoder, fit using training snapshots. Inverse last-layer transformation matches initial physical predictions. Loss remains in physical units; no coefficient-supervision loss was added.",
        "timing_scope":"Pipeline includes PCA/spline fitting, scaler fitting when applicable, training and validation. Compilations, oracle checks, gradient diagnostics, batch generation and final scoring are excluded.",
        "gradient_scope":"Unweighted loss-term gradient norms in each model's own parameter coordinates, evaluated on first/last shared training batches. Norms across parameterizations are not directly invariant."}
    (output/'summary.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    create_report(root,output)
    print(f'Saved diagnostic round: {output}',flush=True)
    return metadata
