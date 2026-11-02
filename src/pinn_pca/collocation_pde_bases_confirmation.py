"""Frozen fresh-seed confirmation of the centered-PCA versus POD contrast."""
import json
from pathlib import Path
from .collocation import _run_collocation_seed
from .collocation_hard_conditions import HardRecorder
from .collocation_pde_bases import run as run_bases

FROZEN=("data_path","width","learning_rate","warmup_steps","train_steps","batch_size","update_every",
        "candidate_count","history_snapshots","independent_count","initial_condition_points",
        "initial_condition_strategy","test_time_count","validation_time_count","test_checkpoints")

def validate_confirmation(config,development):
    if len(config["seeds"])!=10 or len(set(config["seeds"]))!=10 or set(config["seeds"])&set(development["seeds"]):
        raise ValueError("Ten distinct fresh seeds, disjoint from development, are required")
    for key in FROZEN:
        if config[key]!=development[key]:raise ValueError(f"Confirmation must preserve {key}")
    if not config.get("confirmation") or config["primary_comparison"]!="centered_pca_minus_pod_terminal_rmse":
        raise ValueError("Frozen primary comparison is required")

def append_primary_result(config,root,result):
    path=Path(root)/"pca_collocation/notebooks/burgers"/config["notebook_name"]
    notebook=json.loads(path.read_text(encoding="utf-8"))
    primary=result["primary"]
    text=("## Resultado de la comparación principal\n\n"
          "PCA centrada menos POD en RMSE terminal: "
          f"diferencia media {primary['mean_difference']:+.5f}; mejora en {primary['wins']}/10 semillas. "
          f"Intervalo bootstrap pareado 95 % [{primary['bootstrap95'][0]:+.5f}, {primary['bootstrap95'][1]:+.5f}].\n\n"
          "Criterio fijado: media negativa e intervalo completamente por debajo de cero. "
          +("La ventaja se confirma según este criterio en esta cohorte." if result["advantage_confirmed"] else
            "La ventaja no se confirma según este criterio; esto no demuestra equivalencia entre bases.")
          +" Los contrastes secundarios se presentan como descriptivos. Diez semillas limitan la precisión de la estimación.")
    notebook["cells"]=[cell for cell in notebook["cells"] if cell.get("id")!="primary-confirmation"]
    notebook["cells"].append(dict(id="primary-confirmation",cell_type="markdown",metadata={},source=text))
    path.write_text(json.dumps(notebook,ensure_ascii=False,indent=2),encoding="utf-8")

def run(config,root):
    root=Path(root).resolve()
    development=json.loads((root/"pca_collocation/configs/burgers_pde_bases.json").read_text())
    validate_confirmation(config,development)
    output=root/"pca_collocation/results/burgers/pde_bases"/config["run_id"]
    output.mkdir(parents=True,exist_ok=True)
    p=output/"config.json"
    if p.exists() and json.loads(p.read_text())!=config:raise ValueError("Use new run_id for changed protocol")
    p.write_text(json.dumps(config,indent=2))
    for seed in config["seeds"]:
        for selector in ("pca","sobol"):
            folder=root/config["reference_root"]/selector/f"seed_{seed}"
            if not (folder/"constraints.csv").exists():
                print(f"RUN confirmation reference {selector} seed={seed}",flush=True)
                spec=dict(base="sobol",label="Sobol") if selector=="sobol" else dict(
                    base="pca_mean_qr",label="PCA + media 95%",adaptive_basis_threshold=.95,oversample_count=32,direct_rank=True)
                _run_collocation_seed(dict(config,seed=seed,hard_conditions=True,pde_only=True,
                    method_specs={selector:spec},common_selection_seed=True,skip_seed_plots=True),root,folder,observer=HardRecorder(config,folder))
                print(f"DONE confirmation reference {selector} seed={seed}",flush=True)
    run_bases(config,root)
    summary=json.loads((output/"summary.json").read_text())
    primary=next(r for r in summary["paired"] if r["method"]=="pca_qr" and r["reference"]=="pod_qr" and r["metric"]=="test_rmse")
    result=dict(primary=primary,criterion="negative mean and bootstrap95 upper endpoint below zero",
        advantage_confirmed=primary["mean_difference"]<0 and primary["bootstrap95"][1]<0,
        limitation="Ten fresh seeds; failure to confirm does not establish equivalence; secondary contrasts are descriptive")
    (output/"primary_confirmation.json").write_text(json.dumps(result,indent=2))
    append_primary_result(config,root,result)
    print(json.dumps(result,indent=2),flush=True)
