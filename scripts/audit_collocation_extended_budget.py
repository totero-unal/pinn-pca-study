"""Audit completed extended-budget artifacts, including embedded notebook images."""
import argparse
import base64
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='pca_collocation/configs/burgers_budget_100k.json')
args = parser.parse_args()
c = json.loads((ROOT / args.config).read_text())
out = ROOT / 'pca_collocation/results/burgers/pde_bases' / c['run_id']
counts = dict(completed_runs=0, active_sets=0, budget_rows=0, constraint_rows=0,
              embedded_pngs=0, code_cells=0)
max_ic = max_bc = 0.0
expected_sets = c['train_steps'] // c['update_every']
for method in ('pca', 'sobol', 'pod_qr', 'pca_qr'):
    for seed in c['seeds']:
        folder = ((ROOT / c['reference_root']) if method in ('pca', 'sobol') else out) / method / f'seed_{seed}'
        prefix = json.loads((folder / 'prefix_verified.json').read_text())
        assert prefix['passed'] and prefix['through_step'] == c['prefix_step']
        assert prefix['active_sets'] == (c['prefix_step'] - c['warmup_steps']) // c['update_every']
        counts['completed_runs'] += 1
        with np.load(folder / 'active_history.npz') as history:
            if expected_sets is not None:
                assert len(history.files) == expected_sets
            for key in history.files:
                points = history[key]
                assert points.shape == (c['batch_size'], 2), (folder, key, points.shape)
                assert np.isfinite(points).all()
                assert len(np.unique(points, axis=0)) == c['batch_size']
                counts['active_sets'] += 1
        with (folder / 'budget_metrics.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        assert [int(r['step']) for r in rows] == c['test_checkpoints']
        assert all(np.isfinite(float(r[k])) for r in rows for k in
                   ('test_rmse', 'independent_residual_rms', 'initial_condition_rmse', 'boundary_rms'))
        counts['budget_rows'] += len(rows)
        with (folder / 'constraints.csv').open(newline='') as stream:
            constraints = list(csv.DictReader(stream))
        assert len(constraints) == len(c['test_checkpoints'])
        max_ic = max(max_ic, *(float(r['ic_max']) for r in constraints))
        max_bc = max(max_bc, *(float(r['bc_max']) for r in constraints))
        counts['constraint_rows'] += len(constraints)
assert max_ic < 1e-6 and max_bc == 0
notebook = ROOT / 'pca_collocation/notebooks/burgers' / c['notebook_name']
n = json.loads(notebook.read_text(encoding='utf-8'))
for cell in n['cells']:
    if cell['cell_type'] == 'code':
        compile(''.join(cell['source']), str(notebook), 'exec')
        counts['code_cells'] += 1
    for output in cell.get('outputs', []):
        png = output.get('data', {}).get('image/png')
        if png:
            with Image.open(io.BytesIO(base64.b64decode(''.join(png)))) as im:
                im.verify()
            counts['embedded_pngs'] += 1
assert counts['embedded_pngs'] >= 3
result = dict(passed=True, **counts, max_ic=max_ic, max_bc=max_bc)
(out / 'extended_audit.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
