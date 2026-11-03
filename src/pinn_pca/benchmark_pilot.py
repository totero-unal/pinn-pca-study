"""Three-PDE pilot infrastructure. Importing this module never runs training."""
from pathlib import Path
import json
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.linalg import qr
import matplotlib.pyplot as plt

from .models import init_mlp, mlp
from .training import adam_init, adam_step
from .collocation import _save_csv
from .collocation_basis_ablation import read_rows
from .collocation_rank_study import adaptive_basis
from .notebook_images import embed_notebook_images
from .atomic_io import write_json

METHODS=('random','hammersley','random_r','rad','pca_mean')
PROBLEMS=('burgers','wave','allen_cahn')


def uniform(n, seed):
    unit=np.random.default_rng(seed).random((n,2))
    return np.column_stack((2*unit[:,0]-1,unit[:,1])).astype(np.float32)


def hammersley(n, seed):
    indices=np.arange(n);value=np.zeros(n);factor=.5
    while np.any(indices):
        value+=factor*(indices%2);indices//=2;factor*=.5
    unit=np.column_stack(((np.arange(n)+.5)/n,value))
    unit=(unit+np.random.default_rng(seed).random(2))%1
    return np.column_stack((2*unit[:,0]-1,unit[:,1])).astype(np.float32)


def initial(problem,x):
    if problem=='burgers':return -jnp.sin(jnp.pi*x)
    if problem=='wave':return jnp.sin(jnp.pi*x)+.5*jnp.sin(4*jnp.pi*x)
    if problem=='allen_cahn':return x*x*jnp.cos(jnp.pi*x)
    raise ValueError(problem)


def operators(problem, sign=1):
    boundary=-1. if problem=='allen_cahn' else 0.
    def solution(params,x,t):
        u0=jnp.where(jnp.abs(x)==1,boundary,initial(problem,x))
        power=2 if problem=='wave' else 1
        return u0+t**power*(1-x*x)*mlp(params,jnp.stack((x,t)))[0]
    def residual(params,x,t):
        f=lambda xx,tt:solution(params,xx,tt)
        u=f(x,t);ux=jax.grad(f,0)(x,t);ut=jax.grad(f,1)(x,t)
        uxx=jax.grad(lambda xx:jax.grad(f,0)(xx,t))(x)
        if problem=='wave':return jax.grad(lambda tt:jax.grad(f,1)(x,tt))(t)-4*uxx
        if problem=='burgers':return ut+u*ux-(.01/np.pi)*uxx
        return ut-.0001*uxx-sign*5*(u**3-u)
    predict=jax.jit(jax.vmap(lambda params,p:solution(params,p[0],p[1]),in_axes=(None,0)))
    field=jax.jit(jax.vmap(lambda params,p:residual(params,p[0],p[1]),in_axes=(None,0)))
    return solution,predict,field


def field_chunks(fn, params, points, chunk=256):
    return np.concatenate([np.asarray(fn(params,jnp.asarray(points[i:i+chunk]))) for i in range(0,len(points),chunk)])


def prepare(root,c):
    root=Path(root);full=c.get('stage')=='full_benchmark'
    out=root/('pca_collocation/results/benchmark' if full else 'pca_collocation/results/benchmark_pilot')/c['run_id']
    out.mkdir(parents=True,exist_ok=True)
    p=out/'config.json'
    if p.exists() and json.loads(p.read_text(encoding='utf-8'))!=c:raise ValueError('Use another run_id for a changed configuration')
    if full:
        assert c['train_steps']==c['schedule_horizon']==100000
        assert len(c['seeds'])==len(set(c['seeds']))==10 and 3101 not in c['seeds']
        assert tuple(c['methods'])==METHODS and tuple(c['problems'])==PROBLEMS
        assert c['active_count']==2000 and c['update_every']==1000
        assert c['snapshot_every']==50 and c['pca_threshold']==.95
        assert c['hidden_layers']==5 and c['width']==64
    else:
        if set(c['pilot_seeds'])&set(c['confirmation_seeds_reserved']):raise ValueError('Pilot/confirmation overlap')
        if not 0<c['pilot_steps']<c['full_study_steps']:raise ValueError('This runner is pilot-only')
    assert c['update_every']//c['snapshot_every']==20
    write_json(p,c)
    return out


def references(root,c):
    out=prepare(root,c);folder=out/'references';folder.mkdir(exist_ok=True)
    nx,nt=c['test_shape'];x=np.linspace(-1,1,nx);t=np.linspace(0,1,nt)
    xx,tt=np.meshgrid(x,t,indexing='ij');points=np.column_stack((xx.ravel(),tt.ravel()))
    audits=[]
    for problem in c['problems']:
        path=folder/f'{problem}.npz';audit=folder/f'{problem}_audit.json'
        if path.exists() and audit.exists():
            info=json.loads(audit.read_text());assert info['passed'];audits.append(info);continue
        print(f'REFERENCE {problem}',flush=True)
        if problem=='wave':
            truth=np.sin(np.pi*xx)*np.cos(2*np.pi*tt)+.5*np.sin(4*np.pi*xx)*np.cos(8*np.pi*tt)
            info=dict(problem=problem,passed=True,kind='analytic',refinement_relative_difference=0.)
        else:
            values=[]
            for intervals in c['reference_intervals']:
                grid=np.linspace(-1,1,intervals+1);dx=grid[1]-grid[0]
                boundary=-1. if problem=='allen_cahn' else 0.
                u0=grid**2*np.cos(np.pi*grid) if problem=='allen_cahn' else -np.sin(np.pi*grid)
                def rhs(_,interior):
                    u=np.r_[boundary,interior,boundary]
                    diffusion=(u[:-2]-2*u[1:-1]+u[2:])/dx**2
                    if problem=='burgers':return (.01/np.pi)*diffusion-u[1:-1]*(u[2:]-u[:-2])/(2*dx)
                    return .0001*diffusion+c['allen_cahn_reaction_sign']*5*(interior**3-interior)
                sol=solve_ivp(rhs,(0,1),u0[1:-1],method='BDF',t_eval=t,
                    rtol=c['reference_rtol'],atol=c['reference_atol'],
                    jac_sparsity=diags([np.ones(intervals-2),np.ones(intervals-1),np.ones(intervals-2)],[-1,0,1]))
                if not sol.success:raise RuntimeError(sol.message)
                full=np.vstack((np.full(nt,boundary),sol.y,np.full(nt,boundary)))
                values.append(RegularGridInterpolator((grid,t),full)(points).reshape(nx,nt))
            truth=values[-1];delta=float(np.linalg.norm(values[0]-truth)/np.linalg.norm(truth))
            info=dict(problem=problem,kind='central_finite_difference_BDF',refinement_relative_difference=delta,
                passed=delta<c['reference_relative_tolerance'],intervals=c['reference_intervals'],
                note='Two-grid consistency check, not a rigorous error bound or paper ground truth')
        write_json(audit,info)
        if not info['passed']:raise ValueError(f'Reference refinement failed for {problem}: {info}')
        np.savez_compressed(path,points=points.astype(np.float32),truth=truth.ravel(),x=x,t=t)
        audits.append(info)
    write_json(folder/'audit.json',audits)
    return folder


def verify_conditions(c, problem, params):
    solution,_,_=operators(problem,c['allen_cahn_reaction_sign'])
    x=jnp.linspace(-1,1,129);t=jnp.linspace(0,1,129)
    boundary=-1. if problem=='allen_cahn' else 0.
    expected=jnp.where(jnp.abs(x)==1,boundary,initial(problem,x))
    ic=np.asarray(jax.vmap(lambda a:solution(params,a,0.))(x)-expected)
    bc=np.asarray(jax.vmap(lambda b:solution(params,-1.,b))(t))-boundary
    bc2=np.asarray(jax.vmap(lambda b:solution(params,1.,b))(t))-boundary
    velocity=0.
    if problem=='wave':velocity=float(jnp.max(jnp.abs(jax.vmap(lambda a:jax.grad(lambda b:solution(params,a,b))(0.))(x))))
    info=dict(ic_max=float(np.max(np.abs(ic))),bc_max=float(max(np.max(np.abs(bc)),np.max(np.abs(bc2)))),initial_velocity_max=velocity)
    assert max(info.values())<1e-6
    return info


def save_state(path,params,state,step):
    payload={f'layer_{i}_{j}':np.asarray(a) for i,pair in enumerate(params) for j,a in enumerate(pair)}
    if state is not None:
        for name,tree in zip(('adam_m','adam_v'),state[:2]):
            payload.update({f'{name}_{i}_{j}':np.asarray(a) for i,pair in enumerate(tree) for j,a in enumerate(pair)})
        payload['adam_step']=np.asarray(state[2])
    np.savez_compressed(path,step=step,**payload)


def run_experiments(root,c,problem_only=None):
    out=prepare(root,c);references(root,c)
    full=c.get('stage')=='full_benchmark'
    steps=c.get('train_steps',c.get('pilot_steps'))
    seeds=c.get('seeds',c.get('pilot_seeds'))
    problems=[problem_only] if problem_only else c['problems']
    if problem_only and problem_only not in c['problems']:raise ValueError(problem_only)
    status_folder=out/problem_only if problem_only else out
    status_folder.mkdir(parents=True,exist_ok=True)
    total=len(problems)*len(c['methods'])*len(seeds);done=0
    try:
        for problem in problems:
            _,predict,field=operators(problem,c['allen_cahn_reaction_sign'])
            def loss(params,active):return jnp.mean(field(params,active)**2)
            @jax.jit
            def update(params,state,active):
                value,grads=jax.value_and_grad(loss)(params,active)
                lr=.5*c['learning_rate']*(1+jnp.cos(jnp.pi*state[2]/c['schedule_horizon']))
                p,m,v,s=adam_step(params,grads,*state,lr)
                return p,(m,v,s),value
            ref=np.load(out/'references'/f'{problem}.npz');test=ref['points'];truth=ref['truth']
            for seed in seeds:
                init=init_mlp(jax.random.PRNGKey(seed),[2]+[c['width']]*c['hidden_layers']+[1])
                initial_conditions=verify_conditions(c,problem,init)
                validation=uniform(c['validation_count'],seed+100000)
                independent=uniform(c['validation_count'],seed+100001)
                pool=uniform(c['pca_candidates'],seed+200000)
                for method in c['methods']:
                    folder=out/problem/method/f'seed_{seed}';folder.mkdir(parents=True,exist_ok=True)
                    if (folder/'completed.json').exists():
                        marker=json.loads((folder/'completed.json').read_text())
                        assert marker['passed'] and marker['steps']==steps
                        done+=1;print(f'SKIP {problem} {method} seed={seed}',flush=True);continue
                    write_json(status_folder/'status.json',dict(state='running',problem=problem,method=method,seed=seed,completed_runs=done,total_runs=total),critical=False)
                    params=init;state=adam_init(params);best=None;best_loss=np.inf;best_step=0
                    active=hammersley(c['active_count'],seed) if method=='hammersley' else uniform(c['active_count'],seed)
                    starting=active.copy();history=[];traces=[];selection=[];histories={};point_sets={'step_0':active.copy()}
                    start=time.perf_counter();print(f'RUN {problem} {method} seed={seed}',flush=True)
                    for step in range(steps+1):
                        if step:
                            params,state,train_loss=update(params,state,jnp.asarray(active))
                            if not np.isfinite(float(train_loss)):raise FloatingPointError(f'Nonfinite loss at {step}')
                        if method=='pca_mean' and step and step%c['snapshot_every']==0:
                            history.append(field_chunks(field,params,pool))
                        if step%c['validation_every']==0 or step==steps:
                            residual=field_chunks(field,params,validation);val=float(np.mean(residual**2))
                            if not np.isfinite(val):raise FloatingPointError('Nonfinite validation')
                            if val<best_loss:best_loss=val;best=jax.tree.map(lambda a:a.copy(),params);best_step=step
                            traces.append(dict(problem=problem,method=method,seed=seed,step=step,validation_residual_mse=val,elapsed_seconds=time.perf_counter()-start))
                            _save_csv(folder/'trace.csv',traces)
                            print(f'PROGRESS {problem} {method} seed={seed} {step}/{steps}',flush=True)
                            write_json(status_folder/'status.json',dict(state='running',problem=problem,method=method,seed=seed,step=step,completed_runs=done,total_runs=total),critical=False)
                        if full and step in c['state_checkpoints']:
                            save_state(folder/f'state_{step}.npz',params,state,step)
                        if step and step%c['update_every']==0 and step<steps:
                            rng_seed=seed+300000+step
                            k=0
                            if method=='random_r':active=uniform(c['active_count'],rng_seed)
                            elif method=='rad':
                                candidates=uniform(c['rad_candidates'],rng_seed)
                                weights=np.abs(field_chunks(field,params,candidates)).astype(float)**2
                                if not np.isfinite(weights).all():raise FloatingPointError('RAD weights')
                                probability=weights/weights.sum() if weights.sum() else None
                                ids=np.random.default_rng(rng_seed+1).choice(len(candidates),c['active_count'],replace=False,p=probability)
                                active=candidates[ids]
                            elif method=='pca_mean':
                                matrix=np.column_stack(history);assert matrix.shape==(c['pca_candidates'],20)
                                basis,info=adaptive_basis(matrix,'pca_mean_qr',c['pca_threshold']);k=basis.shape[1]
                                _,_,ids=qr(basis.T,mode='economic',pivoting=True)
                                active=np.vstack((pool[ids[:k]],uniform(c['active_count']-k,rng_seed)))
                                histories[f'step_{step}']=matrix
                                selection.append(dict(problem=problem,method=method,seed=seed,step=step,rank=k,
                                    spectral_modes=info['spectral_modes'],guided_fraction=k/c['active_count'],
                                    spectral_fraction=info['spectral_fraction']))
                            assert len(np.unique(active,axis=0))==c['active_count']
                            if method in ('random','hammersley'):np.testing.assert_array_equal(active,starting)
                            if method in ('random_r','rad','pca_mean'):point_sets[f'step_{step}']=active.copy()
                            history=[]
                    terminal_seconds=time.perf_counter()-start
                    if method=='pca_mean':histories[f'step_{steps}']=np.column_stack(history)
                    metrics=[]
                    for label,model,at in (('best_validation',best,best_step),('terminal',params,steps)):
                        pred=field_chunks(predict,model,test);delta=pred-truth
                        metrics.append(dict(problem=problem,method=method,seed=seed,checkpoint=label,step=at,
                            relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(truth)),rmse=float(np.sqrt(np.mean(delta**2))),
                            independent_residual_rms=float(np.sqrt(np.mean(field_chunks(field,model,independent)**2))),
                            pipeline_seconds=terminal_seconds))
                    final_conditions=verify_conditions(c,problem,params)
                    _save_csv(folder/'metrics.csv',metrics);_save_csv(folder/'selection.csv',selection)
                    np.savez_compressed(folder/'active_points.npz',**point_sets)
                    np.savez_compressed(folder/'residual_histories.npz',candidates=pool,**histories)
                    save_state(folder/'terminal_state.npz',params,state,steps)
                    save_state(folder/'best_validation_weights.npz',best,None,best_step)
                    write_json(folder/'constraints.json',dict(initial=initial_conditions,terminal=final_conditions))
                    write_json(folder/'completed.json',dict(passed=True,steps=steps,active_count=len(active)))
                    done+=1
        if not full:report(root,c)
        write_json(status_folder/'status.json',dict(state='complete',completed_runs=done,total_runs=total),critical=False)
    except BaseException as exc:
        import traceback
        details=traceback.format_exc()
        write_json(status_folder/'status.json',dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',completed_runs=done,total_runs=total,error=str(exc)),critical=False)
        try:(status_folder/'last_error.txt').write_text(details,encoding='utf-8')
        except OSError:pass
        raise
    return out


def run_pilot(root,c):
    if c.get('stage')=='full_benchmark':raise ValueError('Use the full benchmark runner')
    return run_experiments(root,c)


def report(root,c):
    root=Path(root);out=prepare(root,c);rows=[];ranks=[]
    for problem in c['problems']:
        for method in c['methods']:
            for seed in c['pilot_seeds']:
                folder=out/problem/method/f'seed_{seed}'
                if not (folder/'completed.json').exists():raise RuntimeError(f'Incomplete: {folder}')
                rows.extend(read_rows(folder/'metrics.csv'))
                if method=='pca_mean':ranks.extend(read_rows(folder/'selection.csv'))
    _save_csv(out/'metrics.csv',rows);_save_csv(out/'pca_rank_diagnostics.csv',ranks)
    fig,axes=plt.subplots(1,len(c['problems']),figsize=(15,4),squeeze=False)
    for ax,problem in zip(axes.flat,c['problems']):
        group=[r for r in rows if r['problem']==problem and r['checkpoint']=='best_validation']
        ax.bar([r['method'] for r in group],[float(r['relative_l2']) for r in group]);ax.set(title=problem,ylabel='Error relativo L2 — piloto')
        ax.tick_params(axis='x',rotation=40)
    fig.tight_layout();fig.savefig(out/'pilot_errors.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(c['problems']),len(c['methods']),figsize=(17,9),squeeze=False)
    for i,problem in enumerate(c['problems']):
        for j,method in enumerate(c['methods']):
            ax=axes[i,j]
            with np.load(out/problem/method/f'seed_{c["pilot_seeds"][0]}'/'active_points.npz') as z:
                key=sorted(z.files,key=lambda s:int(s.split('_')[-1]))[-1];step=int(key.split('_')[-1]);p=z[key]
            k=int(next(r['rank'] for r in ranks if r['problem']==problem and int(r['step'])==step)) if method=='pca_mean' else 0
            ax.scatter(p[k:,0],p[k:,1],s=2,alpha=.2)
            if k:ax.scatter(p[:k,0],p[:k,1],s=25,label=f'QR: {k}');ax.legend(fontsize=7)
            ax.set(title=f'{problem} / {method}',xlabel='x',ylabel='t',xlim=(-1,1),ylim=(0,1))
    fig.tight_layout();fig.savefig(out/'pilot_points.png',dpi=150);plt.close(fig)
    max_fraction=max(float(r['guided_fraction']) for r in ranks)
    finding=dict(stage='pilot_only',completed_runs=len(rows)//2,max_guided_fraction=max_fraction,
        warning='QR selects a small fraction: inspect dilution before freezing full study' if max_fraction<.05 else 'Inspect per-problem rank fractions before freezing full study',
        full_study_authorized_or_started=False)
    write_json(out/'pilot_review.json',finding)
    table='| PDE | Paso | Rango | Fracción QR |\n|---|---:|---:|---:|\n'
    for r in ranks:table+=f"| {r['problem']} | {r['step']} | {r['rank']} | {float(r['guided_fraction']):.4%} |\n"
    text='# Resultados del piloto de protocolo\n\nUna semilla por combinación, 3000 pasos: no demuestra superioridad y no congela automáticamente el estudio completo. El modelo principal minimiza el residuo de validación, sin usar la solución de prueba para elegirlo; se conserva también el terminal.\n\n'+table+'\n'+finding['warning']+'\n\n'
    for name in ('pilot_errors','pilot_points'):text+=f'![{name}](../../results/benchmark_pilot/{c["run_id"]}/{name}.png)\n\n'
    text+='Referencias y restricciones se auditan antes de aceptar cada corrida. Revisar references/*_audit.json, constraints.json y pilot_review.json. La comparación temporal no incluye la preparación única de referencias y sí incluye compilación y observación propia de cada selector; no permite afirmar aceleración. Hammersley usa un desplazamiento módulo uno por semilla para generar repeticiones. Allen–Cahn usa el signo positivo de reacción escrito en el artículo. Los resultados son propios de JAX y no una reproducción exacta de PyTorch.\n'
    p=root/'pca_collocation/notebooks/burgers/34_protocol_pilot_results.ipynb'
    p.write_text(json.dumps(dict(cells=[dict(id='pilot-results',cell_type='markdown',metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding='utf-8')
    embed_notebook_images(p,root)
    return out
