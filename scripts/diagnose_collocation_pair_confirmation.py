"""Offline diagnostics of completed PCA+mean/POD runs; never trains a PINN."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr, subspace_angles
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pinn_pca.collocation_basis_ablation import read_rows, fixed_basis
from pinn_pca.collocation_rank_study import adaptive_basis
from pinn_pca.collocation import _save_csv
from pinn_pca.notebook_images import embed_notebook_images


def projection(basis, matrix):
    total=np.linalg.norm(matrix)**2
    return float(np.linalg.norm(basis.T@matrix)**2/total) if total else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='pca_collocation/results/burgers/pair_confirmation/pca_mean_vs_pod_confirmation_10new_100k_v1')
    args=parser.parse_args()
    source=ROOT/args.source
    c=json.loads((source/'config.json').read_text())
    assert json.loads((source/'status.json').read_text())['state']=='complete'
    out=source/'diagnostics';out.mkdir(exist_ok=True)
    summaries=[];updates=[];snapshots=[];final_points={};reconstructed=0
    results=read_rows(source/'individual_pca_mean_vs_pod.csv')
    bins=(np.linspace(-1,1,17),np.linspace(0,.99,17))
    for seed in c['seeds']:
        print(f'DIAGNOSE seed={seed}',flush=True)
        folders={m:source/m/f'seed_{seed}' for m in ('pca','pod_qr')}
        diagnostics={m:read_rows(p/'selection_diagnostics.csv') for m,p in folders.items()}
        archives={m:np.load(p/'active_history.npz') for m,p in folders.items()}
        per_seed=[]
        assert len(diagnostics['pca'])==len(diagnostics['pod_qr'])==c['train_steps']//c['update_every']
        for a,b in zip(diagnostics['pca'],diagnostics['pod_qr']):
            assert a['update_step']==b['update_step']
            step=int(a['update_step'])-c['update_every'];k=int(a['retained_modes'])
            assert k==int(b['selected_qr_points'])
            active={m:z[f'{m}_{step}'] for m,z in archives.items()}
            np.testing.assert_array_equal(active['pca'][k:],active['pod_qr'][k:])
            points={m:p[:k] for m,p in active.items()}
            sets={m:set(map(tuple,p)) for m,p in points.items()}
            intersection=len(sets['pca']&sets['pod_qr'])
            hist={m:np.histogram2d(p[:,0],p[:,1],bins=bins)[0]/k for m,p in points.items()}
            distances=np.linalg.norm((points['pca'][:,None,:]-points['pod_qr'][None,:,:])/np.array([2.,.99]),axis=2)
            row=dict(seed=seed,selection_step=step,continuation=step-c['warmup_steps'],rank=k,
                guided_overlap=intersection/k,guided_jaccard=intersection/(2*k-intersection),
                histogram_total_variation=float(.5*np.abs(hist['pca']-hist['pod_qr']).sum()),
                symmetric_nearest_distance=float(.5*(distances.min(axis=0).mean()+distances.min(axis=1).mean())),
                pod_own_mean_capture=float(b['mean_projection_fraction']))
            for m,p in points.items():
                row[m+'_central_fraction']=float(np.mean(np.abs(p[:,0])<.1))
                row[m+'_late_fraction']=float(np.mean(p[:,1]>.5))
            updates.append(row);per_seed.append(row)
            final_points[seed]=points
        for archive in archives.values():archive.close()
        r=next(r for r in results if int(r['seed'])==seed)
        item=dict(seed=seed,pca_rmse=float(r['pca_mean']),pod_rmse=float(r['pod']),difference=float(r['difference']),
            rank_mean=float(np.mean([v['rank'] for v in per_seed])),rank_min=min(v['rank'] for v in per_seed),
            rank_max=max(v['rank'] for v in per_seed))
        for key in ('guided_overlap','guided_jaccard','histogram_total_variation','symmetric_nearest_distance','pod_own_mean_capture',
                    'pca_central_fraction','pod_qr_central_fraction','pca_late_fraction','pod_qr_late_fraction'):
            item[key]=float(np.mean([v[key] for v in per_seed]))
        for method,folder in folders.items():
            with np.load(folder/'history_audit.npz') as saved, np.load(folder/'active_history.npz') as active:
                steps=sorted(int(key.split('_')[0]) for key in saved.files if key.endswith('_history'))
                for step in steps:
                    h=saved[f'{step}_history'].astype(float);mean=h.mean(axis=1);centered=h-mean[:,None]
                    # On each source history compare the bases with identical dimension.
                    pca,info=adaptive_basis(h,'pca_mean_qr',.95)
                    k=next(v['rank'] for v in per_seed if v['selection_step']==step)
                    if method=='pca':assert pca.shape[1]==k
                    # POD-source history uses the actual matched dimension, not its autonomous PCA rank.
                    if pca.shape[1]!=k:pca=fixed_basis(h,'pca_mean_qr',k)
                    pod=fixed_basis(h,'pod_qr',k)
                    row=dict(seed=seed,source_method=method,selection_step=step,continuation=step-100,rank=k,
                        mean_energy_fraction=float(h.shape[1]*np.linalg.norm(mean)**2/np.linalg.norm(h)**2),
                        max_subspace_angle_degrees=float(np.degrees(subspace_angles(pca,pod)).max()))
                    for label,basis in (('pca',pca),('pod',pod)):
                        row[label+'_mean_capture']=projection(basis,mean)
                        row[label+'_raw_capture']=projection(basis,h)
                        row[label+'_fluctuation_capture']=projection(basis,centered)
                        _,_,pivots=qr(basis.T,mode='economic',pivoting=True)
                        if label==('pca' if method=='pca' else 'pod'):
                            np.testing.assert_array_equal(active[f'{method}_{step}'][:k],saved[f'{step}_pool'][pivots[:k]])
                            reconstructed+=1
                    snapshots.append(row)
        summaries.append(item)
    _save_csv(out/'seed_summary.csv',summaries)
    _save_csv(out/'selection_diagnostics.csv',updates)
    _save_csv(out/'same_history_basis_comparison.csv',snapshots)
    correlations=[]
    for key in ('rank_mean','guided_overlap','histogram_total_variation','pca_central_fraction','pod_qr_central_fraction'):
        values=[r[key] for r in summaries]
        rho=float(spearmanr(values,[r['difference'] for r in summaries]).statistic) if np.ptp(values)>0 else None
        correlations.append(dict(feature=key,spearman_with_final_difference=rho))
    (out/'exploratory_correlations.json').write_text(json.dumps(correlations,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for r in summaries:
        axes[0].plot(r['rank_mean'],r['difference'],'o');axes[0].annotate(str(r['seed']),(r['rank_mean'],r['difference']),fontsize=7)
    axes[0].axhline(0,color='black',lw=.7);axes[0].set(xlabel='Rango medio',ylabel='RMSE PCA + media − POD')
    axes[0].margins(x=.2,y=.15)
    for seed in c['seeds']:
        group=[r for r in updates if r['seed']==seed]
        # Block summaries preserve the original CSV at full resolution.
        blocks=[group[i:i+100] for i in range(0,len(group),100)]
        axes[1].plot([np.mean([r['continuation'] for r in b]) for b in blocks],
                     [np.mean([r['guided_overlap'] for r in b]) for b in blocks],alpha=.6)
    axes[1].set(xlabel='Pasos de continuación',ylabel='Fracción de puntos QR coincidentes')
    for m,marker in (('pca','o'),('pod_qr','x')):
        group=[r for r in snapshots if r['source_method']==m]
        axes[2].scatter([r['mean_energy_fraction'] for r in group],[r['max_subspace_angle_degrees'] for r in group],marker=marker,label=f'Historia {m}',alpha=.6)
    axes[2].set(xlabel='Fracción de energía de la media',ylabel='Mayor ángulo entre bases (grados)');axes[2].legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'diagnostic_overview.png',dpi=150);plt.close(fig)
    ordered=sorted(summaries,key=lambda r:r['difference']);extremes=ordered[:2]+ordered[-2:]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,r in zip(axes.flat,extremes):
        for m,marker in (('pca','o'),('pod_qr','x')):
            p=final_points[r['seed']][m];ax.scatter(p[:,0],p[:,1],marker=marker,label=m)
        ax.set(xlabel='x',ylabel='t',xlim=(-1,1),ylim=(0,.99),title=f"Semilla {r['seed']}: diferencia {r['difference']:+.4f}");ax.legend()
    fig.suptitle('Última selección QR: dos casos favorables y dos desfavorables a PCA + media')
    fig.tight_layout();fig.savefig(out/'extreme_seed_points.png',dpi=150);plt.close(fig)
    table='| Semilla | Diferencia RMSE | Rango medio | Coincidencia QR media |\n|---:|---:|---:|---:|\n'
    for r in summaries:table+=f"| {r['seed']} | {r['difference']:+.5f} | {r['rank_mean']:.2f} | {r['guided_overlap']:.3f} |\n"
    pooled_mean=np.mean([r['mean_energy_fraction'] for r in snapshots]);capture=np.mean([r['pod_mean_capture'] for r in snapshots])
    overlap=np.mean([r['guided_overlap'] for r in updates])
    distance=np.mean([r['symmetric_nearest_distance'] for r in updates])
    findings=dict(mean_guided_overlap=float(overlap),saved_history_mean_energy_fraction=float(pooled_mean),
        mean_symmetric_nearest_distance=float(distance),
        pca_central_fraction=float(np.mean([r['pca_central_fraction'] for r in updates])),
        pod_central_fraction=float(np.mean([r['pod_qr_central_fraction'] for r in updates])),
        saved_history_pod_mean_capture=float(capture),qr_reconstructions=reconstructed,
        full_training_updates=len(updates),saved_history_comparisons=len(snapshots),
        latest_saved_history_continuation=max(r['continuation'] for r in snapshots))
    (out/'findings.json').write_text(json.dumps(findings,indent=2))
    relative=out.relative_to(ROOT/'pca_collocation').as_posix()
    text=('# Diagnóstico por semilla: PCA + media frente a POD\n\n'
        'Análisis posterior y exploratorio de la confirmación 2111–2120. No se entrenan redes, no se cambian hiperparámetros y no se selecciona un nuevo ganador. Diferencias negativas favorecen PCA + media.\n\n'+table+
        '\n## Qué queda controlado\n\nEl rango y el complemento Sobol coinciden exactamente en cada actualización. Por ello no explican una diferencia de presupuesto entre métodos. El rango puede variar entre semillas, pero su asociación con el resultado es descriptiva. Se analiza solo el subconjunto QR al comparar ubicaciones: incluir los puntos Sobol compartidos inflaría artificialmente la coincidencia. Las regiones |x| < 0.1 y t > 0.5 son descriptores geométricos elegidos después de ver la cohorte, no una demostración de dónde está el error.\n\n'+
        f'Coincidencia QR promedio: {overlap:.3f}. Distancia simétrica media al vecino más cercano: {distance:.5f}, con x y t normalizados por sus longitudes de dominio. Coincidencia exacta baja no implica regiones alejadas. El histograma usa una rejilla 16×16 y pocos puntos QR: su variación total es sensible a los bordes y a la escasez de puntos. En las historias guardadas, la media aporta en promedio {pooled_mean:.3f} de la energía residual y POD captura {capture:.6f} de la energía de esa media. POD sin centrar también puede representar una media persistente; añadirla explícitamente no garantiza una ventaja.\n\n'+
        f'![Diagnóstico](../../{relative}/diagnostic_overview.png)\n\n'+
        '## Comparación sobre la misma historia\n\nPara cada matriz guardada se reconstruyen las dos bases con igual dimensión y se mide la energía bruta, fluctuaciones y media capturadas, además del mayor ángulo entre subespacios. En historias de PCA se reconstruye su regla adaptativa real; en historias de POD se compara una PCA + media de dimensión fija igual al rango usado. Esta última es una comparación geométrica, no una trayectoria alternativa entrenada. Se verifican los puntos QR reales reconstruidos de cada método.\n\n'+
        'Las matrices completas solo están guardadas en pasos totales 100, 200, 1100, 5100 y 10000 (hasta 9900 de continuación). Las conclusiones sobre energía y subespacios se limitan a ese tramo. Los rangos, puntos y diagnósticos POD sí cubren todo el entrenamiento. No hay estados de red guardados para reconstruir mapas de error terminal; los gráficos de puntos no permiten atribuirles causalmente el error final.\n\n'+
        f'![Casos extremos](../../{relative}/extreme_seed_points.png)\n\n'+
        'Los cuatro casos se eligieron retrospectivamente como los dos mejores y los dos peores contrastes: no son una muestra representativa ni un criterio de selección. La última selección ocurrió a 99900 pasos de continuación y se utilizó en el último intervalo.\n\n'+
        '## Archivos y siguiente control\n\nseed_summary.csv resume las diez semillas; selection_diagnostics.csv conserva todas las actualizaciones; same_history_basis_comparison.csv contiene las comparaciones geométricas. exploratory_correlations.json guarda correlaciones de Spearman con solo diez observaciones: son hipótesis, sin inferencia confirmatoria ni interpretación causal.\n\n'
        'Antes de cambiar el selector conviene instrumentar una próxima prueba con historias tardías y estados de red en presupuestos prefijados, para comparar la representación de la media y el error espacial cerca del resultado terminal. No se inicia esa prueba aquí. Estas semillas ya examinadas no pueden presentarse como una confirmación nueva de futuras decisiones.\n\n'
        'Reproducir únicamente el análisis: `python scripts/diagnose_collocation_pair_confirmation.py`.')
    notebook=ROOT/'pca_collocation/notebooks/burgers/33_pair_seed_diagnostics.ipynb'
    notebook.write_text(json.dumps(dict(cells=[dict(id='seed-diagnostics',cell_type='markdown',metadata={},source=text)],metadata={},nbformat=4,nbformat_minor=5),ensure_ascii=False,indent=2),encoding='utf-8')
    embed_notebook_images(notebook,ROOT)
    print(json.dumps(findings,indent=2))


if __name__=='__main__':main()
