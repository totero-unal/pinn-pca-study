# Estudio de tres problemas y cinco selectores

Para ejecutarlos en computadoras distintas y reunir las salidas, seguir [ejecución distribuida e intercambio de resultados](DISTRIBUTED.md). Cada computadora usa un clon local de la misma revisión; los resultados se exportan/importan como ZIP verificados, sin hacer commits de modelos ni historias.

Notebooks preparados, sin ejecutar. Configuración única: `pca_collocation/configs/benchmark_full.json`.

| Notebook | Entrenamientos | Comando desde la raíz |
|---|---:|---|
| [35 — Burgers](35_burgers_benchmark.ipynb) | 50 × 100000 pasos | `python -u scripts/run_collocation_benchmark.py --problem burgers` |
| [36 — Onda](36_wave_benchmark.ipynb) | 50 × 100000 pasos | `python -u scripts/run_collocation_benchmark.py --problem wave` |
| [37 — Allen–Cahn](37_allen_cahn_benchmark.ipynb) | 50 × 100000 pasos | `python -u scripts/run_collocation_benchmark.py --problem allen_cahn` |
| [38 — Agregación](38_benchmark_aggregate.ipynb) | Ninguno | `python scripts/run_collocation_benchmark.py --aggregate` |

Ejecutar secuencialmente en este equipo. El comando equivalente a cada notebook guarda también su registro de consola y genera un notebook `_results.ipynb` con las figuras incrustadas. Alternativamente, abrir el notebook correspondiente con el kernel Python del proyecto y ejecutar sus celdas en orden. No lanzar ambos caminos a la vez.

Preparar PowerShell una vez:

```powershell
Set-Location 'C:\Users\otero\OneDrive\Desktop\pinn-pca-study'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonBenchmark = 'C:\Users\otero\PINNs-otero-toro\.venv-jax-latest\Scripts\python.exe'
```

Después sustituir `python` en cada comando de la tabla por `& $pythonBenchmark`. Por ejemplo:

```powershell
& $pythonBenchmark -u scripts/run_collocation_benchmark.py --problem burgers
```

Se mantienen Random, Hammersley, Random-R, RAD y PCA + media según el piloto inicial, sin ampliar la selección QR. Diez semillas nuevas 3201–3210, red 5×64, 2000 puntos, Adam con coseno, mismo presupuesto por método. El signo de Allen–Cahn, las referencias y el selector Hammersley siguen las definiciones documentadas en el notebook 34. No se reutilizan pesos ni datos de entrenamiento del piloto.

Los resultados están separados en `pca_collocation/results/benchmark/three_pdes_five_selectors_10seeds_100k_v1/{burgers,wave,allen_cahn}/`. Cada carpeta contiene `execution.log`, `status.json`, datos por método/semilla y reportes propios. Las referencias numéricas se preparan una sola vez en `references/`. `aggregate/` contiene los resultados conjuntos. Si la generación del reporte falla después de entrenar, `status.json` puede indicar corridas completas mientras falta `report_complete.json`; consultar el log y regenerar con `--report-only`.

Para retomar un problema, repetir su comando: omite corridas completas y reproduce desde el principio las incompletas. Los estados intermedios guardan Adam para diagnóstico, pero el ejecutor no los restaura automáticamente. Un archivo de estado no prueba que siga activo un proceso tras un cierre abrupto.

Para regenerar un reporte sin entrenar:

```powershell
& $pythonBenchmark scripts/run_collocation_benchmark.py --problem burgers --report-only
```

El notebook 38 debe ejecutarse después de completar los tres problemas. Si faltan archivos, falla sin iniciar entrenamiento. Conserva tablas por problema, ambas políticas de checkpoint y diferencias pareadas por semilla, sin promediar los RMSE de PDE distintas. Los intervalos bootstrap no corrigen comparaciones múltiples.
