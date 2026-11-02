# PCA para seleccionar puntos de colocación

## Estudio completo preparado: notebooks 35–38

La carpeta [notebooks/benchmark](notebooks/benchmark/README.md) contiene los notebooks de Burgers, onda, Allen–Cahn y agregación, **implementados sin ejecutar**. Se conserva el plan original: Random, Hammersley, Random-R, RAD y PCA + media con QR directo y complemento uniforme. Tres problemas × cinco métodos × diez semillas nuevas (3201–3210) × 100000 pasos: 150 corridas. La configuración común es [benchmark_full.json](configs/benchmark_full.json).

Ejecutar `python -u scripts/run_collocation_benchmark.py --problem burgers`, luego cambiar el problema por `wave` y después `allen_cahn`. Al completar los tres, ejecutar `python scripts/run_collocation_benchmark.py --aggregate`. Los comandos completos para PowerShell están en el README de los notebooks. Los scripts generan versiones `_results.ipynb` con imágenes incrustadas y resultados persistentes en `results/benchmark/three_pdes_five_selectors_10seeds_100k_v1/`. Se pueden retomar corridas completas y regenerar reportes con `--report-only`, sin reentrenarlas. Las corridas incompletas se repiten desde el principio.

Se guardan historias residuales tempranas y tardías, todos los lotes, modelos terminales y de mejor validación, y estados con Adam en 0/25000/50000/75000/100000. El modelo principal se elige por residuo de validación, sin usar prueba. La agregación exige las 150 corridas y no puede iniciar entrenamiento. Mantiene métricas por PDE y contrastes pareados, sin promediar RMSE entre problemas ni declarar un ganador global con intervalos sin ajustar.

## Nuevo piloto preparado: tres PDE y cinco selectores

El [notebook 34 de protocolo](notebooks/burgers/34_protocol_pilot.ipynb) describe Burgers, onda y Allen–Cahn; Random, Hammersley, Random-R, RAD y PCA + media. **El piloto ya finalizó sus quince corridas de 3000 pasos**, con red 5×64 tanh, 2000 puntos, Adam con coseno de horizonte 100000 y una semilla exclusiva de desarrollo. No ejecuta las 150 corridas de confirmación propuestas. Configuración: [benchmark_protocol_pilot.json](configs/benchmark_protocol_pilot.json); lógica compartida: `src/pinn_pca/benchmark_pilot.py`.

Resultados en el [notebook 34 de resultados](notebooks/burgers/34_protocol_pilot_results.ipynb). El error relativo L2 del modelo elegido por validación fue, en Burgers / onda / Allen–Cahn: Random 0.13946 / 0.71599 / 0.42644; Hammersley 0.23816 / 0.62782 / 0.54438; Random-R 0.17724 / 0.74820 / 0.51215; RAD 0.17736 / 0.96830 / 0.45440; PCA + media 0.03020 / 0.73672 / 0.50896. Una sola semilla y tres mil pasos no establecen un ganador. PCA aporta entre tres y siete puntos QR de dos mil (0.15–0.35 %): se debe revisar la dilución antes de congelar el estudio. Las referencias numéricas difieren entre mallas 0.436 % en Burgers y 0.0563 % en Allen–Cahn; onda es analítica. `review_audit.json` verifica quince corridas, treinta y tres lotes guardados, seis reconstrucciones QR, selección del modelo por validación, condiciones duras y dos imágenes incrustadas. No se lanzó entrenamiento adicional durante la revisión.

Desde la raíz, ejecutar en PowerShell:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Users\otero\PINNs-otero-toro\.venv-jax-latest\Scripts\python.exe' -u scripts/run_benchmark_protocol_pilot.py
```

Alternativamente, abrir el notebook 34 con el entorno del proyecto y ejecutar todas sus celdas en orden; no lanzar ambos caminos a la vez. El script guarda el registro de consola y genera `34_protocol_pilot_results.ipynb` con imágenes PNG incrustadas. Los datos quedan en `results/benchmark_pilot/three_pdes_five_selectors_pilot_v1/`: configuración, estado, log, referencias verificadas con dos resoluciones, métricas con esquema común, trazas y archivos por problema/método/semilla. Se guardan los modelos terminal y de mejor residuo de validación; prueba no interviene en la elección. Repetir el comando omite corridas completas y repite las incompletas desde el principio. Añadir `--report-only` regenera el reporte sin entrenar.

Antes del estudio completo se debe revisar la fracción guiada: con veinte snapshots el QR puede aportar como máximo aproximadamente 1 % de los 2000 puntos. No se amplía automáticamente ese presupuesto. Allen–Cahn implementa literalmente el signo de reacción positivo `+5(u³−u)` de la sección 3.3 del artículo, y calcula su propia referencia con ese signo; no sustituye datos del problema de reacción opuesta. El refinamiento numérico exige diferencia relativa entre dos mallas menor a 2 %, una comprobación de consistencia sin garantía rigurosa de error. Si falla, el piloto se detiene antes de entrenar. El piloto no establece un ganador ni autoriza automáticamente la confirmación.

Esta línea estudia si los patrones espaciales y temporales de la historia del residuo ayudan a elegir puntos de colocación para entrenar PINNs. Se mantiene la PINN base y se cambia la selección de puntos.

**Experimento principal actual:** [comparación controlada de bases, notebook 15](notebooks/burgers/15_basis_ablation.ipynb). Cambia solo POD / PCA centrada / PCA + media a dimensión fija. El notebook 14 de QR-DEIM queda como referencia auxiliar separada.

## Primer piloto: Burgers

El [notebook del piloto](notebooks/burgers/06_collocation_selection_pilot.ipynb) compara cinco métodos con una semilla: aleatorio uniforme con remuestreo, Sobol, RAD, POD sin centrar + QR y PCA centrado con la media preservada + QR.

Los métodos comparten un calentamiento uniforme de 100 pasos, inicialización, estado de Adam y muestras supervisadas. Cada uno continúa 300 pasos con 128 puntos físicos, renovados cada 100 pasos. La historia inicial usa 4096 candidatos y 20 snapshots; la selección por QR contribuye hasta 32 índices distintos y el resto del lote se completa con puntos Sobol.

Los artefactos del piloto están en `results/burgers/collocation_selection/residual_history_selection_pilot_1_seed_v1/`. Incluyen el mapa de residuos, espectro singular, modos, puntos elegidos, trayectorias de validación y métricas en un conjunto independiente.

Para reproducirlo desde la raíz del repositorio:

```powershell
python scripts/run_collocation_selection_pilot.py --config pca_collocation/configs/burgers_collocation_selection_pilot.json
```

## Confirmación pareada con 10 semillas

La corrida de confirmación usa diez semillas nuevas (2051–2060), mantiene el protocolo del piloto y fija antes de entrenar un objetivo de validación RMSE ≤ 0.35. Compara calidad en un presupuesto de pasos común y registra el primer checkpoint que alcanza el objetivo. Las corridas sin cruce dentro del presupuesto se marcan como censuradas. La tabla incluye diferencias pareadas contra uniforme; los promedios de tiempo entre aciertos se interpretan junto con la tasa de acierto.

```powershell
python scripts/run_collocation_selection_confirmation.py
```

La configuración está en `pca_collocation/configs/burgers_collocation_selection_confirmation.json`. El notebook `pca_collocation/notebooks/burgers/07_collocation_selection_confirmation.ipynb` y los artefactos quedan en `pca_collocation/results/burgers/collocation_selection/residual_history_selection_confirmation_10_seeds_v1/`.

### Lectura inicial

| Método | RMSE de solución (media ± DE) | RMS del residuo independiente | Semillas que alcanzan RMSE ≤ 0.35 |
|---|---:|---:|---:|
| Uniforme | 0.3354 ± 0.0187 | 0.2158 | 7/10 |
| Sobol | 0.3394 ± 0.0123 | 0.2001 | 7/10 |
| RAD | 0.3676 ± 0.0130 | 0.1983 | 3/10 |
| POD + QR | 0.3431 ± 0.0123 | **0.1931** | 6/10 |
| PCA + media + QR | 0.3432 ± 0.0114 | 0.1933 | 5/10 |

En este presupuesto corto, POD/PCA + QR reducen el residuo independiente frente a uniforme, pero no mejoran el RMSE de la solución ni la tasa de llegada al umbral. RAD también reduce ligeramente el residuo, a costa de un RMSE de solución mayor. Estos datos justifican estudiar el mecanismo, pero no concluir que la selección basada en la historia sea superior. Los promedios de tiempo hasta el umbral se reportan solo entre las semillas que lo alcanzan; las demás permanecen censuradas.

## Proporción de puntos elegidos por los modos

El [notebook de proporciones](notebooks/burgers/08_qr_fraction_ablation.ipynb) compara Sobol puro, QR limitado por rango y 32, 64 o 128 puntos elegidos a partir de los modos, para POD y PCA con media. Reutiliza las diez semillas 2051–2060, el lote total de 128 puntos, los 100 pasos de calentamiento y los 300 de continuación.

En la confirmación anterior, el límite de 32 produjo realmente entre 2 y 4 puntos POD y entre 4 y 5 puntos PCA por actualización. Por eso, aumentar únicamente ese límite no permite estudiar la proporción de puntos. La nueva variante inicializa con QR por rango y añade candidatos distintos maximizando la ganancia log-det de la matriz de información de los modos retenidos. Es una extensión explícita de la selección, con regularización de 1e-10; no equivale a tomar pivotes arbitrarios después del rango de QR.

El resto del lote usa un prefijo Sobol compartido por semilla y actualización. Las variantes comparten pesos y estado del optimizador tras el calentamiento, lotes supervisados, candidatos y puntos de evaluación por semilla. El número de modos continúa determinado por energía de 0.99 y máximo 20; 32/64/128 son cantidades de puntos. Los CSV registran las cantidades reales en cada actualización y diferencias pareadas frente a Sobol y frente al QR por rango correspondiente.

```powershell
python scripts/run_qr_fraction_ablation.py
```

Configuración: [burgers_qr_fraction_ablation.json](configs/burgers_qr_fraction_ablation.json). Resultados: `results/burgers/collocation_selection/qr_fraction_ablation_10_seeds_v1/`. Los tiempos observados incluyen calentamiento, selección y evaluaciones; el orden fijo de ejecución y la compilación inicial limitan su comparación. La métrica principal de esta ronda es la calidad a pasos iguales.

La corrida completó las 90 combinaciones. Sobol obtiene RMSE medio de 0.3397; PCA con 64 puntos obtiene 0.3389, con diferencia pareada de −0.0008 ± 0.0078 (DE) y mejora en 6 de 10 semillas. Esa diferencia pequeña y variable no demuestra superioridad. POD con 32 puntos baja el residuo medio a 0.1879 frente a 0.1995 de Sobol, pero su RMSE sube a 0.3456. Con 128 puntos, el residuo final aumenta a 0.2389 (POD) y 0.2454 (PCA).

Las variantes de 128 puntos alcanzan el umbral en las diez semillas durante el entrenamiento, pero terminan bajo él solo en 5 (POD) y 6 (PCA), frente a 7 con Sobol. El notebook distingue primer cruce y estado final e incluye las curvas de validación. El siguiente experimento debería ampliar el presupuesto manteniendo esta distinción antes de seleccionar una proporción definitiva.

## Presupuesto ampliado y estabilidad

El [notebook de presupuesto ampliado](notebooks/burgers/09_qr_fraction_long_budget.ipynb) conserva las nueve variantes y diez semillas de la comparación de proporciones, y amplía la continuación de 300 a 2000 pasos (2100 pasos totales con calentamiento). Se mantienen el optimizador, la arquitectura, los lotes supervisados, el lote físico de 128 puntos y la actualización cada 100 pasos.

```powershell
python scripts/run_qr_fraction_ablation.py --config pca_collocation/configs/burgers_qr_fraction_long_budget.json
```

Se distingue el primer cruce del umbral RMSE de validación ≤ 0.35, terminar bajo él y permanecer bajo él en los últimos cinco checkpoints observados. El resumen compara los errores de solución, residuo, condición inicial y frontera a presupuesto fijo; conserva diferencias pareadas contra Sobol y contra QR por rango. La comparación con el presupuesto anterior comprueba además que los checkpoints tempranos compartidos coincidan hasta tolerancia de 1e-6 y guarda cambios por semilla en `budget_changes_by_seed.csv`.

Los artefactos están en `results/burgers/collocation_selection/qr_fraction_ablation_10_seeds_2k_v1/`. Se puede regenerar el resumen y las imágenes sin entrenar añadiendo `--summarize-only` al comando. Las figuras quedan incrustadas como PNG en el notebook.

En las 90 combinaciones completadas, los controles tempranos coinciden exactamente con la corrida corta. Sobol obtiene RMSE de prueba de 0.1800 ± 0.0359, frente a 0.2051 ± 0.0451 de PCA con QR por rango (la menor media entre las variantes guiadas). PCA con 64 puntos, que estaba cerca de Sobol a 300 pasos, termina en 0.2298 ± 0.0572. Su ventaja media inicial no se mantiene.

Los residuos físicos cuentan otra parte del resultado: Sobol termina con RMS independiente 0.3664, frente a 0.1236 de POD con 32 puntos y 0.1241 de PCA con 64. Las variantes guiadas reducen el residuo y varias mejoran la frontera, pero empeoran el error de solución y la condición inicial respecto a Sobol. Ninguno de los enfoques domina todas las métricas. Las nueve variantes permanecen bajo el umbral de validación 0.35 en las diez semillas durante los últimos cinco controles, por lo que ese umbral deja de distinguir calidad en esta ronda.

## Diagnóstico del equilibrio de pérdidas y gradientes

El [notebook de diagnóstico](notebooks/burgers/10_residual_balance_diagnostics.ipynb) estudia por qué Sobol aproxima mejor la solución mientras POD/PCA reduce más el residuo. Repite tres variantes (Sobol, POD con 32 puntos y PCA con 64), las mismas diez semillas y los 2000 pasos de continuación. La instrumentación compara las métricas finales con las de la corrida anterior, con tolerancia absoluta de 1e-6.

```powershell
python scripts/run_collocation_diagnostics.py
python scripts/run_collocation_diagnostics.py --report-only
```

La [configuración](configs/burgers_collocation_diagnostics.json) fija de antemano la semilla 2051 para las figuras detalladas y los checkpoints 100, 400, 1100 y 2100. Las tablas y curvas resumen todas las semillas. El reporte incluye:

- Solución de referencia, solución predicha, error absoluto y residuo por autodiferenciación, con escalas comunes entre métodos y momentos; perfiles completos y zoom central.
- Todos los puntos activos por actualización, separados entre selección por modos y complemento Sobol, y medidas de cobertura del dominio.
- Pérdidas de datos, física e IC, normas de sus gradientes y cosenos de alineación. Una sonda física fija común permite comparar los modelos sin cambiar los puntos de medición; otra usa sus puntos activos, conservando idénticos datos e IC de la sonda.
- Errores regionales en tiempos de prueba reservados y monitores independientes de residuo, condición inicial global y frontera.

La inspección del objetivo reveló dos detalles del protocolo que el notebook muestra explícitamente: no hay una penalización de frontera en `baseline_ad_loss`, y la condición inicial se impone en los primeros 32 nodos espaciales (`arange(32)`), no distribuidos por todo el dominio. El diagnóstico conserva ambos detalles para reproducir el resultado observado y distingue la IC entrenada del error IC global. Los gradientes euclídeos medidos no incluyen los momentos ni el precondicionamiento de Adam; su oposición local no demuestra por sí sola una relación causal con el error final.

Los artefactos quedan en `results/burgers/collocation_diagnostics/residual_balance_diagnostics_10_seeds_2k_v1/`. Cada semilla conserva campos, parámetros por capa en los cuatro checkpoints, CSV de puntos, sondas, cobertura, errores regionales y protocolo. `replay_comparison.csv` contrasta la repetición instrumentada con la referencia. `--report-only` regenera figuras y notebook desde esos archivos, sin volver a entrenar. Los mapas incluyen todo el dominio temporal para exploración; las métricas de generalización usan los tiempos de prueba. Los tiempos de ejecución incluyen el diagnóstico y no se usan para comparar velocidad.

Las 30 repeticiones reprodujeron exactamente las métricas finales y los monitores de todos los checkpoints de referencia. En el último lote, Sobol ubica en promedio el 5.9% de puntos en la ventana central tardía, POD el 16.0% y PCA el 21.7%. Sus RMSE de prueba en esa ventana son, respectivamente, 0.3736, 0.4352 y 0.4699: más puntos allí no garantizan una solución más precisa. La distancia de cobertura al percentil 95 también aumenta con los métodos guiados, aunque los tres ocupan todas las celdas de la partición gruesa 8×8.

La oposición entre gradiente físico y gradiente conjunto de datos e IC aparece en el 93.5%, 99.5% y 100% de controles posteriores al calentamiento, promediando por semilla. Está presente en los tres métodos y no basta para atribuir causalidad. Los cocientes de normas finales en la sonda común tampoco indican que la física domine siempre más con POD/PCA: sus medias son 19.20 (Sobol), 2.95 (POD) y 1.00 (PCA), con dispersión entre semillas documentada en el notebook. Las curvas logarítmicas muestran media y percentiles 16–84; el resto de bandas y tablas usa media ± DE.

La supervisión IC explícita solo cubre el intervalo [-1, -0.7569], alrededor del 12.2% del ancho espacial. El siguiente control propuesto es distribuir los mismos 32 nodos IC por todo el dominio, manteniendo los demás parámetros, para aislar el efecto de esa cobertura parcial.

## Control: distribuir los mismos nodos de condición inicial

El [notebook del control IC](notebooks/burgers/11_ic_coverage_control.ipynb) compara los primeros 32 nodos espaciales con 32 índices aproximadamente equiespaciados entre ambos extremos del dominio. Se aplica el cambio desde el calentamiento y se mantienen las diez semillas, los tres selectores del diagnóstico, el lote físico de 128 puntos, los pesos de pérdida 1/1/1 y los 2100 pasos totales. La referencia anterior se conserva y se reutiliza para las comparaciones pareadas.

```powershell
python scripts/run_ic_coverage_control.py
python scripts/run_ic_coverage_control.py --report-only
```

La [configuración](configs/burgers_ic_coverage_control.json) activa `initial_condition_strategy: uniform_domain`; los experimentos anteriores conservan por defecto `left_prefix`. La auditoría verifica las divisiones de datos, las sondas comunes, los datos supervisados congelados, el conjunto candidato, la cantidad de nodos IC y el estado calentado compartido entre métodos dentro del control. El estado calentado puede diferir del de la referencia porque la intervención comienza desde el primer paso. No se añade una penalización separada de frontera.

| Método | RMSE prueba antes → después | RMSE IC global antes → después | Residuo RMS antes → después |
|---|---:|---:|---:|
| Sobol | 0.1800 → 0.1734 | 0.2235 → 0.1525 | 0.3664 → 0.5124 |
| POD 32 | 0.2124 → 0.1864 | 0.2525 → 0.1714 | 0.1236 → 0.2160 |
| PCA 64 | 0.2298 → 0.2051 | 0.2656 → 0.1863 | 0.1241 → 0.2108 |

La IC global mejora en las diez semillas para cada método; el RMSE de solución mejora en 6/10 con Sobol, 8/10 con POD y 7/10 con PCA. Sobol conserva el menor RMSE medio. La brecha media de POD frente a Sobol pasa de 0.0324 a 0.0130 y la de PCA de 0.0498 a 0.0317, con dispersión entre semillas detallada en el notebook. Los residuos físicos aumentan en media en los tres métodos: distribuir IC mejora su cumplimiento global, pero no resuelve simultáneamente todos los términos del problema.

Los resultados viven en `results/burgers/ic_coverage_control/ic_uniform_domain_10_seeds_2k_v1/`. Incluyen cambios pareados por semilla, cambios de brecha frente a Sobol, errores regionales, campos, parámetros, gradientes, selección y figuras antes/después. El notebook distingue las métricas de prueba de las visualizaciones y conserva las imágenes incrustadas como PNG.

## IC distribuida: extensión a 5000 pasos de continuación

El [notebook de presupuesto largo con IC distribuida](notebooks/burgers/12_ic_uniform_long_budget.ipynb) amplía los tres métodos del último control de 2000 a 5000 pasos de continuación, además de los 100 pasos comunes de calentamiento. Mantiene los 32 nodos IC distribuidos, las diez semillas, la arquitectura, Adam, los pesos 1/1/1, el conjunto candidato y el lote físico de 128 puntos, renovado cada 100 pasos.

```powershell
python scripts/run_ic_long_budget.py
python scripts/run_ic_long_budget.py --report-only
```

La [configuración](configs/burgers_ic_long_budget.json) fija antes de ejecutar las evaluaciones de prueba en los pasos totales 2100, 3100, 4100 y 5100 (2000, 3000, 4000 y 5000 pasos de continuación). La validación y los monitores independientes siguen cada 100 pasos. El reporte compara métricas a presupuestos iguales, diferencias pareadas por semilla frente al presupuesto anterior y frente a Sobol, y la evolución de validación durante los últimos 500 pasos. El presupuesto no se detiene ni se reajusta usando la prueba.

La ejecución repite el inicio determinista porque la corrida anterior no conservaba todos los momentos de Adam. Mantiene el estado completo del optimizador durante los 5100 pasos de esta corrida y audita que el prefijo reproduzca las trazas, los puntos activos, la historia inicial del residuo y las métricas del paso 2100. No se reinicia el optimizador en ese punto. Los tiempos incluyen el diagnóstico y no se usan para afirmar aceleración.

Los artefactos están en `results/burgers/ic_long_budget/ic_uniform_domain_10_seeds_5k_v1/`: métricas por semilla y presupuesto, cambios pareados, brechas frente a Sobol, auditoría del prefijo, campos, parámetros, gradientes y selección. El notebook conserva las figuras incrustadas como PNG y sus exportaciones PNG/SVG.

Las 30 corridas completadas reproducen exactamente el prefijo anterior, incluidos sus puntos activos. Al pasar de 2100 a 5100 pasos totales, el RMSE de prueba mejora en las diez semillas de cada método:

| Método | RMSE prueba: 2100 → 5100 | Residuo RMS: 2100 → 5100 | RMSE IC global: 2100 → 5100 |
|---|---:|---:|---:|
| Sobol | 0.1734 → 0.0913 | 0.5124 → 0.3877 | 0.1525 → 0.1113 |
| POD 32 | 0.1864 → 0.1188 | 0.2160 → 0.1335 | 0.1714 → 0.1445 |
| PCA 64 | 0.2051 → 0.1289 | 0.2108 → 0.1373 | 0.1863 → 0.1520 |

Sobol conserva el menor RMSE medio en los cuatro presupuestos fijados. A 5100 pasos, POD obtiene menor RMSE que Sobol en 2/10 semillas y PCA en 1/10; los residuos físicos medios de POD/PCA permanecen bastante menores. El entrenamiento adicional ayuda, pero los métodos guiados no alcanzan a Sobol en precisión media dentro del intervalo evaluado. La validación de PCA cambia poco en promedio en los últimos 500 pasos (−0.0012 ± 0.0125); las curvas y diferencias por semilla permiten valorar esa variabilidad sin seleccionar retrospectivamente con prueba.

## IC distribuida: extensión a 10 000 pasos de continuación

El [notebook de 10 000 pasos](notebooks/burgers/13_ic_uniform_10k.ipynb) amplía los tres selectores desde 5000 a 10 000 pasos de continuación, con 100 pasos adicionales de calentamiento (10 100 totales). Se conservan las diez semillas, los 32 nodos IC distribuidos, los 128 puntos físicos activos, la arquitectura, Adam, la tasa y los pesos de pérdida. Esta ronda estudia si las brechas frente a Sobol se reducen con más entrenamiento, distinguiendo ese comportamiento de una reducción del error de todos los métodos.

```powershell
python scripts/run_ic_long_budget.py --config pca_collocation/configs/burgers_ic_10000_steps.json
python scripts/run_ic_long_budget.py --config pca_collocation/configs/burgers_ic_10000_steps.json --report-only
```

La [configuración](configs/burgers_ic_10000_steps.json) fija evaluaciones de prueba en los pasos totales 5100, 6100, 7100, 8100, 9100 y 10 100. La validación sigue cada 100 pasos. El prefijo hasta 5100 se contrasta con la corrida anterior, incluidos monitores, historia inicial, puntos activos y métricas de prueba en el extremo compartido. Se repite el inicio determinista y se conserva el estado de Adam durante toda la ejecución; no se reinicia en el punto de comparación.

Los resultados se guardan en `results/burgers/ic_long_budget/ic_uniform_domain_10_seeds_10k_v1/`, conservando los experimentos anteriores. El reporte incluye cambios pareados por semilla, brechas absolutas y relativas frente a Sobol y tendencias finales de validación. La brecha relativa es la diferencia de RMSE medios dividida por el RMSE medio de Sobol; no es un promedio de cocientes por semilla. Las curvas finitas no demuestran por sí solas convergencia asintótica. Las imágenes quedan incrustadas como PNG en el notebook.

Las 30 corridas reprodujeron exactamente el prefijo hasta 5100 pasos. Al llegar a 10 100 pasos totales:

| Método | RMSE prueba: 5100 → 10 100 | Residuo RMS final | RMSE IC global final |
|---|---:|---:|---:|
| Sobol | 0.0913 → 0.0572 | 0.3107 | 0.0656 |
| POD 32 | 0.1188 → 0.0780 | 0.1009 | 0.0950 |
| PCA 64 | 0.1289 → 0.0971 | 0.0955 | 0.1233 |

El RMSE de solución mejora respecto a 5100 pasos en 9/10 semillas con Sobol, 10/10 con POD y 9/10 con PCA. Sobol mantiene el menor RMSE medio en los seis presupuestos; cada método guiado lo supera en 1/10 semillas al final. La brecha absoluta de POD baja de 0.0275 a 0.0208, pero su brecha relativa sube de 30.1% a 36.4%. Para PCA, la brecha absoluta pasa de 0.0376 a 0.0399 y la relativa de 41.1% a 69.7%. Se observa mejora del error propio, pero no evidencia suficiente de convergencia asintótica hacia Sobol. Los residuos físicos de POD/PCA siguen siendo menores.

## Alcance y referencia

Los experimentos históricos de los notebooks 01–13 implementan snapshots del residuo, SVD y QR con pivoteo, siguiendo la idea de [QR-DEIM para PINNs](https://arxiv.org/html/2501.07700v4). Mantienen fijo el tamaño del lote activo, pero no incluyen la poda adaptativa del artículo: son variantes híbridas, no una reproducción completa de QR-DEIM. El resultado de una semilla sirve para revisar el procedimiento y las visualizaciones, no para afirmar que un método sea superior.

La confirmación histórica mantiene la evaluación del residuo fuera de los puntos que el selector ve y separa el costo compartido del tiempo por método. Sus resultados no permiten juzgar el algoritmo completo del artículo.

## QR-DEIM de referencia con poda y condiciones duras (comparación auxiliar)

El [notebook 14](notebooks/burgers/14_qr_deim_reference.ipynb) abre una comparación independiente. Implementa el algoritmo 1: historia de residuos con signo en cada paso, SVD sin centrar, rango por suma de valores singulares (no cuadrados), QR con pivoteo sobre los modos transpuestos, poda por grado de convergencia, cancelación de reemplazos cuando corresponde y renovación de candidatos disjuntos. No usa relleno Sobol ni log-det dentro de QR-DEIM.

La transformación `u=t*(x+1)*(x-1)*N(x,t)-sin(pi*x)` impone IC y BC. Ambos métodos entrenan solo con la pérdida del residuo, mismo estado inicial de red y Adam y el mismo conjunto uniforme durante el primer periodo. Después Sobol remuestrea el lote completo y QR-DEIM sustituye únicamente k puntos cuando la poda se acepta. Esta pérdida distinta impide comparar directamente estas cifras con los notebooks anteriores.

- [Piloto](configs/burgers_qr_deim_reference_pilot.json): diez semillas, 10 000 pasos, 128 activos, 256 candidatos, periodos de 100, red 3×32 y epsilon 0.005.
- [Tamaños del artículo](configs/burgers_qr_deim_reference_paper.json): 100 000 pasos, 2000 activos, 1000 candidatos, periodos de 1000, red 5×64 y 10 000 puntos de validación. Configuración disponible; **benchmark completo todavía no ejecutado**.
- [Implementación](../src/pinn_pca/qr_deim_reference.py), [reporte](../src/pinn_pca/qr_deim_reference_report.py) y [pruebas](../tests/test_qr_deim_reference.py).
- Artefactos separados: `results/burgers/qr_deim_reference/pilot_10_seeds_10k_v1/`.

```bash
python scripts/run_qr_deim_reference.py --config pca_collocation/configs/burgers_qr_deim_reference_pilot.json
python scripts/run_qr_deim_reference.py --report-only
```

El mejor checkpoint se elige exclusivamente por residuo en validación independiente, cada periodo. Se informa también el último modelo y se evalúa el error de solución sobre la malla de referencia, nunca usada en el ajuste. La frecuencia de validación y el coseno hasta cero son decisiones explícitas de esta implementación JAX; no se promete reproducción bit a bit de PyTorch/GPU. Los tiempos incluyen compilación y diagnóstico. El piso para la poda es 1e-12; al insertar puntos se alinea su residuo de referencia con sus coordenadas. Se archivan matrices de snapshots, índices, grados y puntos para auditar esa operación. Las figuras del notebook quedan incrustadas como PNG.

El piloto completó las 20 corridas. A 10 000 pasos, el checkpoint elegido por validación tiene RMSE medio ± desviación estándar de **0.1535 ± 0.0464 para QR-DEIM** y **0.1634 ± 0.0552 para Sobol**; los residuos RMS medios independientes son 0.4031 y 0.4196, respectivamente. QR-DEIM gana en RMSE solo en **5/10 semillas**, por lo que la diferencia media no establece una ventaja consistente. Para el último modelo, los RMSE medios son 0.1623 y 0.2165: el criterio de selección del checkpoint cambia la magnitud de la diferencia.

El rango k varió entre 1 y 21 (media 8.443); las 1000 actualizaciones de QR-DEIM aceptaron la poda. La rama de cancelación se verificó con una prueba específica. La auditoría de resultados comprobó los 20 historiales, tamaño fijo, emparejamiento inicial, criterio de validación, condiciones duras y reconstrucción de cuatro actualizaciones desde sus matrices guardadas. El siguiente experimento debe ampliar los tamaños hacia la configuración paper antes de atribuir una ventaja al método publicado.

## Línea principal: ablación de la base POD / PCA / PCA + media

El [notebook 15](notebooks/burgers/15_basis_ablation.ipynb) vuelve a la propuesta propia basada en PCA. El objetivo es aislar la construcción de la base, no reproducir QR-DEIM. Se comparan tres bases de **cuatro direcciones totales**: POD sin centrar; PCA centrada; y media temporal más direcciones PCA independientes hasta completar cuatro. La media ocupa un lugar dentro del presupuesto. El control es Sobol.

Las bases se ortonormalizan antes del mismo QR con pivoteo y ampliación log-det (ridge 1e-10). Cada variante guiada selecciona 32 candidatos y añade el mismo complemento de 96 puntos Sobol; el control usa 128 Sobol. Se conserva la red 3×32, la pérdida histórica de datos + PDE + IC, 32 nodos IC distribuidos, Adam constante, diez semillas, 100 pasos uniformes iniciales y 10 000 pasos de continuación. Candidatos fijos (4096), ventana de 20 snapshots y actualización cada 100 pasos. No hay poda ni transformación de salida. Las tres bases empiezan con la misma historia; después sus historias divergen por el entrenamiento adaptativo.

La dimensión 4 se fija antes de ver los resultados y no se presenta como óptima. El umbral de energía y máximo de modos heredados del runner no se usan para decidir rango aquí. Se registra la dimensión efectiva y las fracciones de fluctuaciones/media capturadas. Si no hay rango suficiente, el experimento falla explícitamente. PCA + media ortogonaliza la media primero y omite modos dependientes; no agrega capacidad extra. Esta convención común de ortonormalización hace comparable el selector, pero no reproduce exactamente los antiguos híbridos de rango variable.

- [Configuración](configs/burgers_basis_ablation.json).
- [Implementación de bases y ejecución](../src/pinn_pca/collocation_basis_ablation.py).
- [Reporte](../src/pinn_pca/collocation_basis_report.py) y [pruebas](../tests/test_collocation_basis_ablation.py).
- Resultados: `results/burgers/basis_ablation/fixed_dimension4_10_seeds_10k_v1/`.

```bash
python scripts/run_collocation_basis_ablation.py
python scripts/run_collocation_basis_ablation.py --report-only
```

Se evalúan modelos terminales en presupuestos prefijados (2100, 5100 y 10100 pasos totales), sin elegir por prueba. Se guardan diferencias pareadas entre todas las bases, métricas de solución/residuo/IC/BC, trazas, historial de puntos y figuras PNG incrustadas. Los tiempos incluyen observación y diagnóstico; no permiten afirmar aceleración. El siguiente cambio a estudiar es dimensión/rango, manteniendo historia, candidatos y actualización sin modificaciones.

Las 40 corridas finalizaron. Resultados medios a 10 000 pasos de continuación:

| Método | RMSE de solución ± DE | Residuo RMS independiente |
|---|---:|---:|
| Sobol | 0.05722 ± 0.01961 | 0.31074 |
| POD, dimensión 4 | 0.11304 ± 0.04236 | 0.12041 |
| PCA centrada, dimensión 4 | 0.09508 ± 0.02143 | 0.13799 |
| PCA + media, dimensión total 4 | 0.08846 ± 0.01428 | 0.12084 |

PCA centrada y PCA + media superan a POD en RMSE en 7/10 semillas cada una. PCA + media supera a PCA centrada solo en 5/10: su menor promedio no demuestra una ventaja consistente de incorporar la media. Frente a Sobol, las victorias en RMSE son 0/10, 0/10 y 1/10, respectivamente; las tres bases mejoran el residuo en 10/10 semillas. El RMSE IC global también es mayor para las bases guiadas (0.1351, 0.1177, 0.1168) que para Sobol (0.0656); esto describe un desequilibrio de métricas, sin demostrar su causa. Las 4000 selecciones conservaron sus presupuestos y la dimensión acordada, y Sobol reprodujo exactamente sus trazas previas en las diez semillas.

## Actualización parcial: PCA + media al 95 %

El [notebook 17](notebooks/burgers/17_partial_update.ipynb) compara la reconstrucción PCA, la poda parcial con propuestas PCA, la misma poda con propuestas Sobol y el remuestreo Sobol completo. Se mantienen diez semillas, 100 pasos uniformes + 10000 de continuación, 128 activos, 32 propuestas, candidatos fijos, ventana de 20 snapshots y actualización cada 100 pasos.

Las variantes parciales comparten el lote inicial de la referencia PCA. La primera poda ocurre después de entrenar 100 pasos con ese lote. Se eliminan los 32 puntos con mayor reducción relativa de residuo; si alguno de esos grados es negativo se cancela toda la sustitución. Los nuevos candidatos son disjuntos del lote activo y el residuo basal se realinea después de cada actualización. PCA se ajusta sobre todos los candidatos y QR/log-det se restringe a los elegibles. La media permanece dentro del selector PCA. Los controles conservan la pérdida histórica, sin imponer condiciones duras.

Configuración: [burgers_partial_update.json](configs/burgers_partial_update.json). Implementación: [collocation_partial.py](../src/pinn_pca/collocation_partial.py). Ejecución desde la raíz: `python scripts/run_collocation_partial.py`; para regenerar, añadir `--report-only`. Los resultados quedan en `results/burgers/partial_update/pca95_partial_10_seeds_10k_v1/`, con trazas, puntos y auditorías reconstruibles. El tamaño 32 sigue siendo fijo; seleccionar directamente k pivotes queda para un control posterior.

Las 40 corridas completaron 10000 pasos de continuación. El RMSE medio de solución fue 0.08320 para reconstrucción PCA, 0.11910 para PCA parcial, 0.06076 para Sobol parcial y 0.05722 para Sobol completo. PCA parcial perdió frente a reconstrucción y frente a Sobol parcial en las diez semillas. Los residuos RMS fueron 0.09586, 0.11090, 0.82891 y 0.31074, respectivamente. Esta configuración de poda no mejoró PCA y no se adopta como nueva referencia. Se reconstruyeron 80 actualizaciones guardadas, se verificó la igualdad del inicio de las variantes parciales y la reconstrucción PCA reprodujo su resultado previo en las diez semillas. Esto evalúa la poda con 32 propuestas y la pérdida histórica; no descarta otras cantidades de reemplazo ni otros protocolos.

## QR directo frente a 32 puntos ampliados

El [notebook 18](notebooks/burgers/18_direct_qr.ipynb) conserva PCA + media al 95 %, la reconstrucción completa, candidatos/historia y entrenamiento. Cambia las 32 propuestas por QR + log-det a k pivotes directos, donde k es la dimensión total independiente, y completa con 128−k Sobol. Cambian conjuntamente número de puntos y ampliación; el control no aísla log-det por sí solo. Las referencias del notebook 17 se reutilizan.

Diez semillas a 10000 pasos de continuación: QR directo obtiene RMSE medio 0.07715 ± 0.02436, frente a 0.08320 ± 0.01151 con 32 propuestas; gana en 6/10 semillas. El residuo RMS pasa de 0.09586 a 0.10882. k varía entre 3 y 16, media 12.515. Sobol completo conserva el menor RMSE (0.05722). La mejora media no establece superioridad consistente.

Ejecutar `python scripts/run_collocation_direct_qr.py`. Resultados en `results/burgers/direct_qr/pca95_direct_qr_10_seeds_10k_v1/`. Se verificaron 1000 conjuntos activos, pivotes iniciales y complementos Sobol. Los tiempos no prueban aceleración: la implementación conserva el cálculo ampliado antes de quedarse con los pivotes directos. Esta prueba solo evalúa calidad y presupuesto de puntos.

## Historia por intervalos y candidatos renovados

El [notebook 19](notebooks/burgers/19_interval_history.ipynb) compara la ventana previa con dos brazos nuevos: veinte snapshots cada cinco pasos del último intervalo de cien pasos, sobre candidatos fijos o renovados. PCA + media al 95 %, QR directo, 128 activos y reconstrucción cada cien pasos permanecen iguales. Los candidatos nuevos se fijan antes de recoger cada historia; las filas nunca mezclan coordenadas. La primera selección comparte el warmup anterior. La comparación entre brazos nuevos aísla la renovación; frente a la referencia cambian conjuntamente frecuencia y duración de la historia. Hay mayor costo de observación, sin afirmación de aceleración.

Diez semillas y 10000 pasos de continuación por brazo nuevo. RMSE medio ± DE: ventana anterior 0.07715 ± 0.02436; intervalo fijo 0.07144 ± 0.00977; intervalo renovado 0.07096 ± 0.01749; Sobol 0.05722 ± 0.01961. El intervalo fijo mejora en 6/10 semillas frente a la referencia; renovar mejora en 5/10 frente a fijo. No hay evidencia consistente de ventaja por renovar candidatos. Los residuos respectivos son 0.10882, 0.16801, 0.16187 y 0.31074.

Configuración: configs/burgers_interval_history.json. Ejecutar `python scripts/run_collocation_intervals.py`. Artefactos en results/burgers/interval_history/interval20_every5_10_seeds_10k_v1/, incluidas historias con signo y coordenadas, tiempos de snapshots, puntos y conteos de rango. Se verificaron cien selecciones QR guardadas y dos mil lotes activos. El próximo control debe separar frecuencia de observación y duración de memoria.

## Separar cadencia y duración de memoria

El [notebook 20](notebooks/burgers/20_memory_cadence.ipynb) compara cadencias de 5, 10 y 20 pasos con horizonte 100, y horizontes 100, 200 y 400 con cadencia 5. Candidatos fijos, PCA + media al 95 %, QR directo y reconstrucción cada 100 pasos. Se ejecutaron 40 corridas nuevas y se reutilizaron diez del caso común (5,100), con 10000 pasos de continuación. La ventana larga se llena gradualmente después del estado inicial compartido; los tiempos de snapshots y el estado de llenado quedan registrados. El control no afirma ahorro de tiempo ni selecciona parámetros usando prueba.

RMSE medio de prueba: (5,100) 0.07144; (10,100) 0.07626; (20,100) 0.06970; (5,200) 0.08881; (5,400) 0.07457. Comparados con (5,100), mejoran en 5/10, 6/10, 4/10 y 7/10 semillas. La memoria 400 tiene más victorias pero peor promedio, ilustrando la importancia de la dispersión. No aparece una mejora monotónica por observar más o recordar más. Rangos medios: 5.305, 4.186, 3.178, 6.245 y 8.280, respectivamente.

Ejecutar `python scripts/run_collocation_memory.py`. Configuración: configs/burgers_memory_cadence.json; combinaciones explícitas en VARIANTS del módulo collocation_memory.py. Resultados en results/burgers/memory_cadence/cadence_memory_10_seeds_10k_v1/. Se reconstruyeron 280 selecciones QR y se verificaron 4000 lotes activos nuevos. La prueba temporal cubre las cinco configuraciones, la cadencia y el llenado gradual. Las figuras están incrustadas como PNG.

## Sustitución parcial de k puntos con PCA y QR directo

El [notebook 21](notebooks/burgers/21_rank_pruning.ipynb) mantiene PCA + media al 95 %, candidatos fijos y 20 snapshots cada 5 pasos del último intervalo de 100. Compara reconstrucción con k QR + Sobol frente a poda de k con propuestas PCA o Sobol. El control Sobol utiliza el calendario k de la corrida PCA por semilla e intervalo; es un control emparejado, no un selector autónomo de rango. Los tres brazos comparten el inicio y el primer intervalo; la primera poda ocurre en paso 200. Se excluyen candidatos activos y se realinea el residuo basal tras insertar puntos. Se registran permanencias completas y sobrevivientes censurados por separado.

Veinte corridas nuevas de 10000 pasos, diez referencias reutilizadas. RMSE medio ± DE: reconstrucción 0.07144 ± 0.00977; PCA parcial 0.09870 ± 0.02202; Sobol parcial 0.05674 ± 0.04708. Residuos RMS: 0.16801, 0.08080 y 1.23932, respectivamente. PCA parcial pierde frente a reconstrucción en 10/10 semillas y frente al control parcial Sobol en 9/10. Ambos brazos aceptan 990/990 podas con k entre 2 y 10, media 6.01. El menor residuo de PCA no se traduce en menor error de solución; este resultado no justifica adoptar la poda como nueva referencia ni atribuir el deterioro a cancelaciones.

Ejecutar `python scripts/run_collocation_rank_pruning.py`. Configuración: configs/burgers_rank_pruning.json. Resultados: results/burgers/rank_pruning/pca95_rank_pruning_10_seeds_10k_v1/. Las pruebas cubren k emparejado variable, realineación basal y cancelación. Se reconstruyeron 80 podas y se verificó igualdad de inicio y calendario k. La prueba conserva la pérdida histórica y no permite descartar otros objetivos o protocolos.

## Renovar candidatos dentro de la poda parcial de k

El [notebook 22](notebooks/burgers/22_refresh_pruning.ipynb) mantiene PCA + media al 95 %, QR directo, veinte snapshots cada cinco pasos del intervalo de cien, 128 activos y poda por convergencia. Renueva los 4096 candidatos antes de recoger cada historia, conservándolos fijos durante sus veinte snapshots. Se ejecutaron diez semillas nuevas a 10000 pasos de continuación; los controles de poda fija y reconstrucción fija/renovada se reutilizaron. Todos comparten lote inicial y primer intervalo.

RMSE medio ± DE: poda renovada 0.11016 ± 0.02592; poda fija 0.09870 ± 0.02202; reconstrucción fija 0.07144 ± 0.00977; reconstrucción renovada 0.07096 ± 0.01749. Renovar mejora la poda solo en 2/10 semillas. Pierde frente a reconstrucción fija en 10/10 y frente a reconstrucción renovada en 9/10. Residuo RMS: 0.09063, 0.08080, 0.16801 y 0.16187, respectivamente. Se aceptaron 990/990 podas nuevas; k varió de 2 a 9, media 5.82. Esta prueba no justifica adoptar la poda ni seguir atribuyendo el deterioro a candidatos fijos.

Ejecutar `python scripts/run_collocation_refresh_pruning.py`. Configuración: configs/burgers_refresh_pruning.json. Resultados: results/burgers/refresh_pruning/pca95_refresh_pruning_10_seeds_10k_v1/. Las auditorías reconstruyen cuarenta podas y propuestas QR elegibles, verifican mil conjuntos activos, inicio común y renovación de coordenadas. Las figuras están incrustadas como PNG. El siguiente diagnóstico aconsejado es el balance entre pérdida física y condiciones iniciales/de frontera, sin modificar simultáneamente el selector.

## Balance de pérdidas: IC y frontera

El [notebook 23](notebooks/burgers/23_loss_balance.ipynb) modifica únicamente el objetivo después del warmup común de cien pasos: datos+PDE+IC (referencia); datos+PDE+10 IC; y datos+PDE+IC+BC. Se comparan poda PCA de k y reconstrucción PCA, con candidatos fijos y veinte snapshots cada cinco pasos. BC usa x=±1, objetivo cero y tiempos supervisados del batch de entrenamiento. Se conservan 32 nodos IC. No se combinan todavía IC10 y BC. El factor diez es un contraste exploratorio, no un óptimo validado.

Cuarenta corridas nuevas de diez mil pasos y veinte referencias reutilizadas. RMSE medio (poda / reconstrucción): baseline 0.09870 / 0.07144; IC10 0.11811 / 0.05941; BC1 0.10155 / 0.08146. IC10 mejora la reconstrucción en 8/10 semillas, pero la poda solo en 3/10. BC1 mejora poda en 6/10 con peor promedio y reconstrucción en 2/10. BC reduce su error específico sin garantizar mejor solución; reforzar IC tampoco rescata por sí solo la poda. La mejora de reconstrucción con IC10 debe confirmarse y compararse contra Sobol entrenado bajo la misma pérdida antes de atribuir ventaja al selector.

Se guardan cuatro componentes de pérdida, normas de gradiente sin ponderar/ponderadas y cosenos PDE–IC/PDE–BC sobre probes comunes en pasos 100, 1100, 5100 y 10100. Son diagnósticos locales, no prueba causal completa. Las referencias reutilizadas no incluyen estos diagnósticos adicionales. Se verificaron warmup común, 160 registros finitos y equivalencia de pérdida/gradientes base con la implementación histórica.

Ejecutar `python scripts/run_collocation_loss_balance.py`. Configuración: configs/burgers_loss_balance.json; los tres objetivos están definidos en OBJECTIVES de collocation_loss_balance.py. Resultados: results/burgers/loss_balance/ic_bc_controls_10_seeds_10k_v1/. El notebook incluye tablas de gradientes y figuras PNG incrustadas.

## IC×10 con Sobol y confirmación en nuevas semillas

El [notebook 24](notebooks/burgers/24_ic_confirmation.ipynb) cruza pérdida original/IC×10 y reconstrucción PCA/Sobol bajo el mismo presupuesto. El peso cambia después del warmup común, sin reiniciar Adam. Se ejecutaron diez controles Sobol IC×10 en semillas de desarrollo y cuarenta corridas en nuevas semillas 2071–2080. Treinta referencias previas se reutilizaron. El peso diez y el protocolo se fijaron antes de ejecutar la confirmación; no se mezclan sus resultados con desarrollo.

En confirmación, RMSE medio de PCA pasa de 0.08236 a 0.06191 con IC×10, mejora en 8/10 semillas; el intervalo bootstrap pareado 95 % para el cambio es [-0.03654,-0.00634]. Sobol pasa de 0.06785 a 0.05982, mejora en 7/10, con intervalo que incluye cero. A igual pérdida IC×10, PCA gana a Sobol en solo 4/10 semillas; diferencia media PCA−Sobol +0.00209, intervalo [-0.02733,+0.02672]. No se confirma superioridad en error de solución. PCA sí obtiene menor residuo independiente en 10/10 semillas, también con IC×10. El bootstrap con diez diferencias tiene incertidumbre considerable y no corrige multiplicidad; no debe interpretarse como prueba de equivalencia ni superioridad universal.

Ejecutar `python scripts/run_collocation_ic_confirmation.py`. Configuración: configs/burgers_ic_confirmation.json. Resultados: results/burgers/ic_confirmation/ic10_sobol_confirmation_10new_v1/. manifest.json identifica las ochenta combinaciones y procedencia; cincuenta ejecuciones son nuevas. Se verificó warmup común, grupos de semillas disjuntos y figuras incrustadas. Las pruebas comprueban signos y dispersión del contraste pareado. La confirmación respalda reforzar IC para PCA, pero todavía exige comparar precisión física/solución y costo antes de adoptar el selector como superior a Sobol.

## Condiciones duras conservando datos supervisados

El [notebook 25](notebooks/burgers/25_hard_conditions.ipynb) aplica u=-sin(pi x)+t(1−x²)N(x,t), con corrección numérica en extremos, desde la inicialización. Las derivadas del residuo incluyen la transformación completa. La pérdida es datos+PDE; se retiran penalizaciones IC/BC, conservando los datos supervisados. Se mantienen PCA + media al 95 %, QR directo, reconstrucción, candidatos fijos y veinte snapshots cada cinco pasos. Sobol usa el mismo objetivo. Los veinte brazos IC×10 de referencia se reutilizan del notebook 24; hay veinte corridas nuevas, semillas 2071–2080 y diez mil pasos de continuación. No se trata de otra confirmación independiente: esas semillas ya se estudiaron.

La transformación cambia el espacio de funciones y todo el warmup; solo los parámetros crudos iniciales coinciden entre condiciones. Dentro de cada tipo, PCA y Sobol comparten warmup. RMSE medio ± DE: PCA IC×10 0.06191 ± 0.01464; Sobol IC×10 0.05982 ± 0.04083; PCA hard 0.09571 ± 0.03993; Sobol hard 0.08287 ± 0.04815. Hard mejora respecto a soft en 3/10 semillas para cada selector. PCA hard supera a Sobol hard solo en 2/10, aunque conserva menor residuo en 10/10. Los residuos hard son 0.28964 y 0.59858. IC global queda alrededor de 6.6e-8 y BC es cero. Cumplir las condiciones exactamente no garantizó menor error a este presupuesto; no se adopta automáticamente esta variante.

Ejecutar `python scripts/run_collocation_hard_conditions.py`. Configuración: configs/burgers_hard_conditions.json. Resultados: results/burgers/hard_conditions/hard_ic_bc_data_10seeds_10k_v1/. Las pruebas verifican IC/BC para parámetros arbitrarios, el residuo contra una solución transformada analítica no trivial y la presencia del término supervisado. Se auditaron dos mil lotes activos, cincuenta selecciones QR y ochenta checkpoints de restricciones. Las figuras están incrustadas como PNG. Se conserva el control IC×10 como referencia mientras se estudian limitaciones de optimización; retirar datos supervisados sería otro experimento.

## Objetivo puramente físico con IC/BC duras

El [notebook 26](notebooks/burgers/26_pde_only.ipynb) retira datos supervisados del objetivo y mantiene la transformación de salida, PCA + media al 95 %, QR directo, candidatos fijos, veinte snapshots cada cinco pasos y reconstrucción de 128 activos. Solo se leen las coordenadas físicas del batch. Se aplica pérdida PDE desde la inicialización, también en el warmup; entre objetivos no se iguala artificialmente el estado después del warmup. La referencia usa datos+PDE del notebook 25. Son veinte corridas nuevas a diez mil pasos de continuación y veinte referencias reutilizadas en semillas 2071–2080; esta extensión es exploratoria, no una confirmación independiente.

Con datos+PDE, RMSE medio PCA/Sobol: 0.09571/0.08287. Con solo PDE: 0.13782/0.25193. PCA mejora frente a Sobol en 7/10 semillas y reduce el residuo independiente en 10/10 (0.28619 frente a 0.82013). La diferencia pareada de RMSE PCA−Sobol es −0.11412 con bootstrap 95 % [−0.18472,−0.04187]. El resultado favorece PCA bajo este objetivo y presupuesto; no demuestra superioridad universal. Ambos métodos tienen peor precisión al retirar datos. IC queda en redondeo numérico y BC es cero.

Ejecutar `python scripts/run_collocation_pde_only.py`. Configuración: configs/burgers_pde_only.json. Resultados: results/burgers/pde_only/pde_only_hard_10seeds_10k_v1/. Se verificó independencia de etiquetas y coordenadas supervisadas mediante envenenamiento con NaN y comparación exacta de pérdida/gradientes. Se reconstruyeron cincuenta selecciones QR y se verificaron dos mil lotes activos y ochenta checkpoints de condiciones analíticas. La solución de referencia solo interviene en evaluación, nunca en el objetivo ni en decisiones de entrenamiento. La poda de k bajo el objetivo físico queda para el control siguiente.

## Poda de k con objetivo solo PDE y condiciones duras

El [notebook 27](notebooks/burgers/27_pde_pruning.ipynb) mantiene pérdida puramente física y transformación de salida desde la inicialización. Compara poda de k con propuestas PCA, poda de k con propuestas Sobol emparejadas y referencias de reconstrucción PCA/Sobol del notebook 26. Se conservan candidatos fijos, veinte snapshots cada cinco pasos, 128 activos y PCA + media al 95 %. Todo el cálculo de historia y convergencia utiliza el residuo transformado. Se ejecutaron veinte corridas nuevas, semillas 2071–2080, y se reutilizaron veinte referencias; diez mil pasos de continuación. Son controles exploratorios, no nuevas semillas de confirmación.

RMSE medio ± DE: PCA parcial 0.17866 ± 0.12380; Sobol parcial 0.22878 ± 0.10981; PCA reconstrucción 0.13782 ± 0.06816; Sobol completo 0.25193 ± 0.11611. Residuos RMS: 0.37763, 1.35070, 0.28619 y 0.82013. PCA parcial mejora frente a reconstrucción solo en 3/10 semillas y frente a Sobol parcial en 6/10; los intervalos bootstrap de diferencias de RMSE incluyen cero. Se aceptaron 990/990 podas por brazo, con k de 2 a 7 y media 4.58. No se justifica adoptar la poda como referencia; el deterioro no proviene de cancelaciones. IC queda en redondeo y BC es cero.

Ejecutar `python scripts/run_collocation_pde_pruning.py`. Configuración: configs/burgers_pde_pruning.json. Resultados: results/burgers/pde_pruning/pde_only_rank_pruning_10seeds_10k_v1/. Se verificaron ochenta podas guardadas, calendario k común, inicio idéntico de los brazos parciales y reconstrucción PCA, dos mil lotes activos y permanencias censuradas. La prueba específica confirma que la poda no utiliza el residuo de N sin transformar. Las figuras están incrustadas como PNG.

## Comparación de bases bajo pérdida solo PDE

El [notebook 28](notebooks/burgers/28_pde_bases.ipynb) compara POD sin centrar, PCA centrada y PCA + media con condiciones duras y pérdida puramente física. Se conserva reconstrucción, QR directo, candidatos fijos y veinte snapshots cada cinco pasos. El calendario de dimensiones de PCA + media del notebook 26 se fija antes de ejecutar POD/PCA centrada; cada base propone igual cantidad de puntos y usa el mismo complemento Sobol. La media cuenta dentro del presupuesto de representación. Es un control condicional emparejado: no evalúa las versiones autónomas de rango adaptativo de POD/PCA centrada.

Ejecutar `python scripts/run_collocation_pde_bases.py`. Configuración: configs/burgers_pde_bases.json. Resultados: results/burgers/pde_bases/matched_pde_bases_10seeds_10k_v1/. Diez semillas 2071–2080 con diez mil pasos de continuación; PCA + media y Sobol se reutilizan del notebook 26. Las semillas ya se estudiaron, por lo que es una ablación exploratoria. Se registra energía, fluctuación y media representada para los brazos nuevos. La selección falla si no existe suficiente rango numérico, sin rellenar con direcciones artificiales. Las figuras se incrustan como PNG.

Las veinte corridas nuevas de la comparación física de bases finalizaron. RMSE medio ± DE: POD 0.11846 ± 0.05689; PCA centrada 0.10543 ± 0.04106; PCA + media 0.13782 ± 0.06816; Sobol 0.25193 ± 0.11611. Residuos RMS: 0.30533, 0.31332, 0.28619 y 0.82013. PCA centrada tiene menor promedio que POD, pero lo supera solo en 4/10 semillas; PCA + media supera a POD en 4/10 y a PCA centrada en 4/10. Todos los intervalos bootstrap de diferencias de RMSE entre bases incluyen cero. Frente a Sobol, las victorias son 9/10, 8/10 y 7/10. Esto apoya la selección mediante bases bajo el protocolo, sin demostrar ventaja específica y consistente del centrado. Se reconstruyeron cien selecciones QR y se verificaron dos mil lotes activos, complemento Sobol común y ochenta comprobaciones IC/BC.

## Confirmación de bases con diez semillas nuevas

El [notebook 29](notebooks/burgers/29_pde_bases_confirmation.ipynb) congela el protocolo del notebook 28 y lo ejecuta en semillas 2091–2100. Las cuarenta corridas son nuevas: PCA + media y Sobol generan referencias internas del mismo estudio; POD y PCA centrada usan el calendario de rangos de esa PCA + media, con regla al 95 % fijada antes de ejecutar. La comparación principal es PCA centrada menos POD en RMSE terminal; los contrastes restantes son secundarios y descriptivos. No se mezclan estas semillas con las de desarrollo ni se aumenta el presupuesto de diez mil pasos.

RMSE medio ± DE: POD 0.12834 ± 0.06442; PCA centrada 0.12256 ± 0.06346; PCA + media 0.14940 ± 0.05770; Sobol 0.24395 ± 0.10885. PCA centrada supera a POD en 5/10 semillas; diferencia media −0.00578, bootstrap pareado 95 % [−0.06129,+0.04410]. No se confirma la ventaja específica del centrado según el criterio fijado; esto no establece equivalencia. POD y PCA centrada mejoran a Sobol en 9/10 semillas y PCA + media en 8/10. Residuos RMS respectivos: 0.27063, 0.31890, 0.30514, 0.91721. La evidencia de mejora mediante bases es más clara que una ventaja específica de PCA frente a POD, bajo este protocolo y calendario condicional de rango.

Ejecutar `python scripts/run_collocation_pde_bases_confirmation.py`. Configuración: configs/burgers_pde_bases_confirmation.json. Resultados: results/burgers/pde_bases/matched_pde_bases_confirmation_10fresh_10k_v1/. primary_confirmation.json guarda criterio y resultado principal. Se verificaron cuatro mil lotes activos, ciento cincuenta selecciones QR guardadas, ciento sesenta controles IC/BC, calendarios y complementos Sobol y warmup compartido. El notebook conserva figuras PNG incrustadas y distingue resultados principales y secundarios.

## Presupuesto ampliado hasta cincuenta mil pasos

El [notebook 30](notebooks/burgers/30_long_confirmation.ipynb) conserva las diez semillas 2091–2100 y el protocolo físico, ampliando a 50000 pasos de continuación con evaluaciones prefijadas en 10000 y 25000. Se regenera el prefijo determinista y se mantienen todos los momentos de Adam durante cada corrida; no se reinicia al evaluar los presupuestos. Se verificaron cuarenta prefijos, incluidos cien lotes iniciales por corrida. La regla PCA + media al 95 % continúa generando rangos emparejados después de 10000. No son nuevas semillas independientes.

RMSE medios en 10000 / 25000 / 50000: POD 0.12834 / 0.07297 / 0.04236; PCA centrada 0.12256 / 0.11653 / 0.05741; PCA + media 0.14940 / 0.09725 / 0.04289; Sobol 0.24395 / 0.19227 / 0.17674. Más entrenamiento mejora el promedio de todos los métodos, sin demostrar ventaja consistente de PCA centrada frente a POD. A 50000, PCA centrada gana en 5/10 semillas frente a POD, con diferencia +0.01505 e intervalo bootstrap [−0.02194,+0.05887]. PCA + media gana en 7/10 pero conserva promedio casi igual a POD, con intervalo que incluye cero. Los residuos finales respectivos son 0.12227, 0.16271, 0.14286 y 0.68756.

Ejecutar `python scripts/run_collocation_long_confirmation.py`; regenerar reporte extendido: `python scripts/report_collocation_long_confirmation.py`. Configuración: configs/burgers_long_confirmation.json. Resultados: results/burgers/pde_bases/matched_pde_bases_10fresh_50k_v1/. budget_summary.csv incluye tiempos observados; budget_paired.json conserva contrastes por presupuesto; terminal_stability.csv describe últimos quinientos pasos de validación. Los tiempos incluyen instrumentación también para Sobol y no prueban aceleración. Las curvas finitas no demuestran convergencia asintótica. Se verificaron veinte mil lotes activos, ciento sesenta métricas por presupuesto y ciento sesenta controles IC/BC. Las figuras se incrustan como PNG.

## Extensión completada hasta cien mil pasos

El [notebook 31](notebooks/burgers/31_budget_100k.ipynb) conserva el protocolo y las mismas diez semillas, con evaluaciones adicionales en 75000 y 100000 pasos. La comparación principal de esta extensión es PCA + media frente a POD a 100000. Se reproducen y verifican las métricas y los 500 lotes del prefijo compartido hasta 50000, manteniendo Adam sin reinicios dentro de cada corrida. POD y PCA centrada conservan el calendario de rangos determinado por PCA + media al 95 % y el complemento Sobol emparejado.

| Método | RMSE medio a 50000 | A 75000 | A 100000 | Mediana a 100000 | Residuo RMS a 100000 |
|---|---:|---:|---:|---:|---:|
| PCA + media | 0.04289 | 0.05466 | 0.02042 | 0.01941 | 0.08311 |
| POD | 0.04236 | 0.03190 | 0.02511 | 0.01925 | 0.07321 |
| PCA centrada | 0.05741 | 0.04085 | 0.05086 | 0.02805 | 0.11137 |
| Sobol | 0.17674 | 0.14542 | 0.14917 | 0.14576 | 0.57059 |

PCA + media obtiene el menor RMSE promedio final (18.7 % menor que POD), pero gana solo en 5/10 semillas. Diferencia pareada media −0.00469, bootstrap 95 % [−0.02167,+0.00856]: no se demuestra ventaja consistente frente a POD. Sus medianas son casi iguales y POD conserva menor residuo físico promedio. La mejora con presupuesto no es monótona: PCA + media empeora a 75000 antes de mejorar a 100000. La cohorte no aporta nuevas semillas independientes y no permite afirmar convergencia asintótica. El siguiente control recomendado es confirmar el contraste PCA + media frente a POD a presupuesto fijo de 100000 con nuevas semillas, antes de volver a aumentar entrenamiento.

Ejecutar o retomar: `python scripts/run_collocation_long_confirmation.py --config pca_collocation/configs/burgers_budget_100k.json`. Omite corridas con `prefix_verified.json` y reproduce desde el inicio las incompletas. Regenerar reporte: `python scripts/report_collocation_long_confirmation.py --config pca_collocation/configs/burgers_budget_100k.json`. Auditar: `python scripts/audit_collocation_extended_budget.py`.

Resultados en [matched_pde_bases_10fresh_100k_v1](results/burgers/pde_bases/matched_pde_bases_10fresh_100k_v1/): `budget_summary.csv`, `individual_pca_mean_vs_pod.csv`, `primary_budget_comparison.json` y `extended_audit.json`. Las cuarenta corridas finalizaron; se verificaron cuarenta mil lotes, 240 filas de métricas, 240 controles IC/BC, cien reconstrucciones QR y tres PNG incrustados. El error máximo de IC es 5.96e−8 y BC es cero. Los tiempos observados incluyen instrumentación compartida y no prueban aceleración entre implementaciones.

## Confirmación preparada: PCA + media frente a POD a cien mil pasos

**Ejecución completada y revisada: 20/20 corridas.** Configuración: [burgers_pca_mean_pod_confirmation_100k.json](configs/burgers_pca_mean_pod_confirmation_100k.json). Veinte corridas: PCA + media y POD en diez semillas nuevas 2111–2120, separadas de las configuraciones anteriores. Se conserva el protocolo físico del notebook 31: cien pasos uniformes y cien mil de continuación, rango PCA + media al 95 %, dimensión total que incluye la media, QR directo, mismo complemento Sobol y 128 puntos por actualización. POD utiliza el calendario de rangos de PCA + media para cada nueva semilla. No se reutilizan resultados de las semillas anteriores.

El [notebook 32](notebooks/burgers/32_pca_mean_pod_confirmation_100k.ipynb) contiene los resultados finales. RMSE medio ± DE: PCA + media 0.03745 ± 0.02789; POD 0.03737 ± 0.03814. Medianas: 0.03059 y 0.02147; residuos RMS medios: 0.09488 y 0.09729. PCA + media gana en 4/10 semillas. La diferencia pareada media es +0.000083, con bootstrap 95 % [−0.02362,+0.02307]: no satisface el criterio de ventaja frente a POD y tampoco demuestra equivalencia. La ventaja promedio observada en la cohorte previa no se confirma con estas nuevas semillas. Las auditorías pasan: veinte mil lotes, cincuenta reconstrucciones QR, rangos y complementos emparejados e inicio común. Se comprobaron adicionalmente los veinte marcadores de finalización, cuarenta restricciones IC/BC y el PNG incrustado del notebook. Antes de ampliar otra vez el presupuesto conviene diagnosticar las diferencias por semilla y la representación de la media, conservando esta cohorte como confirmación ya examinada.

### Diagnóstico posterior por semilla

El [notebook 33](notebooks/burgers/33_pair_seed_diagnostics.ipynb) analiza los archivos existentes sin entrenar. Ejecutar `python scripts/diagnose_collocation_pair_confirmation.py`. Guarda CSV, figuras y `findings.json` en el subdirectorio `diagnostics/` de la confirmación. Se verifican los rangos y complementos Sobol en las 10000 actualizaciones emparejadas y se reconstruyen cien selecciones QR sobre matrices guardadas. Compara ambas bases sobre la misma historia, incluyendo energía de la media, capturas y ángulos entre subespacios. Los gráficos de ubicaciones muestran solo puntos QR, excluyendo el fondo compartido.

La coincidencia exacta QR promedio es 12.96 %, pero ambos métodos concentran aproximadamente 98 % de esos puntos en |x| < 0.1; la distancia simétrica al vecino más cercano es 0.05588 en coordenadas normalizadas por las longitudes de dominio. Los rangos medios por semilla están entre 5.307 y 5.620. En las matrices guardadas la media aporta 94.40 % de la energía y POD captura en promedio 99.99998 % de esa media: conservarla explícitamente no garantiza una ventaja exclusiva de PCA. Estos diagnósticos no establecen la causa del resultado terminal. Las matrices completas llegan solo hasta 9900 pasos de continuación, aunque los puntos y rangos cubren los cien mil. Las correlaciones con diez semillas y los cuatro casos extremos son exploratorios. El siguiente control propuesto es guardar historias tardías y estados de red en presupuestos prefijados; este diagnóstico no ejecuta ese nuevo entrenamiento.

La comparación principal es la diferencia pareada de RMSE terminal PCA + media − POD a 100000; criterio de ventaja: extremo superior del intervalo bootstrap pareado al 95 % menor que cero. Se fijan 20000 remuestreos con semilla 6071. Las medianas y el residuo son descriptivos. Solo se evalúa prueba tras el warmup y al final, sin seleccionar presupuestos por prueba. No se combinan cohortes. Un intervalo que incluya cero no demostraría equivalencia.

Desde la raíz del repositorio, con el entorno Python del proyecto:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:\Users\otero\PINNs-otero-toro\.venv-jax-latest\Scripts\python.exe' -u scripts/run_collocation_pca_mean_pod_confirmation.py
```

Los resultados se guardarán automáticamente en `pca_collocation/results/burgers/pair_confirmation/pca_mean_vs_pod_confirmation_10new_100k_v1/`, dentro de este repositorio:

- `execution.log`: salida de consola persistente, con progreso cada 10000 pasos y errores.
- `status.json`: estado, corrida actual y número de corridas completas; si el proceso termina abruptamente puede conservar el último estado registrado, no constituye una prueba de que siga activo.
- `config.json`: protocolo congelado; cambios requieren otro `run_id`.
- `pca/seed_*/` y `pod_qr/seed_*/`: métricas, trazas, selecciones, historias de auditoría y restricciones por corrida. `completed.json` se escribe únicamente después de verificar la corrida.
- Al terminar: `metrics.csv`, `traces.csv`, `summary.csv`, `individual_pca_mean_vs_pod.csv`, `primary_confirmation.json`, `audit_checks.json` y `comparison.png`.
- Si ocurre un error controlable: `last_error.txt`; para una interrupción abrupta, consultar también `execution.log`.

El mismo comando retoma el estudio omitiendo corridas completas verificadas. Una corrida incompleta se reproduce desde el principio: no hay restauración intermedia de Adam. Las corridas completas conservan sus archivos. Si falla solo el reporte, añadir `--report-only` para regenerarlo sin entrenar; esa opción falla explícitamente si falta alguna corrida.

Al finalizar se generará `pca_collocation/notebooks/burgers/32_pca_mean_pod_confirmation_100k.ipynb` con resultados, protocolo, contraste principal e imagen PNG incrustada. Para revisar los resultados en este chat basta con avisar al terminar: los archivos permanecen accesibles en el espacio de trabajo compartido. No se publican ni se suben a un servidor automáticamente.
