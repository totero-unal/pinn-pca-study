# PCA como preprocesamiento para PINNs

Repositorio para evaluar empíricamente si PCA reduce el tiempo de entrenamiento de PINNs conservando precisión de solución y residuo físico en distintos problemas. Se distinguen PCA espacial de snapshots y PCA de coordenadas de entrada.

## Estado global

La [tabla global de benchmarks](BENCHMARKS.md) y el [notebook de síntesis](notebooks/00_benchmark_summary.ipynb) reúnen estados, comparaciones pareadas, fuentes y límites de interpretación. Las métricas se calculan desde artefactos guardados; se conservan todas las variantes de los barridos resumidos y un índice de las corridas anteriores. Las filas comparten semillas y referencias y no se cuentan como experimentos independientes.

**Onda multiescala: cerrada como exploración negativa del protocolo probado**, tras el piloto y la ronda de diagnóstico. El [registro de cierre](reports/wave_multiscale_closure.md) limita las conclusiones a Adam, 2000 pasos, los pesos de pérdida y las representaciones ensayadas.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\build_benchmark_summary.py --apply-closures
```

El comando solo agrega resultados existentes y actualiza documentación; no importa JAX ni ejecuta entrenamiento. Los generadores de reportes de onda conservan el cierre de las corridas registradas.

## Comportamiento al inicio del entrenamiento

El [análisis temprano](reports/early_training.md) y su [notebook](notebooks/01_early_training_comparison.ipynb) comparan validación en pasos 100, 200 y 500, además de la ventana 100–500 y los últimos checkpoints observados antes de presupuestos de tiempo. La comparación principal usa PCA al 99% en diez casos; variantes de k, pesos y escalas se conservan aparte sin contar repetidamente el mismo problema.

En el paso 100, PCA tiene menor RMSE medio en los diez casos, con reducción superior al 5% en ocho. La ventaja es transitoria en varios problemas: en el promedio de la ventana 100–500 hay dos casos con reducción superior al 5%, tres similares y cinco con aumento superior al 5%. Es una descripción de estos datos y de esa banda de similitud; no una prueba de superioridad general. No hay registros comunes de pasos 1–99 ni residuos PDE reservados en todos esos checkpoints.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\analyze_early_training.py
```

El análisis lee trayectorias guardadas y no entrena modelos. Los CSV, huellas de fuentes y cuatro figuras SVG/PNG quedan en `results/early_training/`.

## Primera implementación

La comparación inicial enfrenta una PINN base con una PINN que predice coeficientes temporales sobre una base PCA espacial. Ambas usan los mismos datos, particiones, minibatches, ubicaciones de condición inicial y semillas. Las anchuras se eligieron para que ambas redes tengan cantidades de parámetros cercanas. PCA se ajusta solo con los snapshots de entrenamiento. Para el residuo físico durante el entrenamiento, la base usa autodiferenciación del MLP; PCA usa derivadas espaciales discretas de los modos y autodiferenciación temporal de los coeficientes. Por eso el tiempo compara los pipelines completos y no aísla únicamente el costo de la reducción dimensional.

La corrida piloto configurada usa tres semillas, 1000 pasos y `k=4`. Cada corrida registra RMSE, error relativo L2, residuo RMS de Burgers, tiempo de entrenamiento y tiempo total de la variante PCA. La compilación JAX se ejecuta una vez por arquitectura y se reporta aparte. Las tablas y la gráfica quedan guardadas en `results/burgers/base_vs_pca/pilot_3_seeds_v1/` y también aparecen como outputs del notebook.

## Comparación controlada con autodiferenciación

`configs/burgers_controlled_ad.json` define una segunda comparación, sin reemplazar el piloto. Usa 10 semillas, selección de `k` por 99% de varianza, redes con cantidades de parámetros emparejadas y los mismos lotes supervisados, puntos de colocación, particiones y puntos iniciales por pareja. En ambos modelos se calculan `u_x`, `u_t` y `u_xx` por autodiferenciación de JAX. En PCA, la media y los modos se interpolan con splines cúbicos de Hermite; la derivación de la solución reconstruida pasa por esa interpolación.

La compilación se mide aparte del ajuste. El tiempo de entrenamiento incluye evaluaciones de validación periódicas idénticas; el tiempo del pipeline PCA suma además el ajuste de la base. Se registran trayectorias de RMSE de validación frente al tiempo y tiempos para cruzar umbrales fijados antes de correr (0.30, 0.25 y 0.20). Resultados y curvas quedan en `results/burgers/base_vs_pca/controlled_ad_10_seeds_v1/`.

Ejecutar esta variante con:

```powershell
.\.venv\Scripts\python.exe scripts\run_controlled_experiment.py --config configs\burgers_controlled_ad.json
```

## Presupuesto extendido y barrido de componentes

`configs/burgers_longer_k_sweep.json` continúa con las mismas 10 semillas y extiende el entrenamiento hasta 5000 pasos. Además del PCA que retiene 99% de varianza, compara bases de 4, 8 y 16 componentes, emparejando el número de parámetros con la PINN base. Reporta métricas de validación en los pasos 1000, 2500 y 5000; el conjunto de prueba se usa solo en el último checkpoint. Las curvas, el resumen final y los artefactos se guardan en `results/burgers/base_vs_pca/longer_budget_k_sweep_10_seeds_v1/`.

```powershell
.\.venv\Scripts\python.exe scripts\run_longer_k_sweep.py --config configs\burgers_longer_k_sweep.json
```

## Barrido del peso de la pérdida física

El siguiente experimento fija PCA en `k=8` y compara `lambda_PDE` de 1, 2, 5 y 10 en `L = MSE_data + lambda_PDE * MSE_residual + MSE_initial`. Cada variante PCA usa la misma inicialización, particiones y secuencia de minibatches por semilla. La PINN base con peso físico 1 se conserva como referencia. El presupuesto es de 5000 pasos y los resultados de prueba se calculan solo al final. Los artefactos se guardan en `results/burgers/base_vs_pca/physics_weight_sweep_10_seeds_v1/`.

```powershell
.\.venv\Scripts\python.exe scripts\run_physics_weight_sweep.py --config configs\burgers_physics_weight_sweep.json
```

## Selección del peso con validación y prueba externa

`configs/burgers_validation_selected_lambda.json` define un conjunto externo fijo de 20 tiempos, excluido de las particiones de entrenamiento y validación. Se elige `lambda_PDE` solo con validación: entre los pesos cuya RMSE media no supere en más de 15% la mejor RMSE PCA, se selecciona el de menor residuo PDE medio. Después de fijar el peso, se evalúa una vez el conjunto externo para la variante seleccionada y la PINN base. Los otros pesos no reciben métricas de prueba. Los resultados quedan en `results/burgers/base_vs_pca/validation_selected_lambda_nested_10_seeds_v1/`.

```powershell
.\.venv\Scripts\python.exe scripts\run_validation_selected_sweep.py --config configs\burgers_validation_selected_lambda.json
```

## Benchmark inicial de advección periódica

`configs/advection_base_vs_pca.json` añade el primer problema distinto de Burgers: `u_t + c*u_x = 0` con fronteras periódicas y solución analítica generada con `u0(x) = sin(pi*x) + 0.5*cos(2*pi*x)`. Compara la PINN base contra PCA espacial al 99% de varianza en 10 semillas. Ambos modelos usan los mismos lotes, puntos de colocación y tiempos; autodiferencian el residuo PDE y penalizan la diferencia entre `u(-1,t)` y `u(1,t)`. El archivo de datos analítico se crea en `data/generated/` si todavía no existe. Los resultados se guardan en `results/advection/base_vs_pca/initial_pilot_10_seeds_v1/`.

```powershell
.\.venv\Scripts\python.exe scripts\run_advection_experiment.py --config configs\advection_base_vs_pca.json
```

El siguiente experimento extiende este piloto con un barrido de `k`: compara el `k` que alcanza 99% de varianza y `k=2, 4, 6, 8`, manteniendo semillas, particiones y muestras pareadas. En esta corrida de 10 semillas, el umbral de 99% seleccionó `k=3`; `k=4` obtuvo el menor error de solución (error L2 relativo medio 0.00265 frente a 0.05397 de la PINN base), con residuo PDE medio 0.01596 frente a 0.01196. El resultado sugiere que más varianza retenida no asegura mejor ajuste bajo un presupuesto fijo; es una observación de este benchmark, no una regla general. Las tablas y figuras visibles están en `notebooks/advection/02_pca_k_sweep.ipynb`; los resultados crudos se guardan en `results/advection/base_vs_pca/k_sweep_10_seeds_v1/` y se regeneran con:

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_advection_k_sweep.py --config configs\advection_k_sweep.json
```

## Barrido con una condición inicial periódica más compleja

Para comprobar si el resultado anterior dependía de la condición inicial de dos frecuencias, `configs/advection_fourier_k_sweep.json` define una serie de Fourier de cuatro armónicos. Se repiten las mismas 10 semillas, particiones y presupuesto de 2000 pasos; se comparan la selección al 99% y los `k` fijos disponibles en el barrido. La selección automática eligió `k=4`; entre las variantes PCA, `k=6` tuvo el menor error de solución y `k=8` el menor residuo PDE, aunque ninguna superó a la PINN base en ambas métricas promedio. Todas las variantes PCA fueron más rápidas en esta corrida. La generación analítica acepta coeficientes seno/coseno configurables y la metadata registra la condición inicial usada. Tablas y figuras están en `notebooks/advection/03_fourier_rich_k_sweep.ipynb`; los resultados se guardan en `results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_advection_k_sweep.py --config configs\advection_fourier_k_sweep.json
```

## Presupuesto extendido para la condición Fourier

`configs/advection_fourier_k_sweep_5k.json` repite el mismo barrido con 5000 pasos y checkpoints de validación cada 100. Los primeros 2000 pasos reproducen exactamente las trayectorias de validación del experimento anterior. Al final, `k=8` reduce el error relativo L2 de 0.1089 a 0.0126 y queda por debajo de la PINN base (0.0415); el residuo de la base permanece menor (0.0204 frente a 0.0264). El umbral automático selecciona `k=4` y no muestra una mejora comparable. Las tablas y gráficas están en `notebooks/advection/04_fourier_rich_longer_budget.ipynb`; resultados en `results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_5k_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_advection_k_sweep.py --config configs\advection_fourier_k_sweep_5k.json
```

## Confirmación y cierre del benchmark de advección

`configs/advection_six_harmonic_confirmation.json` evalúa en una tercera condición periódica de seis armónicos la PINN base, `k=8` fijado antes de la corrida y la selección PCA al 99%. Se usan 10 semillas y 5000 pasos; la regla automática eligió `k=6` en todas las semillas. En prueba, `k=8` obtuvo error L2 relativo 0.1790 y residuo PDE 0.4415, por encima de la base (0.0539 y 0.0372); ambas variantes PCA redujeron el tiempo del pipeline y el defecto periódico. La mejora de `k=8` en la condición Fourier anterior no se replicó. Con este resultado se cierra el benchmark de advección, sin seguir ajustando componentes para esta ecuación. El análisis está en `notebooks/advection/05_six_harmonic_confirmation.ipynb`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_advection_k_sweep.py --config configs\advection_six_harmonic_confirmation.json
```

## Benchmark de advección en dimensión alta

`configs/high_dim_advection.json` compara una PINN base con PCA de coordenadas de entrada al 99% para `d=5, 10, 20`, usando diez semillas y 2000 pasos. La solución analítica depende de las dos direcciones principales de una distribución gaussiana con espectro de covarianza decreciente. PCA retuvo `k=3` (aprox. 99,2% de varianza) en las tres dimensiones y la autodiferenciación atraviesa la proyección lineal hasta las coordenadas originales. El residuo PDE promedio fue menor con PCA, pero el error L2 y el tiempo de pipeline no mejoraron consistentemente. Es un caso controlado favorable de baja dimensión intrínseca, y PCA aquí se aplica a las entradas, no a snapshots espaciales.

El notebook está en `notebooks/high_dimensional/01_input_pca_advection.ipynb`; métricas y figuras en `results/high_dimensional_advection/input_pca_vs_base/aligned_low_intrinsic_advection_d5_10_20_10_seeds_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_high_dim_advection.py --config configs\high_dim_advection.json
```

## Alineación entre varianza y relevancia física

`configs/high_dim_alignment_sensitivity_2k_current.json` y `configs/high_dim_alignment_sensitivity_5k.json` comparan una PINN base, PCA al 99% (`k=3`) y PCA con `k=4` en advección de dimensión 20. La solución exacta depende de `q1` y `q4`; `q4` está por debajo del corte de varianza del 99%, así que este control mide empíricamente el efecto de eliminar una dirección de baja varianza pero físicamente activa. Se usan diez semillas, particiones y datos pareados, redes con parámetros emparejados y el mismo código para ambos presupuestos.

Con 2000 pasos, las tres variantes quedan cerca en L2 de prueba (0.522 base, 0.526 PCA al 99%, 0.529 `k=4`). A 5000 pasos, `k=4` logra L2 medio 0.231 ± 0.153 y residuo RMS 0.0232 ± 0.0131; mejora frente a la base en 8/10 y 9/10 semillas respectivamente. PCA al 99% no replica la mejora. El `k=4` no reduce tiempo: el pipeline tarda en promedio 1.17 s más que la base, con alta dispersión entre semillas. El hallazgo respalda conservar la dirección activa de baja varianza en este caso sintético y presupuesto, pero no prueba una mejora general ni una aceleración.

El notebook presenta las métricas pareadas de prueba, las trayectorias de validación y las figuras guardadas. Las corridas crudas quedan en `results/high_dimensional_advection/alignment_sensitivity/low_variance_active_direction_d20_10_seeds_2k_current_v1/` y `results/high_dimensional_advection/alignment_sensitivity/low_variance_active_direction_d20_10_seeds_5k_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_high_dim_alignment_sensitivity.py --config configs\high_dim_alignment_sensitivity_2k_current.json
.\.venv-jax-latest\Scripts\python.exe scripts\run_high_dim_alignment_sensitivity.py --config configs\high_dim_alignment_sensitivity_5k.json
```

## Tiempo hasta una calidad de validación equivalente

`configs/high_dim_time_to_target.json` analiza las trayectorias guardadas de 5000 pasos sin inicializar JAX ni entrenar. El umbral es la media final de validación de la PINN base: RMSE ≤ 0.3685953. PCA con `k=4` lo alcanza en 8/10 semillas, la base en 1/10 y PCA al 99% en 0/10. Las corridas sin cruce se conservan como censuradas. En la única pareja donde ambos alcanzan el umbral, la base tarda 7.645 s y `k=4` tarda 12.066 s; las medianas calculadas solo entre aciertos no permiten concluir una aceleración general.

El análisis adicional limita el tiempo de espera al horizonte observado por ambos modelos dentro de cada semilla. Para `k=4`, la diferencia media PCA − base es −1.337 s, con IC bootstrap 95% [−2.780, 0.264] s; el intervalo incluye cero. Los horizontes varían entre semillas y la regla se eligió retrospectivamente con validación, por lo que este diagnóstico es exploratorio. El objetivo compara RMSE y no verifica el residuo PDE en el instante del cruce. Los tiempos incluyen ajuste PCA y validación periódica; excluyen compilación JAX, generación de lotes y prueba final.

El notebook `notebooks/high_dimensional/03_time_to_target.ipynb` conserva tablas y figuras visibles. Los CSV, el resumen, las huellas de los archivos fuente y las figuras SVG/PNG quedan en `results/high_dimensional_advection/alignment_sensitivity/time_to_target_5k_v1/`. Se requiere Matplotlib (dependencias opcionales de notebooks).

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\analyze_high_dim_time_to_target.py --config configs\high_dim_time_to_target.json
```

El benchmark de dimensión alta se cierra como estudio exploratorio: conservar la dirección física de baja varianza mejora precisión en este caso, mientras que la aceleración a calidad equivalente queda sin confirmar. El siguiente benchmark introduce frecuencias separadas en la solución de una ecuación de segundo orden.

## Piloto de onda multiescala

`configs/wave_multiscale_pilot.json` define `u_tt - 4*u_xx = 0` en `x,t ∈ [0,1]`, con fronteras de desplazamiento cero y velocidad inicial cero. Los casos son `sin(pi*x)*cos(2*pi*t) + 0.5*sin(4*pi*x)*cos(8*pi*t)` y la misma solución más `0.25*sin(9*pi*x)*cos(18*pi*t)`, tomados del apéndice C.3 de Datar et al. (2026), Frozen-PINN. Se adapta el protocolo supervisado del repositorio: no se comparan los tiempos publicados de ese método.

El piloto compara PINN base y PCA espacial de snapshots al 99%, con diez semillas y 2000 pasos por caso. Ambas redes reciben los mismos datos, particiones, condiciones iniciales y puntos de residuo. Se emparejan parámetros dentro de 5%, se usan los mismos pesos de pérdida y se alterna el orden de ejecución. PCA se ajusta solo con tiempos de entrenamiento. JAX autodiferencia las segundas derivadas en ambos modelos; para PCA se preparan splines cúbicos C2 de la media y los modos espaciales.

En esta corrida, PCA eligió `k=2` y `k=3`, con error de proyección de prueba cercano a `2.6e-8`. Aun así, la red reducida se aproxima a la solución nula: sus errores L2 medios son 1.0000 y 1.0001, frente a 0.5378 y 0.5722 de la base. El pipeline PCA es menor (7.19 frente a 8.94 s; 7.95 frente a 9.61 s), pero no constituye una mejora a calidad equivalente. En dos frecuencias, el residuo PCA es menor; en tres, su media aumenta con alta variabilidad. Los errores por armónico y las pérdidas iniciales permiten distinguir capacidad espacial de aprendizaje temporal. Este resultado es un piloto con 2000 pasos y coeficientes PCA sin estandarizar; no establece una limitación general de PCA.

El notebook `notebooks/wave/01_multiscale_wave.ipynb` presenta error de solución, residuo PDE, tiempo de pipeline, curvas de validación y errores por frecuencia. La proyección con coeficientes exactos de prueba aparece solo como diagnóstico de representación. Las imágenes usan la primera semilla preseleccionada. Los resultados se guardan en `results/wave/base_vs_pca/two_and_three_frequencies_10_seeds_2k_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_wave_multiscale.py --config configs\wave_multiscale_pilot.json
.\.venv-jax-latest\Scripts\python.exe scripts\run_wave_multiscale.py --config configs\wave_multiscale_pilot.json --report-only
```

La compilación se hace una vez por arquitectura durante el proceso y se registra aparte. El pipeline incluye ajuste PCA, splines, entrenamiento y validación; excluye preparación de lotes y evaluación final. La opción `--report-only` lee los artefactos sin entrenar ni importar JAX.

## Diagnóstico de la onda y escala de coeficientes

`configs/wave_diagnostic_round.json` conserva el piloto y ejecuta primero un control con coeficientes temporales analíticos de la proyección PCA. Se compara la referencia directa, la base PCA expresada con senos analíticos y los splines C2 usados por la red. Los criterios de continuación, fijados antes del diagnóstico, usan solo validación: error L2 ≤ 1e-4, residuo normalizado ≤ 1e-3 y pérdida oráculo/nula ≤ 0.1. Las 20 combinaciones caso/semilla pasan. El spline no es exacto: su residuo RMS medio de validación es 0.0161 y 0.1881, respectivamente, frente a residuos normalizados de 4.98e-5 y 2.13e-4. Por tanto, queda una limitación numérica que se reporta explícitamente.

La segunda etapa compara base, PCA original y PCA estandarizada con las mismas muestras, semillas, k, arquitectura, pérdida y 2000 pasos. El único cambio de modelo es el decodificador afín de coeficientes, ajustado con entrenamiento; se transforma inversamente la última capa para igualar el campo inicial. No se añade una pérdida de supervisión de coeficientes. Se completan 60 entrenamientos. Los 800 checkpoints de base/PCA original reproducen exactamente el piloto y los archivos fuente conservan sus hashes.

Los errores L2 medios original/estandarizado son 0.999951/0.999885 (dos frecuencias) y 1.000070/1.000039 (tres); estandarizar no corrige el colapso. Las normas de gradiente registradas muestran un fuerte predominio inicial del término PDE dentro de cada parametrización PCA, compatible con desequilibrio de optimización. Las normas dependen de las coordenadas de parámetros y no prueban por sí solas una causa. La conclusión corresponde al protocolo probado; no puede generalizarse a PCA como método de compresión.

El notebook `notebooks/wave/02_diagnostic_round.ipynb` conserva tablas y siete figuras. Resultados, oráculos, escaladores, gradientes, auditoría de reproducción y figuras SVG/PNG quedan en `results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/`.

```powershell
.\.venv-jax-latest\Scripts\python.exe scripts\run_wave_diagnostics.py
.\.venv-jax-latest\Scripts\python.exe scripts\run_wave_diagnostics.py --oracle-only
.\.venv-jax-latest\Scripts\python.exe scripts\run_wave_diagnostics.py --report-only
```

`--report-only` actualiza el notebook sin entrenar ni importar JAX. El pipeline incluye PCA, spline, escalador cuando corresponde, entrenamiento y validación. Excluye controles oráculo, compilación, mediciones de gradiente, generación de lotes y evaluación final.

## Estructura

```text
configs/                  Configuraciones JSON de experimentos
data/raw/                 Datos originales de los benchmarks
notebooks/                Narrativa, ejecución y resultados visibles
results/                  CSV, JSON y figuras compactas por corrida
scripts/                  Entradas de línea de comandos
src/pinn_pca/              Código reutilizable de datos, modelos y métricas
```

## Entorno

Se fija JAX y jaxlib 0.12.0.1 con NumPy 2.5.4 y SciPy 1.18.1. Se recomienda Python 3.14 y CPU para esta primera comparación.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
.\.venv\Scripts\python.exe -m ipykernel install --user --name pinn-pca-study
```

## Ejecutar

Desde la raíz de este repositorio:

```powershell
.\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\burgers_base_vs_pca.json
```

Después, abrir los notebooks de `notebooks/burgers/` para los experimentos de Burgers y los notebooks `notebooks/advection/` para el benchmark y el barrido de componentes PCA. Los notebooks presentan artefactos guardados; los experimentos se regeneran con sus scripts de `scripts/`.

## Política para notebooks y resultados

- Los modelos y cálculos viven en `src/pinn_pca/`; no se duplican entre notebooks.
- Los notebooks conservan outputs de tablas y gráficas para que el análisis sea visible al revisarlos.
- Cada corrida tiene una carpeta con su configuración, métricas por semilla, resumen y figura.
- No se guardan checkpoints ni matrices grandes de predicción por defecto.
- Para nuevas preguntas experimentales, añadir notebooks por benchmark y pregunta, y conservar el código común en `src/`.
