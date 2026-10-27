# Tabla global de benchmarks

Fecha de síntesis: **2026-10-26**. Se agregan artefactos existentes; esta operación no entrena modelos.

## Estados y conclusiones

| Benchmark | Estado | Aplicación de PCA | Conclusión limitada | Evidencia |
| --- | --- | --- | --- | --- |
| Burgers 1D | Evaluación externa completada | PCA espacial de snapshots; red de coeficientes temporales | El peso físico seleccionado con validación reduce residuo y tiempo, pero aumenta el error de solución en prueba externa. | [Notebook](notebooks/burgers/05_validation_selected_lambda.ipynb) |
| Advección periódica 1D | Cerrado exploratorio | PCA espacial de snapshots; red de coeficientes temporales | Las mejoras de precisión con ciertos k en condiciones anteriores no se replican en la confirmación de seis armónicos; los pipelines PCA son menores en esa corrida. | [Notebook](notebooks/advection/05_six_harmonic_confirmation.ipynb) |
| Advección en dimensión alta | Cerrado exploratorio | PCA de coordenadas de entrada; proyección lineal fija | Conservar q4 con k=4 mejora precisión en el control desalineado a 5000 pasos, con alta dispersión; la aceleración a calidad equivalente sigue sin confirmarse. | [Notebook](notebooks/high_dimensional/03_time_to_target.ipynb) |
| Onda multiescala | Cerrado exploratorio; resultado negativo del protocolo | PCA espacial de snapshots; coeficientes originales y estandarizados | PCA conserva los datos, pero ambas redes reducidas siguen próximas a la solución nula. Estandarizar no aporta una mejora práctica; no se respalda aceleración a igual calidad. | [Notebook](notebooks/wave/02_diagnostic_round.ipynb) |

## Cómo leer las métricas

- Media ± desviación estándar entre semillas; los conteos comparan PCA con su base dentro de cada semilla.
- Las magnitudes del residuo y los tiempos absolutos no deben ordenarse entre PDE: cambian unidades, operadores y condiciones de ejecución. Comparar dentro de cada fila.
- Burgers/advección 1D puntúan el residuo de prueba por diferencias finitas; dimensión alta/onda usan AD. La física de entrenamiento usa AD en estas fases.
- Los tiempos son de pipelines individuales según cada protocolo, con compilación separada. No incluyen el costo acumulado de explorar k o pesos. En la onda también se excluyen oráculos y mediciones de gradientes.
- Todas las variantes de los barridos resumidos están incluidas. El archivo cuantitativo conserva fases representativas y finales; el índice histórico conserva los estudios anteriores y controles.
- Las filas reutilizan semillas y comparadores; no son experimentos independientes para sumar tamaños de muestra o combinar significación. No se selecciona un ganador global usando prueba.


## Burgers 1D

PCA espacial de snapshots; red de coeficientes temporales. Diferencias finitas en prueba; AD durante entrenamiento.

| Caso y evidencia | Variante | Semillas / pasos | L2 base → PCA | Residuo base → PCA | Pipeline base → PCA (s) | PCA menor: L2 / PDE / tiempo |
| --- | --- | --- | --- | --- | --- | --- |
| [Prueba externa; lambda_PDE elegida con validación](results/burgers/base_vs_pca/validation_selected_lambda_nested_10_seeds_v1/summary.json); Selección con validación y evaluación externa fija | PCA k=8, lambda_PDE=2; k=8 | 10 / 5000 | 0.184 ± 0.023 → 0.3775 ± 0.0097 | 0.8235 ± 0.14 → 0.1979 ± 0.017 | 11.44 ± 0.43 → 7.197 ± 0.25 | 0/10 / 10/10 / 10/10 |

Límite de interpretación: k=8 fijado para esta fase; el tiempo corresponde al ajuste elegido y excluye el costo del barrido de pesos.


## Advección periódica 1D

PCA espacial de snapshots; red de coeficientes temporales. Diferencias finitas en prueba; AD durante entrenamiento.

| Caso y evidencia | Variante | Semillas / pasos | L2 base → PCA | Residuo base → PCA | Pipeline base → PCA (s) | PCA menor: L2 / PDE / tiempo |
| --- | --- | --- | --- | --- | --- | --- |
| [Dos frecuencias; barrido k](results/advection/base_vs_pca/k_sweep_10_seeds_v1/summary.json); Exploración de componentes | PCA AD 99%; k=3 | 10 / 2000 | 0.05397 ± 0.008 → 0.1125 ± 0.018 | 0.01196 ± 0.0024 → 0.4933 ± 0.071 | 3.072 ± 0.066 → 2.614 ± 0.033 | 0/10 / 0/10 / 10/10 |
| [Dos frecuencias; barrido k](results/advection/base_vs_pca/k_sweep_10_seeds_v1/summary.json); Exploración de componentes | PCA AD k=2; k=2 | 10 / 2000 | 0.05397 ± 0.008 → 0.3653 ± 0.023 | 0.01196 ± 0.0024 → 0.9101 ± 0.095 | 3.072 ± 0.066 → 2.605 ± 0.048 | 0/10 / 0/10 / 10/10 |
| [Dos frecuencias; barrido k](results/advection/base_vs_pca/k_sweep_10_seeds_v1/summary.json); Exploración de componentes | PCA AD k=4; k=4 | 10 / 2000 | 0.05397 ± 0.008 → 0.002649 ± 0.00068 | 0.01196 ± 0.0024 → 0.01596 ± 0.0046 | 3.072 ± 0.066 → 2.633 ± 0.071 | 10/10 / 2/10 / 10/10 |
| [Dos frecuencias; barrido k](results/advection/base_vs_pca/k_sweep_10_seeds_v1/summary.json); Exploración de componentes | PCA AD k=6; k=6 | 10 / 2000 | 0.05397 ± 0.008 → 0.4188 ± 0.047 | 0.01196 ± 0.0024 → 0.3202 ± 0.044 | 3.072 ± 0.066 → 2.678 ± 0.058 | 0/10 / 0/10 / 10/10 |
| [Dos frecuencias; barrido k](results/advection/base_vs_pca/k_sweep_10_seeds_v1/summary.json); Exploración de componentes | PCA AD k=8; k=8 | 10 / 2000 | 0.05397 ± 0.008 → 0.776 ± 0.06 | 0.01196 ± 0.0024 → 0.4665 ± 0.062 | 3.072 ± 0.066 → 2.614 ± 0.093 | 0/10 / 0/10 / 10/10 |
| [Cuatro armónicos; presupuesto extendido](results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_5k_v1/summary.json); Exploración de k con 5000 pasos | PCA AD 99%; k=4 | 10 / 5000 | 0.04153 ± 0.0073 → 0.1664 ± 0.02 | 0.02039 ± 0.003 → 0.5797 ± 0.078 | 7.962 ± 0.23 → 6.83 ± 0.3 | 0/10 / 0/10 / 10/10 |
| [Cuatro armónicos; presupuesto extendido](results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_5k_v1/summary.json); Exploración de k con 5000 pasos | PCA AD k=2; k=2 | 10 / 5000 | 0.04153 ± 0.0073 → 0.4818 ± 0.033 | 0.02039 ± 0.003 → 1.007 ± 0.088 | 7.962 ± 0.23 → 6.818 ± 0.25 | 0/10 / 0/10 / 10/10 |
| [Cuatro armónicos; presupuesto extendido](results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_5k_v1/summary.json); Exploración de k con 5000 pasos | PCA AD k=6; k=6 | 10 / 5000 | 0.04153 ± 0.0073 → 0.02493 ± 0.008 | 0.02039 ± 0.003 → 0.1399 ± 0.031 | 7.962 ± 0.23 → 6.879 ± 0.27 | 9/10 / 0/10 / 10/10 |
| [Cuatro armónicos; presupuesto extendido](results/advection/base_vs_pca/fourier_rich_k_sweep_10_seeds_5k_v1/summary.json); Exploración de k con 5000 pasos | PCA AD k=8; k=8 | 10 / 5000 | 0.04153 ± 0.0073 → 0.01256 ± 0.0022 | 0.02039 ± 0.003 → 0.02639 ± 0.0028 | 7.962 ± 0.23 → 6.834 ± 0.33 | 10/10 / 2/10 / 10/10 |
| [Seis armónicos; confirmación](results/advection/base_vs_pca/six_harmonic_confirmation_10_seeds_5k_v1/summary.json); Condición inicial adicional; k=8 predefinido | PCA AD 99%; k=6 | 10 / 5000 | 0.05391 ± 0.0066 → 0.286 ± 0.032 | 0.03716 ± 0.006 → 0.8401 ± 0.1 | 8.243 ± 0.28 → 7.096 ± 0.2 | 0/10 / 0/10 / 10/10 |
| [Seis armónicos; confirmación](results/advection/base_vs_pca/six_harmonic_confirmation_10_seeds_5k_v1/summary.json); Condición inicial adicional; k=8 predefinido | PCA AD k=8; k=8 | 10 / 5000 | 0.05391 ± 0.0066 → 0.179 ± 0.019 | 0.03716 ± 0.006 → 0.4415 ± 0.068 | 8.243 ± 0.28 → 7.132 ± 0.32 | 0/10 / 0/10 / 10/10 |

Límite de interpretación: Los barridos de k son exploratorios. Se muestran todas sus variantes; k=8 se fijó antes de la condición de seis armónicos.


## Advección en dimensión alta

PCA de coordenadas de entrada; proyección lineal fija. Autodiferenciación en prueba y entrenamiento.

| Caso y evidencia | Variante | Semillas / pasos | L2 base → PCA | Residuo base → PCA | Pipeline base → PCA (s) | PCA menor: L2 / PDE / tiempo |
| --- | --- | --- | --- | --- | --- | --- |
| [Alineado; d=5](results/high_dimensional_advection/input_pca_vs_base/aligned_low_intrinsic_advection_d5_10_20_10_seeds_v1/summary.json); Control favorable de dimensión intrínseca baja | PCA input 99%; k=3 | 10 / 2000 | 0.1168 ± 0.048 → 0.1272 ± 0.053 | 0.01567 ± 0.0033 → 0.01402 ± 0.0024 | 2.504 ± 0.024 → 2.461 ± 0.023 | 4/10 / 8/10 / 9/10 |
| [Alineado; d=10](results/high_dimensional_advection/input_pca_vs_base/aligned_low_intrinsic_advection_d5_10_20_10_seeds_v1/summary.json); Control favorable de dimensión intrínseca baja | PCA input 99%; k=3 | 10 / 2000 | 0.128 ± 0.046 → 0.1145 ± 0.043 | 0.01757 ± 0.003 → 0.01324 ± 0.0032 | 2.573 ± 0.042 → 2.602 ± 0.038 | 9/10 / 8/10 / 1/10 |
| [Alineado; d=20](results/high_dimensional_advection/input_pca_vs_base/aligned_low_intrinsic_advection_d5_10_20_10_seeds_v1/summary.json); Control favorable de dimensión intrínseca baja | PCA input 99%; k=3 | 10 / 2000 | 0.1004 ± 0.036 → 0.1069 ± 0.024 | 0.01654 ± 0.0048 → 0.0142 ± 0.0044 | 2.682 ± 0.14 → 2.747 ± 0.12 | 4/10 / 7/10 / 1/10 |
| [Desalineado; d=20; q4 físicamente activa](results/high_dimensional_advection/alignment_sensitivity/low_variance_active_direction_d20_10_seeds_5k_v1/summary.json); Control de dirección activa de baja varianza | PCA input 99%; k=3 | 10 / 5000 | 0.4679 ± 0.11 → 0.5045 ± 0.014 | 0.04354 ± 0.012 → 0.04521 ± 0.0073 | 12 ± 3.3 → 12.66 ± 3.9 | 3/10 / 5/10 / 3/10 |
| [Desalineado; d=20; q4 físicamente activa](results/high_dimensional_advection/alignment_sensitivity/low_variance_active_direction_d20_10_seeds_5k_v1/summary.json); Control de dirección activa de baja varianza | PCA input k=4; k=4 | 10 / 5000 | 0.4679 ± 0.11 → 0.231 ± 0.15 | 0.04354 ± 0.012 → 0.02318 ± 0.013 | 12 ± 3.3 → 13.17 ± 3.8 | 8/10 / 9/10 / 4/10 |

Límite de interpretación: Caso gaussiano sintético de baja dimensión intrínseca. Umbral del análisis temporal elegido retrospectivamente con validación.


## Onda multiescala

PCA espacial de snapshots; coeficientes originales y estandarizados. Autodiferenciación; modos interpolados con splines C2.

| Caso y evidencia | Variante | Semillas / pasos | L2 base → PCA | Residuo base → PCA | Pipeline base → PCA (s) | PCA menor: L2 / PDE / tiempo |
| --- | --- | --- | --- | --- | --- | --- |
| [Dos frecuencias; diagnóstico final](results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/summary.json); Ablación de escala; inicialización física equivalente | PCA espacial 99%; k=2 | 10 / 2000 | 0.5378 ± 0.039 → 1 ± 0.00048 | 0.07007 ± 0.019 → 0.04215 ± 0.018 | 9.126 ± 0.37 → 7.53 ± 0.55 | 0/10 / 9/10 / 10/10 |
| [Dos frecuencias; diagnóstico final](results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/summary.json); Ablación de escala; inicialización física equivalente | PCA estandarizada 99%; k=2 | 10 / 2000 | 0.5378 ± 0.039 → 0.9999 ± 0.00045 | 0.07007 ± 0.019 → 0.04042 ± 0.0099 | 9.126 ± 0.37 → 7.265 ± 0.2 | 0/10 / 10/10 / 10/10 |
| [Tres frecuencias; diagnóstico final](results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/summary.json); Ablación de escala; inicialización física equivalente | PCA espacial 99%; k=3 | 10 / 2000 | 0.5722 ± 0.037 → 1 ± 0.00046 | 0.07731 ± 0.026 → 0.3158 ± 0.44 | 9.593 ± 0.64 → 7.373 ± 0.3 | 0/10 / 5/10 / 10/10 |
| [Tres frecuencias; diagnóstico final](results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/summary.json); Ablación de escala; inicialización física equivalente | PCA estandarizada 99%; k=3 | 10 / 2000 | 0.5722 ± 0.037 → 1 ± 0.00035 | 0.07731 ± 0.026 → 0.278 ± 0.41 | 9.593 ± 0.64 → 7.267 ± 0.13 | 0/10 / 6/10 / 10/10 |

Límite de interpretación: Adam, 2000 pasos y pesos uniformes. La base también conserva error alto. El desequilibrio de gradientes es una pista, no una causa demostrada; los splines mantienen residuo no nulo.


## Interpretación transversal provisional

La evidencia actual no sostiene una aceleración general que conserve simultáneamente error de solución y residuo físico. Las mejoras puntuales dependen de la selección de componentes, el problema y el entrenamiento; una menor duración a pasos fijos con mayor error no establece eficiencia a calidad equivalente. El umbral del 99% no garantizó conservar la dirección física activa en el control desalineado. Una base capaz de reconstruir datos tampoco garantizó aprender la dinámica en la onda. Estos resultados corresponden a los pipelines evaluados y no prueban una imposibilidad general de PCA.

El benchmark de onda está cerrado con resultado negativo del protocolo.

## Artefactos y reproducción

- [Notebook global](notebooks/00_benchmark_summary.ipynb)
- [Análisis de las primeras iteraciones](notebooks/01_early_training_comparison.ipynb)
- [Comparaciones numéricas CSV](results/benchmark_summary/comparisons.csv)
- [Índice de todas las corridas históricas](results/benchmark_summary/archive_index.csv)
- [Estados y reglas de inclusión](reports/benchmark_registry.json)
- [Cierre de la onda](reports/wave_multiscale_closure.md)

Regenerar: `python scripts/build_benchmark_summary.py`. Para aplicar los cierres registrados a los notebooks existentes, añadir `--apply-closures`. Este script usa solo la biblioteca estándar y no importa JAX. Los generadores de reportes de onda conservan el cierre al regenerar las corridas registradas.
