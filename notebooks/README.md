# Notebooks

Los notebooks presentan una pregunta experimental a la vez. Importan la implementación de `src/pinn_pca/`, leen la configuración bajo `configs/` y guardan resultados en `results/`. Se conservan los outputs compactos de tablas y figuras para que el análisis sea visible al abrir el `.ipynb`.

Los experimentos que tarden mucho deben ejecutarse por configuración o desde `scripts/`; el notebook carga después el resultado guardado para análisis y presentación.

La onda multiescala queda cerrada como estudio exploratorio con resultado negativo del protocolo probado. Sus dos notebooks conservan las métricas y añaden el estado, la decisión de cierre y sus límites.

- `00_benchmark_summary.ipynb`: tabla global de estados y comparaciones numéricas desde resultados guardados, con distinción entre exploración, confirmación y diagnóstico; sin entrenamientos nuevos.
- `01_early_training_comparison.ipynb`: comparación retrospectiva en pasos 100, 200 y 500; promedio de RMSE entre 100–500, presupuestos de tiempo, diferencias por semilla y variantes adicionales. Usa validación y no ejecuta entrenamiento.
- `burgers/01_base_vs_pca.ipynb`: piloto exploratorio de tres semillas.
- `burgers/02_controlled_autodiff.ipynb`: comparación pareada de 10 semillas con el mismo residuo autodiferenciado en ambos modelos, curvas de validación contra tiempo y una celda opcional para regenerar la corrida.
- `burgers/03_longer_budget_k_sweep.ipynb`: presupuesto de 5000 pasos, checkpoints de validación y barrido de componentes PCA para medir el compromiso entre error, residuo físico y tiempo.
- `burgers/04_physics_weight_sweep.ipynb`: barrido del peso de la pérdida PDE para PCA con ocho componentes y evaluación final sobre las mismas particiones pareadas.
- `burgers/05_validation_selected_lambda.ipynb`: regla de selección basada solo en validación y una prueba externa fija, evaluada únicamente para el peso elegido y la PINN base.
- `advection/01_base_vs_pca.ipynb`: primer benchmark de advección periódica con referencia analítica, condición de periodicidad y comparación PINN/PCA pareada.
- `advection/02_pca_k_sweep.ipynb`: efecto de retener 2, 3, 4, 6 u 8 modos en el error de solución, residuo PDE, periodicidad y tiempo.
- `advection/03_fourier_rich_k_sweep.ipynb`: repetición del barrido de componentes con condición inicial periódica de cuatro armónicos.
- `advection/04_fourier_rich_longer_budget.ipynb`: mismo problema y semillas, con 5000 pasos y seguimiento de validación para evaluar el efecto del presupuesto de optimización.
- `advection/05_six_harmonic_confirmation.ipynb`: confirmación del candidato `k=8` en una condición inicial periódica no usada en el barrido, frente a PINN base y selección automática al 99%.
- `high_dimensional/01_input_pca_advection.ipynb`: comparación de PINN base y reducción PCA de coordenadas para advección en `d=5, 10, 20`.
- `high_dimensional/02_variance_physics_alignment.ipynb`: control en `d=20` donde la solución depende de una dirección de baja varianza; compara PCA al 99% (`k=3`), PCA `k=4` y la PINN base a 2000 y 5000 pasos.
- `high_dimensional/03_time_to_target.ipynb`: tiempo hasta un RMSE de validación común en las trayectorias guardadas de 5000 pasos, con corridas censuradas, comparación pareada e intervalos bootstrap.
- `wave/01_multiscale_wave.ipynb`: piloto de onda estacionaria con dos y tres frecuencias; métricas, figuras y cierre exploratorio del benchmark.
- `wave/02_diagnostic_round.ipynb`: control físico con coeficientes exactos, ablación de escala y diagnóstico de gradientes; cierre definitivo de esta etapa con resultados negativos y límites de interpretación.
