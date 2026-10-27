# PCA frente a PINN base en las primeras iteraciones

Análisis retrospectivo de validación guardada, sin nuevos entrenamientos. La comparación principal usa PCA al 99% en diez casos y no cuenta repetidamente prefijos ni controles reproducidos. Los casos comparten familias y semillas; no son diez réplicas independientes.

## Resultado descriptivo

| Checkpoint o ventana | PCA menor >5% | Similar ±5% | PCA mayor >5% |
| --- | --- | --- | --- |
| 100 | 8 | 2 | 0 |
| 200 | 4 | 3 | 3 |
| 500 | 3 | 2 | 5 |
| window_100_500 | 2 | 3 | 5 |

En el paso 100, los diez casos tienen menor RMSE medio con PCA; ocho reducen más del 5%. La ventaja no se mantiene en todos los problemas. En el promedio aproximado de la ventana 100–500, dos casos reducen más del 5%, tres quedan similares y cinco aumentan más del 5%. La banda de similitud es descriptiva y no representa significación estadística.

| Caso | Paso 100 | Paso 200 | Paso 500 | Ventana 100–500 |
| --- | --- | --- | --- | --- |
| Burgers | 0.534× | 0.605× | 0.686× | 0.617× |
| Advección: 2 frecuencias | 0.797× | 1.562× | 2.476× | 1.652× |
| Advección: 4 armónicos | 0.821× | 1.415× | 1.441× | 1.328× |
| Advección: 6 armónicos | 0.892× | 1.231× | 1.444× | 1.245× |
| Entradas alineadas: d=5 | 0.976× | 0.983× | 0.955× | 0.978× |
| Entradas alineadas: d=10 | 0.967× | 0.983× | 0.928× | 0.959× |
| Entradas alineadas: d=20 | 0.924× | 0.932× | 0.920× | 0.937× |
| Entradas desalineadas: d=20 | 0.934× | 0.993× | 0.998× | 0.986× |
| Onda: 2 frecuencias | 0.947× | 0.934× | 1.419× | 1.071× |
| Onda: 3 frecuencias | 0.947× | 0.933× | 1.373× | 1.057× |

Cada valor es media RMSE PCA / media RMSE base: menor que 1 favorece PCA. La ventana se resume con integración trapezoidal de checkpoints observados, dividida por su longitud.

## Alcance

- Los primeros checkpoints comunes son 100, 200 y 500; no se conoce la comparación común de pasos 1–99 ni del estado inicial.
- Se mide calidad de u con validación; no hay residuos PDE reservados para todos esos checkpoints.
- En PCA espacial la base ya incorpora información de snapshots de entrenamiento. Una buena salida del pipeline no demuestra que el optimizador sea intrínsecamente más rápido.
- PCA de entradas y PCA espacial tienen alcances distintos. Se conservan todos los k, pesos físicos y escalas adicionales en tablas separadas, sin elegir el mejor con estos resultados.
- La comparación temporal usa la última evaluación registrada antes de 0.5, 1 y 2 s, con conteo explícito de pares faltantes; no equivale a reevaluar exactamente en esos tiempos.
- La onda mantiene una predicción próxima a cero mientras la base mejora. La superioridad en un checkpoint aislado no garantiza aprendizaje físico ni calidad final.

## Artefactos

- [Notebook con tablas y figuras](../notebooks/01_early_training_comparison.ipynb)
- [Resumen JSON y huellas de fuentes](../results/early_training/summary.json)
- [Comparación por checkpoint](../results/early_training/checkpoint_summary.csv)
- [Resumen de ventana](../results/early_training/window_auc_summary.csv)
- [Configuración y fuentes canónicas](../configs/early_training_analysis.json)

Reproducir: `python scripts/analyze_early_training.py`. No se importa JAX ni se modifican entrenamientos guardados.
