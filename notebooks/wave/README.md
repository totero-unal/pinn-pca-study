# Onda multiescala

**Estado: cerrado como estudio exploratorio con resultado negativo para la configuración ensayada.** Fecha: 26 de octubre de 2026. El [registro de cierre](../../reports/wave_multiscale_closure.md) reúne la evidencia y los límites. La [tabla global](../../BENCHMARKS.md) mantiene el estado del conjunto de benchmarks.

`01_multiscale_wave.ipynb` presenta el piloto del problema de onda del apéndice C.3 de Datar et al. (2026), Frozen-PINN: `u_tt - 4*u_xx = 0` en `[0,1]²`, con desplazamiento cero en fronteras y velocidad inicial cero. Incluye dos frecuencias espaciales (1,4) y tres (1,4,9), diez semillas y 2000 pasos por modelo.

Se compara una PINN base con PCA espacial de snapshots de entrenamiento al 99%. La red PCA predice coeficientes temporales; media y modos se interpolan con splines C2 y JAX autodiferencia las segundas derivadas. El protocolo supervisado y la comparación de tiempos pertenecen a este repositorio, no reproducen el entrenamiento de Frozen-PINN.

El notebook conserva tablas y figuras visibles. Los artefactos se guardan en `results/wave/base_vs_pca/two_and_three_frequencies_10_seeds_2k_v1/`.

```powershell
python scripts/run_wave_multiscale.py --config configs/wave_multiscale_pilot.json
python scripts/run_wave_multiscale.py --config configs/wave_multiscale_pilot.json --report-only
```

La segunda opción actualiza tablas y figuras desde resultados existentes sin importar JAX y conserva el cierre documentado de esta corrida.

## Ronda de diagnóstico

`02_diagnostic_round.ipynb` verifica la física con coeficientes exactos de tiempo antes de entrenar y repite base, PCA original y PCA estandarizada bajo el presupuesto del piloto. Los criterios de continuación se fijan en `configs/wave_diagnostic_round.json` y usan solamente validación. El escalador usa snapshots de entrenamiento; una transformación inversa de la última capa conserva el campo físico inicial.

El oráculo pasa las 20 combinaciones caso/semilla, aunque el spline mantiene un residuo absoluto medio de validación de 0.0161 (dos frecuencias) y 0.1881 (tres). Estandarizar no resuelve el fallo: el L2 sigue cercano a uno. La ronda registra gradientes de los cinco términos para estudiar el desequilibrio de la pérdida. Esto limita la conclusión al protocolo probado; no demuestra una imposibilidad general de PCA.

```powershell
python scripts/run_wave_diagnostics.py
python scripts/run_wave_diagnostics.py --oracle-only
python scripts/run_wave_diagnostics.py --report-only
```

La corrida original se conserva. Los nuevos resultados están en `results/wave/diagnostics/diagnostic_standardized_coefficients_10_seeds_2k_v1/`.

## Decisión de cierre

El piloto y el diagnóstico no respaldan una aceleración a calidad equivalente. Estandarizar coeficientes no corrige el fallo de la red reducida; la base también conserva error considerable. El predominio del gradiente PDE es una pista de desequilibrio del entrenamiento y no una causa única demostrada. La compresión mantiene la información de los datos de estos casos, mientras que los splines conservan un error físico no nulo.

Se detiene esta etapa y se conservan datos, configuraciones, corridas y figuras. Las 40 ejecuciones del piloto y las 60 del diagnóstico comparten semillas y contienen repeticiones de controles; no representan 100 réplicas independientes.
