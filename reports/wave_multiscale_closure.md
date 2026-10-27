## Cierre del benchmark de onda multiescala

**Estado: cerrado como estudio exploratorio con resultado negativo para la configuración evaluada.** Fecha de cierre: 26 de octubre de 2026.

### Alcance y evidencia conservada

Se evaluaron las soluciones de dos y tres frecuencias de `u_tt − 4u_xx = 0`, con diez semillas por caso, Adam, 2000 pasos y cinco términos de pérdida de peso uniforme. PCA espacial al 99% retuvo dos y tres componentes, respectivamente. El piloto ejecutó 40 ajustes y el diagnóstico ejecutó 60; 40 de estos últimos repiten los modelos del piloto. Estos ajustes no constituyen 100 réplicas independientes: se conservaron las mismas diez semillas en cada caso.

La ronda final comparó PINN base, PCA original y PCA con coeficientes estandarizados, manteniendo muestras, particiones, k, arquitectura y pérdida. El escalador se ajustó solo con entrenamiento. La transformación inversa de la última capa conservó el campo físico inicial, con diferencia máxima de 2.98e-8. Los 800 checkpoints de base y PCA original reprodujeron exactamente el piloto.

### Conclusiones respaldadas

- **Representación de los datos:** la proyección PCA en la malla de prueba conserva la solución con error L2 cercano a 2.6e-8. El truncamiento espacial no explica por sí solo el error de la red entrenada en estos casos.
- **Entrenamiento:** el L2 medio final de PCA original/estandarizada es 0.999951/0.999885 con dos frecuencias y 1.000070/1.000039 con tres. Las predicciones siguen próximas a cero. Estandarizar los coeficientes por sí solo no produce una mejora práctica.
- **Referencia base:** sus errores L2 medios son 0.537803 y 0.572163. También conserva un error considerable; este presupuesto no establece una solución de alta precisión para ninguna de las dos arquitecturas.
- **Física de la representación:** los controles con coeficientes exactos pasan los criterios prácticos fijados antes de evaluar validación. El spline conserva residuos RMS medios de validación de 0.0161 y 0.1881; la precisión de los valores de u no se traslada automáticamente a sus derivadas. La reconstrucción oráculo puede alcanzar una pérdida mucho menor que la solución nula.
- **Tiempo:** un menor tiempo de pipeline con error de solución mayor no respalda una aceleración a calidad equivalente. La hipótesis de reducción útil del tiempo no queda respaldada por este benchmark.

### Límites de la interpretación

El predominio inicial del gradiente PDE dentro de cada parametrización PCA es compatible con un desequilibrio de optimización. Las normas dependen de las coordenadas de parámetros y esta observación no demuestra una causa única. La pérdida, el optimizador, el presupuesto, la red temporal y la interpolación siguen formando parte del protocolo evaluado.

El resultado negativo corresponde a esta combinación de PCA y PINN, a estos datos y al presupuesto ensayado. No demuestra una imposibilidad general de PCA, ni que la pérdida de información sea la causa del fallo. Los coeficientes exactos se utilizaron exclusivamente como diagnóstico, no como una red entrenada competidora.

### Decisión de cierre

Se conserva el resultado negativo y se detiene la búsqueda de ajustes dentro de este benchmark. Una investigación futura con nuevas pérdidas, optimizadores o presupuestos debe registrarse como un estudio separado, con criterios fijados antes de evaluar sus resultados. La tabla global conserva el estado y las fuentes de esta conclusión.

Corridas de referencia: `two_and_three_frequencies_10_seeds_2k_v1` y `diagnostic_standardized_coefficients_10_seeds_2k_v1`.
