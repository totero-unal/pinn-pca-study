# PINN PCA Study

Los notebooks 35–38 del estudio de colocación están en [pca_collocation/notebooks/benchmark](pca_collocation/notebooks/benchmark/README.md). Para ejecutar Burgers, onda y Allen–Cahn en tres computadoras y reunir las 150 corridas, seguir la [guía distribuida](pca_collocation/notebooks/benchmark/DISTRIBUTED.md).

Este repositorio reúne dos líneas de investigación sobre PINNs. Cada una tiene su propia carpeta con configuraciones, notebooks y resultados.

| Enfoque | Contenido |
|---|---|
| [PCA como preprocesamiento](pca_preprocessing/README.md) | PCA espacial de snapshots y PCA de coordenadas de entrada; benchmarks de Burgers, advección, dimensión alta y onda. |
| [PCA para seleccionar puntos de colocación](pca_collocation/README.md) | Muestreo uniforme, Sobol, RAD, híbridos POD/PCA + QR y referencia QR-DEIM con poda y condiciones duras. |

## Estructura común

- `src/pinn_pca/`: implementación Python compartida y módulos de cada experimento.
- `scripts/`: entradas para ejecutar experimentos y generar reportes desde la raíz del repositorio.
- `data/`: datos de referencia compartidos y datos analíticos generados.
- `tests/`: pruebas del código reutilizable.
- `pca_preprocessing/` y `pca_collocation/`: configuraciones, notebooks, reportes y resultados separados por pregunta experimental.

## Entorno

Desde la raíz del repositorio, crea un entorno e instala el paquete con las dependencias de notebooks:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[notebooks]"
```

Las instrucciones para reproducir cada línea están en sus respectivos README. Los comandos se ejecutan desde esta raíz para que las rutas de datos y resultados permanezcan consistentes.

## Figuras en los notebooks

Las figuras se guardan como salidas PNG incrustadas en los `.ipynb`: pueden verse al abrir el notebook sin ejecutar sus celdas ni resolver enlaces a archivos locales. Las exportaciones SVG se conservan en los resultados para usos vectoriales. Las celdas de figura también permiten volver a cargar los PNG desde el repositorio al ejecutarlas.

Los generadores de reportes usan este formato. Para actualizar las imágenes incrustadas después de cambiar sus archivos, ejecuta `python scripts/repair_notebook_images.py`. La migración inicial de figuras antiguas que solo existían como SVG admite `--node` y `--sharp-module` para indicar un entorno con Sharp instalado; los notebooks y PNG ya migrados no requieren ese conversor.
