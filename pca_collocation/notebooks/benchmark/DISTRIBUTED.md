# Ejecutar en tres computadoras y reunir los resultados

Usar la misma revisión de Git en las tres computadoras y en la que reúne los datos. No iniciar entrenamiento desde notebooks y scripts a la vez. Estos pasos no necesitan escribir en GitHub desde los equipos de cálculo.

### Corrección de bloqueos de progreso en Windows/OneDrive

El escritor JSON reintenta reemplazos bloqueados, utiliza temporales únicos y conserva intacto el archivo anterior si falla. `status.json` es informativo: si sigue bloqueado, se emite un aviso y continúa el entrenamiento. Configuraciones y marcadores de finalización siguen siendo obligatorios; un fallo persistente al guardar esos datos sí se informa como error. El manejo de errores conserva la excepción original aunque tampoco pueda escribir el estado.

La corrección cambia solo escritura de progreso y manejo de errores, sin modificar pérdida, selección, Adam ni semillas. Se permite explícitamente compatibilidad entre las huellas de código originales y esta corrección revisada. Se conserva la huella original en `environment.json` y se registran las revisiones compatibles utilizadas al retomar. Cambios de versiones de bibliotecas o de código ajenos a esa pareja se siguen rechazando. Así pueden reunirse resultados de equipos que terminaron con la versión original y equipos con esta corrección, usando el importador actualizado.

Si un estudio se detuvo por `PermissionError` en `status.tmp -> status.json`, repetir su comando tras actualizar el código: conserva las corridas completas y reproduce la incompleta desde el principio. No restaura automáticamente el checkpoint intermedio. Un equipo que sigue entrenando puede terminar con su código original; no necesita actualizar durante su ejecución.

## 1. Preparar cada computadora

Instalar Git y Python 3.12 o posterior compatible con las versiones fijadas. En terminal:

```text
git clone https://github.com/totero-unal/pinn-pca-study.git
cd pinn-pca-study
python -m venv .venv
```

Activar el entorno en Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

O en Linux/macOS:

```bash
source .venv/bin/activate
```

Instalar las dependencias y conservar la revisión utilizada:

```text
python -m pip install -e ".[notebooks]"
git rev-parse HEAD
```

El proyecto usa JAX en CPU por defecto. Las instrucciones anteriores no configuran GPU. Los equipos pueden tener tiempos muy distintos; la comparación entre selectores debe interpretarse con esa limitación. El ejecutor registra plataforma, Python, versiones de bibliotecas y huella del código en `environment.json`; no permite cambiar el entorno dentro de un mismo problema ya iniciado.

## 2. Repartir un problema por computadora

Computadora A:

```text
python -u scripts/run_collocation_benchmark.py --problem burgers
```

Computadora B:

```text
python -u scripts/run_collocation_benchmark.py --problem wave
```

Computadora C:

```text
python -u scripts/run_collocation_benchmark.py --problem allen_cahn
```

Cada equipo ejecuta cincuenta corridas de cien mil pasos. El script fija los hilos BLAS por defecto y conserva logs, estados y resultados. Se puede repetir el mismo comando para omitir corridas completas; una corrida incompleta se reproduce desde el principio. No usar una carpeta de resultados compartida/sincronizada entre los equipos mientras entrenan: cada clon trabaja en su disco local. Las referencias se preparan localmente y cada paquete transporta la del problema correspondiente.

## 3. Exportar cuando termine cada problema

En cada equipo, usar el nombre correspondiente:

```text
python scripts/exchange_benchmark_results.py export --problem burgers
python scripts/exchange_benchmark_results.py export --problem wave
python scripts/exchange_benchmark_results.py export --problem allen_cahn
```

Cada comando se ejecuta solo en el equipo que completó ese problema. Genera `benchmark_artifacts/<run_id>_<problema>.zip` con las cincuenta corridas, modelos, historias, referencia, configuración y un manifiesto SHA-256. Exige el reporte completo y el código original del entrenamiento. Los ZIP y los resultados crudos se excluyen de Git para evitar conflictos y archivos grandes. No actualizar ni modificar el código a mitad de un estudio. Se puede copiar cada ZIP mediante USB, OneDrive u otro medio; no es necesario hacer commits de resultados.

## 4. Importar los tres ZIP en un mismo clon

En la computadora que reunirá los resultados, usar la misma revisión de código. Copiar los tres ZIP a `benchmark_artifacts/` y ejecutar (sustituir nombres si se usó `--output` al exportar):

```text
python scripts/exchange_benchmark_results.py import benchmark_artifacts/three_pdes_five_selectors_10seeds_100k_v1_burgers.zip
python scripts/exchange_benchmark_results.py import benchmark_artifacts/three_pdes_five_selectors_10seeds_100k_v1_wave.zip
python scripts/exchange_benchmark_results.py import benchmark_artifacts/three_pdes_five_selectors_10seeds_100k_v1_allen_cahn.zip
python scripts/run_collocation_benchmark.py --aggregate
```

La importación verifica configuración, huella de código, rutas y hashes antes de escribir. Conserva archivos existentes idénticos y rechaza conflictos; no sobrescribe resultados distintos. No ejecuta redes. Después, el agregador audita las 150 corridas y crea el notebook `38_benchmark_aggregate_results.ipynb` con tablas y figuras incrustadas, junto con los CSV de `aggregate/`.

Para regenerar también los reportes individuales en el equipo receptor, ejecutar el comando de cada problema con `--report-only`. No entrena. Los notebooks 35–38 originales permanecen como interfaces ejecutables; las versiones `_results.ipynb` contienen los reportes generados.
