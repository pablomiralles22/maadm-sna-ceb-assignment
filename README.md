# SNA & CEB - Hito 2

## Instalación del entorno de Python

He preparado un entorno conda que se puede instalar desde el archivo `conda-env.yml`.
```bash
conda env create --file conda-env.yml
```

Después de crear el entorno, puedes utilizar `conda activate sna-ceb-assignment` para activarlo, o ejecutar comandos de Python como `conda run -n sna-ceb-assignment <script>`.

## Estructura del código

### Ejercicio 1
Todo el código para este ejercicio se encuentra en el archivo `notebooks/exercise1.ipynb`, salvo los códigos para crear gráficas, que se encuentran en `src/plots.py`.

### Ejercicio 2
Para este ejercicio, el código está bastante dividido. Primero comienzo explicando los distintos archivos de la carpeta `src`:
* `disjoint_set_union.py`. Implementa esta estructura de datos para convertir un individuo con codificación *locus adjacency* en una lista de comunidades de forma eficiente.
* `genetic_operators.py`. Implementa todas las operación de creación, cruce, mutación y selección de individuos.
* `nsga2_utils.py`. Utilidades para el algoritmo `NSGA-II`, en particular la función `fast_non_dominated_sort` y una clase para calcular el *crowding distance*.
* `nsga2.py`. Implementa el algoritmo.
* `plots.py`. Una serie de funciones para dibujar grafos que reutilizo en varios ejercicios.

Este ejercicio, que utiliza el código anteriormente descrito, se divide a su vez en dos partes.
1. Optimización de hiperparámetros en `optuna`. El código Python se encuentra en `scripts/exercise2_optuna_finetune.py`, y para lanzarlo en multiproceso con todos los parámetros que yo utilizo basta con lanzar el script `sh scripts/run_exercise2_optuna_finetune.sh` (desde el directorio raiz).
2. Obtención de soluciones con los hiperparámetros seleccionados. El código de esto se puede encontrar en `notebooks/exercise2-3.ipynb`, aunque solo hasta la sección "Análisis de resultados".

### Ejercicio 3
El código de esto se puede encontrar en `notebooks/exercise2-3.ipynb` a partir de la sección "Análisis de resultados". 

## Replicación
Para replicar la realización de la práctica basta seguir los siguientes pasos:
1. Ejecutar y leer el *notebook* `notebooks/exercise1.ipynb`. El algoritmo de Leiden en la librería `cdlib` no he podido hacerlo determinista, pero no deberían salir resultados muy distintos.
2. Ejecutar el script de búsqueda de hiperparámetros `sh scripts/run_exercise2_optuna_finetune.sh`. De la carpeta de *logs*, en cualquiera de los archivos generados deberían aparecer las mejores configuraciones, que son a su vez un frente de Pareto con las métricas del hipervolumen y la dispersión de Ziztler.
3. Copiar alguna configuración de las anteriores (normalmente escojo una intermedia en el frente de Pareto), y copiarla en el *notebook* `notebooks/exercise2-3.ipynb`, en la primera celda de código de la sección "Obtención de soluciones". A partir de aquí, ejecutar el código y seguir el *notebook*.