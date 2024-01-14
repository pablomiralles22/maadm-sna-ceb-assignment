# SNA & CEB joint assignment

## Installing the Python environment

I have prepared a conda environment that can be installed from the `conda-env.yml` file.
```bash
conda env create --file conda-env.yml
```

After creating the environment, you can use `conda activate sna-ceb-assignment` to activate it, or run python commands as `conda run -n sna-ceb-assignment <script>`.

## Code structure

### Exercise 1
All the code for this exercise is in the file `notebooks/exercise1.ipynb`.

### Exercise 2
The main code 

## Replicating 

## Temp notes
Población inicial con el algoritmo simplón
DSU para locus
Máscara random para cruces, debería ser invariante por permutaciones

Funciones objetivo correlacionadas negativamente

Usar `optuna` para optimizar hiperparámetros.



---
## Discusión

Vamos a intentar dilucidar lo que ocurre. Consideramos para las comunidades originales dos métricas:
* Densidad interna: número de aristas dentro de la comunidad dividido por todas las aristas que podría haber ($n_s \cdot (n_s-1) / 2$).
* Fracción de aristas de nodos de la comunidad que no salen hacia fuera de la comunidad.

Fijado un grafo, estas dos métricas se contraponen: hacer una comunidad más grande favorecerá que todas sus aristas se queden dentro, pero a cambio podría disminuir su densidad si este nodo aporta pocas aristas internas.