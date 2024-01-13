import random

from copy import deepcopy
from typing import Callable, TypeVar, Union, Tuple

# ======================================================= #
# ======================== Types ======================== #
# ======================================================= #

Individual = TypeVar("Individual")
Fitness = TypeVar("Fitness", bound=Tuple[Union[int, float], ...])
MutateFn = Callable[[Individual, float], Individual]


# ============================================================= #
# ======================== Select algs ======================== #
# ============================================================= #


def tournament_select(
    T: int, ranked_population: list[tuple[int, float, Fitness, Individual]]
) -> Individual:
    """
    Devuelve un individuo seleccionado por torneo, devuelve una copia para evitar efectos laterales.
    Intenta minimizar el rango y maximizar el crowding distance.
    """
    selected_population = random.sample(ranked_population, T)

    sort_key = lambda x: (x[0], -x[1])  # min rank, max crowding distance

    fittest = min(selected_population, key=sort_key)[3]
    return deepcopy(fittest)


# ================================================================ #
# ======================== Crossover algs ======================== #
# ================================================================ #


def random_locus_crossover(
    individual_1: list[int], individual_2: list[int], pcross: float
) -> tuple[list[int], list[int]]:
    """
    Función de crossover para permutaciones. Dada una máscara aleatoria, los
    genes que están a True se mantienen, y los que están a False se rellenan
    con los genes que faltan, en el orden en el que aparecen en el otro
    individuo.

    WARNING: los individuos deben ser listas de enteros, que representen los
    índices de los objetos que permutamos
    """
    new_individual_1 = deepcopy(individual_1)
    new_individual_2 = deepcopy(individual_2)

    if random.random() > pcross:
        return new_individual_1, new_individual_2

    individual_size = len(individual_1)
    mask = random.choices([True, False], k=individual_size)

    for i in range(individual_size):
        if mask[i] is True:
            continue
        new_individual_1[i] = individual_2[i]
        new_individual_2[i] = individual_1[i]

    return new_individual_1, new_individual_2


# ============================================================= #
# ======================== Mutate algs ======================== #
# ============================================================= #


def mutate_locus(
    edges: list[list[int]],
    ratio: float,
    individual: list[int],
    pmut: float,
) -> list[int]:
    """
    Operación de mutación para representación locus.
    Con probabilidad pmut, se cambia ratio*100% de los genes por
    un vecino aleatorio.
    """
    new_individual = deepcopy(individual)

    if random.random() < pmut:
        return new_individual

    genes_to_mutate = random.sample(
        range(len(individual)), int(len(individual) * ratio)
    )
    for gene in genes_to_mutate:
        new_individual[gene] = random.choice(edges[gene])

    return new_individual

def mutate_locus_join(
    edges: list[list[int]],
    ratio: float,
    individual: list[int],
    pmut: float,
) -> list[int]:
    """
    Operación de mutación para representación locus.
    Con probabilidad pmut se aplica esta función. 
    Si se aplica, se cambian los genes que apunten a si mismos, o que apunten a
    un nodo que les apunte de vuelta, con probabilidad ratio.
    """
    new_individual = deepcopy(individual)

    if random.random() < pmut:
        return new_individual

    for gene in range(len(individual)):
        if individual[gene] != gene and individual[individual[gene]] != gene:
            continue
        if random.random() > ratio:
            continue
        new_individual[gene] = random.choice(edges[gene])

    return new_individual

def mutate_locus_separate(
    ratio: float,
    individual: list[int],
    pmut: float,
) -> list[int]:
    """
    Operación de mutación para representación locus.
    Con probabilidad pmut se aplica esta función. 
    Si se aplica, se cambian los genes que no apunten a sí mismos
    para que lo hagan.
    """
    new_individual = deepcopy(individual)

    if random.random() < pmut:
        return new_individual

    for gene in range(len(individual)):
        if individual[gene] == gene:
            continue
        if random.random() > ratio:
            continue
        new_individual[gene] = gene

    return new_individual


def mutate_combine(
    mutate_fns: list[MutateFn], individual: Individual, pmut: float
) -> Individual:
    """
    Operación de mutación, elige aleatoriamente entre las que se pasan como parámetro.
    Para ser empleada, se debe instanciar con functools.partial, e.g.:

    functools.partial(mutate_combine, [mutate_reverse, mutate_insert])
    """
    operator = random.choice(mutate_fns)
    return operator(individual, pmut)


# ============================================================= #
# ======================== Create algs ======================== #
# ============================================================= #


def create_locus(edges: list[list[int]], population_size: int = 100):
    """
    Crea una población a partir de una lista de adyacencia
    en formato lista de listas, para la representación locus.
    """
    return [[random.choice(edge) for edge in edges] for _ in range(population_size)]
