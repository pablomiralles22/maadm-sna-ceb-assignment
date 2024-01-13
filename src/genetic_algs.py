import random
import statistics
import numpy as np

from copy import deepcopy
from typing import Callable, TypeVar, Generic, Union, Tuple
from nsga_utils import CrowdingDistanceUtils, fast_non_dominated_sort

# ======================================================= #
# ======================== Types ======================== #
# ======================================================= #

Individual = TypeVar("Individual")
Population = list[Individual]

Fitness = TypeVar("Fitness", bound=Tuple[Union[int, float], ...])

FitnessFn = Callable[[Individual], Fitness]
SelectFn = Callable[[list[tuple[int, Fitness, Individual]]], Individual]  # (rank, fitness, individual)
MutateFn = Callable[[Individual, float], Individual]
CrossoverFn = Callable[[Individual, Individual, float], tuple[Individual, Individual]]

# ========================================================== #
# ======================== Main alg ======================== #
# ========================================================== #


class NSGA2(Generic[Individual, Fitness]):
    def __init__(
        self,
        fitness_fn: FitnessFn,
        select_fn: SelectFn,
        mutate_fn: MutateFn,
        crossover_fn: CrossoverFn,
        pmut: float,
        pcross: float = 0.7,
    ):
        self.fitness_fn = fitness_fn
        self.select_fn = select_fn
        self.mutate_fn = mutate_fn
        self.crossover_fn = crossover_fn
        self.pmut = pmut
        self.pcross = pcross

    def run(
        self,
        population: Population,
        ngen: int = 100,
        trace: int = 0,
    ):
        """
        Evoluciona la población. Devuelve la población ordenada por fitness.
        """

        population_with_fitness = self.__calc_fitness(population)
        fronts = fast_non_dominated_sort(population_with_fitness)
        ranked_population = self.__select_next_population(
            fronts,
            population_with_fitness,
            len(population),
        )

        for ind in range(ngen):
            offspring = self.__generate_offspring(ranked_population)
            offspring_with_fitness = self.__calc_fitness(offspring)

            combined_population = population_with_fitness + offspring_with_fitness
            fronts = fast_non_dominated_sort(combined_population)

            ranked_population = self.__select_next_population(
                fronts,
                combined_population,
                len(population),
            )

            # Log
            if trace > 0 and ind % trace == 0:
                self.__log(ind, ranked_population)

        # Log final result
        if trace > 0:
            self.__log(ind, ranked_population)

        return sorted(ranked_population)

    def __calc_fitness(
        self, population: Population
    ) -> list[tuple[Fitness, Individual]]:
        """
        Devuelve una lista de pares de individuos con sus valores fitness.
        """
        fitness_values = map(self.fitness_fn, population)
        return list(zip(fitness_values, population))

    def __generate_offspring(self, ranked_population: list[tuple[int, Fitness, Individual]]) -> Population:
        offspring = []

        while len(offspring) < len(ranked_population):
            parent1 = self.select_fn(ranked_population)
            parent2 = self.select_fn(ranked_population)
            child1, child2 = self.crossover_fn(parent1, parent2, pcross=self.pcross)
            child1 = self.mutate_fn(child1, pmut=self.pmut)
            child2 = self.mutate_fn(child2, pmut=self.pmut)
            offspring.extend([child1, child2])

        return offspring[: len(ranked_population)]

    def __select_next_population(
        self,
        fronts: list[list[int]],
        population_with_fitness: list[tuple[Fitness, Individual]],
        population_size: int,
    ) -> list[tuple[int, Fitness, Individual]]:  # (rank, fitness, individual)
        """
        Devuelve la siguiente población, seleccionando los mejores individuos.
        Los valores de retorno el rango del individuo (número de frente en el
        que aparece), su fitness y el individuo.
        """
        next_population = []
        remaining_size = population_size

        for ind, front in enumerate(fronts):
            if len(front) <= remaining_size:
                next_population.extend(
                    (ind, population_with_fitness[idx][0], population_with_fitness[idx][1])
                    for idx in front
                )
                remaining_size -= len(front)
            else:
                # Crowding distance selection
                fitness_values = [population_with_fitness[idx][0] for idx in front]
                best_idxs = CrowdingDistanceUtils.get_best(
                    fitness_values,
                    remaining_size,
                )
                next_population.extend(
                    (ind, population_with_fitness[idx][0], population_with_fitness[idx][1])
                    for idx in best_idxs
                )
                break

        return next_population

    def __log(
        self,
        generation_ind: int,
        ranked_population: list[tuple[int, Fitness, Individual]],
    ):
        num_objectives = len(ranked_population[0][1])
        std_dev_fitness = [
            statistics.stdev(fitness[idx] for _, fitness, _ in ranked_population)
            for idx in range(num_objectives)
        ]
        print(f"Generation {generation_ind} - {std_dev_fitness=}")

        for obj in range(num_objectives):
            best_idxs = -1
            best_fitness = tuple(-1 for _ in range(num_objectives))
            for idx, (_, fitness, _) in enumerate(ranked_population):
                if fitness[obj] > best_fitness[obj]:
                    best_fitness = fitness
                    best_idxs = idx
            print(f"\t\tBest value for objective {obj}: {ranked_population[best_idxs][1]}")
            


# ============================================================= #
# ======================== Select algs ======================== #
# ============================================================= #


def tournament_select(
    T: int, ranked_population: list[tuple[int, Fitness, Individual]]
) -> Individual:
    """
    Devuelve un individuo seleccionado por torneo, devuelve una copia para evitar efectos laterales.
    Intenta minimizar el rango. Además, si hay empate, intenta maximizar una combinación convexa al
    azar del fitness. Esto intenta diversificar la población, evitando que si muchas soluciones
    están en una parte del frente de pareto, se seleccione siempre la misma parte.
    """
    selected_population = random.sample(ranked_population, T)

    num_objectives = len(selected_population[0][1])
    logits = np.random.randn(num_objectives)
    probs = np.exp(logits) / np.sum(np.exp(logits))

    sort_key = lambda x: (x[0], -sum(x[1][i] * probs[i] for i in range(num_objectives)))

    fittest = min(selected_population, key=sort_key)[2]
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
