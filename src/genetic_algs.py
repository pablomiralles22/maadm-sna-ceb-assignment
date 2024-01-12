import random
import statistics

from copy import deepcopy
from typing import Callable, TypeVar, Generic, Union, Tuple
from fitness_utils import CrowdingDistanceUtils, fast_non_dominated_sort

# ======================================================= #
# ======================== Types ======================== #
# ======================================================= #

Individual = TypeVar("Individual")
Population = list[Individual]

Fitness = TypeVar("Fitness", bound=Tuple[Union[int, float], ...])

FitnessFn = Callable[[Individual], Fitness]
SelectFn = Callable[[list[tuple[Fitness, Individual]], int], Individual]
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

        for ind in range(ngen):
            offspring = self.__generate_offspring(population)
            offspring_with_fitness = self.__calc_fitness(offspring)

            combined_population = population_with_fitness + offspring_with_fitness
            fronts = fast_non_dominated_sort(combined_population)

            population_with_fitness = self.__select_next_population(
                fronts,
                combined_population,
                len(population),
            )

            population = [individual for _, individual in population_with_fitness]

            # Log
            if trace > 0 and ind % trace == 0:
                self.__log(ind, population_with_fitness)

        # Log final result
        if trace > 0:
            self.__log(ind, population_with_fitness)

        return sorted(population_with_fitness, reverse=True)

    def __calc_fitness(
        self, population: Population
    ) -> list[tuple[Fitness, Individual]]:
        """
        Devuelve una lista de pares de individuos con sus valores fitness.
        """
        fitness_values = map(self.fitness_fn, population)
        return list(zip(fitness_values, population))

    def __generate_offspring(self, population: Population) -> Population:
        offspring = []

        while len(offspring) < len(population):
            parent1, parent2 = random.sample(population, 2)
            child1, child2 = self.crossover_fn(parent1, parent2, pcross=self.pcross)
            child1 = self.mutate_fn(child1, pmut=self.pmut)
            child2 = self.mutate_fn(child2, pmut=self.pmut)
            offspring.extend([child1, child2])

        return offspring[: len(population)]

    def __select_next_population(
        self,
        fronts: list[list[int]],
        population_with_fitness: list[tuple[Fitness, Individual]],
        population_size: int,
    ) -> list[tuple[Fitness, Individual]]:
        next_population = []
        remaining_size = population_size

        for front in fronts:
            if len(front) <= remaining_size:
                next_population.extend(population_with_fitness[idx] for idx in front)
                remaining_size -= len(front)
            else:
                # Crowding distance selection
                fitness_values = [population_with_fitness[idx][0] for idx in front]
                best_idxs = CrowdingDistanceUtils.get_best(
                    fitness_values,
                    remaining_size,
                )
                next_population.extend(population_with_fitness[idx] for idx in best_idxs)
                break

        return next_population

    def __log(
        self,
        generation_ind: int,
        population_with_fitness: list[tuple[Fitness, Individual]],
    ):
        num_objectives = len(population_with_fitness[0][0])
        std_dev_fitness = [
            statistics.stdev(fitness[idx] for fitness, _ in population_with_fitness)
            for idx in range(num_objectives)
        ]
        print(f"Generation {generation_ind} - {std_dev_fitness=}")

        for obj in range(num_objectives):
            best_idxs = -1
            best_fitness = tuple(-1 for _ in range(num_objectives))
            for idx, (fitness, _) in enumerate(population_with_fitness):
                if fitness[obj] > best_fitness[obj]:
                    best_fitness = fitness
                    best_idxs = idx
            print(f"\t\tBest value for objective {obj}: {population_with_fitness[best_idxs][0]}")
            


# ============================================================= #
# ======================== Select algs ======================== #
# ============================================================= #


def tournament_select(
    T: int, fitted_population: list[tuple[Fitness, Individual]]
) -> Individual:
    """
    Devuelve un individuo seleccionado por torneo, devuelve una copia para evitar efectos laterales
    """
    selected_population = random.sample(fitted_population, T)
    fittest = max(selected_population)[1]
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
