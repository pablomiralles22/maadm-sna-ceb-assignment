import random
import statistics
import numpy as np

from copy import deepcopy
from heapq import nlargest
from typing import Callable, TypeVar, Generic, Union, Tuple
from nsga2_utils import CrowdingDistanceUtils, fast_non_dominated_sort

# ======================================================= #
# ======================== Types ======================== #
# ======================================================= #

Individual = TypeVar("Individual")
Population = list[Individual]

Fitness = TypeVar("Fitness", bound=Tuple[Union[int, float], ...])

FitnessFn = Callable[[Individual], Fitness]
SelectFn = Callable[[list[tuple[int, float, Fitness, Individual]]], Individual]  # (rank, crowding_distance, fitness, individual)
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
            population_with_fitness = [
                (fitness, individual) for _, _, fitness, individual in ranked_population
            ]

            # Log
            if trace > 0 and ind % trace == 0:
                self.__log(ind, ranked_population)

        # Log final result
        if trace > 0:
            self.__log(ind, ranked_population)

        return sorted(ranked_population, key=lambda x: (x[0], -x[1]))

    def __calc_fitness(
        self, population: Population
    ) -> list[tuple[Fitness, Individual]]:
        """
        Devuelve una lista de pares de individuos con sus valores fitness.
        """
        fitness_values = map(self.fitness_fn, population)
        return list(zip(fitness_values, population))

    def __generate_offspring(self, ranked_population: list[tuple[int, float, Fitness, Individual]]) -> Population:
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
    ) -> list[tuple[int, float, Fitness, Individual]]:  # (rank, crowding_dist, fitness, individual)
        """
        Devuelve la siguiente población, seleccionando los mejores individuos.
        Los valores de retorno el rango del individuo (número de frente en el
        que aparece), su fitness y el individuo.
        """
        next_population = []
        remaining_size = population_size

        for front_idx, front in enumerate(fronts):
            fitness_values = [population_with_fitness[idx][0] for idx in front]
            crowding_distances = CrowdingDistanceUtils.calculate(fitness_values)

            if len(front) <= remaining_size:
                next_population.extend(
                    (front_idx, crowding_distances[j], *population_with_fitness[front[j]])
                    for j in range(len(front))
                )
                remaining_size -= len(front)
            else:
                # Crowding distance selection
                best_idxs = nlargest(remaining_size, range(len(front)), key=lambda j: crowding_distances[j])
                next_population.extend(
                    (front_idx, crowding_distances[j], *population_with_fitness[front[j]])
                    for j in best_idxs
                )
                break

        return next_population

    def __log(
        self,
        generation_ind: int,
        ranked_population: list[tuple[int, float, Fitness, Individual]],
    ):
        num_objectives = len(ranked_population[0][2])
        std_dev_fitness = [
            statistics.stdev(fitness[idx] for _, _, fitness, _ in ranked_population)
            for idx in range(num_objectives)
        ]
        print(f"Generation {generation_ind} - {std_dev_fitness=}")

        for obj in range(num_objectives):
            best_idxs = -1
            best_fitness = tuple(-1 for _ in range(num_objectives))
            for idx, (_, _, fitness, _) in enumerate(ranked_population):
                if fitness[obj] > best_fitness[obj]:
                    best_fitness = fitness
                    best_idxs = idx
            print(f"\t\tBest value for objective {obj}: {ranked_population[best_idxs][2]}")
            
