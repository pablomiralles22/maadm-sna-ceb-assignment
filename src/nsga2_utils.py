import itertools

from heapq import nlargest
from typing import TypeVar, Union, Tuple

Individual = TypeVar("Individual")
Fitness = TypeVar("Fitness", bound=Tuple[Union[int, float], ...])

def is_dominated(fitness_1: Fitness, fitness_2: Fitness) -> bool:
    """
    Devuelve True si fitness_1 domina a fitness_2
    """
    return any(f1 > f2 for f1, f2 in zip(fitness_1, fitness_2)) and all(
        f1 >= f2 for f1, f2 in zip(fitness_1, fitness_2)
    )

def fast_non_dominated_sort(
    population_with_fitness: list[tuple[Fitness, Individual]]
) -> list[list[int]]:
    """
    Devuelve una lista de frentes de Pareto, donde cada frente es una lista de índices de individuos
    """
    # domination_by_count[i] = cuántos individuos dominan a i
    domination_by_count: dict[int, int] = {
        idx: 0 for idx in range(len(population_with_fitness))
    }
    # dominates[i] = lista de individuos dominados por i
    dominates: dict[int, list[int]] = {
        idx: [] for idx in range(len(population_with_fitness))
    }

    for (idx1, (fitness1, _)), (idx2, (fitness2, _)) in itertools.product(
        enumerate(population_with_fitness), repeat=2
    ):
        if idx1 == idx2:
            continue

        if is_dominated(fitness1, fitness2):
            domination_by_count[idx2] += 1
            dominates[idx1].append(idx2)

    fronts = [[idx for idx, count in domination_by_count.items() if count == 0]]
    front_index = 0

    while len(fronts) > front_index:
        next_front = []

        for idx in fronts[front_index]:
            for dominated_idx in dominates[idx]:
                domination_by_count[dominated_idx] -= 1

                if domination_by_count[dominated_idx] == 0:
                    next_front.append(dominated_idx)

        front_index += 1

        if len(next_front) > 0:
            fronts.append(next_front)

    return fronts

class CrowdingDistanceUtils:
    __INF: float = float("inf")
    __EPS: float = 1e-6

    @classmethod
    def calculate(cls, fitness_values: list[Fitness]) -> list[float]:
        num_objectives = len(fitness_values[0])
        crowding_distances = [0 for _ in range(len(fitness_values))]

        indices = list(range(len(fitness_values)))

        for obj_idx in range(num_objectives):
            # get indices sorted by fitness value of current objective
            # i.e. [fitness_values[j] for j in sorted_idxs] is sorted by the obj_idx-th objective
            sorted_idxs = sorted(indices, key=lambda idx: fitness_values[idx][obj_idx])

            first_idx = sorted_idxs[0]
            last_idx = sorted_idxs[-1]

            min_value = fitness_values[first_idx][obj_idx]
            max_value = fitness_values[last_idx][obj_idx]
            scale = max_value - min_value
            if scale < cls.__EPS:
                scale = 1.


            crowding_distances[first_idx] = cls.__INF
            crowding_distances[last_idx] = cls.__INF

            for prev_j, j, next_j in zip(sorted_idxs, sorted_idxs[1:], sorted_idxs[2:]):
                prev_value = fitness_values[prev_j][obj_idx]
                next_value = fitness_values[next_j][obj_idx]
                crowding_distances[j] += (next_value - prev_value) / scale

        return crowding_distances

