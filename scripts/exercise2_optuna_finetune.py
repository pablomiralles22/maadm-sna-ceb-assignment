import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import networkx as nx
import functools
import argparse
import optuna
import numpy as np
import math

from nsga2 import NSGA2
from genetic_operators import (
    tournament_select,
    random_locus_crossover,
    mutate_locus,
    mutate_locus_join,
    mutate_locus_separate,
    mutate_combine,
    create_locus,
)
from disjoint_set_union import DisjointSetUnion
from metrics import build_metric


# ================================================================================= #
# ============================== Parse arguments ================================== #
# ================================================================================= #

parser = argparse.ArgumentParser(description="Process graph file.")
parser.add_argument(
    "-g", "--graph-file", type=str, help="Path to the graph file", required=True
)
parser.add_argument(
    "-t", "--num-trials", type=int, help="Number of trials for optuna study", default=10
)
parser.add_argument(
    "-s", "--study-name", type=str, help="Name of the study", required=True
)
parser.add_argument(
    "-p",
    "--proc-id",
    type=int,
    help="Id of this process, for parallel execution",
    default=1,
)
parser.add_argument(
    "--budget-fitness-calls", type=int, help="Max number of fitness calls", default=1000
)
parser.add_argument(
    "-m1",
    "--metric-1",
    type=str,
    help="Metric 1 for MO optimization",
    required=True,
    choices=["modularity", "avg_odf", "internal_density"],
)
parser.add_argument(
    "-m2",
    "--metric-2",
    type=str,
    help="Metric 2 for MO optimization",
    required=True,
    choices=["modularity", "avg_odf", "internal_density"],
)
parser.add_argument(
    "--runs-per-experiment",
    type=int,
    help="Number of runs per configuration, to take the mean",
    default=3,
)
args = parser.parse_args()

# ================================================================================= #
# ============================== Helper functions ================================= #
# ================================================================================= #


def community_list_to_dict(communities: list[list[int]]) -> dict[int, int]:
    # list of list of nodes in the same community to dict of node to community
    return {
        node: ind for ind, community in enumerate(communities) for node in community
    }


def community_dict_to_list(community_dict: dict[int, int]) -> list[list[int]]:
    # dict of node to community to list of list of nodes in the same community
    num_communities = max(community_dict.values()) + 1
    communities = [[] for _ in range(num_communities)]
    for node, community in community_dict.items():
        communities[community].append(node)
    return communities


# =============================================================================== #
# ============================== Read graph ===================================== #
# =============================================================================== #

GRAPH_FILE = args.graph_file
graph = nx.read_graphml(GRAPH_FILE)

# create data structures for convinient index access to nodes
nodes = list(graph.nodes())
node_to_idx = {node: idx for idx, node in enumerate(nodes)}
idx_to_node = {idx: node for idx, node in enumerate(nodes)}

edges: list[list[int]] = [
    list(set([node_to_idx[neighbour] for neighbour in graph.neighbors(idx_to_node[idx])] + [idx]))
    for idx in range(graph.number_of_nodes())
]

# =============================================================================== #
# ============================== Evaluate ======================================= #
# =============================================================================== #

def individual_to_communities(individual: list[int]) -> list[list[int]]:
    """
    Given an individual of the genetic algorithm, return the communities it represents,
    with the original labels in the graph.
    """
    dsu = DisjointSetUnion(len(individual))
    for i1, i2 in enumerate(individual):
        dsu.join(i1, i2)
    components = dsu.get_components()
    communities = [[idx_to_node[idx] for idx in component] for component in components]
    return communities


def evaluate(
    population_size: int,
    pcross: float,
    pmut: float,
    mutate_ratio_random: float,
    mutate_ratio_join: float,
    mutate_ratio_separate: float,
    T: int,
    metric_1: str,
    metric_2: str,
):
    ngen = args.budget_fitness_calls // population_size
    mutate_fn_1 = functools.partial(mutate_locus, edges, mutate_ratio_random)
    mutate_fn_2 = functools.partial(mutate_locus_join, edges, mutate_ratio_join)
    mutate_fn_3 = functools.partial(mutate_locus_separate, mutate_ratio_separate)
    mutate_fn = functools.partial(
        mutate_combine, [mutate_fn_1, mutate_fn_2, mutate_fn_3]
    )
    select_fn = functools.partial(tournament_select, T)
    metric_1 = build_metric(metric_1, graph)
    metric_2 = build_metric(metric_2, graph)

    def fitness_fn(individual: list[int]) -> tuple[float, float]:
        communities = individual_to_communities(individual)
        return metric_1(communities), metric_2(communities)

    population = create_locus(edges, population_size)

    ga = NSGA2[list[int], tuple[float, float]](
        fitness_fn=fitness_fn,
        select_fn=select_fn,
        crossover_fn=random_locus_crossover,
        mutate_fn=mutate_fn,
        pcross=pcross,
        pmut=pmut,
    )

    ranked_population = ga.run(population, ngen=ngen)

    return ranked_population


# =================================================================================================== #
# ============================== Optuna optimize hyperparameters ==================================== #
# =================================================================================================== #


def get_pareto_front(
    fitness_values: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    fitness_values.sort(reverse=True)
    max_f2 = float("-inf")
    non_dominated = []
    for f1, f2 in fitness_values:
        if f2 <= max_f2:
            continue
        non_dominated.append((f1, f2))
        max_f2 = f2
    return non_dominated


def hypervolume(pareto_front: list[tuple[float, float]]) -> float:
    """
    Calculate hypervolume of the given fitness values, with reference
    point (0, 0). This is because we want to maximize both metrics,
    and they are in the interval [0, 1].

    WARNING: assume they are sorted.
    """
    # calculate hypervolume in O(N)
    hypervolume = pareto_front[0][0] * pareto_front[0][1]
    for (_, prev_f2), (f1, f2) in zip(pareto_front, pareto_front[1:]):
        hypervolume += f1 * (f2 - prev_f2)

    return hypervolume


def normalized_zitzler(pareto_front: list[tuple[float, float]]) -> float:
    """
    Calculate the normalized Zitzler indicator of the given fitness values,
    normalized by the maximum diagonal size of \sqrt{2} in our case.

    WARNING: assume they are sorted.
    """
    first, last = pareto_front[0], pareto_front[-1]
    l1 = abs(first[0] - last[0])
    l2 = abs(first[1] - last[1])

    return math.sqrt(l1**2 + l2**2) / math.sqrt(2)


def objective(trial):
    population_size = trial.suggest_int("population_size", 25, 150, step=25)
    pcross = trial.suggest_float("pcross", 0.5, 1.0, step=0.05)
    pmut = trial.suggest_float("pmut", 0.0, 0.5, step=0.05)
    mutate_ratio_random = trial.suggest_float("mutate_ratio_random", 0.0, 0.2)
    mutate_ratio_join = trial.suggest_float("mutate_ratio_join", 0.0, 1.0)
    mutate_ratio_separate = trial.suggest_float("mutate_ratio_separate", 0.0, 0.2)
    T = trial.suggest_int("T", 2, 16)

    hypervolumes = []
    normalized_zitzlers = []
    for _ in range(args.runs_per_experiment):
        ranked_population = evaluate(
            population_size=population_size,
            pcross=pcross,
            pmut=pmut,
            mutate_ratio_random=mutate_ratio_random,
            mutate_ratio_join=mutate_ratio_join,
            mutate_ratio_separate=mutate_ratio_separate,
            T=T,
            metric_1=args.metric_1,
            metric_2=args.metric_2,
        )

        fitness_values = [fitness for _, _, fitness, _ in ranked_population]
        # calculate non dominated points in O(N)
        pareto_front = get_pareto_front(fitness_values)
        hypervolumes.append(hypervolume(pareto_front))
        normalized_zitzlers.append(normalized_zitzler(pareto_front))

    return np.mean(hypervolumes), np.mean(normalized_zitzlers)


if __name__ == "__main__":
    # set log file
    log_file = f"logs/{args.study_name}_{args.proc_id}.log"
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = open(log_file, "w")

    db_file = f"db/{args.study_name}.db"
    Path(db_file).parent.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{db_file}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_name,
        directions=["maximize", "maximize"],
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)

    print(study.best_trials)
