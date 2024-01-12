import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import networkx as nx
import functools
import argparse
import optuna

from networkx import community as nxcom
from cdlib import NodeClustering
from cdlib.evaluation import flake_odf, internal_edge_density

from genetic_algs import NSGA2, tournament_select, random_locus_crossover, mutate_locus, create_locus
from disjoint_set_union import DisjointSetUnion


# ================================================================================= #
# ============================== Parse arguments ================================== #
# ================================================================================= #

parser = argparse.ArgumentParser(description='Process graph file.')
parser.add_argument("-g", "--graph-file", type=str, help='Path to the graph file')
parser.add_argument("-t", "--num-trials", type=int, help='Number of trials for optuna study')
parser.add_argument("-l", "--log-file", type=str, help='File to log everything to')
args = parser.parse_args()

# ================================================================================= #
# ============================== Helper functions ================================= #
# ================================================================================= #

def community_list_to_dict(communities: list[list[int]]) -> dict[int, int]:
    # list of list of nodes in the same community to dict of node to community
    return {
        node: ind
        for ind, community in enumerate(communities)
        for node in community
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
    [ node_to_idx[neighbour] for neighbour in graph.neighbors(idx_to_node[idx]) ]
    for idx in range(graph.number_of_nodes())
]

# =============================================================================== #
# ============================== Evaluate ======================================= #
# =============================================================================== #

# unify metrics interface and output range ([0, 1] interval), so we can pass these as parameters
modularity_metric = lambda communities: nxcom.modularity(graph, communities)
internal_density_metric = lambda communities: internal_edge_density(graph, NodeClustering(communities, graph)).score
flake_odf_metric = lambda communities: 1. - flake_odf(graph, NodeClustering(communities, graph)).score

BUDGET_FITNESS_CALLS = 50_000

def evaluate(
    population_size: int,
    pcross: float,
    pmut: float,
    mutate_ratio: float,
    T: int,
    metric_1: callable,
    metric_2: callable,
):
    ngen = BUDGET_FITNESS_CALLS // population_size
    mutate_fn = functools.partial(mutate_locus, edges, mutate_ratio)
    select_fn = functools.partial(tournament_select, T)

    def fitness_fn(individual: list[int]) -> tuple[float, float]:
        # calculate components using disjoint set union
        dsu = DisjointSetUnion(len(individual))
        for i1, i2 in enumerate(individual):
            dsu.join(i1, i2)
        components = dsu.get_components()
        communities = [[idx_to_node[idx] for idx in component] for component in components]
        
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

    population_with_fitness = ga.run(population, ngen=ngen)

    return population_with_fitness

# =================================================================================================== #
# ============================== Optuna optimize hyperparameters ==================================== #
# =================================================================================================== #

def get_pareto_front(fitness_values: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    WARNING: assume they are sorted.
    """
    max_f2 = float("-inf")
    non_dominated = []
    for f1, f2 in fitness_values:
        if f2 <= max_f2:
            continue
        non_dominated.append((f1, f2))
        max_f2 = f2
    return non_dominated

def hypervolume(fitness_values: list[tuple[float, float]]) -> float:
    """
    Calculate hypervolume of the given fitness values, with reference
    point (0, 0). This is because we want to maximize both metrics,
    and they are in the interval [0, 1].

    WARNING: assume they are sorted.
    """
    # calculate non dominated points in O(N)
    non_dominated = get_pareto_front(fitness_values)

    # calculate hypervolume in O(N)
    hypervolume = non_dominated[0][0] * non_dominated[0][1]
    for (_, prev_f2), (f1, f2) in zip(non_dominated, non_dominated[1:]):
        hypervolume += f1 * (f2 - prev_f2)

    return hypervolume


def objective(trial):
    population_size = trial.suggest_int("population_size", 10, 100, step=5)
    pcross = trial.suggest_float("pcross", 0.5, 1.0)
    pmut = trial.suggest_float("pmut", 0.01, 0.45)
    mutate_ratio = trial.suggest_float("mutate_ratio", 0.01, 0.5)
    T = trial.suggest_int("T", 2, 8)
    metric_2 = trial.suggest_categorical("metric_2", ["flake_odf", "internal_density"])

    population_with_fitness = evaluate(
        population_size=population_size,
        pcross=pcross,
        pmut=pmut,
        mutate_ratio=mutate_ratio,
        T=T,
        metric_1=modularity_metric,
        metric_2=(flake_odf_metric if metric_2 == "flake_odf" else internal_density_metric),
    )

    fitness_values = [fitness for fitness, _ in population_with_fitness]
    return hypervolume(fitness_values)


if __name__ == "__main__":
    if args.log_file:
        Path(args.log_file).parent.mkdir(parents=True, exist_ok=True)
        sys.stdout = open(args.log_file, "w")
        sys.stderr = sys.stdout

    study_name = f"amazon"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.num_trials)

    best_params = study.best_params
    best_value = study.best_value

    print("Best Parameters:", best_params)
    print("Best Value:", best_value)
