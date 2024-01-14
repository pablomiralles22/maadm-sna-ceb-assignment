import networkx as nx
import networkx.algorithms.community as nxcom
import numpy as np

from typing import Literal, Callable
from cdlib import NodeClustering
from cdlib.evaluation import internal_edge_density

def avg_odf(graph: nx.Graph, community: object, summary=True) -> float:
    """
    Average Out-Degree Fraction (AODF) is the average of the out-degree fraction.
    Redefined because it was wrong in the library cdlib.
    $$
     \\frac{1}{n_S} \\sum_{u \\in S} \\frac{|\{(u,v)\\in E: v \\not\\in S\}|}{d(u)}
    $$
    """
    values = []
    for community in community.communities:
        community_subgraph = nx.subgraph(graph, community)
        out_degree_fraction = np.mean(
            [
                1. - (community_subgraph.degree(node) / graph.degree(node))
                for node in community_subgraph.nodes()
            ]
        )
        values.append(out_degree_fraction)
    if summary is True:
        return np.mean(values)
    return values

def build_metric(
    metric_name: Literal["modularity", "avg_odf", "internal_density"],
    graph: nx.Graph,
) -> Callable[[list[list[int]]], float]:
    """
    Builds a unified metric function onto [0, 1], where 1 is the best value.
    """
    if metric_name == "modularity":
        return lambda communities: nxcom.modularity(graph, communities)
    elif metric_name == "internal_density":
        return lambda communities: internal_edge_density(graph, NodeClustering(communities, graph)).score
    elif metric_name == "avg_odf":
        return lambda communities: 1.0 - avg_odf(graph, NodeClustering(communities, graph))
    else:
        raise ValueError(f"Unknown metric {metric_name}")
