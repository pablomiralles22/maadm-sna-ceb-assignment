import networkx as nx
import numpy as np

def avg_odf(graph: nx.Graph, community: object) -> float:
    """
    Average Out-Degree Fraction (AODF) is the average of the out-degree fraction
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
    return np.mean(values)