import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# aux functions to get color in hex format
colormap = plt.cm.get_cmap('tab20')
rescale = lambda val: int(val * 255)
get_color_str = lambda x: "#{:02x}{:02x}{:02x}{:02x}".format(*map(rescale, colormap(x)))

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

# ============================================================================== #
# ============================= PLOT FUNCTIONS ================================= #
# ============================================================================== #

def build_community_graph(graph: nx.Graph, communities: list[list[int]]):
    """
    Builds a community graph from a graph and a list of communities.
    """
    node_to_community = community_list_to_dict(communities)

    # Create a new graph
    community_graph = nx.Graph()

    # Add nodes for each community
    for ind, community in enumerate(communities):
        community_graph.add_node(
            ind,
            size=len(community),
            color=get_color_str(ind),
            nodes=community,
        )

    # Add edges between communities
    for node1, node2 in graph.edges():
        community1 = node_to_community[node1]
        community2 = node_to_community[node2]
        if community1 == community2:
            continue
        if community_graph.has_edge(community1, community2):
            community_graph[community1][community2]['weight'] += 1
        else:
            community_graph.add_edge(community1, community2, weight=1)

    return community_graph

def plot_community_graph(
    community_graph: nx.Graph,
    node_scale: float = 40.,
    edge_scale: float = 0.33,
    sorting: dict[int, int] = None,
    ax=None,
):
    node_sizes = [community_graph.nodes[node]['size'] * node_scale for node in community_graph.nodes()]
    node_colors = [community_graph.nodes[node]['color'] for node in community_graph.nodes()]

    # Set edge weights
    edge_weights = [community_graph.edges[edge]['weight'] * edge_scale for edge in community_graph.edges()]

    # Plot the graph
    pos = nx.circular_layout(community_graph)
    if sorting is not None:
        pos = {node: pos[sorting[node]] for node in community_graph.nodes()}
    nx.draw(
        community_graph,
        pos=pos,
        node_size=node_sizes,
        node_color=node_colors,
        width=edge_weights,
        with_labels=True,
        ax=ax,
    )

def plot_circular_community_graph(
    graph: nx.graph,
    communities: list[list[int]],
    node_scale: float = 40.,
    edge_scale: float = 0.33,
    ax=None,
):
    """
    Plots a single community graph with circular layout.
    """
    communities = sorted(communities, key=len, reverse=True)
    community_graph = build_community_graph(graph, communities)
    plot_community_graph(community_graph, node_scale=node_scale, edge_scale=edge_scale, ax=ax)

def plot_circular_community_graph_with_reference(
    graph: nx.Graph,
    communities_1: list[list[int]],
    communities_2: list[list[int]],
    node_scale: float = 40.,
    edge_scale: float = 0.33,
    ax1=None, ax2=None,
):
    """
    Plots a single community graph with circular layout.
    """
    communities_1 = sorted(communities_1, key=len, reverse=True)
    communities_2 = sorted(communities_2, key=len, reverse=True)

    node_to_community_1 = community_list_to_dict(communities_1)

    # Create new graphs
    community_graph_1 = build_community_graph(graph, communities_1)
    community_graph_2 = build_community_graph(graph, communities_2)

    # Adjust community graph 2 to match community graph 1
    positions = []
    for com_2 in community_graph_2.nodes():
        color = np.zeros(4)
        pos = 0.

        nodes = community_graph_2.nodes[com_2]['nodes']
        for node in nodes:
            com_1 = node_to_community_1[node]
            color += np.array(colormap(com_1)) / len(nodes)
            pos += com_1 / len(nodes)
        
        community_graph_2.nodes[com_2]['color'] = "#{:02x}{:02x}{:02x}{:02x}".format(*map(rescale, color))
        positions.append(pos)

    arg_sorting = list(range(len(positions)))
    arg_sorting = sorted(arg_sorting, key=lambda x: positions[x])
    sorting = dict(zip(arg_sorting, list(range(len(arg_sorting)))))

    # Plot graphs
    plot_community_graph(community_graph_1, node_scale=node_scale, edge_scale=edge_scale, ax=ax1)
    plot_community_graph(community_graph_2, node_scale=node_scale, edge_scale=edge_scale, sorting=sorting, ax=ax2)

def plot_bipartite_community_graph(
    graph: nx.Graph,
    communities_1: list[list[int]],
    communities_2: list[list[int]],
    node_scale: float = 40.,
    edge_scale: float = 0.33,
    ax=None,
):
    communities_1 = sorted(communities_1, key=len, reverse=True)

    node_to_community_1 = community_list_to_dict(communities_1)
    node_to_community_2 = community_list_to_dict(communities_2)

    # Create a new graph
    bipartite_community_graph = nx.Graph()

    # Add nodes for each community
    for ind, community in enumerate(communities_1):
        bipartite_community_graph.add_node(ind, size=len(community), color=get_color_str(ind))
    for ind, community in enumerate(communities_2, start=len(communities_1)):
        bipartite_community_graph.add_node(ind, size=len(community))

    # Add edges between communities
    for node in graph.nodes():
        community_1 = node_to_community_1[node]
        community_2 = node_to_community_2[node] + len(communities_1)

        if bipartite_community_graph.has_edge(community_1, community_2):
            bipartite_community_graph[community_1][community_2]['weight'] += 1
        else:
            bipartite_community_graph.add_edge(community_1, community_2, weight=1)
    

    # Set node colors for the second set of nodes
    positions = []
    for node_2, community in enumerate(communities_2, start=len(communities_1)):
        color = np.zeros(4)
        pos = 0.
        total_weight = 0.

        # iterate over all edges
        for node_1 in bipartite_community_graph.neighbors(node_2):
            weight = bipartite_community_graph[node_1][node_2]['weight']

            pos += weight * node_1
            color += np.array(colormap(node_1)) * weight
            total_weight += weight
        
        color /= total_weight
        pos /= total_weight
        bipartite_community_graph.nodes[node_2]['color'] = "#{:02x}{:02x}{:02x}{:02x}".format(*map(rescale, color))
        positions.append(pos)
    
    # Sort the second set of nodes according to the "pos" attribute
    sorted_nodes_2 = sorted(
        list(range(len(communities_2))),
        key=lambda idx: positions[idx]
    )
    mapping = {(node + len(communities_1)): ind for ind, node in enumerate(sorted_nodes_2, start=len(communities_1))}
    bipartite_community_graph = nx.relabel_nodes(bipartite_community_graph, mapping)

    # Set node sizes and colors
    node_sizes = [bipartite_community_graph.nodes[node]['size'] * node_scale for node in bipartite_community_graph.nodes()]
    node_colors = [bipartite_community_graph.nodes[node]['color'] for node in bipartite_community_graph.nodes()]

    # Set edge weights
    edge_weights = [bipartite_community_graph.edges[edge]['weight'] * edge_scale for edge in bipartite_community_graph.edges()]

    # Plot the graph
    pos = nx.bipartite_layout(
        bipartite_community_graph,
        list(range(len(communities_1)))
    )
    nx.draw(
        bipartite_community_graph,
        pos=pos,
        node_size=node_sizes,
        node_color=node_colors,
        width=edge_weights,
        with_labels=False,
        ax=ax,
    )
