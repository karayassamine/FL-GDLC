import networkx as nx
import igraph as ig
import community


def has_strong_community_structure_strength(G):
    # return cal_modularity(G) > 0.4
    return cal_mixing_parameter(G) < 0.1


def has_high_density(G: nx.Graph):
    # returns true if a graph has a high density (greater than 0.1)
    return nx.density(G) > 0.1


def has_high_transitivity(G):
    # returns true if a graph has a high transitivity (greater than 0.5)
    return nx.transitivity(G) > 0.5


def cal_modularity(G):
    modularity = nx.community.modularity(G, get_communities(G))
    return modularity


def cal_mixing_parameter(G, communities=None):
    if communities == None:
        communities = get_communities(G)
    inter_cluster_edges = 0

    # Iterate over all edges in the graph
    for u, v in G.edges():
        for s in communities:  # type: ignore
            # Check if the nodes belong to different communities
            if u in s:
                if v not in s:
                    inter_cluster_edges += 1
                break
    return inter_cluster_edges / G.number_of_edges()


def get_communities(G):
    # coms = nx.community.louvain_communities(G)
    # coms = nx.community.label_propagation_communities(G)  # type: ignore
    import igraph as ig
    G1 = ig.Graph.from_networkx(G)
    part = G1.community_infomap()

    coms = []
    for com in part:
        coms.append([G1.vs[node_index]['label'] for node_index in com])

    return coms
