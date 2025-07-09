master_centralities = [
    "betweenness",          # Index 0
    "local_betweenness",    # Index 1
    "global_betweenness",   # Index 2
    "degree",               # Index 3
    "local_degree",         # Index 4
    "global_degree",        # Index 5
    "eigenvector",          # Index 6
    "closeness",            # Index 7
    "pagerank",             # Index 8
    "local_pagerank",       # Index 9
    "global_pagerank",      # Index 10
    "k_core",               # Index 11
    "k_truss",              # Index 12
    "Comm",                 # Index 13
    "mv"                    # Index 14
]

import numpy as np

def create_unified_features(G, dataset):
    """
    Args:
        G (nx.Graph): NetworkX graph with node attributes for centralities.
        dataset: Object with `cn_measures` attribute listing expected measures.
    Returns:
        np.ndarray: Feature matrix of shape (num_nodes, 15).
    """
    sorted_nodes = sorted(G.nodes())  # Ensure consistent node order
    num_nodes = len(sorted_nodes)
    feature_matrix = np.zeros((num_nodes, len(master_centralities)))
    
    # Get indices of measures present in this dataset's nodes
    present_measures = [measure for measure in dataset.cn_measures if measure in master_centralities]
    present_indices = [master_centralities.index(m) for m in present_measures]
    
    # Populate the feature matrix
    for i, node in enumerate(sorted_nodes):
        node_data = G.nodes[node]
        for measure, idx in zip(present_measures, present_indices):
            feature_matrix[i, idx] = node_data.get(measure, 0.0)  # Use .get() for safety
    
    return feature_matrix