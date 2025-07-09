import numpy as np

def extract_features(G, cn_measures):
    """
    Extract node features dynamically based on the centrality measures available in the network.
    
    Args:
        G (nx.Graph): NetworkX graph with node attributes.
        cn_measures (list): List of centrality measures for this network.
        
    Returns:
        np.ndarray: Feature matrix of shape (num_nodes, len(cn_measures)).
    """
    sorted_nodes = sorted(G.nodes())
    num_nodes = len(sorted_nodes)
    feature_matrix = np.zeros((num_nodes, len(cn_measures)))
    
    for i, node in enumerate(sorted_nodes):
        node_data = G.nodes[node]
        for j, measure in enumerate(cn_measures):
            feature_matrix[i, j] = node_data.get(measure, 0.0)  # Use .get() to handle missing measures
    
    return feature_matrix




from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


# Define the autoencoder

def build_autoencoder(input_dim, latent_dim=8):

    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(latent_dim, activation='relu')(encoder_input)
    
    # Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Build and compile
    autoencoder = Model(encoder_input, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return autoencoder