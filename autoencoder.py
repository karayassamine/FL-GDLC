import numpy as np






from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam


# Define the autoencoder

def build_autoencoder(input_dim, latent_dim=14):
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(latent_dim, activation='relu')(encoder_input)
    
    # Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Build models
    autoencoder = Model(encoder_input, decoded)
    encoder = Model(encoder_input, encoded)
    
    # Compile autoencoder
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                       loss='mse')  # MSE works better for normalized data
    
    return autoencoder, encoder


