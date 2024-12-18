import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from tensorflow import keras
from tensorflow.keras import layers
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('converted_sensor_data_lift_1.csv')

# Clean up column names
data.columns = data.columns.str.strip()

# Convert relevant columns to numeric
cols_to_convert = [
    'Temperature', 'Humidity',
    'Acceleration X', 'Acceleration Y', 'Acceleration Z',
    'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z',
    'Mag X', 'Mag Y', 'Mag Z',
    'Distance'
]

for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Handle missing values
data = data.dropna()

# Select relevant features for clustering
features = data[cols_to_convert]

# Normalize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Build the Autoencoder using Functional API
input_dim = scaled_features.shape[1]
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation='relu')(input_layer)  # Encoding layer
bottleneck = layers.Dense(4, activation='relu')(encoded)  # Bottleneck
decoded = layers.Dense(8, activation='relu')(bottleneck)  # Decoding layer
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)  # Output layer

autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

# Compile and fit the autoencoder
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, shuffle=True)

# Create encoder model
encoder = keras.Model(inputs=input_layer, outputs=bottleneck)
encoded_data = encoder.predict(scaled_features)

# Cluster the encoded data using KMeans
kmeans = KMeans(n_clusters=9, random_state=42)
clusters = kmeans.fit_predict(encoded_data)

# Add the cluster labels to the original DataFrame
data['Cluster'] = clusters

# Save the labeled data to a CSV file
output_file = 'labeled_sensor_data_lift_1.csv'
data.to_csv(output_file, index=False)

print(f"Labeled dataset saved to {output_file}")
