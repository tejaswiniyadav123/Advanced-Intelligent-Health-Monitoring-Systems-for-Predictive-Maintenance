import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, Model
from mpl_toolkits.mplot3d import Axes3D
import daal4py as d4p
import time

# Initialize daal4py
d4p.daalinit()

# Start the timer for the entire process
overall_start_time = time.time()

# Load and preprocess the dataset
data = pd.read_csv('converted_sensor_data_lift_1.csv')
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

# Select features and normalize
features = data[cols_to_convert]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Build Autoencoder
input_dim = scaled_features.shape[1]
input_layer = layers.Input(shape=(input_dim,))
encoded = layers.Dense(8, activation='relu')(input_layer)
bottleneck = layers.Dense(4, activation='relu')(encoded)
decoded = layers.Dense(8, activation='relu')(bottleneck)
output_layer = layers.Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train Autoencoder
start_time = time.time()
autoencoder.fit(scaled_features, scaled_features, epochs=50, batch_size=32, shuffle=True)
print(f"Autoencoder training completed in {time.time() - start_time:.2f} seconds.")

# Extract encoded data
encoder = Model(inputs=input_layer, outputs=bottleneck)
encoded_data = encoder.predict(scaled_features)

# Save encoded data to a temporary CSV for daal4py processing
encoded_data_df = pd.DataFrame(encoded_data, columns=['Encoded_1', 'Encoded_2', 'Encoded_3', 'Encoded_4'])
encoded_data_df.to_csv("local_kmeans_data.csv", index=False)

# Load encoded data for daal4py
data_for_kmeans = pd.read_csv("local_kmeans_data.csv", dtype=np.float32)

# Initialize centroids using daal4py's KMeans init algorithm
start_time = time.time()
init_alg = d4p.kmeans_init(nClusters=3, fptype="float", method="randomDense", distributed=True)
centroids = init_alg.compute(data_for_kmeans).centroids
print(f"Centroid initialization completed in {time.time() - start_time:.2f} seconds.")

# Perform KMeans clustering using daal4py
start_time = time.time()
alg = d4p.kmeans(nClusters=3, maxIterations=50, fptype="float", accuracyThreshold=0, assignFlag=False, distributed=True)
result = alg.compute(data_for_kmeans, centroids)
print(f"KMeans clustering completed in {time.time() - start_time:.2f} seconds.")

clusters = result.assignments

# Add cluster labels to the original data
data['Cluster'] = clusters

# Map clusters to descriptive names
cluster_names = {
    0: 'Low Vibration, Low Temperature (Normal operation)',
    1: 'High Vibration, High Temperature (Severe anomaly or fault)',
    2: 'Moderate Vibration, Balanced Conditions (Potential concern or transitional phase)',
}
data['Cluster_Name'] = data['Cluster'].map(cluster_names)

# Visualize clusters in 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

colors = ['blue', 'red', 'green']
scatter = ax.scatter(
    encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2],
    c=[colors[int(cluster)] for cluster in clusters], alpha=0.7
)

ax.set_title('3D Cluster Visualization using Autoencoder')
ax.set_xlabel('Encoded Feature 1')
ax.set_ylabel('Encoded Feature 2')
ax.set_zlabel('Encoded Feature 3')

# Add legend for clusters
handles = [
    plt.Line2D([0], [0], marker='o', color='w', label=cluster_names[i],
               markerfacecolor=colors[i], markersize=10) for i in range(3)
]
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

# Show plot
plt.show()

# Print final cluster centroids
print("Cluster Centroids:\n", centroids)

# Print total time taken for the process
print(f"Total time taken: {time.time() - overall_start_time:.2f} seconds.")
