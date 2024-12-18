import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # Convert relevant columns to numeric, coercing errors into NaN
    key_features = ['Temperature (C)', 'Humidity (%)', 'Acceleration X', 'Acceleration Y', 'Acceleration Z',
                    'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z', 'Mag X', 'Mag Y', 'Mag Z', 'Distance (cm)']
    for col in key_features:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    return data


# Function for cluster summary
def cluster_summary(data, key_features):
    # Group the data by clusters and calculate mean, min, max, and std for key features
    return data.groupby('Cluster')[key_features].agg(['mean', 'min', 'max', 'std'])


# Function to count and display the cluster distribution
def cluster_distribution(data):
    return data['Cluster'].value_counts().sort_index()


# Function to generate cluster descriptive statistics
def cluster_descriptive_stats(data, key_features):
    return data.groupby('Cluster')[key_features].describe()


# Function to suggest and map cluster names
def map_cluster_names(data):
    cluster_names = {
        0: 'Stable Low Temp, Low Vibration',
        1: 'High Temp, High Vibration',
        2: 'Moderate Temp, Moderate Vibration',
        3: 'Stable High Temp, Low Vibration',
        4: 'Low Temp, High Humidity',
        5: 'High Temp, Low Humidity',
        6: 'Unstable Operation, High Vibration',
        7: 'Low Temp, Low Vibration',
        8: 'Extreme Conditions, High Temp and Vibration'
    }

    data['Cluster Name'] = data['Cluster'].map(cluster_names)
    return data


# Function to save the updated dataset
def save_updated_data(data, output_file_path):
    data.to_csv(output_file_path, index=False)
    print(f"Updated data with cluster names has been saved to {output_file_path}")


# Function for visualizations
def create_visualizations(data, key_features):
    sns.set(style="whitegrid")

    # Pairplot for visualizing relationships between key features
    sns.pairplot(data, hue='Cluster Name', vars=key_features[:6])
    plt.suptitle('Pairplot of Key Features by Cluster', y=1.02)
    plt.show()

    # Boxplots for visualizing the distribution of each key feature within clusters
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(key_features[:6], 1):
        plt.subplot(2, 3, i)
        sns.boxplot(x='Cluster Name', y=feature, data=data)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {feature} by Cluster')
    plt.tight_layout()
    plt.show()

    # 3D scatter plot for the first three features (Acceleration)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(data['Acceleration X'], data['Acceleration Y'], data['Acceleration Z'],
                         c=data['Cluster'], cmap='viridis', s=40)

    # Create a legend for the scatter plot
    legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend1)

    ax.set_xlabel('Acceleration X')
    ax.set_ylabel('Acceleration Y')
    ax.set_zlabel('Acceleration Z')

    # Adjust viewing angle for better aesthetics
    ax.view_init(elev=20, azim=-60)

    plt.title('3D Scatter Plot of Acceleration Features by Cluster')
    plt.show()


# Main script
def main():
    # Step 1: Load and preprocess data
    file_path = 'labeled_sensor_data_lift.csv'
    data = load_and_preprocess_data(file_path)

    # Step 2: Cluster Summary
    key_features = ['Temperature (C)', 'Humidity (%)', 'Acceleration X', 'Acceleration Y', 'Acceleration Z',
                    'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z', 'Mag X', 'Mag Y', 'Mag Z', 'Distance (cm)']
    cluster_summary_key = cluster_summary(data, key_features)
    print("Cluster Summary:\n", cluster_summary_key)

    # Step 3: Cluster Distribution
    cluster_dist = cluster_distribution(data)
    print("Cluster Distribution:\n", cluster_dist)

    # Step 4: Cluster Descriptive Statistics
    distinctive_characteristics = cluster_descriptive_stats(data, key_features)
    print("Distinctive Characteristics by Cluster:\n", distinctive_characteristics)

    # Step 5: Map Cluster Names
    data_with_names = map_cluster_names(data)
    print("Data with Cluster Names:\n", data_with_names.head())

    # Step 6: Save updated dataset
    output_file_path = 'labeled_sensor_data_with_cluster_names.csv'
    save_updated_data(data_with_names, output_file_path)

    # Step 7: Visualizations
    create_visualizations(data_with_names, key_features)


if __name__ == "__main__":
    main()
