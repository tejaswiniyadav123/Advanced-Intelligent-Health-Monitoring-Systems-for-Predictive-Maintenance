import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.spatial.distance import euclidean

# Load the dataset
file_path = 'labeled_sensor_data_lift.csv'
df = pd.read_csv(file_path)

# Define the cluster names (0 to 8)
cluster_names = {
    0: "Stable Temp, Low Vertical Vibration",
    1: "Moderate Temp, High Horizontal Vibration",
    2: "Moderate Conditions, High Activity",
    3: "Stable Temp, High Vertical Impact",
    4: "Balanced Temp, Consistent Movement",
    5: "Moderate Temp, High Stability",
    6: "Cool Temp, High Gyroscope Activity",
    7: "Stable Conditions, Low Vibration",
    8: "Balanced Temp, Low Impact"
}

# Convert necessary columns to numeric, forcing errors to NaN
df['Temperature (C)'] = pd.to_numeric(df['Temperature (C)'], errors='coerce')
df['Humidity (%)'] = pd.to_numeric(df['Humidity (%)'], errors='coerce')
df['Distance (cm)'] = pd.to_numeric(df['Distance (cm)'], errors='coerce')

# Drop rows with any NaN values in feature columns
df_cleaned = df.dropna()

# Ensure that the target clusters are within the defined range (0-8)
df_cleaned = df_cleaned[df_cleaned['Cluster'].isin(cluster_names.keys())]

# Split features and target
X = df_cleaned.drop('Cluster', axis=1)
y = df_cleaned['Cluster']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_scaled, y)

# Calculate cluster centroids
cluster_centroids = []
for i in sorted(cluster_names.keys()):
    if np.any(y == i):
        centroid = X_scaled[y == i].mean(axis=0)
    else:
        centroid = np.nan * np.ones(X_scaled.shape[1])  # Create a NaN array for empty clusters
    cluster_centroids.append(centroid)
cluster_centroids = np.array(cluster_centroids)

def calculate_ttf(feature_vector, cluster_index, cluster_centroids, dangerous_clusters, threshold=0.5):
    if cluster_index in dangerous_clusters:
        return "Low", cluster_names.get(cluster_index, "Unknown")

    valid_dangerous_centroids = [centroid for i, centroid in enumerate(cluster_centroids) if
                                 i in dangerous_clusters and not np.isnan(centroid).any()]

    if len(valid_dangerous_centroids) == 0:
        return "Unknown", cluster_names.get(cluster_index, "Unknown")

    min_distance = min(
        [euclidean(feature_vector, dangerous_centroid) for dangerous_centroid in valid_dangerous_centroids])

    if min_distance < threshold:
        return "Medium", cluster_names.get(cluster_index, "Unknown")

    return "High", cluster_names.get(cluster_index, "Unknown")

def predict_cluster_and_ttf(temp, hum, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, dist):
    new_data = np.array([[temp, hum, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z, dist]])
    new_data_scaled = scaler.transform(new_data)
    cluster_pred = svm_model.predict(new_data_scaled)[0]

    # Ensure the cluster prediction is within the valid range
    if cluster_pred not in cluster_names:
        cluster_name = "Unknown"
        ttf = "Unknown"
    else:
        dangerous_clusters = [1, 6, 8]  # Assuming these are dangerous clusters
        ttf, cluster_name = calculate_ttf(new_data_scaled.flatten(), cluster_pred, cluster_centroids, dangerous_clusters)

    return cluster_pred, cluster_name, ttf

# Tkinter GUI setup
def submit_action():
    try:
        temp = float(temp_entry.get())
        hum = float(hum_entry.get())
        acc_x = float(acc_x_entry.get())
        acc_y = float(acc_y_entry.get())
        acc_z = float(acc_z_entry.get())
        gyro_x = float(gyro_x_entry.get())
        gyro_y = float(gyro_y_entry.get())
        gyro_z = float(gyro_z_entry.get())
        mag_x = float(mag_x_entry.get())
        mag_y = float(mag_y_entry.get())
        mag_z = float(mag_z_entry.get())
        dist = float(dist_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numerical values.")
        return

    cluster_pred, cluster_name, ttf = predict_cluster_and_ttf(temp, hum, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                                                               mag_x, mag_y, mag_z, dist)

    result_label.config(text=f"Predicted Cluster: {cluster_pred} ({cluster_name})\nTime to Failure (TTF): {ttf}")

# Create the main window
root = tk.Tk()
root.title("Lift Health Monitoring System")

# Create and place labels and entries
tk.Label(root, text="Temperature (C):").grid(row=0, column=0, padx=10, pady=5)
temp_entry = tk.Entry(root)
temp_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(root, text="Humidity (%):").grid(row=1, column=0, padx=10, pady=5)
hum_entry = tk.Entry(root)
hum_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(root, text="Acceleration X:").grid(row=2, column=0, padx=10, pady=5)
acc_x_entry = tk.Entry(root)
acc_x_entry.grid(row=2, column=1, padx=10, pady=5)

tk.Label(root, text="Acceleration Y:").grid(row=3, column=0, padx=10, pady=5)
acc_y_entry = tk.Entry(root)
acc_y_entry.grid(row=3, column=1, padx=10, pady=5)

tk.Label(root, text="Acceleration Z:").grid(row=4, column=0, padx=10, pady=5)
acc_z_entry = tk.Entry(root)
acc_z_entry.grid(row=4, column=1, padx=10, pady=5)

tk.Label(root, text="Gyroscope X:").grid(row=5, column=0, padx=10, pady=5)
gyro_x_entry = tk.Entry(root)
gyro_x_entry.grid(row=5, column=1, padx=10, pady=5)

tk.Label(root, text="Gyroscope Y:").grid(row=6, column=0, padx=10, pady=5)
gyro_y_entry = tk.Entry(root)
gyro_y_entry.grid(row=6, column=1, padx=10, pady=5)

tk.Label(root, text="Gyroscope Z:").grid(row=7, column=0, padx=10, pady=5)
gyro_z_entry = tk.Entry(root)
gyro_z_entry.grid(row=7, column=1, padx=10, pady=5)

tk.Label(root, text="Magnetometer X:").grid(row=8, column=0, padx=10, pady=5)
mag_x_entry = tk.Entry(root)
mag_x_entry.grid(row=8, column=1, padx=10, pady=5)

tk.Label(root, text="Magnetometer Y:").grid(row=9, column=0, padx=10, pady=5)
mag_y_entry = tk.Entry(root)
mag_y_entry.grid(row=9, column=1, padx=10, pady=5)

tk.Label(root, text="Magnetometer Z:").grid(row=10, column=0, padx=10, pady=5)
mag_z_entry = tk.Entry(root)
mag_z_entry.grid(row=10, column=1, padx=10, pady=5)

tk.Label(root, text="Distance (cm):").grid(row=11, column=0, padx=10, pady=5)
dist_entry = tk.Entry(root)
dist_entry.grid(row=11, column=1, padx=10, pady=5)

# Submit button
submit_btn = tk.Button(root, text="Submit", command=submit_action)
submit_btn.grid(row=12, column=0, columnspan=2, padx=10, pady=10)

# Result label
result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.grid(row=13, column=0, columnspan=2, padx=10, pady=10)

# Run the Tkinter event loop
root.mainloop()
