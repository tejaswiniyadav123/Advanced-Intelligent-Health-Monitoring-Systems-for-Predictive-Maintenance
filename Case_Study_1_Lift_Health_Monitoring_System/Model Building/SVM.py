import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.spatial.distance import euclidean
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'labeled_sensor_data_lift_1.csv'
df = pd.read_csv(file_path)

# Define the cluster names
cluster_names = {
    0: "Moderate Temp, Low Vertical Vibration",
    1: "Moderate Temp, High Horizontal Vibration",
    2: "Cool Conditions, High Activity",
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

# Split features and target
X = df_cleaned.drop('Cluster', axis=1)
y = df_cleaned['Cluster']

# Verify consistent lengths
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize and train the SVM model
svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

# Extract unique classes in the test set
unique_classes = sorted(y_test.unique())

# Generate the classification report with appropriate target names
classification_rep = classification_report(y_test, y_pred, target_names=[cluster_names[i] for i in unique_classes],
                                           zero_division=0)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rep)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with appropriate labels
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=[cluster_names[i] for i in unique_classes],
            yticklabels=[cluster_names[i] for i in unique_classes])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

# Calculate cluster centroids
cluster_centroids = []
for i in unique_classes:
    if np.any(y_train == i):
        centroid = X_train[y_train == i].mean(axis=0)
    else:
        centroid = np.nan * np.ones(X_train.shape[1])  # Create a NaN array for empty clusters
    cluster_centroids.append(centroid)
cluster_centroids = np.array(cluster_centroids)

# Real-time prediction
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

    dangerous_clusters = [1, 6, 8]  # Assuming these are dangerous clusters
    ttf, cluster_name = calculate_ttf(new_data_scaled.flatten(), cluster_pred, cluster_centroids, dangerous_clusters)

    return cluster_pred, cluster_name, ttf

# Example usage
temp = float(input("Enter Temperature (C): "))
hum = float(input("Enter Humidity (%): "))
acc_x = float(input("Enter Acceleration X: "))
acc_y = float(input("Enter Acceleration Y: "))
acc_z = float(input("Enter Acceleration Z: "))
gyro_x = float(input("Enter Gyroscope X: "))
gyro_y = float(input("Enter Gyroscope Y: "))
gyro_z = float(input("Enter Gyroscope Z: "))
mag_x = float(input("Enter Magnetometer X: "))
mag_y = float(input("Enter Magnetometer Y: "))
mag_z = float(input("Enter Magnetometer Z: "))
dist = float(input("Enter Distance (cm): "))

predicted_cluster, cluster_name, ttf = predict_cluster_and_ttf(temp, hum, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
                                                               mag_x, mag_y, mag_z, dist)
print(f"The predicted cluster is: {predicted_cluster} ({cluster_name})")
print(f"Time to Failure (TTF): {ttf}")
