import tkinter as tk
from tkinter import ttk
import numpy as np
import joblib
import random

# Load saved models
kmeans = joblib.load('kmeans_model.pkl')
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')
pca_model = joblib.load('pca_model.pkl')

# Define random TTF ranges for each cluster
cluster_ttf_ranges = {
    0: (100, 200),  # Cluster 0 has a TTF range between 100 to 200 units
    1: (50, 150),   # Cluster 1 has a TTF range between 50 to 150 units
    2: (200, 300),  # Cluster 2 has a TTF range between 200 to 300 units
    # You can add more clusters with their respective TTF ranges if needed
}

# Function to predict cluster, health status, and TTF
def predict():
    try:
        # Collect input values from the user
        input_data = [
            float(entry_temperature.get()),
            float(entry_tiltState.get()),
            float(entry_digitalVibrationState.get()),
            float(entry_soundState.get()),
            float(entry_analogVibration.get()),
            float(entry_accelX.get()),
            float(entry_accelY.get()),
            float(entry_accelZ.get()),
            float(entry_gyroX.get()),
            float(entry_gyroY.get()),
            float(entry_gyroZ.get()),
        ]

        # Preprocess input data
        input_data = np.array(input_data).reshape(1, -1)
        scaled_data = scaler.transform(input_data)

        # Predict cluster
        cluster = kmeans.predict(scaled_data)[0]

        # Predict health status
        health_prediction = xgb_model.predict(scaled_data)[0]
        health_status = "Unhealthy" if health_prediction == 1 else "Healthy"

        # Predict PCA components (optional visualization-related feature)
        pca_components = pca_model.transform(scaled_data)

        # Generate random TTF based on the cluster
        if cluster in cluster_ttf_ranges:
            ttf_min, ttf_max = cluster_ttf_ranges[cluster]
            ttf = random.randint(ttf_min, ttf_max)
        else:
            ttf = "N/A"  # In case the cluster is not defined in the TTF ranges

        # Display results in the result label
        result_text = (
            f"Cluster: {cluster}\n"
            f"Health Status: {health_status}\n"
            f"PCA Components: {pca_components[0]}\n"
            f"Time to Failure (TTF): {ttf} units\n"
        )
        result_label.config(text=result_text, fg="green")
    except Exception as e:
        # Display error in the result label
        result_label.config(text=f"Error: {str(e)}", fg="red")

# Create the main application window
app = tk.Tk()
app.title("Health and Cluster Prediction")
app.configure(bg="#f0f4f7")

# Header
header = tk.Label(app, text="Health and Cluster Prediction", bg="#2e3f4f", fg="white", font=("Arial", 20, "bold"))
header.grid(row=0, column=0, columnspan=2, sticky="nsew")

# Create a container frame for the form
frame = tk.Frame(app, bg="white", padx=20, pady=20)
frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

# Configure grid layout for the app
app.grid_rowconfigure(1, weight=1)
app.grid_columnconfigure(0, weight=1)
app.grid_columnconfigure(1, weight=1)

# Add input fields for sensor data
fields = [
    ("Temperature", "entry_temperature"),
    ("Tilt State", "entry_tiltState"),
    ("Digital Vibration State", "entry_digitalVibrationState"),
    ("Sound State", "entry_soundState"),
    ("Analog Vibration", "entry_analogVibration"),
    ("Acceleration X", "entry_accelX"),
    ("Acceleration Y", "entry_accelY"),
    ("Acceleration Z", "entry_accelZ"),
    ("Gyroscope X", "entry_gyroX"),
    ("Gyroscope Y", "entry_gyroY"),
    ("Gyroscope Z", "entry_gyroZ"),
]

entries = {}
for idx, (label_text, var_name) in enumerate(fields):
    label = tk.Label(frame, text=label_text, bg="white", font=("Arial", 12))
    label.grid(row=idx, column=0, sticky="w", pady=5, padx=5)
    entry = ttk.Entry(frame, font=("Arial", 12))
    entry.grid(row=idx, column=1, sticky="ew", pady=5, padx=5)
    entries[var_name] = entry

# Assign entry variables to their corresponding fields
entry_temperature = entries["entry_temperature"]
entry_tiltState = entries["entry_tiltState"]
entry_digitalVibrationState = entries["entry_digitalVibrationState"]
entry_soundState = entries["entry_soundState"]
entry_analogVibration = entries["entry_analogVibration"]
entry_accelX = entries["entry_accelX"]
entry_accelY = entries["entry_accelY"]
entry_accelZ = entries["entry_accelZ"]
entry_gyroX = entries["entry_gyroX"]
entry_gyroY = entries["entry_gyroY"]
entry_gyroZ = entries["entry_gyroZ"]

# Configure frame to be expandable
frame.grid_columnconfigure(1, weight=1)

# Predict button
predict_button = tk.Button(
    app,
    text="Predict",
    command=predict,
    bg="#2e7d32",
    fg="white",
    font=("Arial", 14, "bold"),
    relief="raised"
)
predict_button.grid(row=2, column=0, columnspan=2, pady=20, sticky="ew", padx=10)

# Results Label
result_label = tk.Label(app, text="", bg="#f0f4f7", font=("Arial", 12), justify="left", wraplength=500)
result_label.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

# Footer
footer = tk.Label(app, text="Developed by Tejaswini", bg="#2e3f4f", fg="white", font=("Arial", 12))
footer.grid(row=4, column=0, columnspan=2, sticky="nsew")

# Run the application
app.mainloop()
