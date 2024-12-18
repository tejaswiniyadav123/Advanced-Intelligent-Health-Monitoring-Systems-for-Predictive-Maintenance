import numpy as np
import matplotlib.pyplot as plt

# Simulated data
time = np.linspace(0, 100, 100)  # Time points
degradation = 100 - 0.5 * time + np.random.normal(0, 2, size=len(time))  # Degradation curve with noise
failure_threshold = 50

# Plot
plt.figure(figsize=(10, 6))
plt.plot(time, degradation, label='Degradation Curve', color='blue')
plt.axhline(y=failure_threshold, color='red', linestyle='--', label='Failure Threshold')
plt.scatter([80], [50], color='green', label='Predicted Failure Point', zorder=5)  # Example failure point
plt.title('Time-to-Failure Prediction')
plt.xlabel('Time (Days)')
plt.ylabel('Health Metric')
plt.legend()
plt.grid(True)
plt.show()
