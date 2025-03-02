import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_absolute_error

# Load Dataset
data = pd.read_csv("asteroid.csv")

# Preview Data
print(data.head())

# Rename columns for consistency
data.rename(columns={
    'estimated_diameter_max': 'diameter_km',
    'relative_velocity': 'relative_velocity_km_h',
    'miss_distance': 'miss_distance_km',
    'is_hazardous': 'hazardous'
}, inplace=True)

# Convert velocity to km/s (original is in km/h)
data['relative_velocity_km_s'] = data['relative_velocity_km_h'] / 3600

# Estimate Impact Energy (Simplified Model: Kinetic Energy = 0.5 * mass * velocity^2)
data['impact_energy_megatons'] = 0.5 * (data['diameter_km'] ** 3) * (data['relative_velocity_km_s'] ** 2)

# Select Relevant Features
features = ['diameter_km', 'relative_velocity_km_s', 'miss_distance_km', 'absolute_magnitude']
target_class = 'hazardous'  # Binary classification (1 = Hazardous, 0 = Non-hazardous)
target_reg = 'impact_energy_megatons'  # Regression target (Impact Severity)

# Handle Missing Values
data = data.dropna(subset=features + [target_class, target_reg])

# Split Data for Classification (Impact Probability)
X_train, X_test, y_train, y_test = train_test_split(data[features], data[target_class], test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Deep Learning Model for Classification
clf_model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

clf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
clf_model.fit(X_train_scaled, y_train, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate Classification Model
y_pred_class = (clf_model.predict(X_test_scaled) > 0.5).astype(int)
print("Impact Probability Accuracy:", accuracy_score(y_test, y_pred_class))

# Split Data for Regression (Impact Severity)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(data[features], data[target_reg], test_size=0.2, random_state=42)
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

# Deep Learning Model for Regression
reg_model = Sequential([
    Dense(64, activation='relu', input_shape=(len(features),)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

reg_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
reg_model.fit(X_train_reg_scaled, y_train_reg, epochs=50, batch_size=16, validation_split=0.2, verbose=1)

# Evaluate Regression Model
y_pred_reg = reg_model.predict(X_test_reg_scaled)
print("Impact Severity MAE:", mean_absolute_error(y_test_reg, y_pred_reg))

# Monte Carlo Simulation (Statistical Risk Assessment)
simulations = 10000
impact_probabilities = np.random.uniform(0, 1, simulations)  # Simulating asteroid impact chances
severities = np.random.normal(np.mean(data[target_reg]), np.std(data[target_reg]), simulations)
expected_damage = np.mean(impact_probabilities * severities)
print(f"Expected Annual Impact Damage: {expected_damage:.2f} Megatons")

# Visualization
plt.figure(figsize=(10,5))
sns.histplot(severities, bins=30, kde=True, color='red')
plt.title("Distribution of Simulated Impact Severities")
plt.xlabel("Impact Energy (Megatons)")
plt.ylabel("Frequency")
plt.show()