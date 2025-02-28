#!pip install pandas scikit-learn numpy matplotlib seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the data
try:
    # If you run the code from the same folder as the file
    df = pd.read_csv('datos.txt')
except FileNotFoundError:
    # If you run it from another folder, we take the path
    df = pd.read_csv('/media/sf_2025-Hack2Progress/proyecto/datos.txt')

# 2. Data Cleaning and Preprocessing
df = df.dropna()  # Drop rows with missing values

# Convert 'Fecha de prueba' to datetime and extract month
df['Fecha de prueba'] = pd.to_datetime(df['Fecha de prueba'])
df['Mes'] = df['Fecha de prueba'].dt.month

# 3. Feature Engineering and Selection
# We'll use 'Proporción de mezcla', 'Temperatura (°C)', 'Edad (días)', 'Mes' as features
features = ['Proporción de mezcla', 'Temperatura (°C)', 'Edad (días)', 'Mes']
X = df[features]
y = df['Resistencia (MPa)']  # Target variable

# 4. Data Exploration: Histograms of Original Data
plt.figure(figsize=(15, 10))

# Histograms for Features
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Histogram of {feature}')

# Histogram for Target Variable
plt.subplot(2, 3, len(features) + 1)
sns.histplot(df['Resistencia (MPa)'], kde=True)
plt.title('Histogram of Resistencia (MPa)')

plt.tight_layout()
plt.show()

# 5. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train the Random Forest Regressor Model
model = RandomForestRegressor(random_state=42)  # You can adjust hyperparameters here
model.fit(X_train, y_train)

# 7. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 8. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

# 9. Histograms of Predictions
plt.figure(figsize=(8, 6))
sns.histplot(y_pred, kde=True, color='red', label='Predictions')
sns.histplot(y_test, kde=True, color='blue', label='Actual')
plt.title('Histogram of Predictions vs Actual')
plt.xlabel('Resistencia (MPa)')
plt.ylabel('Frequency')
plt.legend()
plt.show()
#10. Now you can use it for predictions
new_data = pd.DataFrame({
    'Proporción de mezcla': [0.6, 0.45],
    'Temperatura (°C)': [25, 20],
    'Edad (días)': [28, 28],
    'Mes': [2, 3]
})
predictions = model.predict(new_data)
print(predictions)
