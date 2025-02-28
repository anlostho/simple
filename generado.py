import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 1. Load the data
try:
    #If you run the code from the same folder as the file
    df = pd.read_csv('datos.txt')
except FileNotFoundError:
    #If you run it from another folder, we take the path
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

# 4. Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train the Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# 6. Make Predictions on the Test Set
y_pred = model.predict(X_test)

# 7. Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2): {r2}")

#8. Now you can use it for predictions
new_data = pd.DataFrame({
    'Proporción de mezcla': [0.6, 0.45],
    'Temperatura (°C)': [25, 20],
    'Edad (días)': [28, 28],
    'Mes': [2, 3]
})
predictions = model.predict(new_data)
print(predictions)
