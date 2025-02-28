import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

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
features = ['Proporción de mezcla', 'Temperatura (°C)', 'Edad (días)', 'Mes']
X = df[features]
y = df['Resistencia (MPa)']

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 6. Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model trained and saved to model.pkl")
