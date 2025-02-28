from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

app = Flask(__name__)

# Load the trained model (we'll train it once and save it)
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    print("Model file not found. Please train the model first.")
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not trained'}), 500

    try:
        data = request.get_json()
        if 'Proporción de mezcla' not in data or 'Temperatura (°C)' not in data:
            return jsonify({'error': 'Missing required parameters: Proporción de mezcla and Temperatura (°C)'}), 400

        mix_proportion = data['Proporción de mezcla']
        temperature = data['Temperatura (°C)']
        if 'Edad (días)' not in data:
            age = 28
        else:
            age = data['Edad (días)']
        if 'Mes' not in data:
            month = 1
        else:
            month = data['Mes']
        
        new_data = pd.DataFrame({
            'Proporción de mezcla': [mix_proportion],
            'Temperatura (°C)': [temperature],
            'Edad (días)': [age],
            'Mes': [month]
        })

        prediction = model.predict(new_data)
        return jsonify({'Resistencia (MPa)': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
