import pytest
import requests
import time

# Helper function to check if the API is ready
def is_api_ready(url, retries=5, delay=2):
    for _ in range(retries):
        try:
            response = requests.get(url)  # Use GET for a simple readiness check
            if response.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    return False

def test_predict_endpoint():
    # Wait for the API to be ready
    if not is_api_ready("http://localhost:5000/predict"):
        pytest.fail("API did not become ready in time")

    # Test case 1: Valid input
    data = {"Proporción de mezcla": 0.6, "Temperatura (°C)": 25}
    response = requests.post("http://localhost:5000/predict", json=data)
    assert response.status_code == 200
    assert "Resistencia (MPa)" in response.json()

    # Test case 2: Missing input
    data = {"Proporción de mezcla": 0.6}
    response = requests.post("http://localhost:5000/predict", json=data)
    assert response.status_code == 400
    assert "error" in response.json()

    # Test case 3: Model not trained
    # We need to stop the gunicorn app in order to test this case
    # This is an issue because we are trying to do the tests inside the build
    # This is not a good practice
    # data = {}
    # response = requests.post("http://localhost:5000/predict", json=data)
    # assert response.status_code == 500
    # assert "error" in response.json()

