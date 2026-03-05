import json
import requests

# Send GET request
r = requests.get("http://127.0.0.1:8000")

# Print status code
print("GET request returned status code:", r.status_code)

# Print welcome message
print("Response:", r.json())


data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send POST request
r = requests.post("http://127.0.0.1:8000/data/", data=json.dumps(data))

# Print status code
print("POST request returned status code:", r.status_code)

# Print prediction result
print("Prediction:", r.json())