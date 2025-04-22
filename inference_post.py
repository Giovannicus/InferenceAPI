import requests

url = "http://127.0.0.1:8000/inference"

data = {
    "sepal_length": 5.1,
    "sepal_width": 3.7,
    "petal_length": 1.4,
    "petal_width": 2.1
}

response = requests.post(url,json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Errore: {response.json()}")
