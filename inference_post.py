import requests

url = "http://127.0.0.1:8000/inference"

data = {
    "sepal_length": 2.1,
    "sepal_width": 1.5,
    "petal_length": 4.4,
    "petal_width": 0.2
}

response = requests.post(url,json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Errore: {response.json()}")
