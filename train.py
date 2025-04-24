import argparse
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os

############################################# MLflow Tracking: Sperimentazione con diversi modelli
print("Im trainiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiing")
# Parsing degli argomenti da riga di comando
parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--train_data", type=str, default="data/train/train.csv")
parser.add_argument("--valid_data", type=str, default="data/valid/valid.csv")
args = parser.parse_args()

# Impostazioni dell'esperimento
mlflow.set_experiment("Iris Classification")

# 1. Carica il dataset Iris incorporato
iris = load_iris()
X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
y_iris = pd.Series(iris.target)

# 2. Carica i dati aggiuntivi da file CSV (se esistono)
train_data_additional = None
if os.path.exists(args.train_data):
    train_data_additional = pd.read_csv(args.train_data)
    # Assumiamo che il file CSV abbia la stessa struttura del dataset Iris
    X_train_additional = train_data_additional.drop('target', axis=1)
    y_train_additional = train_data_additional['target']
    
     # Converti le etichette in numeri se sono stringhe
    if y_train_additional.dtype == 'object':
        # Crea un mapping dalle stringhe ai numeri
        label_map = {
            'setosa': 0,
            'versicolor': 1,
            'virginica': 2
        }
        # Applica il mapping
        y_train_additional = y_train_additional.map(lambda x: label_map.get(x, x))
        # Converti in intero
        y_train_additional = y_train_additional.astype(int)

    # Combina i dati di scikit-learn con i dati aggiuntivi
    X_train = pd.concat([X_iris, X_train_additional], axis = 0, ignore_index=True)
    y_train = pd.concat([y_iris, y_train_additional], axis = 0, ignore_index=True)
else:
    # Se non ci sono dati aggiuntivi, usa solo il dataset di scikit-learn
    X_train = X_iris
    y_train = y_iris
    print("File di training aggiuntivo non trovato. Uso solo il dataset Iris standard.")

# 3. Carica i dati di valid dal file (obbligatorio)
if os.path.exists(args.valid_data):
    test_data = pd.read_csv(args.valid_data)
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']

        # Converti le etichette testuali in numeri se necessario
    if y_test.dtype == 'object':
        label_map = {
            'setosa': 0,
            'versicolor': 1, 
            'virginica': 2
        }
        y_test = y_test.map(lambda x: label_map.get(x, x))
        y_test = y_test.astype(int)
else:
    raise FileNotFoundError(f"Il file di test {args.valid_data} non è stato trovato!")

# Nomi delle feature (colonne)
feature_names = X_train.columns.tolist()

# Avvia un nuovo run di MLflow
with mlflow.start_run(run_name=f"RF-trees{args.n_estimators}-depth{args.max_depth}"):
    # Log dei parametri di esecuzione
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("train_data_path", args.train_data)
    mlflow.log_param("test_data_path", args.valid_data)
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("valid_size", len(X_test))
    
    # MLFLOW TRACKING: Addestramento e tracciamento
    # Addestra il modello
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        max_depth=args.max_depth, 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Valutazione del modello
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Log delle metriche
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall) 
    mlflow.log_metric("f1_score", f1)
    
    # Crea e salva figura feature importan ce
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, model.feature_importances_)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    
    # MLFLOW MODELS: Salva il modello
    mlflow.sklearn.log_model(
        model, 
        "iris_model",
        signature=mlflow.models.signature.infer_signature(X_train, y_train)
    )
    
    print(f"Modello addestrato con accuracy: {accuracy:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    run_id = mlflow.active_run().info.run_id

# MLFLOW REGISTRY: Registra il modello nel Model Registry
client = MlflowClient()

# Registra o recupera il modello nel Registry
try:
    client.create_registered_model("iris-classifier")
    print("Nuovo modello registrato: iris-classifier")
except:
    print("Il modello iris-classifier esiste già nel Registry")

# Aggiungi una nuova versione del modello
model_version = client.create_model_version(
    name="iris-classifier",
    source=f"runs:/{run_id}/iris_model",
    run_id=run_id
)

# Sposta la versione del modello in "Staging"
client.transition_model_version_stage(
    name="iris-classifier",
    version=model_version.version,
    stage="Staging"
)

print(f"Modello registrato come versione {model_version.version} in stato 'Staging'")
print("Per visualizzare i risultati, esegui: mlflow ui --host 0.0.0.0 --port 5000")