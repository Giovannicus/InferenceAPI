import os
import mlflow
import pandas as pd
import numpy as np

# Carica il modello dalla registry
model = mlflow.sklearn.load_model("models:/iris-classifier/Staging")

# Mappa nomi delle classi
target_names = ["setosa", "versicolor", "virginica"]

# Funzione per caricare dati di test
def load_test_data(test_path="data/test/test.csv"):
    if os.path.exists(test_path):
        test_data = pd.read_csv(test_path)
        X_test = test_data.drop('target', axis=1)
        y_test = test_data['target']
        return X_test, y_test
    else:
        raise FileNotFoundError(f"Il file di test {test_path} non è stato trovato!")

# Carica dati di test
try:
    X_test, y_true = load_test_data()
    print(f"Dati di test caricati: {len(X_test)} campioni")
    
    # Fai le predizioni sul set di test completo
    predictions = model.predict(X_test)
    predictions_names = [target_names[pred] for pred in predictions]
    # Mostra risultati
    print("\nPredizioni sui dati di test:")
    print(f"Accuracy: {(predictions_names == y_true).mean():.4f}")

    # Mostra alcuni esempi
    n_samples = min(5, len(X_test))
    print(f"\nPrime {n_samples} predizioni:")

    for i in range(n_samples):
        true_class = y_true.iloc[i] # target_names[y_true.iloc[i]] if y_true.iloc[i] < len(target_names) else f"Unknown {y_true.iloc[i]}"
        pred_class = target_names[predictions[i]] if predictions[i] < len(target_names) else f"Unknown {predictions[i]}"
        print(f"Esempio {i+1}: Vero = {true_class}, Predetto = {pred_class}")
        
except FileNotFoundError as e:
    print(e)
    print("\nUsando esempi predefiniti per la dimostrazione...")
    
    # Esempi predefiniti se i dati di test non sono disponibili
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Probabile Setosa
        [6.7, 3.0, 5.2, 2.3],  # Probabile Virginica
        [5.9, 3.0, 4.2, 1.5],  # Probabile Versicolor
    ]
    
    # Converti in DataFrame per coerenza
    test_df = pd.DataFrame(test_samples, columns=['sepal length (cm)', 'sepal width (cm)', 
                                                 'petal length (cm)', 'petal width (cm)'])
    
    # Fai le predizioni
    predictions = model.predict(test_df)
    
    # Mostra risultati
    print("\nPredizioni con esempi predefiniti:")
    for i, prediction in enumerate(predictions):
        print(f"Esempio {i+1}: {test_samples[i]} → {target_names[prediction]}")