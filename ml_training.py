import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# Funzione per caricare il modello esistente o crearne uno nuovo
def load_or_create_model():
    if os.path.exists("model.pkl"):
        print("Caricamento del modello esistente...")
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    else:
        print("Creazione di un nuovo modello...")
        return RandomForestClassifier(n_estimators=10, max_depth=5)

# Carica il dataset originale di Iris
iris = sns.load_dataset("iris")
X_original = iris.drop("species", axis=1)
y_original = iris["species"]

# Carica i nuovi dati dal file CSV
csv_path = "data/input/train.csv"  # Percorso del file CSV con nuovi dati
if os.path.exists(csv_path):
    print(f"Caricamento nuovi dati da {csv_path}...")
    new_data = pd.read_csv(csv_path)
    
    # Verifica se il CSV ha la colonna "prediction"
    if "prediction" in new_data.columns:
        # Usa i dati e le predizioni come nuovi esempi di addestramento
        X_new = new_data.drop("prediction", axis=1)
        y_new = new_data["prediction"]
        
        # Combina i dati originali con i nuovi dati
        X_combined = pd.concat([X_original, X_new])
        y_combined = pd.concat([y_original, y_new])
        
        print(f"Aggiunto {len(X_new)} nuovi esempi al dataset.")
    else:
        print("Il file CSV non contiene predizioni. Uso solo i dati originali.")
        X_combined = X_original
        y_combined = y_original
else:
    print("Nessun nuovo dato trovato. Uso solo i dati originali.")
    X_combined = X_original
    y_combined = y_original

# Dividi in train e test
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# Carica il modello esistente o creane uno nuovo
model = load_or_create_model()

# Addestra il modello sui dati combinati
model.fit(X_train, y_train)

# Valuta le prestazioni
print(f"Train accuracy: {model.score(X_train, y_train)}")
print(f"Test accuracy: {model.score(X_test, y_test)}")

# Salva il modello aggiornato
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
    print("Modello aggiornato salvato come 'model.pkl'")