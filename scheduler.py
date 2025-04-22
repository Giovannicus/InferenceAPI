import time
import requests
import os
import csv
import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import threading
import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configurazione
API_URL = "http://localhost:8000/inference"  
TRAIN_DIR = "data/train"                    # Cartella per i dati di addestramento
TEST_DIR = "data/test"                      # Cartella per i dati di test
OUTPUT_DIR = "data/output"                  # Cartella per i risultati
MODEL_PATH = "model.pkl"                    # Percorso del modello
TRAIN_INTERVAL = 60                         # Addestra ogni minuto
PREDICT_INTERVAL = 30                       # Predici ogni 30 secondi

# Crea le directory se non esistono
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(os.path.join(TRAIN_DIR, "processed"), exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(os.path.join(TEST_DIR, "processed"), exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_or_create_model():
    """Carica il modello esistente o crea un nuovo modello."""
    if os.path.exists(MODEL_PATH):
        logger.info("Caricamento del modello esistente...")
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        logger.info("Creazione di un nuovo modello...")
        return RandomForestClassifier(n_estimators=10, max_depth=5)

def train_model():
    """Funzione per addestrare il modello con nuovi dati."""
    while True:
        try:
            start_time = time.time()
            logger.info("Controllo nuovi dati di addestramento...")
            
            # Trova i file CSV nella cartella di addestramento
            csv_files = [f for f in os.listdir(TRAIN_DIR) if f.endswith('.csv') and 
                        os.path.isfile(os.path.join(TRAIN_DIR, f))]
            
            if not csv_files:
                logger.info("Nessun nuovo file di addestramento trovato.")
            else:
                logger.info(f"Trovati {len(csv_files)} file di addestramento.")
                
                # Carica il modello esistente
                model = load_or_create_model()
                
                all_data = []
                
                # Elabora ogni file CSV
                for filename in csv_files:
                    file_path = os.path.join(TRAIN_DIR, filename)
                    logger.info(f"Elaborazione di {filename}...")
                    
                    # Leggi il CSV
                    try:
                        df = pd.read_csv(file_path)
                        all_data.append(df)
                        
                        # Sposta il file nella cartella "processed"
                        processed_path = os.path.join(TRAIN_DIR, "processed", filename)
                        os.rename(file_path, processed_path)
                        logger.info(f"File {filename} spostato in processed.")
                    except Exception as e:
                        logger.error(f"Errore nella lettura del file {filename}: {str(e)}")
                
                if all_data:
                    # Combina tutti i dati
                    training_data = pd.concat(all_data, ignore_index=True)
                    
                    if "species" in training_data.columns:
                        # Addestra il modello
                        X = training_data.drop("species", axis=1)
                        y = training_data["species"]
                        
                        model.fit(X, y)
                        
                        # Salva il modello aggiornato
                        with open(MODEL_PATH, "wb") as f:
                            pickle.dump(model, f)
                        
                        accuracy = model.score(X, y)
                        logger.info(f"Modello addestrato e salvato. Accuratezza: {accuracy:.4f}")
                    else:
                        logger.warning("I dati non contengono la colonna 'species'. Addestramento saltato.")
            
            # Calcola il tempo rimanente prima del prossimo addestramento
            elapsed = time.time() - start_time
            sleep_time = max(1, TRAIN_INTERVAL - elapsed)
            logger.info(f"Prossimo addestramento tra {sleep_time:.1f} secondi.")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Errore nel processo di addestramento: {str(e)}")
            time.sleep(TRAIN_INTERVAL)

def predict():
    """Funzione per fare predizioni su nuovi dati."""
    while True:
        try:
            start_time = time.time()
            logger.info("Controllo nuovi dati per predizioni...")
            
            # Trova i file CSV nella cartella di test
            csv_files = [f for f in os.listdir(TEST_DIR) if f.endswith('.csv') and 
                        os.path.isfile(os.path.join(TEST_DIR, f))]
            
            if not csv_files:
                logger.info("Nessun nuovo file di test trovato.")
            else:
                logger.info(f"Trovati {len(csv_files)} file di test.")
                
                # Elabora ogni file CSV
                for filename in csv_files:
                    file_path = os.path.join(TEST_DIR, filename)
                    logger.info(f"Elaborazione di {filename}...")
                    
                    try:
                        # Leggi il CSV
                        with open(file_path, "r") as f:
                            reader = csv.DictReader(f)
                            rows = list(reader)
                        
                        if not rows:
                            logger.warning(f"Il file {filename} non contiene dati.")
                            continue
                        
                        # Prepara il file di output
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_path = os.path.join(OUTPUT_DIR, f"pred_{timestamp}_{filename}")
                        
                        with open(output_path, "w", newline="") as f:
                            # Aggiungi la colonna prediction all'header
                            fieldnames = rows[0].keys()
                            writer = csv.DictWriter(f, fieldnames=list(fieldnames) + ["prediction"])
                            writer.writeheader()
                            
                            # Elabora ogni riga
                            for row in rows:
                                # Prepara i dati per l'API
                                try:
                                    payload = {
                                        "sepal_length": float(row["sepal_length"]),
                                        "sepal_width": float(row["sepal_width"]),
                                        "petal_length": float(row["petal_length"]),
                                        "petal_width": float(row["petal_width"])
                                    }
                                    
                                    # Chiama l'API
                                    response = requests.post(API_URL, json=payload)
                                    prediction = response.json()["prediction"]
                                    
                                    # Aggiungi la predizione alla riga
                                    row["prediction"] = prediction
                                except Exception as e:
                                    logger.error(f"Errore nell'elaborazione della riga: {str(e)}")
                                    row["prediction"] = "error"
                                
                                # Scrivi la riga nel file di output
                                writer.writerow(row)
                        
                        # Sposta il file elaborato in una sottocartella "processed"
                        processed_path = os.path.join(TEST_DIR, "processed", filename)
                        os.rename(file_path, processed_path)
                        
                        logger.info(f"Elaborato {filename}, salvato in {output_path}")
                        
                    except Exception as e:
                        logger.error(f"Errore nell'elaborazione di {filename}: {str(e)}")
            
            # Calcola il tempo rimanente prima della prossima predizione
            elapsed = time.time() - start_time
            sleep_time = max(1, PREDICT_INTERVAL - elapsed)
            logger.info(f"Prossima predizione tra {sleep_time:.1f} secondi.")
            time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Errore nel processo di predizione: {str(e)}")
            time.sleep(PREDICT_INTERVAL)

def main():
    logger.info("Avvio del sistema di addestramento e inferenza automatica")
    
    # Crea thread separati per addestramento e predizione
    train_thread = threading.Thread(target=train_model, daemon=True)
    predict_thread = threading.Thread(target=predict, daemon=True)
    
    # Avvia i thread
    train_thread.start()
    predict_thread.start()
    
    # Mantieni il programma in esecuzione
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Arresto del programma...")

if __name__ == "__main__":
    main()