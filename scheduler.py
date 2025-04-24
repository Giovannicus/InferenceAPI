# schedule.py - Versione avanzata con controllo dei tempi
import time
import subprocess
import schedule
import logging
import signal
from datetime import datetime
import importlib.util
import os

# Carica dinamicamente la configurazione (consente aggiornamenti senza riavvio)
def load_config():
    spec = importlib.util.spec_from_file_location("config", "./config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.FileHandler("mlflow_schedule.log"), logging.StreamHandler()]
)

def run_with_timeout(cmd, timeout):
    """Esegue un comando con timeout e gestisce interruzioni"""
    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        duration = time.time() - start_time
        return {
            "returncode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "duration": duration
        }
    except subprocess.TimeoutExpired:
        process.kill()
        logging.warning(f"Processo terminato per timeout dopo {timeout} secondi")
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": "Timeout",
            "duration": timeout
        }

# Nelle funzioni run_training e run_prediction
def run_training():
    """Esegue l'addestramento"""
    config = load_config()
    logging.info("Avvio addestramento...")
    
    # Se siamo in un container Docker, esegui direttamente
    result = run_with_timeout(
        ["python", "train.py"],  # Rimuovi 'docker exec iris-mlflow-container'
        config.TRAINING_SCHEDULE["max_duration"]
    )

    logging.info(f"Completato con codice: {result['returncode']}")
    logging.info(f"Output: {result['stdout']}")
    if result['stderr']:
        logging.error(f"Errori: {result['stderr']}")
    
    # Salva statistiche di esecuzione
    with open("training_stats.csv", "a") as f:
        f.write(f"{datetime.now()},{result['returncode']},{result['duration']}\n")

def run_prediction():
    """Esegue le predizioni con controllo del tempo"""
    config = load_config()
    logging.info("Esecuzione predizioni...")
    
    # Esegui direttamente lo script Python
    result = run_with_timeout(
        ["python", "predict.py"],  # Comando diretto senza Docker
        config.INFERENCE_SCHEDULE["max_duration"]
    )
    
    logging.info(f"Predizioni completate in {result['duration']:.2f} secondi")
    logging.info(f"Codice: {result['returncode']}")
    logging.info(f"Output: {result['stdout']}")
    if result['stderr']:
        logging.error(f"Errori: {result['stderr']}")
    
    # Salva statistiche di esecuzione
    with open("inference_stats.csv", "a") as f:
        f.write(f"{datetime.now()},{result['returncode']},{result['duration']}\n")

def setup_schedule():
    """Configura la schedulazione in base alle impostazioni"""
    config = load_config()
    
    # Configura l'addestramento
    if config.TRAINING_SCHEDULE["interval"] == "minutely":
        schedule.every(config.TRAINING_SCHEDULE["minutes"]).minutes.do(run_training)
    elif config.TRAINING_SCHEDULE["interval"] == "hourly":
        schedule.every().hour.at(f":{config.TRAINING_SCHEDULE['minutes']}").do(run_training)
    elif config.TRAINING_SCHEDULE["interval"] == "daily":
        schedule.every().day.at(config.TRAINING_SCHEDULE["time"]).do(run_training)
    elif config.TRAINING_SCHEDULE["interval"] == "weekly":
        getattr(schedule.every(), ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"][config.TRAINING_SCHEDULE["weekday"]]).at(config.TRAINING_SCHEDULE["time"]).do(run_training)
    
    # Configura l'inferenza
    if config.INFERENCE_SCHEDULE["interval"] == "minutely":
        schedule.every(config.INFERENCE_SCHEDULE["minutes"]).minutes.do(run_prediction)
    elif config.INFERENCE_SCHEDULE["interval"] == "hourly":
        schedule.every().hour.at(f":{config.INFERENCE_SCHEDULE['minutes']}").do(run_prediction)
    
    logging.info("Schedulazione configurata")

# Gestisce il ricaricamento della configurazione
def reload_config(signum, frame):
    logging.info("Ricaricamento configurazione...")
    schedule.clear()
    setup_schedule()
    logging.info("Configurazione ricaricata")

if __name__ == "__main__":
    # Registra handler per ricaricare la configurazione
    signal.signal(signal.SIGHUP, reload_config)
    
    # Configura la schedulazione iniziale
    setup_schedule()
    
    # Esegui immediatamente se richiesto (tramite flag)
    if os.environ.get("RUN_IMMEDIATELY", "false").lower() == "true":
        run_training()
        run_prediction()
    
    logging.info("Schedulatore avviatoooooooooooo...")
    
    # Loop principale
    try:
        while True:
            schedule.run_pending()
            time.sleep(10)  # Controlla ogni 10 secondi
    except KeyboardInterrupt:
        logging.info("Schedulatore fermato dall'utente")