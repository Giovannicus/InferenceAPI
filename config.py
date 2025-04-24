# config.py - File di configurazione per controllare i tempi
TRAINING_SCHEDULE = {
    "interval": "minutely",  # Opzioni: "minutely", "hourly", "daily", "weekly"
    "time": "02:00",      # Per daily/weekly
    "minutes": .2,        # Per minutely
    "weekday": 1,         # Per weekly (0=lunedì, 6=domenica)
    "max_duration": 3600  # Tempo massimo in secondi per l'addestramento
}

INFERENCE_SCHEDULE = {
    "interval": "minutely",
    "minutes": .1,        # Per minutely/hourly (esegui al minuto 15 di ogni ora)
    "max_duration": 300   # Tempo massimo in secondi per l'inferenza
}