# Usa un'immagine Python come base
FROM python:3.10-slim

# Imposta la directory di lavoro
WORKDIR /app

# Evita la creazione di file __pycache__
ENV PYTHONDONTWRITEBYTECODE=1 
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema necessarie
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia i file dei requisiti e installa le dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto del codice dell'applicazione
COPY . .

# Crea le directory per i dati se non esistono
RUN mkdir -p data/train data/test

# RIMUOVI QUESTE RIGHE
# RUN python prepare_data.py
# RUN python train.py

# Esposizione della porta per MLflow UI
EXPOSE 5000

# Comando predefinito - avvia l'MLflow UI
CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000"]