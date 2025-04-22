# Usa Python 3.12 come immagine base
FROM python:3.12-slim

# Imposta la directory di lavoro
WORKDIR /app

# Copia i file dei requisiti
COPY requirements.txt .

# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutti i file nel container
COPY . .

# Esponi la porta su cui gira l'API
EXPOSE 8000

# Comando per avviare l'applicazione
CMD ["python", "inference_ml.py"]