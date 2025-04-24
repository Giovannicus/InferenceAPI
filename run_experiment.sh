#!/bin/bash
# Salva questo contenuto in un file chiamato run_experiment.sh

# Questo script esegue train.py nel container Docker
# Uso: ./run_experiment.sh <n_estimators> <max_depth>
# Esempio: ./run_experiment.sh 100 10

# Verifica se sono stati forniti i parametri
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Uso: ./run_experiment.sh <n_estimators> <max_depth>"
  exit 1
fi

# Esegui train.py nel container Docker con i parametri forniti
docker exec -it mlflow_mlflow_1 python train.py --n_estimators $1 --max_depth $2


###### DEVI RENDERLO ESEGUIBILE CON 
# chmod +x run_experiment.sh

###### Esegui un esperimento con 200 alberi e profondità massima 15
#./run_experiment.sh 200 15