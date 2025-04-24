# stats.py - Analisi dei tempi di esecuzione
import pandas as pd
import matplotlib.pyplot as plt

# Carica statistiche
training_stats = pd.read_csv("training_stats.csv", names=["timestamp", "status", "duration"], parse_dates=["timestamp"])
inference_stats = pd.read_csv("inference_stats.csv", names=["timestamp", "status", "duration"], parse_dates=["timestamp"])

# Visualizza statistiche di base
print("=== Statistiche di addestramento ===")
print(f"Tempo medio: {training_stats['duration'].mean():.2f} secondi")
print(f"Tempo massimo: {training_stats['duration'].max():.2f} secondi")
print(f"Tasso di successo: {(training_stats['status'] == 0).mean() * 100:.1f}%")

print("\n=== Statistiche di inferenza ===")
print(f"Tempo medio: {inference_stats['duration'].mean():.2f} secondi")
print(f"Tempo massimo: {inference_stats['duration'].max():.2f} secondi")
print(f"Tasso di successo: {(inference_stats['status'] == 0).mean() * 100:.1f}%")

# Crea grafici dei tempi
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.title("Tempi di addestramento")
plt.plot(training_stats["timestamp"], training_stats["duration"])
plt.ylabel("Secondi")

plt.subplot(2, 1, 2)
plt.title("Tempi di inferenza")
plt.plot(inference_stats["timestamp"], inference_stats["duration"])
plt.ylabel("Secondi")
plt.tight_layout()
plt.savefig("execution_times.png")