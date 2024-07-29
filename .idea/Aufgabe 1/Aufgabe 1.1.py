import matplotlib.pyplot as plt
import numpy as np

# Parameter der Bernoulli-Verteilung
p = 0.62

# Erstellen der Bernoulli-Verteilung
x = np.array([0, 1])
y = np.array([1 - p, p])

# Plotten der Verteilung
plt.bar(x, y, tick_label=['Dagegen', 'Dafür'])
plt.xlabel('Ergebnis')
plt.ylabel('Wahrscheinlichkeit')
plt.title('Bernoulli-Verteilung (p = 0.62)')
plt.ylim(0, 1)

# Erwartungswert als vertikale Linie
plt.axvline(x=p, color='r', linestyle='--', label=f'Erwartungswert = {p}')
plt.legend()

plt.show()
