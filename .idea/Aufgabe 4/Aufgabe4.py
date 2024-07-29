import numpy as np
from scipy.stats import t

# Daten der Stichprobe
sample_weights = np.array([884, 947, 860, 889, 897, 849, 852, 868, 882, 899])

# Parameter der Stichprobe
sample_mean = np.mean(sample_weights)
sample_variance = np.var(sample_weights, ddof=1)
n = len(sample_weights)

# Hypothetischer Mittelwert
mu_0 = 895

# Signifikanzniveau
alpha = 0.05

# Teststatistik berechnen
t_stat = (sample_mean - mu_0) / np.sqrt(sample_variance / n)

# Kritischen t-Wert berechnen
t_crit = t.ppf(1 - alpha, df=n-1)

# Ergebnis des Tests
if t_stat > t_crit:
    result = "Lehne die Nullhypothese ab. Das neue System erzeugt höhere Gewichte."
else:
    result = "Kann die Nullhypothese nicht ablehnen. Kein ausreichender Beweis, dass das neue System höhere Gewichte erzeugt."

# Ergebnisse ausgeben
print(f"Stichprobenmittelwert: {sample_mean:.2f} g")
print(f"Stichprobenvarianz: {sample_variance:.2f} g^2")
print(f"t-Statistik: {t_stat:.2f}")
print(f"Kritischer t-Wert: {t_crit:.2f}")
print(result)
