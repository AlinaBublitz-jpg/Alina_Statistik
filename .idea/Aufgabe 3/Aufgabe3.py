import numpy as np
import plotly.graph_objects as go
from scipy.special import gamma
from scipy.optimize import minimize

# Parameter
theta = 1  # Beispielparameter

# Annahmen über die Zufallsvariablen
assumptions = [
    "Annahme Zufallsvariablen:",
    "1. Unabhängigkeit: S1 und S2 sind unabhängig voneinander.",
    "2. Identische Verteilung: Beide Zufallsvariablen folgen der gleichen Exponentialverteilung mit dem Parameter θ.",
    "3. Summenbildung: T ist die Summe der beiden unabhängig exponentialverteilten Zufallsvariablen S1 und S2."
]

# Ausgabe der Annahmen in der Konsole
for assumption in assumptions:
    print(assumption)

# Dichtefunktion der Gamma-Verteilung (k=2, theta)
def gamma_pdf(t, k, theta):
    return (t**(k-1) * np.exp(-t/theta)) / (theta**k * gamma(k))

# Wertebereich für t
t_values = np.linspace(0, 10, 400)
k = 2  # Parameter k für die Gamma-Verteilung

# Berechnung der Dichtefunktion
pdf_values = gamma_pdf(t_values, k, theta)

# Plot mit Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(x=t_values, y=pdf_values, mode='lines', name='Dichtefunktion f_T(t)'))

fig.update_layout(title='Dichtefunktion der Gesamtbandbreite T',
                  xaxis_title='T (Gesamtbandbreite bis zum Ausfall)',
                  yaxis_title='Dichte f_T(t)',
                  showlegend=True)

fig.show()

# Erklärung der Dichtefunktion in der Konsole
print("\nBerechnung der Dichtefunktion der Gesamtbandbreite T für verschiedene Werte von T:")
for t, pdf in zip(t_values, pdf_values):
    print(f'T: {t:.2f}, f_T(t): {pdf:.4f}')

# Beispiel-Stichprobe der Gesamtbandbreiten bis zum Ausfall
sample = np.array([3.5, 4.2, 5.1, 6.3, 7.4])

# Likelihood-Funktion für theta
def likelihood(theta, sample):
    n = len(sample)
    product_term = np.prod(sample)
    sum_term = np.sum(sample)
    return (1 / theta**2)**n * product_term * np.exp(-sum_term / theta)

# Log-Likelihood-Funktion für theta
def log_likelihood(theta, sample):
    n = len(sample)
    sum_log_term = np.sum(np.log(sample[sample > 0]))  # Vermeidung von log(0)
    sum_term = np.sum(sample)
    return -2 * n * np.log(theta) + sum_log_term - sum_term / theta

# Berechnung der Likelihood und Log-Likelihood für verschiedene Werte von theta
theta_values = np.linspace(0.1, 5, 100)
likelihood_values = [likelihood(theta, sample) for theta in theta_values]
log_likelihood_values = [log_likelihood(theta, sample) for theta in theta_values]

# Erklärung der Likelihood- und Log-Likelihood-Berechnung in der Konsole
print("\nBerechnung der Likelihood-Funktion für verschiedene Werte von Theta basierend auf der Stichprobe:")
print(f"Stichprobe: {sample}")
for theta_val, likelihood_val, log_likelihood_val in zip(theta_values, likelihood_values, log_likelihood_values):
    print(f'Theta: {theta_val:.2f}, Likelihood: {likelihood_val:.4f}, Log-Likelihood: {log_likelihood_val:.4f}')

print("\nZusätzliche Berechnung der Log-Likelihood wird durchgeführt, um numerische Stabilität zu gewährleisten und Überlaufprobleme zu vermeiden.\n"
      "Die Log-Likelihood ermöglicht eine genauere Berechnung, insbesondere bei sehr kleinen oder großen Werten der Likelihood.\n"
      "Um die tatsächliche Likelihood zu erhalten, kann das Ergebnis der Log-Likelihood exponentiell umgeformt werden.")

# Plot der Likelihood- und Log-Likelihood-Funktion
fig = go.Figure()

fig.add_trace(go.Scatter(x=theta_values, y=likelihood_values, mode='lines', name='Likelihood-Funktion'))
fig.add_trace(go.Scatter(x=theta_values, y=np.exp(log_likelihood_values), mode='lines', name='Exp(Log-Likelihood)', line=dict(dash='dash')))

fig.update_layout(title='Likelihood- und Log-Likelihood-Funktion für Theta',
                  xaxis_title='Theta',
                  yaxis_title='Likelihood',
                  showlegend=True)

fig.show()

# Erweiterung: Maximum-Likelihood-Schätzung und Erwartungswert
# Tatsächliche Stichprobe aus dem Experiment
actual_sample = np.array([0, 1, 2, 23, 10])

# Negative Log-Likelihood für Minimierung
def neg_log_likelihood(theta, sample):
    return -log_likelihood(theta, sample)

# Schätzung von theta mit Maximum-Likelihood
result = minimize(neg_log_likelihood, x0=1, args=(actual_sample,), bounds=[(1e-5, None)])
theta_mle = result.x[0]

# Berechnung des Erwartungswerts für die Gesamtbandbreite T
expected_value = k * theta_mle

# Ausgabe der Ergebnisse in der Konsole
print(f"\nMaximum-Likelihood-Schätzung für Theta: {theta_mle:.4f}")
print(f"Erwartungswert für die Gesamtbandbreite T: {expected_value:.4f}")

# Erklärung der Berechnung in der Konsole
print("\nDie Maximum-Likelihood-Schätzung für Theta wurde durch Maximierung der Log-Likelihood-Funktion bestimmt.")
print("Der Erwartungswert für die Gesamtbandbreite T ist das Produkt aus k (Anzahl der Router) und der geschätzten Theta.")

# Plot des Erwartungswerts
fig = go.Figure()
fig.add_trace(go.Scatter(x=[theta_mle], y=[expected_value], mode='markers', name='Erwartungswert', marker=dict(size=10, color='red')))

fig.update_layout(title='Erwartungswert der Gesamtbandbreite bis zum Ausfall',
                  xaxis_title='Theta (MLE)',
                  yaxis_title='Erwartungswert',
                  showlegend=True)

fig.show()
