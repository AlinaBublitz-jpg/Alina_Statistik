import numpy as np
import plotly.graph_objects as go
from scipy.stats import gamma

# Gegebene Parameter
alpha_prior = 46
beta_prior = 97
alpha_likelihood = 3
sample_mean = 36.5
n = 10

# Posterior-Parameter
alpha_posterior = alpha_prior + n * alpha_likelihood
beta_posterior = beta_prior + n * sample_mean

# Bayes-Punktschätzer (Erwartungswert der Posterior-Verteilung)
bayes_estimator_mean = alpha_posterior / beta_posterior

# Bayes-Punktschätzer (Modus der Posterior-Verteilung)
if alpha_posterior > 1:
    bayes_estimator_mode = (alpha_posterior - 1) / beta_posterior
else:
    bayes_estimator_mode = 0  # Modus ist undefiniert für alpha <= 1

# Ausgabe der Posterior-Parameter mit Erklärung
print("Posterior-Verteilung von θ:")
print(f"Alpha (Formparameter) der Posterior-Verteilung: {alpha_posterior}")
print(f"Beta (Skalenparameter) der Posterior-Verteilung: {beta_posterior}")

# Erklärung der Posterior-Verteilung
print("\nDie Posterior-Verteilung von θ ist eine Gamma-Verteilung mit folgenden Parametern:")
print(f"Formparameter (alpha): {alpha_posterior}")
print(f"Skalenparameter (beta): {beta_posterior}")

# Erstellen der Posterior-Verteilung
theta_values = np.linspace(0, 1, 1000)
posterior_pdf = gamma.pdf(theta_values, a=alpha_posterior, scale=1/beta_posterior)

print("\nDie Posterior-Dichtefunktion für verschiedene Werte von θ ist wie folgt:")

# Ausgabe der Dichtewerte für einige θ-Werte
for i in range(0, 1000, 200):
    print(f"θ = {theta_values[i]:.3f}, Posterior-Dichte = {posterior_pdf[i]:.6f}")

# Ausgabe des Bayes-Punktschätzers basierend auf dem Erwartungswert
print("\nBayes-Punktschätzer für θ (Erwartungswert der Posterior-Verteilung):")
print(f"θ_Bayes (Erwartungswert) = {bayes_estimator_mean:.6f}")

# Erklärung des Bayes-Punktschätzers basierend auf dem Erwartungswert
print("\nErklärung des Bayes-Punktschätzers basierend auf dem Erwartungswert:")
print("Der Bayes-Punktschätzer ist der Erwartungswert der Posterior-Verteilung. "
      "Bei einer Gamma-Verteilung Gamma(alpha, beta) ist der Erwartungswert durch das Verhältnis "
      "der Parameter alpha und beta gegeben, d.h. Erwartungswert = alpha / beta.")
print(f"In diesem Fall ist der Bayes-Punktschätzer von θ = {alpha_posterior} / {beta_posterior} = {bayes_estimator_mean:.6f}.")

# Ausgabe des Bayes-Punktschätzers basierend auf dem Modus
print("\nBayes-Punktschätzer für θ (Modus der Posterior-Verteilung):")
print(f"θ_Bayes (Modus) = {bayes_estimator_mode:.6f}")

# Erklärung des Bayes-Punktschätzers basierend auf dem Modus
print("\nErklärung des Bayes-Punktschätzers basierend auf dem Modus:")
print("Der Bayes-Punktschätzer ist der Modus der Posterior-Verteilung. "
      "Bei einer Gamma-Verteilung Gamma(alpha, beta) ist der Modus gegeben durch (alpha - 1) / beta, sofern alpha > 1.")
print(f"In diesem Fall ist der Bayes-Punktschätzer von θ = ({alpha_posterior} - 1) / {beta_posterior} = {bayes_estimator_mode:.6f}, da alpha > 1.")

# Visualisierung der Posterior-Verteilung mit Plotly
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=theta_values,
    y=posterior_pdf,
    mode='lines',
    name='Posterior PDF'
))

fig.update_layout(
    title='Posterior-Verteilung von θ',
    xaxis_title='θ',
    yaxis_title='Dichte',
)

fig.show()



