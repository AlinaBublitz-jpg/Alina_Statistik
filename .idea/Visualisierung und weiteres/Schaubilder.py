import matplotlib.pyplot as plt
import numpy as np

# Daten für die diskrete Zufallsvariable
x_discrete = np.arange(1, 11)
y_discrete = np.ones_like(x_discrete) / 10

# Daten für die stetige Zufallsvariable
x_continuous = np.linspace(0, 10, 400)
y_density = np.exp(-(x_continuous - 5)**2 / 2) / np.sqrt(2 * np.pi)

# Verteilungsfunktion für die stetige Zufallsvariable
y_cdf = np.cumsum(y_density) / np.sum(y_density)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Wahrscheinlichkeitsverteilungen', fontsize=16)

# Diskrete Zufallsvariable - Wahrscheinlichkeitsfunktion
axes[0, 0].stem(x_discrete, y_discrete, linefmt='b-', basefmt=" ", markerfmt='bo')
axes[0, 0].set_title('Wahrscheinlichkeitsfunktion', fontsize=14)
axes[0, 0].set_ylabel('f(x)', fontsize=12)
axes[0, 0].set_xlabel('x', fontsize=12)

# Diskrete Zufallsvariable - Verteilungsfunktion
for i in range(len(x_discrete)):
    axes[1, 0].hlines(np.cumsum(y_discrete)[i], x_discrete[i] - 0.5, x_discrete[i], colors='b')
axes[1, 0].plot(x_discrete - 0.5, np.cumsum(y_discrete), 'bo')
axes[1, 0].set_ylabel('F(x)', fontsize=12)
axes[1, 0].set_xlabel('x', fontsize=12)
axes[1, 0].set_title('Verteilungsfunktion', fontsize=14)

# Stetige Zufallsvariable - Dichtefunktion
axes[0, 1].plot(x_continuous, y_density, 'b-')
axes[0, 1].set_ylabel('f(x)', fontsize=12)
axes[0, 1].set_xlabel('x', fontsize=12)
axes[0, 1].set_title('Dichtefunktion', fontsize=14)

# Stetige Zufallsvariable - Verteilungsfunktion
axes[1, 1].plot(x_continuous, y_cdf, 'b-')
axes[1, 1].set_ylabel('F(x)', fontsize=12)
axes[1, 1].set_xlabel('x', fontsize=12)
axes[1, 1].set_title('Verteilungsfunktion', fontsize=14)
axes[1, 1].axhline(1, color='gray', linestyle='--', linewidth=1)

# Hinzufügen der Beschriftungen "Diskrete Zufallsvariable" und "Stetige Zufallsvariable"
fig.text(0.25, 0.92, 'Diskrete Zufallsvariable', fontsize=14, ha='center')
fig.text(0.75, 0.92, 'Stetige Zufallsvariable', fontsize=14, ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.90])
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Wertebereich für die Normalverteilung
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, 0, 1)

# Bereiche für die Standardabweichungen
sigma1_neg = np.linspace(-1, 0, 500)
sigma1_pos = np.linspace(0, 1, 500)
sigma2_neg = np.linspace(-2, -1, 1000)
sigma2_pos = np.linspace(1, 2, 1000)
sigma3_neg = np.linspace(-3, -2, 1000)
sigma3_pos = np.linspace(2, 3, 1000)
extreme_neg = np.linspace(-4, -3, 1000)
extreme_pos = np.linspace(3, 4, 1000)

plt.figure(figsize=(12, 6))

# Normalverteilung zeichnen
plt.plot(x, y, color='black')

# Bereiche einfärben
plt.fill_between(sigma1_neg, norm.pdf(sigma1_neg), color='#0A74DA', alpha=0.6)
plt.fill_between(sigma1_pos, norm.pdf(sigma1_pos), color='#0A74DA', alpha=0.6)
plt.fill_between(sigma2_neg, norm.pdf(sigma2_neg), color='#1F77B4', alpha=0.6)
plt.fill_between(sigma2_pos, norm.pdf(sigma2_pos), color='#1F77B4', alpha=0.6)
plt.fill_between(sigma3_neg, norm.pdf(sigma3_neg), color='#AEC7E8', alpha=0.6)
plt.fill_between(sigma3_pos, norm.pdf(sigma3_pos), color='#AEC7E8', alpha=0.6)
plt.fill_between(extreme_neg, norm.pdf(extreme_neg), color='#C7D8EA', alpha=0.6)
plt.fill_between(extreme_pos, norm.pdf(extreme_pos), color='#C7D8EA', alpha=0.6)

# Labels hinzufügen
plt.text(-0.5, 0.17, '34.1%', horizontalalignment='center', fontsize=12, color='black')
plt.text(0.5, 0.17, '34.1%', horizontalalignment='center', fontsize=12, color='black')
plt.text(-1.5, 0.05, '13.6%', horizontalalignment='center', fontsize=12, color='black')
plt.text(1.5, 0.05, '13.6%', horizontalalignment='center', fontsize=12, color='black')
plt.text(-2.5, 0.02, '2.1%', horizontalalignment='center', fontsize=12, color='black')
plt.text(2.5, 0.02, '2.1%', horizontalalignment='center', fontsize=12, color='black')
plt.text(-3.5, 0.005, '0.1%', horizontalalignment='center', fontsize=12, color='black')
plt.text(3.5, 0.005, '0.1%', horizontalalignment='center', fontsize=12, color='black')

# Standardabweichungen markieren
plt.axvline(0, color='white', linestyle='--')
plt.axvline(-1, color='white', linestyle='--')
plt.axvline(1, color='white', linestyle='--')
plt.axvline(-2, color='white', linestyle='--')
plt.axvline(2, color='white', linestyle='--')
plt.axvline(-3, color='white', linestyle='--')
plt.axvline(3, color='white', linestyle='--')

# Achsen und Titel
plt.xticks([-3, -2, -1, 0, 1, 2, 3], ['$-3\sigma$', '$-2\sigma$', '$-1\sigma$', '$0$', '$1\sigma$', '$2\sigma$', '$3\sigma$'])
plt.yticks(np.arange(0, 0.5, 0.1))
plt.ylabel('Wahrscheinlichkeitsdichte', fontsize=14)
plt.title('Intervalle um $\\mu$ bei der Normalverteilung', fontsize=16)

plt.tight_layout()
plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Wertebereich für die Chi-Quadrat-Verteilung
df = 10  # Freiheitsgrade
x = np.linspace(0, 30, 1000)
y = chi2.pdf(x, df)

# Kritischer Wert für den rechtseitigen Test
alpha = 0.05
crit_value = chi2.ppf(1 - alpha, df)

plt.figure(figsize=(10, 6))

# Chi-Quadrat-Verteilung zeichnen
plt.plot(x, y, label='$\chi^2$ Verteilung', color='blue')

# Kritische Region einfärben
x_crit = np.linspace(crit_value, 30, 1000)
y_crit = chi2.pdf(x_crit, df)
plt.fill_between(x_crit, y_crit, color='red', alpha=0.5, label='Kritische Region')

# Kritischer Wert markieren
plt.axvline(crit_value, color='red', linestyle='--')
plt.text(crit_value, max(y)*0.4, '$\chi^2_{krit}$', horizontalalignment='center', fontsize=12, color='red')

# Nullhypothese markieren
plt.text(df, max(y)*0.8, '$H_0$', horizontalalignment='center', fontsize=12, color='blue')

# Achsen und Titel
plt.xlabel('$\chi^2$')
plt.ylabel('Wahrscheinlichkeitsdichte', fontsize=14)
plt.title('rechtsseitiger Test', fontsize=16)
plt.legend()

# Hypothesen und alpha beschriften
plt.text(20, max(y)*0.9, '$H_0: \sigma^2 \leq \sigma_0^2$', fontsize=12)
plt.text(20, max(y)*0.85, '$H_1: \sigma^2 > \sigma_0^2$', fontsize=12)
plt.text(crit_value + 2, 0.01, '$\\alpha$', fontsize=12)

plt.tight_layout()
plt.show()





import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, poisson

# Gegebene Parameter
alpha_prior = 46
beta_prior = 97
alpha_likelihood = 3
sample_mean = 36.5
n = 10

# Posterior-Parameter
alpha_posterior = alpha_prior + n * alpha_likelihood
beta_posterior = beta_prior + n * sample_mean

# Bereich für θ
theta = np.linspace(0, 1, 1000)

# Prior-Verteilung
prior = gamma.pdf(theta, a=alpha_prior, scale=1/beta_prior)

# Likelihood-Verteilung (angenommen, wir haben die Summe der Daten x)
lambda_likelihood = alpha_likelihood * n / sample_mean
k = np.arange(0, 10)
likelihood = poisson.pmf(k, lambda_likelihood)

# Posterior-Verteilung
posterior = gamma.pdf(theta, a=alpha_posterior, scale=1/beta_posterior)

# Plotten
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Prior
axes[0].plot(theta, prior, 'b-', lw=2)
axes[0].set_title('Prior')
axes[0].set_xlabel('θ')
axes[0].set_ylabel('Density')
axes[0].grid(True)

# Likelihood
axes[1].stem(k, likelihood, basefmt=" ")
axes[1].set_title('Likelihood')
axes[1].set_xlabel('k')
axes[1].set_ylabel('Probability')
axes[1].grid(True)

# Posterior
axes[2].plot(theta, posterior, 'b-', lw=2)
axes[2].set_title('Posterior')
axes[2].set_xlabel('θ')
axes[2].set_ylabel('Density')
axes[2].grid(True)

plt.tight_layout()
plt.show()
