import plotly.graph_objects as go
from scipy import stats
import numpy as np

# Gegebene Daten
n = 10
weights = [884, 947, 860, 889, 897, 849, 852, 868, 882, 899]
mu_old = 895
sigma_old = 21.4
alpha = 0.05

# Berechnungen
mean_new = np.mean(weights)
std_new = np.std(weights, ddof=1)
z_statistic = (mean_new - mu_old) / (sigma_old / np.sqrt(n))
critical_value = stats.norm.ppf(1 - alpha)
p_value = 1 - stats.norm.cdf(z_statistic)

# Diagramm erstellen
fig = go.Figure()

# Verteilung der z-Statistik hinzufügen
x = np.linspace(-4, 4, 1000)
y = stats.norm.pdf(x)
fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='z-Verteilung'))

# Kritischen Wert markieren
fig.add_trace(go.Scatter(x=[critical_value, critical_value], y=[0, max(y)], mode='lines+text',
                         line=dict(color='red', dash='dash'),
                         name=f'Kritischer Wert: {critical_value:.2f}',
                         text=[f'{critical_value:.2f}'], textposition='top right'))

# Teststatistik markieren
fig.add_trace(go.Scatter(x=[z_statistic, z_statistic], y=[0, stats.norm.pdf(z_statistic)], mode='lines+text',
                         line=dict(color='blue', dash='dash'),
                         name=f'Teststatistik: {z_statistic:.2f}',
                         text=[f'{z_statistic:.2f}'], textposition='bottom right'))

# Ablehnungsbereich schattieren
shade_x = np.linspace(critical_value, 4, 100)
shade_y = stats.norm.pdf(shade_x)
fig.add_trace(go.Scatter(x=np.concatenate(([critical_value], shade_x, [4])),
                         y=np.concatenate(([0], shade_y, [0])),
                         fill='toself', fillcolor='rgba(255, 0, 0, 0.3)',
                         line=dict(color='rgba(255, 0, 0, 0)'),
                         name='Ablehnungsbereich'))

# Diagrammanpassungen
fig.update_layout(title='Rechtsseitiger z-Test mit schattiertem Ablehnungsbereich',
                  xaxis_title='z-Wert',
                  yaxis_title='Dichte',
                  showlegend=True)

fig.show()
