import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score

# Gegebene Datenpunkte
data_points = [
    (-3, -2981677.63), (6, -13217884866.6), (-9, -1664871776185.78),
    (-15, -740699415007428.1), (19, -12670131062880296),
    (11, -18391792157894.01), (-17, 3436253692128093.5),
    (-6, -12903479700.16), (14, -355513790324347.25),
    (-11, -19228985223794.45), (-16, -1646734858787233),
    (0, 6.89), (-5, -1404087971.99), (16, -1788484224926809),
    (-10, -5995411025987.84), (10, -6336617671840.71),
    (18, -7050311156535014), (5, -1528271348.97),
    (3, -3398036.45), (15, -790671445328298.9),
    (17, -3515346465556009), (-14, -337398232460715.06),
    (-7, -81014769228.06), (20, -24250183499670304),
    (-2, -17016.65), (7, -88720567608.48)
]

# Extrahiere x und y Werte
x = np.array([point[0] for point in data_points])
y = np.array([point[1] for point in data_points])

# Design-Matrix erstellen (Vandermonde-Matrix)
X = np.vander(x, N=13, increasing=True)

# Lambda-Werte, die untersucht werden sollen
alphas = np.linspace(0.01, 110, 100)  # 100 Werte zwischen 0.01 und 110

# Ridge-Regression mit Kreuzvalidierung
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X, y)
best_alpha = ridge_cv.alpha_

print("\nBest Alpha (Lambda) from RidgeCV:")
print(best_alpha)

# Ridge-Regression mit dem besten Alpha-Wert
ridge_regressor = Ridge(alpha=best_alpha)
ridge_regressor.fit(X, y)
y_pred_ridge = ridge_regressor.predict(X)

# Berechnung des MSE für den besten Alpha-Wert
mse_best_alpha = mean_squared_error(y, y_pred_ridge)
print(f"\nMean Squared Error (MSE) for Best Alpha {best_alpha}: {mse_best_alpha:.2f}")

# OLS-Schätzung der Parameter
ols_params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
y_pred_ols = X @ ols_params

# Berechnung des MSE für OLS
mse_ols = mean_squared_error(y, y_pred_ols)
print(f"\nMean Squared Error (MSE) for OLS: {mse_ols:.2e}")

# Berechnung von Bestimmtheitsmaß (R^2) und MSE für Ridge
r2_ridge = r2_score(y, y_pred_ridge)

# Berechnung von Bestimmtheitsmaß (R^2) für OLS
r2_ols = r2_score(y, y_pred_ols)

print("\nVergleich der Modelle:")
print(f"OLS: R^2 = {r2_ols:.4f}, MSE = {mse_ols:.2e}")
print(f"Ridge: R^2 = {r2_ridge:.4f}, MSE = {mse_best_alpha:.2e}")

# Vorhersagen für Visualisierung
x_vals = np.linspace(min(x), max(x), 400)
X_vals = np.vander(x_vals, N=13, increasing=True)
y_ols = X_vals @ ols_params
y_ridge = ridge_regressor.predict(X_vals)

# Visualisierung
fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Original Data'))
fig.add_trace(go.Scatter(x=x_vals, y=y_ols, mode='lines', name='OLS Model'))
fig.add_trace(go.Scatter(x=x_vals, y=y_ridge, mode='lines', name='Ridge Model', line=dict(dash='dash')))

fig.update_layout(title='OLS vs Ridge Regression',
                  xaxis_title='x',
                  yaxis_title='f(x)',
                  legend=dict(x=0.1, y=0.9))

# OLS und Ridge Regression Parameter anzeigen
param_df = pd.DataFrame({
    'Parameter': [f'α{i}' for i in range(13)],
    'OLS': ols_params,
    'Ridge': ridge_regressor.coef_
})
print("\nAnzeige der Parameter:")
print(param_df)

# Visualisierung des MSE
fig_mse = go.Figure()
fig_mse.add_trace(go.Bar(name='OLS MSE', x=['OLS'], y=[mse_ols], text=[f'{mse_ols:.2e}'], textposition='auto'))
fig_mse.add_trace(go.Bar(name='Ridge MSE', x=['Ridge'], y=[mse_best_alpha], text=[f'{mse_best_alpha:.2e}'], textposition='auto'))

fig_mse.update_layout(title='MSE Vergleich: OLS vs Ridge Regression',
                      xaxis_title='Modell',
                      yaxis_title='MSE',
                      barmode='group')

# Visualisierung des R^2
fig_r2 = go.Figure()
fig_r2.add_trace(go.Bar(name='OLS R²', x=['OLS'], y=[r2_ols], text=[f'{r2_ols:.4f}'], textposition='auto', marker_color='indianred'))
fig_r2.add_trace(go.Bar(name='Ridge R²', x=['Ridge'], y=[r2_ridge], text=[f'{r2_ridge:.4f}'], textposition='auto', marker_color='lightsalmon'))

fig_r2.update_layout(title='R² Vergleich: OLS vs Ridge Regression',
                     xaxis_title='Modell',
                     yaxis_title='R²',
                     barmode='group')

fig_mse.show()
fig_r2.show()
