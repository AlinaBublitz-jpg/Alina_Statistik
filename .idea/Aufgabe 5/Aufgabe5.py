import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import Ridge

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

# OLS-Schätzung der Parameter
ols_params, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
print("\nOLS-Schätzung der Parameter:")
print("OLS-Parameter (α):\n", ols_params)
print("Residuals:", residuals)
print("Rank der Matrix:", rank)
print("Singularwerte der Matrix:", s)

# Ridge-Regression mit erhöhtem alpha-Wert, um die Ill-conditioned Warnung zu mindern
ridge_regressor = Ridge(alpha=10.0)
ridge_regressor.fit(X, y)
ridge_params = ridge_regressor.coef_
print("\nRidge-Regression:")
print("Ridge-Parameter (α):\n", ridge_params)

# Vorhersagen für Visualisierung
x_vals = np.linspace(min(x), max(x), 400)
X_vals = np.vander(x_vals, N=13, increasing=True)
y_ols = X_vals @ ols_params
y_ridge = X_vals @ ridge_params

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
    'Ridge': ridge_params
})
print("\n6. Anzeige der Parameter:")
print(param_df)

fig.show()





import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten
data = [
    (-3, -2981677.63), (6, -13217884866.6), (-9, -1664871776185.78), (-15, -740699415007428.1),
    (19, -12670131062880296), (11, -18391792157894.01), (-17, 3436253692128093.5), (-6, -12903479700.16),
    (14, -355513790324347.25), (-11, -19228985223794.45), (-16, -1646734858787233), (0, 6.89),
    (-5, -1404087971.99), (16, -1788484224926809), (-10, -5995411025987.84), (10, -6336617671840.71),
    (18, -7050311156535014), (5, -1528271348.97), (3, -3398036.45), (15, -790671445328298.9),
    (17, -3515346465556009), (-14, -337398232460715.06), (-7, -81014769228.06),
    (20, -24250183499670304), (-2, -17016.65), (7, -88720567608.48)
]

# Umwandeln in ein DataFrame
df = pd.DataFrame(data, columns=['x', 'y'])

# Designmatrix für polynomiales Modell bis zum Grad 12
X = np.vander(df['x'], 13, increasing=True)
X_df = pd.DataFrame(X, columns=[f'x^{i}' for i in range(13)])

# Berechnung der Korrelationsmatrix
correlation_matrix = X_df.corr()

# Plotten der Korrelationsmatrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Korrelationsmatrix der Designmatrix')
plt.show()

# Interpretation der Korrelationsmatrix
print("Interpretation der Korrelationsmatrix:")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr = correlation_matrix.iloc[i, j]
        if abs(corr) > 0.8:
            print(f"Hohe Korrelation zwischen {correlation_matrix.columns[i]} und {correlation_matrix.columns[j]}: {corr:.2f}")
        elif 0.5 < abs(corr) <= 0.8:
            print(f"Moderate Korrelation zwischen {correlation_matrix.columns[i]} und {correlation_matrix.columns[j]}: {corr:.2f}")
        else:
            print(f"Geringe oder keine Korrelation zwischen {correlation_matrix.columns[i]} und {correlation_matrix.columns[j]}: {corr:.2f}")

# Allgemeine Interpretation
print("\nAllgemeine Interpretation:")
print("Die Korrelationsmatrix zeigt die paarweise Korrelation zwischen den Variablen der Designmatrix.")
print("Hohe absolute Korrelationswerte (über 0.8) weisen auf starke Multikollinearität hin.")
print("Moderate absolute Korrelationswerte (zwischen 0.5 und 0.8) weisen auf moderate Multikollinearität hin.")
print("Geringe absolute Korrelationswerte (unter 0.5) weisen auf keine oder geringe Multikollinearität hin.")



import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Daten
data = [
    (-3, -2981677.63), (6, -13217884866.6), (-9, -1664871776185.78), (-15, -740699415007428.1),
    (19, -12670131062880296), (11, -18391792157894.01), (-17, 3436253692128093.5), (-6, -12903479700.16),
    (14, -355513790324347.25), (-11, -19228985223794.45), (-16, -1646734858787233), (0, 6.89),
    (-5, -1404087971.99), (16, -1788484224926809), (-10, -5995411025987.84), (10, -6336617671840.71),
    (18, -7050311156535014), (5, -1528271348.97), (3, -3398036.45), (15, -790671445328298.9),
    (17, -3515346465556009), (-14, -337398232460715.06), (-7, -81014769228.06),
    (20, -24250183499670304), (-2, -17016.65), (7, -88720567608.48)
]

# Umwandeln in ein DataFrame
df = pd.DataFrame(data, columns=['x', 'y'])

# Designmatrix für polynomiales Modell bis zum Grad 12
X = np.vander(df['x'], 13, increasing=True)

# Berechnung des VIF für jede Variable
vif_data = pd.DataFrame()
vif_data["Feature"] = [f'x^{i}' for i in range(13)]
vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]

# Ausgabe der VIF-Werte und Interpretation
print(vif_data)
print("\nInterpretation der VIF-Werte:")

for i in range(len(vif_data)):
    feature = vif_data.loc[i, "Feature"]
    vif = vif_data.loc[i, "VIF"]

    if vif < 5:
        interpretation = "keine oder geringe Multikollinearität"
    elif 5 <= vif < 10:
        interpretation = "moderate Multikollinearität"
    else:
        interpretation = "hohe Multikollinearität"

    print(f"Feature {feature}: VIF = {vif:.2f} -> {interpretation}")

# Interpretation der VIF-Werte in der Konsole
print("\nAllgemeine Interpretation:")
print("VIF-Werte geben an, wie stark die Varianz eines Regressionskoeffizienten durch die Multikollinearität beeinflusst wird.")
print("Ein VIF-Wert von 1 bedeutet keine Korrelation zwischen dieser unabhängigen Variable und den anderen.")
print("VIF-Werte zwischen 1 und 5 deuten auf keine bis geringe Multikollinearität hin.")
print("VIF-Werte zwischen 5 und 10 weisen auf moderate Multikollinearität hin.")
print("VIF-Werte über 10 deuten auf hohe Multikollinearität hin und sollten genauer untersucht werden.")


# Ausgabe der Designmatrix in der Konsole
print("Designmatrix Ridge (Vandermonde-Matrix) für polynomiales Modell bis Grad 12:")
print(X_df)