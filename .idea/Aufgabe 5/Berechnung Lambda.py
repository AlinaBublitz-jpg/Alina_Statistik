import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

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
alphas = np.linspace(0.01, 110, 100)  # 100 Werte zwischen 0.01 und 11

# Ridge-Regression mit Kreuzvalidierung
ridge_cv = RidgeCV(alphas=alphas, store_cv_results=True)
ridge_cv.fit(X, y)
best_alpha = ridge_cv.alpha_

print("\nBest Alpha (Lambda) from RidgeCV:")
print(best_alpha)

# Ridge-Regression mit dem besten Alpha-Wert
ridge_regressor = Ridge(alpha=best_alpha)
ridge_regressor.fit(X, y)
y_pred = ridge_regressor.predict(X)

# Berechnung des MSE für den besten Alpha-Wert
mse_best_alpha = mean_squared_error(y, y_pred)
print(f"\nMean Squared Error (MSE) für Best Alpha {best_alpha}: {mse_best_alpha:.2f}")

# Plot der MSE-Werte für alle Alpha-Werte
cv_mse_values = np.mean(ridge_cv.cv_values_, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(alphas, cv_mse_values, marker='o')
plt.axvline(best_alpha, color='r', linestyle='--', label=f'Bestes Lambda: {best_alpha}')
plt.title('MSE für unterschiedliche Lambda-Werte in der Ridge-Regression')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.show()
