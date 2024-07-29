import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

# Erstellen der Designmatrix (Vandermonde-Matrix) für ein polynomiales Modell bis zum Grad 12
X = np.vander(x, N=13, increasing=True)

# Umwandeln in DataFrame für die Visualisierung
X_df = pd.DataFrame(X, columns=[f'x^{i}' for i in range(13)])

# Visualisierung der Designmatrix
plt.figure(figsize=(16, 10))
sns.heatmap(X_df, annot=True, cmap='viridis', cbar=False, fmt='.1f', linewidths=0.5, annot_kws={"size": 6})
plt.title('Designmatrix (Vandermonde-Matrix) für polynomiales Modell bis Grad 12')
plt.xlabel('Feature')
plt.ylabel('Datenpunktindex')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()
