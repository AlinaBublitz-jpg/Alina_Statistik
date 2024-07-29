import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
# Definiere die Variable
y = sp.symbols('y')

# Definiere die Verteilungsfunktion F(y)
F_y = 1 - sp.exp(-8 * y**2)

# Berechne die Wahrscheinlichkeit P(2 <= Y <= 4)
P_2_le_Y_le_4 = F_y.subs(y, 4) - F_y.subs(y, 2)

# Ergebnis auswerten
probability = P_2_le_Y_le_4.evalf()
print(probability)


import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Definiere die Variable
y = sp.symbols('y')

# Definiere die Verteilungsfunktion F(y)
F_y = 1 - sp.exp(-8 * y**2)

# Berechne die Wahrscheinlichkeitsdichtefunktion (PDF) als Ableitung der Verteilungsfunktion
PDF_y = sp.diff(F_y, y)

# Ausgabe der Funktionsgleichung der PDF in der Konsole
PDF_y_simplified = sp.simplify(PDF_y)
print(f'Wahrscheinlichkeitsdichtefunktion (PDF) f(y): {PDF_y_simplified}')

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y) für die numerische Berechnung
def f_y(y):
    return 16 * y * np.exp(-8 * y**2)

# Erstelle eine Reihe von y-Werten
y_values = np.linspace(0, 2, 400)

# Berechne die Wahrscheinlichkeitsdichtefunktion für diese y-Werte
PDF_values = f_y(y_values)

# Erstelle das Diagramm
plt.figure(figsize=(12, 8))
plt.plot(y_values, PDF_values, label='Wahrscheinlichkeitsdichtefunktion', color='blue', linewidth=2)

# Achsen beschriften
plt.xlabel('Zeit (Stunden)', fontsize=14)
plt.ylabel('Dichte', fontsize=14)
plt.title('Wahrscheinlichkeitsdichtefunktion der Zeit, um eine Eule zu hören (in Stunden)', fontsize=16)

# Gitternetz hinzufügen
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legende und Achsenlinien hinzufügen
plt.legend(fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Zusätzliche Einstellungen für die Achsen
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Diagramm anzeigen
plt.show()



import numpy as np
import matplotlib.pyplot as plt

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y) in Stunden
def f_y_hours(y):
    return 16 * y * np.exp(-8 * y**2)

# Konvertiere die Funktion zu Minuten
def f_y_minutes(m):
    y = m / 60  # Konvertiere Minuten zu Stunden
    return f_y_hours(y) / 60  # Skaliere die Dichtefunktion für Minuten

# Erstelle eine Reihe von Minuten-Werten
minutes_values = np.linspace(0, 60, 400)

# Berechne die Wahrscheinlichkeitsdichtefunktion für diese Minuten-Werte
PDF_minutes_values = f_y_minutes(minutes_values)

# Histogramm erstellen
plt.figure(figsize=(12, 8))
plt.hist(minutes_values, bins=60, weights=PDF_minutes_values, color='blue', alpha=0.7, edgecolor='black', linewidth=1.2, label='Wahrscheinlichkeitsdichtefunktion')

# Achsen beschriften
plt.xlabel('Zeit (Minuten)', fontsize=14)
plt.ylabel('Dichte', fontsize=14)
plt.title('Histogramm, um eine Eule zu hören (in Minuten)', fontsize=16)

# Gitternetz hinzufügen
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legende und Achsenlinien hinzufügen
plt.legend(fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Zusätzliche Einstellungen für die Achsen
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Diagramm anzeigen
plt.show()


import scipy.integrate as integrate
import numpy as np

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
def f_y(y):
    return 16 * y * np.exp(-8 * y**2)

# Definiere die Funktion, die das Produkt aus y und der Dichtefunktion berechnet
def mean_function(y):
    return y * f_y(y)

# Berechne das Integral von 0 bis ∞
mean_value, error = integrate.quad(mean_function, 0, np.inf)

# Runde den Mittelwert auf 6 Nachkommastellen
mean_value_rounded = round(mean_value, 6)

# Gebe den Mittelwert in der Konsole aus
print(f'Mittelwert der Wahrscheinlichkeitsdichtefunktion: {mean_value_rounded:.6f}')





import scipy.integrate as integrate
import numpy as np

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
def f_y(y):
    return 16 * y * np.exp(-8 * y**2)

# Definiere die Funktion, die das Produkt aus y^2 und der Dichtefunktion berechnet
def expectation_function(y):
    return y**2 * f_y(y)

# Berechne das Integral von 0 bis ∞
expectation_value, error = integrate.quad(expectation_function, 0, np.inf)

# Runde den Erwartungswert auf 6 Nachkommastellen
expectation_value_rounded = round(expectation_value, 6)

# Gebe den Erwartungswert in der Konsole aus
print(f'Erwartungswert der Wahrscheinlichkeitsdichtefunktion: {expectation_value_rounded:.6f}')


import scipy.integrate as integrate
import numpy as np

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
def f_y(y):
    return 16 * y * np.exp(-8 * y**2)

# Definiere die Funktion, die das Produkt aus y und der Dichtefunktion berechnet
def mean_function(y):
    return y * f_y(y)

# Definiere die Funktion, die das Produkt aus y^2 und der Dichtefunktion berechnet
def expectation_function(y):
    return y**2 * f_y(y)

# Berechne den Erwartungswert E(Y) als Integral von 0 bis ∞
mean_value, mean_error = integrate.quad(mean_function, 0, np.inf)

# Berechne den Erwartungswert E(Y^2) als Integral von 0 bis ∞
expectation_value, expectation_error = integrate.quad(expectation_function, 0, np.inf)

# Berechne die Varianz
variance_value = expectation_value - (mean_value**2)

# Runde die Varianz auf 6 Nachkommastellen
variance_value_rounded = round(variance_value, 6)

# Gebe die Varianz in der Konsole aus
print(f'Varianz der Wahrscheinlichkeitsdichtefunktion: {variance_value_rounded:.6f}')



import scipy.optimize as optimize
import numpy as np

# Definiere die Verteilungsfunktion F(y)
def F_y(y):
    return 1 - np.exp(-8 * y**2)

# Definiere die Inverse der Verteilungsfunktion
def inverse_F_y(q):
    return optimize.newton(lambda y: F_y(y) - q, 0.1)

# Berechne die Quartile
q1 = inverse_F_y(0.25)
q2 = inverse_F_y(0.5)
q3 = inverse_F_y(0.75)

# Runde die Quartile auf 6 Nachkommastellen
q1_rounded = round(q1, 6)
q2_rounded = round(q2, 6)
q3_rounded = round(q3, 6)

# Gebe die Quartile in der Konsole aus
print(f'1. Quartil der Wartezeiten: {q1_rounded:.6f}')
print(f'2. Quartil (Median) der Wartezeiten: {q2_rounded:.6f}')
print(f'3. Quartil der Wartezeiten: {q3_rounded:.6f}')








import matplotlib.pyplot as plt

# Berechne die Standardabweichung
std_deviation = np.sqrt(variance_value)

# Berechne den Mittelwert und die Standardabweichung
mean_value = mean_value_rounded
std_deviation_rounded = round(std_deviation, 6)

# Erstelle eine Reihe von y-Werten
y_values = np.linspace(0, 1, 400)

# Berechne die Wahrscheinlichkeitsdichtefunktion für diese y-Werte
PDF_values = [f_y(y) for y in y_values]

# Erstelle das Diagramm
plt.figure(figsize=(12, 8))
plt.plot(y_values, PDF_values, label='Wahrscheinlichkeitsdichtefunktion', color='blue', linewidth=2)

# Zeichne den Mittelwert ein
plt.axvline(mean_value, color='red', linestyle='--', linewidth=2, label=f'Mittelwert = {mean_value:.6f}')

# Zeichne die Standardabweichungen ein
plt.axvline(mean_value - std_deviation_rounded, color='green', linestyle='--', linewidth=2, label=f'Standardabweichung = {mean_value - std_deviation_rounded:.6f}')
plt.axvline(mean_value + std_deviation_rounded, color='green', linestyle='--', linewidth=2, label=f'Standardabweichung = {mean_value + std_deviation_rounded:.6f}')

# Achsen beschriften
plt.xlabel('Zeit (Stunden)', fontsize=14)
plt.ylabel('Dichte', fontsize=14)
plt.title('Wahrscheinlichkeitsdichtefunktion mit Mittelwert und Standardabweichung (in Stunden)', fontsize=16)

# Gitternetz hinzufügen
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legende und Achsenlinien hinzufügen
plt.legend(fontsize=12, loc='upper right')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Zusätzliche Einstellungen für die Achsen
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Diagramm anzeigen
plt.tight_layout()
plt.show()




# Erstelle das Diagramm
plt.figure(figsize=(12, 8))
plt.plot(y_values, PDF_values, color='black', linewidth=2)

# Fülle die Bereiche zwischen den Quartilen
plt.fill_between(y_values, PDF_values, where=(y_values <= q1), color='skyblue', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=((y_values > q1) & (y_values <= q2)), color='lightgreen', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=((y_values > q2) & (y_values <= q3)), color='lightgreen', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=(y_values > q3), color='skyblue', alpha=0.5)

# Zeichne die Quartile ein
plt.axvline(q1, color='blue', linestyle='--', linewidth=2, label=f'1. Quartil = {q1:.6f}')
plt.axvline(q2, color='red', linestyle='--', linewidth=2, label=f'2. Quartil (Median) = {q2:.6f}')
plt.axvline(q3, color='green', linestyle='--', linewidth=2, label=f'3. Quartil = {q3:.6f}')

# Achsen beschriften
plt.xlabel('Zeit (Stunden)', fontsize=14)
plt.ylabel('Dichte', fontsize=14)
plt.title('Wahrscheinlichkeitsdichtefunktion mit Quartilen der Wartezeiten (in Stunden)', fontsize=16)

# Beschriftung der Quartile weiter rechts platzieren
plt.text(q1 + 0.03, max(PDF_values)*0.9, 'Q1', horizontalalignment='center', fontsize=12, color='blue')
plt.text(q2 + 0.03, max(PDF_values)*0.9, 'Q2', horizontalalignment='center', fontsize=12, color='red')
plt.text(q3 + 0.03, max(PDF_values)*0.9, 'Q3', horizontalalignment='center', fontsize=12, color='green')

# Gitternetz hinzufügen
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legende und Achsenlinien hinzufügen
plt.legend(fontsize=12, loc='upper right')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Zusätzliche Einstellungen für die Achsen
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Diagramm anzeigen
plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Berechne die Quartile erneut für die Genauigkeit
q1 = round(inverse_F_y(0.25), 6)
q2 = round(inverse_F_y(0.5), 6)
q3 = round(inverse_F_y(0.75), 6)

# Berechne die Standardabweichung
std_deviation = np.sqrt(variance_value)
std_deviation_rounded = round(std_deviation, 2)  # Runde auf 2 Nachkommastellen

# Mittelwert aus vorheriger Berechnung
mean_value = mean_value_rounded

# Erstelle das Diagramm
plt.figure(figsize=(12, 8))
plt.plot(y_values, PDF_values, color='black', linewidth=2)

# Fülle die Bereiche zwischen den Quartilen
plt.fill_between(y_values, PDF_values, where=(y_values <= q1), color='skyblue', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=((y_values > q1) & (y_values <= q2)), color='lightgreen', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=((y_values > q2) & (y_values <= q3)), color='lightgreen', alpha=0.5)
plt.fill_between(y_values, PDF_values, where=(y_values > q3), color='skyblue', alpha=0.5)

# Zeichne die Quartile ein
plt.axvline(q1, color='blue', linestyle='--', linewidth=2, label=f'1. Quartil = {q1:.6f}')
plt.axvline(q2, color='red', linestyle='--', linewidth=2, label=f'2. Quartil (Median) = {q2:.6f}')
plt.axvline(q3, color='green', linestyle='--', linewidth=2, label=f'3. Quartil = {q3:.6f}')

# Zeichne den Mittelwert ein
plt.axvline(mean_value, color='purple', linestyle='-', linewidth=2, label=f'Mittelwert = {mean_value:.6f}')

# Zeichne die Standardabweichungen ein
plt.axvline(mean_value - std_deviation_rounded, color='orange', linestyle='--', linewidth=2, label=f'-1 Std. Abweichung = {mean_value - std_deviation_rounded:.6f}')
plt.axvline(mean_value + std_deviation_rounded, color='orange', linestyle='--', linewidth=2, label=f'+1 Std. Abweichung = {mean_value + std_deviation_rounded:.6f}')

# Achsen beschriften
plt.xlabel('Zeit (Stunden)', fontsize=14)
plt.ylabel('Dichte', fontsize=14)
plt.title('Wahrscheinlichkeitsdichtefunktion mit Quartilen, Mittelwert und Standardabweichung (in Stunden)', fontsize=16)

# Beschriftung der Quartile weiter rechts platzieren
plt.text(q1 + 0.03, max(PDF_values)*0.9, 'Q1', horizontalalignment='center', fontsize=12, color='blue')
plt.text(q2 + 0.03, max(PDF_values)*0.9, 'Q2', horizontalalignment='center', fontsize=12, color='red')
plt.text(q3 + 0.03, max(PDF_values)*0.9, 'Q3', horizontalalignment='center', fontsize=12, color='green')

# Gitternetz hinzufügen
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Legende und Achsenlinien hinzufügen
plt.legend(fontsize=12, loc='upper right')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

# Zusätzliche Einstellungen für die Achsen
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Diagramm anzeigen
plt.tight_layout()
plt.show()




