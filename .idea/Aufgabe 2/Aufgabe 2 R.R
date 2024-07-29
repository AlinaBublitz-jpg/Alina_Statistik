# Definiere die Funktion F(y) = 1 - exp(-8 * y^2)
F_y <- expression(1 - exp(-8 * y^2))

# Berechne die Ableitung von F(y)
f_y <- D(F_y, "y")

# Ausgabe der Ableitung
cat("Die Ableitung von F(y) ist: f(y) =", deparse(f_y), "\n")

# Installiere und lade plotly, falls noch nicht installiert
if (!requireNamespace("plotly", quietly = TRUE)) {
  install.packages("plotly", repos = "https://cloud.r-project.org/")
}
library(plotly)

# Definiere die PDF der Verteilungsfunktion
pdf <- function(y) {
  16 * y * exp(-8 * y^2)
}

# Definiere die Inverse der Verteilungsfunktion
inverse_F <- function(p) {
  sqrt(-log(1 - p) / (8))
}

# Berechne den Erwartungswert (gegeben) und die Varianz
mu <- 0.3133285
E_Y2 <- 0.125
variance <- E_Y2 - mu^2

# Berechne die Quartile
Q1 <- inverse_F(0.25)
Q2 <- inverse_F(0.5)
Q3 <- inverse_F(0.75)

# Erstelle eine Sequenz von y-Werten
y <- seq(0, 1, length.out = 1000)
density <- pdf(y)

# Erstelle einen DataFrame für plotly
data <- data.frame(y = y, density = density)

# Zeichne das interaktive Diagramm mit plotly
p <- plot_ly(data, x = ~y, y = ~density, type = 'scatter', mode = 'lines', line = list(color = 'blue')) %>%
  add_lines(x = c(mu, mu), y = c(0, max(density)), line = list(dash = 'dash', color = 'red'), name = 'Erwartungswert') %>%
  add_lines(x = c(Q1, Q1), y = c(0, max(density)/2), line = list(dash = 'dot', color = 'green'), name = 'Q1') %>%
  add_lines(x = c(Q2, Q2), y = c(0, max(density)/2), line = list(dash = 'dot', color = 'green'), name = 'Median') %>%
  add_lines(x = c(Q3, Q3), y = c(0, max(density)/2), line = list(dash = 'dot', color = 'green'), name = 'Q3') %>%
  layout(title = 'Wahrscheinlichkeitsdichtefunktion mit Erwartungswert und Quartilen',
         xaxis = list(title = 'y'),
         yaxis = list(title = 'Dichte'))

# Zeige das Diagramm an
p

# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
f_y <- function(y) {
  16 * y * exp(-8 * y^2)
}

# Definiere die Funktion, die das Produkt aus y und der Dichtefunktion berechnet
mean_function <- function(y) {
  y * f_y(y)
}

# Berechne das Integral von 0 bis ∞
result <- integrate(mean_function, lower = 0, upper = Inf)

# Ausgabe des Mittelwerts in der Konsole
cat("Mittelwert der Wahrscheinlichkeitsdichtefunktion:", result$value, "\n")


# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
f_y <- function(y) {
  16 * y * exp(-8 * y^2)
}

# Definiere die Funktion, die das Produkt aus y^2 und der Dichtefunktion berechnet
expectation_function <- function(y) {
  y^2 * f_y(y)
}

# Berechne das Integral von 0 bis ∞
result <- integrate(expectation_function, lower = 0, upper = Inf)

# Ausgabe des Erwartungswerts in der Konsole
cat("Erwartungswert der Wahrscheinlichkeitsdichtefunktion:", round(result$value, 6), "\n")


# Definiere die Wahrscheinlichkeitsdichtefunktion f(y)
f_y <- function(y) {
  16 * y * exp(-8 * y^2)
}

# Definiere die Funktion für den Erwartungswert E(Y)
mean_function <- function(y) {
  y * f_y(y)
}

# Definiere die Funktion für den Erwartungswert E(Y^2)
expectation_function <- function(y) {
  y^2 * f_y(y)
}

# Berechne den Erwartungswert E(Y) von 0 bis ∞
mean_result <- integrate(mean_function, lower = 0, upper = Inf)
mean_value <- mean_result$value

# Berechne den Erwartungswert E(Y^2) von 0 bis ∞
expectation_result <- integrate(expectation_function, lower = 0, upper = Inf)
expectation_value <- expectation_result$value

# Berechne die Varianz
variance_value <- expectation_value - mean_value^2

# Ausgabe der Varianz in der Konsole
cat("Varianz der Wahrscheinlichkeitsdichtefunktion:", round(variance_value, 6), "\n")
