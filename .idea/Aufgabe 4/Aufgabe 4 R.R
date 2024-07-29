# Definiere die Stichprobe
stichprobe <- c(884, 947, 860, 889, 897, 849, 852, 868, 882, 899)

# Berechne den Stichprobenmittelwert
stichprobenmittelwert <- mean(stichprobe)

# Berechne die Stichprobenvarianz manuell
n <- length(stichprobe)
sum_squared_diff <- sum((stichprobe - stichprobenmittelwert)^2)
stichprobenvarianz <- sum_squared_diff / (n - 1)

# Berechne die Stichprobenstandardabweichung
stichproben_sd <- sqrt(stichprobenvarianz)

# Definiere den Populationsmittelwert unter der Nullhypothese
populationsmittelwert <- 895

# Berechne die Teststatistik t
t_statistik <- (stichprobenmittelwert - populationsmittelwert) / (stichproben_sd / sqrt(n))

# Ausgabe der Ergebnisse
cat("Stichprobenmittelwert:", stichprobenmittelwert, "\n")
cat("Stichprobenvarianz:", stichprobenvarianz, "\n")
cat("Stichprobenstandardabweichung:", stichproben_sd, "\n")
cat("Teststatistik t:", t_statistik, "\n")



# Definiere die Stichprobe
stichprobe <- c(884, 947, 860, 889, 897, 849, 852, 868, 882, 899)

# Berechne die Stichprobengröße
n <- length(stichprobe)

# Definiere die Signifikanzebene
alpha <- 0.05

# Berechne die Freiheitsgrade
df <- n - 1

# Berechne den kritischen Wert für einen einseitigen t-Test
kritischer_wert <- qt(1 - alpha, df)

# Ausgabe des kritischen Werts
cat("Kritischer Wert (einseitig, alpha =", alpha, "):", kritischer_wert, "\n")
