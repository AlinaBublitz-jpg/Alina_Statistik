# Gegebene Parameter
xi_17 <- 46
xi_18 <- 97
xi_19 <- 36.5

# Posterior-Parameter berechnen
alpha_post <- 10 * 3 + xi_17
beta_post <- 10 * xi_19 + xi_18

# Posterior-Verteilung
cat("Posterior-Verteilung von θ: Gamma(", alpha_post, ",", beta_post, ")\n")

# Erwartungswert der Posterior-Verteilung (Bayes-Schätzung mit quadratischer Verlustfunktion)
theta_mean <- alpha_post / beta_post
cat("Bayes-Punkt-Schätzung von θ (Erwartungswert der Posterior-Verteilung):", theta_mean, "\n")

# Modus der Posterior-Verteilung
theta_mode <- (alpha_post - 1) / beta_post
cat("Bayes-Punkt-Schätzung von θ (Modus der Posterior-Verteilung):", theta_mode, "\n")
