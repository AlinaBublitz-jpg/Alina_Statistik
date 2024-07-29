import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Erstellen Sie die Abbildung und die Achsen
fig, ax = plt.subplots(figsize=(12, 8))

# Helle Farbe für das Innere der Kreise
light_blue = '#E0F7FA'

# Prior-Kreis
prior_circle = patches.Circle((0.3, 0.75), 0.2, edgecolor='blue', facecolor=light_blue, linewidth=2)
ax.add_patch(prior_circle)
ax.text(0.3, 0.75, 'Prior\n(Vorkenntnisse über θ)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.3, 0.82, 'Prior', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')

# Likelihood-Kreis
likelihood_circle = patches.Circle((0.3, 0.25), 0.2, edgecolor='blue', facecolor=light_blue, linewidth=2)
ax.add_patch(likelihood_circle)
ax.text(0.3, 0.25, 'Likelihood\n(beobachtete\nErgebnisse, θ gegeben)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.3, 0.32, 'Likelihood', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')

# Plus-Zeichen zwischen den Kreisen
ax.text(0.3, 0.5, '+', horizontalalignment='center', verticalalignment='center', fontsize=30)

# Posterior-Kreis
posterior_circle = patches.Circle((0.7, 0.5), 0.2, edgecolor='blue', facecolor=light_blue, linewidth=2)
ax.add_patch(posterior_circle)
ax.text(0.7, 0.5, 'Posterior\n(geupdatetes Wissen)', horizontalalignment='center', verticalalignment='center', fontsize=12)
ax.text(0.7, 0.57, 'Posterior', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold')

# Pfeil von den Kreisen zur Posterior-Verteilung
arrow = patches.FancyArrowPatch((0.45, 0.5), (0.55, 0.5), mutation_scale=20, color='black')
ax.add_patch(arrow)

# Achsen ausblenden
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')  # Sicherstellen, dass die Kreise rund sind
ax.axis('off')

# Titel der Abbildung
plt.title("Wissensgenerierung in der bayesschen Statistik (Quelle: eigene Darstellung)", pad=10)

# Abbildung speichern und anzeigen
plt.savefig('bayesian_statistik.png')
plt.show()
