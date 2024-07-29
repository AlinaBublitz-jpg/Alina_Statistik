import plotly.graph_objects as go

# Wahrscheinlichkeiten
p_dafuer = 0.62
p_dagegen = 1 - p_dafuer

# Prozentsätze
prozent_dafuer = p_dafuer * 100
prozent_dagegen = p_dagegen * 100

# Labels für das Diagramm
labels = ['Dafür', 'Dagegen']
values = [prozent_dafuer, prozent_dagegen]

# Diagramm erstellen
fig = go.Figure(data=[go.Bar(
    x=labels,
    y=values,
    text=[f'{v:.1f}%' for v in values],
    textposition='auto',
    marker_color=['blue', 'red']  # Farben für die Balken
)])

# Layout anpassen
fig.update_layout(
    title='Ergebnis der Abstimmung: "Dafür" oder "Dagegen"',
    xaxis_title='Abstimmung',
    yaxis_title='Prozent',
    yaxis=dict(range=[0, 100]),  # Y-Achse auf 0-100% skalieren
    font=dict(
        family='Arial, sans-serif',
        size=14,
        color='black'
    ),
    legend_title='Ergebnisse',
    legend=dict(
        x=0.85,
        y=1.15,
        traceorder='normal',
        font=dict(
            family='Arial, sans-serif',
            size=12,
            color='black'
        ),
        bgcolor='LightSteelBlue',
        bordercolor='Black',
        borderwidth=2
    )
)

# Maßstab hinzufügen
fig.add_shape(
    type="line",
    x0=-0.4, y0=50, x1=1.4, y1=50,
    line=dict(color="Black", width=2, dash="dash"),
)
fig.add_annotation(
    x=1.5, y=50,
    text="50%",
    showarrow=False,
    yshift=10,
    font=dict(
        family='Arial, sans-serif',
        size=12,
        color='black'
    )
)

fig.show()
