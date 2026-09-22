import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configuration de l'affichage Streamlit
st.set_page_config(
    page_title="Test Plotly - MASI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Visualisation Plotly - Indice MASI")
st.write("Ce graphique confirme le chargement correct du module Plotly.")

# 1. Génération de données historiques pour le test
dates = pd.date_range(start="2025-01-01", periods=100, freq="B")
np.random.seed(42)
variations = np.random.normal(loc=0.0002, scale=0.005, size=len(dates))
points = 13000 * np.exp(np.cumsum(variations))

df = pd.DataFrame({"Date": dates, "MASI": points})

# 2. Construction de la figure Plotly
fig = go.Figure()

# Ajout de la courbe principale
fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["MASI"],
        mode="lines",
        name="Indice MASI",
        line=dict(color="#003366", width=2.5)
    )
)

# Configuration du layout
fig.update_layout(
    title="Évolution récente du MASI",
    xaxis_title="Date",
    yaxis_title="Points",
    template="plotly_white",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=40)
)

# 3. Affichage dans Streamlit
st.plotly_chart(fig, use_container_width=True)

# 4. Affichage d'une métrique simple
dernier_cours = df["MASI"].iloc[-1]
variation_totale = ((df["MASI"].iloc[-1] / df["MASI"].iloc[0]) - 1) * 100

col1, col2 = st.columns(2)
col1.metric("Dernier niveau MASI", f"{dernier_cours:,.2f} pts")
col2.metric("Performance globale", f"{variation_totale:+.2f} %")
