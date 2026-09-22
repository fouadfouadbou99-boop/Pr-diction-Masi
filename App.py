import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 1. Configuration de la page Streamlit
st.set_page_config(
    page_title="MASI - Prédiction & Analyse",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Analyse et Prédiction du MASI")
st.subheader("Visualisation interactive des cours et modèle de projection simple")

# 2. Génération de données fictives (à remplacer par votre source de données)
@st.cache_data
def load_data():
    dates = pd.date_range(start="2023-01-01", periods=400, freq="B")
    np.random.seed(42)
    # Simulation d'une marche aléatoire autour de 12 000 points
    returns = np.random.normal(loc=0.0003, scale=0.008, size=len(dates))
    price_series = 12000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({"Date": dates, "MASI": price_series})
    return df

df = load_data()

# 3. Barre latérale (Sidebar) - Paramètres
st.sidebar.header("⚙️ Paramètres")
horizon = st.sidebar.slider("Horizon de prédiction (jours ouvrés) :", 5, 60, 20)
trend_factor = st.sidebar.slider("Tendance simulée (%) :", -2.0, 2.0, 0.5) / 100

# 4. Calcul de la projection simple
last_date = df["Date"].iloc[-1]
last_price = df["MASI"].iloc[-1]

future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon, freq="B")
future_prices = [last_price * (1 + trend_factor * (i / horizon)) for i in range(1, horizon + 1)]

df_future = pd.DataFrame({"Date": future_dates, "MASI_Pred": future_prices})

# 5. Graphique Plotly Interactive
fig = go.Figure()

# Courbe historique
fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["MASI"],
    mode="lines",
    name="MASI (Historique)",
    line=dict(color="#1f77b4", width=2)
))

# Courbe de prédiction
fig.add_trace(go.Scatter(
    x=df_future["Date"],
    y=df_future["MASI_Pred"],
    mode="lines+markers",
    name="Projection",
    line=dict(color="#ff7f0e", width=2, dash="dash")
))

# Mise en page du graphique
fig.update_layout(
    title="Évolution de l'indice MASI et Projection",
    xaxis_title="Date",
    yaxis_title="Points MASI",
    template="plotly_white",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

# 6. Métriques clés
col1, col2, col3 = st.columns(3)
col1.metric("Dernier cours MASI", f"{last_price:,.2f} pts")
col2.metric("Objectif projeté", f"{future_prices[-1]:,.2f} pts")
col3.metric("Variation attendue", f"{((future_prices[-1] / last_price) - 1) * 100:+.2f} %")

st.divider()

# 7. Affichage du graphique et du tableau de données
st.plotly_chart(fig, use_container_width=True)

with st.expander("📊 Voir les données historiques récentes"):
    st.dataframe(df.tail(15), use_container_width=True)
