import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from src.data_loader import load_data_from_yfinance, load_data_from_csv
from src.model import train_predict_prophet, train_predict_rf

# Configuration de la page
st.set_page_config(
    page_title="MASI & Stock Market Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Application de Prédiction Boursière — Indice MASI")
st.markdown("Analysez et prédisez la tendance des indices ou actions à l'aide de modèles d'apprentissage automatique.")

# Barres latérales (Sidebar)
st.sidebar.header("⚙️ Configuration")

source_type = st.sidebar.radio(
    "Source des données :",
    ["Yahoo Finance (API)", "Téléverser un CSV (Recommandé pour MASI)"]
)

df = pd.DataFrame()

if source_type == "Yahoo Finance (API)":
    ticker = st.sidebar.text_input("Ticker / Symbole", value="^MASI")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Date début", date.today() - timedelta(days=365*3))
    end_date = col2.date_input("Date fin", date.today())
    
    if st.sidebar.button("Charger les données"):
        df = load_data_from_yfinance(ticker, start_date, end_date)

else:
    uploaded_file = st.sidebar.file_uploader("Choisissez un fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data_from_csv(uploaded_file)

# Section d'analyse et prédiction
if not df.empty:
    st.success(f"Données chargées avec succès ! ({len(df)} enregistrements)")
    
    # Onglets d'affichage
    tab1, tab2, tab3 = st.tabs(["📊 Graphique & Historique", "🤖 Prédiction ML", "📑 Raw Data"])
    
    with tab1:
        st.subheader("Historique des cours")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Prix de Clôture", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Indice / Prix", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        st.subheader("Configuration du modèle de prédiction")
        
        col_algo, col_days = st.columns(2)
        model_choice = col_algo.selectbox("Algorithme", ["Prophet (Meta)", "Random Forest"])
        days_to_predict = col_days.slider("Nombre de jours à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Entraînement du modèle et calcul en cours..."):
                if model_choice == "Prophet (Meta)":
                    forecast, pred_df = train_predict_prophet(df, days_to_predict)
                    
                    # Graphique Prophet
                    fig_pred = go.Figure()
                    # Historique
                    fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique", line=dict(color='#1C2D42')))
                    # Prédiction
                    fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction", line=dict(color='#FF4B4B', dash='dash')))
                    # Intervalle de confiance
                    fig_pred.add_trace(go.Scatter(
                        x=pd.concat([pred_df['Date'], pred_df['Date'][::-1]]),
                        y=pd.concat([pred_df['Limite_Haute'], pred_df['Limite_Basse'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 75, 75, 0.2)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name="Intervalle de confiance"
                    ))
                    fig_pred.update_layout(xaxis_title="Date", yaxis_title="Indice", hovermode="x unified")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.write("### Prédictions chiffrées")
                    st.dataframe(pred_df.style.highlight_max(axis=0))

                else:
                    pred_df = train_predict_rf(df, days_to_predict)
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique", line=dict(color='#1C2D42')))
                    fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction RF", line=dict(color='#FF4B4B', dash='dash')))
                    fig_pred.update_layout(xaxis_title="Date", yaxis_title="Indice")
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    st.write("### Prédictions chiffrées")
                    st.dataframe(pred_df)

    with tab3:
        st.subheader("Données brutes")
        st.dataframe(df)

else:
    st.info("Veuillez charger des données via la barre latérale pour commencer l'analyse.")
