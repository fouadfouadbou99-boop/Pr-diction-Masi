import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import date, timedelta

from src.data_loader import scrape_masi_live, load_data_from_yfinance, load_data_from_csv
from src.model import train_predict_prophet, train_predict_rf

# ------------------------------------------------------------------------------
# Configuration de la page Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Prédiction MASI - Bourse de Casablanca",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prédiction de l'indice Boursier MASI")
st.markdown("Application d'analyse et de prédiction pour le marché boursier marocain (Bourse de Casablanca).")

# ------------------------------------------------------------------------------
# Barre latérale (Sidebar) - Configuration & Source des données
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Source des données")

source_type = st.sidebar.radio(
    "Choisissez la méthode d'acquisition :",
    [
        "Scraping MASI en direct (Recommandé)", 
        "Téléverser un CSV (Bourse de Casablanca)", 
        "Yahoo Finance API (ex: ATW.CS, IAM.CS)"
    ]
)

df = pd.DataFrame()

if source_type == "Scraping MASI en direct (Recommandé)":
    st.sidebar.info("Récupère automatiquement les dernières données du MASI depuis le web.")
    if st.sidebar.button("Charger / Rafraîchir les données 🔄"):
        with st.spinner("Scraping du MASI en cours..."):
            df = scrape_masi_live()

elif source_type == "Téléverser un CSV (Bourse de Casablanca)":
    uploaded_file = st.sidebar.file_uploader("Sélectionnez votre fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data_from_csv(uploaded_file)

else:  # Yahoo Finance
    ticker = st.sidebar.text_input("Symbole (ex: ATW.CS pour Attijariwafa, IAM.CS pour Maroc Telecom)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*3))
    end_date = col2.date_input("Fin", date.today())
    
    if st.sidebar.button("Charger depuis Yahoo Finance"):
        with st.spinner("Téléchargement des données..."):
            df = load_data_from_yfinance(ticker, start_date, end_date)

# ------------------------------------------------------------------------------
# Section Principale : Affichage et Prédictions
# ------------------------------------------------------------------------------
if not df.empty:
    st.success(f"Données chargées avec succès ! ({len(df)} enregistrements)")
    
    # Onglets d'organisation
    tab1, tab2, tab3 = st.tabs(["📊 Graphique Historique", "🤖 Modèle de Prédiction", "📑 Données Brutes"])
    
    # --- ONGLET 1 : GRAPHIQUE ---
    with tab1:
        st.subheader("Évolution historique des cours")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['Close'], 
            name="Prix de Clôture / Indice", 
            line=dict(color='#0066CC', width=2)
        ))
        fig.update_layout(
            xaxis_title="Date", 
            yaxis_title="Indice / Prix (MAD)", 
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        # Note : Utilisation de width='stretch' pour respecter les normes Streamlit récentes
        st.plotly_chart(fig, width='stretch')
        
    # --- ONGLET 2 : PRÉDICTION ---
    with tab2:
        st.subheader("Paramètres du modèle de prédiction")
        
        col_algo, col_days = st.columns(2)
        model_choice = col_algo.selectbox("Algorithme de Prédiction", ["Prophet (Meta)", "Random Forest"])
        days_to_predict = col_days.slider("Nombre de jours à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Entraînement du modèle et calcul des projections..."):
                if model_choice == "Prophet (Meta)":
                    forecast, pred_df = train_predict_prophet(df, days_to_predict)
                    
                    fig_pred = go.Figure()
                    # Courbe Historique
                    fig_pred.add_trace(go.Scatter(
                        x=df['Date'], 
                        y=df['Close'], 
                        name="Historique", 
                        line=dict(color='#1C2D42')
                    ))
                    # Courbe Prédiction
                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['Date'], 
                        y=pred_df['Prédiction'], 
                        name="Prédiction Prophet", 
                        line=dict(color='#FF4B4B', dash='dash')
                    ))
                    # Intervalle de confiance
                    fig_pred.add_trace(go.Scatter(
                        x=pd.concat([pred_df['Date'], pred_df['Date'][::-1]]),
                        y=pd.concat([pred_df['Limite_Haute'], pred_df['Limite_Basse'][::-1]]),
                        fill='toself',
                        fillcolor='rgba(255, 75, 75, 0.15)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name="Intervalle de confiance (95%)"
                    ))
                    fig_pred.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Indice / Prix", 
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_pred, width='stretch')
                    
                    st.write("### Tableau des prédictions chiffrées")
                    st.dataframe(pred_df, width='stretch')

                else:  # Random Forest
                    pred_df = train_predict_rf(df, days_to_predict)
                    
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(
                        x=df['Date'], 
                        y=df['Close'], 
                        name="Historique", 
                        line=dict(color='#1C2D42')
                    ))
                    fig_pred.add_trace(go.Scatter(
                        x=pred_df['Date'], 
                        y=pred_df['Prédiction'], 
                        name="Prédiction Random Forest", 
                        line=dict(color='#FF4B4B', dash='dash')
                    ))
                    fig_pred.update_layout(
                        xaxis_title="Date", 
                        yaxis_title="Indice / Prix", 
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig_pred, width='stretch')
                    
                    st.write("### Tableau des prédictions chiffrées")
                    st.dataframe(pred_df, width='stretch')

    # --- ONGLET 3 : RAW DATA ---
    with tab3:
        st.subheader("Aperçu du jeu de données nettoyé")
        st.dataframe(df, width='stretch')

else:
    st.info("👋 Bienvenue ! Veuillez charger les données via la barre latérale pour afficher l'analyse et générer les prédictions.")
