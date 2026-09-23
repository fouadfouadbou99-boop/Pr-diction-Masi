import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta

# ------------------------------------------------------------------------------
# 1. Fonctions de Chargement & Scraping des Données
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def scrape_masi_live() -> pd.DataFrame:
    """
    Récupère en direct les données du MASI via Web Scraping.
    """
    try:
        url = "https://www.leboursier.ma/index-detail/MASI.html"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            for df in tables:
                cols = [str(c).lower() for c in df.columns]
                if any('date' in c or 'séance' in c for c in cols):
                    df.columns = [c.strip().capitalize() for c in df.columns]
                    rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Séance': 'Date'}
                    df.rename(columns=rename_map, inplace=True)
                    
                    if 'Date' in df.columns and 'Close' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                        if df['Close'].dtype == object:
                            df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
                        
                        df = df[['Date', 'Close']].dropna().sort_values(by='Date', ascending=True).reset_index(drop=True)
                        return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors du scraping : {e}")
        return pd.DataFrame()

def load_data_from_yfinance(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        
        if 'Date' in df.columns and 'Close' in df.columns:
            df = df[['Date', 'Close']].dropna()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur Yahoo Finance : {e}")
        return pd.DataFrame()

def load_data_from_csv(uploaded_file) -> pd.DataFrame:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = [c.strip().capitalize() for c in df.columns]
        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close'}
        df.rename(columns=rename_map, inplace=True)

        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Close'].dtype == object:
                df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
            
            df = df[['Date', 'Close']].dropna().sort_values(by='Date', ascending=True).reset_index(drop=True)
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur fichier CSV : {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 2. Algorithme de Prédiction
# ------------------------------------------------------------------------------
def train_predict_rf(df: pd.DataFrame, days_to_predict: int):
    data = df[['Date', 'Close']].copy()
    data['Day_Index'] = np.arange(len(data))
    
    X = data[['Day_Index']]
    y = data['Close']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_index = data['Day_Index'].iloc[-1]
    future_indices = np.array([[last_index + i] for i in range(1, days_to_predict + 1)])
    preds = model.predict(future_indices)
    
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, days_to_predict + 1)]
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Prédiction': preds})
    return pred_df

# ------------------------------------------------------------------------------
# 3. Interface Utilisateur Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Prédiction MASI", page_icon="📈", layout="wide")

st.title("📈 Application de Prédiction Boursière — MASI")

st.sidebar.header("⚙️ Data Source")
source_type = st.sidebar.radio(
    "Choisir la source :",
    ["Scraping MASI en direct", "Téléverser un CSV", "Yahoo Finance"]
)

df = pd.DataFrame()

if source_type == "Scraping MASI en direct":
    if st.sidebar.button("Récupérer le MASI 🔄"):
        with st.spinner("Scraping en cours..."):
            df = scrape_masi_live()

elif source_type == "Téléverser un CSV":
    uploaded_file = st.sidebar.file_uploader("Fichier CSV", type=["csv"])
    if uploaded_file is not None:
        df = load_data_from_csv(uploaded_file)

else:
    ticker = st.sidebar.text_input("Ticker (ex: ATW.CS)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*2))
    end_date = col2.date_input("Fin", date.today())
    if st.sidebar.button("Charger YFinance"):
        df = load_data_from_yfinance(ticker, start_date, end_date)

if not df.empty:
    st.success(f"Données prêtes ({len(df)} lignes)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Raw Data"])
    
    with tab1:
        st.subheader("Historique des cours")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Close", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Prix / Indice", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("Prédiction Machine Learning")
        days_to_predict = st.slider("Jours à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Calcul des prédictions..."):
                pred_df = train_predict_rf(df, days_to_predict)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique", line=dict(color='#1C2D42')))
                fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction RF", line=dict(color='#FF4B4B', dash='dash')))
                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Indice", hovermode="x unified")
                st.plotly_chart(fig_pred, width='stretch')
                
                st.write("### Tableau des prédictions")
                st.dataframe(pred_df, width='stretch')

    with tab3:
        st.dataframe(df, width='stretch')
else:
    st.info("Sélectionnez une source de données dans la barre latérale et cliquez sur le bouton d'action pour commencer.")
