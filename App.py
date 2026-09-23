import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import urllib3
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta

# Désactiver les avertissements SSL pour les connexions HTTP
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Headers globaux pour simuler un navigateur standard
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ------------------------------------------------------------------------------
# 1. Fonctions de Chargement & Scraping
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def scrape_masi_bvc() -> pd.DataFrame:
    """
    Scrape le MASI directement depuis la Bourse de Casablanca.
    Utilise un canal alternatif si le site principal reformatte ses tables.
    """
    # 1er Essai : Bourse de Casablanca (Site Officiel)
    try:
        url_bvc = "https://www.casablanca-bourse.com/fr/indices/masi"
        res = requests.get(url_bvc, headers=HEADERS, timeout=10, verify=False)
        
        if res.status_code == 200:
            tables = pd.read_html(res.text)
            for df in tables:
                cols = [str(c).lower() for c in df.columns]
                if any('date' in c or 'séance' in c for c in cols):
                    df.columns = [c.strip().capitalize() for c in df.columns]
                    rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close', 'Séance': 'Date'}
                    df.rename(columns=rename_map, inplace=True)
                    
                    if 'Date' in df.columns and 'Close' in df.columns:
                        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
                        if df['Close'].dtype == object:
                            df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
                        
                        df = df[['Date', 'Close']].dropna().sort_values(by='Date', ascending=True).reset_index(drop=True)
                        if not df.empty:
                            return df
    except Exception:
        pass

    # 2e Essai (Fallback) : API BVC / Source miroir Casablanca
    try:
        url_fallback = "https://www.leboursier.ma/index-detail/MASI.html"
        res = requests.get(url_fallback, headers=HEADERS, timeout=10, verify=False)
        if res.status_code == 200:
            tables = pd.read_html(res.text)
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

def load_data_from_file(uploaded_file) -> pd.DataFrame:
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Format de fichier non pris en charge.")
            return pd.DataFrame()

        df.columns = [c.strip().capitalize() for c in df.columns]
        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close'}
        df.rename(columns=rename_map, inplace=True)

        if 'Date' in df.columns and 'Close' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            if df['Close'].dtype == object:
                df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
            
            df = df[['Date', 'Close']].dropna().sort_values(by='Date', ascending=True).reset_index(drop=True)
            return df
        else:
            st.error("Le fichier doit obligatoirement contenir les colonnes 'Date' et 'Close' (ou 'Clôture').")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
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
st.set_page_config(page_title="Prédiction MASI — Bourse de Casablanca", page_icon="📈", layout="wide")

st.title("📈 Prédiction Boursière MASI — Bourse de Casablanca")

st.sidebar.header("⚙️ Source de Données")
source_type = st.sidebar.radio(
    "Choisir la source :",
    ["Bourse de Casablanca (Direct)", "Fichier local (CSV / Excel)", "Yahoo Finance"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Direct)":
    with st.spinner("Récupération en cours sur la Bourse de Casablanca..."):
        df = scrape_masi_bvc()

elif source_type == "Fichier local (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader(
        "Importer un fichier (CSV, XLSX, XLS)", 
        type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)

else:
    ticker = st.sidebar.text_input("Ticker Yahoo (ex: ATW.CS)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*2))
    end_date = col2.date_input("Fin", date.today())
    if st.sidebar.button("Charger Yahoo Finance"):
        with st.spinner("Téléchargement depuis Yahoo Finance..."):
            df = load_data_from_yfinance(ticker, start_date, end_date)

# Affichage des résultats
if not df.empty:
    st.success(f"Données de la Bourse chargées avec succès ({len(df)} enregistrements)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Données Brutes"])
    
    with tab1:
        st.subheader("Historique de l'indice MASI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="MASI Close", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Indice", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("Prédiction par Random Forest")
        days_to_predict = st.slider("Nombre de jours à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Entraînement du modèle..."):
                pred_df = train_predict_rf(df, days_to_predict)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique", line=dict(color='#1C2D42')))
                fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction RF", line=dict(color='#FF4B4B', dash='dash')))
                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Indice MASI", hovermode="x unified")
                st.plotly_chart(fig_pred, width='stretch')
                
                st.write("### Tableau des prédictions")
                st.dataframe(pred_df, width='stretch')

    with tab3:
        st.subheader("Données brutes")
        st.dataframe(df, width='stretch')
else:
    st.warning("Aucune donnée disponible. Veuillez vérifier la connexion ou importer un fichier.")
