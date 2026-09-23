import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import urllib3
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import date, datetime, timedelta

# Désactiver les avertissements liés aux certificats SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest"
}

# ------------------------------------------------------------------------------
# 1. Nettoyage & Alignement des Dates
# ------------------------------------------------------------------------------
def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et trie les données chronologiquement."""
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return pd.DataFrame()

    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    
    if df['Close'].dtype == object:
        df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    
    df = df.dropna(subset=['Date', 'Close'])
    df = df.drop_duplicates(subset=['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    return df

# ------------------------------------------------------------------------------
# 2. Récupération Directe de la Data Réelle du MASI
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def scrape_masi_bvc() -> pd.DataFrame:
    """
    Récupère l'historique réel du MASI via scraping et endpoints API.
    """
    # Source 1 : Scraping LeBoursier / LeMatin Bourse
    try:
        url = "https://www.leboursier.ma/index-detail/MASI.html"
        res = requests.get(url, headers=HEADERS, timeout=10, verify=False)
        if res.status_code == 200:
            tables = pd.read_html(res.text)
            for df in tables:
                cols = [str(c).lower() for c in df.columns]
                if any('date' in c or 'séance' in c for c in cols):
                    df.columns = [c.strip().capitalize() for c in df.columns]
                    rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close', 'Séance': 'Date'}
                    df.rename(columns=rename_map, inplace=True)
                    
                    cleaned = clean_and_sort_df(df)
                    if not cleaned.empty and len(cleaned) > 5:
                        return cleaned
    except Exception as e:
        st.error(f"Erreur lors de la tentative de scraping LeBoursier : {e}")

    # Source 2 : Endpoint API BVC / Medias24
    try:
        api_url = "https://www.medias24.com/content/api/bourse/index/MASI/history"
        res = requests.get(api_url, headers=HEADERS, timeout=10, verify=False)
        if res.status_code == 200:
            data = res.json()
            if isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df.rename(columns={'date': 'Date', 'valeur': 'Close', 'close': 'Close'}, inplace=True)
                cleaned = clean_and_sort_df(df)
                if not cleaned.empty:
                    return cleaned
    except Exception:
        pass

    return pd.DataFrame()

def load_data_from_yfinance(ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
    try:
        df = yf.download(ticker, start=start_date, end=end_date)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.reset_index(inplace=True)
        return clean_and_sort_df(df)
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
            st.error("Format de fichier non supporté.")
            return pd.DataFrame()

        df.columns = [c.strip().capitalize() for c in df.columns]
        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close'}
        df.rename(columns=rename_map, inplace=True)

        return clean_and_sort_df(df)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 3. Algorithme de Prédiction
# ------------------------------------------------------------------------------
def train_predict_rf(df: pd.DataFrame, days_to_predict: int):
    data = df[['Date', 'Close']].copy()
    data['Day_Index'] = np.arange(len(data))
    
    X = data[['Day_Index']]
    y = data['Close']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    last_index = data['Day_Index'].iloc[-1]
    last_date = data['Date'].iloc[-1]
    
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='B')
    future_indices = np.array([[last_index + i] for i in range(1, len(future_dates) + 1)])
    preds = model.predict(future_indices)
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Prédiction': preds})
    return pred_df

# ------------------------------------------------------------------------------
# 4. Interface Utilisateur Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Prédiction MASI — Bourse de Casablanca", page_icon="📈", layout="wide")

st.title("📈 Prédiction Boursière MASI — Bourse de Casablanca")

st.sidebar.header("⚙️ Source de Données")
source_type = st.sidebar.radio(
    "Choisir la source :",
    ["Bourse de Casablanca (Direct)", "Fichier local (CSV / Excel)", "Yahoo Finance (Actions)"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Direct)":
    with st.spinner("Récupération de la data réelle du MASI..."):
        df = scrape_masi_bvc()

elif source_type == "Fichier local (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Importer un fichier (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)

else:
    ticker = st.sidebar.text_input("Ticker Yahoo (ex: ATW.CS pour Attijariwafa, IAM.CS pour Maroc Telecom)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*2))
    end_date = col2.date_input("Fin", date.today())
    if st.sidebar.button("Charger depuis Yahoo Finance"):
        with st.spinner("Téléchargement des données..."):
            df = load_data_from_yfinance(ticker, start_date, end_date)

# Affichage des données
if not df.empty:
    st.success(f"Données réelles chargées avec succès ({len(df)} séances boursières). Dernier cours enregistré : **{df['Close'].iloc[-1]:,.2f}** MAD")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Données Brutes Réelles"])
    
    with tab1:
        st.subheader("Historique des cours réels")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Cours / Indice", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Valeur (MAD)", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("Prédiction par Machine Learning (Random Forest)")
        days_to_predict = st.slider("Séances à prédire (Jours ouvrés)", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Calcul des projections..."):
                pred_df = train_predict_rf(df, days_to_predict)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique Réel", line=dict(color='#1C2D42')))
                fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction RF", line=dict(color='#FF4B4B', dash='dash')))
                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Valeur (MAD)", hovermode="x unified")
                st.plotly_chart(fig_pred, width='stretch')
                
                st.write("### Tableau des prédictions")
                st.dataframe(pred_df, width='stretch')
                
                csv_data = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv_data,
                    file_name="predictions_masi.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("Données brutes extraites")
        st.dataframe(df, width='stretch')

else:
    st.error("⚠️ Impossible d'extraire la data réelle en direct actuellement. Le site source bloque l'accès automatique ou est momentanément indisponible. Veuillez utiliser la fonction 'Fichier local' en important votre propre fichier CSV/Excel téléchargé depuis la Bourse de Casablanca.")
