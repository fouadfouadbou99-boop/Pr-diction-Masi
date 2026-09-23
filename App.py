import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta
from bs4 import BeautifulSoup

# Import de curl_cffi pour contourner la protection Cloudflare / Bot detection
try:
    from curl_cffi import requests as curl_requests
    HAS_CURL_CFFI = True
except ImportError:
    import requests as curl_requests
    HAS_CURL_CFFI = False

# ------------------------------------------------------------------------------
# 1. Nettoyage & Alignement des Dates
# ------------------------------------------------------------------------------
def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie et trie les données chronologiquement (anciennes -> récentes)."""
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
# 2. Scraping Anti-Cloudflare & API Directes
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def scrape_masi_bvc() -> pd.DataFrame:
    """
    Récupère la data réelle du MASI en simulant l'empreinte TLS d'un navigateur Chrome.
    """
    # Option A: Contournement Cloudflare sur LeBoursier via impersonation Chrome
    try:
        url = "https://www.leboursier.ma/index-detail/MASI.html"
        if HAS_CURL_CFFI:
            res = curl_requests.get(url, impersonate="chrome120", timeout=12)
        else:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            res = curl_requests.get(url, headers=headers, timeout=12, verify=False)
            
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
    except Exception:
        pass

    # Option B: Reconstitution via l'API Medias24 / LeBoursier
    try:
        api_url = "https://www.medias24.com/content/api/bourse/index/MASI/history"
        if HAS_CURL_CFFI:
            res = curl_requests.get(api_url, impersonate="chrome120", timeout=10)
        else:
            res = curl_requests.get(api_url, timeout=10, verify=False)
            
        if res.status_code == 200:
            json_data = res.json()
            if isinstance(json_data, list) and len(json_data) > 0:
                df = pd.DataFrame(json_data)
                df.rename(columns={'date': 'Date', 'valeur': 'Close', 'close': 'Close', 'c': 'Close', 'd': 'Date'}, inplace=True)
                cleaned = clean_and_sort_df(df)
                if not cleaned.empty:
                    return cleaned
    except Exception:
        pass

    # Option C: Extraction directe Bourse de Casablanca avec contournement TLS
    try:
        bvc_url = "https://www.casablanca-bourse.com/fr/indices/masi"
        if HAS_CURL_CFFI:
            res = curl_requests.get(bvc_url, impersonate="chrome120", timeout=12)
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
# 3. Algorithme de Prédiction (Business Days)
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
    ["Bourse de Casablanca (Scraping Anti-Blocage)", "Fichier local (CSV / Excel)", "Yahoo Finance"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Scraping Anti-Blocage)":
    with st.spinner("Bypass des protections & chargement du MASI en direct..."):
        df = scrape_masi_bvc()

elif source_type == "Fichier local (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Importer un fichier (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)

else:
    ticker = st.sidebar.text_input("Ticker Yahoo (ex: ATW.CS pour Attijariwafa)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*2))
    end_date = col2.date_input("Fin", date.today())
    if st.sidebar.button("Charger depuis Yahoo Finance"):
        with st.spinner("Téléchargement des données..."):
            df = load_data_from_yfinance(ticker, start_date, end_date)

# Affichage des données
if not df.empty:
    st.success(f"Données réelles chargées avec succès ({len(df)} séances boursières). Dernier cours : **{df['Close'].iloc[-1]:,.2f}** MAD")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Données Brutes"])
    
    with tab1:
        st.subheader("Historique des cours réels")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="MASI Close", line=dict(color='#0066CC', width=2)))
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
        st.subheader("Données brutes réelles")
        st.dataframe(df, width='stretch')

else:
    st.error("Impossible d'extraire automatiquement la data en direct actuellement. Le serveur source bloque la requête ou nécessite une mise à jour. Veuillez utiliser la fonction 'Fichier local' en important un fichier CSV/Excel.")
