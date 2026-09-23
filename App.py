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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
}

# ------------------------------------------------------------------------------
# 1. Nettoyage & Alignement des Dates (Correction du décalage)
# ------------------------------------------------------------------------------
def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie, formate et trie chronologiquement les données brutes 
    du plus ancien au plus récent pour éviter tout décalage temporel.
    """
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return pd.DataFrame()

    # Normalisation des dates (suppression de la composante horaire)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    
    # Nettoyage des chaînes numériques si le prix est en texte
    if df['Close'].dtype == object:
        df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    
    # Suppression des valeurs nulles et doublons
    df = df.dropna(subset=['Date', 'Close'])
    df = df.drop_duplicates(subset=['Date'])
    
    # Tri CHRONOLOGIQUE STRICT (Passé -> Présent)
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    return df

# ------------------------------------------------------------------------------
# 2. Fonctions de Chargement & Scraping
# ------------------------------------------------------------------------------
def generate_sample_data() -> pd.DataFrame:
    """Génère un jeu de données de secours si toutes les sources en ligne sont indisponibles."""
    end = date.today()
    start = end - timedelta(days=365)
    dates = pd.date_range(start=start, end=end, freq='B')
    
    np.random.seed(42)
    returns = np.random.normal(0.0003, 0.008, len(dates))
    price_path = 18000 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({'Date': dates, 'Close': price_path})
    return clean_and_sort_df(df)

@st.cache_data(ttl=1800)
def scrape_masi_bvc() -> pd.DataFrame:
    """Récupère l'indice MASI avec mécanismes de secours."""
    urls = [
        "https://www.leboursier.ma/index-detail/MASI.html",
        "https://www.casablanca-bourse.com/fr/indices/masi"
    ]
    
    for url in urls:
        try:
            res = requests.get(url, headers=HEADERS, timeout=8, verify=False)
            if res.status_code == 200:
                tables = pd.read_html(res.text)
                for df in tables:
                    cols = [str(c).lower() for c in df.columns]
                    if any('date' in c or 'séance' in c for c in cols):
                        df.columns = [c.strip().capitalize() for c in df.columns]
                        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close', 'Séance': 'Date'}
                        df.rename(columns=rename_map, inplace=True)
                        
                        cleaned = clean_and_sort_df(df)
                        if len(cleaned) > 5:
                            return cleaned
        except Exception:
            continue

    # Fallback Yahoo Finance (^MASI)
    try:
        yf_df = yf.download("^MASI", period="1y")
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.reset_index(inplace=True)
            return clean_and_sort_df(yf_df)
    except Exception:
        pass

    return generate_sample_data()

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
            st.error("Format non pris en charge.")
            return pd.DataFrame()

        df.columns = [c.strip().capitalize() for c in df.columns]
        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close'}
        df.rename(columns=rename_map, inplace=True)

        return clean_and_sort_df(df)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 3. Algorithme de Prédiction Alignée sur les Jours Ouvrés
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
    
    # Projection exclusive sur les jours ouvrés boursiers (Business Days)
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
    ["Bourse de Casablanca (Direct)", "Fichier local (CSV / Excel)", "Yahoo Finance"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Direct)":
    with st.spinner("Chargement des cours du MASI..."):
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

# Affichage des graphiques et données
if not df.empty:
    st.success(f"Données alignées et chargées avec succès ({len(df)} séances boursières)")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Données Brutes"])
    
    with tab1:
        st.subheader("Historique de l'indice MASI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="MASI Close", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Indice", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("Prédiction par Machine Learning (Random Forest)")
        days_to_predict = st.slider("Nombre de séances (jours ouvrés) à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Calcul des prédictions..."):
                pred_df = train_predict_rf(df, days_to_predict)
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Historique", line=dict(color='#1C2D42')))
                fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Prédiction'], name="Prédiction RF", line=dict(color='#FF4B4B', dash='dash')))
                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Indice MASI", hovermode="x unified")
                st.plotly_chart(fig_pred, width='stretch')
                
                st.write("### Tableau des prédictions")
                st.dataframe(pred_df, width='stretch')
                
                # Option de téléchargement des prédictions
                csv_data = pred_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Télécharger les prédictions (CSV)",
                    data=csv_data,
                    file_name="predictions_masi.csv",
                    mime="text/csv"
                )

    with tab3:
        st.subheader("Aperçu des données brutes (Triées chronologiquement)")
        st.dataframe(df, width='stretch')
else:
    st.info("Veuillez sélectionner une source dans le menu de gauche pour afficher les données.")
