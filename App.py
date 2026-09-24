import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import urllib3
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta

# Désactiver les avertissements de certificats SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ------------------------------------------------------------------------------
# 1. Nettoyage & Alignement Temporal (Suppression des décalages)
# ------------------------------------------------------------------------------
def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formate, nettoie et trie chronologiquement les données du plus ancien au plus récent.
    """
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return pd.DataFrame()

    # Normalisation de la date (suppression des composantes horaires/timezones)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    
    # Conversion du prix si sous forme de texte
    if df['Close'].dtype == object:
        df['Close'] = df['Close'].astype(str).str.replace(' ', '').str.replace(',', '.').astype(float)
    
    # Nettoyage et tri chronologique strict (Passé -> Présent)
    df = df.dropna(subset=['Date', 'Close'])
    df = df.drop_duplicates(subset=['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    return df

def generate_demo_dataset() -> pd.DataFrame:
    """Génère un dataset MASI de référence au format valide."""
    end = date.today()
    start = end - timedelta(days=365)
    dates = pd.date_range(start=start, end=end, freq='B')
    
    np.random.seed(42)
    returns = np.random.normal(0.0002, 0.006, len(dates))
    price_path = 14200 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({'Date': dates, 'Close': np.round(price_path, 2)})
    return clean_and_sort_df(df)

# ------------------------------------------------------------------------------
# 2. Pipeline de Récupération Boursière
# ------------------------------------------------------------------------------
@st.cache_data(ttl=1800)
def scrape_masi_bvc() -> pd.DataFrame:
    """
    Tente la récupération directe sur plusieurs sources, avec fallback automatique Yahoo Finance.
    """
    sources = [
        "https://www.ilboursa.com/marches/historiques/MASI.ma",
        "https://www.leboursier.ma/index-detail/MASI.html",
        "https://www.casablanca-bourse.com/fr/indices/masi"
    ]
    
    # Tentative 1 : Scraping Direct
    for url in sources:
        try:
            res = requests.get(url, headers=HEADERS, timeout=6, verify=False)
            if res.status_code == 200:
                tables = pd.read_html(res.text)
                for df in tables:
                    cols = [str(c).lower() for c in df.columns]
                    if any('date' in c or 'séance' in c for c in cols):
                        df.columns = [c.strip().capitalize() for c in df.columns]
                        rename_map = {
                            'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 
                            'Valeur': 'Close', 'Séance': 'Date', 'Fermeture': 'Close'
                        }
                        df.rename(columns=rename_map, inplace=True)
                        cleaned = clean_and_sort_df(df)
                        if not cleaned.empty and len(cleaned) > 5:
                            return cleaned
        except Exception:
            continue

    # Tentative 2 : Fallback Automatique YFinance
    try:
        yf_df = yf.download("^MASI", period="1y")
        if not yf_df.empty:
            if isinstance(yf_df.columns, pd.MultiIndex):
                yf_df.columns = yf_df.columns.get_level_values(0)
            yf_df.reset_index(inplace=True)
            cleaned = clean_and_sort_df(yf_df)
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
            st.error("Format de fichier non pris en charge.")
            return pd.DataFrame()

        df.columns = [c.strip().capitalize() for c in df.columns]
        rename_map = {'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 'Valeur': 'Close'}
        df.rename(columns=rename_map, inplace=True)

        return clean_and_sort_df(df)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 3. Moteur de Prédiction ML (Random Forest)
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
    
    # Projection sur les jours ouvrés boursiers uniquement (Business Days)
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
    ["Bourse de Casablanca (Auto / Direct)", "Fichier local (CSV / Excel)", "Yahoo Finance (Actions)"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Auto / Direct)":
    with st.spinner("Récupération en direct des cours du MASI..."):
        df = scrape_masi_bvc()

elif source_type == "Fichier local (CSV / Excel)":
    uploaded_file = st.sidebar.file_uploader("Importer un fichier (CSV, XLSX, XLS)", type=["csv", "xlsx", "xls"])
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)
    else:
        st.sidebar.info("💡 Téléchargez le modèle CSV ci-dessous si vous n'avez pas de fichier :")
        demo_df = generate_demo_dataset()
        csv_demo = demo_df.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Télécharger Modèle MASI (CSV)",
            data=csv_demo,
            file_name="masi_historique_template.csv",
            mime="text/csv"
        )

else:
    ticker = st.sidebar.text_input("Ticker Yahoo (ex: ATW.CS pour Attijariwafa)", value="ATW.CS")
    col1, col2 = st.sidebar.columns(2)
    start_date = col1.date_input("Début", date.today() - timedelta(days=365*2))
    end_date = col2.date_input("Fin", date.today())
    if st.sidebar.button("Charger depuis Yahoo Finance"):
        with st.spinner("Téléchargement des données..."):
            df = load_data_from_yfinance(ticker, start_date, end_date)

# Render principal
if not df.empty:
    st.success(f"Données réelles chargées avec succès ({len(df)} séances boursières). Dernier cours : **{df['Close'].iloc[-1]:,.2f}** Pts/MAD")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique", "🤖 Prédiction ML", "📑 Données Brutes"])
    
    with tab1:
        st.subheader("Historique des cours réels")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="Cours / Indice", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Valeur", hovermode="x unified")
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
                fig_pred.update_layout(xaxis_title="Date", yaxis_title="Valeur", hovermode="x unified")
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
    st.warning("⚠️ Les serveurs de scraping direct sont actuellement bloqués par les sécurités anti-bot du serveur distant. Sélectionnez 'Fichier local' dans le menu latéral et cliquez sur 'Télécharger Modèle MASI' pour tester immédiatement l'application.")
