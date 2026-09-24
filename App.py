import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import urllib3
from sklearn.ensemble import RandomForestRegressor
from datetime import date, timedelta

# Désactiver les avertissements SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr,fr-FR;q=0.9,en-US;q=0.8,en;q=0.7",
}

# ------------------------------------------------------------------------------
# 1. Nettoyage & Alignement Chronologique Strict
# ------------------------------------------------------------------------------
def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie les valeurs boursières et trie le DataFrame du passé au présent."""
    if df.empty or 'Date' not in df.columns or 'Close' not in df.columns:
        return pd.DataFrame()

    # Normalisation des dates (suppression des composantes horaires)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    
    # Conversion numérique (nettoyage des espaces insécables et virgules marocaines)
    if df['Close'].dtype == object:
        df['Close'] = (
            df['Close']
            .astype(str)
            .str.replace('\xa0', '', regex=False)
            .str.replace(' ', '', regex=False)
            .str.replace(',', '.', regex=False)
            .astype(float)
        )
    
    df = df.dropna(subset=['Date', 'Close'])
    df = df.drop_duplicates(subset=['Date'])
    df = df.sort_values(by='Date', ascending=True).reset_index(drop=True)
    return df

# ------------------------------------------------------------------------------
# 2. Scraping Direct Anti-Blocage (Bourse de Casablanca)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=900)
def scrape_casablanca_bourse() -> pd.DataFrame:
    """Tentative d'extraction en direct avec contournement de proxy."""
    bvc_url = "https://www.casablanca-bourse.com/fr/indices/masi"
    proxy_url = f"https://api.allorigins.win/get?url={requests.utils.quote(bvc_url)}"
    
    try:
        res = requests.get(proxy_url, timeout=10)
        if res.status_code == 200:
            contents = res.json().get('contents', '')
            if contents:
                tables = pd.read_html(contents)
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

# ------------------------------------------------------------------------------
# 3. Moteur d'Importation Excel / CSV
# ------------------------------------------------------------------------------
def load_data_from_file(uploaded_file) -> pd.DataFrame:
    """Charge et adapte un fichier Excel ou CSV de la Bourse de Casablanca."""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        elif filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            st.error("Format non pris en charge. Veuillez importer un fichier .xlsx, .xls ou .csv")
            return pd.DataFrame()

        df.columns = [str(c).strip().capitalize() for c in df.columns]
        rename_map = {
            'Clôture': 'Close', 'Prix': 'Close', 'Dernier': 'Close', 
            'Valeur': 'Close', 'Cours': 'Close', 'Séance': 'Date'
        }
        df.rename(columns=rename_map, inplace=True)

        return clean_and_sort_df(df)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier Excel/CSV : {e}")
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# 4. Modèle Machine Learning (Random Forest - Business Days)
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
    
    # Alignement exclusif sur les jours ouvrés (freq='B')
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_to_predict, freq='B')
    future_indices = np.array([[last_index + i] for i in range(1, len(future_dates) + 1)])
    preds = model.predict(future_indices)
    
    pred_df = pd.DataFrame({'Date': future_dates, 'Prédiction': preds})
    return pred_df

# ------------------------------------------------------------------------------
# 5. Interface Streamlit
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Prédiction MASI — Bourse de Casablanca", page_icon="📈", layout="wide")

st.title("📈 Prédiction Boursière MASI — Bourse de Casablanca")

st.sidebar.header("⚙️ Source de Données")
source_type = st.sidebar.radio(
    "Choisir le mode d'alimentation :",
    ["Fichier Local (Excel / CSV BVC)", "Bourse de Casablanca (Direct Officiel)"]
)

df = pd.DataFrame()

if source_type == "Bourse de Casablanca (Direct Officiel)":
    with st.spinner("Connexion au serveur BVC..."):
        df = scrape_casablanca_bourse()

else:
    uploaded_file = st.sidebar.file_uploader("Importer l'historique BVC (.xlsx, .xls, .csv)", type=["xlsx", "xls", "csv"])
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)

# Affichage des résultats
if not df.empty:
    st.success(f"Données BVC chargées avec succès ({len(df)} séances boursières). Dernier cours : **{df['Close'].iloc[-1]:,.2f}** Pts/MAD")
    
    tab1, tab2, tab3 = st.tabs(["📊 Graphique Historique", "🤖 Prédiction ML (Random Forest)", "📑 Données Brutes"])
    
    with tab1:
        st.subheader("Historique du MASI")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name="MASI", line=dict(color='#0066CC', width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="Valeur (Pts/MAD)", hovermode="x unified")
        st.plotly_chart(fig, width='stretch')
        
    with tab2:
        st.subheader("Prédictions Boursières (Machine Learning)")
        days_to_predict = st.slider("Séances ouvrées à prédire", min_value=1, max_value=60, value=14)
        
        if st.button("Lancer la prédiction 🚀"):
            with st.spinner("Calcul des trajectoires ML..."):
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
        st.subheader("Données brutes vérifiées")
        st.dataframe(df, width='stretch')

else:
    st.info("💡 Veuillez importer votre fichier Excel (`.xlsx`) ou `.csv` téléchargé depuis la Bourse de Casablanca dans le menu de gauche.")
