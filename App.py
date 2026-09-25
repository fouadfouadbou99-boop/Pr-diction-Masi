import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import urllib3

from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Prédiction MASI",
    page_icon="📈",
    layout="wide"
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

# ------------------------------------------------------------------------------
# Fonctions
# ------------------------------------------------------------------------------

def clean_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:

    if df.empty:
        return pd.DataFrame()

    if "Date" not in df.columns or "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date", "Close"])

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return df


@st.cache_data(ttl=900)
def scrape_casablanca_bourse():

    try:

        url = "https://www.casablanca-bourse.com/fr/indices/masi"

        proxy_url = (
            "https://api.allorigins.win/get?url="
            + requests.utils.quote(url)
        )

        response = requests.get(
            proxy_url,
            timeout=15
        )

        if response.status_code != 200:
            return pd.DataFrame()

        html = response.json().get("contents", "")

        if not html:
            return pd.DataFrame()

        tables = pd.read_html(html)

        for table in tables:

            cols = [str(c).lower() for c in table.columns]

            if any("date" in c for c in cols):

                rename_map = {
                    "Clôture": "Close",
                    "Prix": "Close",
                    "Cours": "Close",
                    "Dernier": "Close",
                    "Valeur": "Close",
                    "Séance": "Date"
                }

                table.rename(
                    columns=rename_map,
                    inplace=True
                )

                clean_df = clean_and_sort_df(table)

                if not clean_df.empty:
                    return clean_df

    except Exception as e:
        st.warning(f"Erreur Bourse de Casablanca : {e}")

    return pd.DataFrame()


def load_data(uploaded_file):

    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file)

        else:
            return pd.DataFrame()

        rename_map = {
            "Clôture": "Close",
            "Prix": "Close",
            "Cours": "Close",
            "Dernier": "Close",
            "Valeur": "Close",
            "Séance": "Date"
        }

        df.rename(columns=rename_map, inplace=True)

        return clean_and_sort_df(df)

    except Exception as e:

        st.error(
            f"Erreur lecture fichier : {e}"
        )

        return pd.DataFrame()


def train_model(df, horizon):

    data = df[["Date", "Close"]].copy()

    data["Index"] = np.arange(len(data))

    X = data[["Index"]]
    y = data["Close"]

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    last_idx = data["Index"].iloc[-1]
    last_date = data["Date"].iloc[-1]

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon,
        freq="B"
    )

    future_idx = np.arange(
        last_idx + 1,
        last_idx + horizon + 1
    ).reshape(-1, 1)

    predictions = model.predict(future_idx)

    pred_df = pd.DataFrame({
        "Date": future_dates,
        "Prédiction": predictions
    })

    return pred_df


# ------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------

st.title("📈 Prédiction du MASI")

source = st.sidebar.radio(
    "Source de données",
    [
        "Fichier Excel / CSV",
        "Bourse de Casablanca"
    ]
)

df = pd.DataFrame()

if source == "Bourse de Casablanca":

    with st.spinner("Chargement des données..."):
        df = scrape_casablanca_bourse()

else:

    uploaded_file = st.sidebar.file_uploader(
        "Importer un fichier",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file:
        df = load_data(uploaded_file)

# ------------------------------------------------------------------------------
# Affichage
# ------------------------------------------------------------------------------

if not df.empty:

    st.success(
        f"{len(df)} observations chargées."
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Historique",
            "🤖 Prédiction",
            "📋 Données"
        ]
    )

    with tab1:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                mode="lines",
                name="MASI"
            )
        )

        fig.update_layout(
            title="Historique du MASI"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with tab2:

        horizon = st.slider(
            "Nombre de jours ouvrés",
            1,
            60,
            14
        )

        if st.button("Lancer la prédiction"):

            pred_df = train_model(
                df,
                horizon
            )

            fig2 = go.Figure()

            fig2.add_trace(
                go.Scatter(
                    x=df["Date"],
                    y=df["Close"],
                    name="Historique"
                )
            )

            fig2.add_trace(
                go.Scatter(
                    x=pred_df["Date"],
                    y=pred_df["Prédiction"],
                    name="Prévision",
                    line=dict(dash="dash")
                )
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            st.dataframe(
                pred_df,
                use_container_width=True
            )

            st.download_button(
                "📥 Télécharger CSV",
                pred_df.to_csv(index=False),
                "predictions_masi.csv",
                "text/csv"
            )

    with tab3:

        st.dataframe(
            df,
            use_container_width=True
        )

else:

    st.info(
        "Importez un fichier CSV/XLSX ou utilisez la source Bourse de Casablanca."
    )
