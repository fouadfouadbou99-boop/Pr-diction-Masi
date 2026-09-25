import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Prédiction MASI",
    page_icon="📈",
    layout="wide"
)

# ------------------------------------------------------------------------------
# Nettoyage
# ------------------------------------------------------------------------------

def clean_and_sort_df(df):

    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "Séance": "Date",
        "Date": "Date",
        "Clôture": "Close",
        "Cours": "Close",
        "Dernier": "Close",
        "Valeur": "Close",
        "Prix": "Close"
    }

    df.rename(columns=rename_map, inplace=True)

    if "Date" not in df.columns:
        return pd.DataFrame()

    if "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace(" ", "", regex=False)
        .str.replace(",", ".", regex=False)
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna()

    df = (
        df.sort_values("Date")
        .drop_duplicates("Date")
        .reset_index(drop=True)
    )

    return df


# ------------------------------------------------------------------------------
# Chargement Fichier
# ------------------------------------------------------------------------------

def load_file(uploaded_file):

    try:

        filename = uploaded_file.name.lower()

        if filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)

        else:
            df = pd.read_excel(uploaded_file)

        return clean_and_sort_df(df)

    except Exception as e:

        st.error(f"Erreur : {e}")

        return pd.DataFrame()


# ------------------------------------------------------------------------------
# Fallback Historique
# ------------------------------------------------------------------------------

@st.cache_data
def load_fallback():

    try:

        df = pd.read_csv(
            "data/masi_historique.csv"
        )

        return clean_and_sort_df(df)

    except Exception:

        return pd.DataFrame()


# ------------------------------------------------------------------------------
# Features ML
# ------------------------------------------------------------------------------

def build_features(df):

    data = df.copy()

    data["Return_1"] = data["Close"].pct_change()

    data["MA5"] = (
        data["Close"]
        .rolling(5)
        .mean()
    )

    data["MA20"] = (
        data["Close"]
        .rolling(20)
        .mean()
    )

    data["Volatility"] = (
        data["Return_1"]
        .rolling(20)
        .std()
    )

    data["Target"] = (
        data["Close"]
        .shift(-1)
    )

    data.dropna(inplace=True)

    return data


# ------------------------------------------------------------------------------
# Modèle
# ------------------------------------------------------------------------------

def train_model(df):

    data = build_features(df)

    features = [
        "Close",
        "Return_1",
        "MA5",
        "MA20",
        "Volatility"
    ]

    X = data[features]
    y = data["Target"]

    model = RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        random_state=42
    )

    model.fit(X, y)

    pred = model.predict(X)

    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)

    return model, data, mae, r2


# ------------------------------------------------------------------------------
# Prévisions
# ------------------------------------------------------------------------------

def forecast(df, horizon=15):

    model, data, mae, r2 = train_model(df)

    forecasts = []

    temp = data.copy()

    last_date = df["Date"].iloc[-1]

    for i in range(horizon):

        row = temp.iloc[-1]

        X = pd.DataFrame([{
            "Close": row["Close"],
            "Return_1": row["Return_1"],
            "MA5": row["MA5"],
            "MA20": row["MA20"],
            "Volatility": row["Volatility"]
        }])

        pred = model.predict(X)[0]

        forecasts.append(
            {
                "Date": last_date + pd.offsets.BDay(i + 1),
                "Prédiction": pred
            }
        )

        new_row = row.copy()

        new_row["Close"] = pred

        temp = pd.concat(
            [
                temp,
                pd.DataFrame([new_row])
            ],
            ignore_index=True
        )

    return pd.DataFrame(forecasts), mae, r2


# ------------------------------------------------------------------------------
# Interface
# ------------------------------------------------------------------------------

st.title("📈 Prévision du MASI")

uploaded_file = st.sidebar.file_uploader(
    "Importer un historique MASI",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file:

    df = load_file(uploaded_file)

else:

    df = load_fallback()

# ------------------------------------------------------------------------------
# Dashboard
# ------------------------------------------------------------------------------

if not df.empty:

    last_close = df["Close"].iloc[-1]

    if len(df) > 1:
        variation = (
            (
                df["Close"].iloc[-1]
                / df["Close"].iloc[-2]
            ) - 1
        ) * 100
    else:
        variation = 0

    ma20 = (
        df["Close"]
        .tail(20)
        .mean()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dernier Cours",
        f"{last_close:.2f}"
    )

    col2.metric(
        "Variation",
        f"{variation:.2f}%"
    )

    col3.metric(
        "Moyenne 20 séances",
        f"{ma20:.2f}"
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Historique",
            "🤖 Prévision",
            "📋 Données"
        ]
    )

    # --------------------------------------------------------------------------

    with tab1:

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="MASI",
                line=dict(color="#004a99")
            )
        )

        fig.update_layout(
            title="Historique MASI"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------------------------

    with tab2:

        horizon = st.slider(
            "Nombre de séances à prévoir",
            1,
            60,
            15
        )

        if st.button("Lancer la prévision"):

            pred_df, mae, r2 = forecast(
                df,
                horizon
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "MAE",
                round(mae, 2)
            )

            c2.metric(
                "R²",
                round(r2, 3)
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
                    line=dict(
                        color="red",
                        dash="dash"
                    )
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
                "📥 Télécharger les prévisions",
                pred_df.to_csv(index=False),
                file_name="previsions_masi.csv",
                mime="text/csv"
            )

    # --------------------------------------------------------------------------

    with tab3:

        st.dataframe(
            df,
            use_container_width=True
        )

else:

    st.warning(
        """
        Aucun historique disponible.

        Importez un fichier CSV/XLSX contenant :

        - Date
        - Close

        Exemple :

        Date,Close
        01/01/2025,17000
        02/01/2025,17025
        """
    )
