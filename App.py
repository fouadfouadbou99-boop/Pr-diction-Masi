import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from models.forecasting import train_model
from models.backtest import walk_forward_validation
from models.predict import recursive_forecast

# =============================================================================
# CONFIG
# =============================================================================

st.set_page_config(
    page_title="Prévision MASI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prévision du MASI")

# =============================================================================
# CHARGEMENT EXCEL
# =============================================================================

@st.cache_data
def load_excel_model(file):

    workbook = {}

    xls = pd.ExcelFile(file)

    for sheet in xls.sheet_names:

        try:

            df = pd.read_excel(
                xls,
                sheet_name=sheet
            )

            workbook[sheet] = df

        except Exception:
            pass

    return workbook


st.sidebar.header("📂 Données")

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier Excel",
    type=["xlsx", "xls"]
)

workbook = {}

if uploaded_file is not None:

    try:

        workbook = load_excel_model(uploaded_file)

        st.sidebar.success(
            "Fichier Excel chargé"
        )

    except Exception as e:

        st.error(
            f"Erreur lecture Excel : {e}"
        )

        st.stop()

else:

    try:

        workbook = load_excel_model(
            "data/masi_model.xlsx"
        )

        st.sidebar.info(
            "Fichier local utilisé"
        )

    except Exception:

        st.warning(
            """
            Aucun fichier Excel trouvé.

            Importez un fichier Excel.
            """
        )

        st.stop()

# =============================================================================
# FEUILLES
# =============================================================================

st.sidebar.markdown("### Feuilles détectées")

for sheet in workbook.keys():
    st.sidebar.write(f"• {sheet}")

# =============================================================================
# SELECTION AUTOMATIQUE
# =============================================================================

if "MASI" in workbook:

    df = workbook["MASI"].copy()

else:

    first_sheet = list(workbook.keys())[0]

    st.warning(
        f"Utilisation automatique de la feuille : {first_sheet}"
    )

    df = workbook[first_sheet].copy()

# =============================================================================
# NORMALISATION COLONNES
# =============================================================================

rename_map = {

    "DATE": "Date",
    "date": "Date",
    "Date": "Date",
    "Séance": "Date",
    "SEANCE": "Date",
    "SEANCES": "Date",

    "Close": "Close",
    "CLOSE": "Close",
    "Clôture": "Close",
    "CLOTURE": "Close",
    "Cours": "Close",
    "COURS": "Close",
    "Prix": "Close",
    "PRIX": "Close",
    "Valeur": "Close",
    "VALEUR": "Close",
}

df.rename(
    columns=rename_map,
    inplace=True
)

st.sidebar.markdown("### Colonnes détectées")
st.sidebar.write(list(df.columns))

if "Date" not in df.columns:

    st.error(
        "Aucune colonne Date détectée."
    )

    st.stop()

if "Close" not in df.columns:

    st.error(
        "Aucune colonne Close détectée."
    )

    st.stop()

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

df = (
    df.dropna()
    .sort_values("Date")
    .reset_index(drop=True)
)

if len(df) < 60:

    st.warning(
        f"Historique limité : {len(df)} lignes"
    )

# =============================================================================
# KPI
# =============================================================================

last_close = float(df["Close"].iloc[-1])

if len(df) > 1:
    prev_close = float(df["Close"].iloc[-2])
else:
    prev_close = last_close

variation = (
    (last_close / prev_close) - 1
) * 100

ma20 = df["Close"].tail(20).mean()

c1, c2, c3 = st.columns(3)

c1.metric(
    "Dernier cours",
    f"{last_close:,.2f}"
)

c2.metric(
    "Variation",
    f"{variation:.2f}%"
)

c3.metric(
    "MA20",
    f"{ma20:.2f}"
)

# =============================================================================
# PARAMETRES
# =============================================================================

horizon = st.sidebar.slider(
    "Horizon prévision",
    1,
    30,
    10
)

# =============================================================================
# TABS
# =============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Historique",
    "🔮 Prévisions",
    "📈 Backtest",
    "📁 Sources"
])

# =============================================================================
# HISTORIQUE
# =============================================================================

with tab1:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="MASI"
        )
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

# =============================================================================
# PREVISION
# =============================================================================

with tab2:

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
        )

        model = train_model(df)

        forecast_df = recursive_forecast(
            model,
            df,
            horizon,
            sigma
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="Historique"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Prediction"],
                name="Prévision"
            )
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.dataframe(
            forecast_df,
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Erreur prévision : {e}"
        )

# =============================================================================
# BACKTEST
# =============================================================================

with tab3:

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
        )

        a, b, c, d = st.columns(4)

        a.metric("MAE", round(mae, 2))
        b.metric("RMSE", round(rmse, 2))
        c.metric("R²", round(r2, 3))
        d.metric("Sigma", round(sigma, 2))

        st.dataframe(
            bt.tail(50),
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Erreur backtest : {e}"
        )

# =============================================================================
# SOURCES
# =============================================================================

with tab4:

    feuille = st.selectbox(
        "Choisir une feuille",
        list(workbook.keys())
    )

    st.dataframe(
        workbook[feuille],
        width="stretch"
    )

# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    "Prévisions indicatives - aucun conseil d'investissement."
)
