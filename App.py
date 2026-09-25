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

            workbook[sheet] = pd.read_excel(
                xls,
                sheet_name=sheet
            )

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

    workbook = load_excel_model(
        uploaded_file
    )

    st.sidebar.success(
        "Fichier Excel chargé"
    )

else:

    st.warning(
        """
        Importez un fichier Excel contenant :

        Date | Close

        dans une feuille appelée MASI
        ou dans la première feuille.
        """
    )

    st.stop()

# =============================================================================
# FEUILLES
# =============================================================================

st.sidebar.subheader(
    "Feuilles détectées"
)

for sheet in workbook.keys():

    st.sidebar.write(
        f"• {sheet}"
    )

# =============================================================================
# CHOIX FEUILLE
# =============================================================================

if "MASI" in workbook:

    df = workbook["MASI"].copy()

else:

    sheet_name = list(
        workbook.keys()
    )[0]

    st.warning(
        f"Utilisation de : {sheet_name}"
    )

    df = workbook[sheet_name].copy()

# =============================================================================
# NORMALISATION COLONNES
# =============================================================================

rename_map = {

    "DATE": "Date",
    "date": "Date",
    "Séance": "Date",
    "SEANCE": "Date",

    "Close": "Close",
    "CLOSE": "Close",
    "Clôture": "Close",
    "CLOTURE": "Close",
    "Cours": "Close",
    "COURS": "Close",
    "Prix": "Close",
    "PRIX": "Close",
    "Valeur": "Close",
    "VALEUR": "Close"
}

df.rename(
    columns=rename_map,
    inplace=True
)

st.sidebar.subheader(
    "Colonnes détectées"
)

st.sidebar.write(
    list(df.columns)
)

# =============================================================================
# VALIDATION
# =============================================================================

if "Date" not in df.columns:

    st.error(
        "Colonne Date introuvable."
    )

    st.stop()

if "Close" not in df.columns:

    st.error(
        "Colonne Close introuvable."
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

if len(df) < 80:

    st.error(
        f"""
        Historique insuffisant.

        Nombre de lignes :
        {len(df)}

        Minimum recommandé :
        80
        """
    )

    st.stop()

# =============================================================================
# KPI
# =============================================================================

last_close = float(
    df["Close"].iloc[-1]
)

prev_close = float(
    df["Close"].iloc[-2]
)

variation = (
    (
        last_close / prev_close
    ) - 1
) * 100

ma20 = (
    df["Close"]
    .tail(20)
    .mean()
)

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

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Historique",
        "🔮 Prévisions",
        "📈 Backtest",
        "📁 Sources"
    ]
)

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
# PREVISIONS
# =============================================================================

with tab2:

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
        )

        model = train_model(df)

        forecast_df = recursive_forecast(
            model=model,
            df=df,
            horizon=horizon,
            sigma=sigma
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
                y=forecast_df["Upper95"],
                line=dict(width=0),
                showlegend=False
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Lower95"],
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(width=0),
                name="IC95%"
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

        st.exception(e)

# =============================================================================
# BACKTEST
# =============================================================================

with tab3:

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
        )

        a, b, c, d = st.columns(4)

        a.metric(
            "MAE",
            f"{mae:.2f}"
        )

        b.metric(
            "RMSE",
            f"{rmse:.2f}"
        )

        c.metric(
            "R²",
            f"{r2:.3f}"
        )

        d.metric(
            "Sigma",
            f"{sigma:.2f}"
        )

        st.dataframe(
            bt.tail(50),
            width="stretch"
        )

    except Exception as e:

        st.exception(e)

# =============================================================================
# SOURCES
# =============================================================================

with tab4:

    selected_sheet = st.selectbox(
        "Choisir une feuille",
        list(workbook.keys())
    )

    st.dataframe(
        workbook[selected_sheet],
        width="stretch"
    )

# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    """
    Prévisions indicatives.

    Modèle :
    HistGradientBoostingRegressor

    Validation :
    Walk Forward Backtesting

    Les résultats ne constituent pas
    un conseil d'investissement.
    """
)
