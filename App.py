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

        df = pd.read_excel(
            xls,
            sheet_name=sheet
        )

        workbook[sheet] = df

    return workbook


# =============================================================================
# FUSION MACRO
# =============================================================================

def build_dataset(workbook):

    if "MASI" not in workbook:
        raise ValueError(
            "Feuille MASI absente"
        )

    df = workbook["MASI"].copy()

    df["Date"] = pd.to_datetime(df["Date"])

    # BAM

    if "BAM" in workbook:

        bam = workbook["BAM"].copy()

        bam["Date"] = pd.to_datetime(
            bam["Date"]
        )

        df = df.merge(
            bam,
            on="Date",
            how="left"
        )

    # CHANGE

    if "CHANGE" in workbook:

        change = workbook["CHANGE"].copy()

        change["Date"] = pd.to_datetime(
            change["Date"]
        )

        df = df.merge(
            change,
            on="Date",
            how="left"
        )

    # MARCHES

    if "MARCHES" in workbook:

        marches = workbook["MARCHES"].copy()

        marches["Date"] = pd.to_datetime(
            marches["Date"]
        )

        df = df.merge(
            marches,
            on="Date",
            how="left"
        )

    df = (
        df.sort_values("Date")
          .ffill()
          .bfill()
          .reset_index(drop=True)
    )

    return df


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header(
    "📂 Données"
)

uploaded_file = st.sidebar.file_uploader(
    "Importer un fichier Excel",
    type=["xlsx", "xls"]
)

if uploaded_file is None:

    st.warning(
        """
        Chargez un fichier Excel.

        Feuilles recommandées :

        • MASI
        • BAM
        • CHANGE
        • MARCHES
        """
    )

    st.stop()

workbook = load_excel_model(
    uploaded_file
)

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
# DATASET FUSIONNE
# =============================================================================

try:

    df = build_dataset(
        workbook
    )

except Exception as e:

    st.exception(e)

    st.stop()

# =============================================================================
# NORMALISATION
# =============================================================================

rename_map = {

    "DATE": "Date",
    "date": "Date",

    "Clôture": "Close",
    "CLOTURE": "Close",
    "Cours": "Close",
    "COURS": "Close",
    "Prix": "Close",
    "VALEUR": "Close"
}

df.rename(
    columns=rename_map,
    inplace=True
)

if "Close" not in df.columns:

    st.error(
        "Colonne Close absente."
    )

    st.stop()

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

df = df.dropna(
    subset=["Close"]
)

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
    (last_close / prev_close) - 1
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

        st.download_button(
            "📥 Télécharger les prévisions",
            forecast_df.to_csv(index=False),
            "previsions_masi.csv",
            "text/csv"
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

        a.metric("MAE", round(mae, 2))
        b.metric("RMSE", round(rmse, 2))
        c.metric("R²", round(r2, 3))
        d.metric("Sigma", round(sigma, 2))

        fig_bt = go.Figure()

        fig_bt.add_trace(
            go.Scatter(
                y=bt["Actual"],
                name="Réel"
            )
        )

        fig_bt.add_trace(
            go.Scatter(
                y=bt["Forecast"],
                name="Prévision"
            )
        )

        st.plotly_chart(
            fig_bt,
            width="stretch"
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
    Modèle : HistGradientBoostingRegressor.
    Validation : Walk Forward Backtesting.
    """
)
