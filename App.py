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
# CHARGEMENT EXCEL MULTI-FEUILLES
# =============================================================================

@st.cache_data
def load_excel_model(path):

    workbook = {}

    xls = pd.ExcelFile(path)

    for sheet in xls.sheet_names:

        try:

            df = pd.read_excel(
                xls,
                sheet_name=sheet
            )

            if "Date" in df.columns:

                df["Date"] = pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                )

            workbook[sheet] = df

        except Exception as e:

            st.warning(
                f"Erreur feuille {sheet}: {e}"
            )

    return workbook


try:

    workbook = load_excel_model(
        "data/masi_model.xlsx"
    )

except Exception as e:

    st.error(
        f"Impossible de lire data/masi_model.xlsx : {e}"
    )

    st.stop()

# =============================================================================
# FEUILLE MASI
# =============================================================================

if "MASI" not in workbook:

    st.error(
        "Feuille MASI absente"
    )

    st.stop()

df = workbook["MASI"].copy()

required = ["Date", "Close"]

for col in required:

    if col not in df.columns:

        st.error(
            f"Colonne obligatoire absente : {col}"
        )

        st.stop()

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

df = (
    df.dropna()
      .sort_values("Date")
      .reset_index(drop=True)
)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.header(
    "📁 Modèle Excel"
)

sheet_names = list(
    workbook.keys()
)

selected_sheet = st.sidebar.selectbox(
    "Consulter une feuille",
    sheet_names
)

horizon = st.sidebar.slider(
    "Horizon prévision",
    1,
    30,
    10
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
    f"{ma20:,.2f}"
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

    fig.update_layout(
        height=600,
        title="Historique MASI"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =============================================================================
# BACKTEST
# =============================================================================

with tab3:

    st.subheader(
        "Walk Forward Validation"
    )

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
        )

        b1, b2, b3, b4 = st.columns(4)

        b1.metric(
            "MAE",
            round(mae, 2)
        )

        b2.metric(
            "RMSE",
            round(rmse, 2)
        )

        b3.metric(
            "R²",
            round(r2, 3)
        )

        b4.metric(
            "Sigma",
            round(sigma, 2)
        )

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
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Erreur backtest : {e}"
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
            model=model,
            df=df,
            horizon=horizon,
            sigma=sigma
        )

        fig_pred = go.Figure()

        fig_pred.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["Close"],
                name="Historique",
                line=dict(color="blue")
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Upper95"],
                line=dict(width=0),
                showlegend=False
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Lower95"],
                fill="tonexty",
                fillcolor="rgba(255,0,0,0.15)",
                line=dict(width=0),
                name="IC 95%"
            )
        )

        fig_pred.add_trace(
            go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Prediction"],
                name="Prévision",
                line=dict(
                    color="red",
                    dash="dash",
                    width=3
                )
            )
        )

        fig_pred.update_layout(
            height=650,
            title="Prévision MASI"
        )

        st.plotly_chart(
            fig_pred,
            use_container_width=True
        )

        st.dataframe(
            forecast_df,
            use_container_width=True
        )

        st.download_button(
            "📥 Télécharger les prévisions",
            forecast_df.to_csv(index=False),
            "previsions_masi.csv",
            "text/csv"
        )

    except Exception as e:

        st.error(
            f"Erreur prévision : {e}"
        )

# =============================================================================
# SOURCES EXCEL
# =============================================================================

with tab4:

    st.subheader(
        f"Feuille : {selected_sheet}"
    )

    st.dataframe(
        workbook[selected_sheet],
        use_container_width=True
    )

    st.download_button(
        f"Télécharger {selected_sheet}",
        workbook[selected_sheet].to_csv(index=False),
        file_name=f"{selected_sheet}.csv",
        mime="text/csv"
    )

# =============================================================================
# FOOTER
# =============================================================================

st.caption(
    """
    Modèle de prévision du MASI avec :
    
    • Validation Walk-Forward
    • Intervalles de confiance 95 %
    • Prévision récursive
    • Données Excel multi-feuilles
    
    Les prévisions sont indicatives et ne constituent pas
    un conseil d'investissement.
    """
)
