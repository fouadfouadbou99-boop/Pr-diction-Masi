import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from connectors.loader import load_masi

from models.forecasting import train_model
from models.backtest import walk_forward_validation
from models.predict import recursive_forecast


# ------------------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------------------

st.set_page_config(
    page_title="Prévision MASI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prévision du MASI")


# ------------------------------------------------------------------------------
# CHARGEMENT
# ------------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_data():
    return load_masi()


with st.spinner("Chargement des données MASI..."):
    df = get_data()

if df is None or df.empty:
    st.error("Impossible de charger les données MASI.")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"])

# ------------------------------------------------------------------------------
# KPI
# ------------------------------------------------------------------------------

last_close = df["Close"].iloc[-1]

previous_close = (
    df["Close"].iloc[-2]
    if len(df) > 1
    else last_close
)

variation = (
    (last_close / previous_close) - 1
) * 100

ma20 = (
    df["Close"]
    .tail(20)
    .mean()
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dernier cours",
    f"{last_close:,.2f}"
)

col2.metric(
    "Variation",
    f"{variation:.2f}%"
)

col3.metric(
    "MA20",
    f"{ma20:,.2f}"
)

# ------------------------------------------------------------------------------
# TABS
# ------------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Historique",
        "🤖 Prévision",
        "📈 Backtest",
        "📋 Données"
    ]
)

# ------------------------------------------------------------------------------
# HISTORIQUE
# ------------------------------------------------------------------------------

with tab1:

    st.subheader("Historique du MASI")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            mode="lines",
            name="MASI",
            line=dict(
                color="#003366",
                width=2
            )
        )
    )

    fig.update_layout(
        height=600,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Indice"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ------------------------------------------------------------------------------
# PREVISIONS
# ------------------------------------------------------------------------------

with tab2:

    st.subheader("Prévision Walk Forward")

    horizon = st.slider(
        "Nombre de séances à prévoir",
        min_value=1,
        max_value=30,
        value=10
    )

    with st.spinner("Validation du modèle..."):

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

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "MAE",
        round(mae, 2)
    )

    c2.metric(
        "RMSE",
        round(rmse, 2)
    )

    c3.metric(
        "R²",
        round(r2, 3)
    )

    c4.metric(
        "σ erreurs",
        round(sigma, 2)
    )

    fig_pred = go.Figure()

    # historique

    fig_pred.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"],
            name="Historique",
            line=dict(color="blue")
        )
    )

    # borne haute

    fig_pred.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper95"],
            line=dict(width=0),
            showlegend=False
        )
    )

    # borne basse

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

    # prévision

    fig_pred.add_trace(
        go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Prediction"],
            name="Prévision",
            line=dict(
                color="red",
                width=3,
                dash="dash"
            )
        )
    )

    fig_pred.update_layout(
        height=650,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Indice MASI"
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
        label="📥 Télécharger les prévisions",
        data=forecast_df.to_csv(index=False),
        file_name="previsions_masi.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# BACKTEST
# ------------------------------------------------------------------------------

with tab3:

    st.subheader("Validation Walk Forward")

    bt, mae, rmse, r2, sigma = (
        walk_forward_validation(df)
    )

    fig_bt = go.Figure()

    fig_bt.add_trace(
        go.Scatter(
            y=bt["Actual"],
            name="Réel",
            line=dict(color="green")
        )
    )

    fig_bt.add_trace(
        go.Scatter(
            y=bt["Forecast"],
            name="Prévision",
            line=dict(color="orange")
        )
    )

    fig_bt.update_layout(
        height=600,
        hovermode="x unified"
    )

    st.plotly_chart(
        fig_bt,
        use_container_width=True
    )

    residuals = (
        bt["Actual"]
        - bt["Forecast"]
    )

    st.write("### Distribution des erreurs")

    st.line_chart(
        residuals
    )

    st.write(
        f"""
        **MAE :** {mae:.2f}

        **RMSE :** {rmse:.2f}

        **R² :** {r2:.3f}

        **Écart-type des erreurs :** {sigma:.2f}
        """
    )

# ------------------------------------------------------------------------------
# DONNEES
# ------------------------------------------------------------------------------

with tab4:

    st.subheader("Données utilisées")

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "📥 Télécharger les données",
        df.to_csv(index=False),
        file_name="masi_data.csv",
        mime="text/csv"
    )

# ------------------------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------------------------

st.caption(
    """
    Les prévisions sont probabilistes et reposent sur un modèle
    d'apprentissage automatique avec validation temporelle Walk Forward.
    Elles ne constituent pas un conseil d'investissement.
    """
)
