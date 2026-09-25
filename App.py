import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from connectors.loader import load_masi

from models.forecasting import train_model
from models.backtest import walk_forward_validation
from models.predict import recursive_forecast

# ==============================================================================
# CONFIG
# ==============================================================================

st.set_page_config(
    page_title="Prévision du MASI",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Prévision du MASI")

# ==============================================================================
# CHARGEMENT DES DONNÉES
# ==============================================================================

@st.cache_data(ttl=3600)
def get_data():
    return load_masi()

df = None

with st.spinner("Chargement des données MASI..."):
    try:
        df = get_data()
    except Exception as e:
        st.warning(f"Erreur chargement automatique : {e}")

# ==============================================================================
# IMPORT MANUEL SI ÉCHEC
# ==============================================================================

if df is None or df.empty:

    st.warning("""
    Impossible de récupérer automatiquement les données MASI.

    Vous pouvez importer un fichier CSV ou Excel contenant :

    - Date
    - Close
    """)

    uploaded_file = st.file_uploader(
        "Importer un historique MASI",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:

        try:

            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)

            else:
                df = pd.read_excel(uploaded_file)

            if "Date" not in df.columns:
                st.error("Colonne Date introuvable")
                st.stop()

            if "Close" not in df.columns:
                st.error("Colonne Close introuvable")
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

        except Exception as e:

            st.error(
                f"Erreur lecture fichier : {e}"
            )
            st.stop()

    else:
        st.stop()

# ==============================================================================
# VALIDATION
# ==============================================================================

if len(df) < 100:

    st.error(
        "Historique insuffisant. Au moins 100 observations sont recommandées."
    )

    st.stop()

# ==============================================================================
# KPI
# ==============================================================================

last_close = float(df["Close"].iloc[-1])

prev_close = float(df["Close"].iloc[-2])

variation = (
    (last_close / prev_close) - 1
) * 100

ma20 = df["Close"].tail(20).mean()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Dernier Cours",
    f"{last_close:,.2f}"
)

col2.metric(
    "Variation Journalière",
    f"{variation:.2f}%"
)

col3.metric(
    "Moyenne Mobile 20j",
    f"{ma20:,.2f}"
)

# ==============================================================================
# ONGLETS
# ==============================================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Historique",
        "🔮 Prévisions",
        "📈 Backtest",
        "📋 Données"
    ]
)

# ==============================================================================
# HISTORIQUE
# ==============================================================================

with tab1:

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
        title="Historique du MASI",
        xaxis_title="Date",
        yaxis_title="Indice",
        hovermode="x unified",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==============================================================================
# BACKTEST
# ==============================================================================

with tab3:

    st.subheader("Validation Walk Forward")

    try:

        bt, mae, rmse, r2, sigma = (
            walk_forward_validation(df)
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

        fig_bt.update_layout(
            title="Backtest Walk Forward",
            height=600
        )

        st.plotly_chart(
            fig_bt,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Erreur backtest : {e}"
        )

# ==============================================================================
# PREVISIONS
# ==============================================================================

with tab2:

    st.subheader("Prévisions")

    horizon = st.slider(
        "Nombre de séances",
        min_value=1,
        max_value=30,
        value=10
    )

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
                    width=3,
                    dash="dash"
                )
            )
        )

        fig_pred.update_layout(
            title="Prévisions MASI",
            xaxis_title="Date",
            yaxis_title="Indice",
            hovermode="x unified",
            height=650
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

# ==============================================================================
# DONNÉES
# ==============================================================================

with tab4:

    st.dataframe(
        df,
        use_container_width=True
    )

    st.download_button(
        "📥 Télécharger les données",
        df.to_csv(index=False),
        "masi_data.csv",
        "text/csv"
    )

# ==============================================================================
# FOOTER
# ==============================================================================

st.caption(
    """
    Les prévisions reposent sur un modèle d'apprentissage automatique
    validé par Walk-Forward Backtesting.

    Elles ne constituent pas un conseil d'investissement.
    """
)
