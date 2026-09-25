import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from connectors.investing import load_masi

from models.forecasting import (
    train_model,
    forecast_next_days
)

from models.validation import (
    evaluate
)

st.set_page_config(
    page_title="MASI Forecast",
    layout="wide"
)

st.title("📈 Prévision MASI")

with st.spinner("Chargement des données..."):

    df = load_masi()

if df is None or df.empty:

    st.error(
        "Aucune donnée disponible."
    )

    st.stop()

st.success(
    f"{len(df)} observations chargées"
)

tab1, tab2, tab3 = st.tabs(
    [
        "Historique",
        "Prévision",
        "Données"
    ]
)

with tab1:

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["Close"]
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

with tab2:

    model, X_test, y_test, data = (
        train_model(df)
    )

    mae, r2 = evaluate(
        model,
        X_test,
        y_test
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

    horizon = st.slider(
        "Jours",
        1,
        20,
        5
    )

    forecasts = forecast_next_days(
        model,
        data,
        horizon
    )

    pred_df = pd.DataFrame(
        {
            "Prévision": forecasts
        }
    )

    st.dataframe(
        pred_df,
        use_container_width=True
    )

with tab3:

    st.dataframe(
        df,
        use_container_width=True
    )
