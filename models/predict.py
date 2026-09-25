import pandas as pd

from models.features import (
    create_features
)

from models.forecasting import (
    FEATURES
)


def recursive_forecast(
    model,
    df,
    horizon,
    sigma
):

    history = df.copy()

    forecasts = []

    last_date = history["Date"].iloc[-1]

    for i in range(horizon):

        feature_df = create_features(
            history
        )

        if feature_df.empty:

            raise ValueError(
                "Impossible de calculer les indicateurs."
            )

        row = feature_df.iloc[-1]

        X = pd.DataFrame(
            [row[FEATURES]]
        )

        prediction = model.predict(X)[0]

        future_date = (
            last_date
            + pd.offsets.BDay(i + 1)
        )

        forecasts.append(
            {
                "Date": future_date,
                "Prediction": float(prediction)
            }
        )

        new_row = pd.DataFrame(
            {
                "Date": [future_date],
                "Close": [prediction]
            }
        )

        history = pd.concat(
            [history, new_row],
            ignore_index=True
        )

    forecast_df = pd.DataFrame(
        forecasts
    )

    forecast_df["Lower95"] = (
        forecast_df["Prediction"]
        - 1.96 * sigma
    )

    forecast_df["Upper95"] = (
        forecast_df["Prediction"]
        + 1.96 * sigma
    )

    return forecast_df
