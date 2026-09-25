import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.ensemble import (
    HistGradientBoostingRegressor
)

from models.features import (
    create_features
)

FEATURES = [
    "Close",
    "Return_1",
    "Return_5",
    "Return_20",
    "MA5",
    "MA20",
    "MA50",
    "EMA20",
    "Momentum",
    "RSI",
    "Volatility"
]


def walk_forward_validation(df):

    data = create_features(df)

    if len(data) < 80:
        raise ValueError(
            f"Données insuffisantes ({len(data)} lignes)."
        )

    actuals = []
    forecasts = []

    start = max(
        50,
        int(len(data) * 0.7)
    )

    for i in range(start, len(data) - 1):

        train = data.iloc[:i]

        test = data.iloc[i:i + 1]

        model = HistGradientBoostingRegressor(
            max_iter=200,
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["Target"]
        )

        prediction = model.predict(
            test[FEATURES]
        )[0]

        actuals.append(
            float(
                test["Target"].iloc[0]
            )
        )

        forecasts.append(
            float(prediction)
        )

    bt = pd.DataFrame(
        {
            "Actual": actuals,
            "Forecast": forecasts
        }
    )

    mae = mean_absolute_error(
        bt["Actual"],
        bt["Forecast"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            bt["Actual"],
            bt["Forecast"]
        )
    )

    r2 = r2_score(
        bt["Actual"],
        bt["Forecast"]
    )

    sigma = (
        bt["Actual"]
        - bt["Forecast"]
    ).std()

    return (
        bt,
        mae,
        rmse,
        r2,
        sigma
    )
