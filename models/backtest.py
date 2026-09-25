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

    if data.empty:

        raise ValueError(
            "Pas assez de données après création des indicateurs."
        )

    if len(data) < 30:

        raise ValueError(
            f"Seulement {len(data)} lignes disponibles."
        )

    actuals = []
    forecasts = []

    start = max(
        20,
        int(len(data) * 0.70)
    )

    for i in range(start, len(data) - 1):

        train = data.iloc[:i]

        test = data.iloc[i:i + 1]

        model = HistGradientBoostingRegressor(
            max_iter=100,
            random_state=42
        )

        model.fit(
            train[FEATURES],
            train["Target"]
        )

        pred = model.predict(
            test[FEATURES]
        )[0]

        actuals.append(
            float(
                test["Target"].iloc[0]
            )
        )

        forecasts.append(
            float(pred)
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
