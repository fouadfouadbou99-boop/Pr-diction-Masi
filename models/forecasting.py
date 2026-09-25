import pandas as pd

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


def train_model(df):

    data = create_features(df)

    split = int(
        len(data) * 0.80
    )

    train = data.iloc[:split]
    test = data.iloc[split:]

    X_train = train[FEATURES]
    y_train = train["Target"]

    X_test = test[FEATURES]
    y_test = test["Target"]

    model = HistGradientBoostingRegressor(
        max_iter=400,
        max_depth=6,
        learning_rate=0.03,
        random_state=42
    )

    model.fit(X_train, y_train)

    return (
        model,
        X_test,
        y_test,
        data
    )


def forecast_next_days(
    model,
    data,
    horizon
):

    forecasts = []

    current = data.copy()

    for _ in range(horizon):

        row = current.iloc[-1]

        X = pd.DataFrame(
            [row[FEATURES]]
        )

        pred = model.predict(X)[0]

        forecasts.append(pred)

    return forecasts
