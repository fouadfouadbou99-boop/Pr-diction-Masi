import pandas as pd
import numpy as np

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

    X_train = train[FEATURES]

    y_train = train["Target"]

    model = HistGradientBoostingRegressor(
        max_iter=500,
        max_depth=6,
        learning_rate=0.03,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model
