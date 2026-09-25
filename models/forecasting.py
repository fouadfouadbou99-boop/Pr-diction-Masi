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

    if len(data) < 50:
        raise ValueError(
            "Pas assez de données pour entraîner le modèle."
        )

    X = data[FEATURES]

    y = data["Target"]

    model = HistGradientBoostingRegressor(
        max_iter=300,
        max_depth=6,
        learning_rate=0.03,
        random_state=42
    )

    model.fit(X, y)

    return model
