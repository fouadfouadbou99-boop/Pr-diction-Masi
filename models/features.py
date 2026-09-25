import pandas as pd
import numpy as np


def compute_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def create_features(df):

    try:

        data = df.copy()

        if "Close" not in data.columns:
            return pd.DataFrame()

        data["Close"] = pd.to_numeric(
            data["Close"],
            errors="coerce"
        )

        data = data.dropna(subset=["Close"])

        data["Return_1"] = data["Close"].pct_change()

        data["Return_5"] = data["Close"].pct_change(5)

        data["Return_20"] = data["Close"].pct_change(20)

        data["MA5"] = data["Close"].rolling(5).mean()

        data["MA20"] = data["Close"].rolling(20).mean()

        data["MA50"] = data["Close"].rolling(50).mean()

        data["EMA20"] = (
            data["Close"]
            .ewm(span=20, adjust=False)
            .mean()
        )

        data["Momentum"] = (
            data["Close"]
            - data["Close"].shift(10)
        )

        data["Volatility"] = (
            data["Return_1"]
            .rolling(20)
            .std()
        )

        data["RSI"] = compute_rsi(
            data["Close"]
        )

        data["Target"] = (
            data["Close"]
            .shift(-1)
        )

        data = data.dropna()

        return data

    except Exception as e:

        print(
            f"Erreur create_features : {e}"
        )

        return pd.DataFrame()
