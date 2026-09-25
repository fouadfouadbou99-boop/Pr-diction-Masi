import pandas as pd


def load_local():

    try:

        df = pd.read_csv(
            "data/masi_historique.csv"
        )

        return df

    except Exception:

        return pd.DataFrame()
