import pandas as pd


def load_local():

    try:

        df = pd.read_csv(
            "data/masi_historique.csv"
        )

        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        )

        df["Close"] = pd.to_numeric(
            df["Close"],
            errors="coerce"
        )

        df = df.dropna()

        df = (
            df.sort_values("Date")
            .reset_index(drop=True)
        )

        return df

    except Exception as e:

        print(
            f"Erreur fichier local : {e}"
        )

        return pd.DataFrame()
`
