import pandas as pd
import requests

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}


def clean_df(df):

    rename_map = {
        "Séance": "Date",
        "Date": "Date",
        "Clôture": "Close",
        "Cours": "Close",
        "Dernier": "Close",
        "Valeur": "Close"
    }

    df.rename(columns=rename_map, inplace=True)

    if "Date" not in df.columns:
        return pd.DataFrame()

    if "Close" not in df.columns:
        return pd.DataFrame()

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce",
        dayfirst=True
    )

    df["Close"] = (
        df["Close"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    return (
        df.dropna()
          .sort_values("Date")
          .drop_duplicates("Date")
          .reset_index(drop=True)
    )


def load_casablanca():

    urls = [
        "https://www.casablanca-bourse.com/fr/indices/masi",
        "https://www.casablanca-bourse.com/fr/live-market"
    ]

    for url in urls:

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                timeout=20
            )

            tables = pd.read_html(response.text)

            for table in tables:

                df = clean_df(table)

                if len(df) > 50:
                    return df

        except Exception:
            continue

    return pd.DataFrame()
