from connectors.casablanca import (
    load_casablanca
)

from connectors.local import (
    load_local
)


def load_masi():

    print(
        "Tentative chargement Casablanca..."
    )

    try:

        df = load_casablanca()

        if (
            df is not None
            and
            not df.empty
        ):
            print(
                f"Casablanca OK : {len(df)} lignes"
            )
            return df

    except Exception as e:

        print(
            f"Erreur Casablanca : {e}"
        )

    print(
        "Passage au fichier local..."
    )

    try:

        df = load_local()

        if (
            df is not None
            and
            not df.empty
        ):
            print(
                f"Local OK : {len(df)} lignes"
            )
            return df

    except Exception as e:

        print(
            f"Erreur local : {e}"
        )

    return None
