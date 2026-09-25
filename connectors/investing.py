from connectors.casablanca import load_casablanca
from connectors.local import load_local


def load_masi():

    loaders = [
        load_casablanca,
        load_local
    ]

    for loader in loaders:

        try:

            df = loader()

            if not df.empty:
                return df

        except Exception:
            pass

    return None
