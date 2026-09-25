from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)


def evaluate(
    model,
    X_test,
    y_test
):

    pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        pred
    )

    r2 = r2_score(
        y_test,
        pred
    )

    return mae, r2
``
