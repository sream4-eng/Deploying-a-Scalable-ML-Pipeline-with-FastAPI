import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data_returns_expected_shapes():
    """
    process_data should return numpy arrays with matching rows for X and y.
    """
    df = pd.read_csv("data/census.csv").sample(n=200, random_state=42)

    X, y, encoder, lb = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    assert X.shape[0] == y.shape[0]
    assert X.shape[0] == 200
    assert encoder is not None
    assert lb is not None


def test_train_model_returns_random_forest():
    """
    train_model should return a trained RandomForestClassifier.
    """
    df = pd.read_csv("data/census.csv").sample(n=200, random_state=42)

    X, y, _, _ = process_data(
        df,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=True,
    )

    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_compute_model_metrics_are_valid_range():
    """
    compute_model_metrics should return precision/recall/f1 within [0, 1].
    """
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    p, r, f1 = compute_model_metrics(y_true, y_pred)

    assert 0.0 <= p <= 1.0
    assert 0.0 <= r <= 1.0
    assert 0.0 <= f1 <= 1.0
