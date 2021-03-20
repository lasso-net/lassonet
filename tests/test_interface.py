from lassonet import LassoNetRegressor, LassoNetClassifier

from sklearn.datasets import load_diabetes, load_digits


def test_regressor():
    X, y = load_diabetes(return_X_y=True)
    model = LassoNetRegressor()
    model.fit(X, y)
    model.score(X, y)


def test_classifier():
    X, y = load_digits(return_X_y=True)
    model = LassoNetClassifier()
    model.fit(X, y)
    model.score(X, y)
