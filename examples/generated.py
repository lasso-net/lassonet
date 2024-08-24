from functools import partial

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lassonet import LassoNetRegressor, plot_path


def linear():
    p = 10
    n = 400
    coef = np.concatenate([np.random.choice([-1, 1], size=p), [0] * p])
    X = np.random.randn(n, 2 * p)

    linear = X.dot(coef)
    noise = np.random.randn(n)

    y = linear + noise
    return X, y


def strong_linear():
    p = 10
    n = 400
    coef = np.concatenate([np.random.choice([-1, 1], size=p), [0] * p])
    X = np.random.randn(n, 2 * p)

    linear = X.dot(coef)
    noise = np.random.randn(n)
    x1, x2, x3, *_ = X.T
    nonlinear = 2 * (x1**3 - 3 * x1) + 4 * (x2**2 * x3 - x3)
    y = 6 * linear + 8 * noise + nonlinear
    return X, y


def friedman_lockout():
    p = 200
    n = 1000
    X = np.random.rand(n, p)
    y = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    return X, y


for generator in [linear, strong_linear, friedman_lockout]:
    X, y = generator()
    X = StandardScaler().fit_transform(X)
    y -= y.mean()
    y /= y.std()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = LassoNetRegressor(verbose=True, path_multiplier=1.01, hidden_dims=(10, 10))

    path = model.path(X_train, y_train)
    import matplotlib.pyplot as plt

    def score(self, X, y, sample_weight=None):
        y_pred = self.predict(X)
        return np.sqrt(1 - r2_score(y, y_pred, sample_weight=sample_weight))

    model.score = partial(score, model)

    plot_path(model, path, X_test, y_test)
    plt.show()
