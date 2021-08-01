from functools import partial

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


from lassonet import LassoNetRegressor, plot_path


def friedman():
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


X, y = friedman()
X = StandardScaler().fit_transform(X)
y -= y.mean()
y /= y.std()
X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetRegressor(verbose=True, path_multiplier=1.002, hidden_dims=(10, 10))

path = model.path(X_train, y_train)


def score(self, X, y, sample_weight=None):
    y_pred = self.predict(X)
    return np.sqrt(1 - r2_score(y, y_pred, sample_weight=sample_weight))


model.score = partial(score, model)


import matplotlib.pyplot as plt

plot_path(model, path, X_test, y_test)
plt.show()
