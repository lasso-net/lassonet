#!/usr/bin/env python
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from lassonet import LassoNetCoxRegressor, plot_path

data = Path(__file__).parent / "data"
X = np.genfromtxt(data / "hnscc_x.csv", delimiter=",", skip_header=1)
y = np.genfromtxt(data / "hnscc_y.csv", delimiter=",", skip_header=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LassoNetCoxRegressor(
    hidden_dims=(100,),
    lambda_start=1e-2,
    path_multiplier=1.02,
    gamma=1,
    verbose=True,
    tie_approximation="breslow",
)

path = model.path(X_train, y_train, return_state_dicts=True)

plot_path(model, path, X_test, y_test)
plt.savefig("cox_regression.png")


model = LassoNetCoxRegressor(
    hidden_dims=(100,),
    lambda_start=1e-2,
    path_multiplier=1.02,
    gamma=1,
    verbose=True,
    tie_approximation="efron",
)

path = model.path(X_train, y_train)

plot_path(model, path, X_test, y_test)
plt.savefig("cox_regression_efron.png")
