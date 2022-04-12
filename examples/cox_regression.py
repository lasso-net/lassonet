#!/usr/bin/env python
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from lassonet import LassoNetCoxRegressor, plot_path

data = Path(__file__).parent / "HNSCC_data"
X = np.genfromtxt(data / "x_glmnet_2.csv", delimiter=",", skip_header=1)
y = np.genfromtxt(data / "y_glmnet_2.csv", delimiter=",", skip_header=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

model = LassoNetCoxRegressor(
    hidden_dims=(10,),
    eps_start=0.1,
    lambda_start=1e-4,
    path_multiplier=1.02,
    verbose=True,
)

path = model.path(X_train, y_train)

plot_path(model, path, X_test, y_test)
plt.savefig("cox_regression.png")
