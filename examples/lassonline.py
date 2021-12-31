#!/usr/bin/env python

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from lassonet import LassoNetRegressor


dataset = load_diabetes()
X = dataset.data
y = dataset.target
_, true_features = X.shape

# standardize
X = StandardScaler().fit_transform(X)
y = scale(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetRegressor(
    hidden_dims=(10,), eps_start=0.1, online_logging="lassonline example"
)
model.path(X_train, y_train)
