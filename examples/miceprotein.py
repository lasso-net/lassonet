#!/usr/bin/env python
# coding: utf-8

"""
We run Lassonet over [the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression).
This dataset consists of protein expression levels measured in the cortex of normal and
trisomic mice who had been exposed to different experimental conditions.
Each feature is the expression level of one protein.
"""

from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from lassonet import LassoNetClassifier, plot_path
from lassonet.interfaces import LassoNetClassifierCV

X, y = fetch_openml(name="miceprotein", return_X_y=True)
# Fill missing values with the mean
X = SimpleImputer().fit_transform(X)
# Convert labels to scalar
y = LabelEncoder().fit_transform(y)

# standardize
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)


model = LassoNetClassifierCV()
path = model.fit(X_train, y_train)
print("Best model scored", model.score(X_test, y_test))
print("Lambda =", model.best_lambda_)

model = LassoNetClassifier()
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein.png")

model = LassoNetClassifier(dropout=0.5)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_dropout.png")

model = LassoNetClassifier(hidden_dims=(100, 100))
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep.png")

model = LassoNetClassifier(hidden_dims=(100, 100), gamma=0.01)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep_l2_weak.png")

model = LassoNetClassifier(hidden_dims=(100, 100), gamma=0.1)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep_l2_strong.png")

model = LassoNetClassifier(hidden_dims=(100, 100), gamma=1)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep_l2_super_strong.png")

model = LassoNetClassifier(hidden_dims=(100, 100), dropout=0.5)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep_dropout.png")

model = LassoNetClassifier(hidden_dims=(100, 100), backtrack=True, dropout=0.5)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_deep_dropout_backtrack.png")

model = LassoNetClassifier(batch_size=64)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_64.png")

model = LassoNetClassifier(backtrack=True)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_backtrack.png")

model = LassoNetClassifier(batch_size=64, backtrack=True)
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_backtrack_64.png")

model = LassoNetClassifier(class_weight=[0.1, 0.2, 0.3, 0.1, 0.3, 0, 0, 0])
path = model.path(X_train, y_train)
plot_path(model, path, X_test, y_test)
plt.savefig("miceprotein_weighted.png")
