#!/usr/bin/env python
# coding: utf-8

"""
Lassonet Demo Notebook - PyTorch

This notebook illustrates the Lassonet method for
feature selection on a classification task.
We will run Lassonet over
[the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression).
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

X, y = fetch_openml(name="miceprotein", return_X_y=True)
# Fill missing values with the mean
X = SimpleImputer().fit_transform(X)
# Convert labels to scalar
y = LabelEncoder().fit_transform(y)

# standardize
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetClassifier(verbose=True)
path = model.path(X_train, y_train)


plot_path(model, path, X_test, y_test)

plt.savefig("miceprotein.png")


model = LassoNetClassifier(batch_size=64, backtrack=True, verbose=True)
path = model.path(X_train, y_train)


plot_path(model, path, X_test, y_test)

plt.savefig("miceprotein_backtrack_64.png")
