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
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from lassonet import LassoNetClassifier


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

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test, y_pred))
    lambda_.append(save.lambda_)


fig = plt.figure(figsize=(12, 12))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, accuracy, ".-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, accuracy, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("classification accuracy")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("miceprotein.png")
