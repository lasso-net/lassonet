#!/usr/bin/env python
# coding: utf-8

"""
Lassonet Demo Notebook - PyTorch

This notebook illustrates the Lassonet method for
feature selection on a classification task.
We will run Lassonet over [the Mice Dataset](https://archive.ics.uci.edu/ml/datasets/Mice%20Protein%20Expression).
This dataset consists of protein expression levels measured in the cortex of normal and trisomic mice who had been exposed to different experimental conditions. Each feature is the expression level of one protein.
"""
# First we import a few necessary packages


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from lassonet import LassoNetClassifier


import pandas as pd


def load_mice():
    df = pd.read_csv("Data_Cortex_Nuclear.csv")
    y = list(df[df.columns[78:81]].itertuples(False))
    classes = {lbl: i for i, lbl in enumerate(sorted(set(y)))}
    y = np.array([classes[lbl] for lbl in y])
    feats = df.columns[1:78]
    X = df[feats].fillna(df.groupby(y)[feats].transform("mean")).values
    X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
    return X, y


X, y = load_mice()


X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetClassifier(eps=1e-3, n_lambdas=1000)
path = model.path(X_train, y_train)

n_selected = []
accuracy = []

for save in path:
    model.load(save.state_dict)
    n_selected.append(save.selected.sum())
    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))

fig = plt.figure(figsize=(9, 6))
plt.grid(True)
plt.plot(n_selected, accuracy, "o-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")
plt.title("Classification accuracy")
plt.savefig("accuracy.png")
