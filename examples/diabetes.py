#!/usr/bin/env python

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from lassonet import LassoNetRegressor, plot_path


dataset = load_diabetes()
X = dataset.data
y = dataset.target
_, true_features = X.shape
# add dummy feature
X = np.concatenate([X, np.random.randn(*X.shape)], axis=1)
feature_names = list(dataset.feature_names) + ["fake"] * true_features

# standardize
X = StandardScaler().fit_transform(X)
y = scale(y)


X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LassoNetRegressor(
    hidden_dims=(10,),
    verbose=True,
)
path = model.path(X_train, y_train)

plot_path(model, path, X_test, y_test)

plt.savefig("diabetes.png")

plt.clf()

n_features = X.shape[1]
importances = model.feature_importances_.numpy()
order = np.argsort(importances)[::-1]
importances = importances[order]
ordered_feature_names = [feature_names[i] for i in order]
color = np.array(["g"] * true_features + ["r"] * (n_features - true_features))[order]


plt.subplot(211)
plt.bar(
    np.arange(n_features),
    importances,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
colors = {"real features": "g", "fake features": "r"}
labels = list(colors.keys())
handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
plt.legend(handles, labels)
plt.ylabel("Feature importance")

_, order = np.unique(importances, return_inverse=True)

plt.subplot(212)
plt.bar(
    np.arange(n_features),
    order + 1,
    color=color,
)
plt.xticks(np.arange(n_features), ordered_feature_names, rotation=90)
plt.legend(handles, labels)
plt.ylabel("Feature order")

plt.savefig("diabetes-bar.png")
