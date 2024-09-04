import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from lassonet import LassoNetRegressor, plot_path


def load(s):
    return np.loadtxt(s, delimiter=",")


X_train = load("xtrain.csv")
y_train = load("ytrain.csv")
X_val = load("xvalid.csv")
y_val = load("yvalid.csv")
X_test = load("xtest.csv")
y_test = load("ytest.csv")

X_train, X_val, X_test = np.split(
    StandardScaler().fit_transform(np.concatenate((X_train, X_val, X_test))), 3
)

y = np.concatenate((y_train, y_val))
y_mean = y.mean()
y_std = y.std()

for y in [y_train, y_val, y_test]:
    y -= y_mean
    y /= y_std


def rrmse(y, y_pred):
    return np.sqrt(1 - r2_score(y, y_pred))


if __name__ == "__main__":

    model = LassoNetRegressor(
        path_multiplier=1.001,
        M=100_000,
        hidden_dims=(10, 10),
        torch_seed=0,
    )
    path = model.path(
        X_train, y_train, X_val=X_val, y_val=y_val, return_state_dicts=True
    )
    print(
        "rrmse:",
        min(rrmse(y_test, model.load(save).predict(X_test)) for save in path),
    )
    plot_path(model, path, X_test, y_test, score_function=rrmse)
    plt.show()
