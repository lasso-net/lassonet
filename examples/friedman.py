import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


from lassonet import LassoNetRegressor, plot_path


def friedman(linear_terms=True):
    n = 1000
    p = 200
    X = np.random.rand(n, p)
    y = 10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
    if linear_terms:
        y += 10 * X[:, 3] + 5 * X[:, 4]
    return X, y


np.random.seed(0)
X, y = friedman(linear_terms=True)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y)

y_std = 0.5 * y_train.std()
y_train += np.random.randn(*y_train.shape) * y_std

for y in [y_train, y_test]:
    y -= y.mean()
    y /= y.std()


def rrmse(y, y_pred):
    return np.sqrt(1 - r2_score(y, y_pred))


for path_multiplier in [1.01, 1.001]:
    print("path_multiplier:", path_multiplier)
    for M in [10, 100, 1_000, 10_000, 100_000]:
        print("M:", M)
        model = LassoNetRegressor(
            hidden_dims=(10, 10),
            random_state=0,
            torch_seed=0,
            path_multiplier=path_multiplier,
            M=M,
        )
        path = model.path(X_train, y_train)
        print(
            "rrmse:",
            min(rrmse(y_test, model.load(save).predict(X_test)) for save in path),
        )
        plot_path(model, path, X_test, y_test, score_function=rrmse)
        plt.savefig(f"friedman_path({path_multiplier})_M({M}).jpg")

path_multiplier = 1.001
print("path_multiplier:", path_multiplier)
for M in [100, 1_000, 10_000, 100_000]:
    print("M:", M)
    model = LassoNetRegressor(
        hidden_dims=(10, 10),
        random_state=0,
        torch_seed=0,
        path_multiplier=path_multiplier,
        M=M,
        backtrack=True,
    )
    path = model.path(X_train, y_train)
    print(
        "rrmse:",
        min(rrmse(y_test, model.load(save).predict(X_test)) for save in path),
    )
    plot_path(model, path, X_test, y_test, score_function=rrmse)
    plt.savefig(f"friedman_path({path_multiplier})_M({M})_backtrack.jpg")


for path_multiplier in [1.01, 1.001]:
    M = 100_000
    print("path_multiplier:", path_multiplier)
    print("M:", M)
    model = LassoNetRegressor(
        hidden_dims=(10, 10),
        random_state=0,
        torch_seed=0,
        path_multiplier=path_multiplier,
        M=M,
        patience=100,
        n_iters=1000,
    )
    path = model.path(X_train, y_train)
    print(
        "rrmse:",
        min(rrmse(y_test, model.load(save).predict(X_test)) for save in path),
    )
    plot_path(model, path, X_test, y_test, score_function=rrmse)
    plt.savefig(f"friedman_path({path_multiplier})_M({M})_long.jpg")

# if __name__ == "__main__":

#     model = LassoNetRegressor(verbose=True, path_multiplier=1.01, hidden_dims=(10, 10))
#     path = model.path(X_train, y_train)

#     plot_path(model, path, X_test, y_test, score_function=rrmse)
#     plt.show()
