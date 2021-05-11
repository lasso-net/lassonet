import matplotlib.pyplot as plt


def plot_path(model, path, X_test, y_test):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    """
    n_selected = []
    score = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        score.append(model.score(X_test, y_test))
        lambda_.append(save.lambda_)

    plt.figure(figsize=(12, 12))

    plt.subplot(311)
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("score")

    plt.subplot(312)
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(313)
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
