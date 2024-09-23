import matplotlib.pyplot as plt

from .interfaces import BaseLassoNetCV
from .utils import confidence_interval, eval_on_path


def plot_path(model, X_test, y_test, *, score_function=None):
    """
    Plot the evolution of the model on the path, namely:
    - lambda
    - number of selected variables
    - score

    Requires to have called model.path(return_state_dicts=True) beforehand.


    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    X_test : array-like
    y_test : array-like
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    """
    # TODO: plot with manually computed score
    score = eval_on_path(
        model, model.path_, X_test, y_test, score_function=score_function
    )
    n_selected = [save.selected.sum() for save in model.path_]
    lambda_ = [save.lambda_ for save in model.path_]

    plt.figure(figsize=(16, 16))

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

    plt.tight_layout()


def plot_cv(model: BaseLassoNetCV, X_test, y_test, *, score_function=None):
    # TODO: plot with manually computed score
    lambda_ = [save.lambda_ for save in model.path_]
    lambdas = [[h.lambda_ for h in p] for p in model.raw_paths_]

    score = eval_on_path(
        model, model.path_, X_test, y_test, score_function=score_function
    )

    plt.figure(figsize=(16, 16))

    plt.subplot(211)
    plt.grid(True)
    first = True
    for sl, ss in zip(lambdas, model.raw_scores_):
        plt.plot(
            sl,
            ss,
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    avg = model.interp_scores_.mean(axis=1)
    ci = confidence_interval(model.interp_scores_)
    plt.plot(
        model.lambdas_,
        avg,
        "g.-",
        markersize=5,
        alpha=0.2,
        label="average cv with 95% CI",
    )
    plt.fill_between(model.lambdas_, avg - ci, avg + ci, color="g", alpha=0.1)
    plt.plot(lambda_, score, "b.-", markersize=5, alpha=0.2, label="test")
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("score")

    plt.subplot(212)
    plt.grid(True)
    first = True
    for sl, path in zip(lambdas, model.raw_paths_):
        plt.plot(
            sl,
            [save.selected.sum() for save in path],
            "r.-",
            markersize=5,
            alpha=0.2,
            label="cross-validation" if first else None,
        )
        first = False
    plt.plot(
        lambda_,
        [save.selected.sum() for save in model.path_],
        "b.-",
        markersize=5,
        alpha=0.2,
        label="test",
    )
    plt.legend()
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")

    plt.tight_layout()
