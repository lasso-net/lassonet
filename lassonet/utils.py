import matplotlib.pyplot as plt
from functools import partial


def plot_path(model, path, X_test, y_test, *, score_function=None):
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
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    """
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    n_selected = []
    score = []
    lambda_ = []
    for save in path:
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        score.append(score_fun(X_test, y_test))
        lambda_.append(save.lambda_)

    plt.figure(figsize=(8, 8))

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

def plot_cox_loss(model, path, X_test, y_test, *, score_function=None):
    """
    Plot the evolution of the model on the path with cox loss, namely:
    - lambda
    - number of selected variables
    - score
    - training loss

    Parameters
    ==========
    model : LassoNetClassifier or LassoNetRegressor
    path
        output of model.path
    X_test : array-like
    y_test : array-like
    score_function : function or None
        if None, use score_function=model.score
        score_function must take as input X_test, y_test
    is_twin : whether to plot two loss and scale the y axis of two loss in the fourth figure
        True: plot two loss (original loss and alternative loss)
        False: plot one loss (original loss)
    ori_loss_type : the name of the original loss, namely 'cox' or 'alternative_cox'
        'cox' is the primary loss we are using to train the model
        'alternative_cox' is the loss we want to compare, which can be customized 
    """
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    n_selected = []
    score = []
    lambda_ = []
    max_score = -1
    max_score_lambda = 0
    for i in range(1, len(path) - 1):
        save = path[i]
        model.load(save.state_dict)
        n_selected.append(save.selected.sum())
        cur_score = score_fun(X_test, y_test)
        score.append(cur_score)
        if cur_score > max_score:
            max_score = cur_score
            max_score_lambda = save.lambda_
        lambda_.append(save.lambda_)
    print("max_score:", max_score, "at lambda=", max_score_lambda)
    plt.figure(figsize=(8, 14))

    ax1 = plt.subplot(311)
    ax1.title.set_text('C-index vs number of selected feature')
    plt.grid(True)
    plt.plot(n_selected, score, ".-")
    plt.xlabel("number of selected features")
    plt.ylabel("C-index") # "score"

    ax2 = plt.subplot(312)
    ax2.title.set_text('C-index vs lambda')
    plt.grid(True)
    plt.plot(lambda_, score, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("C-index") # "score"

    ax3 = plt.subplot(313)
    ax3.title.set_text('number of selected features vs lambda ')
    plt.grid(True)
    plt.plot(lambda_, n_selected, ".-")
    plt.xlabel("lambda")
    plt.xscale("log")
    plt.ylabel("number of selected features")
    plt.tight_layout()