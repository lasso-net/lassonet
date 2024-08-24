#!/usr/bin/env python
"""
Install required packages with:

    pip install scipy joblib tqdm_joblib
"""


from pathlib import Path
from time import time

import numpy as np
import pandas as pd
import sksurv.datasets
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from lassonet import LassoNetCoxRegressorCV, plot_cv
from lassonet.utils import confidence_interval

DATA_PATH = Path(__file__).parent / "data"
DATA_PATH.mkdir(exist_ok=True)

FIGURES_PATH = Path() / "cox_figures"
FIGURES_PATH.mkdir(exist_ok=True)


def transform_one_hot(input_matrix, col_name):
    one_hot_col = pd.get_dummies(input_matrix[col_name], prefix=col_name)
    input_matrix = input_matrix.drop([col_name], axis=1)
    input_matrix = input_matrix.join(one_hot_col)
    return input_matrix


def dump(array, name):
    pd.DataFrame(array).to_csv(DATA_PATH / name, index=False)


def gen_data(dataset):
    if dataset == "breast":
        X, y = sksurv.datasets.load_breast_cancer()
        di_er = {"negative": 0, "positive": 1}
        di_grade = {
            "poorly differentiated": -1,
            "intermediate": 0,
            "well differentiated": 1,
            "unkown": 0,
        }
        X = X.replace({"er": di_er, "grade": di_grade})
        y_temp = pd.DataFrame(y, columns=["t.tdm", "e.tdm"])
        di_event = {True: 1, False: 0}
        y_temp = y_temp.replace({"e.tdm": di_event})
        y = y_temp

    elif dataset == "fl_chain":
        X, y = sksurv.datasets.load_flchain()
        di_mgus = {"no": 0, "yes": 1}
        X = X.replace({"mgus": di_mgus, "creatinine": {np.nan: 0}})
        col_names = ["chapter", "sex", "sample.yr", "flc.grp"]
        for col_name in col_names:
            X = transform_one_hot(X, col_name)
        y_temp = pd.DataFrame(y, columns=["futime", "death"])
        di_event = {True: 0, False: 1}
        y_temp = y_temp.replace({"death": di_event})
        y = y_temp

    elif dataset == "whas500":
        X, y = sksurv.datasets.load_whas500()
        y_temp = pd.DataFrame(y, columns=["lenfol", "fstat"])
        di_event = {True: 1, False: 0}
        y_temp = y_temp.replace({"fstat": di_event})
        y = y_temp

    elif dataset == "veterans":
        X, y = sksurv.datasets.load_veterans_lung_cancer()
        col_names = ["Celltype", "Prior_therapy", "Treatment"]
        for col_name in col_names:
            X = transform_one_hot(X, col_name)
        y_temp = pd.DataFrame(y, columns=["Survival_in_days", "Status"])
        di_event = {False: 0, True: 1}
        y_temp = y_temp.replace({"Status": di_event})
        y = y_temp
    elif dataset == "hnscc":
        raise ValueError("Dataset exists")
    else:
        raise ValueError("Dataset unknown")

    dump(X, f"{dataset}_x.csv")
    dump(y, f"{dataset}_y.csv")


def load_data(dataset):
    path_x = DATA_PATH / f"{dataset}_x.csv"
    path_y = DATA_PATH / f"{dataset}_y.csv"
    if not (path_x.exists() and path_y.exists()):
        gen_data(dataset)
    X = np.genfromtxt(path_x, delimiter=",", skip_header=1)
    y = np.genfromtxt(path_y, delimiter=",", skip_header=1)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    return X, y


def run(
    dataset,
    hidden_dims,
    path_multiplier,
    *,
    random_state,
    tie_approximation="breslow",
    dump_splits=False,
    verbose=False,
):
    X, y = load_data(dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, stratify=y[:, 1], test_size=0.20
    )

    if dump_splits:
        for array, name in [
            (X_train, "x_train"),
            (y_train, "y_train"),
            (X_test, "x_test"),
            (y_test, "y_test"),
        ]:
            dump(array, f"{dataset}_{name}_{random_state}.csv")

    cv = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            X_train, y_train[:, 1]
        )
    )

    model = LassoNetCoxRegressorCV(
        tie_approximation=tie_approximation,
        hidden_dims=hidden_dims,
        path_multiplier=path_multiplier,
        cv=cv,
        torch_seed=random_state,
        verbose=verbose,
    )
    model.path(X_train, y_train)
    plot_cv(model, X_test, y_test)
    plt.savefig(
        FIGURES_PATH
        / (
            f"cox-cv-{dataset}-{random_state}"
            f"-{model.hidden_dims}-{model.path_multiplier}.jpg"
        ),
        dpi=300,
    )

    test_score = model.score(X_test, y_test)

    return test_score


def experiment(dataset, hidden_dims, path_multiplier, n_runs, n_jobs):
    start = time()
    with tqdm_joblib(desc=f"Running on {dataset}", total=n_runs):
        scores = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(run)(
                    dataset,
                    hidden_dims=hidden_dims,
                    path_multiplier=path_multiplier,
                    tie_approximation=tie_approximation,
                    random_state=random_state,
                )
                for random_state in range(n_runs)
            )
        )
    scores_str = np.array2string(
        scores, separator=", ", formatter={"float_kind": "{:.2f}".format}
    )
    log = (
        f"Dataset: {dataset}\n"
        f"Arch: {hidden_dims}\n"
        f"Path multiplier: {path_multiplier}\n"
        f"Score: {scores.mean():.04f} "
        f"± {confidence_interval(scores):.04f} "
        f"with {n_runs} runs\n"
        f"Raw scores: {scores_str}\n"
        f"Running time: {time() - start:.00f}s with {n_jobs} processors\n"
        f"Running time per run per cpu: {(time() - start) * n_jobs / n_runs:.00f}s\n"
        f"{'-' * 50}\n"
    )

    tqdm.write(log)
    with open("cox_experiments.log", "a") as f:
        print(log, file=f)


if __name__ == "__main__":
    """
    run with python3 script.py dataset [method]

    dataset=all runs all experiments

    method can be "breslow" or "efron" (default "efron")
    """

    import sys

    dataset = sys.argv[1]
    tie_approximation = sys.argv[2] if len(sys.argv) > 2 else "efron"
    if dataset == "all":
        datasets = ["breast", "whas500", "veterans", "hnscc"]
        verbose = False
    else:
        datasets = [dataset]
        verbose = 1

    N_RUNS = 15
    N_JOBS = 5  # set to a divisor of `n_runs` for maximal efficiency

    for hidden_dims in [(16, 16), (32,), (32, 16), (64,)]:
        for path_multiplier in [1.01, 1.02]:
            for dataset in datasets:
                experiment(
                    dataset=dataset,
                    hidden_dims=hidden_dims,
                    path_multiplier=path_multiplier,
                    n_runs=N_RUNS,
                    n_jobs=N_JOBS,
                )


# import optuna

# def objective(trial: optuna.Trial):
#     model = LassoNetCoxRegressorCV(
#         tie_approximation="breslow",
#         hidden_dims=(trial.suggest_int("hidden_dims", 8, 128),),
#         path_multiplier=1.01,
#         M=trial.suggest_float("M", 1e-3, 1e3),
#         cv=cv,
#         torch_seed=random_state,
#         verbose=False,
#     )
#     model.fit(X_train, y_train)
#     test_score = model.score(X_test, y_test)
#     trial.set_user_attr("score std", model.best_cv_score_std_)
#     trial.set_user_attr("test score", test_score)
#     print("test score", test_score)
#     return model.best_cv_score_


# if __name__ == "__main__":
#     study = optuna.create_study(
#         storage="sqlite:///optuna.db",
#         study_name="fastcph-lassonet",
#         direction="maximize",
#         load_if_exists=True,
#     )
#     study.optimize(objective, n_trials=100)
