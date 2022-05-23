#!/usr/bin/env python
from pathlib import Path
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold

from sksurv import datasets
import os


import pandas as pd
from tqdm import tqdm
from lassonet.interfaces import LassoNetCoxRegressorCV


DATA_PATH = Path(__file__).parent / "data"


def transform_one_hot(input_matrix, col_name):
    one_hot_col = pd.get_dummies(input_matrix[col_name], prefix=col_name)
    input_matrix = input_matrix.drop([col_name], axis=1)
    input_matrix = input_matrix.join(one_hot_col)
    return input_matrix


def load_data(dataset):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    if dataset == "breast_cancer":
        X, y = datasets.load_breast_cancer()
        di_er = {"negative": 0, "positive": 1}
        di_grade = {
            "poorly differentiated": -1,
            "intermediate": 0,
            "well differentiated": 1,
            "unkown": 0,
        }
        X = X.replace({"er": di_er, "grade": di_grade})
        X.to_csv(DATA_PATH / f"{dataset}_x.csv", index=False)
        y_temp = pd.DataFrame(y, columns=["t.tdm", "e.tdm"])
        di_event = {True: 1, False: 0}
        y_temp = y_temp.replace({"e.tdm": di_event})
        y_temp.to_csv(DATA_PATH / f"{dataset}_y.csv", index=False)

    elif dataset == "fl_chain":
        X, y = datasets.load_flchain()
        di_mgus = {"no": 0, "yes": 1}
        X = X.replace({"mgus": di_mgus, "creatinine": {np.nan: 0}})
        col_names = ["chapter", "sex", "sample.yr", "flc.grp"]
        for col_name in col_names:
            X = transform_one_hot(X, col_name)
        X.to_csv(DATA_PATH / f"{dataset}_x.csv", index=False)
        y_temp = pd.DataFrame(y, columns=["futime", "death"])
        di_event = {True: 0, False: 1}
        y_temp = y_temp.replace({"death": di_event})
        y_temp.to_csv(DATA_PATH / f"{dataset}_y.csv", index=False)

    elif dataset == "whas500":
        X, y = datasets.load_whas500()
        X.to_csv(DATA_PATH / f"{dataset}_x.csv", index=False)
        y_temp = pd.DataFrame(y, columns=["lenfol", "fstat"])
        di_event = {True: 1, False: 0}
        y_temp = y_temp.replace({"fstat": di_event})
        y_temp.to_csv(DATA_PATH / f"{dataset}_y.csv", index=False)

    elif dataset == "veterans_lung_cancer":
        X, y = datasets.load_veterans_lung_cancer()
        col_names = ["Celltype", "Prior_therapy", "Treatment"]
        for col_name in col_names:
            X = transform_one_hot(X, col_name)
        X.to_csv(DATA_PATH / f"{dataset}_x.csv", index=False)
        y_temp = pd.DataFrame(y, columns=["Survival_in_days", "Status"])
        di_event = {False: 0, True: 1}
        y_temp = y_temp.replace({"Status": di_event})
        y_temp.to_csv(DATA_PATH / f"{dataset}_y.csv", index=False)

    elif dataset == "hnscc":
        pass
    else:
        raise ValueError("Dataset unknown")

    X = np.genfromtxt(DATA_PATH / f"{dataset}_x.csv", delimiter=",", skip_header=1)
    y = np.genfromtxt(DATA_PATH / f"{dataset}_y.csv", delimiter=",", skip_header=1)
    X = preprocessing.StandardScaler().fit(X).transform(X)
    return X, y


def run(X, y, *, random_state, dump=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=random_state, stratify=y[:, 1], test_size=0.20
    )
    if dump:
        pd.DataFrame(X_train).to_csv(
            DATA_PATH / f"{dataset}_x_train_{random_state}.csv", index=False
        )
        pd.DataFrame(y_train).to_csv(
            DATA_PATH / f"{dataset}_y_train_{random_state}.csv", index=False
        )
        pd.DataFrame(X_test).to_csv(
            DATA_PATH / f"{dataset}_x_test_{random_state}.csv", index=False
        )
        pd.DataFrame(y_test).to_csv(
            DATA_PATH / f"{dataset}_y_test_{random_state}.csv", index=False
        )

    cv = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state).split(
            X_train, y_train[:, 1]
        )
    )

    model = LassoNetCoxRegressorCV(
        tie_approximation="breslow",
        hidden_dims=(32,),
        path_multiplier=1.01,
        cv=cv,
        torch_seed=random_state,
    )
    model.fit(X_train, y_train)

    tqdm.write(f"train: {model.best_cv_score_:.04f} ± {model.best_cv_score_std_:.04f}")
    tqdm.write(f"features: {model.best_selected_.sum().item()}")
    test_score = model.score(X_test, y_test)
    tqdm.write(f"test: {test_score:.04f}")
    return test_score


if __name__ == "__main__":
    """
    run with python3 script.py dataset
    """

    import sys

    dataset = sys.argv[1]
    assert dataset in {
        "fl_chain",
        "breast_cancer",
        "whas500",
        "veterans_lung_cancer",
        "HNSCC",
    }

    X, y = load_data(dataset)
    scores = np.array(
        [
            run(X, y, random_state=random_state)
            for random_state in tqdm(range(10), desc="Running with different seeds")
        ]
    )
    print(f"Final score: {scores.mean()} ± {scores.std()}")


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
