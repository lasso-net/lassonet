[![PyPI version](https://badge.fury.io/py/lassonet.svg)](https://badge.fury.io/py/lassonet)
[![Downloads](https://static.pepy.tech/badge/lassonet)](https://pepy.tech/project/lassonet)

# LassoNet

LassoNet is a new family of models to incorporate feature selection and neural networks.

LassoNet works by adding a linear skip connection from the input features to the output. A L1 penalty (LASSO-inspired) is added to that skip connection along with a constraint on the network so that whenever a feature is ignored by the skip connection, it is ignored by the whole network.

<a href="https://www.youtube.com/watch?v=bbqpUfxA_OA" target="_blank"><img src="https://raw.githubusercontent.com/lasso-net/lassonet/master/docs/images/video_screenshot.png" width="450" alt="Promo Video"/></a>

## Installation

```
pip install lassonet
```

## Usage

We have designed the code to follow scikit-learn's standards to the extent possible (e.g. [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)).

```
from lassonet import LassoNetClassifierCV 
model = LassoNetClassifierCV() # LassoNetRegressorCV
path = model.fit(X_train, y_train)
print("Best model scored", model.score(X_test, y_test))
print("Lambda =", model.best_lambda_)
```

You should always try to give normalized data to LassoNet as it uses neural networks under the hood.

You can read the full [documentation](https://lassonet.ml/lassonet/api/) or read the [examples](https://github.com/lasso-net/lassonet/tree/master/examples) that cover all features.

## Features

- regression, classification and [Cox regression](https://en.wikipedia.org/wiki/Proportional_hazards_model) with `LassoNetRegressor`, `LassoNetClassifier` and `LassoNetCoxRegressor`.
- cross-validation with `LassoNetRegressorCV`, `LassoNetClassifierCV` and `LassoNetCoxRegressorCV`
- group feature selection with the `groups` argument
- `lambda_start="auto"` heuristic

Note that cross-validation, group feature selection and automatic `lambda_start` selection have not been published in papers, you can read the code or [post as issue](https://github.com/lasso-net/lassonet/issues/new) to request more details.

We are currently working (among others) on adding support for convolution layers, auto-encoders and online logging of experiments.

## Cross-validation

The original paper describes how to train LassoNet along a regularization path. This requires the user to manually select a model from the path and made the `.fit()` method useless since the resulting model is always empty. This feature is still available with the `.path()` method for any model or the `lassonet_path` function and returns a list of checkpoints that can be loaded with `.load()`.

Since then, we integrated support for cross-validation (5-fold by default) in the estimators whose name ends with `CV`. For each fold, a path is trained. The best regularization value is then chosen to maximize the average performance over all folds. The model is then retrained on the whole training dataset to reach that regularization.

## Website

LassoNet's website is [https://lassonet.ml](https://lassonet.ml). It contains many useful references including the paper, live talks and additional documentation.

## References

- Lemhadri, Ismael, Feng Ruan, Louis Abraham, and Robert Tibshirani. "LassoNet: A Neural Network with Feature Sparsity." Journal of Machine Learning Research 22, no. 127 (2021). [pdf](https://arxiv.org/pdf/1907.12207.pdf) [bibtex](https://github.com/lasso-net/lassonet/blob/master/citation.bib)
- Yang, Xuelin, Louis Abraham, Sejin Kim, Petr Smirnov, Feng Ruan, Benjamin Haibe-Kains, and Robert Tibshirani. "FastCPH: Efficient Survival Analysis for Neural Networks." arXiv preprint arXiv:2208.09793 (2022). [pdf](https://arxiv.org/pdf/2208.09793.pdf)
