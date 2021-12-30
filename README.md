[![PyPI version](https://badge.fury.io/py/lassonet.svg)](https://badge.fury.io/py/lassonet)

# LassoNet

This project is about performing feature selection in neural networks.
At the moment, we support fully connected feed-forward neural networks.
LassoNet is based on the work presented in [this paper](https://arxiv.org/abs/1907.12207) ([bibtex here for citation](https://github.com/lasso-net/lassonet/blob/master/citation.bib)).
Here is a [link](https://www.youtube.com/watch?v=bbqpUfxA_OA) to the promo video:

<a href="https://www.youtube.com/watch?v=bbqpUfxA_OA" target="_blank"><img src="https://raw.githubusercontent.com/lasso-net/lassonet/master/docs/images/video_screenshot.png" width="450" alt="Promo Video"/></a>

### Code

We have designed the code to follow scikit-learn's standards to the extent possible (e.g. [linear_model.Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html)).

To install it,

```
pip install lassonet
```

Our plan is to add more functionality that help users understand the important features in neural networks.

### Online logging

We are working on an experimental feature allowing you to log your experiments online.

If you wish to activate it, run:

```py
import lassonet.online
lassonet.online.configure()
```

or use the `online_logging` argument in `LassoNetRegressor` and `LassoNetClassifier`.

### Website

LassoNet's website is [https://lassonet.ml](https://lassonet.ml). It contains many useful references including the paper, live talks and additional documentation.
