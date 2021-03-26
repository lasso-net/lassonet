from math import log
from abc import ABCMeta, abstractmethod, abstractclassmethod
from dataclasses import dataclass
from functools import partial
from typing import List

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split
import torch

from .model import LassoNet


def abstractattr(f):
    return property(abstractmethod(f))


def abstractclsattr(f):
    return property(abstractclassmethod(f))


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    val_loss: float
    regularization: float
    selected: torch.BoolTensor
    n_iters: int


class BaseLassoNet(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        hidden_dims=(100,),
        lambda_=None,
        eps=1e-2,
        n_lambdas=100,
        path_multiplier=None,
        M=10,
        optim=None,
        n_iters=(3000, 400),
        patience=10,
        val_size=0.1,
        device=None,
        random_state=None,
        torch_seed=None,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int, default=(100,)
            Shape of the hidden layers.
        lambda\\_ : None or float, default=None
            Regularization parameter. Not needed for cross-validation estimators.
        eps : float, default=1e-2
            First value on the path. The corresponding `lambda_` will be lambda_max * eps
        n_lambdas : int, default=100
            Number of lambda values to test. Note this is an absolute upper bound,
            as we stop when all coefficients are 0.
        path_multiplier : float or None
            Multiplicative factor (:math:`1 + \\epsilon`) to increase
            the penalty parameter over the path
            If None, it will be computed from n_lambdas.
            If not None, it will prevail over the value of n_lambdas.
        M : float, default=10.0
            Hierarchy parameter.
        optim : torch optimizer or tuple of 2 optimizers, default=None
            Optimizer for initial training and path computation.
            Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
        n_iters : int or pair of int, default=(3000, 400)
            Maximum number of training epochs for initial training and path computation.
            This is an upper-bound on the effective number of epochs, since the model
            uses early stopping.
        patience : int or pair of int, default=10
            Number of epochs to wait without improvement during early stopping.
        val_size : float, default=0.1
            Proportion of data to use for early stopping.
        device : torch device, default=None
            Device on which to train the model using PyTorch.
            Default: GPU if available else CPU
        random_state
            Random state for cross-validation
        torch_seed
            Torch state for model random initialization

        """

        self.hidden_dims = hidden_dims
        self.lambda_ = lambda_
        self.eps = eps
        if path_multiplier is None:
            path_multiplier = (1 / eps) ** (1 / n_lambdas)
        else:
            n_lambdas = -log(eps) // log(path_multiplier)
        self.n_lambdas = n_lambdas
        self.path_multiplier = path_multiplier
        self.M = M
        if optim is None:
            optim = (
                partial(torch.optim.Adam, lr=1e-3),
                partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
            )
        if isinstance(optim, torch.optim.Optimizer):
            optim = (optim, optim)
        self.optim_init, self.optim_path = optim
        if isinstance(n_iters, int):
            n_iters = (n_iters, n_iters)
        self.n_iters_init, self.n_iters_path = n_iters
        if isinstance(patience, int):
            patience = (patience, patience)
        self.patience_init, self.patience_path = patience
        self.val_size = val_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.random_state = random_state
        self.torch_seed = torch_seed

        self.model = None

    @abstractmethod
    def _convert_y(self, y) -> torch.TensorType:
        """Convert y to torch tensor"""
        raise NotImplementedError

    @abstractmethod
    def _output_shape(self, y):
        """Number of model outputs"""
        raise NotImplementedError

    @abstractclsattr
    def last_layer_bias(cls):
        raise NotImplementedError

    @abstractclsattr
    def criterion(cls):
        raise NotImplementedError

    def _init_model(self, X, y):
        """Create a torch model"""
        output_shape = self._output_shape(y)
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model = LassoNet(
            X.shape[1],
            *self.hidden_dims,
            output_shape,
            last_layer_bias=self.last_layer_bias,
        ).to(self.device)

    def _cast_input(self, X, y=None):
        X = torch.FloatTensor(X).to(self.device)
        if y is None:
            return X
        y = self._convert_y(y)
        return X, y

    def fit(self, X, y):
        """Train the model.
        This method cannot be called if you did not provide a value of lambda\\_
        and should be reserved for production environments.
        """
        assert self.lambda_ is not None, "You cannot call fit without providing lambda_"
        self.path(X, y, self.lambda_)
        return self

    def _train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs,
        lambda_,
        optimizer,
        patience=None,
    ):
        model = self.model

        def obj_fun():
            with torch.no_grad():
                model.eval()
                return (
                    self.criterion(model(X_val), y_val).item()
                    + lambda_ * model.regularization().item()
                )

        best_obj = obj_fun()
        epochs_since_best_obj = 0

        for epoch in range(epochs):

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(model(X_train), y_train)
                loss.backward()
                return loss

            model.train()
            optimizer.step(closure)
            if lambda_:
                model.prox(lambda_=lambda_, M=self.M)

            obj = obj_fun()
            if obj < best_obj:
                best_obj = obj
                epochs_since_best_obj = 0
            if patience is not None and epochs_since_best_obj == patience:
                break
            epochs_since_best_obj += 1
        return lambda_, epoch + 1, obj

    @abstractmethod
    def predict(self, X):
        pass

    @property
    def coef_(self):
        """Coefficients of the skip layer
        This allows to use `sklearn.feature_selection.SelectFromModel \
<https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFromModel.html#sklearn.feature_selection.SelectFromModel>`_
        """
        return self.model.skip.weight.cpu().numpy()

    def path(self, X, y, lambda_=None) -> List[HistoryItem]:
        # TODO: disable save_state
        # TODO: doc
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size)
        X_train, y_train = self._cast_input(X_train, y_train)
        X_val, y_val = self._cast_input(X_val, y_val)

        hist = []

        def register(hist, lambda_, n_iters, val_loss):
            hist.append(
                HistoryItem(
                    lambda_=lambda_,
                    state_dict=self.model.cpu_state_dict(),
                    val_loss=val_loss,
                    regularization=self.model.regularization().item(),
                    selected=self.model.input_mask(),
                    n_iters=n_iters,
                )
            )

        if self.model is None:
            self._init_model(X_train, y_train)

        register(
            hist,
            *self._train(
                X_train,
                y_train,
                X_val,
                y_val,
                lambda_=0,
                epochs=self.n_iters_init,
                optimizer=self.optim_init(self.model.parameters()),
                patience=self.patience_init,
            ),
        )
        n_samples, _ = X.shape
        lambda_max = (
            torch.norm(torch.tensor(X.T.dot(y)), p=2, dim=0).max().item() / n_samples
        )
        current_lambda = lambda_max * self.eps
        if lambda_ is not None:
            lambda_max = lambda_
        optimizer = self.optim_path(self.model.parameters())

        while self.model.selected_count() != 0:
            current_lambda *= self.path_multiplier
            if current_lambda > lambda_max:
                break
            register(
                hist,
                *self._train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    lambda_=current_lambda,
                    epochs=self.n_iters_path,
                    optimizer=optimizer,
                    patience=self.patience_path,
                ),
            )
        if lambda_ is not None and current_lambda != lambda_:
            register(
                hist,
                *self._train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    lambda_=current_lambda,
                    optimizer=optimizer,
                    patience=self.patience,
                ),
            )

        return hist

    def load(self, state_dict):
        if self.model is None:
            output_shape, input_shape = state_dict["skip.weight"].shape
            self.model = LassoNet(
                input_shape,
                *self.hidden_dims,
                output_shape,
                last_layer_bias=self.last_layer_bias,
            ).to(self.device)

        self.model.load_state_dict(state_dict)


class LassoNetRegressor(
    RegressorMixin,
    MultiOutputMixin,
    BaseLassoNet,
):
    """Use LassoNet as regressor"""

    def _convert_y(self, y):
        y = torch.FloatTensor(y).to(self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    def _output_shape(self, y):
        return y.shape[1]

    last_layer_bias = True
    criterion = torch.nn.MSELoss(reduction="mean")

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class LassoNetClassifier(
    ClassifierMixin,
    BaseLassoNet,
):
    """Use LassoNet as classifier"""

    def _convert_y(self, y) -> torch.TensorType:
        return torch.LongTensor(y).to(self.device)

    def _output_shape(self, y):
        return (y.max() + 1).item()

    last_layer_bias = False
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X)).argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans
