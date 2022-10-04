from itertools import islice
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from functools import partial
import itertools
from typing import List
import warnings

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import check_cv, train_test_split
import torch
from tqdm import tqdm

from .model import LassoNet
from .cox import CoxPHLoss, concordance_index


def abstractattr(f):
    return property(abstractmethod(f))


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    objective: float  # loss + lambda_ * regularization
    loss: float
    val_objective: float  # val_loss + lambda_ * regularization
    val_loss: float
    regularization: float
    l2_regularization: float
    l2_regularization_skip: float
    selected: torch.BoolTensor
    n_iters: int

    def log(item):
        print(
            f"{item.n_iters} epochs, "
            f"val_objective "
            f"{item.val_objective:.2e}, "
            f"val_loss "
            f"{item.val_loss:.2e}, "
            f"regularization {item.regularization:.2e}, "
            f"l2_regularization {item.l2_regularization:.2e}"
        )


class BaseLassoNet(BaseEstimator, metaclass=ABCMeta):
    def __init__(
        self,
        *,
        hidden_dims=(100,),
        lambda_start="auto",
        lambda_seq=None,
        gamma=0.0,
        gamma_skip=0.0,
        path_multiplier=1.02,
        M=10,
        groups=None,
        dropout=0,
        batch_size=None,
        optim=None,
        n_iters=(1000, 100),
        patience=(100, 10),
        tol=0.99,
        backtrack=False,
        val_size=None,
        device=None,
        verbose=1,
        random_state=None,
        torch_seed=None,
        class_weight=None,
        tie_approximation=None,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int, default=(100,)
            Shape of the hidden layers.
        lambda_start : float, default='auto'
            First value on the path. Leave 'auto' to estimate it automatically.
        lambda_seq : iterable of float
            If specified, the model will be trained on this sequence
            of values, until all coefficients are zero.
            The dense model will always be trained first.
            Note: lambda_start and path_multiplier will be ignored.
        gamma : float, default=0.0
            l2 penalization on the network
        gamma : float, default=0.0
            l2 penalization on the skip connection
        path_multiplier : float, default=1.02
            Multiplicative factor (:math:`1 + \\epsilon`) to increase
            the penalty parameter over the path
        M : float, default=10.0
            Hierarchy parameter.
        groups : None or list of lists
            Use group LassoNet regularization.
            `groups` is a list of list such that `groups[i]`
            contains the indices of the features in the i-th group.
        dropout : float, default = None
        batch_size : int, default=None
            If None, does not use batches. Batches are shuffled at each epoch.
        optim : torch optimizer or tuple of 2 optimizers, default=None
            Optimizer for initial training and path computation.
            Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
        n_iters : int or pair of int, default=(1000, 100)
            Maximum number of training epochs for initial training and path computation.
            This is an upper-bound on the effective number of epochs, since the model
            uses early stopping.
        patience : int or pair of int or None, default=10
            Number of epochs to wait without improvement during early stopping.
        tol : float, default=0.99
            Minimum improvement for early stopping: new objective < tol * old objective.
        backtrack : bool, default=False
            If true, ensures the objective function decreases.
        val_size : float, default=None
            Proportion of data to use for early stopping.
            0 means that training data is used.
            To disable early stopping, set patience=None.
            Default is 0.1 for all models except Cox for which training data is used.
            If X_val and y_val are given during training, it will be ignored.
        device : torch device, default=None
            Device on which to train the model using PyTorch.
            Default: GPU if available else CPU
        verbose : int, default=1
        random_state
            Random state for validation
        torch_seed
            Torch state for model random initialization
        class_weight : iterable of float, default=None
            If specified, weights for different classes in training.
            There must be one number per class.
        tie_approximation: str
            Tie approximation for the Cox model, must be one of ("breslow", "efron").
        """
        assert isinstance(hidden_dims, tuple), "`hidden_dims` must be a tuple"
        self.hidden_dims = hidden_dims
        self.lambda_start = lambda_start
        self.lambda_seq = lambda_seq
        self.gamma = gamma
        self.gamma_skip = gamma_skip
        self.path_multiplier = path_multiplier
        self.M = M
        self.groups = groups
        self.dropout = dropout
        self.batch_size = batch_size
        self.optim = optim
        if optim is None:
            optim = (
                partial(torch.optim.Adam, lr=1e-3),
                partial(torch.optim.SGD, lr=1e-3, momentum=0.9),
            )
        if isinstance(optim, partial):
            optim = (optim, optim)
        self.optim_init, self.optim_path = optim
        if isinstance(n_iters, int):
            n_iters = (n_iters, n_iters)
        self.n_iters = self.n_iters_init, self.n_iters_path = n_iters
        if patience is None or isinstance(patience, int):
            patience = (patience, patience)
        self.patience = self.patience_init, self.patience_path = patience
        self.tol = tol
        self.backtrack = backtrack
        if val_size is None:
            # TODO: use a cv parameter following sklearn's interface
            if isinstance(self, LassoNetCoxRegressor):
                val_size = 0
            else:
                val_size = 0.1
        self.val_size = val_size

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.verbose = verbose

        self.random_state = random_state
        self.torch_seed = torch_seed

        self.model = None
        self.class_weight = class_weight
        if self.class_weight is not None:
            assert isinstance(
                self, LassoNetClassifier
            ), "Weighted loss is only for classification"
            self.class_weight = torch.FloatTensor(self.class_weight).to(self.device)
            self.criterion = torch.nn.CrossEntropyLoss(
                weight=self.class_weight, reduction="mean"
            )
        if isinstance(self, LassoNetCoxRegressor):
            assert (
                self.batch_size is None
            ), "Cox regression does not work with mini-batches"
            assert (
                tie_approximation in CoxPHLoss.allowed
            ), f"`tie_approximation` must be one of {CoxPHLoss.allowed}"

            self.criterion = CoxPHLoss(method=tie_approximation)

    @abstractmethod
    def _convert_y(self, y) -> torch.TensorType:
        """Convert y to torch tensor"""
        raise NotImplementedError

    @abstractstaticmethod
    def _output_shape(cls, y):
        """Number of model outputs"""
        raise NotImplementedError

    @abstractattr
    def criterion(cls):
        raise NotImplementedError

    def _init_model(self, X, y):
        """Create a torch model"""
        output_shape = self._output_shape(y)
        if self.class_weight is not None:
            assert output_shape == len(self.class_weight)
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model = LassoNet(
            X.shape[1],
            *self.hidden_dims,
            output_shape,
            groups=self.groups,
            dropout=self.dropout,
        ).to(self.device)

    def _cast_input(self, X, y=None):
        X = torch.FloatTensor(X).to(self.device)
        if y is None:
            return X
        y = self._convert_y(y)
        return X, y

    def fit(self, X, y, *, X_val=None, y_val=None):
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path_ = self.path(X, y, X_val=X_val, y_val=y_val, return_state_dicts=False)
        return self

    def _train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        *,
        batch_size,
        epochs,
        lambda_,
        optimizer,
        return_state_dict,
        patience=None,
    ) -> HistoryItem:
        model = self.model

        def validation_obj():
            with torch.no_grad():
                return (
                    self.criterion(model(X_val), y_val).item()
                    + lambda_ * model.l1_regularization_skip().item()
                    + self.gamma * model.l2_regularization().item()
                    + self.gamma_skip * model.l2_regularization_skip().item()
                )

        best_val_obj = validation_obj()
        epochs_since_best_val_obj = 0
        if self.backtrack:
            best_state_dict = self.model.cpu_state_dict()
            real_best_val_obj = best_val_obj
            real_loss = float("nan")  # if epochs == 0

        n_iters = 0

        n_train = len(X_train)
        if batch_size is None:
            batch_size = n_train
            randperm = torch.arange
        else:
            randperm = torch.randperm
        batch_size = min(batch_size, n_train)

        for epoch in range(epochs):
            indices = randperm(n_train)
            model.train()
            loss = 0
            for i in range(n_train // batch_size):
                # don't take batches that are not full
                batch = indices[i * batch_size : (i + 1) * batch_size]

                def closure():
                    nonlocal loss
                    optimizer.zero_grad()
                    ans = (
                        self.criterion(model(X_train[batch]), y_train[batch])
                        + self.gamma * model.l2_regularization()
                        + self.gamma_skip * model.l2_regularization_skip()
                    )
                    ans.backward()
                    loss += ans.item() * len(batch) / n_train
                    return ans

                optimizer.step(closure)
                model.prox(lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M)

            if epoch == 0:
                # fallback to running loss of first epoch
                real_loss = loss
            val_obj = validation_obj()
            if val_obj < self.tol * best_val_obj:
                best_val_obj = val_obj
                epochs_since_best_val_obj = 0
            else:
                epochs_since_best_val_obj += 1
            if self.backtrack and val_obj < real_best_val_obj:
                best_state_dict = self.model.cpu_state_dict()
                real_best_val_obj = val_obj
                real_loss = loss
                n_iters = epoch + 1
            if patience is not None and epochs_since_best_val_obj == patience:
                break

        if self.backtrack:
            self.model.load_state_dict(best_state_dict)
            val_obj = real_best_val_obj
            loss = real_loss
        else:
            n_iters = epoch + 1
        with torch.no_grad():
            reg = self.model.l1_regularization_skip().item()
            l2_regularization = self.model.l2_regularization()
            l2_regularization_skip = self.model.l2_regularization_skip()
        return HistoryItem(
            lambda_=lambda_,
            state_dict=self.model.cpu_state_dict() if return_state_dict else None,
            objective=loss + lambda_ * reg,
            loss=loss,
            val_objective=val_obj,
            val_loss=val_obj - lambda_ * reg,
            regularization=reg,
            l2_regularization=l2_regularization,
            l2_regularization_skip=l2_regularization_skip,
            selected=self.model.input_mask().cpu(),
            n_iters=n_iters,
        )

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    def path(
        self,
        X,
        y,
        *,
        X_val=None,
        y_val=None,
        lambda_seq=None,
        lambda_max=float("inf"),
        return_state_dicts=True,
        callback=None,
    ) -> List[HistoryItem]:
        """Train LassoNet on a lambda\_ path.
        The path is defined by the class parameters:
        start at `lambda_start` and increment according to `path_multiplier`.
        The path will stop when no feature is being used anymore.
        callback will be called at each step on (model, history)
        """
        assert (X_val is None) == (
            y_val is None
        ), "You must specify both or none of X_val and y_val"
        sample_val = self.val_size != 0 and X_val is None
        if sample_val:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.val_size, random_state=self.random_state
            )
        elif X_val is None:
            X_train, y_train = X_val, y_val = X, y
        else:
            X_train, y_train = X, y
        X_train, y_train = self._cast_input(X_train, y_train)
        X_val, y_val = self._cast_input(X_val, y_val)

        hist: List[HistoryItem] = []

        # always init model
        self._init_model(X_train, y_train)

        hist.append(
            self._train(
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=0,
                epochs=self.n_iters_init,
                optimizer=self.optim_init(self.model.parameters()),
                patience=self.patience_init,
                return_state_dict=return_state_dicts,
            )
        )
        if callback is not None:
            callback(self, hist)
        if self.verbose > 1:
            print("Initialized dense model")
            hist[-1].log()

        optimizer = self.optim_path(self.model.parameters())

        # build lambda_seq
        if lambda_seq is not None:
            pass
        elif self.lambda_seq is not None:
            lambda_seq = self.lambda_seq
        else:

            def _lambda_seq(start):
                while start <= lambda_max:
                    yield start
                    start *= self.path_multiplier

            if self.lambda_start == "auto":
                # divide by 10 for initial training
                self.lambda_start_ = (
                    self.model.lambda_start(M=self.M)
                    / optimizer.param_groups[0]["lr"]
                    / 10
                )
                lambda_seq = _lambda_seq(self.lambda_start_)
            else:
                lambda_seq = _lambda_seq(self.lambda_start)

        # extract first value of lambda_seq
        lambda_seq = iter(lambda_seq)
        lambda_start = next(lambda_seq)

        is_dense = True
        for current_lambda in itertools.chain([lambda_start], lambda_seq):
            if self.model.selected_count() == 0:
                break
            last = self._train(
                X_train,
                y_train,
                X_val,
                y_val,
                batch_size=self.batch_size,
                lambda_=current_lambda,
                epochs=self.n_iters_path,
                optimizer=optimizer,
                patience=self.patience_path,
                return_state_dict=return_state_dicts,
            )
            if is_dense and self.model.selected_count() < X.shape[1]:
                is_dense = False
                if current_lambda / lambda_start < 2:
                    warnings.warn(
                        f"lambda_start={lambda_start:.3f} "
                        f"{'(selected automatically) ' * (self.lambda_start == 'auto')}"
                        "might be too large.\n"
                        f"Features start to disappear at {current_lambda=:.3f}."
                    )

            hist.append(last)
            if callback is not None:
                callback(self, hist)

            if self.verbose > 1:
                print(
                    f"Lambda = {current_lambda:.2e}, "
                    f"selected {self.model.selected_count()} features "
                )
                last.log()

        self.feature_importances_ = self._compute_feature_importances(hist)
        """When does each feature disappear on the path?"""

        return hist

    @staticmethod
    def _compute_feature_importances(path: List[HistoryItem]):
        """When does each feature disappear on the path?

        Parameters
        ----------
        path : List[HistoryItem]

        Returns
        -------
            feature_importances_
        """

        current = path[0].selected.clone()
        ans = torch.full(current.shape, float("inf"))
        for save in islice(path, 1, None):
            lambda_ = save.lambda_
            diff = current & ~save.selected
            ans[diff.nonzero().flatten()] = lambda_
            current &= save.selected
        return ans

    def load(self, state_dict):
        if isinstance(state_dict, HistoryItem):
            state_dict = state_dict.state_dict
        if self.model is None:
            output_shape, input_shape = state_dict["skip.weight"].shape
            self.model = LassoNet(
                input_shape,
                *self.hidden_dims,
                output_shape,
                groups=self.groups,
                dropout=self.dropout,
            ).to(self.device)

        self.model.load_state_dict(state_dict)
        return self


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

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    criterion = torch.nn.MSELoss(reduction="mean")

    def predict(self, X):
        self.model.eval()
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

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def _convert_y(self, y) -> torch.TensorType:
        y = torch.LongTensor(y).to(self.device)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = self.model(self._cast_input(X)).argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            ans = torch.softmax(self.model(self._cast_input(X)), -1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans


class LassoNetCoxRegressor(
    BaseLassoNet,
):
    """Use LassoNet for Cox regression"""

    criterion = None

    def _convert_y(self, y):
        return torch.FloatTensor(y).to(self.device)

    @staticmethod
    def _output_shape(y):
        return 1

    predict = LassoNetRegressor.predict

    def score(self, X_test, y_test):
        """Concordance index"""
        time, event = y_test.T
        risk = self.predict(X_test)
        return concordance_index(risk, time, event)


class BaseLassoNetCV(BaseLassoNet, metaclass=ABCMeta):
    def __init__(self, cv=None, **kwargs):
        """
        See BaseLassoNet for the parameters

        cv : int, cross-validation generator or iterable, default=None
            Determines the cross-validation splitting strategy.
            Default is 5-fold cross-validation.
            See <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.check_cv.html>
        """
        super().__init__(**kwargs)
        self.cv = check_cv(cv)

    def path(
        self,
        X,
        y,
        *,
        return_state_dicts=True,
    ):
        raw_lambdas_ = []
        self.raw_scores_ = []
        self.raw_paths_ = []

        # TODO: parallelize
        for train_index, test_index in tqdm(
            self.cv.split(X, y),
            total=self.cv.get_n_splits(X, y),
            desc="Choosing lambda with cross-validation",
            disable=self.verbose == 0,
        ):
            split_lambdas = []
            split_scores = []
            raw_lambdas_.append(split_lambdas)
            self.raw_scores_.append(split_scores)

            def callback(model, hist):
                split_lambdas.append(hist[-1].lambda_)
                split_scores.append(model.score(X[test_index], y[test_index]))

            path = super().path(
                X[train_index],
                y[train_index],
                return_state_dicts=False,  # avoid memory cost
                callback=callback,
            )
            self.raw_paths_.append(path)

        # build final path
        lambda_ = min(sl[1] for sl in raw_lambdas_)
        lambda_max = max(sl[-1] for sl in raw_lambdas_)
        self.lambdas_ = []
        while lambda_ < lambda_max:
            self.lambdas_.append(lambda_)
            lambda_ *= self.path_multiplier

        # interpolate new scores
        self.interp_scores_ = np.stack(
            [
                np.interp(np.log(self.lambdas_), np.log(sl[1:]), ss[1:])
                for sl, ss in zip(raw_lambdas_, self.raw_scores_)
            ],
            axis=-1,
        )

        # select best lambda based on cross_validation
        best_lambda_idx = np.nanargmax(self.interp_scores_.mean(axis=1))
        self.best_lambda_ = self.lambdas_[best_lambda_idx]
        self.best_cv_scores_ = self.interp_scores_[best_lambda_idx]
        self.best_cv_score_ = self.best_cv_scores_.mean()

        if self.lambda_start == "auto":
            # forget the previously estimated lambda_start
            self.lambda_start_ = self.lambdas_[0]

        # train with the chosen lambda sequence
        path = super().path(
            X,
            y,
            lambda_seq=self.lambdas_[: best_lambda_idx + 1],
            return_state_dicts=return_state_dicts,
        )
        if isinstance(self, LassoNetCoxRegressor) and not path[-1].selected.any():
            # condition to retrain and avoid having 0 feature which gives score 0
            # TODO: handle backtrack in path even when return_state_dicts=False
            path = super().path(
                X,
                y,
                lambda_seq=[h.lambda_ for h in path[1:-1]],
                return_state_dicts=return_state_dicts,
            )
        self.path_ = path

        self.best_selected_ = path[-1].selected
        return path

    def fit(
        self,
        X,
        y,
    ):
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path(X, y, return_state_dicts=False)
        return self


class LassoNetRegressorCV(BaseLassoNetCV, LassoNetRegressor):
    pass


class LassoNetClassifierCV(BaseLassoNetCV, LassoNetClassifier):
    pass


class LassoNetCoxRegressorCV(BaseLassoNetCV, LassoNetCoxRegressor):
    pass


def lassonet_path(X, y, task, *, X_val=None, y_val=None, **kwargs):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values
    task : str, must be "classification" or "regression"
        Task
    X_val : array-like of shape (n_samples, n_features)
        Validation data
    y_val : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Validation values

    See BaseLassoNet for the other parameters.
    """

    class_ = {
        "classification": LassoNetClassifier,
        "regression": LassoNetRegressor,
        "cox": LassoNetCoxRegressor,
    }.get(task)
    if class_ is None:
        raise ValueError('task must be "classification," "regression," or "cox')
    model = class_(**kwargs)
    return model.path(X, y, X_val=X_val, y_val=y_val)
