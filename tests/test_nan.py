from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.stats import norm

from lassonet import LassoNetRegressor, NonFiniteLossError
from lassonet.interfaces import LassoNetIntervalRegressor
from lassonet.prox import prox


def test_nonfinite_loss_raises():
    X = np.full((20, 3), np.inf)
    y = np.zeros(20)
    model = LassoNetRegressor(hidden_dims=(5,), n_iters=(2, 2))
    with pytest.raises(NonFiniteLossError):
        model.fit(X, y)


def _fake_path(selected):
    return [
        SimpleNamespace(lambda_=lambda_, selected=torch.tensor(s))
        for lambda_, s in zip((1.0, 2.0, 3.0), selected)
    ]


def test_stability_selection_draws_new_subsample(monkeypatch):
    model = LassoNetRegressor()
    calls = []

    def subsample_path(X, y, lambda_seq=None):
        calls.append(lambda_seq)
        if len(calls) == 2:
            raise NonFiniteLossError("Loss is nan.")
        return _fake_path([[True, True], [True, False], [False, False]])

    monkeypatch.setattr(model, "_stability_selection_path", subsample_path)
    with pytest.warns(UserWarning, match="Drawing a new subsample"):
        *_, paths, prob = model.stability_selection(None, None, n_models=3)
    assert len(paths) == 3
    assert len(calls) == 5


def test_stability_selection_stops_after_max_retries(monkeypatch):
    model = LassoNetRegressor()

    def subsample_path(X, y, lambda_seq=None):
        raise NonFiniteLossError("Loss is nan.")

    monkeypatch.setattr(model, "_stability_selection_path", subsample_path)
    with pytest.warns(UserWarning), pytest.raises(NonFiniteLossError):
        model.stability_selection(None, None, max_retries=2)


@pytest.mark.parametrize("lambda_", [0.0, 1e-3])
def test_prox_zero_skip_weight(lambda_):
    # a zero skip weight with non-zero first-layer weights gave nan (lasso-net#75)
    v = torch.zeros(1, 1)
    u = torch.full((10, 1), 0.5)
    beta, theta = prox(v, u, lambda_=lambda_, lambda_bar=0, M=10)
    assert (beta == 0).all()
    assert (theta == 0).all()


def interval_loss(y, mu, sigma):
    y = torch.tensor(y, dtype=torch.float32)
    mu = torch.tensor(mu, dtype=torch.float32).view(-1, 1).requires_grad_()
    loss = LassoNetIntervalRegressor.criterion((mu, torch.tensor(sigma)), y)
    loss.backward()
    return loss.item(), mu.grad


def test_interval_criterion_matches_likelihood():
    sigma = 0.7
    # left (T <= U), interval (U < T <= V), right (T > V), interval in the upper tail
    y = [
        [2.0, 2.0, 1, 0, 0],
        [1.0, 3.0, 0, 1, 0],
        [5.0, 5.0, 0, 0, 1],
        [np.exp(2.2), np.exp(2.3), 0, 1, 0],
    ]
    mu = [0.5, 0.2, 1.0, -1.2]
    loss, grad = interval_loss(y, mu, sigma)

    def z(t, m):
        return (np.log(t) - m) / sigma

    p = [
        norm.cdf(z(2.0, 0.5)),
        norm.cdf(z(3.0, 0.2)) - norm.cdf(z(1.0, 0.2)),
        norm.sf(z(5.0, 1.0)),
        norm.sf(z(np.exp(2.2), -1.2)) - norm.sf(z(np.exp(2.3), -1.2)),
    ]
    assert loss == pytest.approx(-np.mean(np.log(np.array(p) + 1e-8)), rel=1e-5)
    assert torch.isfinite(grad).all()


@pytest.mark.parametrize(
    "y, sigma",
    [
        ([[1e-5, 1e-5, 1, 0, 0]], 0.5),  # far outside: bounded loss
        ([[2.0, 0.0, 1, 0, 0]], 0.5),  # unused V: 0 * nan was nan
        ([[2.0, 2.0, 0, 1, 0]], 0.5),  # empty interval
        ([[1.0, 1.1, 0, 1, 0]], 1e8),  # F(U) = F(V) in float32
    ],
)
def test_interval_criterion_is_finite(y, sigma):
    loss, grad = interval_loss(y, [1.0], sigma)
    assert 0 <= loss <= -np.log(1e-8) + 1e-3
    assert torch.isfinite(grad).all()
