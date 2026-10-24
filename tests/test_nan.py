import numpy as np
import pytest
import torch
from scipy.stats import norm

from lassonet.interfaces import LassoNetIntervalRegressor


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
