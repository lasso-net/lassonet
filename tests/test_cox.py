import itertools

import pytest
import torch

from lassonet.cox import CoxPHLoss, concordance_index


def test_concordance_index_tied_times_row_order():
    # A and B fail at the same time and cannot be compared with each other.
    # Only (A, C) and (B, C) are comparable, and both are concordant.
    risk = [1, 2, 0]
    time = [5, 5, 9]
    event = [1, 1, 0]
    for perm in itertools.permutations(range(3)):
        assert (
            concordance_index(
                [risk[i] for i in perm],
                [time[i] for i in perm],
                [event[i] for i in perm],
            )
            == 1.0
        )


def test_concordance_index_value():
    # (0, 1): comparable and discordant, (0, 2) and (1, 2): concordant
    assert concordance_index([1, 2, 0], [1, 2, 3], [1, 1, 0]) == pytest.approx(2 / 3)


@pytest.mark.parametrize("method", CoxPHLoss.allowed)
def test_cox_loss_tied_times_row_order(method):
    # an event and a censored row share the duration 5
    log_h = torch.tensor([0.3, 1.2, -0.5, 0.8])
    y = torch.tensor([[5.0, 1.0], [5.0, 0.0], [9.0, 1.0], [2.0, 1.0]])
    loss = CoxPHLoss(method)
    expected = loss(log_h, y)
    for perm in itertools.permutations(range(4)):
        p = torch.tensor(perm)
        assert loss(log_h[p], y[p]) == pytest.approx(expected.item())


def test_cox_loss_censored_tie_in_risk_set():
    # the censored row at duration 5 must be in the risk set of the event at 5
    log_h = torch.tensor([0.3, 1.2])
    y = torch.tensor([[5.0, 1.0], [5.0, 0.0]])
    expected = torch.logsumexp(log_h, dim=0) - log_h[0]
    assert CoxPHLoss("breslow")(log_h, y) == pytest.approx(expected.item())
