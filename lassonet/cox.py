"""
implement CoxPHLoss
"""

__all__ = ["CoxPHLoss", "concordance_index"]

import torch
from sortedcontainers import SortedList

from .utils import log_substract, scatter_logsumexp


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. """

    allowed = ("breslow", "efron")

    def __init__(self, method="breslow"):
        assert method in self.allowed, "Method must be one of %s" % self.allowed
        self.method = method

    def forward(self, log_h, y):
        log_h = log_h.flatten()

        durations, events = y.T

        # sort input
        durations, idx = durations.sort(descending=True)
        log_h = log_h[idx]
        events = events[idx]

        event_ind = events.nonzero().flatten()

        # numerator
        log_num = log_h[event_ind].mean()

        # logcumsumexp of events
        event_lcse = torch.logcumsumexp(log_h, dim=0)[event_ind]

        # number of events for each unique risk set
        _, tie_inverses, tie_count = torch.unique_consecutive(
            durations[event_ind], return_counts=True, return_inverse=True
        )

        # position of last event (lowest duration) of each unique risk set
        tie_pos = tie_count.cumsum(axis=0) - 1

        # logcumsumexp by tie for each event
        event_tie_lcse = event_lcse[tie_pos][tie_inverses]

        if self.method == "breslow":
            log_den = event_tie_lcse.mean()

        elif self.method == "efron":
            # based on https://bydmitry.github.io/efron-tensorflow.html

            # logsumexp of ties, duplicated within tie set
            tie_lse = scatter_logsumexp(event_ind, tie_inverses, dim=0)[tie_inverses]
            # multiply (add in log space) with corrective factor
            aux = torch.ones_like(tie_inverses)
            aux[tie_pos[:-1] + 1] -= tie_count[:-1]
            event_id_in_tie = torch.cumsum(aux, dim=0) - 1
            tie_lse += torch.log(event_id_in_tie) - torch.log(tie_count[tie_inverses])

            # denominator
            log_den = log_substract(event_tie_lcse, tie_lse).mean()

        # loss is negative log likelihood
        return log_den - log_num


def concordance_index(risk, time, event):
    """
    O(n log n) implementation of https://square.github.io/pysurvival/metrics/c_index.html
    """
    assert len(risk) == len(time) == len(event)
    n = len(risk)
    order = sorted(range(n), key=time.__getitem__)
    past = SortedList()
    num = 0
    den = 0
    for i in order:
        num += len(past) - past.bisect_right(risk[i])
        den += len(past)
        if event[i]:
            past.add(risk[i])
    return num / den
