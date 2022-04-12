"""
implement CoxPHLoss
"""

__all__ = ["CoxPHLoss", "concordance_index"]

import torch
from sortedcontainers import SortedList


class CoxPHLoss(torch.nn.Module):
    """Loss for CoxPH model. """

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
        event_lgse = torch.logcumsumexp(log_h, dim=0)[event_ind]

        # number of events for each unique risk set
        _, event_tie_count = torch.unique_consecutive(
            durations[event_ind], return_counts=True
        )

        # position of last event (lowest duration) with of each unique risk set
        event_pos = event_tie_count.cumsum(axis=0) - 1

        # denominator
        log_den = (event_tie_count * event_lgse[event_pos]).mean()

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
