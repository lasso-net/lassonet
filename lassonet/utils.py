from itertools import zip_longest
from typing import TYPE_CHECKING, Iterable, List

import scipy.stats
import torch

if TYPE_CHECKING:
    from lassonet.interfaces import HistoryItem


def eval_on_path(model, path, X_test, y_test, *, score_function=None):
    if score_function is None:
        score_fun = model.score
    else:
        assert callable(score_function)

        def score_fun(X_test, y_test):
            return score_function(y_test, model.predict(X_test))

    score = []
    for save in path:
        model.load(save.state_dict)
        score.append(score_fun(X_test, y_test))
    return score


if hasattr(torch.Tensor, "scatter_reduce_"):
    # version >= 1.12
    def scatter_reduce(input, dim, index, reduce, *, output_size=None):
        src = input
        if output_size is None:
            output_size = index.max() + 1
        return torch.empty(output_size, device=input.device).scatter_reduce(
            dim=dim, index=index, src=src, reduce=reduce, include_self=False
        )

else:
    scatter_reduce = torch.scatter_reduce


def scatter_logsumexp(input, index, *, dim=-1, output_size=None):
    """Inspired by torch_scatter.logsumexp
    Uses torch.scatter_reduce for performance
    """
    max_value_per_index = scatter_reduce(
        input, dim=dim, index=index, output_size=output_size, reduce="amax"
    )
    max_per_src_element = max_value_per_index.gather(dim, index)
    recentered_scores = input - max_per_src_element
    sum_per_index = scatter_reduce(
        recentered_scores.exp(),
        dim=dim,
        index=index,
        output_size=output_size,
        reduce="sum",
    )
    return max_value_per_index + sum_per_index.log()


def log_substract(x, y):
    """log(exp(x) - exp(y))"""
    return x + torch.log1p(-(y - x).exp())


def confidence_interval(data, confidence=0.95):
    if isinstance(data[0], Iterable):
        return [confidence_interval(d, confidence) for d in data]
    return scipy.stats.t.interval(
        confidence,
        len(data) - 1,
        scale=scipy.stats.sem(data),
    )[1]


def selection_probability(paths: List[List["HistoryItem"]]):
    """Compute the selection probability of each feature at each step.
    The individual curves are smoothed to that they are monotonically decreasing.

    Input
    -----
    paths: List of List of HistoryItem
        The lambda paths must be the same for all models.

    Output
    ------
    prob: torch.Tensor
        Tensor of shape (n_steps, n_features) containing the selection probability
        of each feature at lambda value.
    expected_wrong: tuple of (Tensor, LongTensor)
        Expected number of wrong features.
        (values, indices) where values are the expected number of wrong features
        and indices are the order of the selected features.
    """
    n_models = len(paths)

    prob = []
    selected = torch.ones_like(paths[0][0].selected)
    iterable = zip_longest(
        *[[it.selected for it in path] for path in paths],
        fillvalue=torch.zeros_like(paths[0][0].selected),
    )
    for its in iterable:
        sel = sum(its) / n_models
        selected = torch.minimum(selected, sel)
        prob.append(selected)
    prob = torch.stack(prob)

    expected_wrong = (
        prob.shape[1] * (prob.mean(dim=1, keepdim=True)) ** 2 / (2 * prob - 1)
    )
    expected_wrong[prob <= 0.5] = float("inf")
    return prob, expected_wrong.min(axis=0).values.sort()
