import torch
from torch import nn
from torch.nn import functional as F

from .prox import inplace_prox, inplace_group_prox


class LassoNet(nn.Module):
    def __init__(self, *dims, groups=None):
        """
        first dimension is input
        last dimension is output
        `groups` is a list of list such that `groups[i]`
        contains the indices of the features in the i-th group

        """
        assert len(dims) > 2
        if groups is not None:
            n_inputs = dims[0]
            all_indices = []
            for g in groups:
                for i in g:
                    all_indices.append(i)
            assert len(all_indices) == n_inputs and set(all_indices) == set(
                range(n_inputs)
            ), f"Groups must be a partition of range(n_inputs={n_inputs})"

        self.groups = groups

        super().__init__()

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        result = self.skip(inp)
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                current_layer = F.relu(current_layer)
        return result + current_layer

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        if self.groups is None:
            with torch.no_grad():
                inplace_prox(
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )
        else:
            with torch.no_grad():
                inplace_group_prox(
                    groups=self.groups,
                    beta=self.skip,
                    theta=self.layers[0],
                    lambda_=lambda_,
                    lambda_bar=lambda_bar,
                    M=M,
                )

    def regularization(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0).sum()

    def input_mask(self):
        with torch.no_grad():
            return torch.norm(self.skip.weight.data, p=2, dim=0) != 0

    def selected_count(self):
        return self.input_mask().sum().item()

    def cpu_state_dict(self):
        return {k: v.detach().clone().cpu() for k, v in self.state_dict().items()}
