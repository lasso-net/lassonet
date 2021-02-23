import math
from itertools import zip_longest

import torch
from torch import nn
from torch.nn import functional as F

from .prox import inplace_prox


class DeepLasso(nn.Module):
    def __init__(self, *dims, residual_hidden=False, last_layer_bias=True):
        """
        The building block of Deep Lasso.
        first dimension is input
        last dimension is output
        """
        assert len(dims) > 2
        assert dims[-1] == 1

        super().__init__()

        self.dims = dims

        n_hidden = len(dims) - 1
        use_bias = [True] * n_hidden
        use_bias[-1] = last_layer_bias
        n_skip = len(dims) - 2 if residual_hidden else 1

        self.skip_connections = nn.ModuleList(
            [nn.Linear(dims[i], dims[-1], bias=False) for i in range(n_skip)]
        )
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=use_bias[i]) for i in range(n_hidden)]
        )

    def forward(self, inp):
        current_layer = inp
        result = 0
        for beta, theta in zip_longest(self.skip_connections, self.layers):
            if beta is not None:
                result += beta(current_layer)
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                # current_layer = torch.sigmoid(current_layer)
                current_layer = F.relu(current_layer)

        return result + current_layer

    def regularization(self):
        with torch.no_grad():
            return sum(
                torch.norm(skip.weight.data, p=1) for skip in self.skip_connections
            )

    def prox(self, *, lambda_, lambda_bar = 0, M = 1):
        with torch.no_grad():
            for beta, theta in zip(self.skip_connections, self.layers):
                inplace_prox(beta = beta, theta = theta, lambda_ = lambda_,
                        lambda_bar = lambda_bar, M = M)

    def input_mask(self):
        return self.skip_connections[0].weight != 0


class DeepLasso12(nn.Module):
    def __init__(self, *dims, residual_hidden=False):
        """
        first dimension is input
        last dimension is output
        """
        assert len(dims) > 2

        super().__init__()

        self.dims = dims

        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        n_skip = len(dims) - 2 if residual_hidden else 1
        self.skip_connections = nn.ModuleList(
            [nn.Linear(dims[i], dims[-1], bias=False) for i in range(n_skip)]
        )

    def forward(self, inp):
        current_layer = inp
        result = 0
        for beta, theta in zip_longest(self.skip_connections, self.layers):
            if beta is not None:
                result += beta(current_layer)
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                current_layer = F.relu(current_layer)
                # current_layer = torch.sigmoid(current_layer)
        return result + current_layer

    def regularization(self):
        with torch.no_grad():
            return sum(
                torch.norm(skip.weight.data, p=2, dim=0).sum()
                for skip in self.skip_connections
            )

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        with torch.no_grad():
            for beta, theta in zip(self.skip_connections, self.layers):
                inplace_prox(beta, theta, lambda_ = lambda_, lambda_bar =
                        lambda_bar, M = M)

    def input_mask(self):
        return torch.norm(self.skip_connections[0].weight, p=2, dim=0) != 0


class SPCALasso(nn.Module):
    def __init__(self, p, k, l, m):
        """
        This is the SPCA lasso of Overleaf/Write-ups/spca_design_new.tex
        p: input dimension. Also gives the output dimension
        k: number of hidden units inside principal components
        l: number of principal components
        m: number of hidden units used to combine principal components
        """
        super().__init__()

        self.dims = p, k, l, m
        self.beta = nn.Linear(p, l)
        self.theta = nn.Linear(p, l * k)
        self.alpha = nn.Parameter(torch.Tensor(l, k))
        self.intermediate = nn.Linear(l,m)
        self.output = nn.Linear(m,p)
        self.linear_output = nn.Linear(l,p)
        self.reset_parameters()
        self.pc = None

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.alpha, a=math.sqrt(5))

    def forward(self, inp):
        p, k, l, m = self.dims
        principal_components = self.beta(inp) + torch.sum(
            F.relu(self.theta(inp)).view(-1, l, k) * self.alpha, dim=-1
            # torch.sigmoid(self.theta(inp)).view(-1, l, k) * self.alpha, dim=-1
        )
        self.pc = principal_components
        intermediate_layer = self.intermediate(principal_components)
        return self.linear_output(principal_components) +\
    self.output(F.relu(intermediate_layer))
    # self.output(torch.sigmoid(intermediate_layer))

    def regularization(self):
        with torch.no_grad():
            return torch.norm(self.beta.weight.data, p=2, dim=0).sum()

    def prox(self, *, lambda_, lambda_bar=0, M=1):
        with torch.no_grad():
            inplace_prox(self.beta, self.theta, lambda_ = lambda_, lambda_bar =
                    lambda_bar, M = M)

    def input_mask(self):
        return torch.norm(self.beta.weight, p = 2, dim = 0) != 0
