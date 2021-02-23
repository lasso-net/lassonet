import torch
from torch.nn import functional as F


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)


def _find_w(v, u, lambda_, lambda_bar, M):
    """
    v has shape (1,) or (1, batches)
    u has shape (k,) or (k, batches)
    """
    vshape = v.shape
    if len(vshape) == 1:
        v = v.view(-1, 1)
        u = u.view(-1, 1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape
    m = torch.arange(k + 1.0).view(-1, 1)
    zeros = torch.zeros(1, batch)

    x = torch.abs(v) / M + torch.cat([zeros, torch.cumsum(u_abs_sorted, dim=0)])
    w = soft_threshold(lambda_bar * m + lambda_ / M, x) / (m + 1 / M ** 2)

    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)
    return torch.gather(w, 0, idx).view(*vshape)


def prox(v, u, lambda_, lambda_bar, M):
    """
    v has shape (1,) or (1, batches)
    u has shape (k,) or (k, batches)
    """
    w = _find_w(v, u, lambda_, lambda_bar, M)
    beta = sign_binary(v) * w / M
    theta = u.sign() * torch.min(soft_threshold(lambda_bar, u.abs()), w)
    return beta, theta


def prox2(v, u, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    
    we want |u|_inf <= ||v||_2 for every batch
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    x = F.relu(1 - a_s / norm_v) / (1 + s * M ** 2)

    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star


def inplace_prox(beta, theta, lambda_, lambda_bar, M):
    beta.weight.data, theta.weight.data = prox2(
        beta.weight.data, theta.weight.data, lambda_ = lambda_, lambda_bar =
        lambda_bar, M = M
    )
