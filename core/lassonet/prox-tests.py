from prox import prox, prox2
import torch

if __name__ == "__main__":

    from random import randrange, random

    for _ in range(10000):
        n = randrange(0, 10)
        k = randrange(0, 10)
        if not n:
            u = torch.randn(size=(k,))
            v = torch.randn(size=(1,))
        else:
            u = torch.randn(size=(k, n))
            v = torch.randn(size=(1, n))
        lambda_ = random()
        lambda_bar = random()
        M = random() 
        beta, theta = prox(v, u, lambda_, lambda_bar, M)
        beta2, theta2 = prox2(v, u, lambda_, lambda_bar, M)
        assert torch.allclose(beta, beta2, rtol=1e-4, atol=1e-7)
        # try:
        #     assert torch.allclose(beta, beta2, rtol=1e-4, atol=1e-7)
        # except AssertionError:
        #     print("beta", beta, "beta2", beta2)
        assert torch.allclose(theta, theta2, rtol=1e-4, atol=1e-7)
