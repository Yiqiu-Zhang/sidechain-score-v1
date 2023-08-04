import tqdm
import torch
import numpy as np

X_MIN = torch.tensor(1e-5)
X_N = 5000   # relative to pi

SIGMA_MIN= torch.tensor(3e-3)
SIGMA_MAX = torch.tensor(2)
SIGMA_N = 5000  # relative to pi

def grad(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):# 为什么需要这个for loop 啊 还是一个累加的关系
        p_ += (x + 2 * np.pi * i) / sigma ** 2 * np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

def p(x, sigma, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + 2 * np.pi * i) ** 2 / 2 / sigma ** 2)
    return p_

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi # [0, pi]
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi # [0, 2pi]

p_ = p(x, sigma[:, None], N=100)
score_ = grad(x, sigma[:, None], N=100) / p_
score_ = torch.tensor(score_)
p_ = torch.tensor(p_)

def score(x: torch.Tensor, # [B,N,4]
          sigma: torch.Tensor): # sigma
    x = (x + torch.pi) % (2 * torch.pi) - torch.pi #[-pi pi]
    sign = torch.sign(x)
    x = torch.log(torch.abs(x) / torch.pi)
    x = (x - torch.log(X_MIN)) / (0 - torch.log(X_MIN)) * X_N
    x = torch.round(torch.clip(x, 0, X_N)).to(int)
    sigma = torch.log(sigma / torch.pi)
    sigma = (sigma - torch.log(SIGMA_MIN)) / (torch.log(SIGMA_MAX) - torch.log(SIGMA_MIN)) * SIGMA_N
    sigma = torch.round(torch.clip(sigma, 0, SIGMA_N)).to(int)
    return -sign.to('cuda') * score_[sigma.to('cuda'), x.to('cuda')].to('cuda')

def sample(sigma):
    out = sigma * torch.randn(*sigma.shape)
    out = (out + torch.pi) % (2 * torch.pi) - torch.pi
    return out

sigma = torch.tensor(sigma)
score_norm_ = score(
    sample(sigma[None].repeat_interleave(10000, 0).flatten()),
    sigma[None].repeat_interleave(10000, 0).flatten()).reshape(10000, -1)
score_norm_ = (score_norm_ ** 2).mean(0)

def score_norm(sigma):
    sigma = torch.log(sigma / torch.pi)
    sigma = (sigma - torch.log(SIGMA_MIN)) / (torch.log(SIGMA_MAX) - torch.log(SIGMA_MIN)) * SIGMA_N
    sigma = torch.round(torch.clip(sigma, 0, SIGMA_N)).to(int)
    return score_norm_[sigma]