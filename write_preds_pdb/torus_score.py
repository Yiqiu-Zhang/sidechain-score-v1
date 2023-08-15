import tqdm
import torch
import numpy as np
import os
X_MIN = torch.tensor(1e-5)
X_N = 5000   # relative to pi

SIGMA_MIN= torch.tensor(3e-3)
SIGMA_MAX = torch.tensor(2)
SIGMA_N = 5000  # relative to pi

def grad(x, sigma, periodic, N=10, ):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += (x + periodic * i) / sigma ** 2 * np.exp(-(x + periodic * i) ** 2 / 2 / sigma ** 2)
    return p_

def p(x, sigma, periodic, N=10):
    p_ = 0
    for i in tqdm.trange(-N, N + 1):
        p_ += np.exp(-(x + periodic * i) ** 2 / 2 / sigma ** 2)
    return p_

def sample(sigma):
    out = sigma * torch.randn(*sigma.shape)
    out = (out + torch.pi) % (2 * torch.pi) - torch.pi
    return out

def score_int(x, sigma_idx):
    x = (x + torch.pi) % (2 * torch.pi) - torch.pi #[-pi pi]
    sign = torch.sign(x)
    x = torch.log(torch.abs(x) / torch.pi)
    x = (x - torch.log(X_MIN)) / (0 - torch.log(X_MIN)) * X_N
    x = torch.round(torch.clip(x, 0, X_N)).to(int)

    return -sign * score_[sigma_idx, x], -sign * score_pi[sigma_idx, x]

x = 10 ** np.linspace(np.log10(X_MIN), 0, X_N + 1) * np.pi # [0, pi]
sigma = 10 ** np.linspace(np.log10(SIGMA_MIN), np.log10(SIGMA_MAX), SIGMA_N + 1) * np.pi # [0, 2pi]

sigma = torch.tensor(sigma)

if os.path.exists('.p.npy'):
    p_ = np.load('.p.npy')
    score_ = np.load('.score.npy')
    score_norm_ = np.load('.score_norm.npy')

    p_pi = np.load('.p_pi.npy')
    score_pi = np.load('.score_pi.npy')
    score_norm_pi = np.load('.score_norm_pi.npy')

else:
    p_ = p(x, sigma[:, None], periodic = 2 *np.pi, N=100)
    np.save('.p.npy', p_)

    score_ = grad(x, sigma[:, None], periodic = 2 *np.pi, N=100) / p_
    np.save('.score.npy', score_)

    score_norm_, score_norm_pi = score_int(sample(sigma[None].repeat_interleave(10000, 0).flatten()),
                           torch.arange(0,SIGMA_N+1).repeat(10000))
    score_norm_ = (score_norm_.reshape(10000, -1) ** 2).mean(0)
    np.save('.score_norm.npy', score_norm_)


    p_pi = p(x, sigma[:, None], periodic = np.pi, N=100)
    np.save('.p_pi.npy', p_)
    score_pi = grad(x, sigma[:, None], periodic= np.pi, N=100) / p_pi
    np.save('.score_pi.npy', score_pi)
    score_norm_pi = (score_norm_pi.reshape(10000, -1) ** 2).mean(0)
    np.save('.score_norm_pi.npy', score_norm_)


p_ = torch.tensor(p_).to('cuda')
score_ = torch.tensor(score_).to('cuda')
score_norm_ = torch.tensor(score_norm_).to('cuda')

score_pi = torch.tensor(score_pi).to('cuda')
p_pi = torch.tensor(p_pi).to('cuda')
score_norm_pi = torch.tensor(score_norm_pi).to('cuda')


def score(x: torch.Tensor, # [B,N,4]
          sigma_idx: torch.Tensor, # sigma_idx
          chi_pi_periodic):
    x = (x + torch.pi) % (2 * torch.pi) - torch.pi #[-pi pi]
    sign = torch.sign(x)
    x = torch.log(torch.abs(x) / torch.pi)
    x = (x - torch.log(X_MIN)) / (0 - torch.log(X_MIN)) * X_N
    x = torch.round(torch.clip(x, 0, X_N)).to(int)

    score = torch.where(chi_pi_periodic, score_pi[sigma_idx, x].to(x.device),
                        score_[sigma_idx, x].to(x.device))

    return -sign * score

def score_norm(sigma_idx,
               chi_pi_periodic):

    score_norm = torch.where(chi_pi_periodic, score_norm_pi[sigma_idx].to(chi_pi_periodic.device),
                             score_norm_[sigma_idx].to(chi_pi_periodic.device))
    return score_norm