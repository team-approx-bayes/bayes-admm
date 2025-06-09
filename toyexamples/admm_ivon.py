import torch
import numpy as np
import math 
from torch import nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '..')
from ivonwithprior import IVON
    
def variational_subproblem_ivon(
    X, 
    y, 
    mprior, 
    sprior, 
    lagr_v, 
    lagr_u, 
    lam,
    
    ivon_lr = 0.002,
    ivon_beta1 = 0.9,
    ivon_beta2 = 0.9995, 
    ivon_hess_init = 0.0 
):
    """
        Runs IVON to solve the following optimization problem

        lam * E_q[loss(theta) + \theta^\top lagr_v + \theta^\top lagr_V \theta] + KL(q||p), 

        where p is a normal N(\theta | mprior, Sprior^{-1}). The posterior is initialized at prior 
        and a diagonal covariance Gaussian. 
    """

    d = X.shape[1]
    C = y.shape[1]

    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(d, C))

        def forward(self, x):
            return x @ self.weight

    lossfn = nn.CrossEntropyLoss()
    n_steps = 1250

    model = LinearModel()
    optimizer = IVON(
        model.parameters(),
        lr = ivon_lr,
        ess = lam,
        beta1=ivon_beta1,
        beta2=ivon_beta2,
        prior_mean=mprior.clone(), 
        prior_prec=sprior.clone(), 
        dual_mean=lagr_v,
        dual_prec=lagr_u, 
        hess_init=ivon_hess_init)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0.0, T_max=n_steps)

    ivon_mc_samples = 1

    optimizer._set_posterior([mprior], [sprior])
    optimizer._set_dual(dual_mean=lagr_v, dual_prec=lagr_u)

    for i in range(n_steps + 1):
        for _ in range(ivon_mc_samples): 
            with optimizer.sampled_params(train=True):
                optimizer.zero_grad()
                output = model(X)
                l = lossfn(output, y)
                l.backward()

        optimizer.step()
        scheduler.step()

    m, s = optimizer._get_posterior()

    return m[0].clone().detach(), s[0].clone().detach()

def bayes_admm_diagonalcov(
    X,
    y,
    indices, 
    n_admm_iters, 
    m_init,
    s_init, 
    s_prior, 
    rho, 
    dual_damping,
    ivon_parameters, 
    temperature = 1.0, 
    alpha = None,
    bregman_admm = False 
): 
    """
        Runs the proposed method (Figure 4 in the paper) for diagonal covariance Gaussians.  

        If bregman_admm is set to True, this is the method of Wang and Banerjee (NIPS 2014)
        which uses the primal coordinate system in the dual update. 
    """

    n_clients = len(indices)

    # init server and client at same point
    m_server = m_init.clone() 
    s_server = s_init.clone() 
    m_clients = [m_init.clone() for k in range(n_clients)]
    s_clients = [s_init.clone() for k in range(n_clients)]

    # init Lagrange multipliers at zero
    lagr_v = [torch.zeros_like(m_init) for k in range(n_clients)]
    lagr_u = [torch.zeros_like(s_init) for k in range(n_clients)]

    it_info = { 
        'm_clients': [m.clone() for m in m_clients], 
        's_clients': [S.clone() for S in s_clients], 
        'm_server' : m_server.clone(), 
        's_server' : s_server.clone() 
    }

    iterates = [it_info] #

    for it in range(n_admm_iters + 1): 
        print(it)
        ### client step 
        for i in range(n_clients):
            m_clients[i], s_clients[i] = \
                variational_subproblem_ivon(
                        X[indices[i]], 
                        y[indices[i]].float(),
                        m_server.clone().detach(),
                        s_server.clone().detach(),
                        -temperature * lagr_v[i] / len(indices[i]),
                        -temperature * lagr_u[i] / len(indices[i]),
                        len(indices[i]) / (temperature * rho),
                        **ivon_parameters)

        ### dual update
        for i in range(n_clients): 
            if bregman_admm == False: 
                lagr_v[i] = lagr_v[i] + dual_damping * rho * (s_clients[i] * m_clients[i] - s_server * m_server)
                lagr_u[i] = lagr_u[i] + dual_damping * rho * (s_clients[i] - s_server)
            else: 
                # Bregman ADMM by Wang and Banerjee uses the expectation parameters in dual update.
                mu2_server = 1 / s_server + m_server ** 2.0 
                mu2_client = 1 / s_clients[i] + m_clients[i] ** 2.0

                lagr_v[i] = lagr_v[i] + dual_damping * rho * (m_clients[i] - m_server)
                lagr_u[i] = lagr_u[i] + dual_damping * rho * (mu2_client - mu2_server)

        ### server update 
        if alpha is None: 
            alpha = 1.0 / (1.0 + n_clients * rho)
        
        sum_v = torch.sum(torch.vstack(lagr_v), dim=0)
        sum_u = torch.sum(torch.vstack(lagr_u), dim=0)

        mean_sm = torch.mean(torch.vstack([s * m for (s, m) in zip(s_clients, m_clients)]), dim=0)
        mean_s = torch.mean(torch.vstack(s_clients), dim=0)

        s_server = (1 - alpha) * mean_s + alpha * (s_prior + sum_u) 
        m_server = ((1 - alpha) * mean_sm + alpha * sum_v) / s_server

        ### save iterates for plotting
        it_info = { 
            'm_clients': [m.clone() for m in m_clients], 
            's_clients': [s.clone() for s in s_clients], 
            'm_server' : m_server.clone(), 
            's_server' : s_server.clone() 
        }

        iterates.append(it_info)

    return iterates