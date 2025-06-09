import torch
import numpy as np
import math 

def solve_admm_subproblem(
    loss_grad_fn, 
    mprior, 
    sprior, 
    lagr_v, 
    lr = 5e-4, 
    momentum = 0.8, 
    num_iters = 25000, 
    eps = 1e-5 # convergence criterion 
):
    # initialize at prior 
    m = mprior.clone() 
    m_prev = mprior.clone()
    
    for i in range(num_iters):
        fullgradient = loss_grad_fn(m) + sprior * (m - mprior) + lagr_v
        m_prev2 = m.clone() # for momentum 
        m = m - lr * fullgradient + momentum * (m - m_prev)
        m_prev = m_prev2.clone()

        if i % 100 == 0:
            g = torch.linalg.norm(fullgradient) 
            if g < eps: 
                break

    if g > eps: 
        print(f'Gradient descent on ADMM subproblem did not converge, gradient norm {g:.2e}.')

    return m 

def von_fullcov(
    loss_grad_fn,  # gradient of loss
    loss_hess_fn,  # Hessian of loss 
    mprior,        # prior mean
    Sprior,        # prior precision
    lagr_v,        # lagrange multiplier 1
    lagr_V,        # lagrange multiplier 2 
    lam,           # weighting 

    num_iters = 25000,    # number of iterations
    num_mc = 1,           # number of MC samples
    lr_m = 1e-4,          # learning rate for m 
    lr_S = 1e-4,          # learning rate for S 
    jitter = 1e-6,        # cholesky stability
    laplace = False       # use delta method
):
    """
        Runs variational online Newton (VON) to solve the following optimization problem

        lam * E_q[loss(theta) + \theta^\top lagr_v + \theta^\top lagr_V \theta] + KL(q||p), 

        where p is a normal N(\theta | mprior, Sprior^{-1}). The posterior is initialized at prior 
        and a full covariance Gaussian. 
    """

    # initialize posterior at prior
    m = mprior.clone()
    S = Sprior.clone()

    for it in range(num_iters + 1):
        rho1 = lr_m / (1 + it * 0.1) 
        rho2 = lr_S / (1 + it * 0.1) 

        # cholesky needed for random sampling 
        chol_S = torch.linalg.cholesky(S + jitter * torch.eye(S.shape[0]))
        cholinv_S = torch.linalg.inv(chol_S.T)

        if laplace == False: 
            # get expected gradient and Hessian by MC sampling
            samples = [] 
            for j in range(num_mc): 
                z = torch.randn_like(m)
                #theta = m + torch.cholesky_solve(z.unsqueeze(1), chol_S).squeeze(1)
                theta = m + torch.matmul(z, cholinv_S)
                samples.append(theta)

            Egrad = torch.stack([loss_grad_fn(theta) for theta in samples]).mean(dim=0)
            Ehess = torch.stack([loss_hess_fn(theta) for theta in samples]).mean(dim=0)
        else: 
            Egrad = loss_grad_fn(m) 
            Ehess = loss_hess_fn(m) 

        # Lagrange multiplier gradients
        lgrad_S = -lagr_V 
        lgrad_m = lagr_v - lagr_V @ m

        # Updates 
        S = (1 - rho2 / lam) * S + rho2 * (Ehess + lgrad_S + Sprior / lam) 
        #S = 0.5 * (S + S.T) # symmetrize 
        m -= rho1 * torch.linalg.solve(S, Egrad + lgrad_m + Sprior @ (m - mprior) / lam)

    return m, S

def bayes_admm_fullcov(
    client_grad_fns, 
    client_hess_fns,
    n_admm_iters, 
    m_init,
    S_init, 
    S_prior, 
    rho, 
    dual_damping,
    von_parameters, 
    temperature = 1.0, 
    alpha = None,
    bregman_admm = False 
): 
    """
        Runs the proposed method (Figure 4 in the paper) for full covariance Gaussians.  

        If bregman_admm is set to True, this is the method of Wang and Banerjee (NIPS 2014)
        which uses the primal coordinate system in the dual update. 
    """

    n_clients = len(client_grad_fns)

    # init server and client at same point
    m_server = m_init.clone() 
    S_server = S_init.clone() 
    m_clients = [m_init.clone() for k in range(n_clients)]
    S_clients = [S_init.clone() for k in range(n_clients)]

    # init Lagrange multipliers at zero
    lagr_v = [torch.zeros_like(m_init) for k in range(n_clients)]
    lagr_V = [torch.zeros_like(S_init) for k in range(n_clients)]

    it_info = { 
        'm_clients': [m.clone() for m in m_clients], 
        'S_clients': [S.clone() for S in S_clients], 
        'm_server' : m_server.clone(), 
        'S_server' : S_server.clone() 
    }

    iterates = [it_info] # save ADMM trajectory

    for it in range(n_admm_iters + 1): 
        print(it)
        ### client step 
        for i in range(n_clients):
            m_clients[i], S_clients[i] = \
                von_fullcov(client_grad_fns[i], 
                            client_hess_fns[i], 
                            m_server, 
                            S_server, 
                            temperature * lagr_v[i], 
                            temperature * lagr_V[i], 
                            1 / (temperature * rho), 
                            **von_parameters)

        ### dual update
        for i in range(n_clients): 
            if bregman_admm == False: 
                lagr_v[i] = lagr_v[i] + dual_damping * rho * (S_clients[i] @ m_clients[i] - S_server @ m_server)
                lagr_V[i] = lagr_V[i] + dual_damping * rho * (S_clients[i] - S_server)
            else: 
                # Bregman ADMM by Wang and Banerjee uses the expectation parameters in dual update.
                mu2_server = torch.linalg.inv(S_server) + torch.outer(m_server, m_server)
                mu2_client = torch.linalg.inv(S_clients[i]) + torch.outer(m_clients[i], m_clients[i])

                lagr_v[i] = lagr_v[i] + dual_damping * rho * (m_clients[i] - m_server)
                lagr_V[i] = lagr_V[i] + dual_damping * rho * (mu2_client - mu2_server)

        ### server update 
        if alpha is None: 
            alpha = 1.0 / (1.0 + n_clients * rho)
        
        d = m_server.shape[0]
        sum_v = torch.sum(torch.vstack(lagr_v), dim=0)
        sum_V = torch.sum(torch.concatenate([x.view(d, d, 1) for x in lagr_V], dim=2), dim=2)

        mean_Sm = torch.mean(torch.vstack([S @ m for (S, m) in zip(S_clients, m_clients)]), dim=0)
        mean_S = torch.mean(torch.concatenate([x.view(d, d, 1) for x in S_clients], dim=2), dim=2)

        S_server = (1 - alpha) * mean_S + alpha * (S_prior + sum_V) 
        m_server = torch.linalg.solve(S_server, (1 - alpha) * mean_Sm + alpha * (sum_v))

        ### save iterates for plotting
        it_info = { 
            'm_clients': [m.clone() for m in m_clients], 
            'S_clients': [S.clone() for S in S_clients], 
            'm_server' : m_server.clone(), 
            'S_server' : S_server.clone() 
        }

        iterates.append(it_info)

    return iterates

def federated_admm(
    client_grad_fns, 
    n_admm_iters, 
    m_init,
    delta, # prior precision 
    rho, 
    dual_damping,
    rho_server,
    gd_parameters 
): 
    """
        Runs standard federated ADMM (as described in appendix B.1). For rho_server = 0, this is Tseng's AMA method.
    """

    n_clients = len(client_grad_fns)

    # init server and client at same point
    m_server = m_init.clone() 
    m_clients = [m_init.clone() for k in range(n_clients)]
    lagr_v = [torch.zeros_like(m_init) for k in range(n_clients)]

    it_info = { 
        'm_clients': [m.clone() for m in m_clients], 
        'm_server' : m_server.clone(), 
    }
    iterates = [it_info] # save ADMM trajectory

    for it in range(n_admm_iters + 1): 

        ### client step 
        for i in range(n_clients):
            # solve ADMM subproblem 
            m_clients[i] = solve_admm_subproblem(
                client_grad_fns[i], 
                m_server, 
                rho, 
                lagr_v[i], 
                **gd_parameters) 

        ### dual update
        for i in range(n_clients): 
            lagr_v[i] = lagr_v[i] + dual_damping * rho * (m_clients[i] - m_server)

        ### server update 
        alpha = 1 / (delta + n_clients * rho_server)
        
        sum_v = torch.sum(torch.vstack(lagr_v), dim=0)
        mean_m = torch.mean(torch.vstack(m_clients), dim=0)

        m_server = (1 - delta * alpha) * mean_m + alpha * sum_v

        ### save iterates for plotting
        it_info = { 
            'm_clients': [m.clone() for m in m_clients], 
            'm_server' : m_server.clone(), 
        }

        iterates.append(it_info)

    return iterates
