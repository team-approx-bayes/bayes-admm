import numpy as np
import math
import matplotlib.pyplot as plt 
import pickle
import torch 
import os
import admm
from torch.autograd.functional import hessian 
from torch.func import grad

torch.set_default_dtype(torch.float64)

def dataset1(N): 
    np.random.seed(1)
    mid_index = math.floor(N/2)
    mu1 = np.stack([-5*np.ones(mid_index), np.ones(N-mid_index)*0], axis=1)
    mu2 = np.stack([ 5*np.ones(mid_index), np.ones(N-mid_index)*0], axis=1)
    std1=0.6
    std2=0.6
    X = np.concatenate([
        np.random.normal(loc=mu1[0:mid_index-5], scale=std1),
        np.random.normal(loc=mu1[mid_index-5:mid_index], scale=10*std1),
        np.array([-2.5,-2.5]).reshape(1,2),
        np.array([-2.4,-2.6]).reshape(1,2),
        np.array([-2.5,-2.4]).reshape(1,2),
        np.random.normal(loc=mu2, scale=std2)]) 
    
    y = np.ones(N + 3)
    y[mid_index+3:] = 0

    return X, y

def dataset2(N): 
    np.random.seed(1)
    mid_index = math.floor(N/2)
    mu1 = np.stack([-3.2*np.ones(mid_index), np.ones(N-mid_index)*0.2], axis=1)
    mu2 = np.stack([-0.4*np.ones(mid_index), np.ones(N-mid_index)*0.9], axis=1)
    std1=0.25
    std2=0.25
    X = np.concatenate([
        np.random.normal(loc=mu1, scale=std1),
        np.random.normal(loc=mu2, scale=std2)])
    y = np.ones(N)
    y[mid_index:N] = 0

    return X, y

X1, y1 = dataset1(100)
X2, y2 = dataset2(200)

X1 = torch.concatenate([torch.tensor(X1, dtype=torch.float64), torch.ones(X1.shape[0], 1)], dim=1)
X2 = torch.concatenate([torch.tensor(X2, dtype=torch.float64), torch.ones(X2.shape[0], 1)], dim=1)

dataset = { 'X1' : X1, 'y1' : y1, 'X2' : X2, 'y2' : y2 } 

os.makedirs("results", exist_ok=True)
with open('results/toydataset.pkl', 'wb') as file:
    pickle.dump(dataset, file)

def logisticloss(theta, X, y):
    # theta (d)
    # X (N times d)
    f = X @ theta # N 
    loss = -f * torch.tensor(y, dtype=torch.float64) + torch.nn.functional.softplus(f) #torch.log(1.0 + torch.exp(f))
    return loss.sum(axis=0)

def get_grad_hess(idx):
    if idx == 0: 
        X = X1 
        y = y1 
    else: 
        X = X2 
        y = y2  

    loss_fn = lambda t: logisticloss(t, X, y)
    grad_fn = grad(loss_fn, argnums=0, has_aux=False)
    hess_fn = lambda t: hessian(loss_fn, t)

    return grad_fn, hess_fn  

## Dataset preparation
loss_grad_fns = [] 
loss_hess_fns = [] 

K = 2 # number of clients 
delta = 2e-1 # prior precision
d = 3 # three parameters (one is bias)
C = 1 # binary classification 

for idx in range(2): 
    g, H = get_grad_hess(idx)
    loss_grad_fns.append(g)
    loss_hess_fns.append(H)

# run proposed ADMM, store the server Gaussian posterior over iterations.

von_parameters = {
    'num_iters' : 10000,
    'num_mc' : 1,       
    'lr_m' : 0.01,      
    'lr_S' : 0.005,    
    'jitter' : 1e-7,
    'laplace' : False 
}

### --- run bayes admm --- 
bayes_admm_iterates = admm.bayes_admm_fullcov(
    loss_grad_fns, 
    loss_hess_fns, 
    2,  # numer iters
    torch.zeros(d*C), 
    torch.eye(d*C), 
    delta * torch.eye(d*C), 
    0.2,  # rho
    1.0,  # dual damping
    von_parameters, 
    temperature = 1.0, 
    alpha = None 
)

# run standard ADMM, store the server mean over iterations. 

gd_parameters = {
    'lr' : 1e-3, 
    'momentum' : 0.8, 
    'num_iters' : 25000, 
    'eps' : 1e-3
}

admm_iterates = admm.federated_admm(
    loss_grad_fns, 
    5, 
    torch.zeros(d*C),
    delta, 
    0.2, 
    1.0, 
    0.2, 
    gd_parameters) 

with open('results/illu_bayes_admm_iterates.pkl', 'wb') as file:
    pickle.dump(bayes_admm_iterates, file)

with open('results/illu_admm_iterates.pkl', 'wb') as file:
    pickle.dump(admm_iterates, file)
