import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd.functional import hessian 
from torch.func import grad
from torchvision import datasets
import admm
import os

torch.set_default_dtype(torch.double)
torch.set_default_device('cpu')
torch.manual_seed(1)

train_set = datasets.MNIST('./data', train=True, download=True)
test_set = datasets.MNIST('./data', train=False, download=True)

train_data = train_set.data.type(torch.double).reshape(-1,1,28,28) / 255
test_data = test_set.data.type(torch.double).reshape(-1,1,28,28) / 255
train_label = F.one_hot(train_set.targets, num_classes=10).type(torch.double)
test_label = F.one_hot(test_set.targets, num_classes=10)

# downsample MNIST to 7x7 
train_data = F.interpolate(train_data, size=7).reshape(-1,7*7).type(torch.double)
test_data = F.interpolate(test_data, size=7).reshape(-1,7*7)

def squareloss(theta, X, y):
    # theta: (dC)
    # X: (N times d)
    f = X @ theta.reshape(X.shape[1], y.shape[1]) # N times C
    loss = 0.5 * torch.sum((f - y) ** 2.0, axis=1)
    return loss.sum(axis=0)

# dataset 
N = 500
C = train_label.shape[1]
d = train_data.shape[1]
delta = 25.0

def create_loss_functions(indices): 
    X = train_data[indices].cpu()
    y = train_label[indices].cpu()
    loss_fn = lambda t: squareloss(t, X, y)
    grad_fn = grad(loss_fn, argnums=0, has_aux=False)
    hess_fn = lambda t: hessian(loss_fn, t)

    return loss_fn, grad_fn, hess_fn

# split into a heterogeneous dataset (separated by classes)
groups = [[0,1], [2,3], [4,5], [6,7], [8,9]]
n_clients = 5
loss_fns = [] 
loss_grad_fns = [] 
loss_hess_fns = [] 

numeric_labels = train_label[0:N].argmax(axis=1)
for i in range(n_clients): 
    idx = [] 
    for digit in groups[i]: 
        idx.append(np.argwhere(numeric_labels == digit)) 

    indices = torch.concatenate(idx, axis=1).squeeze()
    loss_fn, grad_fn, hess_fn = create_loss_functions(indices)
    loss_fns.append(loss_fn) 
    loss_grad_fns.append(grad_fn)
    loss_hess_fns.append(hess_fn)

# batch loss 
gl, _, __ = create_loss_functions(range(N))
global_loss = lambda m : gl(m) + 0.5 * delta * torch.sum(m ** 2.0)

def compute_losses(iterates): 
    losses = [] 
    for i, it_info in enumerate(iterates): 
        loss = global_loss(it_info['m_clients'][0])
        losses.append(loss) 
        print(f"Iter {i}: Loss = {loss:.4f}")
    return losses 

von_parameters = {
    'num_iters' : 50,
    'num_mc' : 1,       
    'lr_m' : 1.0,      
    'lr_S' : 1.0,      
    'jitter' : 1e-7,
    'laplace' : True # since the loss is quadratic, we can do Laplace 
}

### --- run bayes admm --- 
bayes_admm_iterates = admm.bayes_admm_fullcov(
    loss_grad_fns, 
    loss_hess_fns, 
    20, 
    torch.zeros(d*C), 
    torch.eye(d*C), 
    delta * torch.eye(d*C), 
    1.0 / n_clients, 
    1.0, 
    von_parameters, 
    temperature = 1.0, 
    alpha = None 
)

bayes_admm_losses = compute_losses(bayes_admm_iterates) 

### --- run bregman admm --- 
bregman_admm_iterates = admm.bayes_admm_fullcov(
    loss_grad_fns, 
    loss_hess_fns, 
    20, 
    torch.zeros(d*C), 
    torch.eye(d*C), 
    delta * torch.eye(d*C), 
    1.0, 
    0.1, 
    von_parameters, 
    temperature = 1.0, 
    alpha = None,
    bregman_admm = True 
)

bregman_admm_losses = compute_losses(bregman_admm_iterates) 


### --- run vanilla federated admm --- 
gd_parameters = {
    'lr' : 5e-4, 
    'momentum' : 0.8, 
    'num_iters' : 25000, 
    'eps' : 1e-4
}

admm_iterates = admm.federated_admm(
    loss_grad_fns, 
    20, 
    torch.zeros(d*C),
    delta, 
    200.0, 
    1.0, 
    200.0, 
    gd_parameters) 

admm_losses = compute_losses(admm_iterates) 

# save results for plotting 
results = {
    'admm_iterates': admm_iterates,
    'bayes_admm_iterates': bayes_admm_iterates,
    'bregman_admm_iterates': bregman_admm_iterates,
    'admm_losses': admm_losses,
    'bayes_admm_losses': bayes_admm_losses,
    'bregman_admm_losses': bregman_admm_losses,
}

os.makedirs("results", exist_ok=True)
with open('results/onestep.pkl', 'wb') as f:
    pickle.dump(results, f)
