import pickle
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd.functional import hessian 
from torch.func import grad
from torchvision import datasets
import admm_ivon
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

train_data = F.interpolate(train_data, size=14).reshape(-1,14*14)
test_data = F.interpolate(test_data, size=14).reshape(-1,14*14)

def logisticloss(theta, X, y):
    f = X @ theta.reshape(X.shape[1], y.shape[1]) # N times C
    loss = -torch.sum(f * y, axis=1) + torch.logsumexp(f, axis=1)
    return loss.sum(axis=0)

# dataset 
N = 5000
C = train_label.shape[1]
d = train_data.shape[1]
delta = 10.0

def create_loss_functions(indices): 
    X = train_data[indices].cpu()
    y = train_label[indices].cpu()
    loss_fn = lambda t: logisticloss(t, X, y)

    return loss_fn

# split into a heterogeneous dataset (separated by classes)
groups = [[0,1], [2,3], [4,5], [6,7], [8,9]]
n_clients = 5
loss_fns = [] 
data_indices = [] 

numeric_labels = train_label[0:N].argmax(axis=1)
for i in range(n_clients): 
    idx = [] 
    for digit in groups[i]: 
        idx.append(np.argwhere(numeric_labels == digit)) 

    indices = torch.concatenate(idx, axis=1).squeeze()
    loss_fn = create_loss_functions(indices)
    loss_fns.append(loss_fn) 
    data_indices.append(indices)

X = train_data[0:N]
y = train_label[0:N]

# batch loss 
gl = create_loss_functions(range(N))
global_loss = lambda m : gl(m) 

def compute_losses(iterates, jitter = 1e-7): 
    losses = [] 
    num_elbo_mc = 100

    for i, it_info in enumerate(iterates): 
        m = it_info['m_server']
        s = it_info['s_server']

        lossvals = [] 
        for j in range(num_elbo_mc):
            z = torch.randn_like(m)
            theta = m + z / torch.sqrt(s) 

            lossvals.append(global_loss(theta).detach().cpu().numpy())
        
        loss = np.mean(lossvals)
        loss += 0.5 * torch.sum(delta * (m ** 2.0))
        loss += 0.5 * torch.sum(torch.log(s))
        loss += 0.5 * torch.sum(delta / s) #trace_inv 

        losses.append(loss) 
        print(f"Iter {i}: neg-ELBO = {loss:.4f}")

    return losses 

ivon_parameters = {
    'ivon_lr' : 0.002,
    'ivon_beta1' : 0.9,       
    'ivon_beta2' : 0.9995,      
    'ivon_hess_init' : 0.0,      
}

### --- run bayes admm --- 
bayes_admm_iterates = admm_ivon.bayes_admm_diagonalcov(
    X,
    y,
    data_indices,
    25,
    torch.zeros(d*C), 
    delta * torch.ones(d*C), 
    delta * torch.ones(d*C), 
    1.0, 
    1.0, 
    ivon_parameters, 
    temperature = 1.0, 
    alpha = None,
    bregman_admm = False
)

bayes_admm_losses = compute_losses(bayes_admm_iterates) 
print(bayes_admm_losses)

### --- run bregman admm --- 
bregman_admm_iterates = admm_ivon.bayes_admm_diagonalcov(
    X,
    y,
    data_indices,
    25,
    torch.zeros(d*C), 
    delta * torch.ones(d*C), 
    delta * torch.ones(d*C), 
    1.0, 
    0.2, # bregman admm diverged if dual step size is too large
    ivon_parameters, 
    temperature = 1.0, 
    alpha = None,
    bregman_admm = True
)

bregman_admm_losses = compute_losses(bregman_admm_iterates) 
print(bregman_admm_losses)

### --- run PVI --- 
pvi_iterates = admm_ivon.bayes_admm_diagonalcov(
    X,
    y,
    data_indices,
    25,
    torch.zeros(d*C), 
    delta * torch.ones(d*C), 
    delta * torch.ones(d*C), 
    1.0, 
    1.0, 
    ivon_parameters, 
    temperature = 1.0, 
    alpha = 1.0,
    bregman_admm = False
)

pvi_losses = compute_losses(pvi_iterates) 
print(pvi_losses)

### --- run damped PVI --- 
pvi_damped_iterates = admm_ivon.bayes_admm_diagonalcov(
    X,
    y,
    data_indices,
    25,
    torch.zeros(d*C), 
    delta * torch.ones(d*C), 
    delta * torch.ones(d*C), 
    1.0, 
    0.2, 
    ivon_parameters, 
    temperature = 1.0, 
    alpha = 1.0,
    bregman_admm = False
)

pvi_damped_losses = compute_losses(pvi_damped_iterates) 
print(pvi_damped_losses)


# save results for plotting 
results = {
    'bayes_admm_iterates': bayes_admm_iterates,
    'bregman_admm_iterates': bregman_admm_iterates,
    'pvi_iterates': pvi_iterates,
    'pvi_damped_iterates': pvi_damped_iterates,
    'bayes_admm_losses': bayes_admm_losses,
    'bregman_admm_losses': bregman_admm_losses,
    'pvi_losses': pvi_losses,
    'pvi_damped_losses': pvi_damped_losses,
}

os.makedirs("results", exist_ok=True)
with open('results/neg-elbo.pkl', 'wb') as f:
    pickle.dump(results, f)


