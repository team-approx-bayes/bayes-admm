import math
import torch
from torch.optim.optimizer import Optimizer
from torch.nn.utils import parameters_to_vector, vector_to_parameters


# Check if device of param is the same as old_param_device, and warn if not
def _check_param_device(param, old_param_device):
    if old_param_device is None:
        old_param_device = param.get_device() if param.is_cuda else -1
    else:
        warn = False
        if param.is_cuda:  # check if in same gpu
            warn = (param.get_device() != old_param_device)
        else:  # check if in cpu
            warn = (old_param_device != -1)
        if warn:
            raise TypeError('Found two parameters on different devices, this is currently not supported.')
    return old_param_device


# Convert from parameters to a matrix
def parameters_to_matrix(parameters):
    param_device = None
    mat = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        m = param.shape[0]
        mat.append(param.view(m, -1))
    return torch.cat(mat, dim=-1)


# Get parameter gradients as a vector
def parameters_grads_to_vector(parameters):
    param_device = None
    vec = []
    for param in parameters:
        param_device = _check_param_device(param, param_device)
        if param.grad is None:
            raise ValueError('Gradient not available')
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec, dim=-1)


# The AdamFuncReg optimiser. Started from torch.optim.adam
class AdamFuncReg(Optimizer):

    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
                 amsgrad=False, grad_clip=None):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad,
                        grad_clip=grad_clip)
        super(AdamFuncReg, self).__init__(model.parameters(), defaults)

        self.model = model

        parameters = self.param_groups[0]['params']

        # State initialisation
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        self.state['hessian_ggn'] = torch.zeros_like(self.state['mu'])
        self.state['step'] = 0
        self.state['exp_avg'] = torch.zeros_like(self.state['mu'])
        self.state['exp_avg_sq'] = torch.zeros_like(self.state['mu'])
        if amsgrad:
            self.state['max_exp_avg_sq'] = torch.zeros_like(self.state['mu'])

    # Iteration step for this optimiser
    def step(self, closure_data=None):

        parameters = self.param_groups[0]['params']
        p = parameters_to_vector(parameters)
        self.state['mu'] = p.clone().detach()
        mu = self.state['mu']

        self.model.train()
        vector_to_parameters(mu, parameters)
        grad = parameters_grads_to_vector(parameters).detach()

        # L2 regularisation term
        if self.param_groups[0]['weight_decay'] != 0:
            grad.add_(mu, alpha=self.param_groups[0]['weight_decay'])

        # Do gradient norm clipping
        if self.defaults['grad_clip'] is not None:
            grad_norm = torch.norm(grad)
            grad_norm = 1.0 if grad_norm < self.defaults['grad_clip'] else grad_norm / self.defaults['grad_clip']
            grad.div_(grad_norm)

        # Adam update equations
        lr = self.param_groups[0]['lr']
        exp_avg, exp_avg_sq = self.state['exp_avg'], self.state['exp_avg_sq']
        beta1, beta2 = self.param_groups[0]['betas']
        amsgrad = self.param_groups[0]['amsgrad']
        if amsgrad:
            max_exp_avg_sq = self.state['max_exp_avg_sq']

        self.state['step'] += 1
        bias_correction1 = 1 - beta1 ** self.state['step']
        bias_correction2 = 1 - beta2 ** self.state['step']

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.param_groups[0]['eps'])
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.param_groups[0]['eps'])

        step_size = lr / bias_correction1

        mu.addcdiv_(exp_avg, denom, value=-step_size)

        vector_to_parameters(mu, parameters)
