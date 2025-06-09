# Local node workers
import torch
import torch.nn as nn
import numpy as np
from models import LinearModel, MLP, CifarNet, ResNet20, FedDynCifarCNN
from adamfuncreg import AdamFuncReg
import utils
import copy
import torch.nn.functional as F 
from ivonwithprior import IVON
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

try:
    import wandb
except ModuleNotFoundError as err:
    pass


# stacks a dict of parameters into a single vector
def stack(dic):
    return torch.cat([p.view(-1) for p in dic.values()])

# Local workers, where data is stored and gradients are computed
class worker():
    def __init__(self, model_parameters, optimiser_parameters, misc_parameters, trainloader, testloader,
                 client_ind=0, wandb_log=False, use_cuda=False, seed=1):

        # Store worker parameter settings
        self.model_parameters = model_parameters
        self.optimiser_parameters = optimiser_parameters
        self.misc_parameters = misc_parameters
        self.trainloader = trainloader
        self.testloader = testloader
        self.client_ind = client_ind
        self.wandb_log = wandb_log
        self.use_cuda = use_cuda
        self.seed = seed
        self.trainloader_small = None  # For CIFAR, use smaller trainloader for Hessian calculation
        self.dual_first = False  # Whether to do the dual update before the client update
        self.without_trainloaders = False  # For small models, do not use torch.dataloaders, as much quicker
        self.scheduler = None

        # Initialise model
        if self.model_parameters['network_type'] == "Linear":
            self.model = LinearModel(D_in=self.model_parameters['D_in'], D_out=self.model_parameters['D_out'])
            self.without_trainloaders = True
        elif self.model_parameters['network_type'] == "MLP":
            self.model = MLP(D_in=self.model_parameters['D_in'],
                             hidden_sizes=self.model_parameters['hidden_sizes'],
                             D_out=self.model_parameters['D_out'],
                             act_func=self.model_parameters['act_func'])
            self.without_trainloaders = True
        elif self.model_parameters['network_type'] == "CifarNet":
            in_channels = 3
            self.model = CifarNet(in_channels=in_channels, out_channels=self.model_parameters['D_out'])
            self.without_trainloaders = False
        elif self.model_parameters['network_type'] == 'FedDynCifarCNN': 
            in_channels = 3
            self.model = FedDynCifarCNN(n_cls=self.model_parameters['D_out'])
            self.without_trainloaders = False
        elif self.model_parameters['network_type'] == "ResNet20":
            self.model = ResNet20(num_classes=self.model_parameters['D_out'])#in_channels=in_channels, out_channels=self.model_parameters['D_out'])
            self.without_trainloaders = False
        elif self.model_parameters['network_type'] == "ResNet20FRN":
            self.model = ResNet20(num_classes=self.model_parameters['D_out'], normalisation='FRN')#in_channels=in_channels, out_channels=self.model_parameters['D_out'])
            self.without_trainloaders = False
        else:
            raise ValueError("Do not currently support args.network_type:", self.model_parameters['network_type'])

        if self.use_cuda:
            self.model = self.model.cuda()

        # Initialise optimiser
        if self.optimiser_parameters['optimiser_type'] == "Adam":
            self.optimiser = torch.optim.Adam(self.model.parameters(), lr=optimiser_parameters['learning_rate'],
                                              weight_decay=optimiser_parameters['l2_regularisation'])
        elif self.optimiser_parameters['optimiser_type'] == "SGD":
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr=optimiser_parameters['learning_rate'],
                                              weight_decay=optimiser_parameters['l2_regularisation'])
        elif self.optimiser_parameters['optimiser_type'] == "AdamFuncReg":
            self.optimiser = AdamFuncReg(self.model, lr=optimiser_parameters['learning_rate'],
                                         weight_decay=optimiser_parameters['l2_regularisation'],
                                         grad_clip=optimiser_parameters['grad_clip'])
        else:
            raise ValueError("Do not currently support args.opt_type:", self.optimiser_parameters['optimiser_type'])

        # Initialise variables for training
        self.received_global_weights = None
        self.previous_model_weights = None
        self.dual_weights = None

        # If not using torch.dataloader
        if self.without_trainloaders and self.trainloader is not None:
            inputs, labels = self.trainloader
            self.trainloader = [[inputs, labels]]
            self.train_set_size = len(self.trainloader[0][1])
        elif self.trainloader is not None:
            self.train_set_size = len(self.trainloader.dataset)

        # Choose loss function and dual variable update depending on federated learning method
        if self.misc_parameters['federated_method'] == 'FedAvg':
            self.loss_function = self.fedavg_loss
            self.dual_update_function = self.fedavg_dual_update
        elif self.misc_parameters['federated_method'] == 'FedProx':
            self.loss_function = self.fedprox_loss
            self.dual_update_function = self.fedprox_dual_update
        elif (self.misc_parameters['federated_method'] == 'FedADMM'
              or self.misc_parameters['federated_method'] == 'FedDyn'):
            self.loss_function = self.fedadmm_loss
            self.dual_update_function = self.fedadmm_dual_update
            self.dual_weights = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.received_global_weights = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.previous_model_weights = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
        elif self.misc_parameters['federated_method'] == 'FedLap':
            self.loss_function = self.fedlap_loss
            self.dual_update_function = self.fedlap_dual_update
            self.dual_weights = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.received_global_weights = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
        elif self.misc_parameters['federated_method'] == 'FedLapCov':
            self.loss_function = self.fedlapcov_loss
            self.dual_update_function = self.fedlapcov_dual_update
            self.dual_mean = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.dual_prec = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.received_global_mean = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.received_global_prec = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
        elif self.misc_parameters['federated_method'] == 'FedIVON':
            self.loss_function = self.fedavg_loss
            self.dual_update_function = self.fedivon_dual_update
            self.dual_mean = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.dual_prec = torch.zeros_like(utils.return_model_parameters(self.model))  # Init at zeros
            self.dual_first = False

            rho = self.misc_parameters['federated_step_size']
            wd = self.optimiser_parameters['l2_regularisation']
            tau = self.optimiser_parameters['temperature']
            prior_mean = torch.zeros_like(utils.return_model_parameters(self.model))
            prior_prec = wd * torch.ones_like(utils.return_model_parameters(self.model))

            self.received_global_mean = prior_mean.clone()
            self.received_global_prec = prior_prec.clone() 

            print(self.train_set_size)

            # set optimizer parameters so that batch variational objective 
            # is solved instead of a client one. 
            if self.optimiser_parameters['batchupperbound'] == True: 
                self.optimiser = IVON(self.model.parameters(), 
                            lr = optimiser_parameters['learning_rate'],
                            ess = self.train_set_size / tau,
                            beta1=0.9,
                            beta2=optimiser_parameters['beta2'],
                            prior_mean=prior_mean, 
                            prior_prec=prior_prec, 
                            dual_mean=self.dual_mean, # these are zero
                            dual_prec=self.dual_prec, # these are zero 
                            hess_init=optimiser_parameters['h0'],
                            local_wd=optimiser_parameters['localwd'])

            else: 
                self.optimiser = IVON(self.model.parameters(), 
                            lr = optimiser_parameters['learning_rate'],
                            ess = (self.train_set_size / tau) / rho,
                            beta1=0.9,
                            beta2=optimiser_parameters['beta2'],
                            prior_mean=prior_mean, 
                            prior_prec=prior_prec, 
                            dual_mean=self.dual_mean / (self.train_set_size / tau),
                            dual_prec=self.dual_prec / (self.train_set_size / tau),
                            hess_init=optimiser_parameters['h0'],
                            local_wd=optimiser_parameters['localwd'])

            # Scheduler
            if self.optimiser_parameters['scheduler'] == True:
                if self.misc_parameters['federated_method'] == 'FedIVON':
                    self.scheduler = CosineAnnealingLR(self.optimiser, T_max=self.misc_parameters['num_comm_rounds'], eta_min=0.0)
                else:
                    self.scheduler = StepLR(self.optimiser, step_size=1, gamma=0.992)
                                                
        else:
            raise ValueError("Do not currently support args.federated_method:",
                             self.misc_parameters['federated_method'])
        

        # Weights & Biases logging
        if self.wandb_log:
            wandb.watch(self.model)


    # Federated Averaging loss function
    def fedavg_loss(self, criterion, logits, labels):
        loss = criterion(logits, labels)
        return loss

    # Federated Averaging dual update
    def fedavg_dual_update(self): pass


    # FedProx loss function
    def fedprox_loss(self, criterion, logits, labels):
        loss = criterion(logits, labels)
        # L2 regularisation towards global weights
        if self.received_global_weights is not None:
            loss += 0.5*self.misc_parameters['federated_step_size']*torch.norm(
                nn.utils.parameters_to_vector(self.model.parameters()) - self.received_global_weights)
        return loss

    # FedProx dual update
    def fedprox_dual_update(self): pass


    # FedADMM loss function
    def fedadmm_loss(self, criterion, logits, labels):
        loss = criterion(logits, labels)
        # L2 regularisation towards global weights
        if self.received_global_weights is not None:
            loss += 0.5*self.misc_parameters['federated_step_size']*torch.norm(
                nn.utils.parameters_to_vector(self.model.parameters()) - self.received_global_weights)
        # Dot product of current weights and dual weights
        loss += torch.dot(nn.utils.parameters_to_vector(self.model.parameters()), self.dual_weights)
        return loss

    # FedADMM dual update: y_k <-- y_k + rho * (w_k - w_g)
    def fedadmm_dual_update(self):
        self.dual_weights += (self.misc_parameters['federated_step_size']
                              * (utils.return_model_parameters(self.model) - self.received_global_weights))

    # FedLap loss function: N_k*l_k(w) + delta*v_k^T w + half*delta*||w - w_g||^2
    def fedlap_loss(self, criterion, logits, labels):
        loss = criterion(logits, labels) * self.train_set_size
        # L2 regularisation towards global weights
        loss += 0.5*self.misc_parameters['federated_step_size']*torch.norm(
            nn.utils.parameters_to_vector(self.model.parameters()) - self.received_global_weights)
        # Dot product of current weights and dual weights
        loss += (self.misc_parameters['federated_step_size'] *
                 torch.dot(nn.utils.parameters_to_vector(self.model.parameters()), self.dual_weights))

        if self.optimiser_parameters['optimiser_type'] == "SGD":
            # Divide by number of training points again for SGD
            loss = loss / self.train_set_size
        return loss

    # FedLap dual update: v_k <-- v_k + rho * (w_k - w_g)
    def fedlap_dual_update(self):
        self.dual_weights += (self.misc_parameters['dual_update_damping']
                              * (utils.return_model_parameters(self.model) - self.received_global_weights))


    # FedLapCov loss function: N_k*l_k(w) + v_k^T w - half*w * V_k * w + half*||w - w_g||^2_{S_g}
    def fedlapcov_loss(self, criterion, logits, labels):
        loss = criterion(logits, labels) * self.train_set_size
        # L2 regularisation towards global weights
        if self.received_global_mean is not None:
            loss += 0.5*torch.dot((nn.utils.parameters_to_vector(self.model.parameters()) - self.received_global_mean),
                     self.received_global_prec * (nn.utils.parameters_to_vector(self.model.parameters())
                                                  - self.received_global_mean))
            # Terms to do with current weights w and dual weights {v_k, V_k}
            loss += torch.dot(nn.utils.parameters_to_vector(self.model.parameters()), self.dual_mean)
            loss -= 0.5*torch.dot(nn.utils.parameters_to_vector(self.model.parameters()),
                                  self.dual_prec * nn.utils.parameters_to_vector(self.model.parameters()))

        if self.optimiser_parameters['optimiser_type'] == "SGD":
            # Divide by number of training points again for SGD
            loss = loss / self.train_set_size
        return loss

    # FedLapCov dual update: v_k <-- v_k + rho * (S_k w_k - S_g w_g), where S_k = H_k(w_k) - V_k + S_g
    #                        V_k <-- V_k + rho * (S_k - S_g), or, V_k <-- (1-rho) V_k + rho H_k(w_k)
    def fedlapcov_dual_update(self):
        hessian = self.fedlapcov_calculate_hessian()  # This is H_k(w_k)
        S_k = hessian - self.dual_prec + self.received_global_prec
        self.dual_mean += (self.misc_parameters['dual_update_damping']
                              * (S_k * utils.return_model_parameters(self.model) -
                                 self.received_global_prec * self.received_global_mean))
        self.dual_prec = ((1-self.misc_parameters['dual_update_damping'])*self.dual_prec +
                          self.misc_parameters['dual_update_damping'] * hessian)

    # Calculate Hessian of model at current weights (for FedLapCov), using the LaplaceRedux library
    def fedlapcov_calculate_hessian(self):
        if self.model_parameters['network_type'] != 'Linear':
            # Use GPU if not linear model
            self.model = self.model.cuda()

        from laplace import Laplace
        la = Laplace(self.model, "classification",
                     subset_of_weights="all", hessian_structure="diag")
        if self.without_trainloaders:
            dataset = torch.utils.data.TensorDataset(self.trainloader[0][0], self.trainloader[0][1])
            batch_size = 64 if self.model_parameters['network_type'] == 'MLP' else 4000
            trainloader = torch.utils.data.dataloader.DataLoader(dataset=dataset, batch_size=batch_size,
                                                                 shuffle=False, num_workers=2)
        else:
            trainloader = self.trainloader
            if self.trainloader_small is not None:
                trainloader = self.trainloader_small

        la.fit(trainloader)
        hessian = la.posterior_precision.detach() - la.prior_precision

        if self.model_parameters['network_type'] != 'Linear' and not self.use_cuda:
            self.model = self.model.cpu()
            hessian = hessian.cpu()

        return hessian

    # FedIVON dual update: v_k <-- v_k + rho * (s_g m_g - s_k m_k)
    #                      V_k <-- V_k + rho * (S_g - S_k)
    def fedivon_dual_update(self):
        # dont do a dual update when computing the batch upperbound
        if self.optimiser_parameters['batchupperbound'] == True: 
            return 

        if self.received_global_mean is not None: 
            damping = self.misc_parameters['dual_update_damping']

            mean, prec = self.optimiser._get_posterior() 
            self.dual_mean += (damping * self.misc_parameters['federated_step_size']
                                * (self.received_global_prec * self.received_global_mean - prec[0] * mean[0]))
            self.dual_prec += (damping * self.misc_parameters['federated_step_size']
                                * (self.received_global_prec - prec[0]))
        
        tau = self.optimiser_parameters['temperature']
        self.optimiser._set_dual(
            dual_mean=self.dual_mean/(self.train_set_size/tau), 
            dual_prec=self.dual_prec/(self.train_set_size/tau))

    # Federated Averaging train function
    def train(self, num_epochs=1, num_steps=-1):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

        # Criterion for loss
        criterion = nn.CrossEntropyLoss()

        # Before training, test using current model parameters
        if self.misc_parameters['log_interval_epochs'] > 0:
            test_accuracy, test_nll = self.test()
            print('Epoch[0]: Train acc: %.2f, test acc: %.2f, test nll: %f' % (np.nan, test_accuracy, test_nll))

        # Dual update first for FedIVON
        if self.dual_first == True: 
            self.dual_update_function()

        # Train for num_epochs
        self.model.train()

        finished = False 
        epoch = 0
        totalsteps = 0
        while not finished: 
            # Code for calculating minibatches if no torch trainloaders (minibatching manually)
            if self.without_trainloaders:
                minibatch_size = self.optimiser_parameters['batch_size']
                if minibatch_size is None:
                    minibatch_size = len(self.trainloader[0][1])
                minibatch_perm_inds = torch.randperm(len(self.trainloader[0][1]))
                num_iters_per_epoch = int(len(self.trainloader[0][1]) / minibatch_size + 1 - 1e-10)  # Ceiling function
                trainloader = []
                for iter_ind_interm in range(num_iters_per_epoch):
                    if iter_ind_interm < num_iters_per_epoch - 1:
                        inds_for_iter = minibatch_perm_inds[
                                        iter_ind_interm * minibatch_size:(iter_ind_interm + 1) * minibatch_size]
                    else:
                        inds_for_iter = minibatch_perm_inds[iter_ind_interm * minibatch_size:]
                    trainloader.append((self.trainloader[0][0][inds_for_iter], self.trainloader[0][1][inds_for_iter]))
            else:
                trainloader = self.trainloader

            # Go over minibatches
            train_nll = 0
            train_acc = 0
            trainsteps = 0
            for (inputs, labels) in trainloader:
                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                # Forward pass through model and take optimiser step
                if self.misc_parameters['federated_method'] == 'FedIVON':
                    for _ in range(self.optimiser_parameters['mcsamples']):  
                        with self.optimiser.sampled_params(train=True):
                            self.optimiser.zero_grad()
                            logits = self.model(inputs)
                            loss = self.loss_function(criterion, logits, labels)  # Use loss function for federated learning method
                            train_nll += loss.detach().cpu().item()

                            loss.backward()
                    self.optimiser.step()
                else: 
                    self.optimiser.zero_grad()
                    logits = self.model(inputs)
                    loss = self.loss_function(criterion, logits, labels)  # Use loss function for federated learning method
                    loss.backward()
                    if self.optimiser_parameters['grad_clip'] is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.optimiser_parameters['grad_clip'])
                    self.optimiser.step()
                    train_nll += loss.detach().cpu().item()

                # Calculate training metrics
                pred_y = logits.data.max(1, keepdim=True)[1]
                train_acc += pred_y.eq(labels.data.view_as(pred_y)).sum()
                trainsteps += 1 
                totalsteps += 1

            epoch += 1 

            # Check if done
            if (num_steps == -1) and (epoch >= num_epochs):
                finished = True 
            
            if (num_steps > -1) and (totalsteps > num_steps): 
                finished = True 

            # Training metrics are averaged over number of training examples
            train_nll = train_nll / len(trainloader) # since loss uses mean reduction we dont want to divide by the whole size of the dataset here 

            if self.misc_parameters['federated_method'] == 'FedIVON':
                train_nll /= self.optimiser_parameters['mcsamples']

            train_acc = 100.0*float(train_acc) / self.train_set_size

            # Print and store if log_interval_epoch
            if (self.misc_parameters['log_interval_epochs'] > 0 and
                (epoch % self.misc_parameters['log_interval_epochs'] == 0) or epoch==num_epochs-1):
                test_accuracy, test_nll = self.test()
                print('Epoch[%d]: Train acc: %.2f, test acc: %.2f, test nll: %f' %
                      (epoch + 1, train_acc, test_accuracy, test_nll))
                if self.wandb_log:
                    task_id_string = 'Client %d epoch'%(self.client_ind)
                    wandb.log({
                        task_id_string: epoch,
                        'test_nll': test_nll,
                        'test_accuracy': test_accuracy
                    })

                # Report the loss of the variational objective as well
                # if we are doing batch training. 
                if self.misc_parameters['federated_method'] == 'FedIVON' and self.optimiser_parameters['batchupperbound'] == True:
                    expected_loss = train_nll
                    mean, prec = self.optimiser._get_posterior() 
                    kl_term = utils.kldivergence(
                        mean[0], 
                        1.0/prec[0], 
                        self.optimiser.prior_mean, 
                        1.0/self.optimiser.prior_prec)   
                    
                    tau = self.optimiser_parameters['temperature']
                    vo = (self.train_set_size / tau) * expected_loss + kl_term
                    print("  Variational Objective %.2f" % vo)


        # Dual update step (for relevant methods)
        if self.dual_first == False: 
            self.dual_update_function()

        # Step size scheduler
        if self.scheduler is not None:
            self.scheduler.step() 

    # Test the model on 2D grid of points, can specify plot_dataloader if desired (default to self.testloader)
    def test_toydata(self, plot_dataloader=None):
        with torch.no_grad():
            # Test the model on 2D grid of points
            test_outputs_full = []
            self.model.eval()

            if self.without_trainloaders:
                if plot_dataloader is None:
                    inputs, labels = self.testloader
                    testloader = [[inputs, labels]]
                else:
                    inputs, labels = plot_dataloader
                    testloader = [[inputs, labels]]
            else:
                if plot_dataloader is None:
                    testloader = self.testloader
                else:
                    testloader = plot_dataloader

            for inputs, _ in testloader:
                if self.use_cuda:
                    inputs = inputs.cuda()
                if len(inputs.shape) == 1:
                    inputs = inputs.unsqueeze(0)

                logits = self.model(inputs)
                test_outputs_full.append(logits)

            # Concatenate all the outputs and pass through sigmoid
            test_outputs_full = torch.cat(test_outputs_full, dim=0)
            test_outputs_full = torch.softmax(test_outputs_full, dim=-1)

        return test_outputs_full


    # Test this worker's model on dataloader (default to self.testloader)
    def test(self, dataloader=None):
        if dataloader == None:
            dataloader = self.testloader

        if self.without_trainloaders:
            inputs, labels = dataloader
            num_examples = len(labels)
            dataloader = [[inputs, labels]]
        else:
            num_examples = len(dataloader.dataset)

        criterion = torch.nn.CrossEntropyLoss()

        if self.use_cuda:
            self.model = self.model.cuda()

        correct = 0
        negative_log_lik = 0
        with torch.no_grad():
            self.model.eval()
            for inputs, labels in dataloader:

                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Predictions from model
                logits = self.model.forward(inputs)

                # Calculate NLL
                negative_log_lik += criterion(logits, labels.long()) * float(len(labels))

                # Calculate predicted classes and count correctly-predicted datapoints
                pred_y = logits.data.max(1, keepdim=True)[1]
                correct += pred_y.eq(labels.data.view_as(pred_y)).sum()

            # Average negative loglik and test accuracy
            negative_log_lik /= num_examples
            test_accuracy = 100.0*float(correct) / num_examples
 
        self.model.train() # set back to training mode
        return test_accuracy, negative_log_lik.cpu().numpy()

    # Test this worker's model on dataloader (default to self.testloader)
    # This uses the posterior predictive distribution approximated with MC samples
    # Warning: only works for classification (softmax) problems
    def test_bayes(self, dataloader=None, testmcsamples=32):
        # this is only supported for FedIVON right now
        assert self.misc_parameters['federated_method'] == 'FedIVON'

        if dataloader == None:
            dataloader = self.testloader

        if self.without_trainloaders:
            inputs, labels = dataloader
            num_examples = len(labels)
            dataloader = [[inputs, labels]]
        else:
            num_examples = len(dataloader.dataset)

        criterion = torch.nn.CrossEntropyLoss()

        if self.use_cuda:
            self.model = self.model.cuda()

        correct = 0
        negative_log_lik = 0
        with torch.no_grad():
            self.model.eval()
            for inputs, labels in dataloader:

                if self.use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Predictions from model
                sampled_probs = []
                for i in range(testmcsamples):
                    with self.optimiser.sampled_params(train=False):
                        sampled_logit = self.model.forward(inputs)
                        sampled_probs.append(F.softmax(sampled_logit, dim=1))
                prob = torch.mean(torch.stack(sampled_probs), dim=0)                

                # Calculate NLL
                negative_log_lik += nn.NLLLoss()(torch.log(prob), labels.long()) \
                    * float(len(labels))

                # Calculate predicted classes and count correctly-predicted datapoints
                pred_y = prob.data.max(1, keepdim=True)[1]
                correct += pred_y.eq(labels.data.view_as(pred_y)).sum()

            # Average negative loglik and test accuracy
            negative_log_lik /= num_examples
            test_accuracy = 100.0*float(correct) / num_examples

        return test_accuracy, negative_log_lik.cpu().numpy()  
