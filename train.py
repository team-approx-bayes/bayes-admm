# Main file for running the experiments in Tables 1 and 2

import argparse
import copy
import torch
import numpy as np
import torch.nn as nn
import yaml
try:
    import wandb
except ModuleNotFoundError as err:
    pass

from data_generators import (ToyDataGenerator, FederatedMNISTDataGenerator, FederatedHeterogenousMNISTDataGenerator,
                             FederatedCIFAR10DataGenerator)
from local_client_worker import worker
import utils
from utils import set_model_parameters, return_model_parameters, kldivergence


parser = argparse.ArgumentParser()

# Store and load models during training
parser.add_argument('--config_file', type=str, default=None, help='Load parameters from YAML file')
parser.add_argument('--path', type=str, default='store_toyplots/', help='Path to where to store/load models and plots')
parser.add_argument('--store_data', action='store_true', default='False', help='store model and data generator in path')
parser.add_argument('--load_pretrained_workers', nargs='*', type=int, default=[],
                    help='list of indices from which to load pretrained worker states from "load_pretrained_path": '
                         'use "-1" for global worker, use any other number for data generator only')
parser.add_argument('--load_pretrained_path', type=str, default='', help='file from which to load pretrained workers')
parser.add_argument('--plot_visualisation_comm_round', type=int, default=1, help='how often to plot 2D visualisations')
parser.add_argument('--wandb_log', action='store_true', help='whether to log data to Weights & Biases')

# Initial seed and number of runs
parser.add_argument('--seed_init', type=int, default=5, help='what random seed to use')
parser.add_argument('--num_runs', type=int, default=1, help='how many runs to do (each with a  different random seed)')
parser.add_argument('--use_cuda', action='store_true', help='whether to use GPUs / CUDA')

# Setting up federated learning benchmark
parser.add_argument('--benchmark', type=str, default='toy_gaussians', help='what benchmark / dataset to run on')
parser.add_argument('--num_clients', type=int, default=5, help='number of local clients, each with their own data')
parser.add_argument('--num_comm_rounds', type=int, default=100,
                    help='number of communication rounds in federated learning')

# Federated learning method and related hyperparameters
parser.add_argument('--federated_method', type=str, default='FedProx',
                    help='which federated learning method to run: {FedAvg, FedProx, FedADMM, FedDyn, FedLap, FedIVON...}')
parser.add_argument('--federated_step_size', type=float, default=0.1,
                    help='step size hyperparameter (for FedProx, FedADMM, FedDyn, FedLap, FedIVON)')
parser.add_argument('--global_prior_prec', type=float, default=0,
                    help='global prior precision (should be 0 for most methods)')
parser.add_argument('--num_epochs_batch', type=int, default=50,
                    help='number of training epochs for batch upper bound')
parser.add_argument('--local_prior_prec', type=float, default=0,
                    help='local prior precision (FedDyn)')
parser.add_argument('--global_step_size', type=float, default=1,
                    help='step size for global server update (FedADMM)')
parser.add_argument('--dual_update_damping', type=float, default=1,
                    help='damping factor rho for dual update (FedLap)')
parser.add_argument('--dual_update_damping_multiply_Nk_N', type=int, default=0,
                    help='if 1, have client-specific damping factor for dual update (FedLap), multiplied by N_k/N')

# Model and optimisation parameters
parser.add_argument('--network_type', type=str, default='Linear',
                    help='What model to use: {Linear, MLP, ...}')
parser.add_argument('--hidden_sizes', nargs='*', type=int, default=[], help='Hidden layer sizes for MLP')
parser.add_argument('--opt_type', type=str, default='Adam', help='what optimiser to use')
parser.add_argument('--num_epochs_local', type=int, default=20, help='number of training epochs for local workers')
parser.add_argument('--learning_rate_local', type=float, default=0.01, help='learning rate')
parser.add_argument('--grad_clip', type=float, default=0, help='gradient clipping during optimisation (if greater than '
                                                               '0); currently only works with AdamFuncReg optimiser')
parser.add_argument('--beta2_local', type=float, default=0.999, help='hessian momentum')
parser.add_argument('--h0_local', type=float, default=0.0, help='hessian init in first step')
parser.add_argument('--use_scheduler', type=bool, default=False, help='use a learning rate scheduler')
parser.add_argument('--temperature', type=float, default=1.0, help='scaling term in front of KL')
parser.add_argument('--mcsamples', type=int, default=1, help='number of MC samples in IVON')
parser.add_argument('--batch_size', type=int, default=64, help='number of data points in a (mini)batch')
parser.add_argument('--print_server_elbo', type=bool, default=False, help='for debugging, print the variational objective at server')
parser.add_argument('--num_steps_local', type=int, default=-1, help='if set, will do steps rather than epochs on the client')


# Logging and debugging parameters
parser.add_argument('--plot_comm_round', type=int, default=0,
                    help='how often to plot 2D visualisations (if 0, do not plot)')
parser.add_argument('--log_interval_epochs', type=int, default=10,
                    help='how often to test and log metrics (if 0, do not log)')
parser.add_argument('--fedivon_server_stepsize', type=float, default=None,
                    help='step size for global server update (FedIVON)')


# Initialise data generator and local workers
def init_data_generator_local_clients(args, seed):

    # Initialise data generator
    if 'toy_gaussians' in args.benchmark:
        setting = 2 if '_2' in args.benchmark else 1
        setting = 3 if '_3' in args.benchmark else setting
        data_generator = ToyDataGenerator(setting=setting, num_clients=args.num_clients, num_samples=200,
                                          polynomial_degree=3, seed=seed, multiclass=False, path=None)
    elif 'hetmnist' in args.benchmark:
        fashionmnist = True if 'fashion' in args.benchmark else False
        data_generator = FederatedHeterogenousMNISTDataGenerator(num_clients=args.num_clients,
                                                                 fashionmnist=fashionmnist, seed=seed)
    elif 'mnist' in args.benchmark:
        dataset_proportion = 0.1 if '10p' in args.benchmark else 1.
        data_split = 'non_iid' if 'non_iid' in args.benchmark else 'iid'
        fashionmnist = True if 'fashion' in args.benchmark else False
        data_generator = FederatedMNISTDataGenerator(num_clients=args.num_clients, seed=seed,
                                                     dataset_proportion=dataset_proportion, fashionmnist=fashionmnist,
                                                     data_split=data_split, path=None)
    elif 'cifar10' in args.benchmark:
        dataset_proportion = 0.1 if '10p' in args.benchmark else 1.
        data_split = 'non_iid' if 'non_iid' in args.benchmark else 'iid'
        cifar100 = True if 'cifar100' in args.benchmark else False
        data_generator = FederatedCIFAR10DataGenerator(num_clients=args.num_clients, seed=seed,
                                                       dataset_proportion=dataset_proportion,
                                                       data_split=data_split, cifar100=cifar100, path=None)
    else:
        raise ValueError("Do not currently support args.benchmark:", args.benchmark)

    # Separate out args into model and optimiser hyperparameters
    model_parameters = {
        'network_type': args.network_type,
        'act_func': "sigmoid",
        'D_in': data_generator.num_parameters,
        'D_out': data_generator.num_outputs,
        'hidden_sizes': args.hidden_sizes
    }

    # Only use local prior precision / weight decay if FedDyn or FedIVON
    if args.federated_method == 'FedDyn' or args.federated_method == 'FedIVON': 
        local_prior_prec = args.local_prior_prec 
    else:
        local_prior_prec = 0

    # Only do gradient clipping if greater than 0
    if args.grad_clip is not None and args.grad_clip <= 0:
        args.grad_clip = None

    optimiser_parameters = {
        'optimiser_type': args.opt_type,
        'learning_rate': args.learning_rate_local,
        'l2_regularisation': local_prior_prec,
        'grad_clip': args.grad_clip,
        'batch_size': args.batch_size,
        'beta2': args.beta2_local,
        'h0': args.h0_local,
        'temperature' : args.temperature,
        'mcsamples' : args.mcsamples,
        'scheduler' : args.use_scheduler,
        'batchupperbound' : False  
    }

    # Parameters to do with federated learning method and logging/debugging code
    misc_parameters = {
        'federated_method': args.federated_method,
        'federated_step_size': args.federated_step_size,
        'plot_comm_round': args.plot_comm_round,
        'log_interval_epochs': args.log_interval_epochs,
        'dual_update_damping': args.dual_update_damping,
        'num_comm_rounds': args.num_comm_rounds,
    }

    # Initialise local clients
    local_clients_list = []
    for client_ind in range(args.num_clients):

        # FedDyn: federated_step_size is scaled by N_k / N
        if misc_parameters['federated_method'] == 'FedDyn':
            misc_parameters['federated_step_size'] = \
                args.federated_step_size * (data_generator.num_points_per_client[client_ind]
                                            / np.sum(data_generator.num_points_per_client))

        # FedLap: dual_update_damping is scaled by N_k / N
        if args.dual_update_damping_multiply_Nk_N >= 1:
            misc_parameters['dual_update_damping'] = \
                args.dual_update_damping_multiply_Nk_N * data_generator.num_points_per_client[client_ind] / np.sum(data_generator.num_points_per_client)

        # Load local worker from stored file
        if args.path is not None and client_ind in args.load_pretrained_workers:
            new_client = torch.load(args.path)['local_clients_list'][client_ind]

        # If not loading worker, create new worker
        else:
            trainloader, testloader = data_generator.data_split(client_ind)
            if not data_generator.without_trainloaders:
                if 'cifar' not in args.benchmark:
                    trainloader = torch.utils.data.TensorDataset(trainloader)
                    testloader = torch.utils.data.TensorDataset(testloader)
                trainloader = torch.utils.data.dataloader.DataLoader(dataset=trainloader, batch_size=args.batch_size,
                                                                     shuffle=True, num_workers=2)
                testloader = torch.utils.data.dataloader.DataLoader(dataset=testloader, batch_size=args.batch_size,
                                                                     shuffle=True, num_workers=2)

            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            new_client = worker(model_parameters, optimiser_parameters, misc_parameters, trainloader, testloader,
                                client_ind, args.wandb_log, args.use_cuda, seed)

            # Add trainloader_small for FedLapCov Laplace calculation on CIFAR datasets
            if args.federated_method == 'FedLapCov' and 'cifar' in args.benchmark:
                trainloader_small, _ = data_generator.full_data()
                trainloader_small = torch.utils.data.dataloader.DataLoader(dataset=trainloader_small, batch_size=32,
                                                                           shuffle=False, num_workers=2)
                new_client.trainloader_small = trainloader_small

        local_clients_list.append(new_client)

    # Initialise global worker
    trainloader, testloader = data_generator.full_data()
    if not data_generator.without_trainloaders:
        if 'cifar' not in args.benchmark:
            trainloader = torch.utils.data.TensorDataset(trainloader)
            testloader = torch.utils.data.TensorDataset(testloader)
        trainloader = torch.utils.data.dataloader.DataLoader(dataset=trainloader, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=2)
        testloader = torch.utils.data.dataloader.DataLoader(dataset=testloader, batch_size=args.batch_size,
                                                            shuffle=True, num_workers=2)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    global_worker = worker(model_parameters, optimiser_parameters, misc_parameters, trainloader=trainloader,
                           testloader=testloader, client_ind=0, use_cuda=args.use_cuda, seed=seed)

    if args.use_cuda:
        global_worker.model = global_worker.model.cuda()

    return data_generator, local_clients_list, global_worker


# Upper bound: run on all data jointly, as if in one place
def upper_bound_batch_data(args, data_generator, seed):
    # Separate out args into model and optimiser hyperparameters
    model_parameters = {
        'network_type': args.network_type,
        'act_func': "sigmoid",
        'D_in': data_generator.num_parameters,
        'D_out': data_generator.num_outputs,
        'hidden_sizes': args.hidden_sizes
    }

    optimiser_parameters = {
        'optimiser_type': args.opt_type,
        'learning_rate': args.learning_rate_local,
        'l2_regularisation': args.global_prior_prec,
        'grad_clip': args.grad_clip,
        'batch_size': args.batch_size,
        'beta2': args.beta2_local,
        'h0': args.h0_local,
        'temperature' : args.temperature,
        'mcsamples' : args.mcsamples,
        'scheduler' : args.use_scheduler,
        'batchupperbound' : True # special settings so true objective is solved
    }

    # Parameters to do with federated learning method and logging/debugging code
    federated_method_batch = 'FedIVON' if args.federated_method == 'FedIVON' else 'FedAvg'
    misc_parameters = {
        'federated_method': federated_method_batch,
        'federated_step_size': args.federated_step_size,
        'dual_update_damping': args.dual_update_damping,
        'plot_comm_round': args.plot_comm_round,
        'log_interval_epochs': args.log_interval_epochs,
        'num_comm_rounds': args.num_comm_rounds,
    }

    # Initialise batch worker with all data
    trainloader, _ = data_generator.full_data()
    _, testloader = data_generator.data_split(client_ind=0)
    if not data_generator.without_trainloaders:
        if 'cifar' not in args.benchmark:
            trainloader = torch.utils.data.TensorDataset(trainloader)
            testloader = torch.utils.data.TensorDataset(testloader)
        trainloader = torch.utils.data.dataloader.DataLoader(dataset=trainloader, batch_size=args.batch_size,
                                                             shuffle=True, num_workers=2)
        testloader = torch.utils.data.dataloader.DataLoader(dataset=testloader, batch_size=args.batch_size,
                                                            shuffle=True, num_workers=2)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    batch_worker = worker(model_parameters, optimiser_parameters, misc_parameters, trainloader, testloader,
                          client_ind=0, wandb_log=args.wandb_log,
                          use_cuda=args.use_cuda, seed=seed)

    # Train batch worker
    print('Training upper bound batch worker')
    batch_worker.train(num_epochs=args.num_epochs_batch)
    print('')

    # Plot 2D visualisation
    if args.plot_comm_round > 0 and 'toy_gaussians' in args.benchmark:
        test_outputs = batch_worker.test_toydata(plot_dataloader=testloader)
        save_path = None 
        utils.plot_2d_visualisation([test_outputs], data_generator, worker_ind_list_const=range(args.num_clients),
                                    wandb_log=args.wandb_log, path=save_path, title='Batch worker, ')
        
    return batch_worker


# Main function for running experiment
def run_experiment(args, seed):

    # Initialise data generator and local clients list
    data_generator, local_clients_list, global_worker = init_data_generator_local_clients(args=args, seed=seed)

    # Run batch upper bound
    batch_worker = upper_bound_batch_data(args=args, data_generator=data_generator, seed=seed)
    batch_weights = return_model_parameters((batch_worker.model))
    global_worker.train_set_size = batch_worker.train_set_size  # Store total number of datapoints N in global_worker

    # For FedIVON, the server has a precision vector. We create it here.
    if args.federated_method == 'FedIVON':
        mean, prec = local_clients_list[0].optimiser._get_posterior()
        global_prec = torch.ones_like(prec[0]) * args.local_prior_prec 
        set_model_parameters(global_worker.model, torch.zeros_like(mean[0]))

    elif args.federated_method == 'FedLapCov':
        global_prec = args.federated_step_size * torch.ones_like(return_model_parameters(global_worker.model))

    # Start federated learning communication rounds, starting with local client training
    l2_dist_to_batch = []  # For tracking how far from ideal (batch) weights over communication rounds
    global_test_acc = []  # For tracking global server's performance over communication rounds
    global_test_nll = []  # For tracking global server's performance over communication rounds
    global_test_acc_bayes = []  # For tracking global server's performance over communication rounds (Bayesian sampling)
    global_test_nll_bayes = []  # For tracking global server's performance over communication rounds (Bayesian sampling)
    for comm_round_ind in range(args.num_comm_rounds):
        print('\nStarting communication round', comm_round_ind)

        # Train local clients
        test_outputs_list = []  # For 2D visualisation plots
        for client_ind, worker in enumerate(local_clients_list):
            print('Training client', client_ind)
            worker.train(num_epochs=args.num_epochs_local, num_steps=args.num_steps_local)
            print('')

            # Plot 2D visualisation
            if args.plot_comm_round > 0:
                test_outputs = worker.test_toydata()
                test_outputs_list.append(test_outputs)

        # Plot 2D visualisations for local clients
        if args.plot_comm_round > 0 and (comm_round_ind+1) % args.plot_comm_round == 0 and 'toy_gaussians' in args.benchmark:
            save_path = None
            utils.plot_2d_visualisation(test_outputs_list, data_generator,
                                        worker_ind_list_const=range(args.num_clients), wandb_log=args.wandb_log,
                                        path=save_path, title='Local clients, comm round %d, '%(comm_round_ind))

        # Update global weights
        if args.federated_method == 'FedAvg' or args.federated_method == 'FedProx':
            # FedAvg and FedProx: global weights are weighted average of local weights
            global_weights = torch.zeros_like(return_model_parameters(global_worker.model))
            if args.use_cuda:
                global_weights = global_weights.cuda()
            for client_ind, worker in enumerate(local_clients_list):
                # Global weights = sum_k (N_k/N) w_k
                global_weights += ((worker.train_set_size/global_worker.train_set_size)
                                   * return_model_parameters(worker.model))
            set_model_parameters(global_worker.model, global_weights)
        elif args.federated_method == 'FedADMM':
            # FedADMM: according to Gong et al. (2022) paper
            global_weights = return_model_parameters(global_worker.model)
            if args.use_cuda:
                global_weights = global_weights.cuda()
            for client_ind, worker in enumerate(local_clients_list):
                # NB Their delta_k simplifies to 2*w_k^{new} - w_g - w_k^{old} (do not need the dual weight values v_k)
                delta_k = (2*return_model_parameters(worker.model) - worker.received_global_weights
                           - worker.previous_model_weights)
                global_weights += args.global_step_size / args.num_clients * delta_k
            set_model_parameters(global_worker.model, global_weights)
        elif args.federated_method == 'FedDyn':
            # FedDyn: uses history vector
            new_weights = torch.zeros_like(return_model_parameters(global_worker.model))
            if comm_round_ind == 0:
                history_vector = torch.zeros_like(return_model_parameters(global_worker.model))
            else:
                for worker in local_clients_list:
                    history_vector += ((worker.misc_parameters['federated_step_size']/args.num_clients)
                                       *(return_model_parameters(worker.model) -
                                         return_model_parameters(global_worker.model)))

            for worker in local_clients_list:
                new_weights += (1/args.num_clients) * return_model_parameters(worker.model)

            new_weights += history_vector / args.federated_step_size
            set_model_parameters(global_worker.model, new_weights)
        elif args.federated_method == 'FedLap':
            # FedLap: w_g <-- \sum_k v_k
            global_weights = torch.zeros_like(return_model_parameters(global_worker.model))
            for client_ind, worker in enumerate(local_clients_list):
                global_weights += worker.dual_weights
            set_model_parameters(global_worker.model, global_weights)
        elif args.federated_method == 'FedLapCov':
            # FedLapCov: S_g <-- delta + \sum_k V_k; w_g <-- S_g^{-1} \sum_k v_k
            delta = args.federated_step_size
            global_prec = delta * torch.ones_like(return_model_parameters(global_worker.model))
            for client_ind, worker in enumerate(local_clients_list):
                global_prec += worker.dual_prec

            global_mean = torch.zeros_like(return_model_parameters(global_worker.model))
            for client_ind, worker in enumerate(local_clients_list):
                global_mean += worker.dual_mean
            global_mean = global_mean / global_prec
            set_model_parameters(global_worker.model, global_mean)
        elif args.federated_method == 'FedIVON':
            with torch.no_grad(): 
                global_natmean_avg = torch.zeros_like(return_model_parameters(global_worker.model))
                global_prec_avg = torch.zeros_like(return_model_parameters(global_worker.model))
                dual_mean_sum = torch.zeros_like(return_model_parameters(global_worker.model))
                dual_prec_sum = torch.zeros_like(return_model_parameters(global_worker.model))
                for client_ind, worker in enumerate(local_clients_list):
                    mean, prec = worker.optimiser._get_posterior()
                    global_prec_avg += prec[0] / len(local_clients_list)  # (1/K)* \sum_k S_k
                    global_natmean_avg += mean[0] * prec[0] / len(local_clients_list)  # (1/K)* \sum_k S_k m_k
                    dual_mean_sum += worker.dual_mean   # \sum_k v_k
                    dual_prec_sum += worker.dual_prec   # \sum_k V_k

                # IVON update 
                K = len(local_clients_list)

                if args.fedivon_server_stepsize:
                    alpha = args.fedivon_server_stepsize 
                else: 
                    rho = args.federated_step_size 
                    alpha = 1.0 / (1.0 + K * rho)

                prior_precision = args.local_prior_prec  # S_0

                global_prec = (1.0 - alpha) * global_prec_avg + alpha * (prior_precision - dual_prec_sum)
                global_mean = ((1.0 - alpha) * global_natmean_avg - alpha * dual_mean_sum) / global_prec

                # print the variational objective at server (for debugging)
                if args.print_server_elbo: 

                    criterion = nn.CrossEntropyLoss()
                    train_nll = 0
                    for (inputs, labels) in batch_worker.trainloader:
                        if args.use_cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        # Forward pass through model to evaluate expected loss
                        global_params = return_model_parameters(global_worker.model)
                        MC = 128
                        for _ in range(MC):  
                            eps = torch.randn_like(global_mean)
                            theta = global_mean + eps/torch.sqrt(global_prec)
                            set_model_parameters(global_worker.model, theta)
                            logits = global_worker.model(inputs)
                            loss = global_worker.loss_function(criterion, logits, labels) 
                            train_nll += loss.detach().item() 
                        set_model_parameters(global_worker.model, global_params)

                    expected_loss = train_nll / (MC * len(batch_worker.trainloader))
                    kl_term = kldivergence(
                        global_mean, 
                        1.0/global_prec, 
                        torch.zeros_like(global_mean), 
                        (1.0/prior_precision)*torch.ones_like(global_prec))   
                        
                    tau = args.temperature 
                    vo = (global_worker.train_set_size / args.temperature) * expected_loss + kl_term

                    print("Server's Variational Objective %.2f" % vo)

            set_model_parameters(global_worker.model, global_mean)
        else:
            raise ValueError('Invalid federated_method:', args.federated_method)

        # Test using global model
        dataloader = global_worker.trainloader[0] if 'toy' in args.benchmark else global_worker.testloader  # Train acc/nll if toy data
        test_accuracy, test_nll = global_worker.test(dataloader)
        global_test_acc.append(test_accuracy)
        global_test_nll.append(test_nll)
        print('Comm round %d: Test-acc: %.2f, test-nll: %.2f' % (comm_round_ind+1, test_accuracy, test_nll))
        if args.wandb_log:
            wandb.log({
                'Comm round': comm_round_ind,
                'global_nll': test_nll,
                'global_nll_seed%d'%seed: test_nll,
                'global_accuracy': test_accuracy,
                'global_accuracy_seed%d'%seed: test_accuracy
            })

            if args.benchmark == 'cifar10_non_iid' and comm_round_ind > 2 and test_accuracy < 20.0: # early stopping bad runs
                exit()

        if args.federated_method == 'FedIVON':
            test_accuracy_bayes, test_nll_bayes = global_worker.test_bayes(dataloader)
            print(f'  [Bayes Model-Avg] Test-acc: {test_accuracy_bayes:.2f}, test-nll: {test_nll_bayes:.2f}')
            global_test_acc_bayes.append(test_accuracy_bayes)
            global_test_nll_bayes.append(test_nll_bayes)
            if args.wandb_log:
                wandb.log({
                    'Comm round': comm_round_ind,
                    'global_nll_bayes': test_nll_bayes,
                    'global_nll_bayes_seed%d' % seed: test_nll_bayes,
                    'global_accuracy_bayes': test_accuracy_bayes,
                    'global_accuracy_bayes_seed%d' % seed: test_accuracy_bayes
                })

        # Plot 2D visualisation for global weights
        if args.plot_comm_round > 0 and (comm_round_ind+1) % args.plot_comm_round == 0 and 'toy_gaussians' in args.benchmark:
            test_outputs = global_worker.test_toydata()
            save_path = None
            utils.plot_2d_visualisation([test_outputs], data_generator, worker_ind_list_const=range(args.num_clients),
                                        wandb_log=args.wandb_log, path=save_path,
                                        title='Global, comm round %d, '%(comm_round_ind))

        # Calculate L2 distance between global weights and batch weights (upper baseline) if GLM
        if args.network_type == 'Linear':
            l2_norm_global_batch = torch.norm(return_model_parameters(global_worker.model) - batch_weights).item()
            print('L2 distance between global weights and upper baseline (batch) weights:', l2_norm_global_batch)
            l2_dist_to_batch.append(l2_norm_global_batch)
            if args.wandb_log:
                wandb.log({'Comm round': comm_round_ind,
                           'Norm_diff_global_upper': l2_norm_global_batch})

        # Set local workers parameters to global parameters
        for worker in local_clients_list:
            if args.federated_method == 'FedIVON':
                # set_model_parameters(worker.model, return_model_parameters(global_worker.model))
                worker.optimiser._set_prior(return_model_parameters(global_worker.model), global_prec)
                worker.optimiser._set_posterior([return_model_parameters(global_worker.model)], [global_prec])
                worker.received_global_mean = copy.copy(return_model_parameters(global_worker.model))
                worker.received_global_prec = copy.copy(global_prec)
            elif args.federated_method == 'FedLapCov':
                set_model_parameters(worker.model, return_model_parameters(global_worker.model))
                worker.received_global_mean = copy.copy(return_model_parameters(global_worker.model))
                worker.received_global_prec = copy.copy(global_prec)
            else:  
                worker.previous_model_weights = return_model_parameters(worker.model)
                set_model_parameters(worker.model, return_model_parameters(global_worker.model))
                worker.received_global_weights = copy.copy(return_model_parameters(global_worker.model))

        print('')

    # Store accuracies and NLL after specified number of communication rounds
    comm_rounds_of_interest = [10, 25, 50, 100, 150, 200, 250]
    comm_rounds_of_interest = [10, 20] if 'flamby_heart' in args.benchmark else comm_rounds_of_interest
    if args.wandb_log:
        for comm_round_int in comm_rounds_of_interest:
            if comm_round_int <= args.num_comm_rounds and comm_round_int >=3:
                global_nll_avg = np.mean(global_test_nll[comm_round_int-3:comm_round_int])
                global_acc_avg = np.mean(global_test_acc[comm_round_int-3:comm_round_int])
                wandb.log({
                    'global_nll_avg_seed%d_round%d' % (seed, comm_round_int): global_nll_avg,
                    'global_accuracy_avg_seed%d_round%d' % (seed, comm_round_int): global_acc_avg
                })
                if args.federated_method == 'FedIVON':
                    global_nll_avg_bayes = np.mean(global_test_nll_bayes[comm_round_int-3:comm_round_int])
                    global_acc_avg_bayes = np.mean(global_test_acc_bayes[comm_round_int-3:comm_round_int])
                    wandb.log({
                        'global_nll_avg_bayes_seed%d_round%d' % (seed, comm_round_int): global_nll_avg_bayes,
                        'global_accuracy_avg_bayes_seed%d_round%d' % (seed, comm_round_int): global_acc_avg_bayes
                    })
        # Also store the list of final accuracies and NLLs
        wandb.log({
            'global_accuracy_seed%d_list' % seed: global_test_acc,
            'global_nll_seed%d_list' % seed: global_test_nll
        })
        if args.federated_method == 'FedIVON':
            wandb.log({
                'global_accuracy_bayes_seed%d_list' % seed: global_test_acc_bayes,
                'global_nll_bayes_seed%d_list' % seed: global_test_nll_bayes
            })


if __name__ == '__main__':
    args = parser.parse_args()
        
    # Load config file (YAML)
    if args.config_file is not None:
        dict_args = vars(args)
        with open(args.config_file) as f:
            config = yaml.full_load(f)
        dict_args.update(config)

    # Log on Weights & Biases
    if args.wandb_log:
        wandb_run = wandb.init(project='bayesian-admm')  # , reinit=True)
        wandb.config.update(args)
        wandb.config.seed = args.seed_init

        # Overwrite args based on wandb.config to sync argparse arguments with wandb.config
        for key, value in wandb.config.items():
            setattr(args, key, value)

    print(args)

    # Run experiment using settings from args, repeated args.num_runs times with different random seeds
    for seed_add in range(args.num_runs):
        seed = args.seed_init + seed_add
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        run_experiment(args, seed)
