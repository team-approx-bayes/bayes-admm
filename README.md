# BayesADMM
We provide PyTorch code to repeat all experiments from the following paper: 

**Federated ADMM from Bayesian Duality**\
*T. Möllenhoff\*, S. Swaroop\*, F. Doshi-Velez, M.E. Khan*\
preprint, arXiv:xxxx.xxxxx, [paper link](.)

## Figure 3
Go to the toyexamples folder, e.g.,  ``cd toyexamples``. 

Run ``main_elbo.py`` and ``main_onestep.py``and then: 
- ``plot_elbo.py``for Figure 3(a): see ``results/elbo.pdf``. 
- ``plot_onestep.py``for Figure 3(b): see ``results/onestep.pdf``. 

## Figure 4
Run ``main_illustration.py``and then ``plot_illustration.py``. See for example ``results/illu_client0_bayes.pdf`` and the other similarly named files it generates. 

## Deep Learning Experiments
Run ``train.py --config_file [config]`` with: 

### For Table 1
- FMNIST (10 clients, homog.): configs/fmnist_homog10/[method].yaml
- FMNIST (10 clients, heterog.): configs/fmnist_heterog10/[method].yaml
- FMNIST (100 clients, heterog.): configs/fmnist_heterog100/[method].yaml
- CIFAR-10 (CNN from Zenke et al.): configs/cifar10_zenke/[method].yaml
- CIFAR-10 (CNN from Acar et al.): configs/cifar10_acar/[method].yaml

Possible choices for method are: 
- [FedAvg](https://arxiv.org/abs/1602.05629) (fedavg.yaml)
- [FedProx](https://arxiv.org/abs/1812.06127) (fedprox.yaml)
- [FedDyn](https://arxiv.org/abs/2111.04263) (feddyn.yaml)
- [FedLap](https://arxiv.org/abs/2501.17325v1) (fedlap.yaml)
- [FedLapCov](https://arxiv.org/abs/2501.17325v1) (fedlapcov.yaml)
- Our proposed BayesADMM (fedivon.yaml) 

The baseline results for FMNIST and CIFAR-10 (CNN from Zenke et al.) are from [Swaroop et al., 2025](https://arxiv.org/abs/2501.17325v1). We do not provide config files for those experiments here. We do provide configs for all baselines for the CIFAR-10 (CNN from Acar et al.) experiment. 

### For Table 2
- MNIST (100 clients, highly heterog.): configs/mnist_heterog/[method].yaml
- CIFAR-100 (ResNet-20, heterog.): configs/cifar100_resnet20/[method].yaml

For the specific method, select the config file, for instance to run FedProx on CIFAR-100 (ResNet-20, heterog.), use:
```
python3 train.py --config_file configs/cifar100_resnet20/fedprox.yaml
```

Note that results vary per seed, and to obtain the exact numbers from the paper, run the above for seeds 62, 63, and 64 and average the numbers. The seed can be changed in the config file. 


## How to cite

```
@article{MoellenhoffSwaroop2025,
      title={Federated ADMM from Bayesian Duality}, 
      author={M{\"o}llenhoff, Thomas and Swaroop, Siddharth and Doshi-Velez, Finale and Khan, Mohammad Emtiyaz},
      journal={arXiv:xxxx.xxxxx},
      year={2025},
      url={https://arxiv.org/abs/xxxx.xxxxx}
}
```