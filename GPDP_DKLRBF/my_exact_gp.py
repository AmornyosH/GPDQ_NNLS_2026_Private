import os
import torch
import numpy as np
from time import time
from utility import my_NN, my_utils

TORCH_SQRT_3 = torch.math.sqrt(torch.tensor(3.00, dtype=torch.float32))

class myExactGP(torch.nn.Module):
    def __init__(self, params_dict:dict, dataset:dict, parent_alg:str, cuda:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'exact_gp'
        self.kernel_fn = 'rbf'
        self.param_dict = params_dict
        self.env = self.param_dict['environment']

        # Get best trajectory
        _best_dataset = dataset

        self.num_sample = _best_dataset['arr_0']
        self.gp_training_size = self.param_dict['gp_num_sample']
        # self.gp_training_size = self.num_sample
        if self.gp_training_size > self.num_sample:
            self.gp_training_size = self.num_sample
        self.x_dim = int(self.param_dict['state_dim'])
        # self.x_dim =  2
        self.y_dim = int(self.param_dict['action_dim'])
        self.x_train_full = torch.tensor(_best_dataset['observations'], dtype=torch.float32)
        self.y_train_full = torch.tensor(_best_dataset['actions'], dtype=torch.float32)
        self.x_train_org = torch.tensor(self.x_train_full[:self.gp_training_size], dtype=torch.float32)

        self.cuda() if cuda else ...

        print('========== Create new GP record and models!')
        self.mll_append = []  # Loss storage
        # Create hyperparameters
        # self.sigma_p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-02)

        self.feature_extractor = my_NN.MLP_Relu(input_dim=self.x_dim, output_dim=self.x_dim)
        self.feature_extractor_optimizer = torch.optim.Adam(self.feature_extractor.parameters(), lr=1e-03)

        _start = 0
        # Select the first trajectory to be the main training matrices. 
        self.x_train = torch.tensor(self.x_train_full[0+_start:_start+self.gp_training_size, 0:self.x_dim], dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train_full[0+_start:_start+self.gp_training_size], dtype=torch.float32)

        # Initialised training kernels (K, K_inv).
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        print('GP training sample size: ', len(self.x_train))
        self.K = self.rbfKernel(X_1=self.feature_extractor(self.x_train), 
                                X_2=self.feature_extractor(self.x_train), noise=True)

    def rbfKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2

        kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)

        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        x_test = X_s[:, 0:self.x_dim]
        x_test = x_test.view(-1, self.x_dim)

        _z_train = self.feature_extractor(self.x_train)
        _z_test = self.feature_extractor(x_test)

        with torch.no_grad():
            _k_s = self.rbfKernel(X_1=_z_train, X_2=_z_test, noise=False)
            _k_ss = self.rbfKernel(X_1=_z_test, X_2=_z_test, noise=False)
            # Cholesky decomposition
            _L = torch.linalg.cholesky(self.K, upper=False)
            _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            _mean = (_k_s.T @ _alpha)
            _v = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            _var = _k_ss - (_v.T @ _v)

        return _mean, _var

    def myTraining(self, total_epoch:int, ft:bool=False):
        _batch_size = self.gp_training_size # H
        _gradient_step = self.gp_training_size // _batch_size

        _training_record = 0
        while _training_record < total_epoch:
            # _start_time = time()
            for g in range(_gradient_step):
                _batch_x = self.x_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_y = self.y_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_z = self.feature_extractor(_batch_x)

                self.optimizer.zero_grad()
                self.feature_extractor_optimizer.zero_grad()
                _k = self.rbfKernel(X_1=_batch_z, X_2=_batch_z, noise=True)

                _L = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, _batch_y, upper=False), upper=True)
                _mll = (-0.5 * (_batch_y).T @ _alpha) - \
                       torch.sum(torch.log(torch.diagonal(_L))) - \
                       (_batch_size*torch.log(torch.tensor(2*torch.pi))/2)

                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                self.feature_extractor_optimizer.step()
                self.optimizer.step()

                self.mll_append.append(_mll.tolist())

            _training_record += 1
        
        _z_train = self.feature_extractor(self.x_train)
        self.K = self.rbfKernel(X_1=_z_train, X_2=_z_train, noise=True)

        return _mll.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train}, path)