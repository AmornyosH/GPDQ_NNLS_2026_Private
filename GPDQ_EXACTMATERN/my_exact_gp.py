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
        self.kernel_fn = 'matern'
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
        self.y_train_org = torch.tensor(self.y_train_full[:self.gp_training_size], dtype=torch.float32)

        self.cuda() if cuda else ...

        print('========== Create new GP record and models!')
        self.mll_append = []  # Loss storage
        # Create hyperparameters
        # self.sigma_p = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-02)

        _start = 0
        # Select the first trajectory to be the main training matrices. 
        self.x_train = torch.tensor(self.x_train_full[0+_start:_start+self.gp_training_size, 0:self.x_dim], dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train_full[0+_start:_start+self.gp_training_size], dtype=torch.float32)

        # Initialised training kernels (K, K_inv).
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        print('GP training sample size: ', len(self.x_train))
        self.K = self.maternKernel(X_1=self.x_train, X_2=self.x_train, noise=True)

    def maternKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
 
        r_ard = torch.cdist(X_1/self.ell, X_2/self.ell)
        sqrt3_r = TORCH_SQRT_3 * (r_ard)
        kernel = (self.sigma_p**2) * (1 + sqrt3_r) * torch.exp(-sqrt3_r)

        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        x_test = X_s[:, 0:self.x_dim]
        x_test = x_test.view(-1, self.x_dim)

        with torch.no_grad():
            _k_s = self.maternKernel(X_1=self.x_train, X_2=x_test, noise=False)
            _k_ss = self.maternKernel(X_1=x_test, X_2=x_test, noise=False)
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

                self.optimizer.zero_grad()
                _k = self.maternKernel(X_1=_batch_x, X_2=_batch_x, noise=True)

                _L = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, _batch_y, upper=False), upper=True)
                _mll = (-0.5 * (_batch_y).T @ _alpha) - \
                       torch.sum(torch.log(torch.diagonal(_L))) - \
                       (_batch_size*torch.log(torch.tensor(2*torch.pi))/2)

                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                self.optimizer.step()

                self.mll_append.append(_mll.tolist())

            _training_record += 1
            
        self.K = self.maternKernel(X_1=self.x_train, X_2=self.x_train, noise=True)

        return _mll.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train}, path)