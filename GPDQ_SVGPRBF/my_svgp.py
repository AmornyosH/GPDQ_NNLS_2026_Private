import os
import torch
import numpy as np
from time import time

TORCH_SQRT_3 = torch.math.sqrt(torch.tensor(3.00, dtype=torch.float32))

class mySVGP(torch.nn.Module):
    def __init__(self, parent_alg:str, params_dict:dict, dataset:dict, cuda:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'svgp'
        self.kernel_fn = 'rbf'
        self.param_dict = params_dict
        self.env = params_dict['environment']
        self.num_sample = dataset['arr_0']
        self.gp_training_size = self.param_dict['gp_num_sample']
        if self.gp_training_size > self.num_sample:
            self.gp_training_size = self.num_sample
        self.num_inducing = params_dict['gp_num_inducing']
        self.x_dim = int(params_dict['state_dim'])
        self.y_dim = int(params_dict['action_dim'])

        self.training_record_path = '{}/training_records/{}/{}_{}_{}_training_records.zip'.format(self.parent_alg, self.env, self.parent_alg, self.env, self.alg)
        self.x_train_full = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.y_train_full = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.x_train_org = torch.tensor(self.x_train_full, dtype=torch.float32)
        self.y_train_org = torch.tensor(self.y_train_full, dtype=torch.float32)
        self.cuda() if cuda else ...

        # Check for the answer...
        print('========== ({:s}) Create new GP record and models!'.format(self.alg))
        self.mll_append = []  # Loss storage
        self.training_record = 0  # Training record counter
        # Initialise training data
        _start = 0
        self.x_train = self.x_train_full
        self.y_train = self.y_train_full
        # Get inducing indexes
        self.z_train = self.x_train[0:self.num_inducing]
        self.q_mean = self.y_train[0:self.num_inducing]

        # Create hyperparameters
        self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
        self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)
        self.q_mean = torch.nn.Parameter(self.q_mean, requires_grad=True)
        self.q_var = torch.nn.Parameter(torch.ones([self.num_inducing, self.num_inducing]), requires_grad=True)  # Single covariance
        self.z_train = torch.nn.Parameter(self.z_train, requires_grad=True)

        # Summarize parameters
        self.num_sample = len(self.x_train)
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        # print('========== ({:s}) Summarize GP parameters.'.format(self.alg))
        # print('sigma_p: ', self.sigma_p.tolist(), ', sigma_n: ', self.sigma_n.tolist(), ', ell: ', self.ell.tolist())
        # print('q_mean: ', self.q_mean.tolist(), ', q_var: ', torch.mean(torch.diag(self.q_var)).tolist(), ', z_size: ', self.z_train.size())
        self.cov_optimizer = torch.optim.Adam([self.sigma_n, self.ell], lr=1e-02)
        self.ind_optimizer = torch.optim.Adam([self.q_mean, self.q_var, self.z_train], lr=1e-04)  # was 1e-04

        # Initialised training kernels (K, K_inv) (For evaluation).
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    def loadTrainingRecord(self, path:str, cuda:bool):
        _training_records = torch.load(path, map_location=torch.device('cuda' if cuda else 'cpu'))
        # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
        self.mll_append = _training_records['loss_append']  # Loss append
        self.training_record = _training_records['training_record']  # Training record counter
        _state_dict = _training_records['state_dict']
        self.x_train = _training_records['x_train']
        self.y_train = _training_records['y_train']
        
        self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
        self.ell_1 = torch.nn.Parameter(_state_dict['ell_1'], requires_grad=True)
        self.ell_2 = torch.nn.Parameter(_state_dict['ell_2'], requires_grad=True)
        self.ell_3 = torch.nn.Parameter(_state_dict['ell_3'], requires_grad=True)
        self.ell_4 = torch.nn.Parameter(_state_dict['ell_4'], requires_grad=True)

        self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
        self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
        # self.q_var = torch.clip(_state_dict['q_var'], min=0.1)
        self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)

        # self.num_mixtures = self.param_dict['state_dim']
        # self.mean_q = torch.nn.Parameter(_state_dict['mean_q'], requires_grad=True)
        # self.weight_q = torch.nn.Parameter(_state_dict['weight_q'], requires_grad=True)
        # self.v_q = torch.nn.Parameter(_state_dict['v_q'], requires_grad=True)

        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    def rbfKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2

        # Summative Kernel
        # kernel_1 = torch.clip(torch.exp(-(torch.cdist(X_1[:, 1:5]/self.ell_1, X_2[:, 1:5]/self.ell_1)**2)/2), min=1e-08, max=1)
        # kernel_2 = torch.clip(torch.exp(-(torch.cdist(X_1[:, 5:7]/self.ell_2, X_2[:, 5:7]/self.ell_2)**2)/2), min=1e-08, max=1)
        # kernel_3 = torch.clip(torch.exp(-(torch.cdist(X_1[:, 7:11]/self.ell_3, X_2[:, 7:11]/self.ell_3)**2)/2), min=1e-08, max=1)
        # kernel_4 = torch.clip(torch.exp(-(torch.cdist(X_1[:, 0].reshape(-1, 1)/self.ell_4, X_2[:, 0].reshape(-1, 1)/self.ell_4)**2)/2), min=1e-08, max=1)
        # kernel_1 = ((self.sigma_p**2)/4) * torch.exp(-(torch.cdist(X_1[:, 1:5]/self.ell_1, X_2[:, 1:5]/self.ell_1)**2)/2)
        # kernel_2 = ((self.sigma_p**2)/4) * torch.exp(-(torch.cdist(X_1[:, 5:7]/self.ell_2, X_2[:, 5:7]/self.ell_2)**2)/2)
        # kernel_3 = ((self.sigma_p**2)/4) * torch.exp(-(torch.cdist(X_1[:, 7:11]/self.ell_3, X_2[:, 7:11]/self.ell_3)**2)/2)
        # kernel_4 = ((self.sigma_p**2)/4) * torch.exp(-(torch.cdist(X_1[:, 0].reshape(-1, 1)/self.ell_4, X_2[:, 0].reshape(-1, 1)/self.ell_4)**2)/2)
        # kernel = (self.sigma_p**2) * (kernel_1 + kernel_2 + kernel_3 + kernel_4)

        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)

        # Add observation noise
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            # _k_s = self.rbfKernel(X_1=self.z_train, X_2=x_test, noise=False)
            # _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            _k_s = self.rbfKernel(X_1=self.z_train, X_2=x_test, noise=False)
            _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            # _k_s = self.maternKernel(X_1=self.z_train, X_2=x_test, noise=False)
            # _k_ss = self.maternKernel(X_1=x_test, X_2=x_test, noise=False)

            # # We found that deriving mean and variance this way, provide better results.
            # _mean = _k_s.T @ self.K_mm_inv @ self.y_train
            # _var = _k_ss - _k_s.T @ self.K_mm_inv @ _k_s

            # # Cholesky decomposition
            # _L = torch.linalg.cholesky(self.K, upper=False)
            # _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            # _mean = _k_s.T @ _alpha

            # _mean = _k_s.T @ (torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.y_train, upper=False), upper=True))
            _mean = _k_s.T @ (torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_mean, upper=False), upper=True))
            _v = torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False)
            _k_tilde = _k_ss - (_v.T @ _v)
            _1 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, self.q_var, upper=False), upper=True)
            _2 = torch.linalg.solve_triangular(self.L_mm.T, torch.linalg.solve_triangular(self.L_mm, _k_s, upper=False), upper=True)
            _sigma_f = _k_tilde + (_k_s.T @ _1 @ _2)
            _var = (_sigma_f) + (self.sigma_n**2)

            # print(_sigma_f)

        return _mean, _var

    # Training Method for GPR. 
    # The reason we train GP here is to add entropy term to regulate the covariance. 
    def myTraining(self, total_epoch:int, ft:bool=False):
        _batch_size = self.param_dict['gp_batch_size']
        _gradient_step = int(self.num_sample//_batch_size)  
        _training_record = 0

        while _training_record < total_epoch:
        # while self.training_record < total_epoch:
            # _start_time = time()
            _sampling_indexes = torch.randperm(self.num_sample)
            for g in range(_gradient_step):

                # print('sigma_n: ', self.sigma_n)
                # print('ell: ', self.ell_1, self.ell_2, self.ell_3, self.ell_4)
                # print('q_mean: ', self.q_mean)
                # print('q_var: ', self.q_var)

                _batch_x = self.x_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_y = self.y_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_z = self.z_train
                _l_1 = 0
                self.cov_optimizer.zero_grad()
                self.ind_optimizer.zero_grad()

                # Clip q_var
                # _q_var = torch.clip(self.q_var, min=1e-08, max=1)
                _q_var = self.q_var

                _K_mm = self.rbfKernel(X_1=_batch_z, X_2=_batch_z, noise=True)
                _K_nn = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _K_mn = self.rbfKernel(X_1=_batch_z, X_2=_batch_x, noise=False)
                _L_mm = torch.linalg.cholesky(_K_mm, upper=False)
                # _K_mm_inv = torch.inverse(_K_mm)

                # print(_K_mm)
                # print(self.sigma_n)

                _K_tilde = _K_nn - _K_mn.T @ (torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _K_mn, upper=False), upper=True))
                # _K_tilde = _K_nn - _K_mn.T @ _K_mm_inv @ _K_mn

                _1 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _q_var, upper=False), upper=True)
                _3 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_mean, upper=False), upper=True)
                _sigma_n_sq = self.sigma_n**2
                for h in range(_batch_size):
                    _k_i = _K_mn[:, h].view(-1, 1)

                    _mean = _k_i.T @ _3
                    _2 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _k_i, upper=False), upper=True)
                    _sigma_f_sq = torch.abs(_K_tilde[h, h].view(1, 1) + (_k_i.T @ _1 @ _2))

                    # _mean = _k_i.T @ _K_mm_inv @ self.q_mean
                    # _sigma_f_sq = torch.abs(_K_tilde[h, h].view(1, 1) + (_k_i.T @ _K_mm_inv @ self.q_var @ _K_mm_inv @ _k_i))

                    _l_1 += ((-torch.mean(_batch_y[h].view(1, self.y_dim)-_mean, dim=1)**2) / (2*_sigma_n_sq)) - \
                            (torch.log(torch.sqrt(2*torch.pi*_sigma_n_sq))) - \
                            (_sigma_f_sq / (2*_sigma_n_sq))
                    # print(_l_1)
                # _l_1 = torch.clip(_l_1, min=-100000)
                # print('L_1: ', _l_1)

                # KL-divergence term
                # print(self.q_var)
                # _cov_q = torch.sqrt(torch.abs(torch.diagonal(self.q_var))) @ torch.ones(size=[self.num_inducing, self.y_dim])
                _cov_q = torch.sqrt(torch.abs(_q_var)) @ torch.ones(size=[self.num_inducing, self.y_dim])
                _q = torch.distributions.Normal(loc=self.q_mean, scale=_cov_q)
                _u = _q.sample(torch.Size())
                _log_q_u = _q.log_prob(_u)
                
 
                # _p = torch.distributions.Normal(loc=0., 
                #                                 scale=torch.sqrt(torch.abs(torch.diagonal(_K_mm))) @ torch.ones(size=[self.num_inducing, self.y_dim]))
                _p = torch.distributions.Normal(loc=0., 
                                                scale=torch.sqrt(_K_mm) @ torch.ones(size=[self.num_inducing, self.y_dim]))
                # _log_p_u = _p.log_prob(_u)
                # print(_log_q_u)
                # print(_p.log_prob(_u))
                # ELBO
                _elbo = _l_1 - (-torch.sum(torch.exp(_log_q_u)*(_p.log_prob(_u)-_log_q_u)))
                # _elbo = _l_1 - (torch.sum(torch.exp(_log_p_u) * (_log_p_u - _log_q_u)))
                _elbo = -_elbo.mean()  # We maximise ELBO.
                _elbo.backward(retain_graph=True)
                self.cov_optimizer.step()
                self.ind_optimizer.step()

                # print(_elbo)

                self.mll_append.append(_elbo.tolist())

            self.training_record += 1
            _training_record += 1

            # print('Gradient_Step: ', self.training_record, 
            #         ', Loss: ', round(_elbo.tolist(), 4), 
            #         ', Var: ', round(torch.diagonal(self.q_var).mean().tolist(), 4),
            #         ', Obs_var: ', round(self.sigma_n.tolist(), 4),
            #         ', Time_usage: ', round(time()-_start_time, 4))

        # self.recordSaving(path=self.training_record_path)
        self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        # self.K_mm_inv = torch.inverse(self.K_mm)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        return _elbo.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'training_record': self.training_record,
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train,
                    'z_train': self.z_train}, path)