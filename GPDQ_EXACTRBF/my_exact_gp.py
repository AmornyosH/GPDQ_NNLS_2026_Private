import os
import torch
from time import time

TORCH_SQRT_3 = torch.math.sqrt(torch.tensor(3.00, dtype=torch.float32))

class myExactGP(torch.nn.Module):
    def __init__(self, params_dict:dict, dataset:dict, parent_alg:str, cuda:bool=False):
        super().__init__()
        self.parent_alg = parent_alg
        self.alg = 'exact_gp'
        self.kernel_fn = 'matern'
        self.param_dict = params_dict
        self.env = self.param_dict['environment']
        # self.num_sample = params_dict['gp_num_sample']  # Fixed to 1000.
        self.num_sample = dataset['arr_0']
        self.gp_training_size = self.param_dict['gp_num_sample']
        # self.gp_training_size = self.num_sample
        if self.gp_training_size > self.num_sample:
            self.gp_training_size = self.num_sample
        self.x_dim = int(self.param_dict['state_dim'])
        self.y_dim = int(self.param_dict['action_dim'])
        self.x_train_full = torch.tensor(dataset['observations'][0:self.gp_training_size], dtype=torch.float32)
        self.y_train_full = torch.tensor(dataset['actions'][0:self.gp_training_size], dtype=torch.float32)

        # self.training_record_path = '{}/training_records/{}/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.env, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)
        self.training_record_path = '{}/training_records/{}/{}_{}_{}_{}_training_records'.format(self.parent_alg, self.env, self.parent_alg, self.env, self.alg, self.kernel_fn)
        # self.training_record_path = '{}/training_records/walker2d-rbf-1000-medium-expert-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, 
        #                                                                                          self.env, 
        #                                                                                          self.alg, 
        #                                                                                          self.num_sample, 
        #                                                                                          self.kernel_fn)
        # self.training_record_path = '{}/training_records/walker2d-rbf-1000-medium-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)


        # self.training_record_path = '{}/training_records/hopper-rbf-1000-medium-expert-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)
        # self.training_record_path = '{}/training_records/hopper-rbf-1000-medium-replay-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)
        # self.training_record_path = '{}/training_records/hopper-rbf-1000-medium-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)
        # self.training_record_path = '{}/training_records/hopper-medium-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)
        # self.training_record_path = '{}/training_records/hopper-medium-expert-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)

        # self.training_record_path = '{}/training_records/halfcheetah-medium-replay-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.kernel_fn)

        # self.training_record_path = '{}/training_records/halfcheetah-rbf-1000-medium-expert-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, 
        #                                                                                          self.parent_alg, 
        #                                                                                          self.env, 
        #                                                                                          self.alg, 
        #                                                                                          self.num_sample, 
        #                                                                                          self.kernel_fn)

        # self.training_record_path = '{}/training_records/walker2d-rbf-1000-medium-v2/{}_{}_{}_{}_{}_training_records'.format(self.parent_alg, 
        #                                                                                          self.parent_alg, 
        #                                                                                          self.env, 
        #                                                                                          self.alg, 
        #                                                                                          self.num_sample, 
        #                                                                                          self.kernel_fn)

        self.training_checkpoint_path = '{}/training_records/{}_{}_{}_{}_{}_checkpoint'.format(self.parent_alg, 
                                                                                               self.parent_alg, 
                                                                                               self.env, self.alg, 
                                                                                               self.num_sample, 
                                                                                               self.kernel_fn)
        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.alg))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.alg))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== Create new GP record and models!')
            self.mll_append = []  # Loss storage
            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

            # self.mean_s = torch.nn.Parameter(torch.zeros(size=[self.gp_training_size, self.y_dim], dtype=torch.float32), requires_grad=True)
            # self.mean_s_p = my_NN.MLP(input_dim=self.x_dim, output_dim=self.y_dim)

            if self.kernel_fn == 'svm':
                # hyperparameter for svm kernel
                self.num_mixtures = self.x_dim//2
                self.mean_q = torch.nn.Parameter(torch.zeros(self.num_mixtures, self.x_dim), requires_grad=True)
                self.weight_q = torch.nn.Parameter(torch.ones(self.num_mixtures), requires_grad=True)
                self.v_q = torch.nn.Parameter(torch.ones(self.num_mixtures, self.x_dim), requires_grad=True)

            _start = 0
            # Select the first trajectory to be the main training matrices. 
            self.x_train = torch.tensor(self.x_train_full[0+_start:_start+self.gp_training_size], dtype=torch.float32)
            self.y_train = torch.tensor(self.y_train_full[0+_start:_start+self.gp_training_size], dtype=torch.float32)

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== Load training record.')
            # Load training records
            _training_records = torch.load(self.training_record_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            self.mll_append = _training_records['loss_append']  # Loss append
            _state_dict = _training_records['state_dict']

            self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)

            if self.kernel_fn == 'svm':
                # hyperparameter for svm kernel
                self.num_mixtures = self.x_dim//2
                self.mean_q = torch.nn.Parameter(_state_dict['mean_q'], requires_grad=True)
                self.weight_q = torch.nn.Parameter(_state_dict['weight_q'], requires_grad=True)
                self.v_q = torch.nn.Parameter(_state_dict['v_q'], requires_grad=True)

            self.x_train = _training_records['x_train']
            self.y_train = _training_records['y_train']
            self.eval()
        else:
            print('========== Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).')
            exit()

        # Initialised training kernels (K, K_inv).
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        # self.mean_s_p_optimizer = torch.optim.Adam(self.mean_s_p.parameters(), lr=1e-03)

        # self.ell_p = torch.tensor(1.0, dtype=torch.float32)
        # self.p = torch.tensor(1.0, dtype=torch.float32)
        if self.kernel_fn == 'rbf':
            self.K = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        elif self.kernel_fn == 'matern':
            self.K = self.maternKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        elif self.kernel_fn == 'svm':
            self.K = self.svmKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        else:
            print(' No kernel is selected. Please select "rbf" or "svm".')

    def loadTrainingRecord(self, path:str, cuda:bool):
        _training_records = torch.load(path, map_location=torch.device('cuda' if cuda else 'cpu'))
        # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
        self.mll_append = _training_records['loss_append']  # Loss append
        # self.training_record = _training_records['training_record']  # Training record counter
        _state_dict = _training_records['state_dict']
        self.x_train = _training_records['x_train']
        self.y_train = _training_records['y_train']
        # print(_state_dict)
        self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
        self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)

        # self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
        # self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
        # self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)

        # self.num_mixtures = self.param_dict['state_dim']
        # self.mean_q = torch.nn.Parameter(_state_dict['mean_q'], requires_grad=True)
        # self.weight_q = torch.nn.Parameter(_state_dict['weight_q'], requires_grad=True)
        # self.v_q = torch.nn.Parameter(_state_dict['v_q'], requires_grad=True)

        if self.kernel_fn == 'rbf':
            self.K = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        elif self.kernel_fn == 'matern':
            self.K = self.maternKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        elif self.kernel_fn == 'svm':
            self.K = self.svmKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        else:
            print(' No kernel is selected. Please select "rbf" or "svm".')

        self.x_train_full = self.x_train
        self.y_train_full = self.y_train

    def rbfKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)

        # rbf_kernel = torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)
        # # Periodic kernel
        # period_kernel = torch.exp(-(2/(self.ell_p**2))*torch.sin(torch.pi*torch.cdist(X_1, X_2)/self.p)**2)
        # # Combine kernels
        # kernel = (self.sigma_p**2) * period_kernel * rbf_kernel

        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def maternKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        
        r_ard = torch.cdist(X_1/self.ell, X_2/self.ell)
        sqrt3_r = TORCH_SQRT_3 * (r_ard)
        kernel = (self.sigma_p**2) * (1 + sqrt3_r) * torch.exp(-sqrt3_r)

        # rbf_kernel = torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)
        # # Periodic kernel
        # period_kernel = torch.exp(-(2/(self.ell_p**2))*torch.sin(torch.pi*torch.cdist(X_1, X_2)/self.p)**2)
        # # Combine kernels
        # kernel = (self.sigma_p**2) * period_kernel * rbf_kernel

        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def svmKernel(self, X_1, X_2, noise=False):
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2     
        kernel = 0
        for i in range(self.num_mixtures):
            _diff = torch.cdist(X_1*self.v_q[i], X_2*self.v_q[i])
            _exp_term = torch.exp(-2 * (torch.pi**2) * (_diff**2))
            _cos_term = torch.cos(2 * torch.pi * torch.cdist(X_1*self.mean_q[i], X_2*self.mean_q[i]))
            kernel += self.weight_q[i] * (_exp_term * _cos_term)
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))
        return kernel    

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            # _k_s = self.rbfKernel(X_1=self.x_train, X_2=x_test, noise=False)
            # _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)
            # _k_s = self.svmKernel(X_1=self.x_train, X_2=x_test, noise=False)
            # _k_ss = self.svmKernel(X_1=x_test, X_2=x_test, noise=False)
            _k_s = self.maternKernel(X_1=self.x_train, X_2=x_test, noise=False)
            _k_ss = self.maternKernel(X_1=x_test, X_2=x_test, noise=False)


            # # We found that deriving mean and variance this way, provide better results.
            # _mean = _k_s.T @ self.K_inv @ self.y_train
            # _var = _k_ss - _k_s.T @ self.K_inv @ _k_s

            # Cholesky decomposition
            _L = torch.linalg.cholesky(self.K, upper=False)
            _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            _mean = _k_s.T @ _alpha
            _v = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            _var = _k_ss - (_v.T @ _v)

        return _mean, _var

    def myTraining(self, total_epoch:int, ft:bool=False):
        # _batch_size = self.num_sample if not ft else self.x_train.size()[0]  # minibatch size is the whole dataset.
        _batch_size = self.gp_training_size # H
        _gradient_step = self.num_sample // _batch_size
        _optimizer = torch.optim.Adam(self.parameters(), lr=1e-02)
        self.train()  # Set to training mode.
        _training_record = 0
        while _training_record < total_epoch:
            _start_time = time()
            for g in range(_gradient_step):
                _batch_x = self.x_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_y = self.y_train[g*_batch_size:_batch_size+(g*_batch_size)]

                _optimizer.zero_grad()
                # _k = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                # _k = self.svmKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _k = self.maternKernel(X_1=_batch_x, X_2=_batch_x, noise=True)

                _L = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, _batch_y, upper=False), upper=True)

                _mll = (-0.5 * _batch_y.T @ _alpha) - \
                       torch.sum(torch.diagonal(_L)) - \
                       (_batch_size*torch.log(torch.tensor(2*torch.pi))/2)
                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                _optimizer.step()

                self.mll_append.append(_mll.tolist())

            _training_record += 1
            
            # print('Gradient_Step: ', _training_record, 
            #         ', Loss: ', _mll.tolist(), 
            #         ', Time_usage: ', round(time()-_start_time, 4))

        # # Assign 
        # self.x_train = self.x_train_full[0:self.gp_training_size]
        # self.y_train = self.y_train_full[0:self.gp_training_size]

        self.recordSaving(path=self.training_record_path)
        
        # self.K = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        # self.K = self.svmKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.K = self.maternKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        # self.K_inv = torch.linalg.inv(self.K)
        self.eval()  # Set to evaluation mode after the training is finished.

        return _mll.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train}, path)