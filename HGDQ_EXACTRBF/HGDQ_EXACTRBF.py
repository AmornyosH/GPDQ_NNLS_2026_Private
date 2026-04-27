'''
Gaussian Process Diffusion Q-learning
Version: v0
Revision: 1
Remark: PyTorch version
'''
# ============================ Pytorch Related ============================
import torch
CUDA = torch.cuda.is_available()
if CUDA:
    print('CUDA is activated.')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# ============================ Pytorch Related ============================

# ================================== Others ==================================
from HGDQ_EXACTRBF import my_exact_gp
from utility import my_utils, my_NN
from time import time
import numpy as np
import os

class HiearchicalGaussianprocessDiffusionQlearning:
    def __init__(self, params_dict:dict, dataset:dict, ft:bool=False):
        self.ALG = 'HGDQ_EXACTRBF'
        self.ENV_CONFIG = params_dict['environment']
        self.STATE_DIM = int(params_dict['state_dim'])
        self.ACTION_DIM = int(params_dict['action_dim'])
        self.GOAL_DIM = int(params_dict['goal_dim'])
        self.high_state_dim = int(2 + self.GOAL_DIM)  # High-level state dimension
        self.cuda = CUDA
        self.num_action_candidates = 10
        self.NUM_SAMPLE = dataset['observations'].shape[0]
        
        if 'medium-play' in self.ENV_CONFIG:
            self.final_goal = torch.tensor([20.36824249435748, 21.156394661806964], dtype=torch.float32)
        else:
            # umaze
            self.final_goal = torch.tensor([0.36824249435748196, 9.156394661806964], dtype=torch.float32)
        
        # Initialise replay buffers
        self.state_buffer = torch.tensor(dataset['observations'], dtype=torch.float32)        # State buffer (unnormalised)
        self.high_state_buffer = torch.concat([self.state_buffer[:, :2], self.final_goal.repeat(self.NUM_SAMPLE, -1)], dim=1)  # High-level state buffer
        self.next_state_buffer = torch.tensor(dataset['next_observations'], dtype=torch.float32)   # Next state buffer (unnormalised)
        self.high_next_state_buffer = torch.concat([self.next_state_buffer[:, :2], self.final_goal.repeat(self.NUM_SAMPLE, -1)], dim=1)  # High-level next state buffer
        self.action_buffer = torch.tensor(dataset['actions'], dtype=torch.float32)       # Action buffer
        self.reward_buffer = torch.tensor(dataset['rewards'], dtype=torch.float32).view(-1, 1)  
        self.MINIBATCH_SIZE = 256

        # Initialise Diffusion Model's Parameters
        self.initialiseDiffusionParams(schedule='vp', beta_min=1, beta_max=10, num_step=params_dict['diffusion_step'], dec_step=params_dict['diffusion_step'])

        # Initialise Paths
        self.training_record_path = '/home/amornyos/PhD/Packages/resources/training_records/{}/{}/{}_{}_training_records'.format(self.ALG, self.ENV_CONFIG, self.ALG, self.ENV_CONFIG)
        self.testing_record_path = '/home/amornyos/PhD/Packages/resources/test_results/{}/{}_{}_test_results'.format(self.ALG, self.ENV_CONFIG, self.ALG, self.ENV_CONFIG)
        
        # Initialise neural networks
        self.EPSILON_INPUT_DIM = self.high_state_dim + self.GOAL_DIM + self.POS_DIM
        self.EPSILON_BEH_INPUT_DIM = self.GOAL_DIM + self.POS_DIM
        self.HIGH_Q_INPUT_DIM = self.high_state_dim + self.GOAL_DIM
        self.LOW_POLICY_INPUT_DIM = self.STATE_DIM + self.GOAL_DIM
        self.LOW_Q_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM
        # High-level policy
        self.epsilon_beh = my_NN.MLP(input_dim=self.EPSILON_INPUT_DIM, output_dim=self.GOAL_DIM)
        # Low-level policy
        self.low_policy = my_NN.MLP_Relu(input_dim=self.LOW_POLICY_INPUT_DIM, output_dim=self.ACTION_DIM)
        # Value networks
        self.high_q = my_NN.MLP_Qnetwork(input_dim=self.HIGH_Q_INPUT_DIM, output_dim=1)
        self.low_q = my_NN.MLP_Qnetwork(input_dim=self.LOW_Q_INPUT_DIM, output_dim=1)
        # Create target networks
        self.high_q_tar = self.high_q
        self.low_q_tar = self.low_q
        self.epsilon_beh_tar = self.epsilon_beh
        self.low_policy_tar = self.low_policy
        # Declare optimizer for the networks (offline training)
        self.epsilon_optimizer = torch.optim.Adam(self.epsilon_beh.parameters(), lr=3e-04)
        self.high_q_optimizer = torch.optim.Adam(self.high_q.parameters(), lr=3e-04)
        self.low_q_optimizer = torch.optim.Adam(self.low_q.parameters(), lr=3e-04)
        self.low_policy_optimizer = torch.optim.Adam(self.low_policy.parameters(), lr=3e-04)
        # Set to cuda if GPU is available.
        if CUDA:
            self.epsilon_beh.cuda()
            self.high_q.cuda()
            self.low_q.cuda()
            self.low_policy.cuda() 

        # ======================================= Create GP model. =======================================
        # Uncomment out the selected gp model type...
        self.gp_model_type = 'exact'
        best_dataset = my_utils.mazeTrajExtraction(dataset=dataset, max_episode_steps=1000, top_k=100)

        if self.gp_model_type == 'exact':
            self.gp_model = my_exact_gp.myExactGP(params_dict=params_dict, dataset=best_dataset, cuda=CUDA, parent_alg=self.ALG)
        else:
            print('No gp model type. Please try "exact" or "sparse".')

        # ======================================= Create/Load Training Record =======================================
        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'
        # _ans_1 = 'y'

        # Check for the answer...
        # --------------------------- No loading, creating new models
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new record and models!'.format(self.ALG))
            self.training_record = 0
            self.gradient_step = 0
            self.beh_training_record = 0
            self.best_norm_reward_training = 0
            self.norm_reward_training_append = []
            self.epsilon_beh_loss_append = []
            self.high_q_loss_append = []
            self.low_q_loss_append = []
            self.low_policy_loss_append = []

            # # Load pretrained behaviour model
            # _load_beh = torch.load(self.training_record_path+'_pretrained', map_location=torch.device('cpu' if not CUDA else 'cuda'))
            # self.epsilon_beh.load_state_dict(_load_beh['epsilon_beh'])
            # self.epsilon_optimizer.load_state_dict(_load_beh['epsilon_beh_optimizer'])
            # # target network
            # self.epsilon_beh_tar.load_state_dict(_load_beh['epsilon_beh'])
            # self.epsilon_optimizer.load_state_dict(_load_beh['epsilon_beh_optimizer'])

        # --------------------------- Loading, continue training of models
        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            self.loadTrainingRecord()  # Load from the method here <--------- ****
            self.loadGPTrainingRecord()
            self.epsilon_beh.eval()
            self.high_q.eval()
            self.low_q.eval()
            self.low_policy.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (load existing one) or "n" for no (create new)).'.format(self.ALG))
            exit()

    # Training Record Loading Method
    def loadTrainingRecord(self, path:str=None):
        if path is None:
            _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
            # _loaded_training_record = torch.load(self.training_checkpoint_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        else:
            _loaded_training_record = torch.load(path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        self.training_record = _loaded_training_record['training_record']
        self.gradient_step = _loaded_training_record['gradient_step']
        self.epsilon_beh.load_state_dict(_loaded_training_record['epsilon_beh'])
        self.epsilon_beh_tar.load_state_dict(_loaded_training_record['epsilon_beh_tar'])
        self.high_q.load_state_dict(_loaded_training_record['high_q'])
        self.high_q_tar.load_state_dict(_loaded_training_record['high_q_tar'])
        self.low_q.load_state_dict(_loaded_training_record['low_q'])
        self.low_q_tar.load_state_dict(_loaded_training_record['low_q_tar'])
        self.low_policy.load_state_dict(_loaded_training_record['low_policy'])
        self.low_policy_tar.load_state_dict(_loaded_training_record['low_policy_tar'])
        self.epsilon_optimizer.load_state_dict(_loaded_training_record['epsilon_beh_optimizer'])
        self.high_q_optimizer.load_state_dict(_loaded_training_record['high_q_optimizer'])
        self.low_q_optimizer.load_state_dict(_loaded_training_record['low_q_optimizer'])
        self.low_policy_optimizer.load_state_dict(_loaded_training_record['low_policy_optimizer'])
        self.epsilon_beh_loss_append = _loaded_training_record['epsilon_beh_loss_append']
        self.high_q_loss_append = _loaded_training_record['high_q_loss_append']
        self.low_q_loss_append = _loaded_training_record['low_q_loss_append']
        self.low_policy_loss_append = _loaded_training_record['low_policy_loss_append']
        # print('========== ({:s}) Training Record: '.format(self.ALG), self.training_record, ' epoch.', 
        #       ', Gradient steps: ', self.training_record*(self.NUM_SAMPLE//self.MINIBATCH_SIZE))

    # GP Training Record Loading Method
    def loadGPTrainingRecord(self, path=None):
        if path is None:
            _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
            # _loaded_training_record = torch.load(self.training_record_path+'_256', map_location=torch.device('cpu' if not CUDA else 'cuda'))
            # _loaded_training_record = torch.load(self.training_checkpoint_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        else:
            _loaded_training_record = torch.load(path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        _gp_state_dict = _loaded_training_record['gp_state_dict']

        # self.gp_model.sigma_p = torch.nn.Parameter(_gp_state_dict['sigma_p'], requires_grad=True)
        self.gp_model.sigma_n = torch.nn.Parameter(_gp_state_dict['sigma_n'], requires_grad=True)
        self.gp_model.ell = torch.nn.Parameter(_gp_state_dict['ell'], requires_grad=True)
        # self.gp_model.ell_1 = torch.nn.Parameter(_gp_state_dict['ell_1'], requires_grad=True)  # Angles
        # self.gp_model.ell_2 = torch.nn.Parameter(_gp_state_dict['ell_2'], requires_grad=True)  # Velocity
        # self.gp_model.ell_3 = torch.nn.Parameter(_gp_state_dict['ell_3'], requires_grad=True)  # Angular Velocity
        # self.gp_model.ell_4 = torch.nn.Parameter(_gp_state_dict['ell_4'], requires_grad=True)  # z-coordinate
        self.gp_model.x_train = _loaded_training_record['gp_x_train']
        self.gp_model.y_train = _loaded_training_record['gp_y_train']

        # self.gp_model.q_mean = torch.nn.Parameter(_gp_state_dict['q_mean'], requires_grad=True)
        # self.gp_model.q_var = torch.nn.Parameter(_gp_state_dict['q_var'], requires_grad=True)
        # # self.gp_model.q_var = torch.clip(_gp_state_dict['q_var'], min=0.1)
        # self.gp_model.z_train = torch.nn.Parameter(_loaded_training_record['gp_z_train'], requires_grad=True)

        # self.gp_model.optimizer.load_state_dict(_loaded_training_record['gp_optimizer'])

        # self.gp_model.cov_optimizer.load_state_dict(_loaded_training_record['gp_cov_optimizer'])
        # self.gp_model.ind_optimizer.load_state_dict(_loaded_training_record['gp_ind_optimizer'])

        # self.gp_model.K_mm = self.gp_model.rbfKernel(X_1=self.gp_model.z_train, X_2=self.gp_model.z_train, noise=True)
        # self.gp_model.L_mm = torch.cholesky(self.gp_model.K_mm, upper=False)
        if self.gp_model.kernel_fn == 'rbf':
            self.gp_model.K = self.gp_model.rbfKernel(X_1=self.gp_model.x_train, X_2=self.gp_model.x_train, noise=True)
        elif self.gp_model.kernel_fn == 'matern':
            self.gp_model.K = self.gp_model.maternKernel(X_1=self.gp_model.x_train, X_2=self.gp_model.x_train, noise=True)
        # self.gp_model.L_mm = torch.linalg.cholesky(self.gp_model.K_mm, upper=False)

    # Diffusion model's parameters initialisation Method
    def initialiseDiffusionParams(self, schedule='vp', beta_min=0.1, beta_max=10, num_step=50, dec_step=10):
        # Intialise Diffusion Model Parameters
        self.DIFFU_STEPS = num_step
        self.DEC_DIFFU_STEPS = dec_step
        self.BETA_MIN = beta_min
        self.BETA_MAX = beta_max
        self.MIN_DIFFU_SPACE = torch.tensor(-1., dtype=torch.float32)
        self.MAX_DIFFU_SPACE = torch.tensor(1., dtype=torch.float32)
        self.POS_DIM = 4
        self.beta = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha_bar = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.DIFFU_MEAN = torch.tensor(0., dtype=torch.float32)
        self.DIFFU_VAR = torch.tensor(1., dtype=torch.float32)
        self.DIFFU_STD = torch.sqrt(self.DIFFU_VAR)
        self.reverse_mean = 0.
        self.reverse_cov = 0.

        self.POS_EMB = my_utils.sinPositionEncoding(seq_len=self.DIFFU_STEPS, dim=self.POS_DIM)

        # ----- Derive the diffusion schedule
        # For the sake of simplicity, the order of the element in the array represents the forward process order.
        # For example, B_{fp} = [1, 2, 3, ..., N-1, N]  (the elements represent indices.)
        # In order to use for the reverse process, The elements have to be flip the other way around.
        # For example, B_{rp} = [N, N-1, N-2, ..., 2, 1] (the elements represent indices.)
        # Always remember that the diffusion process has the minimum time step is 1 and maximum at N. B = [1, 2, 3, ..., N]
        # ----- Variance Preserving Schedule
        if schedule == 'vp':
            # Derive the diffusion rate (noise schedule) 
            # which means, if it is the reverse, we have to reverse the order of the array.
            for i in range(self.DIFFU_STEPS):
                # Formular for the alpha: np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
                self.alpha[i] = np.exp((-self.BETA_MIN * 1/self.DIFFU_STEPS) - (0.5 * (self.BETA_MAX-self.BETA_MIN) * (2*(i+1)-1)/(np.square(self.DIFFU_STEPS))))
                self.beta[i] = 1 - self.alpha[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])

        # ----- Linear Schedule (Same as original DDPM)
        elif schedule == 'linear':        
            self.beta = np.linspace(start=self.BETA_MIN, stop=self.BETA_MAX, num=self.DIFFU_STEPS)
            for i in range(self.DIFFU_STEPS):
                self.alpha[i] = 1 - self.beta[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])
        
        # Convert them into tensor format.
        self.beta = torch.tensor(self.beta, dtype=torch.float32)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)
        self.POS_EMB = torch.tensor(self.POS_EMB, dtype=torch.float32)

    # Diffusion process (forward process) Method
    def forwardProcess(self, data, epsilon, step:int=None):
        alpha_bars = self.alpha_bar[step]
        x_t_p_1 = (torch.sqrt(alpha_bars) * data) + (torch.sqrt(1-alpha_bars) * epsilon)
        return x_t_p_1

    # Original Reverse Process Method
    def reverseProcess(self, inputs:list, size:int, guide:bool=False, target:bool=False):
        # Initialise noisy data (needed to be clipped).
        x_T = torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD)
        _diffu_steps = self.DIFFU_STEPS 
        _mu_r = []
        _var_r = []
        # Predict guidance params
        if guide:
            _mu_r, _var_r = self.predictGP(x_test=inputs)
            self.mu_r = _mu_r
            self.mu_r = torch.clip(input=_mu_r, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)
            self.var_r = torch.diag(_var_r).unsqueeze(1).expand(-1, self.ACTION_DIM)
            # print(self.var_r)
        
        # Start reverse processes
        for i in range(_diffu_steps): 
            # Define reverse position indices 
            # (Have to be inversed since the diffusion schedule was created in forward process's order.)
            rev_pos = _diffu_steps-i-1
            rev_pos_emb = self.POS_EMB[rev_pos].repeat(size, 1)
            # Predict the noise for the reverse process (e_{theta})
            if not target:
                epsilon_theta_t = self.epsilon_beh(torch.concat((inputs, x_T, rev_pos_emb), dim=1))
            else:
                epsilon_theta_t = self.epsilon_beh_tar(torch.concat((inputs, x_T, rev_pos_emb), dim=1))
            # Reverse process (Stochastic) (DDPM)
            x_t_m_1  = (x_T / torch.sqrt(self.alpha[rev_pos])) - \
                    (self.beta[rev_pos] * epsilon_theta_t / torch.sqrt(self.alpha[rev_pos]*(1-self.alpha_bar[rev_pos])))
            # Guided term
            if guide:
                if i != (self.DIFFU_STEPS-1):
                    # _cov_dp = self.beta[rev_pos] * torch.eye(size) 
                    # _cov_gp = _cov_dp
                    _cov_dp = self.beta[rev_pos]
                else:
                    # _cov_dp = 0. * torch.eye(size)
                    _cov_dp = 0.
                # _cov_gp = torch.clip(self.var_r, min=self.beta[rev_pos]) * torch.eye(size, dtype=torch.float32)
                # _inv_cov_gp = torch.linalg.inv(_cov_gp)
                _cov_gp = torch.clip(self.var_r, min=self.beta[rev_pos])
                # x_t_m_1 += -torch.matmul(torch.matmul(_cov_dp, _inv_cov_gp), x_t_m_1 - self.mu_r)
                # print(_cov_dp.shape, _cov_gp.shape, x_t_m_1.shape, self.mu_r.shape)
                x_t_m_1 += -((_cov_dp / _cov_gp) * (x_t_m_1 - self.mu_r))
                x_t_m_1 = torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

            if i == (_diffu_steps-1):
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * 0) 
            else:
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD))
            
            x_T = x_t_m_1  # Update the previous a (a_i) for the next iteration.

        return torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

    # Output prediction method
    def predict(self, state, size:int, guide:bool=True, target:bool=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).view(-1, self.STATE_DIM)
        return self.reverseProcess(state, size, guide, target)

    # Altered observation computation Method (Currently used)
    def getAlteredObservation_new(self, states, actions=None):
        with torch.no_grad():
            _sampling_size = self.num_action_candidates
            batch_size = len(states)
            
            # Vectorize: reshape and repeat all states at once [batch*sampling, state_dim]
            states_tensor = states.view(batch_size, 1, self.STATE_DIM) \
                                    .expand(batch_size, _sampling_size, self.STATE_DIM) \
                                    .reshape(batch_size * _sampling_size, self.STATE_DIM)

            # Single batched predict call instead of per-sample loop
            a_all = self.predict(state=states_tensor, size=batch_size * _sampling_size, guide=True)

            # Reshape to [batch, sampling, action_dim] for per-sample argmax
            a_all = a_all.view(batch_size, _sampling_size, self.ACTION_DIM)
            states_rep = states_tensor.view(batch_size, _sampling_size, self.STATE_DIM)

            # Compute Q-values in one batched call [batch, sampling, 1]
            q_inputs = torch.cat([states_rep, a_all], dim=2) \
                            .view(batch_size * _sampling_size, self.STATE_DIM + self.ACTION_DIM)
            q_values = self.q_1_tar(q_inputs).view(batch_size, _sampling_size)

            # action_tensor = actions.view(batch_size, 1, self.ACTION_DIM) \
            #                        .expand(batch_size, _sampling_size, self.ACTION_DIM) \
            #                        .reshape(batch_size * _sampling_size, self.ACTION_DIM)
            # actions_rep = action_tensor.view(batch_size, _sampling_size, self.ACTION_DIM)
            # # Norm a
            # _norm_a = 3 * torch.sum(torch.square(actions_rep - a_all), dim=2).view(batch_size, _sampling_size)
            # # Finalise Q-values
            # q_values -= _norm_a

            # Mean Q
            _mean_q = torch.mean(q_values, dim=1, keepdim=True)
            _std_q = torch.std(q_values, dim=1 , keepdim=True) + 1e-03

            # Argmax per sample, gather best actions
            best_indices = q_values.argmax(dim=1)  # [batch]
            best_indices_exp = best_indices.view(batch_size, 1, 1).expand(batch_size, 1, self.ACTION_DIM)
            y_hat = a_all.gather(1, best_indices_exp).squeeze(1)  # [batch, action_dim]

        return y_hat, _mean_q, _std_q

    # Altered observation computation Method (Old-I wrote it my own)
    def getAlteredObservation(self, states):
        # print('Start computing altered memories...')
        with torch.no_grad():
            _y_hat_list = []
            for m in range(len(states)):
                # Altered Observation (My approach)
                # For q-network and diffusion, still using the original observation (x) not the latent.
                _sampling_size = self.num_action_candidates
                _state_n_tensor = torch.reshape(states[m], [-1, self.STATE_DIM]).repeat(_sampling_size, 1)
                _beh_a_n_tensor = torch.reshape(self.gp_model.y_train_org[m], [-1, self.ACTION_DIM]).repeat(_sampling_size, 1)
                # a_i_m_1 = self.predict(state=_state_n_tensor, size=_sampling_size, guide=False)
                a_i_m_1 = self.predict(state=_state_n_tensor, size=_sampling_size, guide=True)

                # Get desired location (a, which max q)
                _q_values = self.q_1(torch.concat([_state_n_tensor, a_i_m_1], dim=1)) 

                # Get norm 
                _norm_a = 10 * torch.sum(torch.square(_beh_a_n_tensor - a_i_m_1), dim=1).reshape(_sampling_size, 1)

                # Get final Q-values
                _q_values -= _norm_a

                _max_q_value = torch.max(_q_values)
                _max_q_indices = torch.where(_q_values == _max_q_value)[0]
                if _max_q_indices.shape[0] > 1:
                    _max_q_index = torch.reshape(_max_q_indices[0], [-1, 1])
                else: 
                    _max_q_index = _max_q_indices[0]
                _a_max = torch.squeeze(a_i_m_1[_max_q_index])
                _y_hat_list.append(_a_max)
        return torch.stack(_y_hat_list, dim=0).view(-1, self.ACTION_DIM)

    # Training Method (for offliine training)
    def behTraining(self, total_epoch:int, eval:bool=False, ft:bool=False):
        # Diffusion Models Training Method (local)
        def _trainDiffusionBeh(inputs, y_true):
            # self.epsilon_optimizer.zero_grad()
            residual_noise = self.epsilon_beh(inputs)
            # Compute for the MSE.
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)
            return _diffu_loss, _diffu_loss.grad

        # Extract dataset
        buffer_size = self.NUM_SAMPLE if not ft else self.state_buffer.size()[0]
        state_buffer = self.state_buffer
        action_buffer = self.action_buffer

        # Initialise parameters
        _batch_size = self.MINIBATCH_SIZE
        _num_gradient_step = buffer_size//_batch_size

        # Set models to training mode.
        self.epsilon_beh.train()

        # ========================= Training Loop Start =========================
        while self.training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0

            # Get shuffle indices
            _sampling_indices = torch.randperm(buffer_size)

            # Start gradient steps loop
            for g in range(_num_gradient_step):
                # Get training batches
                batch_state_tensor = state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                # Prepare data for diffusion learning
                self.epsilon_optimizer.zero_grad()
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]  # Retrieve
                epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])

                # Original
                forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                diffu_loss, _ = _trainDiffusionBeh(inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1), y_true=epsilon_tensor)
                diffu_loss = torch.mean(diffu_loss)

                diffu_loss.backward()
                self.epsilon_optimizer.step()
                diffu_loss_accum += diffu_loss.tolist()

                self.gradient_step += 1

            # Increase training record after epoch finished.
            self.training_record += 1

            # Append loss for recording.
            self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)

            # Print the status.
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', self.gradient_step, 
                  ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))

            # Save the training_records
            self.recordSaving(path=self.training_record_path)
            if self.gradient_step == 1000000:
                self.recordSaving(path=self.training_record_path+'_pretrained')

        # ========================== Training Loop End ==========================

    # Training Method (for offliine training)
    def training(self, total_epoch:int, eval:bool=False, ft:bool=False):
        # Diffusion Models Training Method (local)
        def _trainDiffusionBeh(inputs, y_true):
            # self.epsilon_optimizer.zero_grad()
            residual_noise = self.epsilon_beh(inputs)
            # Compute for the MSE.
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)
            # # Mean of Loss 
            # _diffu_loss = torch.mean(_diffu_loss)
            # _diffu_loss.backward()
            # self.epsilon_optimizer.step()
            return _diffu_loss, _diffu_loss.grad

        # Target Networks Updating Method
        def _updateTargetNetworks():
            # Update the target networks
            high_q_tar_state_dict = self.high_q_tar.state_dict()
            high_q_state_dict = self.high_q.state_dict()
            for key in high_q_state_dict:
                high_q_tar_state_dict[key] = high_q_state_dict[key]*_TAU + high_q_tar_state_dict[key]*(1-_TAU)
            self.high_q_tar.load_state_dict(high_q_tar_state_dict)

            # Update the target networks
            low_q_tar_state_dict = self.low_q_tar.state_dict()
            low_q_state_dict = self.low_q.state_dict()
            for key2 in low_q_state_dict:
                low_q_tar_state_dict[key2] = low_q_state_dict[key2]*_TAU + low_q_tar_state_dict[key2]*(1-_TAU)
            self.low_q_tar.load_state_dict(low_q_tar_state_dict)
            
        # Declare Constants
        _GAMMA = 0.995  # Discount factor.
        _TAU = 0.005  # Tau for soft updating of target model's weights.

        # Extract dataset
        buffer_size = self.NUM_SAMPLE if not ft else self.state_buffer.size()[0]

        # Initialise parameters
        _batch_size = self.MINIBATCH_SIZE
        _num_gradient_step = buffer_size//_batch_size
        _gp_loss = 0

        # Set models to training mode.
        self.epsilon_beh.train()
        self.high_q.train()
        self.low_q.train()
        self.low_policy.train()
        self.gp_model.train()

        # ========================= Training Loop Start =========================
        while self.training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0
            high_q_loss_accum = 0
            low_q_loss_accum = 0
            low_policy_loss_accum = 0
            gp_mean_loss_accum = 0

            # Get shuffle indices
            _sampling_indices = torch.randperm(buffer_size)

            # Start gradient steps loop
            for g in range(_num_gradient_step):
                # Get training batches
                batch_state_tensor = self.state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_high_state_tensor = self.high_state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_high_next_state_tensor = self.high_next_state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = self.action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_reward_tensor = self.reward_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_next_state_tensor = self.next_state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                # --------------------------------- Diffusion Policy Learning ---------------------------------
                # Prepare data for diffusion learning
                self.epsilon_optimizer.zero_grad()
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]  # Retrieve
                epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                # Original
                forward_goal_tensor = self.forwardProcess(data=batch_high_next_state_tensor, epsilon=epsilon_tensor, step=rand_t)
                diffu_loss, _ = _trainDiffusionBeh(inputs=torch.concat([batch_high_state_tensor, forward_goal_tensor, encode_t_tensor], dim=1), y_true=epsilon_tensor)
                diffu_loss = torch.mean(diffu_loss)
                diffu_loss.backward(retain_graph=True)
                self.epsilon_optimizer.step()
                diffu_loss_accum += diffu_loss.tolist()

                # --------------------------------- Temporal Difference Learning (Q-function) ---------------------------------
                # Prepare data for q learning
                self.high_q_optimizer.zero_grad()
                _batch_next_goal = self.predict(state=batch_high_next_state_tensor, size=_batch_size, guide=True, target=False).view(-1, self.GOAL_DIM)
                y_true = batch_reward_tensor + _GAMMA * self.high_q_tar(torch.concat([batch_high_next_state_tensor, _batch_next_goal], dim=1))
                high_q_loss = torch.mean(torch.square(y_true - self.high_q(torch.concat([batch_high_state_tensor, batch_next_state_tensor[:, :2]], dim=1)))) 
                high_q_loss_accum += high_q_loss.tolist()
                high_q_loss.backward(retain_graph=True)
                self.high_q_optimizer.step()

                # --------------------------------- Low-level Q Learning --------------------------------- 
                self.low_q_optimizer.zero_grad()
                _batch_next_action = self.low_policy_tar(torch.concat([batch_state_tensor, _batch_next_goal], dim=1))
                _batch_euc_score = torch.sqrt(torch.sum(torch.square(batch_high_next_state_tensor[:, :2] - _batch_next_goal), dim=1))
                _low_q_target = _batch_euc_score + _GAMMA * self.low_q_tar(torch.concat([batch_next_state_tensor, _batch_next_action]))
                low_q_loss = torch.mean(torch.square(_low_q_target - self.low_q_tar(torch.concat([batch_state_tensor, batch_action_tensor], dim=1))))
                low_q_loss_accum += low_q_loss.tolist()
                low_q_loss.backward(retain_graph=True)
                self.low_q_optimizer.step()
                
                # --------------------------------- Low-level Policy Learning --------------------------------- 
                self.low_policy_optimizer.zero_grad()
                a_pred = self.low_policy(torch.concat([batch_state_tensor, batch_next_state_tensor[:, :2]], dim=1))
                low_policy_loss = -self.low_q(torch.concat([batch_state_tensor, a_pred], dim=1)) + (10 * torch.mean(torch.square(a_pred - batch_action_tensor)))
                low_policy_loss_accum += low_policy_loss.tolist()
                low_policy_loss.backward(retain_graph=True)
                self.low_policy_optimizer.step()

                # Update target networks
                _updateTargetNetworks()

                self.gradient_step += 1

                # Train GP and save checkpoint every 10k step
                if (self.gradient_step % 10000) == 0:
                    # Save checkpoint
                    self.recordSaving(path=self.training_record_path+'_{:d}'.format(self.gradient_step))

            # Increase training record after epoch finished.
            self.training_record += 1

            # Train the gp
            if (self.training_record % 1) == 0:
                _gp_loss = self.gp_model.myTraining(total_epoch=10 if self.gp_model_type == 'sparse' else 10, ft=False)

            # Update Altered observation for every ... epoch.
            if self.training_record % 5 == 0:
                # self.gp_model.y_train = self.getAlteredObservation(self.gp_model.x_train)
                self.gp_model.y_train = self.getAlteredObservation(self.gp_model.x_train_org)
                # _gp_loss = self.gp_model.myTraining(total_epoch=100, ft=False)


            # Append loss for recording.
            self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)
            self.high_q_loss_append.append(high_q_loss_accum/_num_gradient_step)
            self.low_q_loss_append.append(low_q_loss_accum/_num_gradient_step)
            self.low_policy_loss_append.append(low_policy_loss_accum/_num_gradient_step)

            # Print the status.
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', self.gradient_step, 
                  ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', HighQ_loss: ', round(high_q_loss_accum/_num_gradient_step, 4),
                  ', LowQ_loss: ', round(low_q_loss_accum/_num_gradient_step, 4),
                  ', LowPolicy_loss: ', round(low_policy_loss_accum/_num_gradient_step, 4),
                  ', GP_loss: ', round(_gp_loss, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))

            # Save the training_records
            self.recordSaving(path=self.training_record_path)

            # Check for the breaking for evaluation.
            if eval and self.training_record % 5 == 0:
                # self.epsilon_beh.eval()
                # self.q_1.eval()
                # self.q_2.eval()
                break

        # ========================== Training Loop End ==========================

    # ==================================== GP model sections start ====================================
    # Prediction Method (mean, var)
    def predictGP(self, x_test):
        _mean, _var = self.gp_model.predict(X_s=x_test)
        return _mean, _var
    # ===================================== GP model sections end =====================================
    # Training Record Saving Method
    def recordSaving(self, path:str):
        torch.save({'training_record': self.training_record,
                    'gradient_step': self.gradient_step,
                    'epsilon_beh': self.epsilon_beh.state_dict(), 
                    'high_q': self.high_q.state_dict(),
                    'low_q': self.low_q.state_dict(),
                    'low_policy': self.low_policy.state_dict(),
                    'epsilon_beh_tar': self.epsilon_beh_tar.state_dict(),
                    'high_q_tar': self.high_q_tar.state_dict(),
                    'low_q_tar': self.low_q_tar.state_dict(), 
                    'low_policy_tar': self.low_policy_tar.state_dict(),
                    'epsilon_beh_loss_append': self.epsilon_beh_loss_append, 
                    'high_q_loss_append': self.high_q_loss_append,
                    'low_q_loss_append': self.low_q_loss_append, 
                    'low_policy_loss_append': self.low_policy_loss_append,  
                    'epsilon_beh_optimizer': self.epsilon_optimizer.state_dict(), 
                    'high_q_optimizer': self.high_q_optimizer.state_dict(),
                    'low_q_optimizer': self.low_q_optimizer.state_dict(), 
                    'low_policy_optimizer': self.low_policy_optimizer.state_dict(), 
                    'gp_state_dict': self.gp_model.state_dict(), 
                    'gp_loss_append': self.gp_model.mll_append,
                    'gp_x_train': self.gp_model.x_train, 
                    'gp_y_train': self.gp_model.y_train, 
                    'gp_optimizer': self.gp_model.optimizer.state_dict(),
                    }, path)