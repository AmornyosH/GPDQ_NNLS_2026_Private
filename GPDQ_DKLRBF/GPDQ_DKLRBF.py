'''
Gaussian Process Diffusion Q-learning — Deep Kernel Learning variant
Version: v0
Revision: 1
Remark: FeatureNet(full state) → emb_dim → GP kernel (jointly trained via MLL)
'''
# ============================ Pytorch Related ============================
import torch
CUDA = torch.cuda.is_available()
if CUDA:
    print('CUDA is activated.')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# ============================ Pytorch Related ============================

# ================================== Others ==================================
from GPDQ_DKLRBF import my_dkl_gp
from utility import my_utils, my_NN
from time import time
import copy
import numpy as np
import os

class GaussianProcessDiffusionQlearning:
    def __init__(self, params_dict:dict, dataset:dict, ft:bool=False):
        self.ALG = 'GPDQ_DKLRBF'
        self.ENV_CONFIG = params_dict['environment']
        self.STATE_DIM = int(params_dict['state_dim'])
        self.ACTION_DIM = int(params_dict['action_dim'])
        self.cuda = CUDA
        self.num_action_candidates = 10
        self.NUM_SAMPLE = dataset['observations'].shape[0]
        # Initialise replay buffers
        self.state_buffer = torch.tensor(dataset['observations'], dtype=torch.float32)
        self.next_state_buffer = torch.tensor(dataset['next_observations'], dtype=torch.float32)
        self.action_buffer = torch.tensor(dataset['actions'], dtype=torch.float32)
        self.reward_buffer = torch.tensor(dataset['rewards'], dtype=torch.float32).view(-1, 1)
        self.MINIBATCH_SIZE = 256

        self.action_mean = torch.mean(self.action_buffer)
        self.action_std = torch.std(self.action_buffer) + 1e-03
        self.action_dist = torch.distributions.Normal(loc=self.action_mean, scale=self.action_std)

        # Improvement hyperparams
        self.lambda_gp = float(params_dict.get('lambda_gp', 1.0))
        self.eta = float(params_dict.get('eta', 1.0))
        self.ema_decay = float(params_dict.get('ema_decay', 0.995))
        self.gamma = float(params_dict.get('gamma', 0.99 if 'antmaze' in params_dict['environment'] else 0.995))

        # Initialise Diffusion Model's Parameters
        self.initialiseDiffusionParams(schedule='vp', beta_min=1, beta_max=10, num_step=params_dict['diffusion_step'], dec_step=params_dict['diffusion_step'])

        # Initialise Paths
        self.training_record_path = '/home/amornyos/PhD/Packages/resources/training_records/{}/{}/{}_{}_training_records'.format(self.ALG, self.ENV_CONFIG, self.ALG, self.ENV_CONFIG)
        self.testing_record_path = '/home/amornyos/PhD/Packages/resources/test_results/{}/{}_{}_test_results'.format(self.ALG, self.ENV_CONFIG, self.ALG, self.ENV_CONFIG)

        # Initialise neural networks
        self.EPSILON_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM + self.POS_DIM
        self.EPSILON_BEH_INPUT_DIM = self.ACTION_DIM + self.POS_DIM
        self.Q_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM
        self.V_INPUT_DIM = self.STATE_DIM
        self.epsilon_beh = my_NN.MLP(input_dim=self.EPSILON_INPUT_DIM, output_dim=self.ACTION_DIM)
        self.q_1 = my_NN.MLP_Relu(input_dim=self.Q_INPUT_DIM, output_dim=1)
        self.q_2 = my_NN.MLP_Relu(input_dim=self.Q_INPUT_DIM, output_dim=1)
        self.q_1_tar = copy.deepcopy(self.q_1)
        self.q_2_tar = copy.deepcopy(self.q_2)
        self.epsilon_beh_tar = copy.deepcopy(self.epsilon_beh)
        self.epsilon_beh_ema = copy.deepcopy(self.epsilon_beh)

        _q_lr = 3e-04 if 'antmaze' not in self.ENV_CONFIG else 3e-03
        self.epsilon_optimizer = torch.optim.Adam(self.epsilon_beh.parameters(), lr=3e-04)
        self.q_1_optimizer = torch.optim.Adam(self.q_1.parameters(), lr=_q_lr)
        self.q_2_optimizer = torch.optim.Adam(self.q_2.parameters(), lr=_q_lr)
        if CUDA:
            self.epsilon_beh.cuda()
            self.q_1.cuda()
            self.q_2.cuda()

        # ======================================= Create DKL GP model =======================================
        self.gp_model_type = 'dkl'
        best_dataset = my_utils.mazeTrajExtraction(dataset=dataset, max_episode_steps=1000, top_k=100)

        self.gp_model = my_dkl_gp.myDKLGP(
            params_dict=params_dict, dataset=best_dataset, cuda=CUDA, parent_alg=self.ALG)

        # ======================================= Create/Load Training Record =======================================
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'

        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new record and models!'.format(self.ALG))
            self.training_record = 0
            self.gradient_step = 0
            self.beh_training_record = 0
            self.best_norm_reward_training = 0
            self.norm_reward_training_append = []
            self.epsilon_beh_loss_append = []
            self.q_1_loss_append = []

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            self.loadTrainingRecord()
            self.loadGPTrainingRecord()
            self.epsilon_beh.eval()
            self.q_1.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (load existing one) or "n" for no (create new)).'.format(self.ALG))
            exit()

    # Training Record Loading Method
    def loadTrainingRecord(self, path:str=None):
        if path is None:
            _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        else:
            _loaded_training_record = torch.load(path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        self.training_record = _loaded_training_record['training_record']
        self.gradient_step = _loaded_training_record['gradient_step']
        self.epsilon_beh.load_state_dict(_loaded_training_record['epsilon_beh'])
        self.epsilon_beh_tar.load_state_dict(_loaded_training_record['epsilon_beh_tar'])
        self.q_1.load_state_dict(_loaded_training_record['q_1'])
        self.q_1_tar.load_state_dict(_loaded_training_record['q_1_tar'])
        if 'q_2' in _loaded_training_record:
            self.q_2.load_state_dict(_loaded_training_record['q_2'])
            self.q_2_tar.load_state_dict(_loaded_training_record['q_2_tar'])
            self.q_2_optimizer.load_state_dict(_loaded_training_record['q_2_optimizer'])
        else:
            self.q_2.load_state_dict(_loaded_training_record['q_1'])
            self.q_2_tar.load_state_dict(_loaded_training_record['q_1_tar'])
        if 'epsilon_beh_ema' in _loaded_training_record:
            self.epsilon_beh_ema.load_state_dict(_loaded_training_record['epsilon_beh_ema'])
        else:
            self.epsilon_beh_ema.load_state_dict(_loaded_training_record['epsilon_beh'])
        self.epsilon_optimizer.load_state_dict(_loaded_training_record['epsilon_beh_optimizer'])
        self.q_1_optimizer.load_state_dict(_loaded_training_record['q_1_optimizer'])
        self.epsilon_beh_loss_append = _loaded_training_record['epsilon_beh_loss_append']
        self.q_1_loss_append = _loaded_training_record['q_1_loss_append']

    # GP Training Record Loading Method (DKL version: load_state_dict restores FeatureNet + GP params together)
    def loadGPTrainingRecord(self, path=None):
        if path is None:
            _loaded_training_record = torch.load(self.training_record_path, map_location=torch.device('cpu' if not CUDA else 'cuda'))
        else:
            _loaded_training_record = torch.load(path, map_location=torch.device('cpu' if not CUDA else 'cuda'))

        # Restore FeatureNet weights + all GP hyperparameters in one call
        self.gp_model.load_state_dict(_loaded_training_record['gp_state_dict'])
        self.gp_model.x_train = _loaded_training_record['gp_x_train']
        self.gp_model.y_train = _loaded_training_record['gp_y_train']
        # Restore org copies (positionally aligned with x_train; used in getAlteredObservation)
        if 'gp_x_train_org' in _loaded_training_record:
            self.gp_model.x_train_org = _loaded_training_record['gp_x_train_org']
            self.gp_model.y_train_org = _loaded_training_record['gp_y_train_org']
        # Recompute cached embedding + kernel with the restored feature net
        with torch.no_grad():
            self.gp_model.z_train = self.gp_model.feature_net(self.gp_model.x_train)
        self.gp_model.K = self.gp_model.kernel(
            Z_1=self.gp_model.z_train, Z_2=self.gp_model.z_train, noise=True)
        self.gp_model._cache_stale = False

    # Diffusion model's parameters initialisation Method
    def initialiseDiffusionParams(self, schedule='vp', beta_min=0.1, beta_max=10, num_step=50, dec_step=10):
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

        if schedule == 'vp':
            for i in range(self.DIFFU_STEPS):
                self.alpha[i] = np.exp((-self.BETA_MIN * 1/self.DIFFU_STEPS) - (0.5 * (self.BETA_MAX-self.BETA_MIN) * (2*(i+1)-1)/(np.square(self.DIFFU_STEPS))))
                self.beta[i] = 1 - self.alpha[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])
        elif schedule == 'linear':
            self.beta = np.linspace(start=self.BETA_MIN, stop=self.BETA_MAX, num=self.DIFFU_STEPS)
            for i in range(self.DIFFU_STEPS):
                self.alpha[i] = 1 - self.beta[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])

        self.beta = torch.tensor(self.beta, dtype=torch.float32)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)
        self.POS_EMB = torch.tensor(self.POS_EMB, dtype=torch.float32)

    # Best Trajectory Extraction Method
    def multiBestTrajExtraction(self, dataset: dict, max_episode_steps=1000, top_k=10):
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']
        num_samples = len(observations)

        trajectories = []
        current_trajectory = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}
        step_count = 0

        for i in range(num_samples):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            if terminals[i] or step_count >= max_episode_steps or i == num_samples - 1:
                if step_count > 1:
                    trajectories.append({
                        'observations': np.array(current_trajectory['observations']),
                        'actions': np.array(current_trajectory['actions']),
                        'rewards': np.array(current_trajectory['rewards']),
                        'next_observations': np.array(current_trajectory['next_observations'])
                    })
                current_trajectory = {'observations': [], 'actions': [], 'rewards': [], 'next_observations': []}
                step_count = 0

        cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]
        top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]

        all_obs, all_actions, all_rewards, all_next_obs = [], [], [], []
        total_length = 0
        for idx in top_k_indices:
            traj = trajectories[idx]
            all_obs.append(traj['observations'])
            all_actions.append(traj['actions'])
            all_rewards.append(traj['rewards'])
            all_next_obs.append(traj['next_observations'])
            total_length += len(traj['observations'])

        return {
            'arr_0': total_length,
            'observations': np.vstack(all_obs),
            'actions': np.vstack(all_actions),
            'rewards': np.hstack(all_rewards).reshape(-1, 1),
            'next_observations': np.vstack(all_next_obs),
        }

    # Diffusion forward process
    def forwardProcess(self, data, epsilon, step:int=None):
        alpha_bars = self.alpha_bar[step]
        return (torch.sqrt(alpha_bars) * data) + (torch.sqrt(1-alpha_bars) * epsilon)

    # Reverse (denoising) process
    def reverseProcess(self, inputs:list, size:int, guide:bool=False, target:bool=False, use_ema:bool=False):
        x_T = torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD)
        _diffu_steps = self.DIFFU_STEPS
        if guide:
            _mu_r, _var_r = self.predictGP(x_test=inputs)
            self.mu_r = torch.clip(input=_mu_r, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)
            self.var_r = torch.diag(_var_r).unsqueeze(1).expand(-1, self.ACTION_DIM)

        for i in range(_diffu_steps):
            rev_pos = _diffu_steps-i-1
            rev_pos_emb = self.POS_EMB[rev_pos].repeat(size, 1)
            _denoiser_input = torch.concat((inputs, x_T, rev_pos_emb), dim=1)
            if target:
                epsilon_theta_t = self.epsilon_beh_tar(_denoiser_input)
            elif use_ema:
                epsilon_theta_t = self.epsilon_beh_ema(_denoiser_input)
            else:
                epsilon_theta_t = self.epsilon_beh(_denoiser_input)

            x_t_m_1 = (x_T / torch.sqrt(self.alpha[rev_pos])) - \
                       (self.beta[rev_pos] * epsilon_theta_t / torch.sqrt(self.alpha[rev_pos]*(1-self.alpha_bar[rev_pos])))

            if guide:
                _cov_dp = self.beta[rev_pos] if i != (self.DIFFU_STEPS-1) else 0.
                _cov_gp = torch.clip(self.var_r, min=self.beta[rev_pos])
                x_t_m_1 += -self.lambda_gp * ((_cov_dp / _cov_gp) * (x_t_m_1 - self.mu_r))
                x_t_m_1 = torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

            if i == (_diffu_steps-1):
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * 0)
            else:
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD))

            x_T = x_t_m_1

        return torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

    def predict(self, state, size:int, guide:bool=True, target:bool=False, use_ema:bool=False):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32).view(-1, self.STATE_DIM)
        return self.reverseProcess(state, size, guide, target, use_ema)

    def getAlteredObservation_new(self, states, actions=None):
        with torch.no_grad():
            _sampling_size = self.num_action_candidates
            batch_size = len(states)

            states_tensor = states.view(batch_size, 1, self.STATE_DIM) \
                                   .expand(batch_size, _sampling_size, self.STATE_DIM) \
                                   .reshape(batch_size * _sampling_size, self.STATE_DIM)
            a_all = self.predict(state=states_tensor, size=batch_size * _sampling_size, guide=True)
            a_all = a_all.view(batch_size, _sampling_size, self.ACTION_DIM)
            states_rep = states_tensor.view(batch_size, _sampling_size, self.STATE_DIM)

            q_inputs = torch.cat([states_rep, a_all], dim=2) \
                            .view(batch_size * _sampling_size, self.STATE_DIM + self.ACTION_DIM)
            q_values = self.q_1_tar(q_inputs).view(batch_size, _sampling_size)

            _mean_q = torch.mean(q_values, dim=1, keepdim=True)
            _std_q  = torch.std(q_values, dim=1, keepdim=True) + 1e-03

            best_indices = q_values.argmax(dim=1)
            best_indices_exp = best_indices.view(batch_size, 1, 1).expand(batch_size, 1, self.ACTION_DIM)
            y_hat = a_all.gather(1, best_indices_exp).squeeze(1)

        return y_hat, _mean_q, _std_q

    def getAlteredObservation(self, states):
        with torch.no_grad():
            _y_hat_list = []
            for m in range(len(states)):
                _sampling_size = self.num_action_candidates
                _state_n_tensor = torch.reshape(states[m], [-1, self.STATE_DIM]).repeat(_sampling_size, 1)
                _beh_a_n_tensor = torch.reshape(self.gp_model.y_train_org[m], [-1, self.ACTION_DIM]).repeat(_sampling_size, 1)
                a_i_m_1 = self.predict(state=_state_n_tensor, size=_sampling_size, guide=True)
                _q_values = self.q_1(torch.concat([_state_n_tensor, a_i_m_1], dim=1))
                _norm_a = 10 * torch.sum(torch.square(_beh_a_n_tensor - a_i_m_1), dim=1).reshape(_sampling_size, 1)
                _q_values -= _norm_a
                _max_q_value = torch.max(_q_values)
                _max_q_indices = torch.where(_q_values == _max_q_value)[0]
                _max_q_index = _max_q_indices[0] if _max_q_indices.shape[0] == 1 else torch.reshape(_max_q_indices[0], [-1, 1])
                _a_max = torch.squeeze(a_i_m_1[_max_q_index])
                _y_hat_list.append(_a_max)
        return torch.stack(_y_hat_list, dim=0).view(-1, self.ACTION_DIM)

    def behTraining(self, total_epoch:int, eval:bool=False, ft:bool=False):
        def _trainDiffusionBeh(inputs, y_true):
            residual_noise = self.epsilon_beh(inputs)
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)
            return _diffu_loss, _diffu_loss.grad

        buffer_size = self.NUM_SAMPLE if not ft else self.state_buffer.size()[0]
        state_buffer = self.state_buffer
        action_buffer = self.action_buffer
        _batch_size = self.MINIBATCH_SIZE
        _num_gradient_step = buffer_size//_batch_size
        self.epsilon_beh.train()

        while self.training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0
            _sampling_indices = torch.randperm(buffer_size)

            for g in range(_num_gradient_step):
                batch_state_tensor  = state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                self.epsilon_optimizer.zero_grad()
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]
                epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                diffu_loss, _ = _trainDiffusionBeh(
                    inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1),
                    y_true=epsilon_tensor)
                diffu_loss = torch.mean(diffu_loss)
                diffu_loss.backward()
                self.epsilon_optimizer.step()
                diffu_loss_accum += diffu_loss.tolist()
                self.gradient_step += 1

            self.training_record += 1
            self.epsilon_beh_loss_append.append(diffu_loss_accum/_num_gradient_step)
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', self.gradient_step,
                  ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))
            self.recordSaving(path=self.training_record_path)

    def training(self, total_epoch:int, eval:bool=False, ft:bool=False):
        def _getExpectedCumulativeReturn(next_sa_inputs):
            _q1_t = self.q_1_tar(next_sa_inputs)
            _q2_t = self.q_2_tar(next_sa_inputs)
            return batch_reward_tensor + (_GAMMA * torch.min(_q1_t, _q2_t))

        def _trainDiffusionBeh(inputs, y_true):
            residual_noise = self.epsilon_beh(inputs)
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)
            return _diffu_loss, _diffu_loss.grad

        def _updateTargetNetworks():
            for src, tgt in [(self.q_1, self.q_1_tar), (self.q_2, self.q_2_tar)]:
                tgt_sd = tgt.state_dict()
                src_sd = src.state_dict()
                for key in src_sd:
                    tgt_sd[key] = src_sd[key] * _TAU + tgt_sd[key] * (1 - _TAU)
                tgt.load_state_dict(tgt_sd)
            for p_ema, p in zip(self.epsilon_beh_ema.parameters(), self.epsilon_beh.parameters()):
                p_ema.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)

        _GAMMA = self.gamma
        _TAU   = 0.005

        buffer_size       = self.NUM_SAMPLE if not ft else self.state_buffer.size()[0]
        state_buffer      = self.state_buffer
        action_buffer     = self.action_buffer
        reward_buffer     = self.reward_buffer
        next_state_buffer = self.next_state_buffer
        _batch_size       = self.MINIBATCH_SIZE
        _num_gradient_step = buffer_size // _batch_size
        _gp_loss = 0

        self.epsilon_beh.train()
        self.q_1.train()
        self.q_2.train()
        self.gp_model.train()

        while self.training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0
            q_1_loss_accum   = 0
            _sampling_indices = torch.randperm(buffer_size)

            for g in range(_num_gradient_step):
                batch_state_tensor      = state_buffer[_sampling_indices[g*_batch_size : _batch_size+g*_batch_size]]
                batch_action_tensor     = action_buffer[_sampling_indices[g*_batch_size : _batch_size+g*_batch_size]]
                batch_reward_tensor     = reward_buffer[_sampling_indices[g*_batch_size : _batch_size+g*_batch_size]]
                batch_next_state_tensor = next_state_buffer[_sampling_indices[g*_batch_size : _batch_size+g*_batch_size]]

                # ── Diffusion BC loss ──
                self.epsilon_optimizer.zero_grad()
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]
                epsilon_tensor  = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                diffu_loss, _ = _trainDiffusionBeh(
                    inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1),
                    y_true=epsilon_tensor)
                diffu_loss = torch.mean(diffu_loss)
                diffu_loss.backward()
                self.epsilon_optimizer.step()
                diffu_loss_accum += diffu_loss.tolist()

                # ── Clipped double-Q TD ──
                with torch.no_grad():
                    _batch_next_action = self.predict(state=batch_next_state_tensor, size=_batch_size, guide=True, target=False).view(-1, self.ACTION_DIM)
                    y_true_1 = _getExpectedCumulativeReturn(
                        torch.concat([batch_next_state_tensor, _batch_next_action], dim=1))

                _sa_inputs = torch.concat([batch_state_tensor, batch_action_tensor], dim=1)

                self.q_1_optimizer.zero_grad()
                q_1_loss = torch.mean(torch.square(y_true_1 - self.q_1(_sa_inputs)))
                q_1_loss_accum += q_1_loss.tolist()
                q_1_loss.backward()
                self.q_1_optimizer.step()

                self.q_2_optimizer.zero_grad()
                q_2_loss = torch.mean(torch.square(y_true_1 - self.q_2(_sa_inputs)))
                q_2_loss.backward()
                self.q_2_optimizer.step()

                _updateTargetNetworks()
                self.gradient_step += 1

                if (self.gradient_step % 10000) == 0:
                    self.recordSaving(path=self.training_record_path + '_{:d}'.format(self.gradient_step))

            self.training_record += 1

            if (self.training_record % 1) == 0:
                _gp_loss = self.gp_model.myTraining(total_epoch=10, ft=False)

            if self.training_record % 5 == 0:
                self.gp_model.y_train = self.getAlteredObservation(self.gp_model.x_train_org)

            self.epsilon_beh_loss_append.append(diffu_loss_accum / _num_gradient_step)
            self.q_1_loss_append.append(q_1_loss_accum / _num_gradient_step)

            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', self.gradient_step,
                  ', Diffu_loss: ', round(diffu_loss_accum / _num_gradient_step, 4),
                  ', Q1_loss: ', round(q_1_loss_accum / _num_gradient_step, 4),
                  ', GP_loss: ', round(_gp_loss, 4),
                  ', Time/Epoch: ', round(time() - start_time, 4))

            self.recordSaving(path=self.training_record_path)

            if eval and self.training_record % 5 == 0:
                break

    # ── GP prediction ─────────────────────────────────────────────────────────
    def predictGP(self, x_test):
        _mean, _var = self.gp_model.predict(X_s=x_test)
        return _mean, _var

    # ── Checkpoint saving ─────────────────────────────────────────────────────
    def recordSaving(self, path:str):
        torch.save({
            'training_record': self.training_record,
            'gradient_step':   self.gradient_step,
            'epsilon_beh':         self.epsilon_beh.state_dict(),
            'epsilon_beh_tar':     self.epsilon_beh_tar.state_dict(),
            'epsilon_beh_ema':     self.epsilon_beh_ema.state_dict(),
            'q_1':             self.q_1.state_dict(),
            'q_1_tar':         self.q_1_tar.state_dict(),
            'q_2':             self.q_2.state_dict(),
            'q_2_tar':         self.q_2_tar.state_dict(),
            'epsilon_beh_loss_append': self.epsilon_beh_loss_append,
            'q_1_loss_append':         self.q_1_loss_append,
            'epsilon_beh_optimizer':   self.epsilon_optimizer.state_dict(),
            'q_1_optimizer':           self.q_1_optimizer.state_dict(),
            'q_2_optimizer':           self.q_2_optimizer.state_dict(),
            # gp_state_dict includes FeatureNet + log_sigma_p + sigma_n + ell
            'gp_state_dict':   self.gp_model.state_dict(),
            'gp_loss_append':  self.gp_model.mll_append,
            'gp_x_train':      self.gp_model.x_train,
            'gp_y_train':      self.gp_model.y_train,
            # org copies must survive reload so getAlteredObservation targets stay aligned
            'gp_x_train_org':  self.gp_model.x_train_org,
            'gp_y_train_org':  self.gp_model.y_train_org,
            'gp_optimizer':    self.gp_model.optimizer.state_dict(),
        }, path)
