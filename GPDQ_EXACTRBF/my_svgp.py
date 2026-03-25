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
        self.kernel_fn = 'matern'
        self.param_dict = params_dict
        self.env = params_dict['environment']
        # self.num_sample = 10000  # Fixed to 1000.
        # self.num_sample = len(dataset['arr_0'])
        # self.terminals = dataset['terminals']
        # self.num_sample = params_dict['gp_num_sample']
        # self.num_inducing = 1000
        self.num_inducing = params_dict['gp_num_inducing']
        self.x_dim = int(params_dict['state_dim'])
        self.y_dim = int(params_dict['action_dim'])

        self.training_record_path = '{}/training_records/{}/{}_{}_{}_training_records.zip'.format(self.parent_alg, self.env, self.parent_alg, self.env, self.alg)
        # self.training_record_path = '{}/training_records/walker2d_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        # self.training_record_path = '{}/training_records/hopper_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        # self.training_record_path = '{}/training_records/halfcheetah_1st_(rbf-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        # self.training_record_path = '{}/training_records/halfcheetah_2nd_(svm-1000-3hidden)/{}_{}_{}_training_records_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)

        # self.training_checkpoint_path = '{}/training_records/{}_{}_{}_checkpoint_{:d}_{:d}.zip'.format(self.parent_alg, self.parent_alg, self.env, self.alg, self.num_sample, self.num_inducing)
        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
        # if os.path.isfile(self.training_checkpoint_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.alg))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.alg))
        else:
            _ans_1 = 'n'
        _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new GP record and models!'.format(self.alg))
            self.mll_append = []  # Loss storage
            self.training_record = 0  # Training record counter
            # Initialise training data
            _start = 0
            # self.x_train = torch.tensor(dataset['observations'][0+_start:_start+self.num_sample], dtype=torch.float32)
            # self.y_train = torch.tensor(dataset['actions'][0+_start:_start+self.num_sample], dtype=torch.float32)

            _training_data = self.getTrainingData(dataset=dataset)
            self.x_train = torch.tensor(_training_data['observations'], dtype=torch.float32)
            self.y_train = torch.tensor(_training_data['actions'], dtype=torch.float32)

            # Intialise inducing points (Let's fix it first.)
            # _best_z, _best_z_indexes = self.getInitInducingPoints(x_train=self.x_train[0:1000])
            # _sub_z, _sub_z_indexes = self.getInitInducingPoints(x_train=self.x_train[60000:61000])
            # self.z_train = torch.concatenate([_best_z, _sub_z], dim=0)
            # self.indexes = np.concatenate([_best_z_indexes, _sub_z_indexes], axis=0)
            # self.z_train, self.indexes = self.getInitInducingPoints(x_train=self.x_train)
            
            _ind_training_data = self.getZ(dataset=dataset)
            self.z_train = torch.tensor(_ind_training_data['observations'], dtype=torch.float32)
            self.q_mean = torch.tensor(_ind_training_data['actions'], dtype=torch.float32)

            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.ones(size=[1, self.x_dim], dtype=torch.float32), requires_grad=True)

            # self.q_mean = torch.nn.Parameter(self.y_train[self.indexes], requires_grad=True)
            self.q_mean = torch.nn.Parameter(self.q_mean, requires_grad=True)
            self.q_var = torch.nn.Parameter(torch.ones([self.num_inducing, self.num_inducing]), requires_grad=True)  # Single covariance
            self.z_train = torch.nn.Parameter(self.z_train, requires_grad=True)

            # self.num_mixtures = params_dict['state_dim']
            # self.mean_q = torch.nn.Parameter(torch.zeros(self.num_mixtures, self.x_dim), requires_grad=True)
            # self.weight_q = torch.nn.Parameter(torch.ones(self.num_mixtures), requires_grad=True)
            # self.v_q = torch.nn.Parameter(torch.ones(self.num_mixtures, self.x_dim), requires_grad=True)

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.alg))
            # Load training records
            # _training_records = torch.load(self.training_record_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            # # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            # self.mll_append = _training_records['loss_append']  # Loss append
            # self.training_record = _training_records['training_record']  # Training record counter
            # _state_dict = _training_records['state_dict']
            # self.x_train = _training_records['x_train']
            # self.y_train = _training_records['y_train']
            # self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)
            # self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            # self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)
            # self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
            # self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
            self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
            self.loadTrainingRecord(path=self.training_record_path, cuda=cuda)
            # self.loadTrainingRecord(path=self.training_checkpoint_path, cuda=cuda)
            self.eval()
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).'.format(self.alg))
            exit()

        # Summarize parameters
        self.num_sample = len(self.x_train)
        print(self.num_sample)
        self.sigma_p = torch.tensor(1.0, dtype=torch.float32)  # Fixed signal variance to 1.00^2
        # print('========== ({:s}) Summarize GP parameters.'.format(self.alg))
        # print('sigma_p: ', self.sigma_p.tolist(), ', sigma_n: ', self.sigma_n.tolist(), ', ell: ', self.ell.tolist())
        # print('q_mean: ', self.q_mean.tolist(), ', q_var: ', torch.mean(torch.diag(self.q_var)).tolist(), ', z_size: ', self.z_train.size())

        # Initialised training kernels (K, K_inv) (For evaluation).
        # self.K_mm = self.rbfKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.K_mm = self.maternKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    # Best Trajectory Extraction Method (>1)
    def getTrainingData(self, dataset: dict, max_episode_steps=1000, top_k=10):
        _flag = '(dataset_extraction ---------->) '
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        num_samples = len(observations)

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }
        step_count = 0  # Track steps in episode (max 1000)

        for i in range(num_samples):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['terminals'].append(terminals[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # End of trajectory
            if terminals[i] or step_count >= max_episode_steps or i == num_samples - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'terminals': np.array(current_trajectory['terminals']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # Compute cumulative rewards
        cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]
        # cumulative_rewards = sorted(cumulative_rewards, reverse=True)

        # Get indices of top K cumulative rewards
        # top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]  # descending order
        # top_k_indices = np.argsort(cumulative_rewards)  # descending order

        # # top_k_indices = np.argsort(cumulative_rewards)
        # print(cumulative_rewards[0])
        # print(top_k_indices[0])

        stash1 = []  # 100-90
        stash2 = []  # 95-90
        stash3 = []  # 90-85
        stash4 = []  # 85-80
        stash5 = []  # 80-75
        stash6 = []  # 75-70
        stash7 = []  # 70-65
        stash8 = []  # 65-60
        stash9 = []  # 60-55
        stash10 = []  # 55-50
        stash11 = []  # 50-45
        stash12 = []  # 45-40
        stash13 = []  # 40-35
        stash14 = []  # 35-30
        stash15 = []  # 30-25
        stash16 = []  # 25-20
        stash17 = []  # 20-15
        stash18 = []  # 15-10
        stash19 = []  # 10-5
        stash20 = []  # 5-0

        # Get max indices
        max_return = np.max(cumulative_rewards)
        print(f'{_flag} Max return: {max_return}')

        for k in range(len(cumulative_rewards)):
        # for k in range(len(top_k_indices)):
            if cumulative_rewards[k] > max_return * 0.95:
                stash1.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.90 and cumulative_rewards[k] <= max_return * 0.95:
                stash2.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.85 and cumulative_rewards[k] <= max_return * 0.90:
                stash3.append(np.where(cumulative_rewards==cumulative_rewards[k]))      
            elif cumulative_rewards[k] > max_return * 0.80 and cumulative_rewards[k] <= max_return * 0.85:
                stash4.append(np.where(cumulative_rewards==cumulative_rewards[k]))     
            elif cumulative_rewards[k] > max_return * 0.75 and cumulative_rewards[k] <= max_return * 0.80:
                stash5.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.70 and cumulative_rewards[k] <= max_return * 0.75:
                stash6.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.65 and cumulative_rewards[k] <= max_return * 0.70:
                stash7.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.60 and cumulative_rewards[k] <= max_return * 0.65:
                stash8.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.55 and cumulative_rewards[k] <= max_return * 0.60:
                stash9.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.50 and cumulative_rewards[k] <= max_return * 0.55:
                stash10.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.45 and cumulative_rewards[k] <= max_return * 0.50:
                stash11.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.40 and cumulative_rewards[k] <= max_return * 0.45:
                stash12.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.35 and cumulative_rewards[k] <= max_return * 0.40:
                stash13.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.30 and cumulative_rewards[k] <= max_return * 0.35:
                stash14.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.25 and cumulative_rewards[k] <= max_return * 0.30:
                stash15.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.20 and cumulative_rewards[k] <= max_return * 0.25:
                stash16.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.15 and cumulative_rewards[k] <= max_return * 0.20:
                stash17.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.10 and cumulative_rewards[k] <= max_return * 0.15:
                stash18.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.05 and cumulative_rewards[k] <= max_return * 0.10:
                stash19.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.00 and cumulative_rewards[k] <= max_return * 0.05:
                stash20.append(np.where(cumulative_rewards==cumulative_rewards[k]))

        # print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")
        print(f'{_flag} stash1: ', len(stash1))
        print(f'{_flag} stash2: ', len(stash2))
        print(f'{_flag} stash3: ', len(stash3))
        print(f'{_flag} stash4: ', len(stash4))
        print(f'{_flag} stash5: ', len(stash5))
        print(f'{_flag} stash6: ', len(stash6))
        print(f'{_flag} stash7: ', len(stash7))
        print(f'{_flag} stash8: ', len(stash8))
        print(f'{_flag} stash9: ', len(stash9))
        print(f'{_flag} stash10: ', len(stash10))
        print(f'{_flag} stash11: ', len(stash11))
        print(f'{_flag} stash12: ', len(stash12))
        print(f'{_flag} stash13: ', len(stash13))
        print(f'{_flag} stash14: ', len(stash14))
        print(f'{_flag} stash15: ', len(stash15))
        print(f'{_flag} stash16: ', len(stash16))
        print(f'{_flag} stash17: ', len(stash17))
        print(f'{_flag} stash18: ', len(stash18))
        print(f'{_flag} stash19: ', len(stash19))
        print(f'{_flag} stash20: ', len(stash20))

        # # Compute Weight (returns)
        # selected_stash = stash9

        # Collect all top K trajectories and stack them
        all_obs = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_next_obs = []
        total_length = 0
        # print(stash6[0])
        # print(cumulative_rewards[np.squeeze(stash6[0])])

        # # for idx in top_k_indices:
        # for idx in selected_stash:
        #     traj = trajectories[idx]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        for i in range(10):
            traj = trajectories[np.squeeze(stash2[i])]
            all_obs.append(traj['observations'])
            all_actions.append(traj['actions'])
            all_rewards.append(traj['rewards'])
            all_terminals.append(traj['terminals'])
            all_next_obs.append(traj['next_observations'])
            total_length += len(traj['observations'])

        # traj = trajectories[np.squeeze(stash4[0])]
        # all_obs.append(traj['observations'])
        # all_actions.append(traj['actions'])
        # all_rewards.append(traj['rewards'])
        # all_terminals.append(traj['terminals'])
        # all_next_obs.append(traj['next_observations'])
        # total_length += len(traj['observations'])

        for i in range(10):
            traj = trajectories[np.squeeze(stash1[i])]
            all_obs.append(traj['observations'])
            all_actions.append(traj['actions'])
            all_rewards.append(traj['rewards'])
            all_terminals.append(traj['terminals'])
            all_next_obs.append(traj['next_observations'])
            total_length += len(traj['observations'])

        # for i in range(7):
        #     traj = trajectories[np.squeeze(stash3[i])]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        # for i in range(5):
        #     traj = trajectories[np.squeeze(stash9[i])]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        # for i in range(5):
        #     traj = trajectories[np.squeeze(stash10[i])]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        # for i in range(5):
        #     traj = trajectories[np.squeeze(stash11[i])]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        # traj = trajectories[np.squeeze(stash5[0])]
        # all_obs.append(traj['observations'])
        # all_actions.append(traj['actions'])
        # all_rewards.append(traj['rewards'])
        # all_terminals.append(traj['terminals'])
        # all_next_obs.append(traj['next_observations'])
        # total_length += len(traj['observations'])

        # traj = trajectories[np.squeeze(stash6[0])]
        # all_obs.append(traj['observations'])
        # all_actions.append(traj['actions'])
        # all_rewards.append(traj['rewards'])
        # all_terminals.append(traj['terminals'])
        # all_next_obs.append(traj['next_observations'])
        # total_length += len(traj['observations'])

        # Stack everything
        stacked_obs = np.vstack(all_obs)
        stacked_actions = np.vstack(all_actions)
        stacked_rewards = np.hstack(all_rewards)  # rewards are usually 1D
        stacked_terminals = np.hstack(all_terminals)
        stacked_next_obs = np.vstack(all_next_obs)

        # Final dictionary
        best_dataset = {
            'arr_0': total_length,  # Total length of stacked trajectories
            'observations': stacked_obs,
            'actions': stacked_actions,
            'rewards': stacked_rewards.reshape(-1, 1),  # Shape [N, 1]
            'terminals': stacked_terminals,
            'next_observations': stacked_next_obs,
        }

        return best_dataset

    # Best Trajectory Extraction Method (>1)
    def getZ(self, dataset: dict, max_episode_steps=1000, top_k=10):
        _flag = '(dataset_extraction ---------->) '
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        num_samples = len(observations)

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'next_observations': []
        }
        step_count = 0  # Track steps in episode (max 1000)

        for i in range(num_samples):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['terminals'].append(terminals[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # End of trajectory
            if terminals[i] or step_count >= max_episode_steps or i == num_samples - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'terminals': np.array(current_trajectory['terminals']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # Compute cumulative rewards
        cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]
        # cumulative_rewards = sorted(cumulative_rewards, reverse=True)

        # Get indices of top K cumulative rewards
        # top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]  # descending order
        # top_k_indices = np.argsort(cumulative_rewards)  # descending order

        # # top_k_indices = np.argsort(cumulative_rewards)
        # print(cumulative_rewards[0])
        # print(top_k_indices[0])

        stash1 = []  # 100-90
        stash2 = []  # 95-90
        stash3 = []  # 90-85
        stash4 = []  # 85-80
        stash5 = []  # 80-75
        stash6 = []  # 75-70
        stash7 = []  # 70-65
        stash8 = []  # 65-60
        stash9 = []  # 60-55
        stash10 = []  # 55-50
        stash11 = []  # 50-45
        stash12 = []  # 45-40
        stash13 = []  # 40-35
        stash14 = []  # 35-30
        stash15 = []  # 30-25
        stash16 = []  # 25-20
        stash17 = []  # 20-15
        stash18 = []  # 15-10
        stash19 = []  # 10-5
        stash20 = []  # 5-0

        # Get max indices
        max_return = np.max(cumulative_rewards)
        print(f'{_flag} Max return: {max_return}')

        for k in range(len(cumulative_rewards)):
        # for k in range(len(top_k_indices)):
            if cumulative_rewards[k] > max_return * 0.95:
                stash1.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.90 and cumulative_rewards[k] <= max_return * 0.95:
                stash2.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.85 and cumulative_rewards[k] <= max_return * 0.90:
                stash3.append(np.where(cumulative_rewards==cumulative_rewards[k]))      
            elif cumulative_rewards[k] > max_return * 0.80 and cumulative_rewards[k] <= max_return * 0.85:
                stash4.append(np.where(cumulative_rewards==cumulative_rewards[k]))     
            elif cumulative_rewards[k] > max_return * 0.75 and cumulative_rewards[k] <= max_return * 0.80:
                stash5.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.70 and cumulative_rewards[k] <= max_return * 0.75:
                stash6.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.65 and cumulative_rewards[k] <= max_return * 0.70:
                stash7.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.60 and cumulative_rewards[k] <= max_return * 0.65:
                stash8.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.55 and cumulative_rewards[k] <= max_return * 0.60:
                stash9.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.50 and cumulative_rewards[k] <= max_return * 0.55:
                stash10.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.45 and cumulative_rewards[k] <= max_return * 0.50:
                stash11.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.40 and cumulative_rewards[k] <= max_return * 0.45:
                stash12.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.35 and cumulative_rewards[k] <= max_return * 0.40:
                stash13.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.30 and cumulative_rewards[k] <= max_return * 0.35:
                stash14.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.25 and cumulative_rewards[k] <= max_return * 0.30:
                stash15.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.20 and cumulative_rewards[k] <= max_return * 0.25:
                stash16.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.15 and cumulative_rewards[k] <= max_return * 0.20:
                stash17.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.10 and cumulative_rewards[k] <= max_return * 0.15:
                stash18.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.05 and cumulative_rewards[k] <= max_return * 0.10:
                stash19.append(np.where(cumulative_rewards==cumulative_rewards[k]))
            elif cumulative_rewards[k] > max_return * 0.00 and cumulative_rewards[k] <= max_return * 0.05:
                stash20.append(np.where(cumulative_rewards==cumulative_rewards[k]))

        # # print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")
        # print(f'{_flag} stash1: ', len(stash1))
        # print(f'{_flag} stash2: ', len(stash2))
        # print(f'{_flag} stash3: ', len(stash3))
        # print(f'{_flag} stash4: ', len(stash4))
        # print(f'{_flag} stash5: ', len(stash5))
        # print(f'{_flag} stash6: ', len(stash6))
        # print(f'{_flag} stash7: ', len(stash7))
        # print(f'{_flag} stash8: ', len(stash8))
        # print(f'{_flag} stash9: ', len(stash9))
        # print(f'{_flag} stash10: ', len(stash10))
        # print(f'{_flag} stash11: ', len(stash11))
        # print(f'{_flag} stash12: ', len(stash12))
        # print(f'{_flag} stash13: ', len(stash13))
        # print(f'{_flag} stash14: ', len(stash14))
        # print(f'{_flag} stash15: ', len(stash15))
        # print(f'{_flag} stash16: ', len(stash16))
        # print(f'{_flag} stash17: ', len(stash17))
        # print(f'{_flag} stash18: ', len(stash18))
        # print(f'{_flag} stash19: ', len(stash19))
        # print(f'{_flag} stash20: ', len(stash20))

        # Compute Weight (returns)
        selected_stash = stash9

        # Collect all top K trajectories and stack them
        all_obs = []
        all_actions = []
        all_rewards = []
        all_terminals = []
        all_next_obs = []
        total_length = 0
        # print(stash6[0])
        # print(cumulative_rewards[np.squeeze(stash6[0])])

        # # for idx in top_k_indices:
        # for idx in selected_stash:
        #     traj = trajectories[idx]
        #     all_obs.append(traj['observations'])
        #     all_actions.append(traj['actions'])
        #     all_rewards.append(traj['rewards'])
        #     all_terminals.append(traj['terminals'])
        #     all_next_obs.append(traj['next_observations'])
        #     total_length += len(traj['observations'])

        _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash2[0])]['observations']), num=500, endpoint=False, dtype=int)
        traj = trajectories[np.squeeze(stash2[0])]
        all_obs.append(traj['observations'][_indexes])
        all_actions.append(traj['actions'][_indexes])
        all_rewards.append(traj['rewards'][_indexes])
        all_terminals.append(traj['terminals'][_indexes])
        all_next_obs.append(traj['next_observations'][_indexes])
        total_length += len(traj['observations'][_indexes])

        _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash1[0])]['observations']), num=500, endpoint=False, dtype=int)
        traj = trajectories[np.squeeze(stash1[0])]
        all_obs.append(traj['observations'][_indexes])
        all_actions.append(traj['actions'][_indexes])
        all_rewards.append(traj['rewards'][_indexes])
        all_terminals.append(traj['terminals'][_indexes])
        all_next_obs.append(traj['next_observations'][_indexes])
        total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash9[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash9[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash10[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash10[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash11[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash11[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash4[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash4[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash9[0])]['observations']), num=500, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash9[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash5[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash5[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # _indexes = np.linspace(start=0, stop=len(trajectories[np.squeeze(stash6[0])]['observations']), num=250, endpoint=False, dtype=int)
        # traj = trajectories[np.squeeze(stash6[0])]
        # all_obs.append(traj['observations'][_indexes])
        # all_actions.append(traj['actions'][_indexes])
        # all_rewards.append(traj['rewards'][_indexes])
        # all_terminals.append(traj['terminals'][_indexes])
        # all_next_obs.append(traj['next_observations'][_indexes])
        # total_length += len(traj['observations'][_indexes])

        # Stack everything
        stacked_obs = np.vstack(all_obs)
        stacked_actions = np.vstack(all_actions)
        stacked_rewards = np.hstack(all_rewards)  # rewards are usually 1D
        stacked_terminals = np.hstack(all_terminals)
        stacked_next_obs = np.vstack(all_next_obs)

        # Final dictionary
        best_dataset = {
            'arr_0': total_length,  # Total length of stacked trajectories
            'observations': stacked_obs,
            'actions': stacked_actions,
            'rewards': stacked_rewards.reshape(-1, 1),  # Shape [N, 1]
            'terminals': stacked_terminals,
            'next_observations': stacked_next_obs,
        }

        return best_dataset

    def loadTrainingRecord(self, path:str, cuda:bool):
        _training_records = torch.load(path, map_location=torch.device('cuda' if cuda else 'cpu'))
        # _training_records = torch.load(self.training_checkpoint_path, map_location=torch.device('cuda' if cuda else 'cpu'))
        self.mll_append = _training_records['loss_append']  # Loss append
        self.training_record = _training_records['training_record']  # Training record counter
        _state_dict = _training_records['state_dict']
        self.x_train = _training_records['x_train']
        self.y_train = _training_records['y_train']
        
        self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
        self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)

        self.q_mean = torch.nn.Parameter(_state_dict['q_mean'], requires_grad=True)
        # self.q_var = torch.nn.Parameter(_state_dict['q_var'], requires_grad=True)
        self.q_var = torch.clip(_state_dict['q_var'], min=0.1)
        self.z_train = torch.nn.Parameter(_training_records['z_train'], requires_grad=True)

        self.num_mixtures = self.param_dict['state_dim']
        self.mean_q = torch.nn.Parameter(_state_dict['mean_q'], requires_grad=True)
        self.weight_q = torch.nn.Parameter(_state_dict['weight_q'], requires_grad=True)
        self.v_q = torch.nn.Parameter(_state_dict['v_q'], requires_grad=True)

        self.K_mm = self.maternKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

    # def getInitInducingPoints(self, x_train, stop_pt:int=320):
    #     _total_indexes = []
    #     _counter = 0
    #     for i in range(self.num_sample):
    #         # _indexes = np.linspace(start=0, stop=len(x_train), num=self.num_inducing, endpoint=False)
    #         if _counter < stop_pt:
    #             _total_indexes.append(i)
    #         else:
    #             if self.terminals[i]:
    #                 _counter = 0
    #         _counter += 1    
    #         if len(_total_indexes) >= self.num_inducing:
    #             break
    #     return x_train[_total_indexes], _total_indexes
    
    def rbfKernel(self, X_1, X_2, noise=False):
        '''
        Isotropic squared exponential kernel.
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
        Returns:
            (m x n) matrix.
        '''
        # # RBF
        # X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        # X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        # # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        # kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1/self.ell, X_2/self.ell)**2)/2)
        # if noise:
        #     # Noisy observation
        #     kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        # return kernel

        # # SVM
        # X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        # X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2     
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

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            # _k_s = self.rbfKernel(X_1=self.z_train, X_2=x_test, noise=False)
            # _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            _k_s = self.maternKernel(X_1=self.z_train, X_2=x_test, noise=False)
            _k_ss = self.maternKernel(X_1=x_test, X_2=x_test, noise=False)

            # # We found that deriving mean and variance this way, provide better results.
            # _mean = _k_s.T @ self.K_inv @ self.y_train
            # _var = _k_ss - _k_s.T @ self.K_inv @ _k_s

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

        return _mean, _var

    # Training Method for GPR. 
    # The reason we train GP here is to add entropy term to regulate the covariance. 
    def myTraining(self, total_epoch:int, ft:bool=False):
        _batch_size = self.param_dict['gp_batch_size']
        _gradient_step = int(self.num_sample//_batch_size)  
        if self.kernel_fn == 'rbf' or self.kernel_fn == 'matern':
            _cov_optimizer = torch.optim.Adam([self.sigma_n, self.ell], lr=1e-02)
        # _cov_optimizer = torch.optim.Adam([self.sigma_n, self.mean_q, self.v_q, self.weight_q], lr=3e-04)
        # _ind_optimizer = torch.optim.Adam([self.q_mean, self.q_var, self.z_train], lr=1e-04)  # was 1e-04
        _ind_optimizer = torch.optim.Adam([self.q_mean, self.q_var], lr=1e-04)  # was 1e-04
        # self.train()  # Set to training mode.
        _training_record = 0

        while _training_record < total_epoch:
        # while self.training_record < total_epoch:
            # _start_time = time()
            _sampling_indexes = torch.randperm(self.num_sample)
            for g in range(_gradient_step):
                _batch_x = self.x_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_y = self.y_train[_sampling_indexes[g*_batch_size:_batch_size+(g*_batch_size)]]
                _batch_z = self.z_train
                _l_1 = 0
                _cov_optimizer.zero_grad()
                _ind_optimizer.zero_grad()

                _K_mm = self.maternKernel(X_1=_batch_z, X_2=_batch_z, noise=True)
                _K_nn = self.maternKernel(X_1=_batch_x, X_2=_batch_x, noise=True)
                _K_mn = self.maternKernel(X_1=_batch_z, X_2=_batch_x, noise=False)
                _L_mm = torch.linalg.cholesky(_K_mm, upper=False)

                _K_tilde = _K_nn - _K_mn.T @ (torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _K_mn, upper=False), upper=True))

                _1 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_var, upper=False), upper=True)
                _3 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, self.q_mean, upper=False), upper=True)
                _sigma_n_sq = self.sigma_n**2
                for h in range(_batch_size-1):
                    _k_i = _K_mn[:, h].view(-1, 1)
                    _mean = _k_i.T @ _3
                    _2 = torch.linalg.solve_triangular(_L_mm.T, torch.linalg.solve_triangular(_L_mm, _k_i, upper=False), upper=True)
                    _sigma_f_sq = torch.abs(_K_tilde[h, h].view(1, 1) + (_k_i.T @ _1 @ _2))
                    _l_1 += (-(_batch_y[h].view(1, self.y_dim)-_mean)**2) / (2*_sigma_n_sq) - \
                            (torch.log(torch.sqrt(2*torch.pi*_sigma_n_sq))) - \
                            (_sigma_f_sq / (2*_sigma_n_sq))

                # KL-divergence term
                _cov_q = torch.sqrt(torch.abs(torch.diagonal(self.q_var))) @ torch.ones(size=[self.num_inducing, self.y_dim])
                _q = torch.distributions.Normal(loc=self.q_mean, scale=_cov_q)
                _u = _q.sample(torch.Size())
                _log_q_u = _q.log_prob(_u)
                _p = torch.distributions.Normal(loc=0., 
                                                scale=torch.sqrt(torch.abs(torch.diagonal(_K_mm)))@torch.ones(size=[self.num_inducing, self.y_dim]))

                # ELBO
                _elbo = _l_1 - (-torch.sum(torch.exp(_log_q_u)*(_p.log_prob(_u)-_log_q_u)))
                _elbo = -_elbo.mean()  # We maximise ELBO.
                _elbo.backward(retain_graph=True)
                _cov_optimizer.step()
                _ind_optimizer.step()

                self.mll_append.append(_elbo.tolist())

            self.training_record += 1
            _training_record += 1

            # print('Gradient_Step: ', self.training_record, 
            #         ', Loss: ', round(_elbo.tolist(), 4), 
            #         ', Var: ', round(torch.diagonal(self.q_var).mean().tolist(), 4),
            #         ', Obs_var: ', round(self.sigma_n.tolist(), 4),
            #         ', Time_usage: ', round(time()-_start_time, 4))

        # self.recordSaving(path=self.training_record_path)
        self.K_mm = self.maternKernel(X_1=self.z_train, X_2=self.z_train, noise=True)
        self.L_mm = torch.linalg.cholesky(self.K_mm, upper=False)

        return _elbo.tolist()

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'training_record': self.training_record,
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train,
                    'z_train': self.z_train}, path)