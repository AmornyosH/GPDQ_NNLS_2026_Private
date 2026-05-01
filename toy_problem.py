import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt
import torch

from HGDQ_EXACTRBF.HGDQ_EXACTRBF import HiearchicalGaussianprocessDiffusionQlearning

num_data = 1000000
num_test = 1000
num_traj = 1000
dataset = np.load('antmaze-medium-play-v0.npz', allow_pickle=True)

dataset = {'observations': (dataset['arr_0'][0:num_data, :2] / 22)*2 - 1, 
           'actions': (dataset['arr_4'][0:num_data, :2] / 22)*2 - 1, 
           'rewards': dataset['arr_2'][0:num_data] - 1, 
           'terminals': dataset['arr_3'][0:num_data], 
           'next_observations': (dataset['arr_4'][0:num_data, :2] / 22)*2 - 1}

params_dict = {'environment': 'toy_problem', 
               'horizon': 1000, 
               'gp_num_sample': 1000, 
               'state_dim': 2, 
               'action_dim': 2, 
               'goal_dim': 2, 
               'normalise_obs': False, 
               'normalise_reward': True, 
               'diffusion_step': 50}

obs = dataset['observations']
actions = dataset['actions']
rewards = dataset['rewards']
terminals = dataset['terminals']
next_obs = dataset['next_observations']

# plt.scatter(obs[:num_traj, 0], obs[:num_traj, 1])
# # plt.scatter(obs[num_traj:2*num_traj, 0], obs[num_traj:2*num_traj, 1])
# # plt.scatter(obs[2*num_traj:3*num_traj, 0], obs[2*num_traj:3*num_traj, 1])
# plt.show()
    
agent = HiearchicalGaussianprocessDiffusionQlearning(params_dict=params_dict, dataset=dataset)

# EPOCH = 10000
# while agent.training_record < EPOCH:
#     print('Start Offline Behaviour Model Training...')
#     # Train the algorithm.
#     agent.behTraining(total_epoch=EPOCH, eval=False)
# print('Training is done!')

n = 32  # because 10 x 10 = 100
x = np.linspace(-1, 1, n)
y = np.linspace(-1, 1, n)

# X, Y = np.meshgrid(x, y)
# coords = np.stack([X.ravel(), Y.ravel()], axis=-1)  # (100, 2)

# agent.gp_model.x_train_full = torch.tensor(coords, dtype=torch.float32)

pred = agent.predict2(state=dataset['observations'][:num_test], size=num_test, guide=False)
# pred = agent.predict2(state=agent.gp_model.y_train_full[:num_test], size=num_test, guide=True)
# plt.scatter(pred[:, 0].tolist(), pred[:, 1].tolist(), label='pred')
plt.scatter(agent.gp_model.x_train_full[:num_test, 0].cpu(), agent.gp_model.x_train_full[:num_test, 1].cpu(), label='GP')
plt.scatter(pred[:, 0].tolist(), pred[:, 1].tolist(), label='pred')
# plt.scatter(dataset['next_observations'][:, 0], dataset['next_observations'][:, 1], label='gt')

plt.legend()
plt.show()
