import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import matplotlib.pyplot as plt

from HGDQ_EXACTRBF.HGDQ_EXACTRBF import HiearchicalGaussianprocessDiffusionQlearning

num_data = 100000
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

# for i in range(num_data):
#     if step_count 

# plt.scatter(obs[:num_traj, 0], obs[:num_traj, 1])
# plt.scatter(obs[num_traj:2*num_traj, 0], obs[num_traj:2*num_traj, 1])
# plt.scatter(obs[2*num_traj:3*num_traj, 0], obs[2*num_traj:3*num_traj, 1])
# plt.show()
    
agent = HiearchicalGaussianprocessDiffusionQlearning(params_dict=params_dict, dataset=dataset)

# EPOCH = 1000
# while agent.training_record < EPOCH:
#     print('Start Offline Behaviour Model Training...')
#     # Train the algorithm.
#     agent.behTraining(total_epoch=EPOCH, eval=False)
# print('Training is done!')

pred = agent.predict2(state=dataset['observations'][:num_test], size=num_test, guide=False)
# pred = agent.predict2(state=agent.gp_model.x_train_full[:num_test], size=num_test, guide=True)
plt.scatter(pred[:, 0].tolist(), pred[:, 1].tolist(), label='pred')
plt.scatter(agent.gp_model.x_train_full[:, 0], agent.gp_model.x_train_full[:, 1], label='GP')
# plt.scatter(dataset['next_observations'][:, 0], dataset['next_observations'][:, 1], label='gt')
plt.legend()
plt.show()
