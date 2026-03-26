# =================================================== Modules ==================================================
from GPDQ_EXACTRBF.GPDQ_EXACTRBF import GaussianProcessDiffusionQlearning 
from utility import dataset_preprocessing

import os
import numpy as np
from time import time, sleep
import argparse
import random
import torch
import matplotlib.pyplot as plt
import gym
import d4rl
# ==============================================================================================================


# ============================================== Local Functions ===============================================
def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def addArguments(parser):
    parser.add_argument('--env', default='walker2d-v2-medium-expert', help='D4RL dataset (Mujoco, Antmaze, ...), default:walker2d-v2-medium-expert')
    parser.add_argument('--alg', default="GPDQ", help='Algorithm: ["GPDQ"], default:GPDQ')
    parser.add_argument('--task', default='training', help='Task: ["training", "testing"], default:training')
    parser.add_argument('--gradient_step', default=100000, help='Gradient Step, default:1000000')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--rendering', default='no-render', help='Rendering: [render, no-render], default:no-render')

def getParamsDict(env):
    return {'simple_sine' : {'environment': 'simple_sine', 'horizon': (4*np.pi//0.05)//2, 'gp_num_sample': 125, 'gp_num_inducing': 25, 'gp_batch_size': 25, 'state_dim': 1, 'action_dim': 1, 'normalise_reward': False,'diffusion_step': 25, 'task': TASK}, 
            'walker2d-medium-expert-v2' :  {'environment': 'walker2d-medium-expert-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'walker2d-medium-replay-v2' :  {'environment': 'walker2d-medium-replay-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 100, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'walker2d-medium-v2' :  {'environment': 'walker2d-medium-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1024, 'gp_batch_size': 1024, 'state_dim': 17, 'action_dim': 6, 'normalise_obs': True, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'walker2d-random-v2' :  {'environment': 'walker2d-medium-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'walker2d-expert-v2' :  {'environment': 'walker2d-expert-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'hopper-medium-expert-v2' :  {'environment': 'hopper-medium-expert-v2', 'horizon': 1000, 'gp_num_sample': 2000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 11, 'action_dim': 3, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'hopper-medium-replay-v2' :  {'environment': 'hopper-medium-replay-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 11, 'action_dim': 3, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'hopper-medium-v2' :  {'environment': 'hopper-medium-v2', 'horizon': 1000, 'gp_num_sample': 2000, 'gp_num_inducing': 1000, 'gp_batch_size': 2000, 'state_dim': 11, 'action_dim': 3, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'hopper-random-v2' :  {'environment': 'hopper-random-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 11, 'action_dim': 3, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'halfcheetah-medium-expert-v2' :  {'environment': 'halfcheetah-medium-expert-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'halfcheetah-medium-replay-v2' :  {'environment': 'halfcheetah-medium-replay-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'halfcheetah-medium-v2' :  {'environment': 'halfcheetah-medium-v2', 'horizon': 1000, 'gp_num_sample': 2000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': False, 'diffusion_step': 5, 'task': TASK},
            'halfcheetah-random-v2' :  {'environment': 'halfcheetah-random-v2', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 17, 'action_dim': 6, 'normalise_reward': True, 'diffusion_step': 5, 'task': TASK},
            'antmaze-umaze-v0' :  {'environment': 'antmaze-umaze-v0', 'horizon': 1000, 'gp_num_sample': 2000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 29, 'action_dim': 8, 'normalise_reward': False, 'diffusion_step': 5, 'task': TASK},
            'antmaze-umaze-diverse-v0' :  {'environment': 'antmaze-umaze-diverse-v0', 'horizon': 1000, 'gp_num_sample': 2000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 29, 'action_dim': 8, 'normalise_reward': False, 'diffusion_step': 5, 'task': TASK},
            'antmaze-medium-play-v0' :  {'environment': 'antmaze-medium-play-v0', 'horizon': 1000, 'gp_num_sample': 1000, 'gp_num_inducing': 1000, 'gp_batch_size': 256, 'state_dim': 29, 'action_dim': 8, 'normalise_reward': False, 'diffusion_step': 5, 'task': TASK},
           }

def interactionMuJoCo(agent, seed, random:bool=False) -> float:
    # `````````` Initialise/Reset observation parameters
    step = 0  # Initialise step counter
    env.seed(seed)
    state = env.reset()
    truncated = False
    terminated = False    
    trajectory_reward = 0
    state_exps = []
    action_exps = []
    reward_exps = []
    next_state_exps = []
    noise_start_time = 300
    max_step = 300

    while not terminated and not truncated if not UNCERTAINTY else not truncated:

        # print(state[0], '    ', state[1])

        if not random:
            norm_state = (state - mean_obs) / (std_obs + 1e-08)
            action = agent.predict(state=norm_state, size=1).detach().cpu().tolist() # Change to cpu memories and make it a list
            # action = agent.getAlteredObservation_old(states=torch.tensor(np.reshape(state, [-1, agent.STATE_DIM]), dtype=torch.float32)).detach().cpu().tolist()
            # action = agent.predict(state=state, size=1, guide=False).detach().cpu().tolist()
            action = np.squeeze(action)
        else:
            action = np.random.randn(6)
        
        if UNCERTAINTY:
            if step > noise_start_time and step < (noise_start_time+100):
                action[3:6] = [0., 0., 0.]
                # action[5] = 0.
                # action = [0., 0., 0.]

            # action[3:6] = [0., 0., 0.]

        # next_state, reward, terminated, truncated = env.step(action)
        next_state, reward, terminated, _ = env.step(action)

        # # Densing reward
        # reward = my_utils.singleDenseReward(state=state)

        trajectory_reward += reward
        
        # Store experiences
        state_exps.append(state)
        action_exps.append(action)
        reward_exps.append(reward)
        next_state_exps.append(next_state)
        
        # Increment
        step += 1
        state = next_state

        # env.render()

    np.stack(state_exps, axis=0)
    np.stack(action_exps, axis=0)
    np.stack(reward_exps, axis=0)  
    reward_exps = np.reshape(reward_exps, [-1, 1])
    np.stack(next_state_exps, axis=0)

    return state_exps, action_exps, reward_exps, next_state_exps

def evaluatingMuJoCo(eval_times:int=10, step:int=0):
    _TEST_TIME = eval_times
    _TEST_SEEDS = [2203+(n*(9**n)) for n in range(eval_times)]  # MuJoCo
    # _TEST_SEEDS = [2203 + (n * 100) for n in range(eval_times)]
    # _TEST_SEEDS = [2203+(n+(9**n)) for n in range(eval_times)]  # AntMaze
    # _TEST_SEEDS = [2203, 2212, 3486786604]
    # _TEST_SEEDS = [2212, 4390, 28447]  # Hopper Medium-Expert
    # _TEST_SEEDS = [28447, 28446, 28448] 
    # _TEST_SEEDS = [2203+(2203*(5*n)) for n in range(eval_times)]  # good
    # _TEST_SEEDS = [2203+(4213*(11*n)) for n in range(eval_times)]
    # _EXPERT_REWARD = 4580.59  # Raw reward of SAC with Stochastic Policy
    _EXPERT_REWARD = 4592.30  # D4RL
    # _EXPERT_REWARD = 13091.87
    # _EXPERT_REWARD = 3453.56
    return_append = []
    norm_return_append = []

    # print('********************************* Evaluation Start *********************************')
    for e in range(_TEST_TIME):
        env = gym.make(args.env)
        # setGlobalSeed(_TEST_SEEDS[e])
        setGlobalSeed(seed)
        # Testing only (No fine-tuning)
        # ========== Start the program
        # Check for the rendering
        # Run evaluated policy
        for _ in range(1):
            _, _, reward_exps, _ = interactionMuJoCo(agent, _TEST_SEEDS[e], random=False)
            # _, _, reward_exps, _ = interactionMuJoCo(agent, seed, random=False)
            # _, _, random_reward_exps, _ = interactionMuJoCo(agent, _TEST_SEEDS[e], random=True)
            trajectory_reward = np.sum(reward_exps)
            norm_trajectory_reward = env.get_normalized_score(trajectory_reward) * 100
            # random_trajectory_reward = np.sum(random_reward_exps)
            return_append.append(trajectory_reward)
            norm_return_append.append(norm_trajectory_reward)
        env.close()

        # print('Test: ', e+1, '/', _TEST_TIME, 
        # ', Accum_Reward: ', round(trajectory_reward, 2), 
        # ', Norm_Accum_Reward: ', round(norm_trajectory_reward , 2), 
        # ', Seed: ', _TEST_SEEDS[e])

    avg_return = np.mean(return_append)
    avg_norm_return = np.mean(norm_return_append)
    std_norm_return = np.std(norm_return_append)
    # norm_return_append.sort(reverse=True)
    # avg_max_3_seeds = np.mean(norm_return_append[0:3])#
    # std_max_3_seed = np.std(norm_return_append[0:3])

    # _tar_q = torch.sum(agent.q_1(torch.concat([agent.gp_model.x_train, agent.gp_model.y_train], dim=1)))

    # _pred_a = agent.predict(state=agent.gp_model.x_train, size=agent.gp_model.gp_training_size)
    # _pred_q = torch.sum(agent.q_1(torch.concat([agent.gp_model.x_train, _pred_a], dim=1)))

    # _err_q = _tar_q - _pred_q

    print('Summary',
        #   ', Epoch: ', agent.training_record,
          ', Gradient_Step: ', agent.gradient_step,
          ', Avg_Accum_Reward: ', round(avg_return, 4), 
          ', Norm_Avg_Accum_Reward: ', round(avg_norm_return , 4), 
          ', Norm_Std_Accum_Reward: ', round(std_norm_return, 4),
        #   ', Sum_Q: ', round(_err_q.tolist(), 4)
        #   ', Avg_max_3_seeds: ', round(avg_max_3_seeds, 2), 
        #   ', Std_max_3_seeds: ', round(std_max_3_seed, 2)
          )
    # print('********************************** Evaluation End **********************************')
    return avg_return, avg_norm_return
    # return avg_return, avg_max_3_seeds

# ==============================================================================================================


# ============================================= Main Program Start =============================================
seed = 2203
setGlobalSeed(seed)

parser = argparse.ArgumentParser(description=
                                 'GPDQ (Transaction of Neural Network and Learning Systems 2026) PyTorch Args')
addArguments(parser)
args = parser.parse_args()

# ========== Global constants
ENV_NAME = args.env
TASK = args.task
RENDER = True if args.rendering == 'render' else False
UNCERTAINTY = True if args.eval_mode == 'shifted' else False

# Get parameters dictionary of the environment.
params_dict = getParamsDict(env=ENV_NAME)
env = gym.make(args.env)
dataset = d4rl.qlearning_dataset(env)

# Dataset Preprocessing
dataset, mean_obs, std_obs = dataset_preprocessing.datasetPreprocessing(dataset=dataset, 
                                                     obs_tuned=params_dict[args.env]['normalise_obs'], 
                                                     reward_tuned=params_dict[args.env]['normalise_reward'])



# -------------------------------------- Create the agent ------------------------------------------------------
if args.alg == 'GPDQ':                                                                                        
    agent = GaussianProcessDiffusionQlearning(params_dict=params_dict[args.env], dataset=dataset)   
# --------------------------------------------------------------------------------------------------------------



# -------------------------------------- Training (Offline) ----------------------------------------------------
EPOCH = int(args.gradient_step) / (agent.NUM_SAMPLE//agent.MINIBATCH_SIZE)
if args.task == 'training':
    # Train Diffusion Policy and Value functions.
    while agent.training_record < EPOCH:
        print('Start Offline Training...')
        # Train the algorithm.
        agent.training(total_epoch=EPOCH, eval=False)
    print('Training is done!')
# --------------------------------------------------------------------------------------------------------------



# ------------------------------------ Training Behaviour Policy (Offline) -------------------------------------
elif args.task == 'beh_training':
    while agent.training_record < EPOCH:
        print('Start Offline Behaviour Model Training...')
        # Train the algorithm.
        agent.behTraining(total_epoch=EPOCH, eval=False)
    print('Training is done!')
# --------------------------------------------------------------------------------------------------------------



# ---------------------------------------------- Testing -------------------------------------------------------
elif args.task == 'testing':
    # # Start environment
    # _reward_append = []
    # _folder_path = 'GPDQ_exact_2/training_records/hopper-medium-v2'
    # for filename in os.listdir(_folder_path):
    #     _file_path = os.path.join(_folder_path, filename)
    #     agent.loadTrainingRecord(path=_file_path)
    #     # agent.gp_model.loadTrainingRecord(path=agent.gp_model.training_record_path, cuda=agent.cuda)
    #     _, norm_reward_append = evaluatingMuJoCo(eval_times=5)
    #     _reward_append.append(norm_reward_append)
    # np.save(agent.evaluation_path, _reward_append) # save the evaluation rewards.
    
    j = 1000000 # now at 405
    # j = 1100000
    _reward_append = []
    while j <= 2000000:
        # if j >= 256:
        #     _increment = 1
        # elif j == 250:
        #     _increment = 6
        # else:
        #     _increment = 10
        _increment = 10000
        agent.loadTrainingRecord(path=agent.training_record_path+'_{:d}'.format(j))
        agent.loadGPTrainingRecord(path=agent.training_record_path+'_{:d}'.format(j))
        # agent.gp_model.loadTrainingRecord(path=agent.gp_model.training_record_path+'_{:d}'.format(j), cuda=agent.cuda)
        _, norm_reward_append = evaluatingMuJoCo(eval_times=5)
        _reward_append.append(norm_reward_append)
        j += _increment
        # if j == 70:
        #     j = 64
        # np.savez(agent.evaluation_path, _reward_append, j) # save the evaluation rewards.

    # # Plot to see the selected trajectories
    # x = agent.gp_model.x_train[:, 0].detach().cpu().numpy()
    # y = agent.gp_model.x_train[:, 1].detach().cpu().numpy()
    # sc = plt.scatter(x, y, c=np.arange(len(x)), cmap='viridis', s=50, edgecolor='k')
    # # plt.plot(y, x)
    # plt.colorbar(sc, shrink=0.8)
    # # plt.scatter(x, y)
    # # plt.xlim(0, 10)
    # # plt.ylim(0, 10)
    # plt.show()

    # j = 10000 # now at 405
    # _reward_append = []
    # _increment = 10000
    # while j <= 2000000:
    #     agent.loadTrainingRecord(path=agent.training_record_path+'_{:d}'.format(j))
    #     agent.loadGPTrainingRecord(path=agent.training_record_path+'_{:d}'.format(j))
    #     # agent.gp_model.loadTrainingRecord(path=agent.gp_model.training_record_path+'_{:d}'.format(j), cuda=agent.cuda)
    #     _, norm_reward_append = evaluatingMuJoCo(eval_times=5, step=j)
    #     _reward_append.append(norm_reward_append)
    #     j += _increment
    #     # if j == 70:
    #     #     j = 64
    # np.save(agent.evaluation_path, _reward_append) # save the evaluation rewards.

    # # agent.loadTrainingRecord(path=agent.training_record_path+'_255')
    # _, norm_reward_append = evaluatingMuJoCo(eval_times=5)
    # np.save(agent.evaluation_path, norm_reward_append) # save the evaluation rewards.

    # checkpoint = '461'
    # agent.loadTrainingRecord(path=agent.training_record_path+'_'+checkpoint)
    # agent.loadGPTrainingRecord(path=agent.training_record_path+'_'+checkpoint)
    # # agent.loadTrainingRecord(path=agent.training_record_path)
    # # agent.gp_model.loadTrainingRecord(path=agent.gp_model.training_record_path+'_'+checkpoint, cuda=agent.cuda)
    # # agent.gp_model.loadTrainingRecord(path=agent.gp_model.training_record_path, cuda=agent.cuda)
    # _, _ = evaluatingMuJoCo(eval_times=10)
# --------------------------------------------------------------------------------------------------------------


# ============================================== Main Program End ==============================================