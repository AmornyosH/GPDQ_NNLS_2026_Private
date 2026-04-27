import numpy as np
import torch
import matplotlib.pyplot as plt

def rangeConversion(input, new_min, new_max, old_min=-1, old_max=1) -> float:
    '''
    Range Conversion Method (Mapping from old range to a new range) \\
    This method will map the value in the defined old range (default = [-1, 1])
    in to a new defined range (no default). 
    Formula = new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min))
    
    Input Arguments: sample:Any, new_min, new_max, old_min=-1, old_max=1
    Return: sample in new range (sample \in [new_min, new_max])
    '''
    # Default of old range is [-1, 1]
    # Define parameters
    # old_length = (old_max - old_min) / 2  # length between old mid point and old marginal points.
    # old_new_diff = new_max - (old_max/old_length)  # the difference between old marginal point and new marginal point.
    # return tf.clip_by_value(new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min)), 
    #                         clip_value_min=new_min, 
    #                         clip_value_max=new_max)
    return torch.clip(new_min + ((input - old_min) * (new_max - new_min) / (old_max - old_min)), 
                            min=new_min, 
                            max=new_max)

def klDivergence(input_1, input_2):
    '''
    KL-Divergence Computation Method \\
    This method will return the Kullback-Leibler Divergence (D_KL) between input_1 and input_2.\\
    Remind that D_KL between input_1 and input_2 is not equal to D_KL between input_2 and input_1.

    Input Arguments: input_1:Any, input_2:Any (Numpy Format)
    Return: D_KL(input_1 || input_2) (Numpy Format)
    '''
    # old_policy = tf.gather_nd(old_policy, indices=slice_indices.astype(int))
    # return tf.reduce_mean(-tf.reduce_sum(old_policy * tf.math.log(new_policy/old_policy), axis=1)).numpy()
    kl = -np.sum(input_1 * np.log(input_2/input_1)) / len(input_1)
    return kl

# Sinusoidal Position Encoding (Same as the original Transformer.)
def sinPositionEncoding(seq_len, dim, N=10000):
    output = np.zeros([seq_len, dim], dtype=float)
    for k in range(seq_len):
        for i in np.arange(int(dim/2)):
            denominator = np.power(N, 2*i/dim)
            output[k, 2*i] = np.sin(k/denominator)
            output[k, 2*i+1] = np.cos(k/denominator)
    return output

# Positional encoding method (My implementation)
def sinusoidalEncoding(input, seq_len):
    t_space = np.linspace(start=-1, stop=1, num=seq_len, dtype=float)  # [-1, -0.5, 0, 0.5, 1]
    return np.sin(2 * np.pi * t_space[input-1])  # sin(2 * pi * t)

# Best Trajectory Extraction Method (>1)
def bestTrajExtraction(dataset: dict, max_episode_steps=1000, top_k=10):
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

    print(f"{_flag} Extracted {len(trajectories)} trajectories (max 1000 steps each).")

    # Compute cumulative rewards
    cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]
    top_k_indices = np.argsort(cumulative_rewards)  # descending order

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

    # Compute Weight (returns)
    selected_stash = np.argmax([len(stash1), len(stash2), len(stash3), len(stash4), len(stash5), len(stash6), len(stash7), len(stash8), len(stash9), len(stash10), 
                                len(stash11), len(stash12), len(stash13), len(stash14), len(stash15), len(stash16), len(stash17), len(stash18), len(stash19), len(stash20)])


    # Collect all top K trajectories and stack them
    all_obs = []
    all_actions = []
    all_rewards = []
    all_terminals = []
    all_next_obs = []
    total_length = 0
    print(stash6[0])
    print(cumulative_rewards[np.squeeze(stash6[0])])

    traj = trajectories[np.squeeze(stash10[0])]
    all_obs.append(traj['observations'])
    all_actions.append(traj['actions'])
    all_rewards.append(traj['rewards'])
    all_terminals.append(traj['terminals'])
    all_next_obs.append(traj['next_observations'])
    total_length += len(traj['observations'])

    traj = trajectories[np.squeeze(stash9[0])]
    all_obs.append(traj['observations'])
    all_actions.append(traj['actions'])
    all_rewards.append(traj['rewards'])
    all_terminals.append(traj['terminals'])
    all_next_obs.append(traj['next_observations'])
    total_length += len(traj['observations'])

    traj = trajectories[np.squeeze(stash8[0])]
    all_obs.append(traj['observations'])
    all_actions.append(traj['actions'])
    all_rewards.append(traj['rewards'])
    all_terminals.append(traj['terminals'])
    all_next_obs.append(traj['next_observations'])
    total_length += len(traj['observations'])


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
def multiBestTrajExtraction(dataset: dict, max_episode_steps=1000, top_k=10):
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
        'next_observations': []
    }
    step_count = 0  # Track steps in episode (max 1000)

    for i in range(num_samples):
        current_trajectory['observations'].append(observations[i])
        current_trajectory['actions'].append(actions[i])
        current_trajectory['rewards'].append(rewards[i])
        current_trajectory['next_observations'].append(next_observations[i])
        step_count += 1

        # End of trajectory
        if terminals[i] or step_count >= max_episode_steps or i == num_samples - 1:
            if step_count == 1:
                ... 
            else:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
            # Reset
            current_trajectory = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': []
            }
            step_count = 0

    print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

    # Compute cumulative rewards
    cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]

    # Get indices of top K cumulative rewards
    top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]  # descending order

    print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")

    # Collect all top K trajectories and stack them
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    total_length = 0

    for idx in top_k_indices:
        traj = trajectories[idx]
        all_obs.append(traj['observations'])
        all_actions.append(traj['actions'])
        all_rewards.append(traj['rewards'])
        all_next_obs.append(traj['next_observations'])
        total_length += len(traj['observations'])

    # Stack everything
    stacked_obs = np.vstack(all_obs)
    stacked_actions = np.vstack(all_actions)
    stacked_rewards = np.hstack(all_rewards)  # rewards are usually 1D
    stacked_next_obs = np.vstack(all_next_obs)

    # Final dictionary
    best_dataset = {
        'arr_0': total_length,  # Total length of stacked trajectories
        'observations': stacked_obs,
        'actions': stacked_actions,
        'rewards': stacked_rewards.reshape(-1, 1),  # Shape [N, 1]
        'next_observations': stacked_next_obs,
    }

    return best_dataset

# Maze Problems Trajectory Extraction Method
def mazeTrajExtraction(dataset:dict, max_episode_steps=1000, top_k=10):
    print('Extract maze trajectories!!!!!')
    observations = dataset['observations']
    actions = dataset['actions']
    rewards = dataset['rewards']
    terminals = dataset['terminals']
    next_observations = dataset['next_observations']

    # Find the initial state first
    num_samples = len(observations)

    _init = np.array([0., 0.], dtype=float)
    # _target = np.array([0.36824249435748196, 9.156394661806964], dtype=float)
    # _target = np.array([1, 0.5], dtype=float)
    # _target = np.array([5.5, 6.0], dtype=float)
    _target = np.array([20.36824249435748, 21.156394661806964], dtype=float)
    _threshold = 1.0

    trajectories = []
    current_trajectory = {
        'observations': [],
        'actions': [],
        'rewards': [],
        'next_observations': []
    }
    step_count = 0  # Track steps in episode (max 1000)

    for i in range(num_samples):
        # Current Euclidean Distance
        _dist = np.linalg.norm(next_observations[i, :2]-_target)
        _init_dist = np.linalg.norm(observations[i, 0:2]-_init)

        if step_count == 0:
            # if _init_dist < 1:
            if _dist > _threshold and _init_dist < _threshold:   # <==== Use this condition for both antmaze-umaze datasets.
                current_trajectory['observations'].append(observations[i])
                current_trajectory['actions'].append(actions[i])
                current_trajectory['rewards'].append(rewards[i])
                current_trajectory['next_observations'].append(next_observations[i])
                step_count += 1
            else:
                # Reset
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'terminals': [],
                    'next_observations': []
                }
                step_count = 0
        else:
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            if terminals[i] or step_count >= max_episode_steps or i == num_samples-1:  
            # if _dist < _threshold or step_count >= max_episode_steps or i == num_samples-1:  
                # if terminals[i]:
                if _dist < _threshold:
                    trajectories.append({
                        'observations': np.array(current_trajectory['observations']),
                        'actions': np.array(current_trajectory['actions']),
                        'rewards': np.array(current_trajectory['rewards']),
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
                

    print(f"Extracted {len(trajectories)} trajectories (max {max_episode_steps} steps each).")

    # Compute cumulative rewards
    cumulative_rewards = [np.sum(traj['rewards']) for traj in trajectories]

    # Get indices of top K cumulative rewards
    top_k_indices = np.argsort(cumulative_rewards)[-top_k:][::-1]  # descending order
    # top_k_indices = np.argsort(cumulative_rewards)[::-1]  # descending order

    print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")

    # Collect all top K trajectories and stack them
    all_obs = []
    all_actions = []
    all_rewards = []
    all_next_obs = []
    total_length = 0

    for idx in top_k_indices:
        traj = trajectories[idx]
        all_obs.append(traj['observations'])
        all_actions.append(traj['actions'])
        all_rewards.append(traj['rewards'])
        all_next_obs.append(traj['next_observations'])
        total_length += len(traj['observations'])

    # print(f"Top {top_k} cumulative rewards: {[cumulative_rewards[i] for i in top_k_indices]}")

    # Stack everything
    stacked_obs = np.vstack(all_obs)
    stacked_actions = np.vstack(all_actions)
    stacked_rewards = np.hstack(all_rewards)  # rewards are usually 1D
    stacked_next_obs = np.vstack(all_next_obs)

    # Final dictionary
    best_dataset = {
        'arr_0': total_length,  # Total length of stacked trajectories
        'observations': stacked_obs,
        'actions': stacked_actions,
        'rewards': stacked_rewards.reshape(-1, 1),  # Shape [N, 1]
        'next_observations': stacked_next_obs,
    }

    # plt.scatter(stacked_obs[:1000, 0], stacked_obs[:1000, 1])
    # plt.show()

    return best_dataset