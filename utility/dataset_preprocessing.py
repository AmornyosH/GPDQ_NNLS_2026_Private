import numpy as np
import matplotlib.pyplot as plt

def datasetPreprocessingMuJoCo(dataset, obs_tuned:bool=False, reward_tuned:bool=False):
    """
    Preprocess D4RL MuJoCo dataset for offline RL algorithms (IQL-style normalization).

    Args:
        reward_scale (float): Factor to scale rewards (default 1.0, set to 1/1000 for IQL style).
        clip_actions (bool): Whether to clip actions to environment bounds.

    Returns:
        dict: Preprocessed dataset with keys:
              'observations', 'actions', 'rewards', 'terminals', 'next_observations'
    """
    
    # Rewards Normalisation Method (Followed IQL) (Used for Gym-MuJoCo tasks only)
    def normaliseRewardsIQL(dataset):
        """
        IQL-style reward normalization with max 1000-step trajectories.
        Splits dataset, computes returns, normalizes rewards.
        """

        # === Step 1: Split dataset into trajectories (max 1000 steps each) ===
        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        next_observations = dataset['next_observations']

        trajectories = []
        current_trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'next_observations': []
        }
        step_count = 0  # To track steps within a trajectory (max 1000)

        for i in range(len(observations)):
            current_trajectory['observations'].append(observations[i])
            current_trajectory['actions'].append(actions[i])
            current_trajectory['rewards'].append(rewards[i])
            current_trajectory['next_observations'].append(next_observations[i])
            step_count += 1

            # Check for episode end: either terminal=True or max 1000 steps
            if terminals[i] or step_count >= 1000 or i == len(observations) - 1:
                trajectories.append({
                    'observations': np.array(current_trajectory['observations']),
                    'actions': np.array(current_trajectory['actions']),
                    'rewards': np.array(current_trajectory['rewards']),
                    'next_observations': np.array(current_trajectory['next_observations'])
                })
                # Reset for next trajectory
                current_trajectory = {
                    'observations': [],
                    'actions': [],
                    'rewards': [],
                    'next_observations': []
                }
                step_count = 0

        print(f"Extracted {len(trajectories)} trajectories (max 1000 steps each).")

        # === Step 2: Compute returns ===
        returns = [np.sum(traj['rewards']) for traj in trajectories]
        returns_sorted = sorted(returns)
        R_min = returns_sorted[0]
        R_max = returns_sorted[-1]
        print(f"Return range: min={R_min:.2f}, max={R_max:.2f}")

        # === Step 3: Normalize rewards globally ===
        scaling_factor = 1.0 / (R_max - R_min + 1e-8)
        normalized_rewards = rewards * scaling_factor * 1000.0

        print(f"Rewards normalized with scaling factor: {scaling_factor:.6f}")

        return normalized_rewards
    
    obs = dataset['observations'].copy()
    actions = dataset['actions'].copy()
    rewards = dataset['rewards'].copy()
    terminals = dataset['terminals'].copy()
    next_obs = dataset['next_observations'].copy()

    if obs_tuned:
        mean_obs = np.mean(obs, axis=0)
        std_obs = np.std(obs, axis=0) + 1e-3
        normalized_obs = (obs - mean_obs) / std_obs
        obs = normalized_obs

        # Compute next observations
        next_obs = np.roll(normalized_obs, -1, axis=0)
        # # Handle episode boundaries: next_obs after terminal is meaningless, optional to zero out
        # next_obs[terminals==1] = 0.0
    else:
        mean_obs = None
        std_obs = None

    if reward_tuned:
        rewards = normaliseRewardsIQL(dataset=dataset)

    # Handle episode boundaries: next_obs after terminal is meaningless, optional to zero out
    next_obs[terminals==1] = 0.0

    preprocessed_dataset = {
        'observations': obs,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'next_observations': next_obs,
    }

    print(f"Dataset preprocessed:")
    print(f"  Observations mean/std: {np.mean(preprocessed_dataset['observations'], axis=0)[:5]} / {np.std(preprocessed_dataset['observations'], axis=0)[:5]}")
    print(f"  Rewards mean/std: {np.mean(preprocessed_dataset['rewards']):.3f} / {np.std(preprocessed_dataset['rewards']):.3f}")
    print(f"  Actions min/max: {preprocessed_dataset['actions'].min(axis=0)} / {preprocessed_dataset['actions'].max(axis=0)}")

    return preprocessed_dataset, mean_obs, std_obs

def datasetPreprocessingMaze(dataset, obs_tuned:bool=False, reward_tuned:bool=False):
    """
    Preprocess D4RL MuJoCo dataset for offline RL algorithms (IQL-style normalization).

    Args:
        reward_scale (float): Factor to scale rewards (default 1.0, set to 1/1000 for IQL style).
        clip_actions (bool): Whether to clip actions to environment bounds.

    Returns:
        dict: Preprocessed dataset with keys:
              'observations', 'actions', 'rewards', 'terminals', 'next_observations'
    """

    obs = dataset['observations'].copy()
    actions = dataset['actions'].copy()
    rewards = dataset['rewards'].copy()
    terminals = dataset['terminals'].copy()
    next_obs = dataset['next_observations'].copy()

    if obs_tuned:
        mean_obs = np.mean(obs, axis=0)
        std_obs = np.std(obs, axis=0) + 1e-3
        normalized_obs = (obs - mean_obs) / std_obs
        obs = normalized_obs

        # Compute next observations
        next_obs = np.roll(normalized_obs, -1, axis=0)
        # # Handle episode boundaries: next_obs after terminal is meaningless, optional to zero out
        # next_obs[terminals==1] = 0.0
    else:
        mean_obs = None
        std_obs = None

    if reward_tuned:
        rewards -= 1

    terminals[rewards>0] = True
    # print(obs[np.where(terminals==True), :2])
    # # Handle episode boundaries: next_obs after terminal is meaningless, optional to zero out
    # next_obs[terminals==1] = 0.0

    preprocessed_dataset = {
        'observations': obs,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'next_observations': next_obs,
    }

    print(f"Dataset preprocessed Maze:")
    print(f"  Observations mean/std: {np.mean(preprocessed_dataset['observations'], axis=0)[:5]} / {np.std(preprocessed_dataset['observations'], axis=0)[:5]}")
    print(f"  Rewards mean/std: {np.mean(preprocessed_dataset['rewards']):.3f} / {np.std(preprocessed_dataset['rewards']):.3f}")
    print(f"  Actions min/max: {preprocessed_dataset['actions'].min(axis=0)} / {preprocessed_dataset['actions'].max(axis=0)}")

    # plt.scatter(obs[:1000, 0], obs[:1000, 1])
    # plt.show()

    return preprocessed_dataset, mean_obs, std_obs

def normaliseObservation(observation, mean, std):
    return (observation - mean) / std