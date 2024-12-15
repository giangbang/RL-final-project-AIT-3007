import numpy as np
def process_batch(batchs):
    # Initialize dictionaries to hold lists for each field
    batch_blue = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'state': [],
        'next_state': [],
        'dones': []
    }
    batch_red = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'state': [],
        'next_state': [],
        'dones': []
    }

    for episode in batchs:
        # Initialize lists to collect per-episode data
        obs_blue_episode = []
        actions_blue_episode = []
        rewards_blue_episode = []
        next_obs_blue_episode = []
        dones_blue_episode = []

        obs_red_episode = []
        actions_red_episode = []
        rewards_red_episode = []
        next_obs_red_episode = []
        dones_red_episode = []

        states_episode = []
        next_states_episode = []

        for transition in episode:
            # Collect shared states
            states_episode.append(transition['state'])          # Shape: (45, 45, 5)
            next_states_episode.append(transition['next_state'])  # Shape: (45, 45, 5)

            # Collect data for the blue team
            obs_blue_episode.append(transition['obs']['blue'])        # Shape: (transitions, N_blue, 13, 13, 5)
            actions_blue_episode.append(transition['actions']['blue'])   # Shape: (transitions, N_blue, )
            rewards_blue_episode.append(transition['rewards']['blue'])   # Shape: (trainsitions, 1,)
            next_obs_blue_episode.append(transition['next_obs']['blue']) # Shape: (transitions, N_blue, 13, 13, 5)
            dones_blue_episode.append(transition['dones']['blue'])      # Shape: (transitions, 1, )

            # Collect data for the red team
            obs_red_episode.append(transition['obs']['red'])
            actions_red_episode.append(transition['actions']['red'])
            rewards_red_episode.append(transition['rewards']['red'])
            next_obs_red_episode.append(transition['next_obs']['red'])
            dones_red_episode.append(transition['dones']['red'])

        # Convert lists to numpy arrays for the blue team
        obs_blue_episode = np.array(obs_blue_episode)            # Shape: ( transitions, N_blue, 13, 13, 5)
        actions_blue_episode = np.array(actions_blue_episode)    # Shape: ( transitions, N_blue)
        rewards_blue_episode = np.array(rewards_blue_episode)    # Shape: ( transitions, 1)
        next_obs_blue_episode = np.array(next_obs_blue_episode)  # Shape: ( transitions, N_blue, 13, 13, 5)
        dones_blue_episode = np.array(dones_blue_episode)        # Shape: ( transitions, 1)

        # Append to batch_blue
        batch_blue['obs'].append(obs_blue_episode) 
        batch_blue['actions'].append(actions_blue_episode)
        batch_blue['rewards'].append(rewards_blue_episode)
        batch_blue['next_obs'].append(next_obs_blue_episode)
        batch_blue['dones'].append(dones_blue_episode)

        # Convert lists to numpy arrays for the red team
        obs_red_episode = np.array(obs_red_episode)
        actions_red_episode = np.array(actions_red_episode)
        rewards_red_episode = np.array(rewards_red_episode)
        next_obs_red_episode = np.array(next_obs_red_episode)
        dones_red_episode = np.array(dones_red_episode)

        # Append to batch_red
        batch_red['obs'].append(obs_red_episode)
        batch_red['actions'].append(actions_red_episode)
        batch_red['rewards'].append(rewards_red_episode)
        batch_red['next_obs'].append(next_obs_red_episode)
        batch_red['dones'].append(dones_red_episode)


        # Convert states to arrays
        states_episode = np.array(states_episode)            # Shape: (episode_n, 45, 45, 5)
        next_states_episode = np.array(next_states_episode)   # Shape: (episode_n, 45, 45, 5)

        # Append states to both batches
        batch_blue['state'].append(states_episode)
        batch_blue['next_state'].append(next_states_episode)
        batch_red['state'].append(states_episode)
        batch_red['next_state'].append(next_states_episode)

    # Stack data along the batch dimension for the blue team
    for key in batch_blue:
        batch_blue[key] = np.stack(batch_blue[key], axis=0)
        # Now, batch_blue[key] has shape (B, episode_n, ...)

    # Stack data along the batch dimension for the red team
    for key in batch_red:
        batch_red[key] = np.stack(batch_red[key], axis=0)
        # Now, batch_red[key] has shape (B, episode_n, ...)

    return batch_blue, batch_red