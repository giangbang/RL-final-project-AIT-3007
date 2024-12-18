import numpy as np
import pickle
import torch
from tensordict.tensordict import TensorDict
from typing import List

class Normalizer:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0

    def update(self, data):
        batch_mean = np.mean(data, axis=0)
        batch_var = np.var(data, axis=0)
        batch_count = data.shape[0]

        self.mean, self.var, self.count = self._update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, data):
        return (data - self.mean) / (np.sqrt(self.var) + 1e-8)

    @staticmethod
    def _update_mean_var_count(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def update_normalize(self, data):
        self.update(data)
        return self.normalize(data)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'mean': self.mean, 'var': self.var, 'count': self.count}, f)

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.mean = data['mean']
            self.var = data['var']
            self.count = data['count']



def process_batch(batchs, normalizer_obs_b: Normalizer=None, normalizer_obs_r: Normalizer=None, normalizer_state: Normalizer=None):
    # Initialize dictionaries to hold lists for each field
    batch_blue = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'state': [],
        'next_state': [],
        'dones': [],
        'prev_actions': []
    }
    batch_red = {
        'obs': [],
        'actions': [],
        'rewards': [],
        'next_obs': [],
        'state': [],
        'next_state': [],
        'dones': [],
        'prev_actions': []  
    }

    max_length_blue = 0
    max_length_red = 0

    for episode in batchs:
        # Initialize lists to collect per-episode data
        obs_blue_episode = []
        actions_blue_episode = []
        rewards_blue_episode = []
        next_obs_blue_episode = []
        dones_blue_episode = []
        prev_actions_blue_episode = []

        obs_red_episode = []
        actions_red_episode = []
        rewards_red_episode = []
        next_obs_red_episode = []
        dones_red_episode = []
        prev_actions_red_episode = []

        states_episode = []
        next_states_episode = []

        for transition in episode:
            # Collect shared states
            if normalizer_state is not None:
                transition['state'] = normalizer_state.update_normalize(transition['state'])
                transition['next_state'] = normalizer_state.update_normalize(transition['next_state'])

            states_episode.append(transition['state'])          # Shape: (45, 45, 5)
            next_states_episode.append(transition['next_state'])  # Shape: (45, 45, 5)

            # Collect data for the blue team
            if normalizer_obs_b is not None:
                transition['obs']['blue'] = normalizer_obs_b.update_normalize(transition['obs']['blue'])
                transition['next_obs']['blue'] = normalizer_obs_b.update_normalize(transition['next_obs']['blue'])
            obs_blue_episode.append(transition['obs']['blue'])        # Shape: (transitions, N_blue, 13, 13, 5)
            actions_blue_episode.append(transition['actions']['blue'])   # Shape: (transitions, N_blue)
            rewards_blue_episode.append(transition['rewards']['blue'])   # Shape: (transitions, 1)
            next_obs_blue_episode.append(transition['next_obs']['blue']) # Shape: (transitions, N_blue, 13, 13, 5)
            dones_blue_episode.append(transition['dones']['blue'])      # Shape: (transitions, 1)
            prev_actions_blue_episode.append(transition['prev_actions']['blue'])

            # Collect data for the red team
            if normalizer_obs_r is not None:
                transition['obs']['red'] = normalizer_obs_r.update_normalize(transition['obs']['red'])
                transition['next_obs']['red'] = normalizer_obs_r.update_normalize(transition['next_obs']['red'])
            obs_red_episode.append(transition['obs']['red'])
            actions_red_episode.append(transition['actions']['red'])
            rewards_red_episode.append(transition['rewards']['red'])
            next_obs_red_episode.append(transition['next_obs']['red'])
            dones_red_episode.append(transition['dones']['red'])
            prev_actions_red_episode.append(transition['prev_actions']['red'])

        # Update maximum lengths
        max_length_blue = max(max_length_blue, len(obs_blue_episode))
        max_length_red = max(max_length_red, len(obs_red_episode))

        # Convert lists to numpy arrays for the blue team
        obs_blue_episode = np.array(obs_blue_episode)            # Shape: (transitions, N_blue, 13, 13, 5)
        actions_blue_episode = np.array(actions_blue_episode)    # Shape: (transitions, N_blue)
        rewards_blue_episode = np.array(rewards_blue_episode)    # Shape: (transitions, 1)
        next_obs_blue_episode = np.array(next_obs_blue_episode)  # Shape: (transitions, N_blue, 13, 13, 5)
        dones_blue_episode = np.array(dones_blue_episode)        # Shape: (transitions, 1)
        prev_actions_blue_episode = np.array(prev_actions_blue_episode)

        # Append to batch_blue
        batch_blue['obs'].append(obs_blue_episode) 
        batch_blue['actions'].append(actions_blue_episode)
        batch_blue['rewards'].append(rewards_blue_episode)
        batch_blue['next_obs'].append(next_obs_blue_episode)
        batch_blue['dones'].append(dones_blue_episode)
        batch_blue['prev_actions'].append(prev_actions_blue_episode)

        # Convert lists to numpy arrays for the red team
        obs_red_episode = np.array(obs_red_episode)
        actions_red_episode = np.array(actions_red_episode)
        rewards_red_episode = np.array(rewards_red_episode)
        next_obs_red_episode = np.array(next_obs_red_episode)
        dones_red_episode = np.array(dones_red_episode)
        prev_actions_red_episode = np.array(prev_actions_red_episode)

        # Append to batch_red
        batch_red['obs'].append(obs_red_episode)
        batch_red['actions'].append(actions_red_episode)
        batch_red['rewards'].append(rewards_red_episode)
        batch_red['next_obs'].append(next_obs_red_episode)
        batch_red['dones'].append(dones_red_episode)
        batch_red['prev_actions'].append(prev_actions_red_episode)

        # Convert states to arrays
        states_episode = np.array(states_episode)            # Shape: (episode_n, 45, 45, 5)
        next_states_episode = np.array(next_states_episode)   # Shape: (episode_n, 45, 45, 5)

        # Append states to both batches
        batch_blue['state'].append(states_episode)
        batch_blue['next_state'].append(next_states_episode)
        batch_red['state'].append(states_episode)
        batch_red['next_state'].append(next_states_episode)

    # Pad sequences and create masks for the blue team
    for key in ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'state', 'next_state', 'prev_actions']:
        max_len = max_length_blue
        padded = []
        masks = []
        for seq in batch_blue[key]:
            pad_length = max_len - len(seq)
            if pad_length > 0:
                padding_shape = (pad_length,) + seq.shape[1:]
                padding = np.zeros(padding_shape, dtype=seq.dtype)
                padded_seq = np.concatenate([seq, padding], axis=0)
                # mask = np.concatenate([np.ones(len(seq)), np.zeros(pad_length)])
            else:
                padded_seq = seq
                mask = np.ones(len(seq))
            padded.append(padded_seq)
            # masks.append(mask)
        batch_blue[key] = np.stack(padded, axis=0)
        # batch_blue[f'{key}_mask'] = np.stack(masks, axis=0)

    # Pad sequences and create masks for the red team
    for key in ['obs', 'actions', 'rewards', 'next_obs', 'dones', 'state', 'next_state', 'prev_actions']:
        max_len = max_length_red
        padded = []
        # masks = []
        for seq in batch_red[key]:
            pad_length = max_len - len(seq)
            if pad_length > 0:
                padding_shape = (pad_length,) + seq.shape[1:]
                padding = np.zeros(padding_shape, dtype=seq.dtype)
                padded_seq = np.concatenate([seq, padding], axis=0)
                # mask = np.concatenate([np.ones(len(seq)), np.zeros(pad_length)])
            else:
                padded_seq = seq
                # mask = np.ones(len(seq))
            padded.append(padded_seq)
            # masks.append(mask)
        batch_red[key] = np.stack(padded, axis=0)
        # batch_red[f'{key}_mask'] = np.stack(masks, axis=0)

    return batch_blue, batch_red

def collate_episodes(episodes: List[TensorDict], device: torch.device):
    """
    Collate a list of episodes into batched TensorDict with padding.

    Args:
        episodes (List[TensorDict]): List of episodes to collate.
        device (torch.device): Device to place tensors on.

    Returns:
        TensorDict: Batched episodes with padding and masks.
    """
    # Determine the maximum episode length in the batch
    max_length = max(episode["o_b"].shape[0] for episode in episodes)

    batched = {}
    masks = torch.zeros(len(episodes), max_length, dtype=torch.bool)

    for key in episodes[0].keys():
        # Collect all steps for this key
        key_data = [episode[key] for episode in episodes]
        # Determine the shape excluding time
        shape = key_data[0].shape[1:]
        # Initialize batched tensor with padding
        if key_data[0].dtype == torch.bool:
            pad_value = False
        elif key_data[0].dtype == torch.float32:
            pad_value = 0.0
        else:
            pad_value = 0

        batched[key] = torch.zeros(len(episodes), max_length, *shape)
        batched[key].fill_(pad_value)

        for i, episode in enumerate(key_data):
            length = episode.shape[0]
            batched[key][i, :length] = episode
            masks[i, :length] = 1  # Mark valid steps

    batched_tensordict = TensorDict(batched, batch_size=[len(episodes)])
    batched_tensordict.set("mask", masks)
    return batched_tensordict

def process_episodes(batch):
    batch_blue = {
        'o' : batch['o_b'],
        'a' : batch['a_b'],
        'r' : batch['r_b'],
        'n_o' : batch['n_o_b'],
        'd' : batch['do'],
        's' : batch['s'],
        'n_s' : batch['n_s'],
        'p_a' : batch['p_a_b']
    }
    batch_red = {
        'o' : batch['o_r'],
        'a' : batch['a_r'],
        'r' : batch['r_r'],
        'n_o' : batch['n_o_r'],
        'd' : batch['do'],
        's' : batch['s'],
        'n_s' : batch['n_s'],
        'p_a' : batch['p_a_r']
    }
    return batch_blue, batch_red