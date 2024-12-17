import numpy as np
import pickle

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

def process_batch(batch, normalizer_obs_b: Normalizer=None, normalizer_obs_r: Normalizer=None, normalizer_state: Normalizer=None):
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


    # Collect shared states
    if normalizer_state is not None:
        batch['state'] =  np.array([normalizer_state.update_normalize(e) for e in batch['state']])
        batch['next_state'] = np.array([normalizer_state.update_normalize(e) for e in batch['next_state']])

    # states_episode.append(batch['state'])          # Shape: (45, 45, 5)
    # next_states_episode.append(batch['next_state'])  # Shape: (45, 45, 5)

    # Collect data for the blue team
    if normalizer_obs_b is not None:
        batch_blue['obs'] = np.array([normalizer_obs_b.update_normalize(e['blue']) for e in batch['obs']])
        batch_blue['next_obs'] = np.array([normalizer_obs_b.update_normalize(e['blue']) for e in batch['next_obs']])
        # Collect data for the red team
    if normalizer_obs_r is not None:
        batch_red['obs'] = np.array([normalizer_obs_r.update_normalize(e['red']) for e in batch['obs']])
        batch_red['next_obs'] = np.array([normalizer_obs_r.update_normalize(e['red']) for e in batch['next_obs']])

    # Append to batch_blue
    batch_blue['actions'] = np.array([e['blue'] for e in batch['actions']])
    batch_blue['rewards'] = np.array([e['blue'] for e in batch['rewards']])
    batch_blue['dones'] = np.array([e['blue'] for e in batch['dones']])

    # Append to batch_red
    batch_red['actions'] = np.array([e['red'] for e in batch['actions']])
    batch_red['rewards'] = np.array([e['red'] for e in batch['rewards']])
    batch_red['dones'] = np.array([e['red'] for e in batch['dones']])


    # Convert states to arrays
    # states_episode = np.array(states_episode)            # Shape: (episode_n, 45, 45, 5)
    # next_states_episode = np.array(next_states_episode)   # Shape: (episode_n, 45, 45, 5)

    # Append states to both batches
    batch_blue['state'] = batch['state']
    batch_blue['next_state'] = batch['next_state']
    batch_red['state'] = batch['state']
    batch_red['next_state'] = batch['next_state']

    # # Stack data along the batch dimension for the blue team
    # for key in batch_blue:
    #     batch_blue[key] = np.stack(batch_blue[key], axis=0)
    #     # Now, batch_blue[key] has shape (B, episode_n, ...)

    # # Stack data along the batch dimension for the red team
    # for key in batch_red:
    #     batch_red[key] = np.stack(batch_red[key], axis=0)
    #     # Now, batch_red[key] has shape (B, episode_n, ...)

    return batch_blue, batch_red