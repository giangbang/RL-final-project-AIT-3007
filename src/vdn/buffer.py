import torch
import numpy as np
import collections
from utils import device

class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        """Update buffer with a new transition
        :param transition: tuple of (state, action, reward, next_state, done)
        """
        self.buffer.append(transition)

    def sample_chunk(self, batch_size, chunk_size):
        """Sample a batch of chunk_size transitions from the buffer
        :param batch_size: number of transitions to sample
        :param chunk_size: length of horizon of each batch
        :return: tuple of (states, actions, rewards, next_states, dones),
        their shapes are respectively:
        [batch_size, chunk_size, n_agents, ...obs_shape],
        [batch_size, chunk_size, n_agents],
        [batch_size, chunk_size, n_agents],
        [batch_size, chunk_size, n_agents, ...obs_shape],
        [batch_size, chunk_size, n_agents]
        """
        start_idx = np.random.randint(0, len(self.buffer) - chunk_size, batch_size)
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []

        for idx in start_idx:
            for chunk_step in range(idx, idx + chunk_size):
                # (state, action, reward, next_state, done) * num_agent
                s, a, r, s_prime, done = self.buffer[chunk_step]
                s_lst.append(s)
                a_lst.append(a)
                r_lst.append(r)
                s_prime_lst.append(s_prime)
                done_lst.append(done)
        num_agents = len(s_lst[0])
        obs_shape = s_lst[0][0].shape

        s_lst = np.array(s_lst).reshape(batch_size, chunk_size, num_agents, *obs_shape)
        a_lst = np.array(a_lst).reshape(batch_size, chunk_size, num_agents)
        r_lst = np.array(r_lst).reshape(batch_size, chunk_size, num_agents)
        s_prime_lst = np.array(s_prime_lst).reshape(batch_size, chunk_size, num_agents, *obs_shape)
        done_lst = np.array(done_lst, dtype=bool).reshape(batch_size, chunk_size, num_agents)
        return (
            torch.tensor(s_lst, dtype=torch.float32).to(device),
            torch.tensor(a_lst, dtype=torch.float32).to(device),
            torch.tensor(r_lst, dtype=torch.float32).to(device),
            torch.tensor(s_prime_lst, dtype=torch.float32).to(device),
            torch.tensor(done_lst, dtype=torch.float32).to(device)
        )

    def size(self):
        return len(self.buffer)