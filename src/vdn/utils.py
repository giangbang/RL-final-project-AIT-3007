import torch
import numpy as np
import random
from dataclasses import dataclass

# Init
seed = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class VdnHyperparameters:
    lr: float = 0.001
    gamma: float = 0.99
    batch_size: int = 2048
    update_iter: int = 20
    buffer_limit: int = 9000
    update_target_interval: int = 20
    max_episodes: int = 500
    max_epsilon: float = 0.9
    min_epsilon: float = 0.1
    episode_min_epsilon: int = 200
    test_episodes: int = 10
    warm_up_steps: int = 3000
    chunk_size: int = 1
    recurrent: bool = False

def save_model(model, name):
    torch.save(model.state_dict(), f'{name}.pth')

def save_data(data, name='data'):
    np.save(f'{name}.npy', data)

def reseed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_output_dim(input_dim, kernel_size, stride, padding):
    return (input_dim - kernel_size + 2 * padding) // stride + 1