import numpy as np
from typing import List, Dict

class ReplayBuffer:
    """
    Replay Buffer that stores episodes separately for multi-agent environments.
    """

    def __init__(self, memory_size: int, field_names: List[str]):
        """
        Initialize the replay buffer.

        Args:
            memory_size (int): Maximum number of episodes to store.
            field_names (List[str]): List of fields to store in each transition.
        """
        self.memory_size = memory_size
        self.episodes = []  # List to store episodes
        self.field_names = field_names

    def save_episode(self, episode: List[Dict[str, any]]):
        """
        Save an entire episode to the replay buffer.

        Args:
            episode (List[Dict[str, any]]): List of transitions in the episode.
        """
        if len(self.episodes) >= self.memory_size:
            self.episodes.pop(0)  # Remove oldest episode
        self.episodes.append(episode)

    def sample(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=False)
        sampled_episodes = [self.episodes[idx] for idx in indices]
        return sampled_episodes

    def __len__(self) -> int:
        return len(self.episodes)
