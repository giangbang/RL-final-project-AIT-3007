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
        self.field_names = field_names
        self.episodes = []  # List to store episodes

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

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, memory_size: int, field_names: List[str], alpha: float = 0.6):
        super().__init__(memory_size, field_names)
        self.priorities = np.zeros((memory_size,), dtype=np.float32)
        self.alpha = alpha
        self.next_idx = 0

    def save_episode(self, episode: List[Dict[str, any]]):
        max_prio = self.priorities.max() if self.episodes else 1.0
        if len(self.episodes) < self.memory_size:
            self.episodes.append(episode)
        else:
            self.episodes[self.next_idx] = episode
        self.priorities[self.next_idx] = max_prio
        self.next_idx = (self.next_idx + 1) % self.memory_size

    def sample(self, batch_size: int) -> List[Dict[str, np.ndarray]]:
        if len(self.episodes) == self.memory_size:
            probs = self.priorities ** self.alpha
        else:
            probs = self.priorities[:self.next_idx] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.episodes), size=batch_size, replace=False, p=probs)
        sampled_episodes = [self.episodes[idx] for idx in indices]
        return sampled_episodes, indices

    def update_priorities(self, indices: List[int], priorities: List[float]):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio