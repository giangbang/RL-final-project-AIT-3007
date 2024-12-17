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


import numpy as np
from typing import List

class PrioritizedReplayBuffer:
    def __init__(self, memory_size: int, field_names: List[str], alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer
        
        Args:
            memory_size: Maximum size of buffer
            field_names: Names of fields to store in transitions
            alpha: How much prioritization to use (0 = uniform, 1 = full priority)
            beta: Importance sampling correction factor
        """
        self.memory_size = memory_size
        self.field_names = field_names
        self.alpha = alpha
        self.beta = beta
        
        # Initialize storage buffers
        self.buffer = {name: [] for name in field_names}
        self.priorities = np.zeros(memory_size, dtype=np.float32)
        self.position = 0
        self.size = 0

    def save_transition(self, transition):
        """Save a transition with maximum priority"""
        # Set max priority for new transitions
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        # Store transition
        for key, value in transition.items():
            if self.size < self.memory_size:
                self.buffer[key].append(value)
            else:
                self.buffer[key][self.position] = value
                
        # Store priority
        self.priorities[self.position] = max_prio
        
        self.position = (self.position + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self, batch_size: int):
        if self.size == 0:
            raise ValueError("Cannot sample from empty buffer")

        # Clip and normalize the priorities to avoid issues with large values
        max_priority = 1e5  # Set a maximum threshold for priority
        clipped_priorities = np.clip(self.priorities[:self.size], 1e-6, max_priority)  # Prevent very large priorities
        
        # Normalize priorities
        probs = clipped_priorities / np.sum(clipped_priorities)
        
        # Handle NaN or inf values
        probs = np.nan_to_num(probs, nan=1.0)  # Replace NaNs with 1.0 (or a reasonable default)
        probs = np.clip(probs, 1e-8, 1.0)  # Ensure no zero or extremely small probabilities

        # If the sum of probabilities is zero (can happen with very small values), fall back to uniform sampling
        if np.sum(probs) == 0:
            print("Warning: Probabilities sum to 0, using uniform sampling instead.")
            probs = np.ones_like(probs) / len(probs)

        # Calculate weighted probabilities
        probs = probs ** self.alpha  # Apply prioritization factor
        probs = np.nan_to_num(probs, nan=1.0)  # Replace NaNs with 1.0
        probs = np.clip(probs, 1e-8, 1.0)  # Ensure probabilities are in a valid range

        # Normalize again after applying alpha
        probs = probs / np.sum(probs)
        
        # Ensure probabilities sum to 1
        assert np.abs(np.sum(probs) - 1.0) < 1e-5, "Probabilities do not sum to 1"

        # Sample transitions based on the probabilities
        indices = np.random.choice(self.size, batch_size, p=probs)

        # Calculate importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = np.nan_to_num(weights, nan=1.0)  # Replace NaNs with 1.0
        weights = np.clip(weights, 1e-8, None)  # Ensure weights are within a valid range
        weights = weights / weights.max()  # Normalize weights

        # Return the sampled batch of transitions
        batch = {
            key: [self.buffer[key][i] for i in indices]
            for key in self.field_names
        }

        return batch, indices, weights

    def __len__(self):
        return self.size


    def update_priorities(self, indices: List[int], priorities: float):
        """
        Update the priorities of specific episodes.

        Args:
            indices (List[int]): List of indices of the episodes to update.
            priorities (List[float]): New priority values for the episodes.
        """
        for idx, prio in zip(indices, priorities):
            # Ensure priorities are positive and finite
            prio = max(prio, 1e-6)
            if not np.isfinite(prio):
                prio = 1e-6
            self.priorities[idx] = prio + 1e-6  # Adding a small constant to prevent zero priority
