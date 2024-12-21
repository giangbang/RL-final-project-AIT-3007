import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Dict, Tuple, Optional
from pathlib import Path
import logging
from magent2.environments import battle_v4
from torch_model import QNetwork
from torch.utils.data import Dataset, DataLoader

@dataclass
class Config:
    """Configuration parameters"""
    batch_size: int = 512
    gamma: float = 0.99
    target_update_freq: int = 1000
    train_freq: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: float = 0.995
    num_episodes: int = 500
    buffer_capacity: int = 10000
    learning_rate: float = 1e-4
    checkpoint_freq: int = 100
    map_size: int = 45
    max_cycles: int = 300

class ReplayBuffer(Dataset):
    """Experience replay buffer implementation"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        state, action, reward, next_state, done = self.buffer[idx]
        return (
            torch.FloatTensor(state).permute(2, 0, 1),
            torch.LongTensor([action]),
            torch.FloatTensor([reward]),
            torch.FloatTensor(next_state).permute(2, 0, 1),
            torch.FloatTensor([done])
        )

class DQNAgent:
    """DQN Agent implementation"""
    def __init__(self, config: Config):
        logging.info("Initializing DQNAgent...")
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        self.env = battle_v4.env(map_size=config.map_size, max_cycles=config.max_cycles)
        
        # Initialize networks
        self.q_network = QNetwork(
            self.env.observation_space("blue_0").shape,
            self.env.action_space("blue_0").n
        ).to(self.device)
        
        self.target_network = QNetwork(
            self.env.observation_space("blue_0").shape,
            self.env.action_space("blue_0").n
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.learning_rate)
        self.replay_buffer = ReplayBuffer(config.buffer_capacity)
        self.epsilon = config.epsilon_start
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.rand() < self.epsilon:
            return self.env.action_space("blue_0").sample()
        
        state_tensor = torch.tensor(state).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
            
        train_loader = DataLoader(
            self.replay_buffer,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        for states, actions, rewards, next_states, dones in train_loader:
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            dones = dones.to(self.device)

            q_values = self.q_network(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)

            loss = F.mse_loss(q_values, target_q_values)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self):
        """Main training loop"""
        logging.info("Starting training...")
        print("Training started...")  # Immediate console feedback
        
        for episode in range(self.config.num_episodes):
            self.env.reset()
            total_reward = 0
            done = {agent: False for agent in self.env.agents}

            while not all(done.values()):
                for agent in self.env.agent_iter():
                    obs, reward, termination, truncation, _ = self.env.last()
                    agent_team = agent.split("_")[0]

                    if termination or truncation:
                        action = None
                        done[agent] = True
                    else:
                        if agent_team == "blue":
                            action = self.select_action(obs)
                            next_obs = self.env.observe(agent)
                            self.replay_buffer.add(obs, action, reward, next_obs, termination or truncation)
                            self.step_count += 1
                            total_reward += reward
                        else:
                            action = self.env.action_space(agent).sample()

                    self.env.step(action)

                if self.step_count % self.config.train_freq == 0:
                    self.optimize_model()

            self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

            if self.step_count % self.config.target_update_freq == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            print(f"Episode {episode + 1}/{self.config.num_episodes}, "
                  f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

            if (episode + 1) % self.config.checkpoint_freq == 0:
                self.save_checkpoint(episode + 1)

        self.save_model()

    def save_checkpoint(self, episode: int):
        checkpoint_path = f"blue_checkpoint_{episode}.pt"
        torch.save(self.q_network.state_dict(), checkpoint_path)
        print(f"Model checkpoint saved at episode {episode}")

    def save_model(self):
        weights_dir = Path("weight_models")
        
        # Save model
        model_path = weights_dir / "blue.pt"
        torch.save(self.q_network.state_dict(), str(model_path))
        print(f"Training complete. Model saved as '{model_path}'")

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    print("Starting program...")
    setup_logging()
    config = Config()
    agent = DQNAgent(config)
    agent.train()

if __name__ == "__main__":
    main()