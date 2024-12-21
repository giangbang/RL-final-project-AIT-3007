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
import time
import sys

@dataclass
class Config:
    """Configuration parameters"""
    batch_size: int = 512
    gamma: float = 0.99
    target_update_freq: int = 1000
    train_freq: int = 4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.99
    num_episodes: int = 500
    buffer_capacity: int = 20000
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
        logging.debug("Initializing DQNAgent...")
        try:
            self.config = config
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logging.info(f"Using device: {self.device}")
            
            self.env = battle_v4.env(map_size=config.map_size, max_cycles=config.max_cycles)
            logging.debug("Environment created successfully")
            
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
            self.replay_buffer = ReplayBuffer(capacity=config.buffer_capacity)
            self.epsilon = config.epsilon_start
            self.step_count = 0

        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise

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
            
        try:
            train_loader = DataLoader(
                self.replay_buffer,
                batch_size=min(self.config.batch_size, len(self.replay_buffer)),
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
                logging.debug(f"Training batch processed, buffer size: {len(self.replay_buffer)}")
                
        except Exception as e:
            logging.error(f"Error during optimization: {str(e)}")
            raise

    def train(self):
        """Main training loop"""
        logging.info("Starting training loop...")
        try:
            for episode in range(self.config.num_episodes):
                logging.debug(f"Starting episode {episode + 1}")
                self.env.reset()
                total_reward = 0
                done = {agent: False for agent in self.env.agents}
                observations = {}
                episode_start_time = time.time()

                while not all(done.values()):
                    for agent in self.env.agent_iter():
                        obs, reward, termination, truncation, _ = self.env.last()
                        agent_team = agent.split("_")[0]
                        
                        if agent not in observations:
                            observations[agent] = obs
                        next_obs = observations.get(agent, obs)

                        if termination or truncation:
                            action = None
                            done[agent] = True
                        else:
                            if agent_team == "blue":
                                action = self.select_action(obs)
                                self.replay_buffer.add(obs, action, reward, next_obs, termination or truncation)
                                observations[agent] = next_obs
                                self.step_count += 1
                                total_reward += reward
                            else:
                                action = self.env.action_space("red_0").sample()

                        self.env.step(action)

                    if self.step_count % self.config.train_freq == 0:
                        self.optimize_model()

                # Update epsilon
                self.epsilon = max(self.config.epsilon_end, self.epsilon * self.config.epsilon_decay)

                # Update target network
                if self.step_count % self.config.target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                episode_duration = time.time() - episode_start_time
                logging.info(f"Episode {episode + 1} completed in {episode_duration:.2f}s, "
                           f"Total Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.4f}")

                # Save checkpoint
                if (episode + 1) % self.config.checkpoint_freq == 0:
                    self.save_checkpoint(episode + 1)

            self.save_model()
                
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint_path = f"blue_vs_final_checkpoint_{episode}.pt"
        torch.save(self.q_network.state_dict(), checkpoint_path)
        logging.info(f"Model checkpoint saved at episode {episode}")

    def save_model(self):
        """Save final model"""
        torch.save(self.q_network.state_dict(), "blue_vs_random.pt")
        logging.info("Training complete. Model saved as 'blue_vs_random.pt'")

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.DEBUG,  # Changed to DEBUG level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def main():
    """Main entry point"""
    print("Starting program...")  # Immediate console feedback
    try:
        setup_logging()
        logging.info("Logging setup complete")
        
        config = Config()
        logging.info("Configuration loaded")
        
        agent = DQNAgent(config)
        logging.info("Agent initialized successfully")
        
        agent.train()
        
    except Exception as e:
        logging.error(f"Program failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()