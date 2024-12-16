import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from magent2.environments import battle_v4
from torch_model import QNetwork
import torch.optim as optim
from final_torch_model import QNetwork as FinalQNetwork
import time

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

def initialize_environment():
    return battle_v4.env(map_size=45, max_cycles=300)

def initialize_q_network(env, device):
    obs_space = env.observation_space("blue_0").shape
    act_space = env.action_space("blue_0").n
    return QNetwork(obs_space, act_space).to(device)

def preprocess_state(state, device):
    state = np.array(state)
    return torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(device)

def train(env, q_network, target_network, optimizer, replay_buffer, device, config):
    start_time = time.time()
    time_limit = 7200  # 2 hours in seconds
    
    for episode in range(config['num_episodes']):
        # Check time limit
        if time.time() - start_time > time_limit:
            print(f"Time limit of 2 hours reached")
            torch.save(q_network.state_dict(), "blue_final.pt")
            print("Training stopped. Model saved as 'blue_final.pt'")
            return
            
        # Training loop
        env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            ...existing code...
            
        # Logging with time
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            print(f"Episode {episode}, Time: {elapsed_time/3600:.1f}h")
            print(f"Reward: {episode_reward:.2f}")
            print("-" * 50)

def main():
    # Initialize environment and devices
    env = initialize_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Q-networks
    q_network = initialize_q_network(env, device)
    target_network = initialize_q_network(env, device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    red_final_network = FinalQNetwork(
        env.observation_space("red_0").shape, 
        env.action_space("red_0").n
    ).to(device)
    red_final_network.load_state_dict(
        torch.load("red_final.pt", map_location=device)
    )
    red_final_network.eval()

    def red_final_policy(obs):
        observation = torch.tensor(obs, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_final_network(observation)
        return q_values.argmax().item()

    # Optimizer and Replay Buffer
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=10000)

    # Hyperparameters
    config = {
        'batch_size': 64,
        'gamma': 0.99,
        'target_update_freq': 1000,
        'train_freq': 4,
        'epsilon_start': 1.0,
        'epsilon_end': 0.1,
        'epsilon_decay': 0.995,
        'num_episodes': 800
    }
    epsilon = config['epsilon_start']
    step_count = 0

    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space("blue_0").sample()
        else:
            state_tensor = preprocess_state(state, device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()

    def optimize_model():
        if len(replay_buffer) < config['batch_size']:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(config['batch_size'])

        states = torch.tensor(states).float().permute(0, 3, 1, 2).to(device)
        next_states = torch.tensor(next_states).float().permute(0, 3, 1, 2).to(device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        q_values = q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + config['gamma'] * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values.squeeze(), target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Main training loop
    train(env, q_network, target_network, optimizer, replay_buffer, device, config)

    torch.save(q_network.state_dict(), "blue.pt")
    print("Training complete. Model saved as 'blue.pt'")

if __name__ == "__main__":
    main()
