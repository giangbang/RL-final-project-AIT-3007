import torch
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from magent2.environments import battle_v4
from torch_model import QNetwork
import torch.optim as optim

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

def train():
    # Initialize environment and devices
    env = initialize_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize Q-networks
    q_network = initialize_q_network(env, device)
    target_network = initialize_q_network(env, device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    # Optimizer and Replay Buffer
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    replay_buffer = ReplayBuffer(capacity=10000)

    # Hyperparameters
    batch_size = 64
    gamma = 0.99
    target_update_freq = 1000
    train_freq = 4
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    num_episodes = 500
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
        if len(replay_buffer) < batch_size:
            return
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states).float().permute(0, 3, 1, 2).to(device)
        next_states = torch.tensor(next_states).float().permute(0, 3, 1, 2).to(device)
        actions = torch.tensor(actions).long().unsqueeze(1).to(device)
        rewards = torch.tensor(rewards).float().to(device)
        dones = torch.tensor(dones).float().to(device)

        q_values = q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values.squeeze(), target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Main training loop
    for episode in range(num_episodes):
        env.reset()
        observations = {agent: env.observe(agent) for agent in env.agents if agent.startswith("blue")}
        total_reward = 0
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            for agent in env.agent_iter():
                obs, reward, termination, truncation, _ = env.last()
                agent_team = agent.split("_")[0]

                if termination or truncation:
                    action = None
                    done[agent] = True
                else:
                    if agent_team == "blue":
                        action = select_action(obs, epsilon)
                        next_obs = env.observe(agent)
                        replay_buffer.add(obs, action, reward, next_obs, termination or truncation)
                        step_count += 1
                        total_reward += reward
                    else:
                        action = env.action_space(agent).sample()

                env.step(action)

            if step_count % train_freq == 0:
                optimize_model()

        # Update epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Sync target network
        if step_count % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

        if (episode + 1) % 100 == 0:
            checkpoint_path = f"blue_checkpoint_{episode + 1}.pt"
            torch.save(q_network.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at episode {episode + 1}")

    torch.save(q_network.state_dict(), "blue.pt")
    print("Training complete. Model saved as 'blue.pt'")

if __name__ == "__main__":
    train()
