import torch
import torch.nn.functional as F
import numpy np
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

def train(env, q_network, target_network, optimizer, replay_buffer, red_final_policy, device, config):
    start_time = time.time()
    time_limit = 7200  # 2 hours
    epsilon = config['epsilon_start']
    step_count = 0
    best_reward = float('-inf')
    rewards_history = []
    
    for episode in range(config['num_episodes']):
        if time.time() - start_time > time_limit:
            print(f"Time limit of 2 hours reached")
            torch.save(q_network.state_dict(), "blue_final.pt")
            return
            
        env.reset()
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
                        
                        if step_count % config['train_freq'] == 0:
                            optimize_model()
                    else:
                        action = red_final_policy(obs)

                env.step(action)

        # Update epsilon
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])
        
        # Update target network
        if step_count % config['target_update_freq'] == 0:
            target_network.load_state_dict(q_network.state_dict())
            
        # Track best model
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(q_network.state_dict(), "blue_best.pt")
            
        rewards_history.append(total_reward)
        
        # Logging
        if episode % 10 == 0:
            elapsed_time = time.time() - start_time
            avg_reward = np.mean(rewards_history[-10:])
            print(f"Episode {episode}/{config['num_episodes']}")
            print(f"Time: {elapsed_time/3600:.1f}h")
            print(f"Avg Reward: {avg_reward:.2f}")
            print(f"Epsilon: {epsilon:.4f}")
            print(f"Buffer size: {len(replay_buffer)}")
            print("-" * 50)
            
        # Save checkpoint
        if (episode + 1) % 100 == 0:
            torch.save(q_network.state_dict(), f"blue_checkpoint_{episode+1}.pt")

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

    # Move red_final_policy definition outside
    red_final_policy = lambda obs: get_action(obs, red_final_network)

    # Main training loop
    train(env, q_network, target_network, optimizer, replay_buffer, red_final_policy, device, config)

    torch.save(q_network.state_dict(), "blue.pt")
    print("Training complete. Model saved as 'blue.pt'")

if __name__ == "__main__":
    main()
