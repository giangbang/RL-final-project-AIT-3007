import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from torch_model import QNetwork
from magent2.environments import battle_v4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay Buffer để lưu trữ experience
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Training function
def train_blue_agent(episodes=10, batch_size=128, gamma=0.99, epsilon_start=1.0, 
                     epsilon_end=0.01, epsilon_decay=0.995):
    env = battle_v4.env(map_size=45, render_mode=None)
    
    # Khởi tạo networks
    q_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    target_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    optimizer = optim.Adam(q_network.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    replay_buffer = ReplayBuffer()
    epsilon = epsilon_start
    
    for episode in range(episodes):
        env.reset()
        episode_reward = 0
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            if agent.startswith("blue"):
                # Kiểm tra termination/truncation trước khi quyết định action
                if termination or truncation:
                    action = None
                else:
                    # Chuyển observation sang tensor
                    state = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0).to(device)
                    
                    # Epsilon-greedy action selection
                    if random.random() < epsilon:
                        action = env.action_space(agent).sample()
                    else:
                        with torch.no_grad():
                            q_values = q_network(state)
                            action = torch.argmax(q_values, dim=1).item()
                
                # Thực hiện action
                env.step(action)
                
                # Lấy next state
                if not (termination or truncation):
                    next_observation, _, _, _, _ = env.last()
                    next_state = torch.FloatTensor(next_observation).permute(2, 0, 1).unsqueeze(0).to(device)
                    done = False
                else:
                    next_state = state  # Dummy value khi episode kết thúc
                    action = None
                    done = True
                
                # Lưu experience vào replay buffer
                if not done:
                    replay_buffer.push(state.cpu(), action, reward, next_state.cpu(), done)
                
                # Training khi có đủ samples
                if len(replay_buffer) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    states, actions, rewards, next_states, dones = zip(*batch)
                    
                    states = torch.cat(states).to(device)
                    actions = torch.LongTensor(actions).to(device)
                    rewards = torch.FloatTensor(rewards).to(device)
                    next_states = torch.cat(next_states).to(device)
                    dones = torch.FloatTensor(dones).to(device)
                    
                    # Tính current Q values
                    current_q_values = q_network(states).gather(1, actions.unsqueeze(1))
                    
                    # Tính target Q values
                    with torch.no_grad():
                        next_q_values = target_network(next_states).max(1)[0]
                        target_q_values = rewards + gamma * next_q_values * (1 - dones)
                    
                    # Update network
                    loss = criterion(current_q_values.squeeze(), target_q_values)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                episode_reward += reward
            else:
                # Random action cho red agents
                if termination or truncation:
                    action = None
                else:
                    action = env.action_space(agent).sample()
                env.step(action)
        
        # Update target network
        if episode % 1 == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
        
        # Lưu model
        if episode % 1 == 0:
            torch.save(q_network.state_dict(), f"blue_agent_episode_{episode}.pt")
    
    env.close()
    return q_network


# Train agent
import time
start_time = time.time()

trained_network = train_blue_agent()
torch.save(trained_network.state_dict(), "blue.pt")

end_time = time.time()
print(f"Training took {end_time - start_time:.2f} seconds")