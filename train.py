from magent2.environments import battle_v4
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import os
# QNetwork architecture
lass QNetwork(nn.Module):
   def __init__(self, input_shape, num_actions):
       super().__init__()
       self.conv = nn.Sequential(
           nn.Conv2d(input_shape[2], 32, kernel_size=3, stride=1),
           nn.ReLU(),
           nn.Conv2d(32, 64, kernel_size=3, stride=1),
           nn.ReLU(),
           nn.Conv2d(64, 64, kernel_size=3, stride=1),
           nn.ReLU()
       )
       
       # Tính toán kích thước đầu ra của conv layers
       conv_out_size = self._get_conv_out(input_shape)
       
       self.fc = nn.Sequential(
           nn.Linear(conv_out_size, 512),
           nn.ReLU(),
           nn.Linear(512, num_actions)
       )
   
   def _get_conv_out(self, shape):
       o = self.conv(torch.zeros(1, shape[2], shape[0], shape[1]))
       return int(np.prod(o.size()))
   
   def forward(self, x):
       conv_out = self.conv(x).view(x.size()[0], -1)
       return self.fc(conv_out)
# Replay Buffer để lưu trữ experience
class ReplayBuffer:
   def __init__(self, capacity):
       self.buffer = deque(maxlen=capacity)
   
   def push(self, state, action, reward, next_state, done):
       self.buffer.append((state, action, reward, next_state, done))
   
   def sample(self, batch_size):
       return random.sample(self.buffer, batch_size)
   
   def __len__(self):
       return len(self.buffer)
def train():
   # Hyperparameters
   BUFFER_SIZE = 100000
   BATCH_SIZE = 64
   GAMMA = 0.99
   LEARNING_RATE = 1e-4
   EPISODES = 1000
   EPSILON_START = 1.0
   EPSILON_END = 0.01
   EPSILON_DECAY = 0.995
    # Khởi tạo môi trường
   env = battle_v4.env(map_size=45)
   
   # Khởi tạo mạng Q và optimizer
   q_network = QNetwork(
       env.observation_space("blue_0").shape,
       env.action_space("blue_0").n
   )
   target_network = QNetwork(
       env.observation_space("blue_0").shape,
       env.action_space("blue_0").n
   )
   target_network.load_state_dict(q_network.state_dict())
   
   optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
   replay_buffer = ReplayBuffer(BUFFER_SIZE)
   epsilon = EPSILON_START
    for episode in range(EPISODES):
       env.reset()
       episode_reward = 0
       
       for agent in env.agent_iter():
           observation, reward, termination, truncation, info = env.last()
           
           if agent.startswith("blue"):
               state = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0)
               
               # Epsilon-greedy action selection
               if random.random() < epsilon:
                   action = env.action_space(agent).sample()
               else:
                   with torch.no_grad():
                       q_values = q_network(state)
                       action = torch.argmax(q_values, dim=1).item()
               
               # Thực hiện action và lưu experience
               env.step(action)
               next_observation, next_reward, next_termination, next_truncation, next_info = env.last()
               next_state = torch.FloatTensor(next_observation).permute(2, 0, 1).unsqueeze(0)
               
               replay_buffer.push(
                   state, action, reward,
                   next_state, termination or truncation
               )
               
               episode_reward += reward
               
               # Training
               if len(replay_buffer) > BATCH_SIZE:
                   batch = replay_buffer.sample(BATCH_SIZE)
                   states, actions, rewards, next_states, dones = zip(*batch)
                   
                   states = torch.cat(states)
                   actions = torch.tensor(actions)
                   rewards = torch.tensor(rewards, dtype=torch.float32)
                   next_states = torch.cat(next_states)
                   dones = torch.tensor(dones, dtype=torch.float32)
                   
                   current_q = q_network(states).gather(1, actions.unsqueeze(1))
                   next_q = target_network(next_states).max(1)[0].detach()
                   target_q = rewards + (1 - dones) * GAMMA * next_q
                   
                   loss = nn.MSELoss()(current_q.squeeze(), target_q)
                   
                   optimizer.zero_grad()
                   loss.backward()
                   optimizer.step()
           
           else:  # Random actions for other agents
               if termination or truncation:
                   action = None
               else:
                   action = env.action_space(agent).sample()
               env.step(action)
       
       # Update target network
       if episode % 10 == 0:
           target_network.load_state_dict(q_network.state_dict())
       
       # Decay epsilon
       epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
       
       print(f"Episode {episode}, Reward: {episode_reward}, Epsilon: {epsilon:.3f}")
       
       # Save model periodically
       if episode % 100 == 0:
           torch.save(q_network.state_dict(), "blue.pt")
    env.close()
if __name__ == "__main__":
   train()