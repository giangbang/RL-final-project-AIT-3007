from magent2.environments import battle_v4
from torch_model import QNetwork
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import random

def train():
    env = battle_v4.env(map_size=45, max_cycles=300)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    q_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    target_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)
    replay_buffer = []
    buffer_size = 10000
    batch_size = 64
    gamma = 0.99
    target_update_freq = 1000
    train_freq = 4

    def random_policy(env, agent, obs):
        return env.action_space(agent).sample()

    def select_action(state, epsilon):
        if np.random.rand() < epsilon:
            return env.action_space("blue_0").sample()
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()

    num_episodes = 10
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    epsilon = epsilon_start
    step_count = 0

    for episode in range(num_episodes):
        env.reset()
        observations = {agent: env.observe(agent) for agent in env.agents if agent.startswith("blue")}
        total_reward = 0
        done = {agent: False for agent in env.agents}

        while not all(done.values()):
            for agent in env.agent_iter():
                obs, reward, termination, truncation, _ = env.last()
                agent_team = agent.split("_")[0]

                if agent_team == "blue":
                    if termination or truncation:
                        action = None
                        done[agent] = True
                    else:
                        state = observations[agent]
                        action = select_action(state, epsilon)
                        env.step(action)
                        next_state = env.observe(agent)
                        total_reward += reward

                        replay_buffer.append((state, action, reward, next_state, termination or truncation))
                        if len(replay_buffer) > buffer_size:
                            replay_buffer.pop(0)
                        observations[agent] = next_state

                        if len(replay_buffer) >= batch_size and step_count % train_freq == 0:
                            # Optimize model
                            batch = random.sample(replay_buffer, batch_size)
                            states, actions, rewards, next_states, dones = zip(*batch)

                            states = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                            actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
                            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
                            next_states = torch.tensor(next_states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
                            dones = torch.tensor(dones, dtype=torch.float32).to(device)

                            q_values = q_network(states).gather(1, actions)
                            with torch.no_grad():
                                next_q_values = target_network(next_states).max(1)[0]
                            target_q_values = rewards + gamma * next_q_values * (1 - dones)

                            loss = F.mse_loss(q_values.squeeze(), target_q_values)
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        if step_count % target_update_freq == 0:
                            target_network.load_state_dict(q_network.state_dict())

                        step_count += 1
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)
                else:
                    if termination or truncation:
                        action = None
                    else:
                        action = random_policy(env, agent, obs)
                    env.step(action)

        # Hiển thị thông tin sau mỗi tập
        print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}")

        # Lưu mô hình sau mỗi 100 tập
        if (episode + 1) % 100 == 0:
            checkpoint_path = f"blue_checkpoint_{episode + 1}.pt"
            torch.save(q_network.state_dict(), checkpoint_path)
            print(f"Model checkpoint saved at episode {episode + 1}")

    # Lưu mô hình cuối cùng
    torch.save(q_network.state_dict(), "blue.pt")
    print("Training complete. Model saved as 'blue.pt'")

if __name__ == "__main__":
    train()