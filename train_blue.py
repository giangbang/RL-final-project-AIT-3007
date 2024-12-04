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
            # Chuyển state thành numpy array trước nếu chưa phải
            state = np.array(state)
            state_tensor = torch.from_numpy(state).float().permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = q_network(state_tensor)
            return q_values.argmax().item()

    def optimize_model():
        batch = random.sample(replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển đổi thành numpy arrays trước
        states = np.array(states)
        next_states = np.array(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)

        # Sau đó chuyển thành tensors
        states = torch.from_numpy(states).float().permute(0, 3, 1, 2).to(device)
        next_states = torch.from_numpy(next_states).float().permute(0, 3, 1, 2).to(device)
        actions = torch.from_numpy(actions).long().unsqueeze(1).to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        dones = torch.from_numpy(dones).float().to(device)

        q_values = q_network(states).gather(1, actions)
        with torch.no_grad():
            next_q_values = target_network(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = F.mse_loss(q_values.squeeze(), target_q_values)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
                        # Lựa chọn hành động cho agent blue
                        action = select_action(obs, epsilon)

                        # Lưu trữ thông tin để huấn luyện
                        next_obs = env.observe(agent)
                        replay_buffer.append((obs, action, reward, next_obs, termination or truncation))
                        if len(replay_buffer) > buffer_size:
                            replay_buffer.pop(0)

                        step_count += 1

                    env.step(action)
                    total_reward += reward
                else:
                    if termination or truncation:
                        action = None
                    else:
                        # Chính sách ngẫu nhiên cho agent red
                        action = random_policy(env, agent, obs)
                    env.step(action)

            # Tối ưu mô hình sau khi tất cả agents đã thực hiện một bước
            if len(replay_buffer) >= batch_size and step_count % train_freq == 0:
                optimize_model()

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