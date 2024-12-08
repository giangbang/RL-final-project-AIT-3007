import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from magent2.environments import battle_v4

############################################
# Hyperparameters
############################################
num_episodes = 1000
max_cycles = 300
gamma = 0.99
lambda_gae = 0.95
clip_range = 0.2
entropy_coef = 0.01
value_loss_coef = 0.5
lr = 1e-3
batch_size = 1024
epochs_per_update = 10
h_size = 64  # hidden size for networks

############################################
# Environment Setup
############################################
# Create a parallel environment
env = battle_v4.parallel_env(map_size=45, minimap_mode=False, step_reward=-0.005,
                             dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2,
                             max_cycles=max_cycles)
env.reset(seed=42)

agents = env.agents
team_red = [agent for agent in agents if 'red' in agent]
team_blue = [agent for agent in agents if 'blue' in agent]

# My intention is to train a policy for the blue team to defeat the red team.

# For simplicity, let's assume we focus on controlling the red team and treat blue as part of the environment.
# In a real scenario, you may train both sides or fix one side's policy.

obs_shape = env.observation_space(team_red[0]).shape
act_space = env.action_space(team_red[0])
num_actions = act_space.n
num_agents = len(team_red)

############################################
# Neural Network Definitions
############################################

class Actor(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(np.prod(input_dim), hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        logits = self.net(x)
        return logits


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


############################################
# Utilities for MAPPO
############################################

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = torch.zeros_like(rewards)
    last_adv = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_done = 1.0 if dones[t] else 0.0
        else:
            next_value = values[t+1]
            next_done = 1.0 if dones[t+1] else 0.0

        delta = rewards[t] + gamma * next_value * (1 - next_done) - values[t]
        advantages[t] = last_adv = delta + gamma * lam * (1 - next_done) * last_adv
    returns = advantages + values
    return advantages, returns


def rollout(env, actor, critic, steps=max_cycles):
    # Storage for one episode
    # We'll store data for all agents in team_red at each step
    obs_list = []
    actions_list = []
    log_probs_list = []
    values_list = []
    rewards_list = []
    dones_list = []

    # Reset environment
    obs, _ = env.reset()
    done = False
    for _ in range(steps):
        # Extract red team observations
        red_obs = np.array([obs[agent] for agent in team_red])
        # Flatten or reshape as needed
        red_obs_t = torch.tensor(red_obs, dtype=torch.float32)
        # Actor forward
        logits = actor(red_obs_t.view(num_agents, -1))
        dist = Categorical(logits=logits)
        actions = dist.sample()

        # Compute values from critic (centralized)
        # Centralized critic can take concatenation of all agents' observations as state
        # For simplicity, we consider the concatenation of all red agents' observations as "state"
        state_t = red_obs_t.view(1, -1)
        values = critic(state_t).detach().item()  # single value for team? 
        # NOTE: For a large number of agents, one might consider separate critics or feeding all obs
        # In this simplified example, we treat the entire team's concatenated obs as one state input.
        # Ideally, you'd use a more sophisticated method to handle the full global state (including blue).

        # Step environment with chosen actions (only supply red actions)
        action_dict = {}
        for i, agent_name in enumerate(team_red):
            action_dict[agent_name] = actions[i].item()
        # For blue team, random actions (or a fixed policy)
        for agent_name in team_blue:
            action_dict[agent_name] = env.action_space(agent_name).sample()

        next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

        # Convert termination/truncation into a done signal
        # If all red agents done or environment ended
        done = all(terminations[a] or truncations[a] for a in env.agents)
        # if chung ta cho gop. thanh` 1 done co the gay ra truong hop ma thieu' mat qua trinh truoc khi ket thuc, vi du trong truong hop
        # 2vs1, neu ta gop lai thanh 1 done khi do cac agent se khong the hoc duoc trong truong hop 2vs1 nen lam gi 

        # Store transition
        obs_list.append(red_obs_t)
        actions_list.append(actions)
        log_probs_list.append(dist.log_prob(actions).detach())
        values_list.append(torch.tensor([values], dtype=torch.float32))
        # For simplicity, sum rewards of red team agents (an alternative is to store individual)
        # red_reward = np.sum([rewards[a] for a in team_red])
        red_reward = 0
        for a in team_red:
            try:
                red_reward += rewards[a]
            except KeyError:
                continue

        rewards_list.append(torch.tensor([red_reward], dtype=torch.float32))
        dones_list.append(done)

        obs = next_obs
        if done:
            break

    # Convert collected data
    obs_list = torch.stack(obs_list)
    actions_list = torch.stack(actions_list)
    log_probs_list = torch.stack(log_probs_list)
    values_list = torch.stack(values_list)
    rewards_list = torch.stack(rewards_list)
    dones_list = torch.tensor(dones_list, dtype=torch.float32)

    return obs_list, actions_list, log_probs_list, values_list, rewards_list, dones_list


############################################
# Training Setup
############################################

actor = Actor(obs_shape, num_actions, hidden_dim=h_size)
critic = Critic(np.prod(obs_shape)*num_agents, hidden_dim=h_size)  # critic sees all red agents' obs
actor_optim = optim.Adam(actor.parameters(), lr=lr)
critic_optim = optim.Adam(critic.parameters(), lr=lr)

for episode in range(num_episodes):
    # Collect a rollout
    obs_batch, actions_batch, old_log_probs_batch, values_batch, rewards_batch, dones_batch = rollout(env, actor, critic, steps=max_cycles)

    # Compute advantages using GAE
    advantages, returns = compute_gae(rewards_batch, values_batch.squeeze(-1), dones_batch, gamma=gamma, lam=lambda_gae)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # PPO-Style Update
    for _ in range(epochs_per_update):
        # Recompute log_probs and values with current policy
        logits = actor(obs_batch.view(-1, np.prod(obs_shape)))
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions_batch)
        entropy = dist.entropy().mean()

        # Centralized critic value
        # For simplicity, we assume obs_batch stores only red agents' obs.
        # We must reconstruct the state input for the critic at each time step:
        # Since we parameter-shared among red agents, we combine their obs by concatenation:
        # NOTE: In this simplistic example, obs_batch is (T, N, obs_dim), T steps, N agents
        # We first reshape obs_batch to combine all agent obs per timestep
        T = obs_batch.shape[0]
        N = obs_batch.shape[1]
        state_batch = obs_batch.view(T, -1)  # (T, N*obs_dim)
        values_new = critic(state_batch).squeeze(-1)

        # Compute ratios for PPO
        ratio = torch.exp(new_log_probs - old_log_probs_batch.detach())
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

        # Value loss
        value_loss = ((returns - values_new)**2).mean()

        # Update actor
        actor_optim.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        actor_optim.step()

        # Update critic
        critic_optim.zero_grad()
        value_loss_coef * value_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
        critic_optim.step()

    # Simple logging
    total_reward = rewards_batch.sum().item()
    print(f"Episode {episode}/{num_episodes} - Reward: {total_reward:.2f}")


print("Training finished.")
env.close()
