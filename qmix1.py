import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from magent2.environments import battle_v4
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
import wandb

from typing import List

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Q-Network for Agents
# -------------------------
class AgentQNetwork(nn.Module):
    """
    This network takes agent observations as input and produces Q-values for each action.
    """
    def __init__(self, num_actions=21):
        super().__init__()
        # Input: (13,13,5)
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(32*13*13, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, obs):
        # obs shape: (batch, N_agents, 13,13,5)
        # We'll reshape to feed each agent separately if needed
        B, N, H, W, C = obs.shape
        obs = obs.permute(0,1,4,2,3).contiguous() # (B, N, C, H, W)
        obs = obs.view(B*N, C, H, W)
        x = self.conv(obs)
        x = x.view(x.size(0), -1) # (B*N, 32*13*13)
        q = self.fc(x) # (B*N, num_actions)
        q = q.view(B, N, -1) # (B, N, num_actions)
        return q

# -------------------------
# Mixing Network
# -------------------------
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_shape=(45,45,5), embed_dim=32):
        """
        This network takes agent Q-values and global state as input
        and produces a single Q-value for the team.
        """
        super().__init__()
        # We'll encode the state with a CNN or a small network
        self.conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # After pooling: 16 dims
        self.hyper_w_1 = nn.Linear(16, num_agents*embed_dim)
        self.hyper_b_1 = nn.Linear(16, embed_dim)

        self.hyper_w_final = nn.Linear(16, embed_dim)
        self.hyper_b_final = nn.Linear(16, 1)

        self.num_agents = num_agents
        self.embed_dim = embed_dim

    def forward(self, agent_qs, state):
        # agent_qs: (B, N)
        # state: (B, H, W, C)
        B, N = agent_qs.size()

        # Encode state
        state = state.permute(0,3,1,2).float() # (B, C, H, W)
        s = self.conv(state)
        s = s.view(B, -1) # (B, 16)

        w1 = torch.abs(self.hyper_w_1(s)).view(B, self.num_agents, self.embed_dim) # ensure non-negativity if needed
        b1 = self.hyper_b_1(s).view(B, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1) # (B, 1, embed_dim)

        w_final = torch.abs(self.hyper_w_final(s)).view(B, self.embed_dim, 1)
        b_final = self.hyper_b_final(s).view(B, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + b_final # (B, 1, 1)
        q_tot = q_tot.view(B)
        return q_tot

# -------------------------
# QMIX Class
# -------------------------
class QMIX:
    def __init__(self, num_agents, state_shape, agent_ids, num_actions=21, lr=1e-3, gamma=0.99, device=device):
        self.num_agents = num_agents
        self.gamma = gamma
        self.agent_ids = agent_ids

        self.agent_q_network = AgentQNetwork(num_actions=num_actions)
        self.agent_q_network.to(device)
        self.mixing_network = MixingNetwork(num_agents, state_shape=state_shape)
        self.mixing_network.to(device)
        
        self.target_agent_q_network = AgentQNetwork(num_actions=num_actions)
        self.target_agent_q_network.to(device)
        self.target_mixing_network = MixingNetwork(num_agents, state_shape=state_shape)
        self.target_mixing_network.to(device)
        
        # Load weights to target
        self.target_agent_q_network.load_state_dict(self.agent_q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        self.optimizer = optim.Adam(list(self.agent_q_network.parameters()) + list(self.mixing_network.parameters()), lr=lr)

        # Watch models with wandb
        wandb.watch(self.agent_q_network, log_freq=100)
        wandb.watch(self.mixing_network, log_freq=100)
    def select_actions(self, obs, epsilon=0.1):
        # obs shape: (N_agents, 13,13,5)
        obs = obs.unsqueeze(0) # add batch dimension
        with torch.no_grad():
            q_values = self.agent_q_network(obs) # (1, N, actions)
        q_values = q_values.squeeze(0) # (N, actions)
        actions = {}
        for i, agent in enumerate(self.agent_ids):
            if np.random.rand() < epsilon:
                a = np.random.randint(q_values.size(-1))
            else:
                a = q_values[i].argmax().item()
            actions[agent] = a
        return actions

    def update(self, batch):
        # batch is sampled from replay buffer
        # batch['obs']: (B, N, 13,13,5)
        # batch['actions']: (B, N)
        # batch['rewards']: (B)
        # batch['next_obs']: (B, N, 13,13,5)
        # batch['state']: (B, H, W, C)
        # batch['next_state']: (B, H, W, C)
        # batch['dones']: (B)
        batch['dones'] = batch['dones'].astype(np.int8)

        obs = torch.tensor(batch['obs'], device=device)
        actions = torch.tensor(batch['actions'], dtype=torch.long, device=device)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32, device=device)
        next_obs = torch.tensor(batch['next_obs'], device=device)
        state = torch.from_numpy(batch['state']).to(device)
        next_state = torch.from_numpy(batch['next_state']).to(device)
        dones = torch.tensor(batch['dones'], dtype=torch.int8).to(device)

        # Current Q-values
        q_values = self.agent_q_network(obs) # (B, N, actions)
        chosen_q = torch.gather(q_values, dim=2, index=actions.unsqueeze(-1)).squeeze(-1) # (B, N)

        # Target Q-values
        with torch.no_grad():
            target_q_values = self.target_agent_q_network(next_obs) # (B, N, actions)
            # Double Q-learning: select max actions from current net
            max_actions = target_q_values.argmax(dim=2, keepdim=True)
            max_q = torch.gather(target_q_values, 2, max_actions).squeeze(-1) # (B, N)
        
        # Compute Q_tot and target Q_tot
        # Q_tot is the sum of Q-values for all agents
        q_tot = self.mixing_network(chosen_q, state)
        target_q_tot = self.target_mixing_network(max_q, next_state)
        target = rewards + self.gamma * (1 - dones) * target_q_tot

        loss = ((q_tot - target.detach())**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    # TODO:
    # Soft update or periodic hard update of target networks
    # Here we do periodic hard update:
    def update_target(self):
        for param, target_param in zip(self.agent_q_network.parameters(), self.target_agent_q_network.parameters()):
            target_param.data.copy_(param.data)
        for param, target_param in zip(self.mixing_network.parameters(), self.target_mixing_network.parameters()):
            target_param.data.copy_(param.data)

# -------------------------
# Training Loop (Conceptual)
# -------------------------

class ReplayBuffer:
    def __init__(self, memory_size, field_names):
        self.memory_size = memory_size
        self.fields = {name: [] for name in field_names}
        self.idx = 0

    def save_to_memory(self, *args):
        keys = list(self.fields.keys())
        for i, arg in enumerate(args):
            self.fields[keys[i]].append(arg)
        if len(self.fields["obs"]) > self.memory_size:
            for k in self.fields.keys():
                self.fields[k].pop(0)

    def sample(self, batch_size):
        idxs = np.random.randint(0, len(self.fields["obs"]), batch_size)
        batch = {k: [] for k in self.fields.keys()}
        for k in self.fields.keys():
            batch[k] = [self.fields[k][i] for i in idxs]
        return batch

    def __len__(self):
        return len(self.fields["obs"])
        
def process_batch(batch, field_names, batch_size):
    batch_blue = {k: [] for k in field_names}
    batch_red = {k: [] for k in field_names}
    
    # Step 1: Process state and next_state first
    # These should be (B, H, W, C) = (B, 45, 45, 5)
    for field in ['state', 'next_state']:
        batch_state = np.stack(batch[field], axis=0)  # Stack along batch dimension
        batch_blue[field] = batch_state
        batch_red[field] = batch_state
    
    batch_blue['rewards'] = np.stack([batch['rewards'][i]['blue'] for i in range(batch_size)], axis=0)
    batch_red['rewards'] = np.stack([batch['rewards'][i]['red'] for i in range(batch_size)], axis=0)

    batch_blue['dones'] = np.stack([batch['dones'][i]['blue'] for i in range(batch_size)], axis=0)
    batch_red['dones'] = np.stack([batch['dones'][i]['red'] for i in range(batch_size)], axis=0)

    # Step 2: Process other fields
    for field in field_names:
        if field in ['state', 'next_state', 'rewards', 'dones']:
            continue
            
        for i in range(batch_size):
            current_batch = batch[field][i]
            blue_agents = [a for a in current_batch.keys() if a.startswith("blue")]
            red_agents = [a for a in current_batch.keys() if a.startswith("red")]
            
            # Step 3: Handle observations differently from other fields
            if field in ['obs', 'next_obs']:
                # Shape should be (B, N, 13, 13, 5)
                if blue_agents:
                    blue_obs = np.stack([current_batch[a] for a in blue_agents], axis=0)  # (N, 13, 13, 5)
                    batch_blue[field].append(blue_obs)
                if red_agents:
                    red_obs = np.stack([current_batch[a] for a in red_agents], axis=0)  # (N, 13, 13, 5)
                    batch_red[field].append(red_obs)
            else:
                # For actions, rewards, dones: Shape should be (B, N)
                if blue_agents:
                    blue_values = np.array([current_batch[a] for a in blue_agents])  # (N,)
                    batch_blue[field].append(blue_values)
                if red_agents:
                    red_values = np.array([current_batch[a] for a in red_agents])  # (N,)
                    batch_red[field].append(red_values)
            # Step 4: Stack all batches
    for field in field_names:
        if field not in ['state', 'next_state'] and len(batch_blue[field]) > 0:
            batch_blue[field] = np.stack(batch_blue[field], axis=0)  # Stack along batch dimension
        if field not in ['state', 'next_state'] and len(batch_red[field]) > 0:
            batch_red[field] = np.stack(batch_red[field], axis=0)  # Stack along batch dimension
            
    return batch_blue, batch_red

if __name__ == "__main__":
    import supersuit as ss
    wandb.login(key="37d305fc5fac9b15b88ec48208250be56185986e")
        # Initialize wandb
    wandb.init(project="QMIX_Project 1", config={ 
        "learning_rate": 1e-3,
        "batch_size": 32,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_decay": 0.9998,
        "epsilon_min": 0.05,
        "num_episodes": 50,
        "update_step" : 10
    })
    config = wandb.config

    # Update variables with config values
    num_episodes = config.num_episodes
    batch_size = config.batch_size
    epsilon = config.epsilon_start
    epsilon_decay = config.epsilon_decay
    epsilon_min = config.epsilon_min
    learning_rate = config.learning_rate
    gamma = config.gamma
    update_step = config.update_step

    num_envs = 2 
    # env = battle_v4.parallel_env(map_size=30, minimap_mode=False, max_cycles=300, seed=10)
    env = battle_v4.parallel_env(map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, 
            extra_features=False)
    env = ss.black_death_v3(env)
    # env = ss.pad_observations_v0(env)
    # env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])

    env.reset()

    # Number of blue agents is known after reset
    # env.possible_agents contains both red_ and blue_ agents
    blue_agents = [agent for agent in env.possible_agents if agent.startswith("blue_")]
    num_blue_agents = len(blue_agents)

    red_agents = [agent for agent in env.possible_agents if agent.startswith("red_")]
    num_red_agents = len(red_agents)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize your QMIX agents with the learning rate from config
    qmix_blue = QMIX(
        num_agents=num_blue_agents,
        state_shape=(45, 45, 5),
        agent_ids=blue_agents,
        num_actions=21,
        lr=learning_rate,
        gamma=gamma
    )
    qmix_red = QMIX(
        num_agents=num_red_agents,
        state_shape=(45, 45, 5),
        agent_ids=red_agents,
        num_actions=21,
        lr=learning_rate,
        gamma=gamma
    )
    # Off-policy training: we need a replay buffer
    # rb = MultiAgentReplayBuffer(
    #     memory_size=100000,
    #     field_names=["obs", "action", "reward", "next_obs", "done", "state", "next_state"],
    #     agent_ids=env.possible_agents,  # Use possible_agents instead of agents
    #     device="cpu"
    # )
    field_names = ["obs", "actions", "rewards", "next_obs", "dones", "state", "next_state"]
    rb = ReplayBuffer(memory_size=1000,
         field_names=field_names)

    num_episodes = 1
    batch_size = 2
    epsilon = 1.0
    epsilon_decay = 0.999

    count = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        # We will collect transitions after all blue agents have acted once (a "joint step")
        # the purpose of joint step is to collect joint transitions for QMIX training
        global_state = env.state() # global state for mixing

        episode_reward_blue = 0
        episode_reward_red = 0

        terminated = False
        while not terminated:
            
            count += 1
            # Convert obs to batch form
            blue_agents = [agent for agent in env.agents if agent.startswith("blue")]
            obs_array_blue = np.stack([obs[a] for a in blue_agents], axis=0)
            qmix_blue.agent_ids = blue_agents
            actions_blue = qmix_blue.select_actions(torch.from_numpy(obs_array_blue).to(device), epsilon=epsilon) # return shape: (N_agents,)

            red_agents = [agent for agent in env.agents if agent.startswith("red")]
            qmix_red.agent_ids = red_agents
            obs_array_red = np.stack([obs[a] for a in red_agents], axis=0)
            actions_red = qmix_red.select_actions(torch.from_numpy(obs_array_red).to(device), epsilon=epsilon) # return shape: (N_agents,)

            actions = {}
            for i, agent in enumerate(blue_agents):
                actions[agent] = actions_blue[agent]
            
            for i, agent in enumerate(red_agents):
                actions[agent] = actions_red[agent]

            # Step environment
            next_obs, rewards, terminations, truncations, info = env.step(actions)
            dones = {}
            for agent in env.agents:
                dones[agent] = terminations[agent] or truncations[agent]

            next_global_state = env.state()

            # Compute rewards: sum or average relevant ones for blue agents
            # The environment: each agent gets individual reward. We can sum them for QMIX training
            # done_all = np.all(list(dones.values()))
            blue_done = np.all([dones[a] for a in blue_agents])
            red_done = np.all([dones[a] for a in red_agents])
            dones = {"blue": blue_done, "red": red_done}
            done_all = np.all(list(dones.values()))

            reward_blue = sum([rewards[a] for a in blue_agents])
            reward_red = sum([rewards[a] for a in red_agents])

            episode_reward_blue += reward_blue
            episode_reward_red += reward_red

            # Store transition in replay buffer
            # Flatten structures and store:
            reward_save = {"blue": reward_blue, "red": reward_red}
            rb.save_to_memory(
                obs, # dict of observations of all agents
                actions, # dict of actions of all agents
                reward_save, # dict of rewards of all agents
                next_obs, # dict of next observations of all agents
                dones, # dict of done flags of all agents
                global_state,
                next_global_state
            )

            obs = next_obs
            global_state = next_global_state
            terminated = done_all

            # After episode, decay epsilon
            epsilon = max(epsilon * epsilon_decay, 0.05)

            # Training step
            if len(rb) > batch_size:
                batch = rb.sample(batch_size)
                field_names = list(batch.keys())
                batch_blue, batch_red = process_batch(batch, field_names, batch_size)
                
                # Now batch_blue and batch_red should have the correct shapes:
                # batch_blue['obs']: (B, N, 13, 13, 5)
                # batch_blue['actions']: (B, N)
                # batch_blue['rewards']: (B, N)
                # batch_blue['next_obs']: (B, N, 13, 13, 5)
                # batch_blue['state']: (B, 45, 45, 5)
                # batch_blue['next_state']: (B, 45, 45, 5)
                # batch_blue['dones']: (B, N)
                
                loss_blue = qmix_blue.update(batch_blue)
                loss_red = qmix_red.update(batch_red)

                wandb.log({
                        "loss_blue": loss_blue.item(),
                        "loss_red": loss_red.item(),
                        "epsilon": epsilon,
                    })
                if count % update_step == 0:
                    qmix_blue.update_target()
                    qmix_red.update_target()
                    print(count)
                if count == 1000:
                    print(terminated)
        if count == 1000:
            print(terminated)
        wandb.log({
        "episode_reward_blue": episode_reward_blue,
        "episode_reward_red": episode_reward_red,
        "episode": ep
    })
        # After training:
        # if ep % 50 == 0:
        #     # Save models
        #     torch.save(qmix_blue.agent_q_network.state_dict(), f"./model/qmix_blue_ep{ep}.pth")
        #     torch.save(qmix_red.agent_q_network.state_dict(), f"./model/qmix_red_ep{ep}.pth")

        #     # # Log the saved models to wandb
        #     # wandb.save(f"qmix_blue_ep{ep}.pth")
        #     # wandb.save(f"qmix_red_ep{ep}.pth")

    wandb.finish()

    print("Training complete.")
    # After training:
    # The Q-networks and mixing network learned to produce coordinated policies.
    # Blue agents should better focus fire and coordinate attacks to secure kills efficiently.