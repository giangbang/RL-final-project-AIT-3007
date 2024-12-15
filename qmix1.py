import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from magent2.environments import battle_v4
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
# from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
# from magent2.
import wandb
import supersuit
from typing import List
# load pretrained model from torchvision
import torchvision.models as models

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')
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
            nn.Conv2d(5, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(2),  # Reduces H and W by half
            nn.Conv2d(32, 64, kernel_size=3, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, dilation=2),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(128*3*3, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: ones for weights, zeros for biases
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        # After pooling: 16 dims  
        self.state_dim = 32 
        self.num_agents = num_agents
        self.embedding_dim = embed_dim
        # Layers for hypernetwork
        self.hyper_w_1 = nn.Sequential(
        nn.Linear(self.state_dim, self.embedding_dim),
        nn.ReLU(),
        nn.Linear(self.embedding_dim, self.num_agents * self.embedding_dim)
        )
        self.hyper_w_final = nn.Sequential(
        nn.Linear(self.state_dim, self.embedding_dim),
        nn.ReLU(),
        nn.Linear(self.embedding_dim, self.embedding_dim)
        )

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embedding_dim)
        self.hyper_b_final = nn.Sequential(
        nn.Linear(self.state_dim, self.embedding_dim),
        nn.ReLU(),
        nn.Linear(self.embedding_dim, 1)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Conv layers: Kaiming initialization with fan_out mode
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: ones for weights, zeros for biases
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Linear):
                # Different initialization for different linear layers
                if m in self.hyper_w_1.modules() or m in self.hyper_w_final.modules():
                    # Hypernetwork weights: scaled-down Kaiming initialization
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                    m.weight.data.mul_(0.1)
                else:
                    # Other linear layers: regular Kaiming initialization
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, agent_qs, state):
        # agent_qs: (B, N)
        # state: (B, H, W, C)
        B, N = agent_qs.size()

        # Encode state
        state = state.permute(0,3,1,2).float() # (B, C, H, W)
        s = self.conv(state)
        s = s.view(B, -1) # (B, 32)

        w1 = torch.abs(self.hyper_w_1(s)).view(B, self.num_agents, self.embedding_dim) # ensure non-negativity if needed
        b1 = self.hyper_b_1(s).view(B, 1, self.embedding_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1) # (B, 1, embed_dim)

        w_final = torch.abs(self.hyper_w_final(s)).view(B, self.embedding_dim, 1)
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
        # self.agent_q_network = torch,nn.DataParallel(self.agent_q_network)
        self.agent_q_network.to(device)
        self.agent_q_network = torch.compile(self.agent_q_network)

        self.mixing_network = MixingNetwork(num_agents, state_shape=state_shape)
        # self.mixing_network = torch.nn.DataParallel(self.mixing_network)
        self.mixing_network.to(device)
        self.mixing_network = torch.compile(self.mixing_network)
        
        self.target_agent_q_network = AgentQNetwork(num_actions=num_actions)
        # self.target_agent_q_network = torch.nn.DataParallel(self.target_agent_q_network)
        self.target_agent_q_network.to(device)
        self.target_agent_q_network = torch.compile(self.target_agent_q_network)

        self.target_mixing_network = MixingNetwork(num_agents, state_shape=state_shape)
        # self.target_mixing_network = torch.nn.DataParallel(self.target_mixing_network)
        self.target_mixing_network.to(device)
        self.target_mixing_network = torch.compile(self.target_mixing_network)  
        
        # Load weights to target
        self.target_agent_q_network.load_state_dict(self.agent_q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

        self.optimizer = optim.Adam(list(self.agent_q_network.parameters()) + list(self.mixing_network.parameters()), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=20, T_mult=2, eta_min=1e-6 )
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

        torch.nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)

        self.optimizer.step()
        self.scheduler.step()
        return loss
    # TODO:
    # Soft update or periodic hard update of target networks
    # Here we do periodic hard update:
    
    def update_target(self, tau=0.4):
        for param, target_param in zip(self.agent_q_network.parameters(), self.target_agent_q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for param, target_param in zip(self.mixing_network.parameters(), self.target_mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


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

def train(config):
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
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=200, 
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
    # rb = MultiAgentReplayBuffer(
    #     memory_size=100000,
    #     field_names=["obs", "action", "reward", "next_obs", "done", "state", "next_state"],
    #     agent_ids=env.possible_agents,  # Use possible_agents instead of agents
    #     device="cpu"
    # )
    field_names = ["obs", "actions", "rewards", "next_obs", "dones", "state", "next_state"]
    rb = ReplayBuffer(memory_size=10000,
         field_names=field_names)

    count = 0
    for ep in range(num_episodes):
        obs, _ = env.reset()
        # We will collect transitions after all blue agents have acted once (a "joint step")
        # the purpose of joint step is to collect joint transitions for QMIX training
        global_state = env.state() # global state for mixing

        episode_reward_blue = 0
        episode_reward_red = 0

        blue_agents = [agent for agent in env.agents if agent.startswith("blue")]
        red_agents = [agent for agent in env.agents if agent.startswith("red")]
        terminated = False

        while not terminated:
            
            # count += 1
            # Convert obs to batch form
            # blue_agents = [agent for agent in env.agents if agent.startswith("blue")]
            obs_array_blue = np.stack([obs[a] for a in blue_agents], axis=0)
            qmix_blue.agent_ids = blue_agents
            actions_blue = qmix_blue.select_actions(torch.from_numpy(obs_array_blue).to(device), epsilon=epsilon) # return shape: (N_agents,)

            # red_agents = [agent for agent in env.agents if agent.startswith("red")]
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
            blue_agents = [agent for agent in env.agents if agent.startswith("blue_")]
            red_agents = [agent for agent in env.agents if agent.startswith("red_")]
            # Compute rewards: sum or average relevant ones for blue agents
            # The environment: each agent gets individual reward. We can sum them for QMIX training
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
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # Training step
            if len(rb) > batch_size:
                count += 1
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
                        "loss" : loss_blue.item() + loss_red.item()
                    })
                if count % update_step == 0:
                    qmix_blue.update_target()
                    qmix_red.update_target()
                    print(count)

        wandb.log({
        "episode_reward_blue": episode_reward_blue,
        "episode_reward_red": episode_reward_red,
        "episode": ep
        })
        if ep+1 % 20 == 0:
                save_path_blue = os.path.join("./model", f"qmix_blue_ep{ep}.pth")
                save_path_red = os.path.join("./model", f"qmix_red_ep{ep}.pth")
                torch.save(qmix_blue.agent_q_network.state_dict(), save_path_blue)
                torch.save(qmix_red.agent_q_network.state_dict(), save_path_red)


    wandb.finish()
    print("Training complete.")
    # After training:
    # The Q-networks and mixing network learned to produce coordinated policies.
    # Blue agents should better focus fire and coordinate attacks to secure kills efficiently.
    
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="QMIX training for MAgent2 battle environment")
    
    # Training hyperparameters
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for training")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--epsilon_start", type=float, default=1.0,
                        help="Starting value of epsilon for epsilon-greedy exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.9998,
                        help="Decay rate of epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.06,
                        help="Minimum value of epsilon")
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to train")
    parser.add_argument("--update_step", type=int, default=200,
                        help="Number of steps between target network updates")
    
    # # Environment parameters
    # parser.add_argument("--map_size", type=int, default=45,
    #                     help="Size of the battle map")
    # parser.add_argument("--max_cycles", type=int, default=1000,
    #                     help="Maximum number of cycles per episode")
    
    # # Wandb parameters
    # parser.add_argument("--wandb_project", type=str, default="QMIX_Project_1",
    #                     help="Name of the wandb project")
    # parser.add_argument("--wandb_key", type=str, required=True,
    #                     help="Wandb API key")
    # parser.add_argument("--use_wandb", action="store_true",
    #                     help="Whether to use wandb for logging")
    
    # # Model parameters
    # parser.add_argument("--save_model", action="store_true",
    #                     help="Whether to save model checkpoints")
    # parser.add_argument("--model_dir", type=str, default="./model",
    #                     help="Directory to save model checkpoints")
    # parser.add_argument("--save_interval", type=int, default=50,
    #                     help="Number of episodes between model saves")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    import supersuit as ss
    import os

    args = parse_args()
    
    wandb.login(key="")
        # Initialize wandb
    wandb.init(
        project="QMIX_Project_1",
        config={
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gamma": args.gamma,
            "epsilon_start": args.epsilon_start,
            "epsilon_decay": args.epsilon_decay,
            "epsilon_min": args.epsilon_min,
            "num_episodes": args.num_episodes,
            "update_step": args.update_step,
        }
    )
    config = wandb.config

    train(config)











    # sweep_configuration = {
    #     "method" : "bayes",
    #     "name" : "QMIX_Project 2",
    #     "metric" : {"name" : "loss", "goal" : "minimize"},
    #     "parameters" : {
    #         "learning_rate" : {"min" : 5e-4, "max" : 2e-3},
    #         "batch_size" : {"values" : [16, 64, 256, ]},
    #         "gamma" : {"values" : [0.6, 0.7, 0.9, 0.99]},
    #         "epsilon_start" : {"value" : 1.0},
    #         "epsilon_decay" : {"value" : 0.9999},
    #         "epsilon_min" : {"value" : 0.05},
    #         "num_episodes" : {"value" : 10000},
    #         "update_step" : {"values" : [5, 40, 100]}
    #     }
    # }
    # sweep_id = wandb.sweep(sweep_configuration, project="QMIX_Project 1")
    # wandb.agent(sweep_id=sweep_id, function=train(config), count=10)