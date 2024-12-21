import gc
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List
from torch.nn import functional as F
from typing import Dict, Tuple, List
import wandb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('high')

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
    def forward(self, x):
        return self.conv(x)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=3,
                               stride=stride, dilation=dilation, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Automatically create downsample layer if needed
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResidualFCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(ResidualFCBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(out_features, out_features)
        
        if in_features != out_features:
            self.residual = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(dropout)
            )
        else:
            self.residual = nn.Identity()
        
        # Initialization
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        residual = self.residual(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class SharedActorNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super(SharedActorNetwork, self).__init__()
        self.cnn = nn.Sequential(
            DepthwiseSeparableConv(5, 32, stride=2),
            DepthwiseSeparableConv(32, 64, stride=2),
            DepthwiseSeparableConv(64, 64, stride=2),
            nn.BatchNorm2d(64),
            # ResidualBlock(32, 32, stride=2),
            # nn.Flatten(),
        )
        self.fc1 = nn.Linear(self.cnn_output_dim, 128)
        # Calculate the size after CNN layers
        self._create_fc(observation_shape)
        self.fc2 = nn.Sequential(
            ResidualFCBlock(128, 128),
            nn.Linear(128, action_shape),
        )
        self.gru = nn.GRUCell(128, 128)

        self.embed = nn.Embedding(action_shape, 16)
    
    def _create_fc(self, observation_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_dim = dummy_output.view(-1).shape[0]
    
    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, 128, device=device)

    def forward(self, x, action, hidden = None):
        x = self.cnn(x)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        
        x = torch.cat([x, self.embed(action)], dim=-1)
        x = self.fc1(x)

        hidden = self.gru(x, hidden)
        x = self.fc2(hidden)
        action_probs = F.softmax(x, dim=-1)
        return x, action_probs, hidden
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QNetwork(nn.Module):
    def __init__(self, observation_shape, action_shape=21):
        super(QNetwork, self).__init__()
        self.cnn = nn.Sequential(
            DepthwiseSeparableConv(5, 16, stride=2),
            nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            DepthwiseSeparableConv(16,32, stride=2),

            DepthwiseSeparableConv(32, 64, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2),
            # DepthwiseSeparableConv(32, 32, stride=2),
            DepthwiseSeparableConv(64, 128, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
        )
        # Calculate the size after CNN layers
        # self._create_fc(observation_shape)
        self.fc1 = nn.Linear(128+8, 128)
        self.fc2 = nn.Sequential(
            ResidualFCBlock(256, 128),
            nn.Linear(128, action_shape),
        )
        self.gru = nn.GRUCell(128, 256)

        self.embed = nn.Embedding(action_shape, 8)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _create_fc(self, observation_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *observation_shape)
            dummy_output = self.cnn(dummy_input)
            self.cnn_output_dim = dummy_output.view(-1).shape[0]
    
    def init_hidden(self, batch_size, select=False):
        return torch.zeros(batch_size, 256, device=device)

    def action_none(self, batch_size):
        return torch.zeros(batch_size, dtype=torch.long, device=device)


    def forward(self, x, action, hidden = None):
        x = self.cnn(x)
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        hidden = hidden.detach()
        if action is None:
            action = self.action_none(x.size(0))
        x = torch.cat([x, self.embed(action)], dim=-1)
        x = self.fc1(x)
        hidden = self.gru(x, hidden)

        q = self.fc2(hidden)
        # action_probs = F.softmax(x, dim=-1)
        return q, hidden
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_shape=(5, 45, 45), embed_dim=64, use_attention=True):
        """
        Mixing network for QMIX with optional attention mechanism.
        
        Args:
            num_agents (int): Number of agents.
            state_shape (tuple): Shape of the global state.
            embed_dim (int): Embedding dimension for hypernetworks.
            use_attention (bool): Whether to use attention mechanism.
        """
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        self.use_attention = use_attention

        self.state_encoder = nn.Sequential(
            # Initial feature extraction
            DepthwiseSeparableConv(5, 32, stride=2),
            nn.BatchNorm2d(32),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            ResidualBlock(32, 32),

            nn.MaxPool2d(2),

            ResidualBlock(32, 64, stride=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            # DepthwiseSeparableConv(64, 128, stride=2),
            # nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True),

            # # Downsample 45->23
            # # Downsample 23->12
            # # Downsample 12->6
            # Final reduction 6->1
            nn.Conv2d(64, 128, kernel_size=6),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # ResidualBlock(64, 64),
            # nn.ReLU(inplace=True)
        )
        self.state_dim = 128

        # Hypernetworks for weights
        self.hyper_w_1 = nn.Sequential(
            ResidualFCBlock(self.state_dim, embed_dim),
            nn.Linear(embed_dim, num_agents * embed_dim),
        )
        self.hyper_w_final = nn.Sequential(
            ResidualFCBlock(self.state_dim, embed_dim),
        )

        # Hypernetworks for biases
        self.hyper_b_1 = ResidualFCBlock(self.state_dim, embed_dim)
        self.hyper_b_final = nn.Sequential(
            # ResidualFCBlock(self.state_dim, 1),
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        
        # self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming Initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Linear layer initialization
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, agent_qs, state):
        """
        Forward pass for MixingNetwork.
        
        Args:
            agent_qs (torch.Tensor): Agent Q-values with shape (batch_size*transitions, num_agents).
            state (torch.Tensor): Global state with shape (batch_size*transitions, C, H, W).
        
        Returns:
            torch.Tensor: Total Q-value with shape (batch_size, 1).
        """
        batch_size = state.size(0)
        state = self.state_encoder(state).view(batch_size, -1)  # Shape: (batch_size*transition, state_dim)

        # Generate hypernetwork weights and biases
        w1 = torch.abs(self.hyper_w_1(state)).view(batch_size, self.num_agents, self.embed_dim)  # (batch_size*transition, num_agents, embed_dim)
        b1 = self.hyper_b_1(state).view(batch_size, 1, self.embed_dim)  # (batch_size*transition, 1, embed_dim)
        hidden = F.elu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (batch_size*transiition, 1, embed_dim)

        w_final = torch.abs(self.hyper_w_final(state)).view(batch_size, self.embed_dim, 1)  # (batch_size*transitions, embed_dim, 1)
        b_final = self.hyper_b_final(state).view(batch_size, 1, 1)  # (batch_size*transitions, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + b_final  # (batch_size*transitions, 1, 1)

        q_tot = q_tot.view(batch_size, -1)  # (batch_size, transitions)
        return q_tot

    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class QMIX:
    """
    QMIX implementation for multi-agent reinforcement learning.
    """
    def __init__(
        self,
        num_agents: int,
        agent_ids,
        agent_ids1,
        state_shape: Tuple[int, int, int],
        device: torch.device,
        num_actions: int = 21,
        lr: float = 1e-3,
        gamma: float = 0.99,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.device = device
        self.agent_ids = agent_ids
        self.agent_ids1 = agent_ids1
        # Initialize networks
        self.agent_q_network = QNetwork(observation_shape=(5,13,13), action_shape=21).to(device)
        # self.agent_q_network1 = QNetwork(observation_shape=(5,13,13), action_shape=21).to(device)
        # self.agent_q_network2 = QNetwork(observation_shape=(5,13,13), action_shape=21).to(device)
        self.mixing_network = MixingNetwork(num_agents, state_shape=state_shape).to(device)

        self.target_agent_q_network = QNetwork(observation_shape=(5,13,13), action_shape=21).to(device)
        self.target_mixing_network = MixingNetwork(num_agents, state_shape=state_shape).to(device)
        
        # Compile networks for potential speed-up (PyTorch 2.0+)
        try:
            self.agent_q_network = torch.compile(self.agent_q_network)
            self.mixing_network = torch.compile(self.mixing_network)
            self.target_agent_q_network = torch.compile(self.target_agent_q_network)
            self.target_mixing_network = torch.compile(self.target_mixing_network)
        except Exception as e:
            print(f"Compilation failed: {e}. Using regular execution.")
        # # If torch.compile fails (e.g., not using PyTorch 2.0), proceed without compiling
  
        self.update_target_hard()
        # Optimizer and scheduler
        self.optimizer = optim.Adam(
            params=list(self.agent_q_network.parameters()) + list(self.mixing_network.parameters()),
            lr=lr,
        )
        self.scheduler1 = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=300, gamma=0.9
        )
        self.lambda_ = 0.5
        self.eligibility_traces = {}
        for name, param in self.named_parameters():
            self.eligibility_traces[name] = torch.zeros_like(param.data)

    def named_parameters(self):
        # Generator for all named parameters
        for name, param in self.agent_q_network.named_parameters():
            yield f'agent_q_{name}', param
        for name, param in self.mixing_network.named_parameters():
            yield f'mixing_{name}', param
        # self.loss_fn = nn.MSELoss().to(device)
        # Log models with wandb
        # wandb.watch(self.agent_q_network, log_freq=1000)
        # wandb.watch(self.mixing_network, log_freq=1000)

    def select_actions(self, obs: torch.Tensor, prev_action: None, 
                       hidden = None,
                        epsilon: float = 0.1) -> Dict[str, int]:
        """
        Select actions using epsilon-greedy policy.

        Args:
            obs (torch.Tensor): Observations with shape (N_agents, C, H, W).
            epsilon (float): Probability of choosing a random action.

        Returns:
            Dict[str, int]: Mapping from agent IDs to selected actions.
        """
        # Add batch dimension
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)  # Shape: (H, W, C)
        # obs = obs.unsqueeze(0)  # Shape: (1, H, W, c)
        N, E, H, W, C = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3).contiguous()        # Shape: (B, N, C, H, W)
        obs = obs.view(-1,C,H,W).contiguous()
        if prev_action is not None:
            prev_action = torch.tensor(prev_action, dtype=torch.long, device=self.device)  # Shape: 
            prev_action = prev_action.view(-1)
        # else:
        #     prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            q_values, hidden = self.agent_q_network(obs, prev_action, hidden)  # Shape: (num_actions)
        
        # for agent in self.agent_ids:
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, q_values.shape[1], (q_values.shape[0],), device=q_values.device)
        else:
            action = torch.argmax(q_values, dim=1)

        action = action.view(N, E)
        actions = {}
        for i, agent in enumerate(self.agent_ids):
            actions[agent] = action[i].cpu().numpy()
        return actions, hidden

    def select_actions1(self, obs: torch.Tensor, prev_action: None, 
                       hidden = None,
                        epsilon: float = 0.1) -> Dict[str, int]:
        """
        Select actions using epsilon-greedy policy.

        Args:
            obs (torch.Tensor): Observations with shape (N_agents, C, H, W).
            epsilon (float): Probability of choosing a random action.

        Returns:
            Dict[str, int]: Mapping from agent IDs to selected actions.
        """
        # Add batch dimension
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)  # Shape: (H, W, C)
        # obs = obs.unsqueeze(0)  # Shape: (1, H, W, c)
        N, E, H, W, C = obs.shape
        obs = obs.permute(0, 1, 4, 2, 3).contiguous()        # Shape: (B, N, C, H, W)
        obs = obs.view(-1,C,H,W).contiguous()
        if prev_action is not None:
            prev_action = torch.tensor(prev_action, dtype=torch.long, device=self.device)  # Shape: 
            prev_action = prev_action.view(-1)
        # else:
        #     prev_action = torch.zeros(1, dtype=torch.long, device=self.device)
        # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            q_values, hidden = self.agent_q_network(obs, prev_action, hidden)  # Shape: (num_actions)
        
        # for agent in self.agent_ids:
        if torch.rand(1).item() < epsilon:
            action = torch.randint(0, q_values.shape[1], (q_values.shape[0],), device=q_values.device)
        else:
            action = torch.argmax(q_values, dim=1)

        action = action.view(N, E)
        actions = {}
        for i, agent in enumerate(self.agent_ids1):
            actions[agent] = action[i].cpu().numpy()
        return actions, hidden
        # q_values = q_values.cpu().numpy()

        # if np.random.rand() < epsilon:
            # action = np.random.randint(0, q_values.shape[1])
        # else:
        #     action = np.argmax(q_values).astype(np.uint16)
        # # check if observation is padded observation ( all zeros) set action equal to 0
        # return action, hidden
    
    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        self.agent_q_network.train()
        self.mixing_network.train()
        self.target_agent_q_network.eval()
        self.target_mixing_network.eval()

        """
        Update the QMIX networks based on a batch of experiences.

        Args:
            batch (Dict[str, np.ndarray]): Batch of experiences.

        Returns:
            float: The computed loss value.
        """

        obs = batch['o']
        actions = batch['a'].long()
        rewards = batch['r'] # Shape: (B, T)
        state = batch['s']
        dones = batch['d'] # Shape: (B, T)

        B, T = obs.shape[0], obs.shape[1]

        total_loss = 0.0
        hidden = None
        hidden_target = None
        q_evals = []
        q_targets = []
        loss = 0
        entropy = 0
        bs =  2
        for b in range(0, B, bs):
            bs = b+bs
            if bs > B:
                break
            q_evals = []
            q_targets = []
            loss = 0
            for i in range(T-1): # Transitions

                # Extract sub-batch for the current episode
                obs_sb = obs[b: bs, i]           # Shape: ( N, H, W, C)
                # obs_sb = obs[:,i]           # Shape: (B, N, H, W, C)
                # obs_sb = obs_sb.unsqueeze(0)  # Shape: (1, N, H, W, C)

                actions_sb = actions[b: bs, i]      # Shape: (N,)
                # actions_sb = actions[:, i]      # Shape: (B, N, 1)
                # actions_sb = actions_sb.unsqueeze(0)  # Shape: (1, N, 1)

                # rewards_sb = rewards[b, i+1]      # Shape: (N,)
                # # rewards_sb = rewards[:, i+1]      # Shape: (B, N)
                # rewards_sb = rewards_sb.unsqueeze(0)  # Shape: (1, N)

                next_obs_sb = obs[b: bs, i+1]    # Shape: (N, H, W, C)
                # next_obs_sb = obs[:,i+1]    # Shape: (B, N, H, W, C)
                # next_obs_sb = next_obs_sb.unsqueeze(0)  # Shape: (1, N, H, W, C)

                # done_sb = dones[b, i+1]      # Shape: (N,)
                # # done_sb = dones[:, i+1]      # Shape: (B, N)
                # done_sb = done_sb.unsqueeze(0)  # Shape: (1, N)
                if i != 0:
                    prev_actions_sb = actions[b: bs,i-1]  # Shape: (B, N)
                    # prev_actions_sb = prev_actions_sb.unsqueeze(0)  # Shape: (1, N)
                else:
                    prev_actions_sb = None


                obs_sb = obs_sb.permute(0, 1, 4, 2, 3).contiguous()        # Shape: (B, N, C, H, W)
                next_obs_sb = next_obs_sb.permute(0, 1, 4, 2, 3).contiguous()

                # Flatten batch and agent dimensions for processing
                sb_size, N_agents = obs_sb.shape[0], obs_sb.shape[1]
                obs_sb_flat = obs_sb.view(-1, *obs_sb.shape[2:]).contiguous()          # Shape: (sb_size * N, C, H, W)
                actions_sb_flat = actions_sb.contiguous().view(-1).contiguous()                     # Shape: (sb_size * N)
                next_obs_sb_flat = next_obs_sb.view(-1, *next_obs_sb.shape[2:]).contiguous()  # Shape: (sb_size * N, C, H, W)

                if prev_actions_sb is not None:
                    prev_actions_sb_flat = prev_actions_sb.contiguous().view(-1).contiguous()  # Shape: (sb_size * N)
                else:
                    prev_actions_sb_flat = None
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        # Compute current Q-values
                    q_values, hidden = self.agent_q_network(obs_sb_flat, prev_actions_sb_flat, hidden=hidden)              # Shape: (sb_size * N, num_actions)
                    chosen_actions = actions_sb_flat.unsqueeze(1)             # Shape: (sb_size * N, 1)
                    chosen_q_values = q_values.gather(1, chosen_actions).squeeze(1)  # Shape: (sb_size * N)
                    chosen_q_values = chosen_q_values.view(sb_size, N_agents)  # Shape: (sb_size, N)

                    
                    # entropy regularization
                    # entropy += -torch.sum(F.softmax(q_values, dim=1) * F.log_softmax(q_values, dim=1), dim=1).mean()/T*(1-done_sb) # (sb_size * N)

                # Compute Double Q-learning targets
                    with torch.no_grad():
                        target_q_values, hidden_target = self.target_agent_q_network(next_obs_sb_flat, actions_sb_flat, hidden=hidden_target)  # (sb_size*N, num_actions)
                        _, max_actions = target_q_values.max(dim=1)  # (sb_size*N,)
                        max_target_q_values = target_q_values.gather(1, max_actions.unsqueeze(1)).squeeze(1)  # (sb_size*N,)
                        max_target_q_values = max_target_q_values.view(sb_size, N_agents)  # (episodes, agents)

                q_evals.append(chosen_q_values)
                q_targets.append(max_target_q_values)

                # Compute Q_tot and target Q_tot
            q_evals = torch.stack(q_evals, dim=1) # Shape: (sb_size, N, T-1)
            q_targets = torch.stack(q_targets, dim=1)

            dones_sb = dones[b:bs, 1:].contiguous()  # Shape: (1, T-1)
            # dones_sb = dones_sb.unsqueeze(0)  # Shape: (1, T-1)
            rewards_sb = rewards[b: bs, 1:].contiguous()  # Shape: (1, T-1)
            # rewards_sb = rewards_sb.unsqueeze(0)  # Shape: (1, T-1)
            # q_evals = q_evals.permute(0, 2, 1).contiguous() # Shape: (sb_size, T-1, N)
            # q_targets = q_targets.permute(0, 2, 1).contiguous()

            q_evals = q_evals.view(-1, *q_evals.shape[2:]).contiguous()  # Shape: (sb_size*(T-1), N)
            q_targets = q_targets.view(-1, *q_targets.shape[2:]).contiguous()

            s = state[b:bs, :-1].contiguous()  # Shape: ( T-1, C_s, H_s, W_s)
            # s = state[:, :-1].contiguous()  # Shape: ( T-1, C_s, H_s, W_s)
            # s = s.unsqueeze(0)  # Shape: (1, T-1, C_s, H_s, W_s)
            s = s.permute(0, 1, 4, 2, 3).contiguous()  # Shape: (1, T-1, C_s, H_s, W_s)
            s = s.view(-1, *s.shape[2:]).contiguous()

            n_s = state[b:bs, 1:].contiguous()
            # n_s = state[:, 1:].contiguous()
            # n_s = n_s.unsqueeze(0)
            n_s = n_s.permute(0, 1, 4, 2, 3).contiguous()
            n_s = n_s.view(-1, *n_s.shape[2:]).contiguous()
            
            # Compute Q_tot and target Q_tot
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                q_tot = self.mixing_network(q_evals, s)  # Shape: (sb_size, transitions)
                # q_tot = q_tot.squeeze(1)  # Shape: (sb_size,)

                with torch.no_grad():
                    target_q_tot = self.target_mixing_network(q_targets, n_s)  # Shape: (sb_size, transitions)
                    # Compute targets
                    # target_q_tot = target_q_tot.squeeze(1)  # Shape: (sb_size,)
                    rewards_sb = rewards_sb.view(-1, 1)  # Shape: (sb_size,)
                    dones_sb = dones_sb.view(-1, 1)  # Shape: (sb_size,)
                    targets = rewards_sb + self.gamma * (1 - dones_sb) * target_q_tot  # Shape: (sb_size, 1)

                masked_td = (q_tot - targets)*(1-dones_sb)  # Shape: (sb_size, 1)
                done_sum = torch.sum(1-dones_sb, dtype=torch.float32)
                if done_sum:
                    loss += (masked_td ** 2).sum()/done_sum/B  # Shape: (1,)
                    # entropy += -torch.sum(F.softmax(q_values, dim=1) * F.log_softmax(q_values, dim=1), dim=1).mean()/done_sum*(1-done_sb)/B # (sb_size * N)
                    # loss += 0.03*entropy
    # loss = loss/obs.size(0)
                    loss.backward()

                    # Gradient clipping
                    nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
                    nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)
            
                    
                    self.optimizer.step()
                    for name, param in self.named_parameters():
                        if param.grad is not None:
                            # Accumulate gradients
                            self.eligibility_traces[name] = self.gamma * self.lambda_ * self.eligibility_traces[name] + param.grad.data

                            # Update parameters using eligibility traces
                            param.data -= self.optimizer.param_groups[0]['lr'] * self.eligibility_traces[name]
                    self.scheduler1.step()
                    self.optimizer.zero_grad()

                total_loss += loss
        # Free up GPU memory
        del obs_sb, actions_sb,  next_obs_sb 
        del obs_sb_flat, actions_sb_flat, next_obs_sb_flat

        # self.scheduler1.step()
        return total_loss

    def update_target_hard(self):
        """
        Perform a hard update of target networks.
        """
        self.target_agent_q_network.load_state_dict(self.agent_q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def update_target_soft(self, tau: float = 0.1):
        """
        Perform a soft update of target networks.

        Args:
            tau (float): Soft update coefficient.
        """
        for target_param, param in zip(self.target_agent_q_network.parameters(), self.agent_q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def num_params(self) -> int:
        # calculate total paramater of AgentQNetwork and MixingNetwork
        return sum(p.numel() for p in self.agent_q_network.parameters()) + sum(p.numel() for p in self.mixing_network.parameters())



# if __name__ == "__main__":
    # Initialize environment
    # device = 
    # share = QNetwork((5, 13, 13), 5)
    # mixing = MixingNetwork(81)
    # # a = QMIX(5, (45, 45, 5), device)
    # # a = mixing()
    # # print(np.random( 13, 13, 5).shape)32
    # start= time.time()
    # # a.select_actions(np.random.rand( 13, 13, 5), None)
    # print(share.num_params(), mixing.num_params())
    # print(time.time()-start)