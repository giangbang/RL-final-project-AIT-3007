import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from typing import List
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision('medium')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
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
        out = self.leaky_relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.leaky_relu(out)

        return out

class ResidualFCBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.1):
        super(ResidualFCBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True)
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
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.fc2.bias, 0)

    def forward(self, x):
        residual = self.residual(x)
        out = self.fc1(x)
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.leaky_relu(out)
        return out

class AgentQNetwork(nn.Module):
    """
    Custom Q-Network for RL Agents without Pretrained Weights.
    """
    def __init__(self, num_actions=21, input_channels=5):
        super().__init__()
        
        # Initial convolution layers
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),  # Downsample
            ResidualBlock(128, 128),
            ResidualBlock(128, 256, stride=2),  # Downsample
            ResidualBlock(256, 256),
            ResidualBlock(256, 512, stride=2),  # Downsample
        )
        
        # Calculate the flattened feature size
        # Assuming input image size is (C, H, W) = (5, 45, 45)
        # After two downsampling layers with stride=2: H and W become 11 (45/2 -> 22, then 11)
        self.feature_dim = 512 * 2 * 2  # Adjust based on actual input size
        
        self.fc = nn.Sequential(
            ResidualFCBlock(self.feature_dim, 512),
            ResidualFCBlock(512, 256),
            ResidualFCBlock(256, 128),
            nn.Linear(128, num_actions)
        )
        
        # Initialization for the final layer
        nn.init.kaiming_normal_(self.fc[-1].weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.fc[-1].bias, 0)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass for AgentQNetwork.

        Args:
            x (torch.Tensor): Input observations with shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Q-values for each action with shape (batch_size, num_actions).
        """
        x = self.input_conv(x)  # Shape: (batch_size, 256, 11, 11)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 256*11*11)
        q_values = self.fc(x)  # Shape: (batch_size, num_actions)
        return q_values
    
# -------------------------
# Mixing Network
# -------------------------
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_shape=(45,45,5), embed_dim=256, use_attention=True):
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

        # Simplified state encoder
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_shape[2], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResidualBlock(64, 128, stride=2),  # Downsample
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResidualBlock(128, 256, stride=2),  # Downsample
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # Output: (batch_size, 64, 1, 1)
        )
        self.state_dim = 256

        # Hypernetworks for weights
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_agents * embed_dim)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Hypernetworks for biases
        self.hyper_b_1 = nn.Linear(self.state_dim, embed_dim)
        self.hyper_b_final = nn.Sequential(
            nn.Linear(self.state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Optional attention mechanism
        if self.use_attention:
            self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=4, batch_first=True)

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
            agent_qs (torch.Tensor): Agent Q-values with shape (batch_size, num_agents).
            state (torch.Tensor): Global state with shape (batch_size, C, H, W).
        
        Returns:
            torch.Tensor: Total Q-value with shape (batch_size, 1).
        """
        batch_size = state.size(0)
        state = self.state_encoder(state).view(batch_size, -1)  # Shape: (batch_size, state_dim)

        # Generate hypernetwork weights and biases
        w1 = torch.abs(self.hyper_w_1(state)).view(batch_size, self.num_agents, self.embed_dim)  # (batch_size, num_agents, embed_dim)
        b1 = self.hyper_b_1(state).view(batch_size, 1, self.embed_dim)  # (batch_size, 1, embed_dim)
        hidden = F.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1)  # (batch_size, 1, embed_dim)

        w_final = torch.abs(self.hyper_w_final(state)).view(batch_size, self.embed_dim, 1)  # (batch_size, embed_dim, 1)
        b_final = self.hyper_b_final(state).view(batch_size, 1, 1)  # (batch_size, 1, 1)

        if self.use_attention:
            # Apply attention mechanism
            attn_output, _ = self.attention(hidden, hidden, hidden)  # (batch_size, 1, embed_dim)
            q_tot = torch.bmm(attn_output, w_final) + b_final  # (batch_size, 1, 1)
        else:
            q_tot = torch.bmm(hidden, w_final) + b_final  # (batch_size, 1, 1)

        q_tot = q_tot.view(batch_size, 1)  # (batch_size, 1)
        return q_tot
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from magent2.environments import battle_v4

class QMIX:
    """
    QMIX implementation for multi-agent reinforcement learning.
    """
    def __init__(
        self,
        num_agents: int,
        state_shape: Tuple[int, int, int],
        agent_ids: List[str],
        device: torch.device,
        num_actions: int = 21,
        lr: float = 1e-3,
        gamma: float = 0.99,
        sub_batch_size: int = 64,
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.agent_ids = agent_ids
        self.device = device
        self.sub_batch_size = sub_batch_size

        # Initialize networks
        self.agent_q_network = AgentQNetwork(num_actions=num_actions).to(device)
        self.mixing_network = MixingNetwork(num_agents, state_shape=state_shape).to(device)
        self.target_agent_q_network = AgentQNetwork(num_actions=num_actions).to(device)
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
        self.optimizer = optim.AdamW(
            params=list(self.agent_q_network.parameters()) + list(self.mixing_network.parameters()),
            lr=lr,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6,
        )
        self.scheduler1 = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.8
        )

        self.loss_fn = nn.SmoothL1Loss().to(device)
        # Log models with wandb
        wandb.watch(self.agent_q_network, log_freq=100)
        wandb.watch(self.mixing_network, log_freq=100)

    def select_actions(self, obs: torch.Tensor,  epsilon: float = 0.1) -> Dict[str, int]:
        """
        Select actions using epsilon-greedy policy.

        Args:
            obs (torch.Tensor): Observations with shape (N_agents, C, H, W).
            epsilon (float): Probability of choosing a random action.

        Returns:
            Dict[str, int]: Mapping from agent IDs to selected actions.
        """
        # Add batch dimension

        N, H, W, C = obs.shape
        obs = obs.view(N,C,H,W).contiguous()
        obs = obs.float().to(self.device)  # Shape: (1, 1, N_agents, C, H, W)
        with torch.no_grad():
            q_values = self.agent_q_network(obs)  # Shape: (1, 1, N_agents, num_actions)
        q_values = q_values.squeeze(0).squeeze(1)  # Shape: (N_agents, num_actions)
        q_values = q_values.cpu().numpy()

        actions = {}
        for i, agent in enumerate(self.agent_ids):
            if np.random.rand() < epsilon:
                action = np.random.randint(q_values.shape[1])
            else:
                action = np.argmax(q_values[i]).astype(np.uint16)
            actions[agent] = action
        return actions
    
    def update(self, batch: Dict[str, np.ndarray], ep: int) -> float:
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

        # Convert batch data to tensors on CPU initially
        obs = torch.tensor(batch['obs'], dtype=torch.float32)            # (B, T, N, H, W, C)
        actions = torch.tensor(batch['actions'], dtype=torch.int64)        # (B, T, N)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32)     # (B, T,)
        next_obs = torch.tensor(batch['next_obs'], dtype=torch.float32)   # (B, T, N, H, W, C)
        state = torch.tensor(batch['state'], dtype=torch.float32)         # (B, T, H_s, W_s, C_s)
        next_state = torch.tensor(batch['next_state'], dtype=torch.float32)  # Same shape as state
        dones = torch.tensor(batch['dones'], dtype=torch.int64)            # (B, T,)

        B, T = obs.shape[0], obs.shape[1]

        # Determine the number of sub-batches
        sub_batch_size = self.sub_batch_size
        num_sub_batches = (T + sub_batch_size - 1) // sub_batch_size  # Ceiling division

        total_loss = 0.0
        priorities = []
        for i in range(B):
            # Process each episode individually
            loss_prio = 0
            for sb in range(num_sub_batches):
                start = sb * sub_batch_size
                end = min((sb + 1) * sub_batch_size, T)

                # Extract sub-batch for the current episode
                obs_sb = obs[i, start:end].to(self.device)             # Shape: (sb_size, N, H, W, C)
                actions_sb = actions[i, start:end].to(self.device)      # Shape: (sb_size, N)
                rewards_sb = rewards[i, start:end].to(self.device)      # Shape: (sb_size,)
                next_obs_sb = next_obs[i, start:end].to(self.device)    # Shape: (sb_size, N, H, W, C)
                state_sb = state[i, start:end].to(self.device)          # Shape: (sb_size, H_s, W_s, C_s)
                next_state_sb = next_state[i, start:end].to(self.device)  # Shape: (sb_size, H_s, W_s, C_s)
                dones_sb = dones[i, start:end].to(self.device)          # Shape: (sb_size,)

                # Permute dimensions if necessary (e.g., from (sb_size, N, H, W, C) to (sb_size, N, C, H, W))
                obs_sb = obs_sb.permute(0, 1, 4, 2, 3).contiguous()        # Shape: (sb_size, N, C, H, W)
                next_obs_sb = next_obs_sb.permute(0, 1, 4, 2, 3).contiguous()

                # Flatten batch and agent dimensions for processing
                sb_size, N_agents = obs_sb.shape[0], obs_sb.shape[1]
                obs_sb_flat = obs_sb.view(-1, *obs_sb.shape[2:]).contiguous()          # Shape: (sb_size * N, C, H, W)
                actions_sb_flat = actions_sb.view(-1).contiguous()                     # Shape: (sb_size * N)
                next_obs_sb_flat = next_obs_sb.view(-1, *next_obs_sb.shape[2:]).contiguous()  # Shape: (sb_size * N, C, H, W)

                # Compute current Q-values
                q_values = self.agent_q_network(obs_sb_flat)              # Shape: (sb_size * N, num_actions)
                chosen_actions = actions_sb_flat.unsqueeze(1)             # Shape: (sb_size * N, 1)
                chosen_q_values = q_values.gather(1, chosen_actions).squeeze(1)  # Shape: (sb_size * N)
                chosen_q_values = chosen_q_values.view(sb_size, N_agents)  # Shape: (sb_size, N)

                # entropy regularization
                entropy = -torch.sum(F.softmax(q_values, dim=1) * F.log_softmax(q_values, dim=1), dim=1).mean() # (sb_size * N)

               # Compute Double Q-learning targets
                with torch.no_grad():
                    target_q_values = self.target_agent_q_network(next_obs_sb_flat)  # (sb_size*N, num_actions)
                    _, max_actions = target_q_values.max(dim=1)  # (sb_size*N,)
                    max_target_q_values = target_q_values.gather(1, max_actions.unsqueeze(1)).squeeze(1)  # (sb_size*N,)
                    max_target_q_values = max_target_q_values.view(sb_size, N_agents)  # (B*T, N)


                state_sb = state_sb.permute(0, 3, 1, 2).contiguous()  # Shape: (sb_size, C_s, H_s, W_s)
                next_state_sb = next_state_sb.permute(0, 3, 1, 2).contiguous()
                
                # Compute Q_tot and target Q_tot

                q_tot = self.mixing_network(chosen_q_values, state_sb)  # Shape: (sb_size, 1)
                q_tot = q_tot.squeeze(1)  # Shape: (sb_size,)

                with torch.no_grad():
                    target_q_tot = self.target_mixing_network(max_target_q_values, next_state_sb)  # Shape: (sb_size, 1)
                    # Compute targets
                    target_q_tot = target_q_tot.squeeze(1)  # Shape: (sb_size,)
                    targets = rewards_sb + self.gamma * (1 - dones_sb) * target_q_tot  # Shape: (sb_size, 1)

                # Compute loss
                loss = self.loss_fn(q_tot, targets.detach()) - 0.01 * entropy
                total_loss += loss.item()
                loss_prio += loss.item()
                # Backpropagation and optimization
                # self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)


                # Free up GPU memory
                del obs_sb, actions_sb, rewards_sb, next_obs_sb, state_sb, next_state_sb, dones_sb
                del obs_sb_flat, actions_sb_flat, next_obs_sb_flat
                # torch.cuda.empty_cache()
            priorities.append(loss_prio)
        self.scheduler1.step()
        # Return the average loss over all sub-batches
        average_loss = total_loss / (B * num_sub_batches)
        return average_loss, priorities

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