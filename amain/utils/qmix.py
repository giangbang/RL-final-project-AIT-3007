import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from typing import List
from torch.nn import functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('medium')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Add skip connection if input and output channels differ
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out

class AgentQNetwork(nn.Module):
    """
    Simplified Q-Network for agents with optional Swin Transformer.
    """
    def __init__(self, num_actions=21, use_swin=True):
        super().__init__()
        
        # Initial convolution layers
        self.input_conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            ResidualBlock(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, padding=1, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, dilation=2),
        )
        
        # Optional Swin Transformer
        self.use_swin = use_swin
        if self.use_swin:
            self.swin = models.efficientnet_b0(weights='IMAGENET1K_V1')
            # Allow fine-tuning with a smaller learning rate
            for param in self.swin.parameters():
                param.requires_grad = False 
            
            # Modify Swin to output desired feature size
            self.swin.head = nn.Identity()
            swin_output_dim = 1000 # Swin-T base output dimension
        else:
            # If not using Swin, add additional convolutional layers
            self.additional_conv = nn.Sequential(
                ResidualBlock(64, 128),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1))
            )
            swin_output_dim = 128
        
        # Final fully connected layer
        self.fc = nn.Linear(swin_output_dim, num_actions)
        
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

    def forward(self, x):
        """
        Forward pass for AgentQNetwork.

        Args:
            x (torch.Tensor): Input observations with shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Q-values for each action with shape (batch_size, num_actions).
        """
        x = self.input_conv(x)  # Shape: (batch_size, 64, H, W)
        
        if self.use_swin:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
            features = self.swin(x)  # Shape: (batch_size, 768)
        else:
            x = self.additional_conv(x)  # Shape: (batch_size, 128, 1, 1)
            features = x.view(x.size(0), -1)  # Shape: (batch_size, 128)
        
        q_values = self.fc(features)  # Shape: (batch_size, num_actions)
        return q_values


# -------------------------
# Mixing Network
# -------------------------
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_shape=(45,45,5), embed_dim=32, use_attention=True):
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
            nn.AdaptiveAvgPool2d((1,1))  # Output: (batch_size, 64, 1, 1)
        )
        self.state_dim = 64

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
        
        # Get parameter groups
        # Allow Swin parameters to be fine-tuned with a lower learning rate
        param_groups = [
            {
                'params': list(self.agent_q_network.input_conv.parameters()) + 
                        #   list(self.agent_q_network..parameters()) +
                        #   list(self.agent_q_network.final_conv.parameters()) +
                          list(self.agent_q_network.fc.parameters()) +
                          list(self.mixing_network.parameters()),
                'lr': lr
            },
            {
                'params': list(self.agent_q_network.swin.parameters()),
                'lr': lr * 0.1  # 10x smaller learning rate for Swin
            }
        ]
        # Compile networks for potential speed-up (PyTorch 2.0+)
        try:
            self.agent_q_network = torch.compile(self.agent_q_network)
            self.mixing_network = torch.compile(self.mixing_network)
            self.target_agent_q_network = torch.compile(self.target_agent_q_network)
            self.target_mixing_network = torch.compile(self.target_mixing_network)
        except Exception as e:
            print(f"Compilation failed: {e}. Using regular execution.")
        # If torch.compile fails (e.g., not using PyTorch 2.0), proceed without compiling

        # self.agent_q_network = torch.nn.DataParallel(self.agent_q_network)
        # self.mixing_network = torch.nn.DataParallel(self.mixing_network)
        # self.target_agent_q_network = torch.nn.DataParallel(self.target_agent_q_network)
        # self.target_mixing_network = torch.nn.DataParallel(self.target_mixing_network)
        # Synchronize target networks with main networks
        self.update_target_hard()

        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            params=param_groups,
            lr=lr,
            weight_decay=1e-4,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=2, eta_min=1e-6,
        )
        self.scheduler1 = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=32, gamma=0.8
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
        obs = obs.to(self.device)  # Shape: (1, 1, N_agents, C, H, W)
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

        for i in range(B):
            # Process each episode individually
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
                loss = self.loss_fn(q_tot, targets.detach())
                total_loss += loss.item()

                # Backpropagation and optimization
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
                nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.scheduler.step()

                # Free up GPU memory
                del obs_sb, actions_sb, rewards_sb, next_obs_sb, state_sb, next_state_sb, dones_sb
                del obs_sb_flat, actions_sb_flat, next_obs_sb_flat
                # torch.cuda.empty_cache()
        
        if ep % 10 == 0:
            self.scheduler1.step()
        # Return the average loss over all sub-batches
        average_loss = total_loss / (B * num_sub_batches)
        return average_loss

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