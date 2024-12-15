import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from typing import List
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision('high')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
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
    This network takes agent observations as input and produces Q-values for each action.
    """
    def __init__(self, num_actions=21):
        super().__init__()
        
        # Initial conv to process the input
        self.input_conv = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 16),
        )
        
        # Final convolution to match Swin input
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        
        # Swin Transformer for feature extraction
        self.swin = models.swin_v2_t(weights='IMAGENET1K_V1')

        # setting learning rate of swin to small number
     


        # turn off gradients for the swin model
        # for param in self.swin.parameters():
        #     param.requires_grad = False
        
        # Final fully connected layer
        self.fc = nn.Linear(1000, num_actions)
        
        self._initialize_weights()

    
    def _initialize_weights(self):
        for m in [self.res_blocks.modules(), self.input_conv.modules(), self.final_conv.modules(), self.fc.modules()]:
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
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                m.weight.data.mul_(0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)



    def forward(self, obs):
        # obs shape: (batch, n_eps,  N_agents, 13,13,5)
        # We'll reshape to feed each agent separately if needed
        B, N, T, H, W, C = obs.shape
        obs = obs.permute(0,1,2,5,3,4).contiguous() # (B, T, N, C, H, W)
        obs = obs.view(-1, C, H, W).contiguous() # (B*T*N, C, H, W)
        
        # Process through conv layers with residual connections
        x = self.input_conv(obs)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        
        # Ensure the input size matches Swin's requirements
        if x.size(-1) != 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Process through Swin
        features = self.swin(x)
        
        # Final Q-values
        q = self.fc(features)
        q = q.view(B*T, N, -1).contiguous()  # (B, T, N, num_actions)
        
        return q
    def get_params_groups(self):
        """
        Group parameters for different learning rates
        """
        swin_params = list(self.swin.parameters())
        other_params = (list(self.input_conv.parameters()) + 
                       list(self.res_blocks.parameters()) +
                       list(self.final_conv.parameters()) +
                       list(self.fc.parameters()))
        
        return {
            'swin': swin_params,
            'other': other_params
        }

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
            ResidualBlock(5, 16),
            ResidualBlock(16, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 16),
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
        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, agent_qs, state):
        # agent_qs: (B,T, N)
        # state: (B, T, H, W, C)
        B, T, H, W, C = state.size()

        # Encode state
        state = state.permute(0,1,4,2,3).contiguous().float() # (B,T, C, H, W)
        state = state.view(-1, 5, 45, 45).contiguous() # (B*T, C, H, W)
        # agent_qs = agent_qs.view(B*T, N).contiguous() # (B*T, N)
        s = self.conv(state)
        s = s.view(B*T, -1) # (B, 32)

        w1 = torch.abs(self.hyper_w_1(s)).view(B*T, self.num_agents, self.embedding_dim) # ensure non-negativity if needed
        b1 = self.hyper_b_1(s).view(B*T, 1, self.embedding_dim)
        hidden = torch.relu(torch.bmm(agent_qs.unsqueeze(1), w1) + b1) # (B*T, 1, embed_dim)

        w_final = torch.abs(self.hyper_w_final(s)).view(B*T, self.embedding_dim, 1)
        b_final = self.hyper_b_final(s).view(B*T, 1, 1)

        q_tot = torch.bmm(hidden, w_final) + b_final # (B*T, 1, 1)
        q_tot = q_tot.view(B*T, 1) # (B, T, 1)
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
    ):
        self.num_agents = num_agents
        self.gamma = gamma
        self.agent_ids = agent_ids
        self.device = device


        # Initialize networks
        self.agent_q_network = AgentQNetwork(num_actions=num_actions).to(device)
        self.mixing_network = MixingNetwork(num_agents, state_shape=state_shape).to(device)
        self.target_agent_q_network = AgentQNetwork(num_actions=num_actions).to(device)
        self.target_mixing_network = MixingNetwork(num_agents, state_shape=state_shape).to(device)
        
        # Get parameter groups
        param_groups = []
        
        ps_groups = self.agent_q_network.get_params_groups()
        # Add Swin parameters with small learning rate
        param_groups.append({
            'params': ps_groups['swin'],
            'lr': lr * 0.02  # 50x smaller learning rate for Swin
        })
        
        # Add other parameters with normal learning rate
 
                
        param_groups.append({
            'params': (
                list(ps_groups['other']) +
                list(self.mixing_network.parameters())
            ),
            'lr': lr
        })
        # Compile networks for potential speed-up (PyTorch 2.0+)
        try:
            self.agent_q_network = torch.compile(self.agent_q_network)
            self.mixing_network = torch.compile(self.mixing_network)
            self.target_agent_q_network = torch.compile(self.target_agent_q_network)
            self.target_mixing_network = torch.compile(self.target_mixing_network)
        except Exception as e:
            print(f"Compilation failed: {e}. Using regular execution.")
            # If torch.compile fails (e.g., not using PyTorch 2.0), proceed without compiling
        # torch.nn.DataParallel(self.agent_q_network)
        # torch.nn.DataParallel(self.mixing_network)
        # torch.nn.DataParallel(self.target_agent_q_network)
        # torch.nn.DataParallel(self.target_mixing_network)
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
        obs = obs.unsqueeze(0).unsqueeze(1).to(self.device)  # Shape: (1, 1, N_agents, C, H, W)
        with torch.no_grad():
            q_values = self.agent_q_network(obs)  # Shape: (1, 1, N_agents, num_actions)
        q_values = q_values.squeeze(0).squeeze(1)  # Shape: (N_agents, num_actions)
        q_values = q_values.cpu().numpy()

        actions = {}
        for i, agent in enumerate(self.agent_ids):
            if np.random.rand() < epsilon:
                action = np.random.randint(q_values.shape[1])
            else:
                action = np.argmax(q_values[i])
            actions[agent] = action
        return actions

    def update(self, batch: Dict[str, np.ndarray]) -> float:
        """
        Update the QMIX networks based on a batch of experiences.

        Args:
            batch (Dict[str, np.ndarray]): Batch of experiences.

        Returns:
            float: The computed loss value.
        """
        # Convert batch data to tensors
        obs = torch.tensor(batch['obs'], dtype=torch.float32).to(self.device)            # (B, T, N, H, W, C)
        actions = torch.tensor(batch['actions'], dtype=torch.long).to(self.device)        # (B, T, N)
        rewards = torch.tensor(batch['rewards'], dtype=torch.float32).to(self.device)     # (B, T, N)
        next_obs = torch.tensor(batch['next_obs'], dtype=torch.float32).to(self.device)   # (B, T, N, H, W, C)
        state = torch.tensor(batch['state'], dtype=torch.float32).to(self.device)         # (B, T, H_s, W_s, C_s)
        next_state = torch.tensor(batch['next_state'], dtype=torch.float32).to(self.device)  # Same shape as state
        dones = torch.tensor(batch['dones'], dtype=torch.float32).to(self.device)         # (B, T, N)


        # Current Q-values
        q_values = self.agent_q_network(obs)  # Shape: (B, T, N, num_actions)
        actions = actions.unsqueeze(-1).expand(-1, -1, q_values.size(-1))  # Shape: (B, N, num_actions)
        chosen_q = torch.gather(q_values, dim=2, index=actions).squeeze(-1)  # Shape: (B, N)

        # Target Q-values
        with torch.no_grad():
            target_q_values = self.target_agent_q_network(next_obs)  # Shape: (B, N, num_actions)
            max_actions = torch.argmax(target_q_values, dim=2, keepdim=True)  # Shape: (B, N, 1)
            max_q = torch.gather(target_q_values, 2, max_actions).squeeze(-1)  # Shape: (B, N)

        # Compute Q_tot and target Q_tot
        q_tot = self.mixing_network(chosen_q, state)  # Shape: (B, 1)
        target_q_tot = self.target_mixing_network(max_q, next_state)  # Shape: (B, 1)
        target = rewards.unsqueeze(1) + self.gamma * (1 - dones.unsqueeze(1)) * target_q_tot  # Shape: (B, 1)

        # Compute loss
        loss = nn.MSELoss()(q_tot, target.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
        nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    def update_target_hard(self):
        """
        Perform a hard update of target networks.
        """
        self.target_agent_q_network.load_state_dict(self.agent_q_network.state_dict())
        self.target_mixing_network.load_state_dict(self.mixing_network.state_dict())

    def update_target_soft(self, tau: float = 0.4):
        """
        Perform a soft update of target networks.

        Args:
            tau (float): Soft update coefficient.
        """
        for target_param, param in zip(self.target_agent_q_network.parameters(), self.agent_q_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.target_mixing_network.parameters(), self.mixing_network.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)