import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.models as models
from typing import List
from torch.nn import functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_float32_matmul_precision('medium')

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, dilation=dilation, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.ReLU(inplace=True)
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
    def __init__(self, in_features, out_features, dropout=0.1, negative_slope=0):
        super(ResidualFCBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.leaky_relu = nn.ReLU(inplace=True)
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
        out = self.leaky_relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += residual
        out = self.leaky_relu(out)
        return out

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

class AgentQNetwork(nn.Module):
    """
    Custom Q-Network for RL Agents without Pretrained Weights.
    """
    def __init__(self, num_actions=21, input_channels=5):
        super().__init__()
        
        self.input_conv = nn.Sequential(
            # Initial feature extraction with depthwise separable conv
            DepthwiseSeparableConv(input_channels, 16), 
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16, 16),
            # First block: 16->32 channels with spatial reduction
            ResidualBlock(16, 32, stride=2),  # Size reduction 13->7
            # nn.ReLU(inplace=True),
            ResidualBlock(32, 32),
            # Second block: 32->48 channels with spatial reduction 
            ResidualBlock(32, 64, stride=2),  # Size reduction 7->4
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            # Final block: efficient feature extraction
            DepthwiseSeparableConv(64, 128, stride=2),  # Size reduction 4->2
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            ResidualBlock(128, 128),
            DepthwiseSeparableConv(128, 128, stride=2),
            nn.Flatten()  # Size reduction 2->1
        )
        # Calculate the flattened feature size
        # Assuming input image size is (C, H, W) = (5, 45, 45)
        # After two downsampling layers with stride=2: H and W become 11 (45/2 -> 22, then 11)

        a_embed, id_embed = 64, 64 
        self.action_embed = nn.Embedding(21, a_embed) 
        self.agent_id_embed = nn.Embedding(81, id_embed)

        self.feature_dim = 128 + a_embed + id_embed
        self.hidden_dim = 256

        self.fc1 = nn.Linear(self.feature_dim, self.hidden_dim)         
        self.rnn = nn.GRUCell(self.hidden_dim, 256)

        self.fc = nn.Sequential(
            ResidualFCBlock(self.hidden_dim, 256),
            nn.Dropout(0.2),
            ResidualFCBlock(256, 256),
            nn.Dropout(0.2),
            nn.Linear(256, num_actions)
        )
        
        # Initialization for the final layer
        nn.init.kaiming_normal_(self.fc[-1].weight, mode='fan_in', nonlinearity='linear')
        nn.init.constant_(self.fc[-1].bias, 0)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRUCell):
                nn.init.kaiming_normal_(m.weight_ih)
                nn.init.kaiming_normal_(m.weight_hh)
                nn.init.constant_(m.bias_ih, 0)
                nn.init.constant_(m.bias_hh, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.kaiming_normal_(m.weight)

    # create a zeros tensor in case previous actions is None
    # and in case previous hidden state is None
    def init_hidden(self):
        # make hidden states on same device as model
        return torch.zeros(1, self.hidden_dim).to(device) # self.fc1.weight is used as a reference to the device



    def forward(self, x, a=None, id=None, hidden=None):
        """
        Forward pass for AgentQNetwork.

        Args:
            x (torch.Tensor): Input observations with shape (batch_size, C, H, W).

        Returns:
            torch.Tensor: Q-values for each action with shape (batch_size, num_actions).
        """
        x = x.to(device)
        # find index of padded observation
        original_x = x.clone()
        original_x = original_x.view(original_x.size(0), -1).abs().sum(dim=1)
        obs_mask = original_x == 0


        x = self.input_conv(x)  # Shape: (batch_size, 256)
        x = x.view(x.size(0), -1)  # Shape:     (batch_size, 256)
        # Provide default embeddings if a or id is None
        if a is not None and isinstance(a, torch.Tensor):
            a = a.to(device)
            action_mask = (a == 0)
            a = self.action_embed(a.long())
        elif isinstance(a, np.ndarray):
            a = torch.tensor(a, device=x.device, dtype=torch.long, requires_grad=False)
            action_mask = (a == 0)
            a = self.action_embed(a)
        else:
            a = torch.zeros(x.size(0), device=x.device,dtype=torch.long, requires_grad=False)
            action_mask = (a == 0)
            a = self.action_embed(a)
        
        if id is not None:
            id = self.agent_id_embed(id)
        else:
            size = int(x.size(0)/81)
            id = torch.tensor([[i for i in range(81)] for i in range(size)], device=x.device, dtype=torch.long, requires_grad=False)
            id = self.agent_id_embed(id)
            id = id.view(x.size(0), -1)
        combined_mask = obs_mask & action_mask
        if hidden is None:
            hidden = self.init_hidden()
            hidden = hidden.expand(x.size(0), -1).to(device)
        hidden = hidden.detach()
        # concatenate action embedding and state embedding
        x = torch.cat([x, a, id], dim=1)
        x = self.fc1(x) # Shape: (batch_size, 256+30+30)
        h = self.rnn(x, hidden) # Shape: (batch_size, 256)
        q_values = self.fc(h)  # Shape: (batch_size, num_actions)

        combined_mask = combined_mask.unsqueeze(1).expand(q_values.size())
        q_values = q_values.masked_fill(combined_mask, 0.0)
        return q_values, h
    
# -------------------------
# Mixing Network
# -------------------------
class MixingNetwork(nn.Module):
    def __init__(self, num_agents, state_shape=(45,45,5), embed_dim=128, use_attention=True):
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
            DepthwiseSeparableConv(state_shape[2], 8),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            ResidualBlock(8, 8),
            # Downsample 45->23
            DepthwiseSeparableConv(8, 16, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ResidualBlock(16, 16),
            # Downsample 23->12
            DepthwiseSeparableConv(16, 32, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualBlock(32, 32),           

            # Downsample 12->6
            DepthwiseSeparableConv(32, 64, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            ResidualBlock(64, 64),
            # Final reduction 6->1
            nn.Conv2d(64, 128, kernel_size=6),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            # nn.ReLU(inplace=True)
        )
        self.state_dim = 128

        # Hypernetworks for weights
        self.hyper_w_1 = nn.Sequential(
            ResidualFCBlock(self.state_dim, embed_dim, negative_slope=0),
            # nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_agents * embed_dim),
            nn.ReLU(inplace=True)
        )
        self.hyper_w_final = nn.Sequential(
            ResidualFCBlock(self.state_dim, embed_dim, negative_slope=0),
            # nn.ReLU(inplace=    True),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True)
        )

        # Hypernetworks for biases
        self.hyper_b_1 = ResidualFCBlock(self.state_dim, embed_dim, negative_slope=0)
        self.hyper_b_final = nn.Sequential(
            ResidualFCBlock(self.state_dim, embed_dim,negative_slope=0),
            # nn.ReLU(inplace=True),  
            nn.Linear(embed_dim, 1),
            nn.ReLU(inplace=True)
        )
        
        # Optional attention mechanism
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


        attn_output, _ = self.attention(hidden, hidden, hidden)  # (batch_size, 1, embed_dim)
        q_tot = torch.bmm(attn_output, w_final) + b_final  # (batch_size, 1, 1)


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
        # try:
        #     self.agent_q_network = torch.compile(self.agent_q_network)
        #     self.mixing_network = torch.compile(self.mixing_network)
        #     self.target_agent_q_network = torch.compile(self.target_agent_q_network)
        #     self.target_mixing_network = torch.compile(self.target_mixing_network)
        # except Exception as e:
        #     print(f"Compilation failed: {e}. Using regular execution.")
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
            self.optimizer, step_size=100, gamma=0.9
        )

        self.loss_fn = nn.SmoothL1Loss().to(device)
        # Log models with wandb
        # wandb.watch(self.agent_q_network, log_freq=100)
        # wandb.watch(self.mixing_network, log_freq=100)

    def select_actions(self, obs: torch.Tensor, prev_actions: None, 
                       agent_ids = None,
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

        N, H, W, C = obs.shape
        obs = obs.view(N,C,H,W).contiguous()
        obs = obs.float().to(self.device)  # Shape: (1, 1, N_agents, C, H, W)
        if prev_actions is not None:
            prev_actions = torch.tensor(prev_actions, dtype=torch.long).to(self.device)  # Shape: (1, 1, N_agents)
        with torch.no_grad():
            q_values, hidden = self.agent_q_network(obs, prev_actions, agent_ids, hidden)  # Shape: (1, 1, N_agents, num_actions)
        q_values = q_values.squeeze(0).squeeze(1)  # Shape: (N_agents, num_actions)
        q_values = q_values.cpu().numpy()

        actions = {}
        for i, agent in enumerate(self.agent_ids):
            if np.random.rand() < epsilon:
                action = np.random.randint(q_values.shape[1])
            else:
                action = np.argmax(q_values[i]).astype(np.uint16)
            actions[agent] = action
            # check if observation is padded observation ( all zeros) set action equal to 0
            if torch.sum(obs) == 0:
                actions[agent] = 0
        return actions, hidden
    
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

        # # Convert batch data to tensors on CPU initially
        # obs = torch.tensor(batch['o'], dtype=torch.float32)            # (B, T, N, H, W, C)
        # actions = torch.tensor(batch['a'], dtype=torch.int64)        # (B, T, N)
        # rewards = torch.tensor(batch['r'], dtype=torch.float32)     # (B, T,)
        # next_obs = torch.tensor(batch['n_o'], dtype=torch.float32)   # (B, T, N, H, W, C)
        # state = torch.tensor(batch['s'], dtype=torch.float32)         # (B, T, H_s, W_s, C_s)
        # next_state = torch.tensor(batch['n_s'], dtype=torch.float32)  # Same shape as state
        # dones = torch.tensor(batch['d'], dtype=torch.int64)            # (B, T,)
        # prev_actions = torch.tensor(batch['p_a'], dtype=torch.int64)  # (B, T, N)
        obs = batch['o']
        actions = batch['a'].long()
        rewards = batch['r']
        next_obs = batch['n_o']
        state = batch['s']
        next_state = batch['n_s']
        dones = batch['d']
        prev_actions = batch['p_a'].long()


        B, T = obs.shape[0], obs.shape[1]

        # Determine the number of sub-batches
        # sub_batch_size = self.sub_batch_size
        # num_sub_batches = (T + sub_batch_size - 1) // sub_batch_size  # Ceiling division

        total_loss = 0.0
        priorities = []
        # agent_ids = agent_ids.view(1, 81).expand(B, 81).view(-1).to(device)
        hidden = None
        hidden_target = None
        for i in range(T):
            # Process each episode individually
            loss_prio = 0
            # for sb in range(num_sub_batches):
            #     start = sb * sub_batch_size
            #     end = min((sb + 1) * sub_batch_size, T)

            # Extract sub-batch for the current episode
            obs_sb = obs[:,i].to(self.device)            # Shape: (sb_size, N, H, W, C)
            actions_sb = actions[:, i].to(self.device)      # Shape: (sb_size, N)
            rewards_sb = rewards[:, i].to(self.device)      # Shape: (sb_size,)
            next_obs_sb = next_obs[:,i ].to(self.device)    # Shape: (sb_size, N, H, W, C)
            state_sb = state[:, i].to(self.device)          # Shape: (sb_size, H_s, W_s, C_s)
            next_state_sb = next_state[:, i].to(self.device)  # Shape: (sb_size, H_s, W_s, C_s)
            dones_sb = dones[:, i].to(self.device)          # Shape: (sb_size,)
            prev_actions_sb = prev_actions[:, i].to(self.device)  # Shape: (sb_size, N)

            # Permute dimensions if necessary (e.g., from (sb_size, N, H, W, C) to (sb_size, N, C, H, W))
            obs_sb = obs_sb.permute(0, 1, 4, 2, 3).contiguous()        # Shape: (sb_size, N, C, H, W)
            next_obs_sb = next_obs_sb.permute(0, 1, 4, 2, 3).contiguous()

            # Flatten batch and agent dimensions for processing
            sb_size, N_agents = obs_sb.shape[0], obs_sb.shape[1]
            obs_sb_flat = obs_sb.view(-1, *obs_sb.shape[2:]).contiguous()          # Shape: (sb_size * N, C, H, W)
            actions_sb_flat = actions_sb.contiguous().view(-1).contiguous()                     # Shape: (sb_size * N)
            next_obs_sb_flat = next_obs_sb.view(-1, *next_obs_sb.shape[2:]).contiguous()  # Shape: (sb_size * N, C, H, W)
            prev_actions_sb_flat = prev_actions_sb.contiguous().view(-1).contiguous()  # Shape: (sb_size * N)

            # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            # Compute current Q-values
            q_values, hidden = self.agent_q_network(obs_sb_flat, prev_actions_sb_flat,  hidden=hidden)              # Shape: (sb_size * N, num_actions)
            chosen_actions = actions_sb_flat.unsqueeze(1)             # Shape: (sb_size * N, 1)
            chosen_q_values = q_values.gather(1, chosen_actions).squeeze(1)  # Shape: (sb_size * N)
            chosen_q_values = chosen_q_values.view(sb_size, N_agents)  # Shape: (sb_size, N)

            # entropy regularization
            entropy = -torch.sum(F.softmax(q_values, dim=1) * F.log_softmax(q_values, dim=1), dim=1).mean() # (sb_size * N)

        # Compute Double Q-learning targets
            with torch.no_grad():
                target_q_values, hidden_target = self.target_agent_q_network(next_obs_sb_flat, actions_sb_flat, hidden=hidden_target)  # (sb_size*N, num_actions)
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
            loss = self.loss_fn(q_tot, targets.detach()) - 0.03 * entropy
            # loss_prio += loss.item()
            # Backpropagation and optimization
            # self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.agent_q_network.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(self.mixing_network.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            # Free up GPU memory
            del obs_sb, actions_sb, rewards_sb, next_obs_sb, state_sb, next_state_sb, dones_sb
            del obs_sb_flat, actions_sb_flat, next_obs_sb_flat
            # torch.cuda.empty_cache()
            priorities.append(loss_prio)

            self.scheduler1.step()
        # Return the average loss over all sub-batches
        return total_loss, priorities

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
#     # Test QMIX
#     num_agents = 81 
#     state_shape = (45, 45, 5)
#     agent_ids = [f"blue_{i}" for i in range(81)]
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     num_actions = 21
#     qmix = QMIX(num_agents=num_agents, state_shape=state_shape, agent_ids=agent_ids, num_actions=num_actions, device=device)
#     print(f"Total number of parameters: {qmix.num_params()})")
#     input = torch.randn(162, 5, 13, 13)
#     actions = torch.randint(0, 20, (162,))
#     agent_ids = torch.tensor([i for i in range(81)])
#     aqn = AgentQNetwork(num_actions=num_actions).to(device)
#     # print(aqn(input).shape)
#     mn = MixingNetwork(num_agents=num_agents, state_shape=state_shape).to(device)
#     agent_qs = torch.randn(2, 81)
#     state = torch.randn(2, 5, 45, 45)
#     print(mn(agent_qs.to(device), state.to(device)).shape)
#     a = aqn(input.to(device) ,actions.to(device), agent_ids.to(device))
