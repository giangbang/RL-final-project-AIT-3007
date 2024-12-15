import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    """
    Recurrent Neural Network module for QMIX agents.

    Args:
        input_shape (int): Dimension of input features (obs + n_actions + n_agents if applicable).
        args (Namespace): Configuration arguments containing rnn_hidden_dim and n_actions.
    """
    def __init__(self, input_shape: int, args):
        super(RNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, obs: torch.Tensor, hidden_state: torch.Tensor):
        """
        Forward pass through the RNN.

        Args:
            obs (torch.Tensor): Input observations, shape (batch_size, input_shape).
            hidden_state (torch.Tensor): Hidden state from previous step, shape (batch_size, rnn_hidden_dim).

        Returns:
            q (torch.Tensor): Q-values for actions, shape (batch_size, n_actions).
            h (torch.Tensor): Updated hidden state, shape (batch_size, rnn_hidden_dim).
        """
        x = F.relu(self.fc1(obs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixNet(nn.Module):
    """
    QMix Network to combine individual agent Q-values into a global Q-value.

    Args:
        args (Namespace): Configuration arguments containing state_shape, n_agents,
                         qmix_hidden_dim, hyper_hidden_dim, and two_hyper_layers.
    """
    def __init__(self, args):
        super(QMixNet, self).__init__()
        self.args = args

        # Define hypernetworks
        if args.two_hyper_layers:
            self.hyper_w1 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim)
            )
            self.hyper_w2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim)
            )
        else:
            self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
            self.hyper_w2 = nn.Linear(args.state_shape, self.args.qmix_hidden_dim)

        # Bias hypernetworks
        self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
        if args.two_hyper_layers:
            self.hyper_b2 = nn.Sequential(
                nn.Linear(args.state_shape, args.hyper_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.hyper_hidden_dim, 1)
            )
        else:
            self.hyper_b2 = nn.Linear(args.state_shape, 1)

    def forward(self, q_values: torch.Tensor, states: torch.Tensor):
        """
        Forward pass through the QMix network.

        Args:
            q_values (torch.Tensor): Individual agent Q-values, shape (batch_size, n_agents).
            states (torch.Tensor): Global state, shape (batch_size, state_shape).

        Returns:
            q_total (torch.Tensor): Combined Q-value, shape (batch_size, 1).
        """
        batch_size = q_values.size(0)

        # Compute hypernetwork weights
        w1 = torch.abs(self.hyper_w1(states))  # Ensure positivity
        b1 = self.hyper_b1(states)

        # Reshape weights and biases
        w1 = w1.view(batch_size, self.args.n_agents, self.args.qmix_hidden_dim)
        b1 = b1.view(batch_size, 1, self.args.qmix_hidden_dim)

        # Compute hidden layer
        hidden = F.elu(torch.bmm(q_values.unsqueeze(1), w1) + b1)  # (batch_size, 1, qmix_hidden_dim)

        # Compute second hypernetwork weights and biases
        w2 = torch.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)

        w2 = w2.view(batch_size, self.args.qmix_hidden_dim, 1)
        b2 = b2.view(batch_size, 1, 1)

        # Compute final Q_total
        q_total = torch.bmm(hidden, w2) + b2  # (batch_size, 1, 1)
        q_total = q_total.view(batch_size, -1)  # (batch_size, 1)
        return q_total


class QMIX:
    """
    QMIX algorithm implementation for multi-agent reinforcement learning.

    Args:
        args (Namespace): Configuration arguments containing model parameters and training settings.
    """
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if self.args.cuda and torch.cuda.is_available() else "cpu")
        
        self.n_actions = self.args.n_actions
        self.n_agents = self.args.n_agents
        self.state_shape = self.args.state_shape
        self.obs_shape = self.args.obs_shape

        input_shape = self.obs_shape
        if self.args.last_action:
            input_shape += self.n_actions
        if self.args.reuse_network:
            input_shape += self.n_agents

        # Initialize neural networks
        self.eval_rnn = RNN(input_shape, self.args).to(self.device)
        self.target_rnn = RNN(input_shape, self.args).to(self.device)
        self.eval_qmix_net = QMixNet(self.args).to(self.device)
        self.target_qmix_net = QMixNet(self.args).to(self.device)

        # Load model if specified
        self.model_dir = os.path.join(self.args.model_dir, self.args.alg, self.args.map)
        if self.args.load_model:
            path_rnn = os.path.join(self.model_dir, 'rnn_net_params.pkl')
            path_qmix = os.path.join(self.model_dir, 'qmix_net_params.pkl')
            if os.path.exists(path_rnn) and os.path.exists(path_qmix):
                self.eval_rnn.load_state_dict(torch.load(path_rnn, map_location=self.device))
                self.eval_qmix_net.load_state_dict(torch.load(path_qmix, map_location=self.device))
                print(f'Successfully loaded the model: {path_rnn} and {path_qmix}')
            else:
                raise FileNotFoundError(f"Model files not found in {self.model_dir}.")

        # Synchronize target networks
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        # Set up optimizer
        self.eval_parameters = list(self.eval_qmix_net.parameters()) + list(self.eval_rnn.parameters())
        if self.args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=self.args.lr)
        else:
            raise ValueError(f"Unsupported optimizer type: {self.args.optimizer}")

        # Initialize hidden states
        self.eval_hidden = None
        self.target_hidden = None

        print('Initialized QMIX algorithm.')

    def learn(self, batch: dict, max_episode_len: int, train_step: int, epsilon=None):
        """
        Perform a learning step with a batch of experiences.

        Args:
            batch (dict): Batch of experiences.
            max_episode_len (int): Maximum length of episodes in the batch.
            train_step (int): Current training step.
            epsilon (float, optional): Epsilon value for exploration (unused).
        """
        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)

        # Move batch data to the correct device and convert to tensors
        for key, value in batch.items():
            if key == 'u':
                batch[key] = torch.tensor(value, dtype=torch.long, device=self.device)
            else:
                batch[key] = torch.tensor(value, dtype=torch.float32, device=self.device)

        s = batch['s']
        s_next = batch['s_next']
        u = batch['u']
        r = batch['r']
        avail_u = batch['avail_u']
        avail_u_next = batch['avail_u_next']
        terminated = batch['terminated']
        mask = 1 - batch["padded"].float()

        # Get Q-values for current and next states
        q_evals, q_targets = self.get_q_values(batch, max_episode_len)

        # Gather Q-values for actions taken
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)

        # Mask unavailable actions in target Q-values
        q_targets[avail_u_next == 0.0] = -1e9  # Use a large negative value instead of -9999999 for numerical stability
        q_targets = q_targets.max(dim=3)[0]

        # Compute total Q-values using QMix networks
        q_total_eval = self.eval_qmix_net(q_evals, s)
        q_total_target = self.target_qmix_net(q_targets, s_next)

        # Compute target Q-values
        targets = r + self.args.gamma * q_total_target * (1 - terminated)

        # Compute TD error
        td_error = q_total_eval - targets.detach()
        masked_td_error = mask * td_error

        # Compute loss
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        # Update target networks periodically
        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch: dict, transition_idx: int):
        """
        Prepare inputs for the RNN networks by including last actions and agent IDs.

        Args:
            batch (dict): Batch of experiences.
            transition_idx (int): Current transition index.

        Returns:
            inputs (torch.Tensor): Prepared inputs for current transition.
            inputs_next (torch.Tensor): Prepared inputs for next transition.
        """
        obs = batch['o'][:, transition_idx]
        obs_next = batch['o_next'][:, transition_idx]
        u_onehot = batch['u_onehot']

        episode_num = obs.size(0)
        inputs = [obs]
        inputs_next = [obs_next]

        if self.args.last_action:
            if transition_idx == 0:
                # Previous action is zero for the first transition
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx], device=self.device))
            else:
                inputs.append(u_onehot[:, transition_idx - 1].to(self.device))
            inputs_next.append(u_onehot[:, transition_idx].to(self.device))

        if self.args.reuse_network:
            # Add agent IDs as one-hot vectors
            agent_ids = torch.eye(self.n_agents, device=self.device).unsqueeze(0).expand(episode_num, -1, -1)
            inputs.append(agent_ids.reshape(episode_num * self.n_agents, -1))
            inputs_next.append(agent_ids.reshape(episode_num * self.n_agents, -1))

        # Concatenate all inputs
        inputs = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs], dim=1)
        inputs_next = torch.cat([x.reshape(episode_num * self.n_agents, -1) for x in inputs_next], dim=1)

        return inputs, inputs_next

    def get_q_values(self, batch: dict, max_episode_len: int):
        """
        Compute Q-values for all transitions in the batch.

        Args:
            batch (dict): Batch of experiences.
            max_episode_len (int): Maximum length of episodes in the batch.

        Returns:
            q_evals (torch.Tensor): Q-values from evaluation network, shape (episode_num, max_episode_len, n_agents, n_actions).
            q_targets (torch.Tensor): Q-values from target network, shape (episode_num, max_episode_len, n_agents, n_actions).
        """
        episode_num = batch['o'].shape[0]
        q_evals = []
        q_targets = []

        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)

            # Forward pass through evaluation and target RNNs
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            # Reshape Q-values back to (episode_num, n_agents, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)

            q_evals.append(q_eval)
            q_targets.append(q_target)

        # Stack Q-values across transitions to get shape (episode_num, max_episode_len, n_agents, n_actions)
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def init_hidden(self, episode_num: int):
        """
        Initialize hidden states for evaluation and target RNNs.

        Args:
            episode_num (int): Number of episodes in the batch.
        """
        # Initialize hidden states on the correct device
        self.eval_hidden = torch.zeros(episode_num, self.n_agents, self.args.rnn_hidden_dim, device=self.device)
        self.target_hidden = torch.zeros(episode_num, self.n_agents, self.args.rnn_hidden_dim, device=self.device)

    def save_model(self, train_step: int):
        """
        Save the current model parameters.

        Args:
            train_step (int): Current training step, used to name the saved model files.
        """
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        qmix_path = os.path.join(self.model_dir, f'{num}_qmix_net_params.pkl')
        rnn_path = os.path.join(self.model_dir, f'{num}_rnn_net_params.pkl')

        torch.save(self.eval_qmix_net.state_dict(), qmix_path)
        torch.save(self.eval_rnn.state_dict(), rnn_path)
        print(f'Model saved at step {train_step}: {qmix_path}, {rnn_path}')
