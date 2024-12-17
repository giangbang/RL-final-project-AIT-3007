import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from qmix import RNNAgent
from rewards import _calc_reward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN_Trainer():
    def __init__(self, replay_buffer, n_agents, obs_dim, action_shape, action_dim, hidden_dim, 
                 target_update_interval, lr=5e-4, epsilon_start=1.0, epsilon_end=0.05, 
                 epsilon_decay=0.995, lambda_reward=1):
        self.replay_buffer = replay_buffer
        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.target_update_interval = target_update_interval
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.lambda_reward = lambda_reward
        
        # Khởi tạo agent chính và target agent
        self.agent = RNNAgent(obs_dim, action_shape, action_dim, hidden_dim, self.epsilon).to(device)
        self.target_agent = RNNAgent(obs_dim, action_shape, action_dim, hidden_dim, epsilon=0.0).to(device)

        self._update_targets()
        self.update_cnt = 0
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.AdamW(self.agent.parameters(), lr=lr, weight_decay=0.001)

    def get_action(self, state, hidden_in):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        action, hidden_out = self.agent.get_action(state, hidden_in)

        return action, hidden_out

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_observation, episode_state, episode_next_state, episode_action,
                           episode_reward, episode_next_observation):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_observation, episode_state, episode_next_state, episode_action,
                                episode_reward, episode_next_observation)
    
    def update(self, batch_size):
        current_loss = 100
        total_epoch = 0
        num_epoch = 100

        hidden_in, hidden_out, observation, state, next_state, action, reward, next_observation = self.replay_buffer.sample(batch_size)

        # Chuyển đổi dữ liệu
        observation = torch.FloatTensor(np.array(observation)).to(device)
        next_observation = torch.FloatTensor(np.array(next_observation)).to(device)
        action = torch.LongTensor(np.array(action)).to(device)
        reward = torch.FloatTensor(np.array(reward)).unsqueeze(-1).to(device)

        while(current_loss > 0.1 and total_epoch < 10):
            for epoch in range(1, num_epoch + 1):
                # Tính current Q values
                agent_outs, _ = self.agent(observation, hidden_in)
                chosen_action_qvals = torch.gather(
                    agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)

                # Tính target Q values
                target_agent_outs, _ = self.target_agent(next_observation, hidden_out)
                target_max_qvals = target_agent_outs.max(dim=-1)[0]

                # Tính reward và targets
                targets = self._build_td0_targets(reward, target_max_qvals)

                # Tính loss và update
                loss = self.criterion(chosen_action_qvals, targets.detach())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                current_loss = loss.item()
            
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}/{total_epoch+1}, Loss: {current_loss}')

            self.update_cnt += 1
            if self.update_cnt % self.target_update_interval == 0:
                self._update_targets()
            total_epoch += 1

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.agent.epsilon = self.epsilon

        return current_loss, None, None, None
    
    def _build_td0_targets(self, rewards, target_qs, gamma=0.99):
        """
        Tính toán target Q-values theo công thức Q-learning: Q(s,a) = r + γ max_a' Q(s',a')
        
        @params:
            rewards: [#batch, #sequence, 1] - Phần thưởng tức thời
            target_qs: [#batch, #sequence, 1] - Q values từ target network
        @return:
            ret: [#batch, #sequence, 1] - Target Q values
        """
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = rewards[:, -1]
        # Q(s,a) = r + γ max_a' Q(s',a')
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = rewards[:, t] + gamma * target_qs[:, t+1]
        return ret

    def _update_targets(self):
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.agent.state_dict(), path+'_agent')

    def load_model(self, path, map_location):
        self.agent.load_state_dict(torch.load(path+'_agent', map_location=map_location, weights_only=True))
        self.agent.eval()
