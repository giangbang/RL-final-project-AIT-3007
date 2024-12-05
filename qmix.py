import numpy as np
import supersuit
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random
from os import path
import pickle
import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class ReplayBufferGRU:
    """ 
    Replay buffer for agent with GRU network additionally storing previous action, 
    initial input hidden state and output hidden state of GRU.
    And each sample contains the whole episode instead of a single step.
    'hidden_in' and 'hidden_out' are only the initial hidden state for each episode, for GRU initialization.

    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, hidden_in, hidden_out, state, action, last_action, reward, next_state):      
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (
            hidden_in, hidden_out, state, action, last_action, reward, next_state)
        self.position = int((self.position + 1) %
                            self.capacity)  # as a ring buffer

    def sample(self, batch_size):
        s_lst, a_lst, la_lst, r_lst, ns_lst, hi_lst, ho_lst = [], [], [], [], [], [], []
        batch = random.sample(self.buffer, batch_size)
        min_seq_len = float('inf')
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            min_seq_len = min(len(state), min_seq_len)
            hi_lst.append(h_in) # h_in: (1, batch_size=1, n_agents, hidden_size)
            ho_lst.append(h_out)
        hi_lst = torch.cat(hi_lst, dim=-3).detach()  # cat along the batch dim
        ho_lst = torch.cat(ho_lst, dim=-3).detach()

        # strip sequence length
        for sample in batch:
            h_in, h_out, state, action, last_action, reward, next_state = sample
            sample_len = len(state)
            start_idx = int((sample_len - min_seq_len)/2)
            end_idx = start_idx+min_seq_len
            s_lst.append(state[start_idx:end_idx])
            a_lst.append(action[start_idx:end_idx])
            la_lst.append(last_action[start_idx:end_idx])
            r_lst.append(reward[start_idx:end_idx])
            ns_lst.append(next_state[start_idx:end_idx])
        return hi_lst, ho_lst, s_lst, a_lst, la_lst, r_lst, ns_lst

    def __len__(
            self):  # cannot work in multiprocessing case, len(replay_buffer) is not available in proxy of manager!
        return len(self.buffer)

    def get_length(self):
        return len(self.buffer)

class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Conv2d(5, 5, 3),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        # x shape: [..., 13, 13, 5]
        # Permute để chuyển channels về đúng format của CNN
        orig_shape = x.shape[:-3]
        x = x.view(-1, 13, 13, 5)
        x = x.permute(0, 3, 1, 2)  # [..., 5, 13, 13]
        x = self.cnn(x)
        x = x.reshape(*orig_shape, -1)  # Flatten CNN output
        return x
        
class RNNAgent(nn.Module):
    '''
    @brief:
        evaluate Q value given a state and the action
    '''

    def __init__(self, num_inputs, action_shape, num_actions, hidden_size, feature_extractor):
        super(RNNAgent, self).__init__()

        self.num_inputs = num_inputs
        self.action_shape = action_shape
        self.num_actions = num_actions
        
        self.feature_extractor = feature_extractor

        self.linear1 = nn.Linear(num_inputs+action_shape*num_actions, hidden_size) #405+21 -> 64
        self.linear2 = nn.Linear(hidden_size, hidden_size)  #64 -> 64
        self.rnn = nn.GRU(hidden_size, hidden_size) #64 -> 64
        self.linear3 = nn.Linear(hidden_size, hidden_size) #64 -> 64
        self.linear4 = nn.Linear(hidden_size, action_shape*num_actions) #64 -> 21

    def forward(self, state, action, hidden_in):
        '''
        @params:
            state: [#batch, #sequence, #agent, #n_feature]
            action: [#batch, #sequence, #agent, action_shape]
        @return:
            qs: [#batch, #sequence, #agent, action_shape, num_actions]
        '''
        #  to [#sequence, #batch, #agent, #n_feature]
        bs, seq_len, n_agents, _, _, _= state.shape
        
        # Feature extraction from 2D observation
        state = self.feature_extractor(state)  # [batch, sequence, agents, cnn_features]

        state = state.permute(1, 0, 2, 3)
        action = action.permute(1, 0, 2, 3)
        action = F.one_hot(action, num_classes=self.num_actions).squeeze(-2)
        action = action.view(seq_len, bs, n_agents, -1) # [#batch, #sequence, #agent, action_shape*num_actions]

        # Concatenate with action
        x = torch.cat([state, action], -1)
        x = x.view(seq_len, bs*n_agents, -1) # change x to [#sequence, #batch*#agent, -1] to meet rnn's input requirement
        hidden_in = hidden_in.view(1, bs*n_agents, -1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x,  hidden = self.rnn(x, hidden_in)
        x = F.relu(self.linear3(x))
        x = self.linear4(x) # [#sequence, #batch, #agents, #action_shape*#actions]
        # [#sequence, #batch, #agent, #head * #action]
        x = x.view(seq_len, bs, n_agents, self.action_shape, self.num_actions)
        hidden = hidden.view(1, bs, n_agents, -1)
        # categorical over the discretized actions
        qs = F.softmax(x, dim=-1)
        qs = qs.permute(1, 0, 2, 3, 4)  # permute back [#batch, #sequence, #agents, #action_shape, #actions]

        return qs, hidden

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @brief:
            for each distributed agent, generate action for one step given input data
        @params:
            state: [n_agents, n_feature]
            last_action: [n_agents, action_shape]
        '''
        state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device) # add #sequence and #batch: [[#batch, #sequence, n_agents, n_feature]] 
        last_action = torch.LongTensor(
            last_action).unsqueeze(0).unsqueeze(0).to(device)  # add #sequence and #batch: [#batch, #sequence, n_agents, action_shape]
        hidden_in = hidden_in.unsqueeze(1) # add #batch: [#batch, n_agents, hidden_dim]
        agent_outs, hidden_out = self.forward(state, last_action, hidden_in)  # agents_out: [#batch, #sequence, n_agents, action_shape, action_dim]; hidden_out same as hidden_in
        dist = Categorical(agent_outs)

        if deterministic:
            action = np.argmax(agent_outs.detach().cpu().numpy(), axis=-1)
        else:
            action = dist.sample().squeeze(0).squeeze(0).detach().cpu().numpy()  # squeeze the added #batch and #sequence dimension
        
        # Clear unnecessary tensors
        del state, last_action, hidden_in, agent_outs, dist
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return action, hidden_out  # [n_agents, action_shape]

class QMix(nn.Module):
    def __init__(self, state_dim, n_agents, action_shape, embed_dim=64, hypernet_embed=128, abs=True, feature_extractor=None):
        """
        Critic network class for Qmix. Outputs centralized value function predictions given independent q value.
        :param args: (argparse) arguments containing relevant model information.
        """
        super(QMix, self).__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim*n_agents*action_shape # #features*n_agents
        self.action_shape = action_shape

        self.embed_dim = embed_dim
        self.hypernet_embed = hypernet_embed
        self.abs = abs
        
        self.feature_extractor = feature_extractor

        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(self.hypernet_embed, self.action_shape * self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, self.hypernet_embed),
                                            nn.ReLU(inplace=True),
                                           nn.Linear(self.hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(
            self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        """
        Compute actions from the given inputs.
        @params:
            agent_qs: [#batch, #sequence, #agent, #action_shape]
            states: [#batch, #sequence, #agent, #features*action_shape]
        :param agent_qs: q value inputs into network [batch_size, #agent, action_shape]
        :param states: state observation.
        :return q_tot: (torch.Tensor) return q-total .
        """
        bs = agent_qs.size(0)
        
        # Feature extraction from 2D observation
        states = self.feature_extractor(states)
    
        states = states.reshape(-1, self.state_dim)  # [#batch*#sequence, action_shape*#features*#agent]
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents*self.action_shape)  # [#batch*#sequence, 1, #agent*#action_shape]
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)  # [#batch*#sequence, action_shape*self.embed_dim*#agent]
        b1 = self.hyper_b_1(states)  # [#batch*#sequence, self.embed_dim]
        w1 = w1.view(-1, self.n_agents*self.action_shape, self.embed_dim)  # [#batch*#sequence, #agent*action_shape, self.embed_dim]
        b1 = b1.view(-1, 1, self.embed_dim)   # [#batch*#sequence, 1, self.embed_dim]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [#batch*#sequence, 1, self.embed_dim]

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)  # [#batch*#sequence, self.embed_dim]
        w_final = w_final.view(-1, self.embed_dim, 1)  # [#batch*#sequence, self.embed_dim, 1]
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)  # [#batch*#sequence, 1, 1]
        # Compute final output
        y = torch.bmm(hidden, w_final) + v  
        # Reshape and return
        q_tot = y.view(bs, -1, 1) # [#batch, #sequence, 1]
        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = torch.abs(self.hyper_w_1(states))
        w_final = torch.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim*self.action_shape)
        w_final = w_final.view(-1, self.embed_dim*self.action_shape, 1)
        k = torch.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / torch.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim*self.action_shape, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim*self.action_shape)
        v = self.V(states).view(-1, 1, 1)
        b = torch.bmm(b1, w_final) + v
        return b


class QMix_Trainer():
    def __init__(self, replay_buffer, n_agents, state_dim, action_shape, action_dim, hidden_dim, hypernet_dim, target_update_interval, lr=0.001, logger=None):
        self.replay_buffer = replay_buffer

        self.action_dim = action_dim
        self.action_shape = action_shape
        self.n_agents = n_agents
        self.target_update_interval = target_update_interval
        self.feature_extractor = CNNFeatureExtractor()
        self.agent = RNNAgent(state_dim, action_shape,
                              action_dim, hidden_dim, feature_extractor=self.feature_extractor).to(device)
        self.target_agent = RNNAgent(
            state_dim, action_shape, action_dim, hidden_dim, feature_extractor=self.feature_extractor).to(device)
        
        self.mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim, feature_extractor=self.feature_extractor).to(device)
        self.target_mixer = QMix(state_dim, n_agents, action_shape,
                          hidden_dim, hypernet_dim, feature_extractor=self.feature_extractor).to(device)
        
        self._update_targets()
        self.update_cnt = 0
        
        self.criterion = nn.MSELoss()

        # self.optimizer = optim.Adam(
        #     list(self.agent.parameters())+list(self.mixer.parameters()), lr=lr)
        all_params = set(self.agent.parameters()) | set(self.mixer.parameters())
        self.optimizer = optim.Adam(all_params, lr=lr)

    def sample_action(self):
        probs = torch.FloatTensor(
            np.ones(self.action_dim)/self.action_dim).to(device)
        dist = Categorical(probs)
        action = dist.sample((self.n_agents, self.action_shape))

        return action.type(torch.FloatTensor).numpy()

    def get_action(self, state, last_action, hidden_in, deterministic=False):
        '''
        @return:
            action: w/ shape [#active_as]
        '''

        action, hidden_out = self.agent.get_action(state, last_action, hidden_in, deterministic=deterministic)

        return action, hidden_out

    def push_replay_buffer(self, ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                           episode_reward, episode_next_state):
        '''
        @brief: push arguments into replay buffer
        '''
        self.replay_buffer.push(ini_hidden_in, ini_hidden_out, episode_state, episode_action, episode_last_action,
                                episode_reward, episode_next_state)

    def update(self, batch_size):
        # 1. Lấy batch từ replay buffer
        hidden_in, hidden_out, state, action, last_action, reward, next_state = self.replay_buffer.sample(
            batch_size)
        
        # Converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        state = np.array(state)
        next_state = np.array(next_state) 
        action = np.array(action)
        last_action = np.array(last_action)
        reward = np.array(reward)
        
        state = torch.FloatTensor(state).to(device) # [#batch, sequence, #agents, #features*action_shape]
        next_state = torch.FloatTensor(next_state).to(device)
        action = torch.LongTensor(action).to(device) # [#batch, sequence, #agents, #action_shape]
        last_action = torch.LongTensor(last_action).to(device)
        reward = torch.FloatTensor(reward).unsqueeze(-1).to(device) # reward is scalar, add 1 dim to be [reward] at the same dim

        # 2. Tính current Q values
        agent_outs, _ = self.agent(state, last_action, hidden_in) # [#batch, #sequence, #agent, action_shape, num_actions]
        chosen_action_qvals = torch.gather(  # [#batch, #sequence, #agent, action_shape]
            agent_outs, dim=-1, index=action.unsqueeze(-1)).squeeze(-1)
        qtot = self.mixer(chosen_action_qvals, state) # [#batch, #sequence, 1]

        # 3. Tính target Q values
        target_agent_outs, _ = self.target_agent(next_state, action, hidden_out)
        target_max_qvals = target_agent_outs.max(dim=-1, keepdim=True)[0] # [#batch, #sequence, #agents, action_shape]
        target_qtot = self.target_mixer(target_max_qvals, next_state)

        # 4. Tính reward và targets
        reward = self._calc_reward(reward)
        targets = self._build_td_lambda_targets(reward, target_qtot)

        # 5. Tính loss và update
        loss = self.criterion(qtot, targets.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_cnt += 1
        if self.update_cnt % self.target_update_interval == 0:
            self._update_targets()

        return loss.item()

    def _build_td_lambda_targets(self, rewards, target_qs, gamma=0.99, td_lambda=0.6):
        '''
        @params:
            rewards: [#batch, #sequence, 1]
            target_qs: [#batch, #sequence, 1]
        '''
        ret = target_qs.new_zeros(*target_qs.shape)
        ret[:, -1] = target_qs[:, -1]
        # backwards recursive update of the "forward view"
        for t in range(ret.shape[1] - 2, -1, -1):
            ret[:, t] = td_lambda * gamma * ret[:, t+1] + (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t+1])
        return ret

    def _update_targets(self):
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_agent.parameters(), self.agent.parameters()):
            target_param.data.copy_(param.data)

    def _calc_reward(self, rewards):
        # Tạo mask cho các agent còn sống (reward != 0)
        alive_mask = (rewards != 0).float()  # [batch, sequence, agents, 1]
        
        # Tính tổng số agent còn sống tại mỗi timestep
        num_alive = alive_mask.sum(dim=2, keepdim=True)  # [batch, sequence, 1, 1]
        num_alive = torch.clamp(num_alive, min=1.0)  # Tránh chia cho 0
        
        # Tính mean reward chỉ cho các agent còn sống
        rewards = (rewards * alive_mask).sum(dim=2, keepdim=True) / num_alive  # [batch, sequence, 1, 1]
        rewards = rewards.squeeze(-1)  # [batch, sequence, 1]
        return rewards

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        torch.save(self.agent.state_dict(), path+'_agent')
        torch.save(self.mixer.state_dict(), path+'_mixer')

    def load_model(self, path):
        self.agent.load_state_dict(torch.load(path+'_agent'))
        self.mixer.load_state_dict(torch.load(path+'_mixer'))

        self.agent.eval()
        self.mixer.eval()