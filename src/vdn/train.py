import numpy as np
import torch
import time
import torch.optim as optim
import torch.nn.functional as F
import random

from torch.amp import GradScaler, autocast

from model import ReplayBuffer
from utils import TeamManager

# Init
seed = 25
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_model(model, name):
    torch.save(model.state_dict(), f'{name}.pth')

def save_data(data, name='data'):
    np.save(f'{name}.npy', data)

def reseed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import sys
import os
# Thêm thư mục gốc của project vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.rule_based.model import RuleBasedAgent
rule_base_agent = RuleBasedAgent(my_team='red')

def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
    q.train()
    q_target.eval()
    chunk_size = chunk_size if q.recurrent else 1
    losses = []

    scaler = GradScaler('cuda')
    
    for i in range(update_iter):
        # Get data from buffer
        states, actions, rewards, next_states, dones = memory.sample_chunk(batch_size, chunk_size)

        hidden = q.init_hidden(batch_size).to(device)
        target_hidden = q_target.init_hidden(batch_size).to(device)
        
        loss = 0
        for step_i in range(chunk_size):
            with autocast('cuda'):
                q_out, hidden = q(states[:, step_i].to(device), hidden)  # [batch_size, num_agents, n_actions]
                q_out = q_out.to(device)
                hidden = hidden.to(device)
                q_a = q_out.gather(2, actions[:, step_i, :].unsqueeze(-1).long().to(device)).squeeze(-1)  # [batch_size, num_agents]: q values of actions taken
                sum_q = (q_a * (1 - dones[:, step_i].to(device))).sum(dim=1, keepdims=True)  # [batch_size, 1]
    
                with torch.no_grad():
                    max_q_prime, target_hidden = q_target(next_states[:, step_i].to(device), target_hidden.detach())
                    target_hidden = target_hidden.to(device)
                    max_q_prime = max_q_prime.max(dim=2)[0].squeeze(-1)  # [batch_size, num_agents]
                    target_q = rewards[:, step_i, :].to(device).sum(dim=1, keepdims=True)  # [batch_size, 1]
                    target_q += gamma * ((1 - dones[:, step_i].to(device)) * max_q_prime.to(device)).sum(dim=1, keepdims=True)
            
                loss += F.smooth_l1_loss(sum_q, target_q.detach())
            
                # Create a mask for each agent separately
                done_mask = dones[:, step_i].to(device).bool()  # Shape: (batch_size, num_agents)
                
                # Lấy chỉ số batch và agent nơi done_mask == 1
                batch_indices, agent_indices = torch.where(done_mask)
                
                # Số lượng agents đã kết thúc
                num_terminated = len(batch_indices)
                
                if num_terminated > 0:  # Chỉ xử lý nếu có agent nào bị kết thúc
                    # Khởi tạo hidden states mới cho tất cả các agents bị kết thúc
                    new_hidden = q.init_hidden(batch_size=num_terminated).to(device)  # Shape: (num_terminated, num_agents, hx_size)
                    new_target_hidden = q_target.init_hidden(batch_size=num_terminated).to(device)  # Same shape
                
                    # Lấy hidden states tương ứng với từng agent
                    new_hidden_agents = new_hidden[range(num_terminated), agent_indices, :]  # Shape: (num_terminated, hx_size)
                    new_target_hidden_agents = new_target_hidden[range(num_terminated), agent_indices, :]  # Same shape
                
                    # Gán các hidden states mới vào các vị trí tương ứng trong tensor `hidden` và `target_hidden`
                    hidden[batch_indices, agent_indices, :] = new_hidden_agents
                    target_hidden[batch_indices, agent_indices, :] = new_target_hidden_agents

        losses.append(loss.item())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), grad_clip_norm, norm_type=2)
        scaler.step(optimizer)
        scaler.update()

    print('Loss: ' + " ".join([str(round(loss, 2)) for loss in losses]))
    return losses

def run_episode(env, q, memory=None, random_rate=0, epsilon=0.1):
    """Run an episode in self-play mode
    :return: total score of the episode
    """
    observations, infos = env.reset()
    team_manager = TeamManager(env.agents)
    teams = team_manager.get_teams()
    my_team = team_manager.get_my_team()
    hidden = q.init_hidden()
    opponent_hidden = q.init_hidden()  # Hidden state for the opponent
    score = 0.0

    while not team_manager.has_terminated_teams():
        # Fill rows with zeros for terminated agents
        for agent in team_manager.agents:
            if agent not in observations or observations[agent] is None:
                observations[agent] = np.zeros(q.n_obs, dtype=np.float32)
                team_manager.terminate_agent(agent)

        # Get observations for the current team and opponent
        my_team_observations = team_manager.get_info_of_team(my_team, observations)
        opponent_team = team_manager.get_other_team()
        opponent_observations = team_manager.get_info_of_team(opponent_team, observations)

        # Get actions for my team
        obs_tensor = torch.tensor(np.array(list(my_team_observations.values()))).unsqueeze(0)
        actions, hidden = q.sample_action(obs_tensor, hidden, epsilon)
        my_team_actions = {
            agent: action
            for agent, action in zip(
                my_team_observations.keys(), actions.squeeze(0).cpu().data.numpy().tolist()
            )
        }

        # Get actions for the opponent team (self-play logic)
        opponent_obs_batch = np.array(list(opponent_observations.values()))  # Batch observations
        opponent_obs_tensor = torch.tensor(opponent_obs_batch, dtype=torch.float32)  # Convert to tensor
        opponent_actions = rule_base_agent.get_action(opponent_obs_tensor)
        opponent_team_actions = dict(zip(opponent_observations.keys(), opponent_actions.tolist()))

        # Combine actions
        agent_actions = {**my_team_actions, **opponent_team_actions}

        # Terminated agents use None action
        for agent in team_manager.terminated_agents:
            agent_actions[agent] = None

        # Step the environment
        observations, agent_rewards, agent_terminations, agent_truncations, agent_infos = env.step(agent_actions)
        score += sum(team_manager.get_info_of_team(my_team, agent_rewards, 0).values())

        if memory is not None:
            # Fill rows with zeros for terminated agents
            next_observations = [
                observations[agent]
                if agent in observations and observations[agent] is not None
                else np.zeros(q.n_obs, dtype=np.float32)
                for agent in team_manager.get_my_agents()
            ]
            my_team_actions = [
                agent_actions[agent]
                if agent in agent_actions and agent_actions[agent] is not None
                else 0
                for agent in team_manager.get_my_agents()
            ]

            memory.put((
                list(my_team_observations.values()),
                my_team_actions,
                list(team_manager.get_info_of_team(my_team, agent_rewards, 0).values()),
                next_observations,
                list(team_manager.get_info_of_team(
                    my_team,
                    TeamManager.merge_terminates_truncates(agent_terminations, agent_truncations)).values())
            ))

        # Check for termination
        for agent, done in agent_terminations.items():
            if done:
                team_manager.terminate_agent(agent)
        for agent, done in agent_truncations.items():
            if done:
                team_manager.terminate_agent(agent)

        # Break if the other team has less than 3 agents
        if len(team_manager.get_other_team_remains()) <= 3:
            break

    print('Score:', score)
    return score

def run_model_train_test(
        env,
        test_env,
        model,
        target_model,
        save_name,
        team_manager,
        hp,
        train_fn,
        run_episode_fn,
        num_test_runs=1,
        mix_net = None,
        mix_net_target = None,
):
    """
    Run training and testing loop of a model

    :param env: Training environment
    :param test_env: Testing environment
    :param model: training model
    :param target_model: target model
    :param save_name: name to save the model
    :param team_manager: TeamManager
    :param hp: Hyperparameters
    :param train_fn: training function
    :param run_episode_fn: function to run an episode
    :return: train_scores, test_scores
    """
    reseed(seed)
    # create env.
    memory = ReplayBuffer(hp.buffer_limit)

    # Setup env
    test_env.reset(seed=seed)
    env.reset(seed=seed)
    my_team = team_manager.get_my_team()
    print("My team: ", my_team)

    # Load target model
    target_model.load_state_dict(model.state_dict())
    if mix_net is not None:
        mix_net_target.load_state_dict(mix_net.state_dict())

    train_score = 0
    test_score = 0
    train_scores = []
    test_scores = []
    losses = []

    # Train and test
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    if mix_net is not None:
        optimizer = optim.Adam([{'params': model.parameters()}, {'params': mix_net.parameters()}], lr=hp.lr)
    start_train = time.time()
    for episode_i in range(hp.max_episodes):
        start = time.time()
        print(f'Episodes {episode_i + 1} / {hp.max_episodes}')
        # Collect data
        epsilon = max(hp.min_epsilon,
                      hp.max_epsilon - (hp.max_epsilon - hp.min_epsilon) * (episode_i / (hp.episode_min_epsilon)))
        model.eval()
        train_score = run_episode_fn(env, model, memory, epsilon=epsilon)
        train_scores.append(train_score)

        if train_score > 200:
            hp.min_epsilon = 0.05

        # Train
        if memory.size() > hp.warm_up_steps:
            print("Training phase:")
            model.train()

            if mix_net is not None:
                mix_net.train()
                episode_losses = train_fn(
                    model, target_model, mix_net, mix_net_target, memory, optimizer,
                    hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
                )
            else:
                episode_losses = train_fn(
                    model, target_model, memory, optimizer,
                    hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
                )
            losses.append(episode_losses)

        if episode_i % hp.update_target_interval == 0 and episode_i > 0:
            target_model.load_state_dict(model.state_dict())
            if mix_net is not None:
                mix_net_target.load_state_dict(mix_net.state_dict())

        # Test
        if episode_i >= hp.max_episodes - 40:
            print('Test phase:')
            model.eval()
            avg_test_score = 0
            for _ in range(num_test_runs):
                test_score = evaluate_model(test_env, hp.test_episodes, model, run_episode_fn)
                avg_test_score += test_score
            avg_test_score /= num_test_runs

            test_scores.append(avg_test_score)

            save_model(model, f'{save_name}-{episode_i}')

            print(f"Avg Test score: {avg_test_score:.2f} | Episode: {episode_i}")
            print('#' * 90)

        print(f'Time: {time.time() - start}')
        print(f'Total Time: {time.time() - start_train}')
        print('-'*90)

    env.close()
    test_env.close()

    return train_scores, test_scores, losses