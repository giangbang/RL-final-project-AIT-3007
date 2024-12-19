import numpy as np
import torch
import time
import torch.optim as optim
import torch.nn.functional as F

from torch.amp import GradScaler, autocast

from buffer import ReplayBuffer
from team import TeamManager
from utils import reseed, save_model, device, seed
from eval import evaluate_model

import sys
import os
# Thêm thư mục gốc của project vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

def train(q, q_target, memory, optimizer, gamma, batch_size, update_iter=10, chunk_size=10, grad_clip_norm=5):
    q.train()
    q_target.eval()
    chunk_size = chunk_size if q.recurrent else 1
    losses = []

    scaler = GradScaler()
    
    for i in range(update_iter):
        # Get data from buffer
        states, actions, rewards, next_states, dones = memory.sample_chunk(batch_size, chunk_size)

        hidden = q.init_hidden(batch_size).to(device)
        target_hidden = q_target.init_hidden(batch_size).to(device)
        
        loss = 0
        for step_i in range(chunk_size):
            with autocast():
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

def run_episode(env, q, opponent_q, memory=None, random_rate=0, epsilon=0.1):
    """Run an episode in self-play mode
    :return: total score of the episode
    """
    observations, infos = env.reset()
    team_manager = TeamManager(env.agents)
    teams = team_manager.get_teams()
    my_team = team_manager.get_my_team()
    opponent_team = team_manager.get_other_team()
    
    hidden = q.init_hidden()
    opponent_hidden = q.init_hidden()
    score = 0.0

    while not team_manager.has_terminated_teams():
        # Fill rows with zeros for terminated agents
        for agent in team_manager.agents:
            if agent not in observations or observations[agent] is None:
                observations[agent] = np.zeros(q.n_obs, dtype=np.float32)
                team_manager.terminate_agent(agent)

        # Get observations for the current team and opponent
        my_team_observations = team_manager.get_info_of_team(my_team, observations)
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
        opponent_obs_tensor = torch.tensor(np.array(list(opponent_observations.values()))).unsqueeze(0)
        opponent_actions, opponent_hidden = opponent_q.sample_action(opponent_obs_tensor, opponent_hidden, epsilon)
        opponent_team_actions = {
            agent: action
            for agent, action in zip(
                opponent_observations.keys(), opponent_actions.squeeze(0).cpu().data.numpy().tolist()
            )
        }

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
        model_team1,
        model_team2,
        target_model_team1,
        target_model_team2,
        save_name_team1,
        save_name_team2,
        team_manager,
        hp,
        train_fn,
        run_episode_fn,
        num_test_runs=1,
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
    memory_team1 = ReplayBuffer(hp.buffer_limit)
    memory_team2 = ReplayBuffer(hp.buffer_limit)

    test_env.reset(seed=seed)
    env.reset(seed=seed)

    target_model_team1.load_state_dict(model_team1.state_dict())
    target_model_team2.load_state_dict(model_team2.state_dict())

    # Setup env

    train_scores_team1, train_scores_team2 = [], []
    test_scores_team1, test_scores_team2 = [], []
    losses_team1, losses_team2 = [], []

    optimizer_team1 = optim.Adam(model_team1.parameters(), lr=hp.lr)
    optimizer_team2 = optim.Adam(model_team2.parameters(), lr=hp.lr)

    # Train and test
    start_train = time.time()
    for episode_i in range(hp.max_episodes):
        start = time.time()
        print(f'Episodes {episode_i + 1} / {hp.max_episodes}')
        # Collect data
        epsilon = max(hp.min_epsilon,
                      hp.max_epsilon - (hp.max_epsilon - hp.min_epsilon) * (episode_i / (hp.episode_min_epsilon)))
        
        model_team1.eval()
        model_team2.eval()
        
        train_score_team1 = run_episode_fn(env, model_team1, model_team2, memory_team1, epsilon=epsilon)
        train_score_team2 = run_episode_fn(env, model_team2, model_team1, memory_team2, epsilon=epsilon)

        train_scores_team1.append(train_score_team1)
        train_scores_team2.append(train_score_team2)

        if train_score_team1 > 200 or train_score_team2 > 200:
            hp.min_epsilon = 0.05

        # Train models
        if memory_team1.size() > hp.warm_up_steps:
            print("Training Team 1:")
            model_team1.train()
            episode_losses_team1 = train_fn(
                model_team1, target_model_team1, memory_team1, optimizer_team1,
                hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
            )
            losses_team1.append(episode_losses_team1)

        if memory_team2.size() > hp.warm_up_steps:
            print("Training Team 2:")
            model_team2.train()
            episode_losses_team2 = train_fn(
                model_team2, target_model_team2, memory_team2, optimizer_team2,
                hp.gamma, hp.batch_size, hp.update_iter, hp.chunk_size
            )
            losses_team2.append(episode_losses_team2)

        if episode_i % hp.update_target_interval == 0 and episode_i > 0:
            target_model_team1.load_state_dict(model_team1.state_dict())
            target_model_team2.load_state_dict(model_team2.state_dict())

        # Test phase
        if episode_i >= hp.max_episodes - 20:
            print("Test phase for both teams:")
            model_team1.eval()
            model_team2.eval()

            avg_test_score_team1 = 0
            avg_test_score_team2 = 0

            for _ in range(num_test_runs):
                avg_test_score_team1 += evaluate_model(test_env, hp.test_episodes, model_team1, model_team2, run_episode_fn)
                avg_test_score_team2 += evaluate_model(test_env, hp.test_episodes, model_team2, model_team1,  run_episode_fn)

            avg_test_score_team1 /= num_test_runs
            avg_test_score_team2 /= num_test_runs

            test_scores_team1.append(avg_test_score_team1)
            test_scores_team2.append(avg_test_score_team2)

            save_model(model_team1, f'vdn-{save_name_team1}-{episode_i}')
            save_model(model_team1, f'vdn-{save_name_team2}-{episode_i}')

            print(f"Team 1 Avg Test Score: {avg_test_score_team1:.2f}")
            print(f"Team 2 Avg Test Score: {avg_test_score_team2:.2f}")
            print('#' * 90)

        print(f'Time: {time.time() - start}')
        print(f'Total Time: {time.time() - start_train}')
        print('-' * 90)

    env.close()
    test_env.close()

    return train_scores_team1, train_scores_team2, test_scores_team1, test_scores_team2, losses_team1, losses_team2