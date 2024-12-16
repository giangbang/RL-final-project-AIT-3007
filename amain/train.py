from utils.qmix import QMIX
from utils.rb import ReplayBuffer, PrioritizedReplayBuffer
# from utils.normalization import Normalizer
from magent2.environments import battle_v4
import supersuit as ss
import torch
import numpy as np
import os
import wandb
from utils.process import process_batch 

def normalize_data( data):
    # data shape : (H, W, C)
    mean = np.mean(data, axis=(0, 1))
    std = np.std(data, axis=(0, 1))
    return (data - mean) / (std + 1e-8)


def train(config):
    # Update variables with config values
    num_episodes = config.num_episodes
    batch_size = config.batch_size
    epsilon = config.epsilon_start
    epsilon_decay = config.epsilon_decay
    epsilon_min = config.epsilon_min
    learning_rate = config.learning_rate
    gamma = config.gamma
    update_step = config.update_step
    sub_bs = config.sub_bs
    num_envs = 2 
    # env = battle_v4.parallel_env(map_size=30, minimap_mode=False, max_cycles=300, seed=10)
    env = battle_v4.parallel_env(map_size=45, minimap_mode=False, step_reward=-0.005,
            dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=10, 
            extra_features=False)
    env = ss.black_death_v3(env)

    env.reset()
    blue_agents = [agent for agent in env.possible_agents if agent.startswith("blue_")]
    num_blue_agents = len(blue_agents)

    red_agents = [agent for agent in env.possible_agents if agent.startswith("red_")]
    num_red_agents = len(red_agents)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize QMIX agents
    qmix_blue = QMIX(
        num_agents=num_blue_agents,
        state_shape=(45, 45, 5),
        agent_ids=blue_agents,
        device=device,
        num_actions=21,
        lr=learning_rate,
        gamma=gamma,
        sub_batch_size=sub_bs
    )
    qmix_red = QMIX(
        num_agents=num_red_agents,
        state_shape=(45, 45, 5),
        agent_ids=red_agents,
        device=device,
        num_actions=21,
        lr=learning_rate,
        gamma=gamma,
        sub_batch_size=sub_bs
    )

    # Initialize replay buffer
    field_names = ["obs", "actions", "rewards", "next_obs", "dones", "state", "next_state"]
    rb = PrioritizedReplayBuffer(memory_size=60, field_names=field_names)

    blue_agents = [agent for agent in env.agents if agent.startswith("blue")]
    red_agents = [agent for agent in env.agents if agent.startswith("red")]
    obs_shape = (13,13,5)
    state_shape = (45,45,5)
    count = 0
    # normalizer_obs_b = Normalizer(shape=obs_shape)
    # normalizer_obs_r = Normalizer(shape=obs_shape)
    # normalizer_state = Normalizer(shape=state_shape)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        # We will collect transitions after all blue agents have acted once (a "joint step")
        # the purpose of joint step is to collect joint transitions for QMIX training
        global_state = env.state() # global state for mixing
        global_state = normalize_data(global_state)
        
        episode_reward_blue = 0
        episode_reward_red = 0

        terminated = False
        episode = []

        while not terminated:
            
            obs_array_blue = np.stack([obs[a] for a in blue_agents], axis=0)
            obs_array_blue = normalize_data(obs_array_blue)
            actions_blue = qmix_blue.select_actions(torch.from_numpy(obs_array_blue).to(device), epsilon=epsilon) # return shape: (N_agents,)


            obs_array_red = np.stack([obs[a] for a in red_agents], axis=0)
            obs_array_red = normalize_data(obs_array_red)
            actions_red= qmix_red.select_actions(torch.from_numpy(obs_array_red).to(device), epsilon=epsilon) # return shape: (N_agents,)

            actions = {}
            for i, agent in enumerate(blue_agents):
                actions[agent] = actions_blue[agent]
            for i, agent in enumerate(red_agents):
                actions[agent] = actions_red[agent]

            # Step environment
            next_obs, rewards, terminations, truncations, info = env.step(actions)
            dones = {}
            for agent in env.agents:
                dones[agent] = terminations[agent] or truncations[agent]

            next_global_state = env.state()
            next_global_state = normalize_data(next_global_state)

            done_all = np.all(list(dones.values()))

            reward_blue = sum([rewards[a] for a in blue_agents])
            reward_red = sum([rewards[a] for a in red_agents])

            episode_reward_blue += reward_blue
            episode_reward_red += reward_red

            a_blue = np.array([actions_blue[a] for a in blue_agents])
            a_red = np.array([actions_red[a] for a in red_agents])

            next_obs_array_blue = np.stack([next_obs[a] for a in blue_agents], axis=0)
            next_obs_array_blue = normalize_data(next_obs_array_blue)

            next_obs_array_red = np.stack([next_obs[a] for a in red_agents], axis=0)
            next_obs_array_red = normalize_data(next_obs_array_red)

            obs_save = {"blue": obs_array_blue, "red": obs_array_red} # obs_array_blue: (81, 13, 13, 5)
            reward_save = {"blue": reward_blue, "red": reward_red} # reward_blue: (1,)
            next_obs_save = {"blue": next_obs_array_red, "red": next_obs_array_red} # (81, 13, 13, 5)

            actions = {"blue": a_blue, "red": a_red} # actions_blue: (81,)
            dones = {"blue": done_all, "red": done_all} # done_all: bool
            global_state = np.array(global_state)


            transition = {
                "obs": obs_save,
                "actions": actions,
                "rewards": reward_save,
                "next_obs": next_obs_save,
                "dones": dones,
                "state": global_state,
                "next_state": next_global_state
            }
            episode.append(transition)

            obs = next_obs
            global_state = next_global_state
            terminated = done_all

            # After episode, decay epsilon
            epsilon = max(epsilon * epsilon_decay, epsilon_min)

            # Training step
            if len(rb) >= batch_size:
                batch, ids = rb.sample(batch_size)

                count += batch_size
                # batch['obs']: (B, transitions, N, 13, 13, 5)
                # batch['actions']: (B, transitions,N)
                # batch['rewards']: (B, transitions, )
                # batch['next_obs']: (B, transitions, 13, 13, 5)
                # batch['state']: (B, transitions, 45, 45, 5)
                # batch['next_state']: (B, transitions, 45, 45, 5)
                # batch['dones']: (B, transitions, )
                batch_blue, batch_red = process_batch(batch)
                loss_blue, priorities_b = qmix_blue.update(batch_blue, ep)
                loss_red, priorities_r = qmix_red.update(batch_red,ep)

                wandb.log({
                        "loss_blue": loss_blue,
                        "loss_red": loss_red,
                        "epsilon": epsilon,
                        "loss" : (loss_blue + loss_red)/2
                    })
                qmix_blue.update_target_soft(config.tau)
                qmix_red.update_target_soft(config.tau)
                priorities = [(a+b)/2 for a,b in zip(priorities_b, priorities_r)]

                rb.update_priorities(ids, priorities)
                # print(count)

        rb.save_episode(episode)
        wandb.log({
        "episode_reward_blue": episode_reward_blue,
        "episode_reward_red": episode_reward_red,
        "episode": ep
        })
        if ((ep+1) % update_step) == 0:
                save_path_blue = os.path.join("./model", f"qmix_blue_ep{ep}.pth")
                save_path_red = os.path.join("./model", f"qmix_red_ep{ep}.pth")
                torch.save(qmix_blue.agent_q_network.state_dict(), save_path_blue)
                torch.save(qmix_red.agent_q_network.state_dict(), save_path_red)

    wandb.finish()