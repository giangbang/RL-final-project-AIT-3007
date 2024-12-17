from .utils.qmix import QMIX
from rb import ReplayBuffer, PrioritizedReplayBuffer
import gc
# from utils.normalization import Normalizer
from magent2.environments import battle_v4
import supersuit as ss
import torch
import numpy as np
import os
import wandb
from .utils.process import process_batch, Normalizer 
import random
import numpy as np
import torch
import time

def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# def normalize_data( data):
#     # data shape : (H, W, C)
#     mean = np.mean(data, axis=(0, 1))
#     std = np.std(data, axis=(0, 1))
#     # I want to take care situation where std is 0
#     return (data - mean) / (std + 1e-8)
def add_parameter_noise(model, stddev=0.01):
    """Adds parameter-space noise to a model."""
    with torch.no_grad():
        for param in model.parameters():
            noise = torch.randn_like(param) * stddev
            param.add_(noise)

def train(config):
    # Update variables with config values
    # Set save time (e.g., 2 hours)
    save_time_seconds = 2 * 3600  # 2 hours in seconds
    start_time = time.time()
    set_seed(config.seed)
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
    env = battle_v4.parallel_env(map_size=45, minimap_mode=False, step_reward=-0.008,
            dead_penalty=-0.2, attack_penalty=-0.1, attack_opponent_reward=0.8, max_cycles=360, 
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
    
    rb = PrioritizedReplayBuffer(memory_size=20, field_names=field_names)

    blue_agents = [agent for agent in env.agents if agent.startswith("blue")]
    red_agents = [agent for agent in env.agents if agent.startswith("red")]
    obs_shape = (13,13,5)
    state_shape = (45,45,5)
    count = 0
    normalizer_obs_b = Normalizer(shape=obs_shape)
    normalizer_obs_r = Normalizer(shape=obs_shape)
    normalizer_state = Normalizer(shape=state_shape)
    for ep in range(num_episodes):
        obs, _ = env.reset()
        # We will collect transitions after all blue agents have acted once (a "joint step")
        # the purpose of joint step is to collect joint transitions for QMIX training
        global_state = env.state().astype(np.float32) # global state for mixing
        global_state = normalizer_state.update_normalize(global_state)

        obs_array_blue = np.stack([obs[a] for a in blue_agents], axis=0, dtype=np.float32)
        obs_array_blue = normalizer_obs_b.update_normalize(obs_array_blue)

        obs_array_red = np.stack([obs[a] for a in red_agents], axis=0, dtype=np.float32)
        obs_array_red = normalizer_obs_r.update_normalize(obs_array_red)

        episode_reward_blue = 0
        episode_reward_red = 0

        terminated = False
        episode = []
        count += 1
        if ep < 200:
            add_parameter_noise(qmix_blue.agent_q_network)
            add_parameter_noise(qmix_red.agent_q_network)
        while not terminated:
            actions_blue = qmix_blue.select_actions(torch.from_numpy(obs_array_blue).to(device), epsilon=epsilon) # return shape: (N_agents,)

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

            next_global_state = env.state().astype(np.float32)
            next_global_state = normalizer_state.update_normalize(next_global_state)

            done_all = np.all(list(dones.values()))

            reward_blue = sum([rewards[a] for a in blue_agents])
            reward_red = sum([rewards[a] for a in red_agents])

            episode_reward_blue += reward_blue
            episode_reward_red += reward_red

            a_blue = np.array([actions_blue[a] for a in blue_agents])
            a_red = np.array([actions_red[a] for a in red_agents])

            next_obs_array_blue = np.stack([next_obs[a] for a in blue_agents], axis=0, dtype=np.float32)
            next_obs_array_blue = normalizer_obs_b.update_normalize(next_obs_array_blue)

            next_obs_array_red = np.stack([next_obs[a] for a in red_agents], axis=0, dtype=np.float32)
            next_obs_array_red = normalizer_obs_r.update_normalize(next_obs_array_red)

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

            obs_array_blue = next_obs_array_blue
            obs_array_red = next_obs_array_red
            global_state = next_global_state
            terminated = done_all

            # # After episode, decay epsilon
            # epsilon = max(epsilon * epsilon_decay, epsilon_min)

        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        # Save episode to replay buffer
        rb.save_episode(episode)
        wandb.log({
        "episode_reward_blue": episode_reward_blue,
        "episode_reward_red": episode_reward_red,
        "episode": ep
        })

        # Train QMIX
        if len(rb) >= batch_size and ep >= 13:
                batch, ids = rb.sample(batch_size)

                # batch['obs']: (B, transitions, N, 13, 13, 5)
                # batch['actions']: (B, transitions,N)
                # batch['rewards']: (B, transitions, )
                # batch['next_obs']: (B, transitions, 13, 13, 5)
                # batch['state']: (B, transitions, 45, 45, 5)
                # batch['next_state']: (B, transitions, 45, 45, 5)
                # batch['dones']: (B, transitions, )
                batch_blue, batch_red = process_batch(batch, normalizer_obs_b=normalizer_obs_b, normalizer_obs_r=normalizer_obs_r, normalizer_state=normalizer_state)
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

        if ((ep+1) % update_step) == 0:
                save_path_blue = os.path.join("/kaggle/working/", f"qmix_blue_ep{ep}.pth")
                save_path_red = os.path.join("/kaggle/working/", f"qmix_red_ep{ep}.pth")
                torch.save(qmix_blue.agent_q_network.state_dict(), save_path_blue)
                torch.save(qmix_red.agent_q_network.state_dict(), save_path_red)
            # Check if the save time has passed
        elapsed_time = time.time() - start_time
        if elapsed_time >= save_time_seconds:
            print(f"Saving model after {save_time_seconds / 3600} hours of training...")
            torch.save(qmix_blue.state_dict(), 'model_bafter_2_hours.pth')
            torch.save(qmix_red.state_dict(), 'model_rafter_2_hours.pth')
            break  # Optionally, stop training after saving the model
    wandb.finish()