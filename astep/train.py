import tempfile
from qmix2 import QMIX
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gc
from magent2.environments import battle_v4
import time
import numpy as np
import random
from tensordict.tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage, RandomSampler, LazyTensorStorage
# from perstep import AgentQNetwork
from process2 import process_episodes, collate_episodes
import wandb
import threading
def set_seed(seed: int):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Update target network periodically
def create_sample(blue_agents, red_agents):
    obs_array_blue = {
        agent: np.zeros((13,13,5))
        for agent in blue_agents 
    }
    obs_array_red = {
        agent: np.zeros((13,13,5))
        for agent in red_agents
    }
    action_array_blue = {
        agent: 0
        for agent in blue_agents
    }
    action_array_red = {
        agent: 0
        for agent in red_agents
    }
    reward_array_blue = {
        agent: 0.0
        for agent in blue_agents 
    }
    reward_array_red =  {
        agent: 0.0
        for agent in  red_agents
   }
    done_blue = False   
    done_red = False
    return obs_array_blue, obs_array_red, action_array_blue, action_array_red, reward_array_blue, reward_array_red, done_blue, done_red

def train(config):
    start_time = time.time()
    save_time_seconds = 2 * 60 * 60  # Save model after 2 hours of training
    set_seed(config.seed)
    env = battle_v4.env(map_size=45, minimap_mode=False, step_reward=-0.2,
            dead_penalty=-0.2, attack_penalty=-0.1, attack_opponent_reward=1.8, max_cycles=150, 
            extra_features=False, seed=config.seed)
    max_length = 150
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.batch_size
    lr = config.lr
    gamma = config.gamma
    num_episodes = config.num_episodes
    update_step = config.update_step
    tau = config.tau
    epsilon = config.epsilon_start
    epsilon_decay = config.epsilon_decay
    epsilon_min = config.epsilon_min
    qmix_blue = QMIX(num_agents=81, state_shape=(5, 45, 45), device=device, lr=lr, gamma=gamma)
    qmix_red = QMIX(num_agents=81, state_shape=(5, 45, 45), device=device, lr=lr, gamma=gamma)

    buffersize = 100
    # storage_dir = os.path.join(os.getcwd(), "storage1")
    with tempfile.TemporaryDirectory(dir="/home/trnmah/284-home/tmp3") as storage_dir:
        storage = LazyMemmapStorage(buffersize, scratch_dir=storage_dir)
        replay_buffer = TensorDictReplayBuffer(
            storage=storage,
            sampler=RandomSampler(),
            batch_size=batch_size,
        )
        env.reset()
        agent_names = env.possible_agents
        blue_agents = [agent for agent in agent_names if agent.startswith("blue_")]
        red_agents = [agent for agent in agent_names if agent.startswith("red_")]
        N = 162
        for ep in range(num_episodes):
            env.reset()
            done = False
            episode_transitions = []
            h_r = None
            h_b = None
            p_r = None
            p_b = None
            state = env.state()
            do = {agent: False for agent in agent_names}
            termination = False
            while not termination: 
                state = env.state()
                obs_array_blue, obs_array_red, action_array_blue, action_array_red, reward_array_blue, reward_array_red, done_blue, done_red = create_sample(blue_agents, red_agents)
                episode_reward_b = 0
                episode_reward_r = 0
                for id, agent in enumerate(env.agents):
                    obs, reward, termination, truncation, _ = env.last()
                    done = termination or truncation
                    if done:
                        break
                    name = agent.startswith("red")        
                    if name:
                        action, h_r = qmix_red.select_actions(obs, p_r, h_r, epsilon)
                    else:
                        action, h_b = qmix_blue.select_actions(obs, p_b, h_b, epsilon)

                    env.step(action)
                    # _, reward, _, _, _ = env.last()
                    if name:
                        obs_array_red[agent] = obs
                        action_array_red[agent] = action
                        reward_array_red[agent] = reward
                        episode_reward_r += reward
                    else:
                        obs_array_blue[agent] = obs
                        action_array_blue[agent] = action
                        reward_array_blue[agent] = reward
                        episode_reward_b += reward
                    do[agent] = done    

                obs_array_blue = np.array([obs_array_blue[agent] for agent in blue_agents]) # (N, 13, 13, 5)
                obs_array_red = np.array([obs_array_red[agent] for agent in red_agents])
                action_array_blue = np.array([action_array_blue[agent] for agent in blue_agents]) # (N, 1)
                action_array_red = np.array([action_array_red[agent] for agent in red_agents]) 
                reward_array_blue = np.array([reward_array_blue[agent] for agent in blue_agents]) # (N, 1)
                reward_array_red = np.array([reward_array_red[agent] for agent in red_agents])
                done_blue = np.array([done], dtype=np.uint8) # (N, 1)
                done_red = np.array([done], dtype=np.uint8)
                transition = {
                    "o_b": torch.from_numpy(obs_array_blue).float(),
                    "o_r": torch.from_numpy(obs_array_red).float(),
                    "a_b": torch.from_numpy(action_array_blue).long(),
                    "a_r": torch.from_numpy(action_array_red).long(),
                    "r_b": torch.tensor(reward_array_blue, dtype=torch.float32),
                    "r_r": torch.tensor(reward_array_red, dtype=torch.float32),
                    "d_b": torch.tensor(done_blue, dtype=torch.uint32),
                    "d_r": torch.tensor(done_red, dtype=torch.uint32),
                    "s": torch.from_numpy(state).float(),
                    "index": torch.tensor(ep+1, dtype=torch.uint32)  # Add episode_id
                }            

                episode_transitions.append(transition)
                del obs_array_blue, obs_array_red, action_array_blue, action_array_red, reward_array_blue, reward_array_red, done_blue, done_red
                gc.collect()
                termination = np.all(list(do.values()))
            wandb.log({"episode_reward_b": episode_reward_b, "episode_reward_r": episode_reward_r})

            episode_tensordict = {}
            keys = episode_transitions[0].keys()
            for key in keys:
                key_data = [transi[key] for transi in episode_transitions] # (T, ...)
                # if key == "index":
                #     stacked = key_data[0] # [1]
                # else:
                stacked = torch.stack(key_data, dim=0)
                episode_tensordict[key] = stacked.unsqueeze(0) # (1, T, ...)

            episode_tensordict = TensorDict(episode_tensordict, batch_size=[1])
            episode_tensordict = collate_episodes(episode_tensordict, max_length=max_length)

            replay_buffer.extend(episode_tensordict)

            del episode_transitions
            gc.collect()

            epsilon = max(epsilon_min, epsilon * epsilon_decay)
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size, device=device)
                # batch = batch.to(device)
                batch_blue, batch_red = process_episodes(batch)
                # train(batch)
                loss_blue = qmix_blue.update(batch_blue)
                loss_red = qmix_red.update(batch_red)
                qmix_blue.update_target_soft(tau)
                qmix_red.update_target_soft(tau)
                wandb.log({"loss_blue": loss_blue, "loss_red": loss_red})

            if (ep+1) % update_step == 0:
                qmix_blue.update_target_hard()
                qmix_red.update_target_hard()
                


            if ((ep+1) % 50) == 0:
                    save_path_blue = "qmix_blue_ep{}.pth".format(ep)
                    save_path_red = ".qmix_red_ep{}.pth".format(ep)
                    torch.save(qmix_blue.agent_q_network.state_dict(), save_path_blue)
                    torch.save(qmix_red.agent_q_network.state_dict(), save_path_red)
                    torch.save(qmix_blue.mixing_network.state_dict(), "mn_blue_ep{}.pth".format(ep))
                    torch.save(qmix_red.mixing_network.state_dict(), "mn_red_ep{}.pth".format(ep))
                    # normalizer_obs_b.save("normalizer_obs_b{}.pkl".format(ep))
                    # normalizer_obs_r.save("normalizer_obs_r.pkl".format(ep))
                    # normalizer_state.save("normalizer_state.pkl".format(ep))
                # Check if the save time has passed
            elapsed_time = time.time() - start_time
            if (save_time_seconds - elapsed_time) <= 100:
                print(f"Saving model after {save_time_seconds / 3600} hours of training...")
                qmix_blue.update_target_hard()
                qmix_red.update_target_hard()
                torch.save(qmix_blue.agent_q_network.state_dict(), 'aqn_model_bafter_2_hours.pth')
                torch.save(qmix_red.agent_q_network.state_dict(), 'aqn_model_rafter_2_hours.pth')
                torch.save(qmix_blue.mixing_network.state_dict(), 'mn_model_btarget_after_2_hours.pth')
                torch.save(qmix_red.mixing_network.state_dict(), 'mn_model_rtarget_after_2_hours.pth')
                break  # Optionally, stop training after saving the model
        wandb.finish()
    # normalizer_obs_b.save("normalizer_obs_b.pkl")
    # normalizer_obs_r.save("normalizer_obs_r.pkl")
    # normalizer_state.save("normalizer_state.pkl")
# storage_dir.cleanup()
    
