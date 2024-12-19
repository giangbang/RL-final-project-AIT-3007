import os
import time
import numpy as np
import torch
import cv2
import argparse
from magent2.environments import battle_v4
from src.qmix.qmix import QMix_Trainer, ReplayBuffer, CNNFeatureExtractor
from torch_model import QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_blue_policy(model_path, hidden_dim=64, hypernet_dim=128):
    """
    Khởi tạo policy cho team blue sử dụng QMIX model đã train
    """
    # Khởi tạo environment để lấy các thông số
    dummy_env = battle_v4.env(map_size=45, minimap_mode=False, extra_features=False)
    dummy_env.reset()
    
    # Khởi tạo CNN feature extractor để lấy kích thước output
    dummy_cnn = CNNFeatureExtractor()
    obs_dim = dummy_cnn.get_output_dim(dummy_env.observation_space("blue_0").shape[:-1])
    state_dim = dummy_cnn.get_output_dim(dummy_env.state().shape[:-1])
    action_dim = dummy_env.action_space("blue_0").n
    action_shape = 1
    n_agents = len(dummy_env.agents)//2
    
    # Khởi tạo replay buffer và QMIX trainer
    replay_buffer = ReplayBuffer(2)  # Chỉ cần buffer size nhỏ vì không train
    learner = QMix_Trainer(
        replay_buffer=replay_buffer,
        n_agents=n_agents,
        obs_dim=obs_dim,
        state_dim=state_dim,
        action_shape=action_shape,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        hypernet_dim=hypernet_dim,
        target_update_interval=10,
        epsilon_start=0.0,  # Không cần epsilon vì đang evaluate
        epsilon_end=0.0,
        epsilon_decay=1.0
    )
    
    # Load model đã train
    learner.load_model(model_path, map_location=device)
    
    # Khởi tạo hidden state
    hidden_states = {i: torch.zeros(1, 1, hidden_dim).to(device) for i in range(n_agents)}
    
    def policy(env, agent_id, obs):
        """
        Policy cho một agent trong team blue
        """
        nonlocal hidden_states
        
        # Lưu observation vào dictionary
        agent_idx = int(agent_id.split("_")[1])

        # Get action từ model
        action, new_hidden = learner.get_action(obs, hidden_states[agent_idx])
        hidden_states[agent_idx] = new_hidden
        
        # Trả về action cho agent hiện tại
        return action[0][0]
        
    return policy

def get_random_policy():
    """
    Random policy cho team red
    """
    def policy(env, agent_id, obs):
        return env.action_space(agent_id).sample()
    return policy

def get_pretrain_red_policy(q_network):
    """
    Policy cho team red sử dụng pretrained model
    """
    def policy(env, agent_id, obs):
        observation = (
            torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            q_values = q_network(observation)
        return torch.argmax(q_values, dim=1).cpu().numpy()[0]
    return policy

def evaluate(env, blue_policy, red_policy, n_episodes=100, max_cycles=1000, save_video=False):
    """
    Đánh giá hiệu suất của các policy
    """
    red_win = []
    blue_win = []
    red_tot_rw = []
    blue_tot_rw = []
    env.reset()
    n_agent_each_team = len(env.agents)//2
    
    if save_video:
        vid_dir = "video"
        os.makedirs(vid_dir, exist_ok=True)
        fps = 35
        frames = []
    
    for episode in range(n_episodes):
        env.reset()
        red_reward = 0
        blue_reward = 0
        who_loses = None
        n_dead = {"red": 0, "blue": 0}
        
        for agent in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            agent_team = agent.split("_")[0]
            if agent_team == "red":
                red_reward += reward
            else:
                blue_reward += reward

            if env.unwrapped.frames >= max_cycles and who_loses is None:
                who_loses = "red" if n_dead["red"] > n_dead["blue"] else "draw"
                who_loses = "blue" if n_dead["red"] < n_dead["blue"] else who_loses

            if termination or truncation:
                action = None  # this agent has died
                n_dead[agent_team] = n_dead[agent_team] + 1

                if (
                    n_dead[agent_team] == n_agent_each_team
                    and who_loses is None  # all agents are terminated at the end of episodes
                ):
                    who_loses = agent_team
            else:
                if agent_team == "red":
                    action = red_policy(env, agent, observation)
                else:
                    action = blue_policy(env, agent, observation)

            env.step(action)
            
            try:
                if save_video and agent == env.agents[0]:
                    frames.append(env.render())
            except Exception as e:
                break
                
        red_win.append(who_loses == "blue")
        blue_win.append(who_loses == "red")

        red_tot_rw.append(red_reward / n_agent_each_team)
        blue_tot_rw.append(blue_reward / n_agent_each_team)
        
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"Current Blue Winrate: {np.mean(blue_win):.3f}")
            print(f"Current Blue Average Reward: {np.mean(blue_tot_rw):.3f}")
            print("---")
        
        if save_video and frames:
            height, width, _ = frames[0].shape
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out = cv2.VideoWriter(
                os.path.join(vid_dir, f"qmix_eval_{timestamp}.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 
                fps,
                (width, height),
            )
            for frame in frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()
            print("Done recording evaluation video")


    return {
        "winrate_red": np.mean(red_win),
        "winrate_blue": np.mean(blue_win),
        "average_rewards_red": np.mean(red_tot_rw),
        "average_rewards_blue": np.mean(blue_tot_rw),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate QMIX agents')
    parser.add_argument('--model_path', type=str, default='model/qmix', help='path to model')
    parser.add_argument('--n_episodes', type=int, default=1, help='number of episodes')
    parser.add_argument('--max_cycles', type=int, default=300, help='max cycles per episode')
    parser.add_argument('--render', action='store_true', help='render the environment')
    parser.add_argument('--save_video', action='store_true', help='save evaluation video')

    args = parser.parse_args()
    
    render_mode = "rgb_array" if args.save_video else None
    if args.render:
        render_mode = "human"
    # Khởi tạo environment
    env = battle_v4.env(map_size=45, minimap_mode=False, extra_features=False, render_mode=render_mode)
    
    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)
    
    # Khởi tạo policies
    blue_policy = get_blue_policy(args.model_path)
    red_policy = get_random_policy()
    # Evaluate
    results = evaluate(env, blue_policy, red_policy, args.n_episodes, args.max_cycles, args.save_video)    
    print("\nFinal Results Random:")
    print(f"Blue Winrate: {results['winrate_blue']:.3f}")
    print(f"Red Winrate: {results['winrate_red']:.3f}")
    print(f"Blue Average Reward: {results['average_rewards_blue']:.3f}")
    print(f"Red Average Reward: {results['average_rewards_red']:.3f}")
    
    red_policy = get_pretrain_red_policy(q_network)
    # Evaluate
    results = evaluate(env, blue_policy, red_policy, args.n_episodes, args.max_cycles, args.save_video)    
    print("\nFinal Results Pretrain:")
    print(f"Blue Winrate: {results['winrate_blue']:.3f}")
    print(f"Red Winrate: {results['winrate_red']:.3f}")
    print(f"Blue Average Reward: {results['average_rewards_blue']:.3f}")
    print(f"Red Average Reward: {results['average_rewards_red']:.3f}")

    env.close()