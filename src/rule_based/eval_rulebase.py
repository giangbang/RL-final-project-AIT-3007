import os
import time
import numpy as np
import torch
import cv2
import argparse
from magent2.environments import battle_v4

import sys
import os
# Thêm thư mục gốc của project vào PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.torch_model import QNetwork
from src.final_torch_model import QNetwork as QNetwork_final
from model import RuleBasedAgent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                    action = blue_policy.get_action(torch.from_numpy(observation).permute(2, 0, 1).unsqueeze(0))[0].item()

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
    env = battle_v4.env(map_size=45, max_cycles=300, minimap_mode=False, extra_features=False, render_mode=render_mode)
    
    # Blue agent
    blue_policy = RuleBasedAgent(my_team='blue')
   
    # Khởi tạo policies
    red_policy = get_random_policy()
    # Evaluate
    results = evaluate(env, blue_policy, red_policy, args.n_episodes, args.max_cycles, args.save_video)    
    print("\nFinal Results Random:")
    print(f"Blue Winrate: {results['winrate_blue']:.3f}")
    print(f"Red Winrate: {results['winrate_red']:.3f}")
    print(f"Blue Average Reward: {results['average_rewards_blue']:.3f}")
    print(f"Red Average Reward: {results['average_rewards_red']:.3f}")
    
    # Red.pt
    env.reset()
    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("../../weight_models/red.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)
    red_policy = get_pretrain_red_policy(q_network)
    # Evaluate
    results = evaluate(env, blue_policy, red_policy, args.n_episodes, args.max_cycles, args.save_video)    
    print("\nFinal Results Pretrain:")
    print(f"Blue Winrate: {results['winrate_blue']:.3f}")
    print(f"Red Winrate: {results['winrate_red']:.3f}")
    print(f"Blue Average Reward: {results['average_rewards_blue']:.3f}")
    print(f"Red Average Reward: {results['average_rewards_red']:.3f}")
    
    # Red_final.pt
    env.reset()
    q_network = QNetwork_final(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("../../weight_models/red_final.pt", weights_only=True, map_location="cpu")
    )
    q_network.to(device)
    red_policy = get_pretrain_red_policy(q_network)
    # Evaluate
    results = evaluate(env, blue_policy, red_policy, args.n_episodes, args.max_cycles, args.save_video)    
    print("\nFinal Results Final Pretrain:")
    print(f"Blue Winrate: {results['winrate_blue']:.3f}")
    print(f"Red Winrate: {results['winrate_red']:.3f}")
    print(f"Blue Average Reward: {results['average_rewards_blue']:.3f}")
    print(f"Red Average Reward: {results['average_rewards_red']:.3f}")

    env.close()