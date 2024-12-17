import torch
from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

def create_video(frames, filename, vid_dir, fps=35):
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, filename),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

def run_episode(env, blue_network, red_policy):
    frames = []
    env.reset()
    
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            agent_team = agent.split("_")[0]
            if agent_team == "blue":
                # Blue agent policy
                observation = torch.FloatTensor(observation).permute(2, 0, 1).unsqueeze(0)
                with torch.no_grad():
                    q_values = blue_network(observation)
                action = q_values.argmax().item()
            else:
                # Red agent policy
                action = red_policy(observation, env, agent)

        env.step(action)
        if agent == "blue_0":
            frames.append(env.render())
            
    return frames

if __name__ == "__main__":
    # Setup
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)

    # Load models
    blue_network = QNetwork(
        env.observation_space("blue_0").shape,
        env.action_space("blue_0").n
    ).to(device)
    blue_network.load_state_dict(torch.load("blue.pt", map_location=device))
    blue_network.eval()

    red_network = QNetwork(
        env.observation_space("red_0").shape,
        env.action_space("red_0").n
    ).to(device)
    red_network.load_state_dict(torch.load("red.pt", map_location=device))
    red_network.eval()

    red_final_network = FinalQNetwork(
        env.observation_space("red_0").shape,
        env.action_space("red_0").n
    ).to(device)
    red_final_network.load_state_dict(torch.load("red_final.pt", map_location=device))
    red_final_network.eval()

    # Define policies
    def random_policy(obs, env, agent):
        return env.action_space(agent).sample()

    def red_policy(obs, env, agent):
        observation = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_network(observation)
        return q_values.argmax().item()

    def red_final_policy(obs, env, agent):
        observation = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_final_network(observation)
        return q_values.argmax().item()

    # Record episodes
    # Blue vs Random
    frames = run_episode(env, blue_network, random_policy)
    create_video(frames, "blue_vs_random.mp4", vid_dir)
    print("Done recording blue vs random agents")

    # Blue vs Red
    frames = run_episode(env, blue_network, red_policy)
    create_video(frames, "blue_vs_red.mp4", vid_dir)
    print("Done recording blue vs red agents")

    # Blue vs Red Final
    frames = run_episode(env, blue_network, red_final_policy)
    create_video(frames, "blue_vs_red_final.mp4", vid_dir)
    print("Done recording blue vs red_final agents")

    env.close()