from magent2.environments import battle_v4
import os
import cv2
from torch_model import QNetwork
import torch
import numpy as np

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained blue agent
    blue_q_network = QNetwork(env.observation_space("blue_0").shape, env.action_space("blue_0").n).to(device)
    blue_q_network.load_state_dict(torch.load("blue.pt", map_location=device))
    blue_q_network.eval()

    # Define policies
    def blue_policy(observation):
        observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = blue_q_network(observation)
        action = q_values.argmax(dim=1).item()
        return action

    def random_policy(env, agent, observation):
        return env.action_space(agent).sample()

    # Function to change red agents to green in frames
    def change_red_to_green(frame):
        # Chuyển màu đỏ ([255, 0, 0]) thành màu lục ([0, 255, 0])
        mask = np.all(frame == [255, 0, 0], axis=-1)
        frame[mask] = [0, 255, 0]
        return frame

    # Simulation 1: blue.pt vs green agents (red agents displayed as green with random policy)
    frames = []
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                action = blue_policy(observation)
            elif agent_handle == "red":
                action = random_policy(env, agent, observation)
            else:
                action = env.action_space(agent).sample()

        env.step(action)

        if agent == "blue_0":
            frame = env.render()
            frame = change_red_to_green(frame)  # Thay đổi màu đỏ thành màu lục
            frames.append(frame)

    if frames:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            os.path.join(vid_dir, "blue_vs_green.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print("Đã ghi video blue.pt đấu với agents màu lục (ngẫu nhiên)")
    else:
        print("Không có khung hình nào được ghi lại cho blue.pt vs green agents")

    # Simulation 2: blue.pt vs red.pt
    # Load pretrained red agent
    red_q_network = QNetwork(env.observation_space("red_0").shape, env.action_space("red_0").n).to(device)
    red_q_network.load_state_dict(torch.load("red.pt", map_location=device))
    red_q_network.eval()

    def red_policy(observation):
        observation = torch.tensor(observation, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = red_q_network(observation)
        action = q_values.argmax(dim=1).item()
        return action

    frames = []
    env.reset()
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
                action = blue_policy(observation)
            elif agent_handle == "red":
                action = red_policy(observation)
            else:
                action = env.action_space(agent).sample()

        env.step(action)

        if agent == "blue_0":
            frames.append(env.render())

    if frames:
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(
            os.path.join(vid_dir, "blue_vs_red.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        out.release()
        print("Đã ghi video blue.pt đấu với red.pt")
    else:
        print("Không có khung hình nào được ghi lại cho blue.pt vs red.pt")

    env.close()
