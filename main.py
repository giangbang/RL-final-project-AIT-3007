from magent2.environments import battle_v4
import os
import cv2
import torch
from torch_model import QNetwork
from final_torch_model import QNetwork as FinalQNetwork

def load_model(model_path, env, agent_type, device="cpu"):
    if "final" in model_path:
        network = FinalQNetwork(
            env.observation_space(f"{agent_type}_0").shape,
            env.action_space(f"{agent_type}_0").n
        )
    else:
        network = QNetwork(
            env.observation_space(f"{agent_type}_0").shape,
            env.action_space(f"{agent_type}_0").n
        )
    network.load_state_dict(torch.load(model_path, map_location=device))
    network.eval()
    return network

def get_action(obs, network):
    observation = torch.Tensor(obs).float().permute([2, 0, 1]).unsqueeze(0)
    with torch.no_grad():
        q_values = network(observation)
    return torch.argmax(q_values, dim=1).item()

def record_video(env, blue_network, red_network=None, filename="battle.mp4"):
    frames = []
    env.reset()
    
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()

        if termination or truncation:
            action = None
        else:
            agent_team = agent.split("_")[0]
            if agent_team == "blue":
                action = get_action(obs, blue_network)
            else:  # red team
                if red_network:
                    action = get_action(obs, red_network)
                else:
                    action = env.action_space(agent).sample()

        env.step(action)
        if agent == env.agents[0]:
            frames.append(env.render())

    # Save video
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        filename,
        cv2.VideoWriter_fourcc(*"mp4v"),
        35,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()

if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)

    # Load blue model
    blue_network = load_model("blue_v2.pt", env, "blue")

    # Scenario 1: Blue vs Random
    record_video(
        env, 
        blue_network,
        filename=os.path.join(vid_dir, "blue_vs_random.mp4")
    )
    print("Done recording blue vs random")

    # Scenario 2: Blue vs Red
    red_network = load_model("red.pt", env, "red")
    record_video(
        env,
        blue_network,
        red_network,
        filename=os.path.join(vid_dir, "blue_vs_red.mp4")
    )
    print("Done recording blue vs red")

    # Scenario 3: Blue vs Red Final
    red_final_network = load_model("red_final.pt", env, "red")
    record_video(
        env,
        blue_network,
        red_final_network,
        filename=os.path.join(vid_dir, "blue_vs_red_final.mp4")
    )
    print("Done recording blue vs red_final")

    env.close()